"""Peak picking.

Translated from MiningSuite ``+sig/peaks.m`` (BSD-3, Olivier Lartillot;
originally Olivier Lartillot & University of Jyväskylä, 2007-2009).

MiningSuite quirks kept on purpose, to be checked against the MIRtoolbox
1.8.2 fixtures:

- ``Peaks.val`` is read on the normalised curve; ``Peaks.val_original``
  converts it back to the signal's units.
- Quadratic interpolation is skipped for the first two positions
  (``p > 2`` in 1-based ``peaks.m``).
- A peak on the last position is interpolated against the padding value
  (±inf), which gives ``pos == 0`` and ``val == nan``.
- ``normalize="global"`` scales by the minimum and maximum of the whole
  signal, all channels and frames together.
"""

from dataclasses import dataclass, replace

import numpy as np

from mirpy.core.data import Peaks, Signal

ORDERS = ("amplitude", "abscissa")
NORMALIZATIONS = ("global", "local", "none")


@dataclass(frozen=True)
class _SearchOptions:
    limit: float
    contrast: float
    select_first: float
    reso: float | None
    reso_first: bool
    total: int | None
    order: str
    interpolate: bool
    no_begin: bool
    no_end: bool
    normalize: str


def peaks(
    x: Signal,
    *,
    total: int | None = None,
    order: str = "amplitude",
    valleys: bool = False,
    contrast: float = 0.1,
    select_first: float = 0.0,
    threshold: float | None = None,
    interpolate: bool = True,
    reso: float | None = None,
    reso_first: bool = False,
    normalize: str = "global",
    no_begin: bool = False,
    no_end: bool = False,
) -> Signal:
    """Find the peaks of each curve of a signal.

    Equivalent to MIRtoolbox ``mirpeaks``. Each curve is searched along
    its elements (e.g. the bins of each spectrum frame), or along frames
    when the signal has one element per frame (e.g. framed RMS).

    A local maximum is kept only if it rises at least `contrast` above
    the lowest point separating it from its neighbouring kept peaks.
    Heights are compared on the curve scaled to [0, 1] (see `normalize`).

    Parameters
    ----------
    x : Signal
        Signal to search.
    total : int, optional
        Keep only the `total` highest peaks of each curve. MATLAB
        ``'Total'``; default: all.
    order : {"amplitude", "abscissa"}
        Order of the peaks of each curve: highest first, or by position.
        MATLAB ``'Order'``.
    valleys : bool
        Find minima instead of maxima. MATLAB ``'Valleys'``.
    contrast : float
        Minimum height difference between a peak and the neighbouring
        notches, on the normalised scale. 0 keeps every local maximum.
        MATLAB ``'Contrast'``.
    select_first : float
        When two peaks are not separated by a notch deep enough, the later
        one replaces the earlier one only if it is higher by more than
        this value. MATLAB ``'SelectFirst'``; given without a value in
        MATLAB, it means ``contrast / 2``.
    threshold : float, optional
        Minimum peak height on the normalised scale (for valleys, maximum
        depth, counted from the bottom). MATLAB ``'Threshold'``.
    interpolate : bool
        Refine position and height with a parabola through the peak and
        its two neighbours. MATLAB ``'Interpol'``.
    reso : float, optional
        Remove peaks closer than `reso` (in units of the searched axis) to
        a higher peak. MATLAB ``'Reso'``.
    reso_first : bool
        With `reso`, keep the earlier of two close peaks instead of the
        higher one. MATLAB ``'First'``.
    normalize : {"global", "local", "none"}
        Scale the values to [0, 1] using the minimum and maximum of the
        whole signal, or of each curve, or not at all. MATLAB
        ``'Normalize'``.
    no_begin, no_end : bool
        Do not accept a peak on the first, or last, position of a curve.
        MATLAB ``'NoBegin'``, ``'NoEnd'``.

    Returns
    -------
    Signal
        `x` with :attr:`Signal.peaks` set.

    See Also
    --------
    scipy.signal.find_peaks : peak picking with prominence and distance.
    librosa.util.peak_pick : onset-style peak picking.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Signal
    >>> curve = Signal(np.array([0, 1, 0, 0, 3, 0, 2, 0.0])[None, :, None], sr=1)
    >>> peaks(curve, interpolate=False).peaks.index
    array([4, 6, 1])
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")

    threshold = 0.0 if threshold is None else (1 - threshold if valleys else threshold)
    options = _SearchOptions(
        limit=threshold if normalize == "none" else max(contrast, threshold),
        contrast=contrast,
        select_first=select_first,
        reso=reso,
        reso_first=reso_first,
        total=total,
        order=order,
        interpolate=interpolate,
        no_begin=no_begin,
        no_end=no_end,
        normalize=normalize,
    )
    axis, positions = _peak_axis(x)
    curves, channels, frames = _curves(x, axis)
    sign = -1.0 if valleys else 1.0
    curves = sign * curves
    offset, scale = _global_scaling(curves) if normalize == "global" else (0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        curves = (curves - offset) / scale

    rows = [_curve_peaks(curve, positions, options) for curve in curves]
    found = [
        (channel, frame, *row) for channel, frame, row in zip(channels, frames, rows, strict=True)
    ]
    return replace(
        x,
        peaks=_assemble(axis, found, sign, offset, scale),
        params={**x.params, "peaks": {**vars(options), "valleys": valleys}},
    )


def _peak_axis(x: Signal) -> tuple[str, np.ndarray]:
    """Return the axis to search and the position of each point along it."""
    if x.is_framed and x.n_elements == 1:
        return "frame", x.frame_starts
    if x.x_values is not None:
        return "element", x.x_values
    if not x.is_framed:
        return "element", x.times
    return "element", np.arange(x.n_elements) / x.sr


def _curves(x: Signal, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the curves as rows, with the channel and frame of each."""
    if axis == "frame":
        return x.data[:, 0, :], np.arange(x.n_channels), np.full(x.n_channels, -1)
    curves = np.moveaxis(x.data, 1, -1).reshape(-1, x.n_elements)
    channels, frames = np.divmod(np.arange(curves.shape[0]), x.n_frames)
    return curves, channels, frames


def _global_scaling(curves: np.ndarray) -> tuple[float, float]:
    low, high = np.nanmin(curves), np.nanmax(curves)
    return low, high - low


def _curve_peaks(y: np.ndarray, x: np.ndarray, o: _SearchOptions):
    """Search one curve (``search`` in ``peaks.m``).

    Returns 0-based indices, positions, heights on the normalised scale,
    and the local scaling ``(offset, scale)`` applied to the curve.
    """
    local = (0.0, 1.0)
    if o.normalize == "local":
        local = (np.nanmin(y), np.nanmax(y) - np.nanmin(y))
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (y - local[0]) / local[1]

    # Padded curve: the value of point k is padded[k + 1].
    padded = np.concatenate(
        [[np.inf if o.no_begin else -np.inf], y, [np.inf if o.no_end else -np.inf]]
    )
    rise = np.diff(padded)
    with np.errstate(invalid="ignore"):
        candidates = np.flatnonzero((padded[1:-1] >= o.limit) & (rise[:-1] > 0) & (rise[1:] <= 0))
    if candidates.size and o.contrast:
        candidates = _select_contrasted(padded, candidates, o.contrast, o.select_first)
    if candidates.size and o.reso:
        candidates = _remove_close(padded, x, candidates, o.reso, o.reso_first)
    candidates = _sort(padded, candidates, o.total, o.order)

    if o.interpolate:
        pos, val = _interpolate(padded, x, candidates)
    else:
        pos, val = x[candidates], padded[candidates + 1]
    rank = np.argsort(np.argsort(-padded[candidates + 1], kind="stable"), kind="stable")
    return candidates, pos, val, rank, local


def _select_contrasted(padded, candidates, contrast, select_first) -> np.ndarray:
    """Keep the peaks separated by notches deeper than `contrast`.

    Line-by-line port of the selection loop in ``peaks.m``. MATLAB ``min``
    and ``max`` ignore NaN, hence ``np.fmin``/``np.fmax``.
    """
    selected = []
    current = 0
    notch = np.inf  # lowest point since the current peak (bufmin)
    height = padded[candidates[0] + 1]  # current peak height (bufmax)
    previous_notch = np.fmin.reduce(padded[: candidates[0] + 1])  # oldbufmin
    for k in range(1, len(candidates)):
        between = padded[candidates[k - 1] + 2 : candidates[k] + 1]
        notch = np.fmin(notch, np.fmin.reduce(between))
        new = padded[candidates[k] + 1]
        if height - notch < contrast:
            # No contrastive notch: the higher peak may take over.
            if new > height and (
                new - height > select_first or height - previous_notch < contrast
            ):
                current, height = k, new
                previous_notch = np.fmin(previous_notch, notch)
                notch = np.inf
            elif new - height <= select_first:
                height = np.fmax(height, new)
                previous_notch = np.fmin(previous_notch, notch)
        else:
            if height - previous_notch < contrast:
                # The current peak is too weak and is dropped.
                previous_notch = np.fmin(previous_notch, notch)
            else:
                selected.append(candidates[current])
                previous_notch = notch
            current, height = k, new
            notch = np.inf
    last = candidates[current]
    if height - previous_notch >= contrast and (
        height - np.fmin.reduce(padded[last + 2 :]) >= contrast
    ):
        selected.append(last)
    return np.array(selected, dtype=np.intp)


def _remove_close(padded, x, candidates, reso, reso_first) -> np.ndarray:
    """Remove peaks closer than `reso` to a higher one (``'Reso'``)."""
    candidates = candidates[np.argsort(-padded[candidates + 1], kind="stable")]
    removed = set()
    for j in range(len(candidates) - 1):
        for k in range(j + 1, len(candidates)):
            if abs(x[candidates[k]] - x[candidates[j]]) < reso:
                earlier_is_lower = reso_first and candidates[j] > candidates[k]
                removed.add(j if earlier_is_lower else k)
    return np.delete(candidates, sorted(removed))


def _sort(padded, candidates, total, order) -> np.ndarray:
    by_height = candidates[np.argsort(-padded[candidates + 1], kind="stable")]
    if total is not None:
        by_height = by_height[:total]
    return np.sort(by_height) if order == "abscissa" else by_height


def _interpolate(padded, x, candidates) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic refinement of position and height, as in ``peaks.m``."""
    pos = x[candidates].astype(np.float64)
    val = padded[candidates + 1].copy()
    inner = candidates >= 2
    p = candidates[inner]
    peak, before, after = padded[p + 1], padded[p], padded[p + 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (after - before) / (2 * (2 * peak - after - before))
        val[inner] = peak - 0.25 * (before - after) * r
        next_x = x[np.minimum(p + 1, len(x) - 1)]
        toward_next = (1 - r) * x[p] + r * next_x
        toward_previous = (1 + r) * x[p] - r * x[p - 1]
    pos[inner] = np.where(r >= 0, toward_next, np.where(r < 0, toward_previous, 0.0))
    return pos, val


def _assemble(axis, found, sign, offset, scale) -> Peaks:
    columns = {
        name: [] for name in ("channel", "frame", "index", "pos", "val", "val_original", "rank")
    }
    for channel, frame, index, pos, val, rank, (local_offset, local_scale) in found:
        columns["channel"].append(np.full(len(index), channel))
        columns["frame"].append(index if axis == "frame" else np.full(len(index), frame))
        columns["index"].append(index)
        columns["pos"].append(pos)
        columns["val"].append(val)
        columns["val_original"].append(
            sign * ((val * local_scale + local_offset) * scale + offset)
        )
        columns["rank"].append(rank)
    return Peaks(axis, **{name: np.concatenate(parts) for name, parts in columns.items()})
