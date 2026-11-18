"""Audio loading with MIRtoolbox ``miraudio`` options.

Translated from MiningSuite ``+sig/+signal/options.m``, the ``after``
method of ``+sig/@Signal/Signal.m``, ``+sig/@Signal/center.m``,
``trim.m``, ``resample.m``, ``fwr.m``, and the read-time excerpt in
``+sig/@design/design.m`` and ``+sig/evaleach.m`` (BSD-3, Olivier
Lartillot).
"""

import warnings
from dataclasses import replace
from os import PathLike

import numpy as np
from scipy.signal import resample_poly

from mirpy.core import framing
from mirpy.core.data import Audio
from mirpy.core.matlab_compat import matlab_round
from mirpy.core.options import FrameArg

EXTRACT_UNITS = ("s", "sp")
TRIM_CHOICES = ("both", "start", "end")
DEFAULT_TRIM_THRESHOLD = 0.06
TRIM_FRAME_LENGTH = 100  # samples, fixed in trim.m
TRIM_FRAME_HOP = 10  # samples, fixed in trim.m
RESAMPLE_KAISER_BETA = 5.0  # MATLAB resample default


def audio(
    source,
    sr: float | None = None,
    *,
    mono: bool = True,
    channels=None,
    center: bool = False,
    sampling: float | None = None,
    extract: tuple[float, float] | None = None,
    extract_unit: str = "s",
    trim: bool | str = False,
    trim_threshold: float = DEFAULT_TRIM_THRESHOLD,
    fwr: bool = False,
    frame: FrameArg = False,
) -> Audio:
    """Load a waveform and apply the basic MIRtoolbox audio options.

    Equivalent to MIRtoolbox ``miraudio``. Options are applied in this
    order: extract, channel selection, mix, center, resample, full-wave
    rectification, trim, frame.

    Parameters
    ----------
    source : str, path-like or array_like
        Audio file path, or a waveform of shape ``(n_samples,)`` or
        ``(n_channels, n_samples)``.
    sr : float, optional
        Sampling rate in Hz. Required when `source` is an array, not
        allowed for a file.
    mono : bool
        Average the channels into one. MATLAB ``'Mono'``.
    channels : int or sequence of int, optional
        Keep only these channels (0-based), before mixing. MATLAB
        ``'Channel'`` (1-based).
    center : bool
        Subtract the mean of each channel. MATLAB ``'Center'``.
    sampling : float, optional
        Resample to this rate in Hz. MATLAB ``'Sampling'``. Original and
        new rates must be whole numbers of Hz.
    extract : (float, float), optional
        Keep only the excerpt between these bounds. MATLAB ``'Extract'``.
    extract_unit : {"s", "sp"}
        ``"s"``: times in seconds, both ends included, as MIRtoolbox.
        ``"sp"``: 0-based sample indices, end excluded (Python slice).
        MATLAB ``'Extract', 1, 100, 'sp'`` is ``extract=(0, 100)``.
    trim : bool or {"both", "start", "end"}
        Remove leading and/or trailing silence. ``True`` means ``"both"``.
        MATLAB ``'Trim'`` with ``'BothEnds'``, ``'JustStart'``, ``'JustEnd'``.
    trim_threshold : float
        Frames whose RMS, scaled between 0 (quietest) and 1 (loudest),
        is at or below this value count as silence. MATLAB
        ``'TrimThreshold'``.
    fwr : bool
        Full-wave rectification (absolute value). MATLAB ``'FWR'``.
    frame : bool, FrameSpec or tuple
        Decompose into frames; ``True`` uses 50 ms frames with half
        overlap. See :func:`mirpy.frame`.

    Returns
    -------
    Audio

    See Also
    --------
    librosa.load : loads, mixes to mono and resamples.

    Examples
    --------
    >>> import numpy as np
    >>> x = audio(np.ones(1000), sr=1000, extract=(0.1, 0.2))
    >>> x.n_elements, x.start
    (101, 0.1)
    """
    y, sr, path = _read(source, sr)
    first = 0
    if extract is not None:
        first, stop = _extract_bounds(extract, extract_unit, sr, y.shape[1])
        y = y[:, first:stop]
    if channels is not None:
        y = y[np.atleast_1d(channels)]
    x = Audio.from_array(y, sr, mono=mono, start=first / sr, source=path)
    if center:
        x = _center(x)
    if sampling is not None:
        x = _resample(x, sampling)
    if fwr:
        x = _fwr(x)
    if trim:
        x = _trim(x, "both" if trim is True else trim, trim_threshold)
    x = framing.frame(x, frame)
    params = {
        "mono": mono,
        "channels": channels,
        "center": center,
        "sampling": sampling,
        "extract": extract,
        "extract_unit": extract_unit,
        "trim": trim,
        "trim_threshold": trim_threshold,
        "fwr": fwr,
    }
    return replace(x, params={**params, **x.params})


def _read(source, sr: float | None) -> tuple[np.ndarray, float, str | None]:
    """Return the waveform as ``(n_channels, n_samples)``, its rate and path."""
    if isinstance(source, str | PathLike):
        if sr is not None:
            raise ValueError("sr is only accepted with an array; files carry their own rate")
        import soundfile

        y, file_sr = soundfile.read(source, dtype="float64", always_2d=True)
        return y.T, float(file_sr), str(source)
    if sr is None:
        raise ValueError("sr is required when source is an array")
    y = np.atleast_2d(np.asarray(source, dtype=np.float64))
    if y.ndim != 2:
        raise ValueError(
            f"source must have shape (n_samples,) or (n_channels, n_samples), got {y.shape}"
        )
    return y, float(sr), None


def _extract_bounds(extract, unit: str, sr: float, n_samples: int) -> tuple[int, int]:
    """Return the 0-based ``[first, stop)`` samples of an excerpt.

    In seconds, follows ``@design/design.m``: ``round(t * sr)`` with both
    ends included, clipped to the signal.
    """
    if unit not in EXTRACT_UNITS:
        raise ValueError(f"extract_unit must be one of {EXTRACT_UNITS}, got {unit!r}")
    t1, t2 = extract
    if unit == "s":
        first, last = np.clip(matlab_round(np.array([t1, t2]) * sr), 0, n_samples - 1)
        first, stop = int(first), int(last) + 1
    else:
        first, stop = (int(np.clip(i, 0, n_samples)) for i in (t1, t2))
    if stop <= first:
        raise ValueError(f"extract={extract} selects no samples of a {n_samples}-sample signal")
    return first, stop


def _center(x: Audio) -> Audio:
    return replace(x, data=x.data - x.data.mean(axis=1, keepdims=True))


def _resample(x: Audio, new_sr: float) -> Audio:
    if new_sr == x.sr:
        return x
    if not (float(new_sr).is_integer() and float(x.sr).is_integer()):
        raise ValueError(f"Resampling needs whole-number rates, got {x.sr} -> {new_sr} Hz")
    data = resample_poly(
        x.data, int(new_sr), int(x.sr), axis=1, window=("kaiser", RESAMPLE_KAISER_BETA)
    )
    return replace(x, data=data, sr=float(new_sr))


def _fwr(x: Audio) -> Audio:
    return replace(x, data=np.abs(x.data))


def _trim(x: Audio, where: str, threshold: float) -> Audio:
    """Cut the quiet start and/or end, as in ``@Signal/trim.m``.

    RMS is computed on 100-sample frames with a 10-sample hop over the
    sum of the channels, then scaled to [0, 1]. The cut keeps one quiet
    frame on each side. As in ``trim.m``, the end cut falls at the
    *start* of the last kept frame.
    """
    if where not in TRIM_CHOICES:
        raise ValueError(f"trim must be True, False or one of {TRIM_CHOICES}, got {where!r}")
    rms = _trim_rms(x.data[:, :, 0].sum(axis=0))
    span = np.ptp(rms) if rms.size else 0.0
    loud = np.flatnonzero((rms - rms.min()) / span > threshold) if span else np.empty(0)
    if loud.size == 0:
        warnings.warn("No frame above the trim threshold; signal not trimmed.", stacklevel=3)
        return x

    first, last = 0, x.n_elements - 1
    if where in ("both", "start"):
        first = max(loud[0] - 1, 0) * TRIM_FRAME_HOP
    if where in ("both", "end"):
        last = min(loud[-1] + 1, rms.size - 1) * TRIM_FRAME_HOP
    return replace(x, data=x.data[:, first : last + 1], start=x.start + first / x.sr)


def _trim_rms(y: np.ndarray) -> np.ndarray:
    n_frames = (y.size - TRIM_FRAME_LENGTH) // TRIM_FRAME_HOP + 1
    if n_frames < 1:
        return np.empty(0)
    starts = np.arange(n_frames) * TRIM_FRAME_HOP
    frames = y[starts[:, np.newaxis] + np.arange(TRIM_FRAME_LENGTH)]
    return np.linalg.norm(frames, axis=1) / np.sqrt(TRIM_FRAME_LENGTH)
