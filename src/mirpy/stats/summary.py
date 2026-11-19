"""Statistics of features along frames.

``mean``, ``std`` and ``median`` are translated from MiningSuite
``+sig/mean.m``, ``+sig/std.m`` and ``+sig/median.m`` (BSD-3, Olivier
Lartillot), but reduce along *frames* as the MIRtoolbox manual says
(MiningSuite reduces along elements when there are several, which would
average MFCC ranks together). NaN values are ignored, as ``mirstat`` does.

``stat`` is not in MiningSuite. It is written from the description of
``mirstat`` in the MIRtoolbox manual. The ``Period*`` details (plain
autocorrelation, entropy of its absolute value, frame rate from the first
frame step) were fitted to the MIRtoolbox fixtures, where they match to
the fifth digit.
"""

import warnings

import numpy as np

from mirpy.core.data import FRAME_AXIS, Signal
from mirpy.features._common import relative_entropy

STAT_FIELDS = ("Mean", "Std", "Slope", "PeriodFreq", "PeriodAmp", "PeriodEntropy")
MIN_PERIOD_FRAMES = 3


def mean(x: Signal) -> Signal:
    """Compute the mean of a feature along frames, ignoring NaN.

    Equivalent to MIRtoolbox ``mirmean``.

    Parameters
    ----------
    x : Signal
        Any feature; an unframed one is returned as a single frame.

    Returns
    -------
    Signal
        Unframed, one value per channel and element, named
        ``"<name>_Mean"`` (the ``mirexport`` column name).

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Signal
    >>> x = Signal(np.array([1.0, np.nan, 3.0]).reshape(1, 1, 3), sr=1, name="RMS", frame_rate=1)
    >>> m = mean(x)
    >>> m.name, m.data.item()
    ('RMS_Mean', 2.0)
    """
    return _summary(x, _ignoring_nan(np.nanmean, x.data), "Mean")


def std(x: Signal) -> Signal:
    """Compute the standard deviation of a feature along frames, ignoring NaN.

    Equivalent to MIRtoolbox ``mirstd``. Uses ``n - 1`` in the
    denominator (MATLAB ``std``); a single value has a standard deviation
    of 0, as in MATLAB.

    Parameters
    ----------
    x : Signal
        Any feature.

    Returns
    -------
    Signal
        Unframed, named ``"<name>_Std"``.
    """
    return _summary(x, _nanstd(x.data), "Std")


def median(x: Signal) -> Signal:
    """Compute the median of a feature along frames, ignoring NaN.

    Equivalent to MIRtoolbox ``mirmedian``.

    Parameters
    ----------
    x : Signal
        Any feature.

    Returns
    -------
    Signal
        Unframed, named ``"<name>_Median"``.
    """
    return _summary(x, _ignoring_nan(np.nanmedian, x.data), "Median")


def stat(x: Signal) -> dict[str, Signal]:
    """Compute the summary statistics of a feature along frames.

    Equivalent to MIRtoolbox ``mirstat``. NaN frames are discarded first.

    Parameters
    ----------
    x : Signal
        Any feature.

    Returns
    -------
    dict of str to Signal
        For a framed feature, one unframed :class:`Signal` per field, each
        named ``"<name>_<field>"``:

        - ``Mean``, ``Std``: see :func:`mean`, :func:`std`.
        - ``Slope``: least-squares slope of the curve, centred and scaled
          to unit variance, against time scaled to [-0.5, 0.5].
        - ``PeriodFreq``: frequency (Hz) of the highest positive local
          maximum of the curve's autocorrelation (not centred, no window,
          all lags; the last lag counts as a maximum when the curve rises
          to it). NaN when there is none. The lag is converted with the
          first frame step, as MIRtoolbox does (44100/1102 Hz, not 40 Hz,
          for 50 ms frames with half overlap at 44.1 kHz).
        - ``PeriodAmp``: that peak's height, with lag 0 scaled to 1. NaN
          when the autocorrelation never rises again, 0 when it does but
          no maximum is above 0 (MIRtoolbox 1.8.2 fixtures).
        - ``PeriodEntropy``: relative entropy (see :func:`mirpy.entropy`)
          of the absolute value of that autocorrelation.

        An unframed feature is returned unchanged, as ``{"Mean": x}``.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Signal
    >>> ramp = Signal(np.arange(10.0).reshape(1, 1, 10), sr=1, name="RMS", frame_rate=1)
    >>> sorted(stat(ramp))
    ['Mean', 'PeriodAmp', 'PeriodEntropy', 'PeriodFreq', 'Slope', 'Std']
    """
    if not x.is_framed:
        return {"Mean": x}
    curves = x.data.reshape(-1, x.n_frames)
    rate = _frame_step_rate(x)
    periodicity = np.array([_periodicity(curve, rate) for curve in curves])
    shape = x.data.shape[:2]
    fields = {
        "Mean": mean(x),
        "Std": std(x),
        "Slope": _summary(x, np.array([_slope(c) for c in curves]).reshape(shape), "Slope"),
    }
    for column, field in enumerate(("PeriodFreq", "PeriodAmp", "PeriodEntropy")):
        fields[field] = _summary(x, periodicity[:, column].reshape(shape), field)
    return fields


def _summary(x: Signal, values: np.ndarray, field: str) -> Signal:
    """Wrap ``(n_channels, n_elements)`` values in an unframed :class:`Signal`."""
    return Signal(
        data=values[:, :, np.newaxis],
        sr=x.sr,
        name=f"{x.name}_{field}",
        unit=x.unit,
        x_name=x.x_name,
        x_unit=x.x_unit,
        x_values=x.x_values,
        start=x.start,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
    )


def _ignoring_nan(reduce, data: np.ndarray) -> np.ndarray:
    """Reduce along frames; an all-NaN curve gives NaN without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reduce(data, axis=FRAME_AXIS)


def _nanstd(data: np.ndarray) -> np.ndarray:
    count = np.sum(np.isfinite(data), axis=FRAME_AXIS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(data, axis=FRAME_AXIS, ddof=1)
    return np.where(count == 1, 0.0, spread)


def _slope(curve: np.ndarray) -> float:
    """Slope S solving S * T = C in the least-squares sense (manual, mirstat)."""
    time = np.linspace(-0.5, 0.5, curve.size)
    keep = np.isfinite(curve)
    if keep.sum() < 2:
        return np.nan
    values, time = curve[keep], time[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = (values - values.mean()) / values.std(ddof=1)
    return float(np.sum(time * normalised) / np.sum(time**2))


def _periodicity(curve: np.ndarray, frame_rate: float) -> tuple[float, float, float]:
    """Return (PeriodFreq, PeriodAmp, PeriodEntropy) of one feature curve."""
    values = curve[np.isfinite(curve)]
    if values.size < MIN_PERIOD_FRAMES:
        return np.nan, np.nan, np.nan
    r = np.correlate(values, values, mode="full")[values.size - 1 :]
    if r[0] == 0:
        return np.nan, np.nan, np.nan
    r = r / r[0]
    entropy = float(relative_entropy(np.abs(r).reshape(1, -1, 1), center=False).item())
    maxima = _local_maxima(r)
    if maxima.size == 0:
        return np.nan, np.nan, entropy
    positive = maxima[r[maxima] > 0]
    if positive.size == 0:
        return np.nan, 0.0, entropy
    peak = int(positive[np.argmax(r[positive])])
    return frame_rate / peak, r[peak], entropy


def _frame_step_rate(x: Signal) -> float:
    """Frames per second from the first frame step (MIRtoolbox), else the nominal rate."""
    starts = x.frame_starts
    if starts is None or starts.size < 2:
        return x.frame_rate
    return 1 / (starts[1] - starts[0])


def _local_maxima(r: np.ndarray) -> np.ndarray:
    """Lags of the local maxima after the first descent from lag 0, the last lag included."""
    rising = np.flatnonzero(np.diff(r) > 0)
    if rising.size == 0:
        return rising
    lags = np.arange(rising[0] + 1, r.size)
    above_left = r[lags] > r[lags - 1]
    above_right = np.append(r[lags[:-1]] >= r[lags[:-1] + 1], True)
    return lags[above_left & above_right]
