"""Autocorrelation function.

Translated from MiningSuite ``+sig/autocor.m``, ``+sig/+autocor/``
(``options.m``, ``main.m``, ``after.m``) and ``+sig/@Autocor/``
(``Autocor.m``, ``normalize.m``) (BSD-3, Olivier Lartillot).

Lag range, in samples ``k`` (``k / sr`` seconds):

- default: ``0 <= k <= min(ceil(max * sr), ceil(n / 2) - 1)``;
- explicit bounds: ``after.m`` cuts the range again with ``round``, so
  ``round(min_lag * sr) <= k <= round(max_lag * sr)``.
"""

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np

from mirpy.core import framing
from mirpy.core.data import Signal
from mirpy.core.matlab_compat import matlab_round, matlab_window, xcorr
from mirpy.core.options import FrameArg
from mirpy.signal._lag_axis import as_frequencies
from mirpy.signal.spectrum import Spectrum

SCALES = ("coeff", "biased", "unbiased", "none", "coeff_x_channels")
AUTO = "auto"
DEFAULT_MAX_LAG = 0.05  # seconds, for audio sampled above 1 kHz
LOW_RATE = 1000  # Hz: below it, no default lag limit and rectangular window
LAG_DIGITS = 9  # rounding before ceil, so 44100 / 441 Hz stays 100 lags
STANDARD_EXPONENT = 2.0  # generalized=2 is the ordinary autocorrelation


@dataclass(frozen=True, eq=False)
class Autocor(Signal):
    """An autocorrelation function, one per frame.

    ``data.shape == (n_channels, n_lags, n_frames)``; ``x_values`` holds
    the lags in seconds (in Hz for the autocorrelation of a spectrum), or
    the corresponding frequencies ``1 / lag`` when computed with
    ``freq=True``.

    Parameters
    ----------
    of_spectrum : bool
        Whether this is the autocorrelation of a spectrum.
    normalized : bool
        Whether it has been divided by the window's autocorrelation.
    """

    name: str = "Autocor"
    x_name: str = "lag"
    x_unit: str = "s"
    of_spectrum: bool = False
    normalized: bool = False


def autocor(
    x: Signal,
    *,
    frame: FrameArg = False,
    min_lag: float | None = None,
    max_lag: float | None = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    scale: str = "coeff",
    generalized: float = STANDARD_EXPONENT,
    window: str | None = AUTO,
    normal_window: bool | None = None,
    halfwave: bool = False,
    freq: bool = False,
) -> Autocor:
    """Compute the autocorrelation of each frame.

    Equivalent to MIRtoolbox ``mirautocor``. Each frame is centred,
    windowed, autocorrelated, and divided by the autocorrelation of the
    window, which compensates for the window's taper.

    Parameters
    ----------
    x : Signal
        Audio (or any time signal), framed or not, or a :class:`Spectrum`
        (lags are then in Hz). An :class:`Autocor` is not recomputed; only
        `halfwave` applies.
    frame : bool, FrameSpec or tuple
        Frame the signal first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    min_lag, max_lag : float, optional
        Lag range in seconds (in Hz for a spectrum). Default: 0 to 50 ms
        for audio sampled above 1 kHz, otherwise no upper limit. Never
        beyond half the frame. MATLAB ``'Min'``, ``'Max'``.
    min_freq, max_freq : float, optional
        The same range given as frequencies: lags from ``1 / max_freq``
        to ``1 / min_freq``. MATLAB ``'Min', f, 'Hz'``. Not combinable
        with `min_lag`/`max_lag`.
    scale : {"coeff", "biased", "unbiased", "none", "coeff_x_channels"}
        Scaling of MATLAB ``xcorr``: lag 0 equals 1; divided by the frame
        length; by the overlap at each lag; none; or divided by lag 0 of
        the sum of all channels. MATLAB ``'coeff'``, ``'biased'``, etc.
    generalized : float
        Exponent ``k`` of the generalized autocorrelation
        ``ifft(abs(fft(x)) ** k)`` (Tolonen & Karjalainen, 2000). 2 is the
        ordinary autocorrelation. MATLAB ``'Generalized'`` / ``'Compres'``
        (``'Compres'`` alone means 0.67). With ``k != 2`` there is no
        window normalisation and `scale` is ignored.
    window : str or None
        Window applied to each frame. Default: MATLAB ``hanning`` for
        audio sampled at 1 kHz or more, rectangular below; none for a
        spectrum. MATLAB ``'Window'``.
    normal_window : bool, optional
        Divide by the window's autocorrelation. Default: on for signals,
        off for spectra. MATLAB ``'NormalWindow'``.
    halfwave : bool
        Set negative values to zero. MATLAB ``'Halfwave'``.
    freq : bool
        Express the lag axis as frequency, ``1 / lag``. MATLAB ``'Freq'``.

    Returns
    -------
    Autocor

    See Also
    --------
    librosa.autocorrelate : autocorrelation without windowing or lag units.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio, peaks
    >>> t = np.arange(4410) / 44100
    >>> a = autocor(Audio.from_array(np.sin(2 * np.pi * 200 * t), sr=44100))
    >>> first = peaks(a, no_begin=True, order="abscissa").peaks.pos[0]
    >>> round(float(first), 4)  # one period of 200 Hz
    0.005
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    if (min_freq, max_freq) != (None, None) and (min_lag, max_lag) != (None, None):
        raise ValueError("Give the lag range either in seconds or in Hz, not both")
    if isinstance(x, Autocor):
        return _halfwave(x) if halfwave else x

    x = framing.frame(x, frame)
    of_spectrum = isinstance(x, Spectrum)
    sampling = 1 / (x.x_values[1] - x.x_values[0]) if of_spectrum else x.sr
    explicit_range = (min_lag, max_lag, min_freq, max_freq) != (None,) * 4
    if min_freq is not None or max_freq is not None:
        min_lag, max_lag = _lags_from_freqs(min_freq, max_freq)
    if max_lag is None:
        max_lag = DEFAULT_MAX_LAG if not of_spectrum and x.sr > LOW_RATE else np.inf
    if normal_window is None:
        normal_window = not of_spectrum
    if window == AUTO:
        window = _default_window(normal_window, of_spectrum, x.sr)

    min_lag = min_lag or 0.0
    first, last = _main_lag_range(min_lag, max_lag, sampling, x.n_elements)
    if explicit_range:
        first, last = _after_lag_range(first, last, min_lag, max_lag, sampling)
    if max_freq is not None:
        # MIRtoolbox rounds a frequency bound up to a whole lag (fixtures, 2026-11-18)
        first = max(first, math.ceil(round(sampling / max_freq, LAG_DIGITS)))

    data, window_autocor = _correlate(x, first, last, scale, generalized, window)
    normalized = bool(normal_window and window_autocor is not None)
    if normalized:
        data = data / window_autocor
    lags = np.arange(first, last + 1) / sampling
    x_values = lags
    if freq:
        data, x_values = as_frequencies(data, lags)
    result = Autocor(
        data=data,
        sr=x.sr,
        x_values=x_values,
        x_name="frequency" if freq else "lag",
        x_unit="Hz" if freq or of_spectrum else "s",
        start=x.start,
        frame_rate=x.frame_rate,
        frame_length=x.frame_length,
        frame_starts=x.frame_starts,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
        of_spectrum=of_spectrum,
        normalized=normalized,
    )
    if halfwave:
        result = _halfwave(result)
    options = {
        "min_lag": min_lag, "max_lag": max_lag, "scale": scale, "generalized": generalized,
        "window": window, "normal_window": normal_window, "halfwave": halfwave, "freq": freq,
    }  # fmt: skip
    return replace(result, params={**result.params, "autocor": options})


def _lags_from_freqs(min_freq, max_freq):
    """Convert a frequency range (Hz) into a lag range (s)."""
    min_lag = None if max_freq is None else 1 / max_freq
    max_lag = None if not min_freq else 1 / min_freq
    return min_lag, max_lag


def _default_window(normal_window: bool, of_spectrum: bool, sr: float) -> str | None:
    if not normal_window:
        return None
    if of_spectrum or sr >= LOW_RATE:
        return "hanning"
    return "rectangular"


def _main_lag_range(min_lag, max_lag, sampling, n) -> tuple[int, int]:
    """Lag range in samples chosen by ``+sig/+autocor/main.m`` (0-based)."""
    first = math.floor(min_lag * sampling)
    if first + 1 > n:
        warnings.warn(
            "The lag range starts beyond the end of the signal; minimum lag set to 0.",
            stacklevel=3,
        )
        first = 0
    last = math.ceil(n / 2) - 1
    if np.isfinite(max_lag):
        last = min(last, math.ceil(max_lag * sampling))
    if last < first:
        warnings.warn("Signal or frame too short for this range of lags.", stacklevel=3)
    return first, last


def _after_lag_range(first, last, min_lag, max_lag, sampling) -> tuple[int, int]:
    """Second cut of ``+sig/+autocor/after.m``, with MATLAB ``round``."""
    first = max(first, int(matlab_round(min_lag * sampling)))
    if np.isfinite(max_lag):
        last = min(last, int(matlab_round(max_lag * sampling)))
    return first, last


def _correlate(x: Signal, first, last, scale, generalized, window):
    """Return the autocorrelation at lags ``first..last`` and the window's own."""
    data = x.data - x.data.mean(axis=1, keepdims=True)
    taper = None if window is None else matlab_window(window, x.n_elements)
    if taper is not None:
        data = data * taper[None, :, None]
    if generalized != STANDARD_EXPONENT:
        magnitude = np.abs(np.fft.fft(data, axis=1)) ** generalized
        return np.fft.ifft(magnitude, axis=1).real[:, first : last + 1], None

    lags = slice(last + first, 2 * last + 1)  # lag k sits at index last + k of xcorr(.., last)
    if scale == "coeff_x_channels":
        # Each channel is divided by lag 0 of the sum of all channels.
        energy = xcorr(data.sum(axis=0, keepdims=True), 0, "none", axis=1)
        c = xcorr(data, last, "none", axis=1)[:, lags] / energy
    else:
        c = xcorr(data, last, scale, axis=1)[:, lags]
    window_autocor = None
    if window == "hanning":
        # Quirk: indexed by position in the returned range, not by the actual lag
        positions = np.arange(last + 1 - first)
        window_autocor = hanning_autocor(positions, x.n_elements)[None, :, None]
    elif taper is not None:
        window_autocor = xcorr(taper, last, "coeff")[lags][None, :, None]
    return c, window_autocor


def hanning_autocor(positions: np.ndarray, n: int) -> np.ndarray:
    """Normalised autocorrelation of a continuous Hann window, as MIRtoolbox divides by.

    MIRtoolbox 1.8.2 does not divide by the autocorrelation of the sampled
    window but by that of a continuous Hann window, at ``tau = (j + 1) /
    (n + 1)``, where ``j`` is the position in the returned lag range: when
    the range starts at lag 53, lag 53 is divided by ``R(1 / (n + 1))``.
    Measured on its outputs (frame lengths 512 to 4096 samples, cut and
    full lag ranges, agreement to 1e-12), 2026-11-18::

        R(tau) = ((1 - tau) (2 + cos 2 pi tau) + 3 / (2 pi) sin 2 pi tau) / 3
    """
    tau = (np.asarray(positions) + 1) / (n + 1)
    angle = 2 * np.pi * tau
    return ((1 - tau) * (2 + np.cos(angle)) + 3 / (2 * np.pi) * np.sin(angle)) / 3


def _halfwave(a: Autocor) -> Autocor:
    return replace(a, data=np.maximum(a.data, 0))
