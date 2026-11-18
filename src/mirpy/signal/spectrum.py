"""Magnitude spectrum, with auditory band and dB options.

Translated from MiningSuite ``+sig/spectrum.m``, ``+sig/+spectrum/``
(``options.m``, ``main.m``, ``after1.m``, ``after2.m``),
``+sig/@Spectrum/Spectrum.m``, ``+aud/spectrum.m`` and
``+aud/+spectrum/after.m`` (BSD-3, Olivier Lartillot).

MiningSuite quirks kept on purpose, to be checked against MIRtoolbox:

- ``terhardt=True`` zeroes the first bin even when it is not 0 Hz.
- ``gauss`` uses a kernel normalised by ``1 / (2 pi sigma)`` and is
  shifted by one sample (``y(4*sigma:end)`` in 1-based MATLAB).
"""

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import lfilter

from mirpy.core import framing
from mirpy.core.data import Signal
from mirpy.core.matlab_compat import matlab_window, nextpow2
from mirpy.core.options import FrameArg
from mirpy.signal import auditory_bands

BANDS = ("freq", "mel", "bark")
LINEAR, LOG10, DECIBEL = 0, 1, 10  # Spectrum.log_scale values
EPSILON = 1e-16  # added before log10 and after band weighting, as MiningSuite
GAUSS_HALF_WIDTH = 4  # kernel spans +-4 sigma


@dataclass(frozen=True, eq=False)
class Spectrum(Signal):
    """A magnitude spectrum, one per frame.

    ``data.shape == (n_channels, n_bins, n_frames)``; ``x_values`` holds
    the bin frequencies in Hz, or the band numbers (from 1) after Mel or
    Bark weighting. ``sr`` stays the sampling rate of the analysed signal.

    Parameters
    ----------
    power : {1, 2}
        1 for magnitude, 2 for power (squared magnitude).
    log_scale : {0, 1, 10}
        0 linear, 1 ``log10``, 10 decibels.
    phase : numpy.ndarray, optional
        Phase of each bin, same shape as `data`, when requested.
    """

    name: str = "Spectrum"
    x_name: str = "frequency"
    x_unit: str = "Hz"
    power: int = 1
    log_scale: int = LINEAR
    phase: np.ndarray | None = None

    @property
    def is_band_spectrum(self) -> bool:
        """Whether bins have been grouped into Mel or Bark bands."""
        return self.x_name != "frequency"


def spectrum(
    x: Signal,
    *,
    frame: FrameArg = False,
    window: str | None = "hamming",
    min_freq: float = 0.0,
    max_freq: float = np.inf,
    res: float | None = None,
    min_res: float = 0.0,
    length: int | None = None,
    zero_pad: int | bool = 0,
    phase: bool = False,
    normal: bool = False,
    power: bool = False,
    time_smooth: int = 0,
    band: str = "freq",
    n_bands: int | None = None,
    terhardt: bool = False,
    mask: bool = False,
    log: bool = False,
    db: bool | float = False,
    smooth: int = 0,
    gauss: int = 0,
) -> Spectrum:
    """Compute the magnitude spectrum of each frame.

    Equivalent to MIRtoolbox ``mirspectrum``. The FFT is not normalised.
    By default the FFT length is the next power of two at or above the
    frame length.

    Parameters
    ----------
    x : Signal
        Audio (or any time signal), framed or not. A :class:`Spectrum` is
        not recomputed: only the post-processing options are applied.
    frame : bool, FrameSpec or tuple
        Frame the signal first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    window : str or None
        Window applied to each frame: ``"hamming"`` (MATLAB symmetric
        form), ``"hanning"``, any :func:`scipy.signal.get_window` name, or
        ``None``/``"rectangular"``. MATLAB ``'Window'``.
    min_freq, max_freq : float
        Keep only bins between these frequencies (Hz). MATLAB ``'Min'``,
        ``'Max'``.
    res : float, optional
        Frequency resolution in Hz; sets the FFT length to
        ``ceil(sr / res)``. MATLAB ``'Res'``.
    min_res : float
        Minimum frequency resolution in Hz; zero-pads frames that are too
        short to reach it. MATLAB ``'MinRes'``.
    length : int, optional
        FFT length in samples. MATLAB ``'Length'``.
    zero_pad : int or bool
        Number of zeros appended before the FFT; ``True`` appends as many
        as the frame length. MATLAB ``'ZeroPad'``.
    phase : bool
        Keep the phase in :attr:`Spectrum.phase`. MATLAB ``'Phase'``.
    normal : bool
        Divide each frame by its Euclidean norm. MATLAB ``'Normal'``.
    power : bool
        Square the magnitude. MATLAB ``'Power'``.
    time_smooth : int
        Moving average over this many frames. MATLAB ``'TimeSmooth'``.
    band : {"freq", "mel", "bark"}
        Group the bins into Mel bands or critical (Bark) bands. MATLAB
        ``'Mel'``, ``'Bark'``.
    n_bands : int, optional
        Number of Mel bands (default 40). MATLAB ``'Bands'``.
    terhardt : bool
        Weight by Terhardt's outer-ear model. MATLAB ``'Terhardt'``.
    mask : bool
        Apply the critical-band masking spread (Mel or Bark spectra only).
        MATLAB ``'Mask'``.
    log : bool
        ``log10`` of the values. MATLAB ``'log'``.
    db : bool or float
        Decibels. A number keeps only that many dB below the maximum of
        each channel, shifted to start at 0. MATLAB ``'dB'``.
    smooth : int
        Moving average over this many bins. MATLAB ``'Smooth'``.
    gauss : int
        Gaussian smoothing over bins, with this standard deviation (in
        bins). MATLAB ``'Gauss'``.

    Returns
    -------
    Spectrum

    See Also
    --------
    librosa.stft : complex short-time Fourier transform.
    librosa.feature.melspectrogram : Mel spectrogram (Slaney filters by default).

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> tone = Audio.from_array(np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000), sr=8000)
    >>> s = spectrum(tone)
    >>> float(s.x_values[np.argmax(s.data[0, :, 0])])
    1000.0
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    if isinstance(x, Spectrum):
        result = x
    else:
        result = _fft_magnitude(
            framing.frame(x, frame), window, res, min_res, length, zero_pad, phase
        )
    if min_freq > 0 or max_freq < np.inf:
        result = _band_cut(result, min_freq, max_freq)
    if time_smooth:
        result = replace(result, data=_moving_average(result.data, time_smooth, axis=2))
    if power and result.power == 1:
        result = replace(result, data=result.data**2, power=2, name=f"Power {result.name}")
    if normal:
        result = replace(
            result, data=result.data / np.linalg.norm(result.data, axis=1, keepdims=True)
        )
    if terhardt:
        weights = auditory_bands.terhardt_weights(result.x_values)
        result = replace(result, data=result.data * weights[None, :, None])
    if band != "freq" and not result.is_band_spectrum:
        result = _to_bands(result, band, n_bands)
    if mask:
        result = _mask(result)
    if log or db:
        result = _to_log(result, db)
    if smooth:
        result = replace(result, data=_moving_average(result.data, smooth, axis=1))
    if gauss:
        result = replace(result, data=_gauss_smooth(result.data, gauss))
    options = {
        "window": window, "min_freq": min_freq, "max_freq": max_freq, "res": res,
        "min_res": min_res, "length": length, "zero_pad": zero_pad, "phase": phase,
        "normal": normal, "power": power, "time_smooth": time_smooth, "band": band,
        "n_bands": n_bands, "terhardt": terhardt, "mask": mask, "log": log, "db": db,
        "smooth": smooth, "gauss": gauss,
    }  # fmt: skip
    return replace(result, params={**result.params, "spectrum": options})


def fft_length(n: int, sr: float, res, min_res, length, zero_pad) -> int:
    """Return the FFT length chosen by ``+sig/+spectrum/main.m``."""
    if zero_pad:
        n += n if zero_pad is True else zero_pad
    if length is not None:
        return int(length)
    if res is not None:
        return math.ceil(sr / res)
    if min_res and n < sr / min_res:
        n = max(n, sr / min_res)
    return 2 ** nextpow2(n)


def _fft_magnitude(x: Signal, window, res, min_res, length, zero_pad, keep_phase) -> Spectrum:
    data = x.data
    if window not in (None, "rectangular"):
        data = data * matlab_window(window, x.n_elements)[None, :, None]
    n_fft = fft_length(x.n_elements, x.sr, res, min_res, length, zero_pad)
    coefficients = np.fft.rfft(data, n=n_fft, axis=1)
    return Spectrum(
        data=np.abs(coefficients),
        sr=x.sr,
        x_values=np.arange(coefficients.shape[1]) * x.sr / n_fft,
        start=x.start,
        frame_rate=x.frame_rate,
        frame_length=x.frame_length,
        frame_starts=x.frame_starts,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
        phase=np.angle(coefficients) if keep_phase else None,
    )


def _band_cut(s: Spectrum, min_freq: float, max_freq: float) -> Spectrum:
    keep = np.flatnonzero((s.x_values >= min_freq) & (s.x_values <= max_freq))
    if keep.size == 0:
        raise ValueError(f"No spectrum bin between {min_freq} and {max_freq} Hz")
    bins = slice(keep[0], keep[-1] + 1)
    phase = None if s.phase is None else s.phase[:, bins]
    return replace(s, data=s.data[:, bins], x_values=s.x_values[bins], phase=phase)


def _to_bands(s: Spectrum, band: str, n_bands: int | None) -> Spectrum:
    if band == "mel":
        weights = auditory_bands.mel_weights(
            s.x_values, n_bands or auditory_bands.DEFAULT_MEL_BANDS
        )
        if auditory_bands.has_undefined_bands(weights):
            warnings.warn(
                "Spectrum resolution too low for the Mel transformation; some Mel bands "
                "are empty. Use a frequency resolution of at least 66 Hz.",
                stacklevel=3,
            )
        label = "Mel"
    else:
        weights = auditory_bands.bark_weights(s.x_values)
        label = "Bark"
    data = np.einsum("bf,cft->cbt", weights, s.data) + EPSILON
    return replace(
        s,
        data=data,
        x_values=np.arange(1, weights.shape[0] + 1, dtype=float),
        x_name=f"{label} bands",
        x_unit="",
        name=f"{label}-Spectrum",
        phase=None,
    )


def _mask(s: Spectrum) -> Spectrum:
    if not s.is_band_spectrum:
        warnings.warn("mask=True needs a Mel or Bark spectrum; option ignored.", stacklevel=3)
        return s
    spread = auditory_bands.mask_spread(s.n_elements)
    return replace(s, data=np.einsum("ij,cjt->cit", spread, s.data) + EPSILON)


def _to_log(s: Spectrum, db: bool | float) -> Spectrum:
    data, log_scale, power = s.data, s.log_scale, s.power
    if log_scale == LINEAR:
        data, log_scale = np.log10(data + EPSILON), LOG10
    if db and log_scale == LOG10:
        data = data * (20 if power == 1 else 10)
        log_scale, power = DECIBEL, 2
    if db is not True and db and np.isfinite(db):
        data = data - data.max(axis=(1, 2), keepdims=True)
        data = np.maximum(data, -db) + db
    return replace(s, data=data, log_scale=log_scale, power=power, phase=None)


def _moving_average(data: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Causal moving average, MATLAB ``filter(ones(1,n)/n, 1, x)``."""
    return lfilter(np.ones(n) / n, 1, data, axis=axis)


def _gauss_smooth(data: np.ndarray, sigma: int) -> np.ndarray:
    """Gaussian smoothing over bins, as ``gausssmooth`` in ``after2.m``."""
    half = GAUSS_HALF_WIDTH * sigma
    k = np.arange(-half, half + 1)
    kernel = 1 / sigma / 2 / np.pi * np.exp(-(k**2) / 2 / sigma**2)
    padded = np.pad(data, ((0, 0), (0, half), (0, 0)))
    smoothed = lfilter(kernel, 1, padded, axis=1)
    return smoothed[:, half - 1 : half - 1 + data.shape[1]]
