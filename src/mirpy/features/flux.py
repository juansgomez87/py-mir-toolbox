"""Flux: distance between successive frames.

Translated from MiningSuite ``+sig/flux.m``, ``+sig/+flux/main.m``,
``+sig/+flux/options.m`` and ``+aud/flux.m`` (BSD-3, Olivier Lartillot).
The ``median`` scaling factor defaults to 1, as in the MIRtoolbox manual
(MiningSuite uses 1.3).

MiningSuite quirks kept on purpose, to be checked against MIRtoolbox, in
``complex`` flux:

- the phase deviation is wrapped in *cycles* (``d/(2 pi) - round(d/(2
  pi))``) but then passed to ``cos`` as if it were in radians;
- value ``k`` compares frames ``k`` to ``k + 2`` but is placed at frame
  ``k + 1``, like plain flux, so onsets show one frame early.
"""

from dataclasses import replace

import numpy as np

from mirpy.core import framing
from mirpy.core.data import ELEMENT_AXIS, FRAME_AXIS, Signal
from mirpy.core.matlab_compat import matlab_round
from mirpy.core.options import FrameArg
from mirpy.features._common import feature_signal, with_params
from mirpy.signal.filterbank import filterbank
from mirpy.signal.spectrum import Spectrum, spectrum

DISTANCES = ("euclidean", "cityblock", "cosine")
INC_DISTANCES = ("euclidean", "cityblock")
DISTANCE_ALIASES = {"city": "cityblock"}
SUB_BANDS = ("manual", "gammatone", "2channels")
# +aud/flux.m: 10 bands with edges at 50 Hz * 2^k, order-2 filters
SUB_BAND_CUTOFFS = [-np.inf, *(50.0 * 2.0 ** np.arange(9)), np.inf]
SUB_BAND_ORDER = 2
DEFAULT_MEDIAN = (0.2, 1.0)  # window length in seconds, scaling factor


def flux(
    x: Signal,
    *,
    frame: FrameArg = True,
    dist: str = "euclidean",
    inc: bool = False,
    complex: bool = False,
    median: tuple[float, float] | bool | None = None,
    halfwave: bool = False,
    sub_band: str | None = None,
) -> Signal:
    """Compute the distance between each frame and the next, e.g. the spectral flux.

    Equivalent to MIRtoolbox ``mirflux``. Value ``i`` compares frames
    ``i`` and ``i + 1`` and is placed at frame ``i + 1``.

    Parameters
    ----------
    x : Signal
        Audio waveform (framed, then turned into a spectrum), or any
        framed curve: spectrum, MFCC, chromagram, ...
    frame : bool, FrameSpec or tuple
        Frame settings for a waveform, which is always framed; ``True``
        uses 50 ms frames with half overlap. MATLAB ``'Frame'``.
    dist : {"euclidean", "cityblock", "cosine"}
        Distance between successive frames. ``"city"`` is accepted for
        ``"cityblock"``. MATLAB ``'Dist'``.
    inc : bool
        Sum only increases between frames (``"euclidean"`` or
        ``"cityblock"`` only). MATLAB ``'Inc'``.
    complex : bool
        Use magnitude and phase (Bello et al., 2004); spectral flux only.
        Gives one value fewer. MATLAB ``'Complex'``.
    median : (float, float) or bool, optional
        Subtract ``C`` times the moving median over ``l`` seconds on each
        side; ``True`` uses ``(0.2, 1.0)``. MATLAB ``'Median', l, C``.
    halfwave : bool
        Set negative values to 0 (after `median`). MATLAB ``'Halfwave'``.
    sub_band : {"manual", "gammatone", "2channels"}, optional
        Split a waveform into frequency bands first and compute one flux
        per band. ``"manual"`` is MIRtoolbox's default: 10 bands with
        edges at 50 Hz times powers of 2. MATLAB ``'SubBand'``.

    Returns
    -------
    Signal
        One value per channel and pair of frames.

    See Also
    --------
    librosa.onset.onset_strength : spectral flux on a log-Mel spectrogram, increases only.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> t = np.arange(8000) / 8000
    >>> step = Audio.from_array(np.where(t < 0.5, 0.0, np.sin(2 * np.pi * 440 * t)), sr=8000)
    >>> f = flux(step)
    >>> 0.45 < float(f.frame_starts[np.argmax(f.data[0, 0])]) < 0.5
    True
    """
    dist = DISTANCE_ALIASES.get(dist.lower(), dist.lower())
    _check_options(dist, inc, sub_band)
    curves = _framed_curves(x, frame, complex, sub_band)
    if complex:
        if not isinstance(curves, Spectrum) or curves.phase is None:
            raise ValueError("complex=True is only defined for spectral flux of a waveform")
        values = _complex_flux(curves.data, curves.phase)
    elif inc:
        values = _increase_flux(curves.data, dist)
    else:
        values = _flux(curves.data, dist)
    starts = curves.frame_starts[1 : 1 + values.shape[-1]]
    result = feature_signal(curves, values, _name(curves), frame_starts=starts)
    if median:
        result = _subtract_median(result, *(DEFAULT_MEDIAN if median is True else median))
    if halfwave:
        result = replace(result, data=np.maximum(result.data, 0))
    options = {
        "dist": dist, "inc": inc, "complex": complex, "median": median,
        "halfwave": halfwave, "sub_band": sub_band,
    }  # fmt: skip
    return with_params(result, "flux", options)


def _check_options(dist: str, inc: bool, sub_band: str | None) -> None:
    allowed = INC_DISTANCES if inc else DISTANCES
    if dist not in allowed:
        raise ValueError(f"dist must be one of {allowed}, got {dist!r}")
    if sub_band is not None and sub_band not in SUB_BANDS:
        raise ValueError(f"sub_band must be one of {SUB_BANDS}, got {sub_band!r}")


def _framed_curves(x: Signal, frame: FrameArg, keep_phase: bool, sub_band) -> Signal:
    """Return the framed curves to compare; a waveform becomes a spectrogram."""
    if x.x_values is not None:
        if not x.is_framed:
            raise ValueError("flux needs a framed input")
        return x
    if sub_band == "manual":
        x = filterbank(x, cutoffs=SUB_BAND_CUTOFFS, order=SUB_BAND_ORDER)
    elif sub_band is not None:
        x = filterbank(x, sub_band)
    if not x.is_framed:
        x = framing.frame(x, frame or True)
    return spectrum(x, phase=keep_phase)


def _flux(d: np.ndarray, dist: str) -> np.ndarray:
    current, following = d[..., :-1], d[..., 1:]
    if dist == "euclidean":
        return np.sqrt(np.sum((following - current) ** 2, axis=ELEMENT_AXIS))
    if dist == "cityblock":
        return np.sum(np.abs(following - current), axis=ELEMENT_AXIS)
    dot = np.sum(current * following, axis=ELEMENT_AXIS)
    norms = np.linalg.norm(current, axis=ELEMENT_AXIS) * np.linalg.norm(
        following, axis=ELEMENT_AXIS
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - dot / norms


def _increase_flux(d: np.ndarray, dist: str) -> np.ndarray:
    increase = np.maximum(0, np.diff(d, axis=FRAME_AXIS))
    if dist == "euclidean":
        return np.sqrt(np.sum(increase**2, axis=ELEMENT_AXIS))
    return np.sum(increase, axis=ELEMENT_AXIS)


def _complex_flux(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Bello et al. (2004) deviation from the predicted magnitude and phase."""
    deviation = np.diff(phase, n=2, axis=FRAME_AXIS) / (2 * np.pi)
    deviation = deviation - matlab_round(deviation)  # in cycles; see module docstring
    now, before = magnitude[..., 2:], magnitude[..., 1:-1]
    squared = now**2 + before**2 - 2 * now * before * np.cos(deviation)
    # Rounding can push it just below 0; MATLAB's sqrt would then give a
    # tiny imaginary number, whose real part is 0.
    distance = np.sqrt(np.maximum(squared, 0))
    return np.sum(distance, axis=ELEMENT_AXIS)


def _subtract_median(x: Signal, length: float, scale: float) -> Signal:
    """Subtract a scaled moving median over ``round(length * frame_rate)`` frames per side."""
    half = int(matlab_round(length * x.frame_rate))
    data = x.data.copy()
    n = x.n_frames
    for i in range(n):
        window = x.data[..., max(0, i - half) : min(n, i + half + 1)]
        data[..., i] = x.data[..., i] - scale * np.median(window, axis=FRAME_AXIS)
    return replace(x, data=data)


def _name(x: Signal) -> str:
    if isinstance(x, Spectrum):
        return "Spectral flux"
    return f"{x.name} flux"
