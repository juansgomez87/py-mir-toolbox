"""Statistical description of a spectral distribution.

Translated from MiningSuite ``+sig/centroid.m``, ``+sig/+centroid/algo.m``,
``+sig/spread.m``, ``+sig/+spread/algo.m``, ``+sig/skewness.m``,
``+sig/kurtosis.m``, ``+sig/flatness.m`` and ``+sig/entropy.m`` (BSD-3,
Olivier Lartillot). Changes made to follow the MIRtoolbox manual:

- On a waveform, the statistics are computed on its spectrum
  (MiningSuite ``centroid`` defaults to the waveform itself).
- The moments are taken over the curve plus its own minimum (per channel
  and frame). Not in MiningSuite or the manual; found with the ragtime
  fixtures, where it matches MIRtoolbox to 1e-15 (it explains the "slight
  overestimate" of the centroid reported by Kazazis et al. 2017).
- The ``'MinRMS'`` and ``'MaxEntropy'`` gates, absent from MiningSuite,
  are written from the manual; see :func:`mirpy.features._common.quiet_frames`.
- Skewness and kurtosis are NaN wherever the spread is (its ``'MinRMS'``
  gate), as in the MIRtoolbox 1.8.2 fixtures of all four demo files.
"""

import numpy as np

from mirpy.core.data import ELEMENT_AXIS, Signal
from mirpy.core.options import FrameArg
from mirpy.features._common import (
    as_distribution,
    distribution,
    feature_signal,
    quiet_frames,
    relative_entropy,
    with_params,
)
from mirpy.signal.spectrum import Spectrum

# MIRtoolbox manual defaults. The centroid 'MinRMS' default is clipped in
# the manual's PDF (".00..."); 0.005 as for entropy, to confirm.
CENTROID_MIN_RMS = 0.005
CENTROID_MAX_ENTROPY = 0.95
SPREAD_MIN_RMS = 0.01
FLATNESS_MIN_RMS = 0.01
ENTROPY_MIN_RMS = 0.005


def centroid(
    x: Signal,
    *,
    frame: FrameArg = False,
    min_rms: float = CENTROID_MIN_RMS,
    max_entropy: float | None = CENTROID_MAX_ENTROPY,
) -> Signal:
    """Compute the centroid (first moment) of a distribution, e.g. the spectral centroid.

    Equivalent to MIRtoolbox ``mircentroid``.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    min_rms : float
        Frames whose spectrum RMS is below this ratio of the highest one
        give NaN. 0 disables. MATLAB ``'MinRMS'``.
    max_entropy : float or None
        Frames whose :func:`entropy` is above this value give NaN (flat
        spectra, e.g. noise or low-pitched tones). ``None`` or 1
        disables. MATLAB ``'MaxEntropy'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN for quasi-silent frames.

    See Also
    --------
    librosa.feature.spectral_centroid : on the STFT magnitude, no gates.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> tone = Audio.from_array(np.sin(2 * np.pi * 1024 * np.arange(8192) / 8192), sr=8192)
    >>> round(float(centroid(tone).data.item()))
    1024
    """
    d = as_distribution(x, frame)
    values = _centroid(d.data, d.x_values)
    gated = quiet_frames(d, min_rms)
    if max_entropy is not None and max_entropy < 1:
        gated |= relative_entropy(d.data, center=False) > max_entropy
    values[gated] = np.nan
    result = feature_signal(d, values, _name(d, "centroid"), d.x_unit)
    return with_params(result, "centroid", {"min_rms": min_rms, "max_entropy": max_entropy})


def spread(x: Signal, *, frame: FrameArg = False, min_rms: float = SPREAD_MIN_RMS) -> Signal:
    """Compute the standard deviation of a distribution around its centroid.

    Equivalent to MIRtoolbox ``mirspread``.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    min_rms : float
        Quasi-silent frame threshold; see :func:`centroid`. MATLAB ``'MinRMS'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN for quasi-silent frames.

    See Also
    --------
    librosa.feature.spectral_bandwidth : same quantity for ``p=2``.
    """
    d = as_distribution(x, frame)
    values = _spread(d.data, d.x_values)
    values[quiet_frames(d, min_rms)] = np.nan
    result = feature_signal(d, values, _name(d, "spread"), d.x_unit)
    return with_params(result, "spread", {"min_rms": min_rms})


def skewness(x: Signal, *, frame: FrameArg = False) -> Signal:
    """Compute the skewness (third standardised moment) of a distribution.

    Equivalent to MIRtoolbox ``mirskewness``.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN where :func:`spread` is NaN.
    """
    d = as_distribution(x, frame)
    values = _standard_moment(d.data, d.x_values, 3)
    values[quiet_frames(d, SPREAD_MIN_RMS)] = np.nan
    return with_params(feature_signal(d, values, _name(d, "skewness")), "skewness", {})


def kurtosis(x: Signal, *, frame: FrameArg = False) -> Signal:
    """Compute the kurtosis (fourth standardised moment) of a distribution.

    Equivalent to MIRtoolbox ``mirkurtosis``. Not the excess kurtosis: a
    Gaussian gives 3, as in MIRtoolbox (the manual says excess).

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN where :func:`spread` is NaN.
    """
    d = as_distribution(x, frame)
    values = _standard_moment(d.data, d.x_values, 4)
    values[quiet_frames(d, SPREAD_MIN_RMS)] = np.nan
    return with_params(feature_signal(d, values, _name(d, "kurtosis")), "kurtosis", {})


def flatness(x: Signal, *, frame: FrameArg = False, min_rms: float = FLATNESS_MIN_RMS) -> Signal:
    """Compute the ratio of the geometric to the arithmetic mean of a distribution.

    Equivalent to MIRtoolbox ``mirflatness``. Close to 1 for a flat
    (noise-like) spectrum, close to 0 for a peaky (tonal) one. The
    geometric mean is computed as MiningSuite does, as a product of
    ``d ** (1/n)``, to keep its rounding.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    min_rms : float
        Quasi-silent frame threshold; see :func:`centroid`. MATLAB ``'MinRMS'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN for quasi-silent frames.

    See Also
    --------
    librosa.feature.spectral_flatness : on the power spectrum, with an amplitude floor.
    """
    d = as_distribution(x, frame)
    n = d.n_elements
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        geometric = np.prod(d.data ** (1 / n), axis=ELEMENT_AXIS)
        values = geometric / np.mean(d.data, axis=ELEMENT_AXIS)
    values[np.isinf(values)] = 0
    values[quiet_frames(d, min_rms)] = np.nan
    result = feature_signal(d, values, _name(d, "flatness"))
    return with_params(result, "flatness", {"min_rms": min_rms})


def entropy(
    x: Signal, *, frame: FrameArg = False, center: bool = False, min_rms: float = ENTROPY_MIN_RMS
) -> Signal:
    """Compute the relative Shannon entropy of a distribution, between 0 and 1.

    Equivalent to MIRtoolbox ``mirentropy``. Negative values are set to 0,
    the curve is scaled to sum to 1, and the entropy is divided by
    ``log(n_elements)``, the entropy of a flat curve.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first), a
        :class:`~mirpy.Spectrum`, or any curve with ``x_values``.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    center : bool
        Subtract the mean of each curve before removing negative values.
        MATLAB ``'Center'``.
    min_rms : float
        Quasi-silent frame threshold; see :func:`centroid`. MATLAB ``'MinRMS'``.

    Returns
    -------
    Signal
        One value per channel and frame. NaN for quasi-silent frames.
    """
    d = as_distribution(x, frame)
    values = relative_entropy(d.data, center)
    values[quiet_frames(d, min_rms)] = np.nan
    result = feature_signal(d, values, _name(d, "entropy"))
    return with_params(result, "entropy", {"center": center, "min_rms": min_rms})


def _name(d: Signal, statistic: str) -> str:
    """E.g. "Spectral centroid" for a spectrum, "Centroid" otherwise."""
    if isinstance(d, Spectrum):
        return f"Spectral {statistic}"
    return statistic.capitalize()


def _positions(f: np.ndarray) -> np.ndarray:
    return f[np.newaxis, :, np.newaxis]


def _weights(d: np.ndarray) -> np.ndarray:
    """Each curve plus its minimum, normalised to sum to 1 (MIRtoolbox behaviour)."""
    return distribution(d + d.min(axis=ELEMENT_AXIS, keepdims=True))


def _centroid(d: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.sum(_positions(f) * _weights(d), axis=ELEMENT_AXIS)


def _spread(d: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.sqrt(_central_moment(d, f, 2))


def _central_moment(d: np.ndarray, f: np.ndarray, order: int) -> np.ndarray:
    deviation = _positions(f) - _centroid(d, f)[:, np.newaxis, :]
    return np.sum(deviation**order * _weights(d), axis=ELEMENT_AXIS)


def _standard_moment(d: np.ndarray, f: np.ndarray, order: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _central_moment(d, f, order) / _spread(d, f) ** order
