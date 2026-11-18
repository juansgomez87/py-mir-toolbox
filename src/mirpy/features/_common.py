"""Helpers shared by the feature operators."""

from dataclasses import replace

import numpy as np

from mirpy.core.data import ELEMENT_AXIS, Signal
from mirpy.core.options import FrameArg
from mirpy.signal.spectrum import Spectrum, spectrum


def as_distribution(x: Signal, frame: FrameArg) -> Signal:
    """Return the curve a spectral statistic is computed on.

    A waveform (no ``x_values``) is framed if asked and turned into a
    :class:`~mirpy.Spectrum` with default options, as MIRtoolbox does. A
    signal that already has positions along its elements (spectrum,
    autocorrelation, cepstrum, ...) is used as is.
    """
    if x.x_values is not None:
        return x
    return spectrum(x, frame=frame)


def feature_signal(x: Signal, values: np.ndarray, name: str, unit: str = "") -> Signal:
    """Wrap per-frame values of shape ``(n_channels, n_frames)`` in a :class:`Signal`.

    Keeps the sampling and framing of `x`, so the values sit at its frame
    centres.
    """
    return Signal(
        data=values[:, np.newaxis, :],
        sr=x.sr,
        name=name,
        unit=unit,
        start=x.start,
        frame_rate=x.frame_rate,
        frame_length=x.frame_length,
        frame_starts=x.frame_starts,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
    )


def quiet_frames(x: Signal, min_rms: float) -> np.ndarray:
    """Return a ``(n_channels, n_frames)`` mask of quasi-silent frames.

    MIRtoolbox ``'MinRMS'``: a frame is quasi-silent when the RMS of its
    spectrum is below `min_rms` times the highest RMS over the frames.
    Written from the MIRtoolbox manual (MiningSuite has no such option);
    taking the RMS over the spectrum bins is an assumption to check
    against the fixtures. Only spectra are gated.
    """
    if not min_rms or not isinstance(x, Spectrum):
        return np.zeros((x.n_channels, x.n_frames), dtype=bool)
    energy = np.sqrt(np.mean(x.data**2, axis=ELEMENT_AXIS))
    return energy < min_rms * energy.max(axis=-1, keepdims=True)


def distribution(d: np.ndarray) -> np.ndarray:
    """Normalise each curve of ``(channel, element, frame)`` data to sum to 1."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / d.sum(axis=ELEMENT_AXIS, keepdims=True)


def with_params(x: Signal, operator: str, options: dict) -> Signal:
    """Record the options an operator used, for export metadata."""
    return replace(x, params={**x.params, operator: options})
