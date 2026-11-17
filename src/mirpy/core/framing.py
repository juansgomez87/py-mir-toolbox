"""Frame decomposition.

Translated from MiningSuite ``+sig/frame.m`` (BSD-3, Olivier Lartillot).
"""

import math
import warnings
from dataclasses import replace

import numpy as np

from mirpy.core.data import Signal
from mirpy.core.options import FrameArg, FrameSpec, as_frame_spec

DEFAULT_FRAME = FrameSpec()


def frame(x: Signal, spec: FrameArg = True) -> Signal:
    """Decompose a signal into successive, possibly overlapping frames.

    Equivalent to MIRtoolbox ``mirframe``. The frame length is rounded down
    to a whole number of samples; the hop may be fractional, and frame
    ``i`` (0-based) starts at sample ``floor(i * hop)``. The number of
    frames is ``floor((n_samples - length) / hop) + 1``, so the end of the
    signal that does not fill a whole frame is dropped.

    Parameters
    ----------
    x : Signal
        An unframed signal.
    spec : bool, FrameSpec or tuple
        Frame settings. ``True`` uses 50 ms frames with a hop of half a
        frame. See :func:`mirpy.core.options.as_frame_spec`.

    Returns
    -------
    Signal
        Same type as `x`, with ``data.shape == (n_channels, length, n_frames)``.
        `x` is returned unchanged, with a warning, when it is already
        framed, when the frame length is shorter than one sample, or when
        it is longer than the signal.

    See Also
    --------
    librosa.util.frame : frames with an integer hop and no unit handling.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy.core.data import Audio
    >>> x = Audio.from_array(np.zeros(1000), sr=1000)
    >>> frame(x, (0.1, 0.5)).data.shape
    (1, 100, 19)
    """
    frame_spec = as_frame_spec(spec, DEFAULT_FRAME)
    if frame_spec is None:
        return x
    if x.is_framed:
        warnings.warn("Signal is already decomposed into frames; ignored.", stacklevel=2)
        return x

    length = frame_spec.length_samples(x.sr)
    if length == 0:
        warnings.warn(
            "Frame length shorter than one sample; no frame decomposition.", stacklevel=2
        )
        return x

    starts = frame_start_samples(x.n_elements, length, frame_spec.hop_samples(x.sr), x.start, x.sr)
    if starts.size == 0:
        warnings.warn("Frame length longer than the signal; no frame decomposition.", stacklevel=2)
        return x

    indices = starts[:, np.newaxis] + np.arange(length)
    framed = x.data[:, indices, 0]  # (channel, frame, element)
    return replace(
        x,
        data=np.moveaxis(framed, 1, -1),
        x_values=None,
        frame_rate=frame_spec.frame_rate(x.sr),
        frame_length=frame_spec.length_seconds(x.sr),
        frame_starts=x.start + starts / x.sr,
        params={**x.params, "frame": frame_spec},
    )


def frame_start_samples(
    n_samples: int, length: int, hop: float, start: float, sr: float
) -> np.ndarray:
    """Return the 0-based first sample of each frame.

    Frames are aligned on a grid of multiples of `hop` counted from time 0,
    so a signal that starts at ``start > 0`` is cut at the same instants as
    the full recording. Arithmetic follows ``frame.m`` step by step (in
    1-based indices) so that floating-point rounding matches MATLAB.

    Parameters
    ----------
    n_samples : int
        Number of samples in the signal.
    length : int
        Frame length in samples.
    hop : float
        Hop in samples, possibly fractional.
    start : float
        Time of the first sample, in seconds.
    sr : float
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray of int
        Empty when no full frame fits in the signal.
    """
    first_sample = math.floor(start * sr) + 1
    first_frame = math.ceil((first_sample - 1) / hop) + 1
    offset = math.floor((first_frame - 1) * hop) + 1 - first_sample
    n_frames = math.floor((n_samples - length - offset) / hop) + 1
    if n_frames < 1:
        return np.empty(0, dtype=np.intp)
    one_based = np.floor(np.arange(n_frames) * hop + 1).astype(np.intp) + offset
    return one_based - 1
