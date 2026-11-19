"""Energy features: RMS and low-energy rate.

``rms`` is translated from MiningSuite ``+sig/rms.m`` (BSD-3, Olivier
Lartillot), with one fix: MiningSuite divides the sum of squares by the
signal length in *seconds* (``Ssize``), which gives sqrt(sr) times the
RMS. mirpy follows MIRtoolbox 1.8.2 as measured on its outputs: a frame is
divided by its number of samples N, an unframed signal by N - 1 (RMS of
``ragtime.wav`` = 0.01793, as in the manual).

``lowenergy`` is not in MiningSuite. It is written from the MIRtoolbox
manual: the share of frames whose energy is below the average
(Tzanetakis & Cook, 2002), or the Average Silence Ratio (Feng, Zhuang &
Pan, 2003).
"""

import numpy as np

from mirpy.core import framing
from mirpy.core.data import ELEMENT_AXIS, Signal
from mirpy.core.options import FrameArg
from mirpy.features._common import feature_signal, with_params

LOW_ENERGY_THRESHOLD = 1.0
ASR_THRESHOLD = 0.5


def rms(x: Signal, *, frame: FrameArg = False, median: bool = False) -> Signal:
    """Compute the root-mean-square energy of the signal, or of each frame.

    Equivalent to MIRtoolbox ``mirrms``. Like MIRtoolbox, the sum of
    squares is divided by the frame length for framed signals, but by
    ``n_samples - 1`` for an unframed signal.

    Parameters
    ----------
    x : Signal
        Audio waveform, framed or not.
    frame : bool, FrameSpec or tuple
        Frame the signal first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    median : bool
        Take the median of the squares instead of their mean
        ("root-median-square"). MATLAB ``'Median'``.

    Returns
    -------
    Signal
        One value per channel and frame, ``data.shape == (n_channels, 1, n_frames)``.

    See Also
    --------
    librosa.feature.rms : framed RMS, centred frames by default.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> tone = Audio.from_array(np.sin(2 * np.pi * 100 * np.arange(8000) / 8000), sr=8000)
    >>> round(float(rms(tone, frame=True).data.mean()), 4)
    0.7071
    """
    x = framing.frame(x, frame)
    squares = x.data**2
    if median:
        values = np.sqrt(np.median(squares, axis=ELEMENT_AXIS))
    else:
        count = x.n_elements if x.is_framed else x.n_elements - 1
        values = np.sqrt(np.sum(squares, axis=ELEMENT_AXIS) / count)
    result = feature_signal(x, values, "RMS")
    return with_params(result, "rms", {"median": median})


def lowenergy(
    x: Signal,
    *,
    frame: FrameArg = True,
    threshold: float | None = None,
    asr: bool = False,
) -> Signal:
    """Compute the share of frames with less energy than the average.

    Equivalent to MIRtoolbox ``mirlowenergy``.

    Parameters
    ----------
    x : Signal
        Audio waveform, or a framed RMS curve from :func:`rms`.
    frame : bool, FrameSpec or tuple
        Frame settings for a waveform; ``True`` uses 50 ms frames with half
        overlap. A waveform is always framed. MATLAB ``'Frame'``.
    threshold : float, optional
        Ratio to the average energy below which a frame counts as low.
        Default 1, or 0.5 with `asr`. MATLAB ``'Threshold'``.
    asr : bool
        Average Silence Ratio: use the mean square (RMS without the square
        root). MATLAB ``'ASR'``.

    Returns
    -------
    Signal
        One value in [0, 1] per channel, unframed.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Signal
    >>> curve = Signal(np.array([1, 1, 1, 9.0]).reshape(1, 1, 4), sr=8000, frame_rate=40,
    ...                frame_length=0.05, frame_starts=np.arange(4) * 0.025)
    >>> float(lowenergy(curve).data.item())
    0.75
    """
    energy = _energy_curve(x, frame)
    if asr:
        energy = energy**2
    if threshold is None:
        threshold = ASR_THRESHOLD if asr else LOW_ENERGY_THRESHOLD
    average = energy.mean(axis=-1, keepdims=True)
    share = np.mean(energy < threshold * average, axis=-1)
    result = Signal(
        data=share[:, np.newaxis, np.newaxis],
        sr=x.sr,
        name="Low energy",
        start=x.start,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
    )
    return with_params(result, "lowenergy", {"threshold": threshold, "asr": asr})


def _energy_curve(x: Signal, frame: FrameArg) -> np.ndarray:
    """RMS of each frame, ``(n_channels, n_frames)``; `x` may already be one."""
    is_rms_curve = x.is_framed and x.n_elements == 1
    if not is_rms_curve:
        if not x.is_framed:
            x = framing.frame(x, frame or True)
        x = rms(x)
    return x.data[:, 0, :]
