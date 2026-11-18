"""Zero-crossing rate.

Translated from MiningSuite ``+sig/zerocross.m`` (BSD-3, Olivier Lartillot).
"""

import numpy as np

from mirpy.core import framing
from mirpy.core.data import ELEMENT_AXIS, Signal
from mirpy.core.options import FrameArg
from mirpy.features._common import feature_signal, with_params

PER = ("second", "sample")
DIRECTIONS = ("one", "both")


def zerocross(
    x: Signal, *, frame: FrameArg = False, per: str = "second", direction: str = "one"
) -> Signal:
    """Compute the rate of sign changes of the signal, or of each frame.

    Equivalent to MIRtoolbox ``mirzerocross``. The number of sign changes
    is divided by the number of samples (not the number of sample pairs),
    as in MiningSuite.

    Parameters
    ----------
    x : Signal
        Audio waveform, framed or not, or any curve with evenly spaced
        ``x_values`` (the rate is then per unit of the x axis).
    frame : bool, FrameSpec or tuple
        Frame the signal first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    per : {"second", "sample"}
        Sign changes per second, or per sample. MATLAB ``'Per'``.
    direction : {"one", "both"}
        Count changes in one direction only (half of all changes), or in
        both. MATLAB ``'Dir'``.

    Returns
    -------
    Signal
        One value per channel and frame.

    See Also
    --------
    librosa.feature.zero_crossing_rate : per-sample rate, both directions, centred frames.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> tone = Audio.from_array(np.sin(2 * np.pi * 100 * np.arange(8000) / 8000 + 0.1), sr=8000)
    >>> float(zerocross(tone).data.item())
    100.0
    """
    if per not in PER:
        raise ValueError(f"per must be one of {PER}, got {per!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    x = framing.frame(x, frame)
    d = x.data
    changes = np.sum(d[:, 1:] * d[:, :-1] < 0, axis=ELEMENT_AXIS).astype(float)
    if per == "second":
        changes *= _rate(x)
    if direction == "one":
        changes /= 2
    result = feature_signal(x, changes / x.n_elements, "Zero-crossing rate")
    return with_params(result, "zerocross", {"per": per, "direction": direction})


def _rate(x: Signal) -> float:
    """Elements per second (sampling rate), or per x unit for other curves."""
    if x.x_values is None:
        return x.sr
    return 1 / (x.x_values[1] - x.x_values[0])
