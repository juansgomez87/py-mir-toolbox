"""High-frequency energy: roll-off and brightness.

Translated from MiningSuite ``+sig/rolloff.m`` and ``+aud/brightness.m``
(BSD-3, Olivier Lartillot). The ``'MinRMS'`` gate, absent from
MiningSuite, is written from the MIRtoolbox manual.
"""

import numpy as np

from mirpy.core.data import ELEMENT_AXIS, Signal
from mirpy.core.options import FrameArg
from mirpy.features._common import as_distribution, feature_signal, quiet_frames, with_params

# The manual's PDF clips the roll-off default (".8..."); 85 % is the
# MiningSuite default and the value mirfeatures uses.
ROLLOFF_THRESHOLD = 0.85
ROLLOFF_MIN_RMS = 0.005
BRIGHTNESS_CUTOFF = 1500.0
BRIGHTNESS_MIN_RMS = 0.005
PERCENT = 100


def rolloff(
    x: Signal,
    *,
    frame: FrameArg = False,
    threshold: float = ROLLOFF_THRESHOLD,
    min_rms: float = ROLLOFF_MIN_RMS,
) -> Signal:
    """Compute the frequency below which a given share of the spectral energy lies.

    Equivalent to MIRtoolbox ``mirrolloff``. Energy is accumulated on the
    magnitude spectrum, and the result is the first bin where the running
    sum reaches `threshold` times the total.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first) or a
        :class:`~mirpy.Spectrum`.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    threshold : float
        Share of the energy, as a ratio (0.85) or a percentage (85).
        MATLAB ``'Threshold'``.
    min_rms : float
        Frames whose spectrum RMS is below this ratio of the highest one
        give NaN. MATLAB ``'MinRMS'``.

    Returns
    -------
    Signal
        Frequency in Hz, one per channel and frame.

    See Also
    --------
    librosa.feature.spectral_rolloff : same idea, 85 % default, on the STFT magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> tone = Audio.from_array(np.sin(2 * np.pi * 1024 * np.arange(8192) / 8192), sr=8192)
    >>> float(rolloff(tone, threshold=0.5).data.item())
    1024.0
    """
    if threshold > 1:
        threshold /= PERCENT
    s = as_distribution(x, frame)
    cumulative = np.cumsum(s.data, axis=ELEMENT_AXIS)
    reached = cumulative >= threshold * cumulative[:, -1:, :]
    first = np.argmax(reached, axis=ELEMENT_AXIS)
    values = np.where(reached.any(axis=ELEMENT_AXIS), s.x_values[first], np.nan)
    values[quiet_frames(s, min_rms)] = np.nan
    result = feature_signal(s, values, "Roll-off", "Hz")
    return with_params(result, "rolloff", {"threshold": threshold, "min_rms": min_rms})


def brightness(
    x: Signal,
    *,
    frame: FrameArg = False,
    cutoff: float = BRIGHTNESS_CUTOFF,
    min_rms: float = BRIGHTNESS_MIN_RMS,
) -> Signal:
    """Compute the share of the spectral energy above a cut-off frequency.

    Equivalent to MIRtoolbox ``mirbrightness``.

    Parameters
    ----------
    x : Signal
        Audio waveform (its spectrum is computed first) or a
        :class:`~mirpy.Spectrum`.
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    cutoff : float
        Cut-off frequency in Hz. 1000 Hz (Laukka, Juslin & Bresin, 2005)
        and 3000 Hz (Juslin, 2000) are also used. MATLAB ``'CutOff'``.
    min_rms : float
        Frames whose spectrum RMS is below this ratio of the highest one
        give NaN. MATLAB ``'MinRMS'``.

    Returns
    -------
    Signal
        Value between 0 and 1, one per channel and frame.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> t = np.arange(8000) / 8000
    >>> mix = Audio.from_array(np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 3000 * t), sr=8000)
    >>> round(float(brightness(mix).data.item()), 1)
    0.5
    """
    s = as_distribution(x, frame)
    above = (s.x_values > cutoff)[np.newaxis, :, np.newaxis]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.sum(s.data * above, axis=ELEMENT_AXIS) / np.sum(s.data, axis=ELEMENT_AXIS)
    values[quiet_frames(s, min_rms)] = np.nan
    result = feature_signal(s, values, "Brightness")
    return with_params(result, "brightness", {"cutoff": cutoff, "min_rms": min_rms})
