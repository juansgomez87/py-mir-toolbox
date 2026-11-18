"""Auditory frequency weightings applied to a spectrum.

Translated from MiningSuite ``+aud/+spectrum/after.m`` (BSD-3, Olivier
Lartillot). The Mel filter constants are Slaney's (Auditory Toolbox MFCC).
"""

import numpy as np

MEL_LOWEST_FREQ = 133.3333
MEL_LINEAR_SPACING = 66.66666666
MEL_LOG_SPACING = 1.0711703
MEL_MAX_LINEAR_BANDS = 13
DEFAULT_MEL_BANDS = 40

# Upper edge (Hz) of each of the 24 critical bands.
BARK_UPPER_FREQS = 10.0 * np.array(
    [10, 20, 30, 40, 51, 63, 77, 92, 108, 127, 148, 172,
     200, 232, 270, 315, 370, 440, 530, 640, 770, 950, 1200, 1550]
)  # fmt: skip


def mel_weights(freqs: np.ndarray, n_bands: int = DEFAULT_MEL_BANDS) -> np.ndarray:
    """Return triangular Mel filters, shape ``(n_bands, len(freqs))``.

    The first 13 bands are spaced linearly from 133.33 Hz by 66.67 Hz,
    the others logarithmically (ratio 1.0711703). Each triangle has unit
    area (height ``2 / (upper - lower)``).
    """
    n_linear = min(MEL_MAX_LINEAR_BANDS, n_bands)
    edges = np.empty(n_bands + 2)
    edges[:n_linear] = MEL_LOWEST_FREQ + np.arange(n_linear) * MEL_LINEAR_SPACING
    n_log = n_bands + 2 - n_linear
    edges[n_linear:] = edges[n_linear - 1] * MEL_LOG_SPACING ** np.arange(1, n_log + 1)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = ((freqs > lower) & (freqs <= center)) * (freqs - lower) / (center - lower)
    falling = ((freqs > center) & (freqs < upper)) * (upper - freqs) / (upper - center)
    return 2 / (upper - lower) * (rising + falling)


def has_undefined_bands(weights: np.ndarray) -> bool:
    """Whether an empty band precedes a non-empty one (spectrum too coarse)."""
    empty = weights.sum(axis=1) == 0
    return bool(np.any(empty[:-1] & ~empty[1:]))


def bark_weights(freqs: np.ndarray) -> np.ndarray:
    """Return rectangular critical-band filters, shape ``(n_bands, len(freqs))``.

    Bands stop at the first critical band whose upper edge exceeds the
    highest frequency, as in MiningSuite.
    """
    above = np.flatnonzero(BARK_UPPER_FREQS > freqs[-1])
    n_bands = above[0] + 1 if above.size else len(BARK_UPPER_FREQS)
    weights = np.zeros((n_bands, len(freqs)))
    first = 0
    for band, upper in enumerate(BARK_UPPER_FREQS[:n_bands]):
        members = first + np.flatnonzero(freqs[first:] <= upper)
        weights[band, members] = 1
        if members.size:
            first = members[-1] + 1
    return weights


def terhardt_weights(freqs: np.ndarray) -> np.ndarray:
    """Return Terhardt's (1979) outer-ear power weighting for each frequency.

    As in MiningSuite, the first value is set to 0 whatever its frequency.
    """
    khz = freqs[1:] / 1000
    with np.errstate(divide="ignore"):
        db = -3.64 * khz**-0.8 + 6.5 * np.exp(-0.6 * (khz - 3.3) ** 2) - 0.001 * khz**4
    weights = np.zeros(len(freqs))
    weights[1:] = (10 ** (db / 20)) ** 2
    return weights


def mask_spread(n_bands: int) -> np.ndarray:
    """Return the spreading matrix of the critical-band masking model.

    Schroeder et al. (1979) spreading function, applied across bands.
    """
    distance = np.arange(n_bands)[:, None] - np.arange(n_bands)[None, :] + 0.474
    return 10 ** ((15.81 + 7.5 * distance - 17.5 * np.sqrt(1 + distance**2)) / 10)
