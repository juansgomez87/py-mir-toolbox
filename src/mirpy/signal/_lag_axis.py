"""Lag and quefrency axes expressed as frequencies (MIRtoolbox ``'Freq'`` option)."""

import numpy as np


def as_frequencies(data: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop lag 0 and order the element axis by increasing frequency ``1 / lag``.

    MIRtoolbox's ``'Freq'`` output (checked against the 1.8.2 fixtures)
    lists frequencies from low to high, i.e. lags from long to short.
    """
    keep = np.flatnonzero(lags > 0)[::-1]
    return data[:, keep], 1 / lags[keep]
