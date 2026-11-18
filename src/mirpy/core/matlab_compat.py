"""NumPy versions of MATLAB built-ins whose behaviour differs from NumPy's."""

import numpy as np
from scipy.signal import get_window


def matlab_round(x):
    """Round half away from zero, like MATLAB ``round``.

    NumPy's :func:`numpy.round` rounds half to even, so ``np.round(2.5)``
    is 2 while MATLAB ``round(2.5)`` is 3.

    Examples
    --------
    >>> matlab_round([0.5, 1.5, 2.5, -2.5])
    array([ 1.,  2.,  3., -3.])
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def nextpow2(n: float) -> int:
    """Return the smallest ``p`` with ``2**p >= abs(n)``, like MATLAB ``nextpow2``.

    Examples
    --------
    >>> nextpow2(1024), nextpow2(1025), nextpow2(0.3)
    (10, 11, -1)
    """
    if n == 0:
        return 0
    mantissa, exponent = np.frexp(abs(n))
    return int(exponent - (mantissa == 0.5))


def matlab_window(name: str, n: int) -> np.ndarray:
    """Return a window of length `n` as MATLAB ``window(@name, n)`` does.

    MATLAB windows are *symmetric* by default, while scipy's are
    periodic (``fftbins=True``). MATLAB ``hanning`` also differs from
    ``hann``: it leaves out the zero end-points.

    Parameters
    ----------
    name : str
        ``"rectangular"``, ``"hanning"``, or any window name known to
        :func:`scipy.signal.get_window` (e.g. ``"hamming"``, ``"hann"``,
        ``"blackman"``).
    n : int
        Window length.

    Examples
    --------
    >>> matlab_window("hanning", 3)
    array([0.5, 1. , 0.5])
    >>> matlab_window("hamming", 3)
    array([0.08, 1.  , 0.08])
    """
    if name == "rectangular":
        return np.ones(n)
    if name == "hanning":
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))
    return get_window(name, n, fftbins=False)


XCORR_SCALES = ("none", "biased", "unbiased", "coeff")


def xcorr(x, maxlag: int, scale: str = "none", axis: int = -1) -> np.ndarray:
    """Autocorrelation at lags ``-maxlag..maxlag``, like MATLAB ``xcorr(x, maxlag, scale)``.

    Parameters
    ----------
    x : array_like
        Real signal(s); correlated along `axis`.
    maxlag : int
        Largest lag, in samples. Lags beyond the signal length are zero.
    scale : {"none", "biased", "unbiased", "coeff"}
        No scaling; divide by the length; divide by the number of
        overlapping samples at each lag; divide by the zero-lag value.
    axis : int
        Axis along which to correlate.

    Returns
    -------
    numpy.ndarray
        Same shape as `x` except along `axis`, which has ``2 * maxlag + 1``
        values; lag 0 is in the middle.

    Examples
    --------
    >>> xcorr([1.0, 2.0, 3.0], 2)
    array([ 3.,  8., 14.,  8.,  3.])
    """
    if scale not in XCORR_SCALES:
        raise ValueError(f"scale must be one of {XCORR_SCALES}, got {scale!r}")
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    n = x.shape[-1]
    n_fft = 2 ** nextpow2(2 * n - 1)
    spectrum = np.fft.rfft(x, n_fft)
    full = np.fft.irfft(spectrum * np.conj(spectrum), n_fft)  # lag k at index k mod n_fft
    lags = np.arange(-maxlag, maxlag + 1)
    c = np.where(np.abs(lags) < n, full[..., lags % n_fft], 0.0)
    match scale:
        case "biased":
            c = c / n
        case "unbiased":
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(np.abs(lags) < n, c / (n - np.abs(lags)), 0.0)
        case "coeff":
            with np.errstate(divide="ignore", invalid="ignore"):
                c = c / c[..., maxlag : maxlag + 1]
    return np.moveaxis(c, -1, axis)
