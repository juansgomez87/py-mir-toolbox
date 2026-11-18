"""NumPy versions of MATLAB built-ins whose behaviour differs from NumPy's."""

import numpy as np


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
