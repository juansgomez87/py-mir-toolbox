"""Sum across channels (filterbank bands).

Translated from MiningSuite ``+sig/sum.m`` and the ``sum`` method of
``+sig/@data/data.m`` (BSD-3, Olivier Lartillot).
"""

import math
from dataclasses import replace

import numpy as np

from mirpy.core.data import CHANNEL_AXIS, Signal


def sum(x: Signal, *, mean: bool = False, adjacent: int = 1) -> Signal:
    """Sum the channels of a signal, e.g. the bands of a filterbank.

    Equivalent to MIRtoolbox ``mirsum``.

    Parameters
    ----------
    x : Signal
        Signal with one or more channels.
    mean : bool
        Divide by the number of channels. As in MiningSuite, this is the
        total number of channels even when `adjacent` groups them.
        MATLAB ``'Mean'``.
    adjacent : int
        Sum groups of this many neighbouring channels instead of all of
        them; the last group may be smaller. MATLAB ``'Adjacent'``.

    Returns
    -------
    Signal
        Same type as `x`, with one channel, or ``ceil(n_channels /
        adjacent)`` channels.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Signal
    >>> x = Signal(np.arange(5.0).reshape(5, 1, 1), sr=1)
    >>> sum(x).data.ravel(), sum(x, adjacent=2).data.ravel()
    (array([10.]), array([1., 5., 4.]))
    """
    if adjacent < 2:
        data = x.data.sum(axis=CHANNEL_AXIS, keepdims=True)
    else:
        groups = np.arange(0, x.n_channels, adjacent)
        data = np.add.reduceat(x.data, groups, axis=CHANNEL_AXIS)
    if mean:
        data = data / x.n_channels
    labels = None if adjacent < 2 or x.channel_labels is None else _group_labels(x, adjacent)
    options = {"mean": mean, "adjacent": adjacent}
    return replace(x, data=data, channel_labels=labels, params={**x.params, "sum": options})


def _group_labels(x: Signal, adjacent: int) -> tuple[str, ...]:
    n_groups = math.ceil(x.n_channels / adjacent)
    labels = x.channel_labels
    return tuple(
        f"{labels[g * adjacent]} .. {labels[min((g + 1) * adjacent, x.n_channels) - 1]}"
        for g in range(n_groups)
    )
