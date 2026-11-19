"""Histogram of the values of any signal or feature.

Translated from MiningSuite ``+sig/histogram.m`` (BSD-3, Olivier
Lartillot), with two changes from the MIRtoolbox manual: bins are equally
spaced between the minimum and maximum (MiningSuite uses ``histcounts``,
whose automatic edges are rounded to "nice" values), and the ``'Ampli'``
option is added. MiningSuite's ``'Integer'`` option is not ported.

Each bin is closed, ``lo <= v <= hi``, so a value on an inner edge is
counted in both bins, as in MIRtoolbox 1.8.2 (pianoA4 fixture, 2026-11-19:
8 samples on an edge, counts summing to n + 8).
"""

from dataclasses import dataclass

import numpy as np

from mirpy.core.data import ELEMENT_AXIS, FRAME_AXIS, Signal

DEFAULT_BINS = 10


@dataclass(frozen=True, eq=False)
class Histogram(Signal):
    """Counts (or summed values) per bin.

    ``data.shape == (n_channels, n_bins, n_frames)``; ``x_values`` holds
    the bin centres.

    Parameters
    ----------
    edges : numpy.ndarray
        The ``n_bins + 1`` bin edges, MIRtoolbox's ``'Bins'``.
    """

    name: str = "Histogram"
    x_name: str = "value"
    edges: np.ndarray | None = None


def histo(x: Signal, *, number: int = DEFAULT_BINS, ampli: bool = False) -> Histogram:
    """Count the values of a signal in equally spaced bins.

    Equivalent to MIRtoolbox ``mirhisto``. A curve with several elements
    (an audio waveform, a spectrum, ...) gives one histogram per frame;
    a framed one-element feature (e.g. framed RMS) gives one histogram of
    its values over the frames. NaN values are ignored.

    Parameters
    ----------
    x : Signal
        Any signal or feature.
    number : int
        Number of bins. MATLAB ``'Number'``.
    ampli : bool
        Sum the values falling in each bin instead of counting them.
        MATLAB ``'Ampli'``.

    Returns
    -------
    Histogram
        The bins are shared by all channels and frames: they span the
        smallest to the largest value of `x`.

    See Also
    --------
    numpy.histogram : same edges, but half-open inner bins.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> h = histo(Audio.from_array(np.array([0.0, 0.1, 0.1, 1.0]), sr=8000), number=2)
    >>> h.data.ravel(), h.x_values
    (array([3., 1.]), array([0.25, 0.75]))
    """
    along_frames = x.n_elements == 1 and x.is_framed
    data = np.swapaxes(x.data, ELEMENT_AXIS, FRAME_AXIS) if along_frames else x.data
    finite = data[np.isfinite(data)]
    edges = np.histogram_bin_edges(finite, bins=number)
    counts = np.zeros((data.shape[0], number, data.shape[FRAME_AXIS]))
    for channel in range(data.shape[0]):
        for frame in range(data.shape[FRAME_AXIS]):
            values = data[channel, :, frame]
            values = values[np.isfinite(values)]
            counts[channel, :, frame] = _closed_bin_counts(values, edges, ampli)
    keeps_frames = x.is_framed and not along_frames
    return Histogram(
        data=counts,
        sr=x.sr,
        unit=x.unit if ampli else "",
        x_values=(edges[:-1] + edges[1:]) / 2,
        x_unit=x.unit,
        start=x.start,
        frame_rate=x.frame_rate if keeps_frames else 0.0,
        frame_length=x.frame_length if keeps_frames else 0.0,
        frame_starts=x.frame_starts if keeps_frames else None,
        channel_labels=x.channel_labels,
        source=x.source,
        params={**x.params, "histo": {"number": number, "ampli": ampli}},
        edges=edges,
    )


def _closed_bin_counts(values: np.ndarray, edges: np.ndarray, ampli: bool) -> np.ndarray:
    """Count (or sum) the values in each closed bin ``[edges[i], edges[i + 1]]``."""
    in_bin = (values[None, :] >= edges[:-1, None]) & (values[None, :] <= edges[1:, None])
    weights = values if ampli else np.ones_like(values)
    return in_bin @ weights
