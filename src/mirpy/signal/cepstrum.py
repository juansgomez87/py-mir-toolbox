"""Real cepstrum.

Translated from MiningSuite ``+sig/cepstrum.m`` and
``+sig/@Cepstrum/Cepstrum.m`` (BSD-3, Olivier Lartillot).

MiningSuite quirks kept on purpose, to be checked against MIRtoolbox:

- The cepstrum is ``abs(fft(log(...)))``, not ``ifft``: values are
  ``N + 1`` times the textbook real cepstrum.
- The mirrored spectrum has ``N + 1`` points for an ``N``-point FFT, yet
  quefrency ``k`` is labelled ``k / sr``.

The second range cut in ``after`` is not ported: for a cepstrum computed
in the same call it uses the same bounds as ``main`` and changes nothing.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from mirpy.core import framing
from mirpy.core.data import Signal
from mirpy.core.options import FrameArg
from mirpy.signal._lag_axis import as_frequencies
from mirpy.signal.spectrum import Spectrum, spectrum

DEFAULT_MIN_QUEFRENCY = 0.0002  # seconds
DEFAULT_MAX_QUEFRENCY = 0.05  # seconds
LOG_FLOOR = 1e-12  # added before the log, as MiningSuite


@dataclass(frozen=True, eq=False)
class Cepstrum(Signal):
    """A real cepstrum, one per frame.

    ``data.shape == (n_channels, n_quefrencies, n_frames)``; ``x_values``
    holds the quefrencies in seconds, or ``1 / quefrency`` in Hz when
    computed with ``freq=True``.
    """

    name: str = "Cepstrum"
    x_name: str = "quefrency"
    x_unit: str = "s"


def cepstrum(
    x: Signal,
    *,
    frame: FrameArg = False,
    min_quefrency: float = DEFAULT_MIN_QUEFRENCY,
    max_quefrency: float = DEFAULT_MAX_QUEFRENCY,
    min_freq: float | None = None,
    max_freq: float | None = None,
    freq: bool = False,
) -> Cepstrum:
    """Compute the real cepstrum of each frame.

    Equivalent to MIRtoolbox ``mircepstrum``: the magnitude spectrum is
    mirrored, its logarithm taken, and the magnitude of its FFT kept.

    Parameters
    ----------
    x : Signal
        Audio (or any time signal), framed or not, or a :class:`Spectrum`.
        Audio goes through :func:`mirpy.spectrum` with default options.
        A :class:`Cepstrum` is returned unchanged.
    frame : bool, FrameSpec or tuple
        Frame the signal first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    min_quefrency, max_quefrency : float
        Quefrency range in seconds. MATLAB ``'Min'``, ``'Max'``.
    min_freq, max_freq : float, optional
        The same range given as frequencies: quefrencies from
        ``1 / max_freq`` to ``1 / min_freq``. Override the quefrency
        bounds. MATLAB ``'Min', f, 'Hz'``.
    freq : bool
        Express the quefrency axis as frequency, ``1 / quefrency``.
        MATLAB ``'Freq'``.

    Returns
    -------
    Cepstrum

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> t = np.arange(2048) / 22050
    >>> y = sum(np.sin(2 * np.pi * 200 * h * t) / h for h in range(1, 20))
    >>> c = cepstrum(Audio.from_array(y, sr=22050), min_quefrency=0.002)
    >>> round(1 / float(c.x_values[np.argmax(c.data[0, :, 0])]))  # f0 in Hz
    200
    """
    if isinstance(x, Cepstrum):
        return x
    if not isinstance(x, Spectrum):
        x = spectrum(framing.frame(x, frame))
    if max_freq is not None:
        min_quefrency = 1 / max_freq
    if min_freq is not None:
        max_quefrency = 1 / min_freq

    first = math.ceil(min_quefrency * x.sr)  # 0-based; MATLAB ceil(min*sr) + 1
    stop = math.ceil(max_quefrency * x.sr)  # exclusive; MATLAB end index, inclusive
    mirrored = np.concatenate([x.data[:, :-1], x.data[:, ::-1]], axis=1)
    full = np.abs(np.fft.fft(np.log(mirrored + LOG_FLOOR), axis=1))
    stop = min(stop, full.shape[1] // 2)
    quefrencies = np.arange(first, stop) / x.sr
    data, x_values = full[:, first:stop], quefrencies
    if freq:
        data, x_values = as_frequencies(data, quefrencies)
    result = Cepstrum(
        data=data,
        sr=x.sr,
        x_values=x_values,
        x_name="frequency" if freq else "quefrency",
        x_unit="Hz" if freq else "s",
        start=x.start,
        frame_rate=x.frame_rate,
        frame_length=x.frame_length,
        frame_starts=x.frame_starts,
        channel_labels=x.channel_labels,
        source=x.source,
        params=x.params,
    )
    options = {"min_quefrency": min_quefrency, "max_quefrency": max_quefrency, "freq": freq}
    return replace(result, params={**result.params, "cepstrum": options})
