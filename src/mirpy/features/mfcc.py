"""Mel-frequency cepstral coefficients.

Translated from MiningSuite ``+aud/mfcc.m`` and ``+aud/@Mfcc/Mfcc.m``
(BSD-3, Olivier Lartillot). MiningSuite's DCT matrix, with rank 0 scaled
by sqrt(2)/2, is the orthonormal DCT-II (``scipy.fft.dct(norm="ortho")``);
it is built explicitly so that any rank can be asked for.
"""

from dataclasses import dataclass, replace

import numpy as np

from mirpy.core.data import Signal
from mirpy.core.options import FrameArg
from mirpy.features._common import with_params
from mirpy.signal.spectrum import Spectrum, spectrum

DEFAULT_BANDS = 40
DEFAULT_RANKS = range(1, 14)
DEFAULT_RADIUS = 2


@dataclass(frozen=True, eq=False)
class Mfcc(Signal):
    """Mel-frequency cepstral coefficients, one set per frame.

    ``data.shape == (n_channels, n_ranks, n_frames)``; ``x_values`` holds
    the ranks (0 is the energy term).

    Parameters
    ----------
    delta : int
        Order of temporal differentiation: 0 for MFCC, 1 for delta-MFCC,
        2 for delta-delta-MFCC.
    """

    name: str = "MFCC"
    x_name: str = "rank"
    delta: int = 0


def mfcc(
    x: Signal,
    *,
    frame: FrameArg = False,
    bands: int = DEFAULT_BANDS,
    rank=DEFAULT_RANKS,
    delta: int = 0,
    radius: int = DEFAULT_RADIUS,
    return_spectrum: bool = False,
) -> Mfcc | tuple[Mfcc, Spectrum]:
    """Compute the Mel-frequency cepstral coefficients of the signal, or of each frame.

    Equivalent to MIRtoolbox ``mirmfcc``: a Mel-band spectrum in log scale
    (:func:`mirpy.spectrum` with ``band="mel", log=True``), then a
    discrete cosine transform over the bands.

    Parameters
    ----------
    x : Signal
        Audio waveform, or a :class:`~mirpy.Spectrum` (turned into Mel
        bands and log scale if it is not already).
    frame : bool, FrameSpec or tuple
        Frame a waveform first; ``True`` uses 50 ms frames with half
        overlap. MATLAB ``'Frame'``.
    bands : int
        Number of Mel bands. MATLAB ``'Bands'``.
    rank : iterable of int
        Coefficients to keep. Rank 0 is the average energy; the default
        is 1 to 13, as MIRtoolbox. Ranks are the same numbers as in
        MATLAB (they are not indices). MATLAB ``'Rank'``.
    delta : int
        Order of temporal differentiation (frames needed). Each order
        drops ``radius`` frames at both ends. MATLAB ``'Delta'``.
    radius : int
        Number of frames on each side used for the least-squares slope.
        MATLAB ``'Radius'``.
    return_spectrum : bool
        Also return the Mel log spectrum, MIRtoolbox's second output.

    Returns
    -------
    Mfcc or (Mfcc, Spectrum)

    See Also
    --------
    librosa.feature.mfcc : HTK/Slaney Mel filters, dB scaling, ranks from 0.
    librosa.feature.delta : Savitzky-Golay derivative, keeps every frame.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> noise = Audio.from_array(np.random.default_rng(0).standard_normal(8000), sr=8000)
    >>> mfcc(noise, frame=True).data.shape
    (1, 13, 39)
    """
    ranks = np.atleast_1d(np.asarray(rank, dtype=int))
    mel = spectrum(x, frame=frame, band="mel", n_bands=bands, log=True)
    coefficients = np.einsum("rb,cbt->crt", dct_matrix(mel.n_elements, ranks), mel.data)
    result = Mfcc(
        data=coefficients,
        sr=mel.sr,
        x_values=ranks.astype(float),
        start=mel.start,
        frame_rate=mel.frame_rate,
        frame_length=mel.frame_length,
        frame_starts=mel.frame_starts,
        channel_labels=mel.channel_labels,
        source=mel.source,
        params=mel.params,
    )
    for _ in range(delta):
        result = _delta(result, radius)
    options = {"bands": bands, "rank": ranks.tolist(), "delta": delta, "radius": radius}
    result = with_params(result, "mfcc", options)
    return (result, mel) if return_spectrum else result


def dct_matrix(n_bands: int, ranks: np.ndarray) -> np.ndarray:
    """Return the ``(n_ranks, n_bands)`` DCT matrix of MiningSuite ``mfcc.m``."""
    bands = np.arange(n_bands)
    matrix = np.cos(np.outer(ranks, 2 * bands + 1) * np.pi / 2 / n_bands) / np.sqrt(n_bands / 2)
    matrix[ranks == 0] *= np.sqrt(2) / 2
    return matrix


def _delta(c: Mfcc, radius: int) -> Mfcc:
    """One order of least-squares differentiation over ``2 * radius + 1`` frames."""
    n_out = c.n_frames - 2 * radius
    if n_out < 1:
        raise ValueError(
            f"Delta with radius {radius} needs more than {2 * radius} frames, got {c.n_frames}"
        )
    slope = np.zeros(c.data.shape[:2] + (n_out,))
    for i in range(1, radius + 1):
        ahead = c.data[:, :, radius + i : radius + i + n_out]
        behind = c.data[:, :, radius - i : radius - i + n_out]
        slope += i * (ahead - behind)
    slope /= 2 * np.sum(np.arange(1, radius + 1) ** 2)
    starts = None if c.frame_starts is None else c.frame_starts[radius : radius + n_out]
    return replace(c, data=slope, frame_starts=starts, delta=c.delta + 1, name=f"Delta-{c.name}")
