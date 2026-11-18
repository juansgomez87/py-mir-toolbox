"""Filterbank decomposition into frequency bands.

Translated from MiningSuite ``+sig/filterbank.m``, ``+sig/+filterbank/``
(``options.m``, ``main.m``, ``specif.m``) and ``+aud/filterbank.m``
(BSD-3, Olivier Lartillot). The gammatone filters come from the vendored
``python_auditory_toolbox`` (Apache-2.0, Malcolm Slaney), which matches
the ``ERBFilters`` code inlined in ``+aud/filterbank.m``.

Bands are put on the channel axis, so the input must be mono.

MiningSuite quirk kept on purpose, to be checked against MIRtoolbox: in
``"2channels"`` mode the high band is the 70-1000 Hz band-pass output,
half-wave rectified, then high-passed at 1 kHz. Tolonen & Karjalainen
(2000) high-pass first and low-pass after rectification.
"""

from dataclasses import replace

import numpy as np
from scipy.signal import butter, ellip, lfilter, sosfilt

from mirpy._vendor import auditory_toolbox
from mirpy.core import framing
from mirpy.core.data import Signal
from mirpy.core.options import FrameArg, FrameSpec
from mirpy.signal import auditory_bands

KINDS = ("gammatone", "2channels", "manual")
PRESETS = ("scheirer", "klapuri", "mel", "bark")
DEFAULT_FRAME = FrameSpec(size=0.5, hop=0.1)  # 'FrameManual', .5, .1 in options.m
DEFAULT_CUTOFFS = (-np.inf, 1000.0, np.inf)
ELLIP_PASSBAND_RIPPLE = 3  # dB
ELLIP_STOPBAND_ATTENUATION = 40  # dB
TWO_CHANNEL_LOW_BAND = (70.0, 1000.0)  # Hz
TWO_CHANNEL_HIGH_EDGE = 1000.0  # Hz
TWO_CHANNEL_HIGH_TOP = 10000.0  # Hz, used when sr >= 20 kHz
TWO_CHANNEL_WIDE_RATE = 20000.0  # Hz
TWO_CHANNEL_LOW_ORDER, TWO_CHANNEL_HIGH_ORDER = 4, 2


def filterbank(
    x: Signal,
    kind: str | None = None,
    *,
    n_channels: int = 10,
    channels=None,
    lowest: float = 50.0,
    cutoffs=None,
    hop: int = 1,
    order: int = 4,
    preset: str | None = None,
    frame: FrameArg = False,
) -> Signal:
    """Split a mono signal into frequency bands, one per channel.

    Equivalent to MIRtoolbox ``mirfilterbank``.

    Parameters
    ----------
    x : Signal
        Unframed mono signal.
    kind : {"gammatone", "2channels", "manual"}, optional
        ``"gammatone"``: ERB-spaced gammatone filters (Patterson &
        Holdsworth; Slaney, 1993). ``"2channels"``: the two-channel model
        of Tolonen & Karjalainen (2000). ``"manual"``: elliptic filters
        between `cutoffs`. Default: ``"manual"`` when `cutoffs` or
        `preset` is given, otherwise ``"gammatone"``.
    n_channels : int
        Number of gammatone bands. MATLAB ``'NbChannels'``.
    channels : int or sequence of int, optional
        Keep only these bands (0-based, lowest first). MATLAB
        ``'Channel'`` (1-based).
    lowest : float
        Centre frequency of the lowest gammatone band, in Hz. MATLAB
        ``'Lowest'``.
    cutoffs : sequence of float, optional
        Band edges in Hz for ``"manual"``; ``-inf``/``inf`` make the first
        and last bands low-pass and high-pass. Default ``(-inf, 1000,
        inf)``. Edges above the Nyquist frequency are dropped. MATLAB
        ``'CutOff'``.
    hop : int
        Band ``i`` spans ``cutoffs[i]`` to ``cutoffs[i + hop]``; 2 gives
        half-overlapping bands. MATLAB ``'Hop'``.
    order : int
        Elliptic filter order (doubled for band-pass). MATLAB ``'Order'``.
    preset : {"scheirer", "klapuri", "mel", "bark"}, optional
        Predefined `cutoffs`: Scheirer (1998) octave bands, Klapuri (1999)
        bands, 40 half-overlapping Mel bands, or the critical bands.
    frame : bool, FrameSpec or tuple
        Frame the filtered bands; ``True`` uses 0.5 s frames with a hop of
        0.1 frame. MATLAB ``'Frame'``.

    Returns
    -------
    Signal
        Same type as `x`, with one channel per band (lowest first) and
        band names in ``channel_labels``.

    See Also
    --------
    librosa.filters.mel : triangular Mel filters applied to a spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy import Audio
    >>> bands = filterbank(Audio.from_array(np.random.randn(22050), sr=22050))
    >>> bands.n_channels, bands.channel_labels[0]
    (10, '50 Hz')
    """
    if x.n_channels != 1 or x.is_framed:
        raise ValueError("filterbank needs an unframed mono signal")
    kind = _resolve_kind(kind, cutoffs, preset)
    if kind == "gammatone":
        data, labels = _gammatone(x.data[0, :, 0], x.sr, n_channels, lowest)
    elif kind == "2channels":
        data, labels = _two_channels(x.data[0, :, 0], x.sr)
    else:
        if preset is not None:
            cutoffs, hop = _preset_cutoffs(preset, hop)
        data, labels = _elliptic(x.data[0, :, 0], x.sr, cutoffs or DEFAULT_CUTOFFS, hop, order)
    if channels is not None:
        keep = np.atleast_1d(channels)
        data, labels = data[keep], [labels[i] for i in keep]
    result = replace(x, data=data[:, :, np.newaxis], channel_labels=tuple(labels))
    result = framing.frame(result, DEFAULT_FRAME if frame is True else frame)
    options = {
        "kind": kind, "n_channels": n_channels, "channels": channels, "lowest": lowest,
        "cutoffs": cutoffs, "hop": hop, "order": order, "preset": preset,
    }  # fmt: skip
    return replace(result, params={**result.params, "filterbank": options})


def _resolve_kind(kind, cutoffs, preset) -> str:
    manual_options = cutoffs is not None or preset is not None
    if kind is None:
        kind = "manual" if manual_options else "gammatone"
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if manual_options and kind != "manual":
        raise ValueError("cutoffs and preset are only used by kind='manual'")
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    return kind


def _gammatone(y, sr, n_channels, lowest):
    """Gammatone bands, lowest first (MiningSuite reverses Slaney's order)."""
    coefficients = auditory_toolbox.MakeErbFilters(sr, n_channels, lowest)
    centers = auditory_toolbox.ErbSpace(lowest, sr / 2, n_channels)
    bands = auditory_toolbox.ErbFilterBank(y, coefficients)
    return bands[::-1], [f"{f:.0f} Hz" for f in centers[::-1]]


def _two_channels(y, sr):
    nyquist = sr / 2
    low_b, low_a = butter(
        TWO_CHANNEL_LOW_ORDER, np.array(TWO_CHANNEL_LOW_BAND) / nyquist, btype="bandpass"
    )
    if sr < TWO_CHANNEL_WIDE_RATE:
        high_b, high_a = butter(
            TWO_CHANNEL_HIGH_ORDER, TWO_CHANNEL_HIGH_EDGE / nyquist, btype="highpass"
        )
    else:
        edges = np.array([TWO_CHANNEL_HIGH_EDGE, TWO_CHANNEL_HIGH_TOP]) / nyquist
        high_b, high_a = butter(TWO_CHANNEL_HIGH_ORDER, edges, btype="bandpass")
    low = lfilter(low_b, low_a, y)
    high = lfilter(high_b, high_a, np.maximum(low, 0))
    return np.stack([low, high]), ["low", "high"]


def _preset_cutoffs(preset, hop):
    if preset == "scheirer":
        return [-np.inf, 200, 400, 800, 1600, 3200, np.inf], hop
    if preset == "klapuri":
        exponents = np.r_[0:3, (9 + np.arange(18)) / 3]
        return list(44 * 2.0**exponents), hop
    if preset == "mel":
        return list(auditory_bands.mel_edges(auditory_bands.DEFAULT_MEL_BANDS)), 2
    return list(auditory_bands.BARK_UPPER_FREQS), hop


def usable_cutoffs(cutoffs, sr) -> list[float]:
    """Drop edges above the Nyquist frequency, as ``specif.m``.

    A last edge above Nyquist becomes ``inf`` (high-pass); others are removed.
    """
    kept = [float(f) for f in cutoffs]
    j = 0
    while j < len(kept):
        if np.isfinite(kept[j]) and kept[j] > sr / 2:
            if j == len(kept) - 1:
                kept[j] = np.inf
            else:
                del kept[j]
                continue
        j += 1
    return kept


def _elliptic(y, sr, cutoffs, hop, order):
    edges = usable_cutoffs(cutoffs, sr)
    n_bands = len(edges) - hop
    if n_bands < 1:
        raise ValueError(f"cutoffs {cutoffs} with hop={hop} define no band")
    nyquist = sr / 2
    bands, labels = [], []
    for low, high in zip(edges[:n_bands], edges[hop:], strict=True):
        if np.isinf(low):
            sos = _ellip(order, high / nyquist, "lowpass")
        elif np.isinf(high):
            sos = _ellip(order, low / nyquist, "highpass")
        else:
            sos = _ellip(order, [low / nyquist, high / nyquist], "bandpass")
        bands.append(sosfilt(sos, y))
        labels.append(f"{low:g}-{high:g} Hz")
    return np.stack(bands), labels


def _ellip(order, cutoff, btype):
    return ellip(
        order,
        ELLIP_PASSBAND_RIPPLE,
        ELLIP_STOPBAND_ATTENUATION,
        cutoff,
        btype=btype,
        output="sos",
    )
