"""Data classes holding the output of every mirpy operator.

Translated from MiningSuite ``+sig/@Signal/Signal.m`` and ``+sig/@data/data.m``
(BSD-3, Olivier Lartillot). MiningSuite stores data with named dimensions
in any order; mirpy uses one fixed axis order instead:

``data.shape == (n_channels, n_elements, n_frames)``

- *channel*: audio channel or filterbank band;
- *element*: samples of an unframed signal, samples within a frame, or
  bins of a spectrum-like representation;
- *frame*: 1 for an unframed signal.

This is the order librosa uses (channels, features, frames). MIRtoolbox
uses ``(element, frame, channel)``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

N_AXES = 3
CHANNEL_AXIS, ELEMENT_AXIS, FRAME_AXIS = 0, 1, 2
PEAK_AXES = ("element", "frame")
SAMPLE_TOLERANCE = 1e-9  # so 0.05 s * 44100 Hz counts as 2205 samples


@dataclass(frozen=True, eq=False)
class Peaks:
    """Peaks found by :func:`mirpy.peaks`, one row per peak.

    Rows are grouped by curve (channel, then frame) and, within a curve,
    listed in the order requested from :func:`mirpy.peaks`.

    Parameters
    ----------
    axis : {"element", "frame"}
        Axis searched: along the elements of each frame (e.g. spectrum
        bins), or along the frames of a one-element feature (e.g. framed
        RMS). For ``"frame"``, `frame` equals `index`.
    channel, frame, index : array_like of int
        Channel, frame and 0-based position along `axis` of each peak.
    pos : array_like of float
        Peak position in units of the searched axis, refined by quadratic
        interpolation when requested.
    val : array_like of float
        Peak height on the normalised curve, as returned by MiningSuite.
        With the default ``normalize="global"`` it lies in [0, 1].
    val_original : array_like of float
        Peak height in the units of the signal (negated back for valleys).
    rank : array_like of int
        0 for the highest peak of its curve, 1 for the next, and so on.
    """

    axis: str
    channel: np.ndarray
    frame: np.ndarray
    index: np.ndarray
    pos: np.ndarray
    val: np.ndarray
    val_original: np.ndarray
    rank: np.ndarray

    def __post_init__(self):
        if self.axis not in PEAK_AXES:
            raise ValueError(f"axis must be one of {PEAK_AXES}, got {self.axis!r}")
        for name in ("channel", "frame", "index", "rank"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.intp))
        for name in ("pos", "val", "val_original"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        if len({len(getattr(self, name)) for name in self._columns()}) > 1:
            raise ValueError("All peak columns must have the same length")

    @staticmethod
    def _columns() -> tuple[str, ...]:
        return ("channel", "frame", "index", "pos", "val", "val_original", "rank")

    def __len__(self) -> int:
        return len(self.index)

    def at(self, channel: int = 0, frame: int | None = None) -> "Peaks":
        """Return the peaks of one channel, and of one frame if given."""
        keep = self.channel == channel
        if frame is not None:
            keep &= self.frame == frame
        return Peaks(self.axis, *(getattr(self, name)[keep] for name in self._columns()))


@dataclass(frozen=True, eq=False)
class Signal:
    """A signal or feature, with its sampling and framing information.

    Objects are immutable: operators return new objects, built with
    :func:`dataclasses.replace`.

    Parameters
    ----------
    data : array_like, shape (n_channels, n_elements, n_frames)
        Values. Converted to a float64 array.
    sr : float
        Sampling rate of the underlying signal, in Hz. Kept after framing.
    name : str
        Name of the quantity, e.g. ``"Audio"`` or ``"Centroid"``.
    unit : str
        Unit of the values.
    x_name, x_unit : str
        Name and unit of the element axis, e.g. ``"frequency"``, ``"Hz"``.
    x_values : array_like, optional
        Position of each element along the element axis (e.g. bin
        frequencies). ``None`` when elements are samples.
    start : float
        Time of the first sample, in seconds.
    frame_rate : float
        Frames per second; 0 for an unframed signal.
    frame_length : float
        Frame length in seconds; 0 for an unframed signal.
    frame_starts : array_like, optional
        Start time of each frame, in seconds; ``None`` when unframed.
    channel_labels : tuple of str, optional
        One label per channel (e.g. filterbank band names).
    source : str, optional
        Path of the audio file the signal comes from.
    params : mapping
        Options used to compute the signal, kept for export metadata.
    peaks : Peaks, optional
        Peaks found by :func:`mirpy.peaks`.
    """

    data: np.ndarray
    sr: float
    name: str = "Signal"
    unit: str = ""
    x_name: str = ""
    x_unit: str = ""
    x_values: np.ndarray | None = None
    start: float = 0.0
    frame_rate: float = 0.0
    frame_length: float = 0.0
    frame_starts: np.ndarray | None = None
    channel_labels: tuple[str, ...] | None = None
    source: str | None = None
    params: Mapping = field(default_factory=dict)
    peaks: Peaks | None = None

    def __post_init__(self):
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != N_AXES:
            raise ValueError(
                f"data must have shape (n_channels, n_elements, n_frames), got {data.shape}"
            )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        for name in ("x_values", "frame_starts"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, np.asarray(values, dtype=np.float64))
        self._check_lengths()

    def _check_lengths(self):
        if self.x_values is not None and len(self.x_values) != self.n_elements:
            raise ValueError("x_values must have one value per element")
        if self.frame_starts is not None and len(self.frame_starts) != self.n_frames:
            raise ValueError("frame_starts must have one value per frame")
        if self.channel_labels is not None and len(self.channel_labels) != self.n_channels:
            raise ValueError("channel_labels must have one label per channel")

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.data.shape[CHANNEL_AXIS]

    @property
    def n_elements(self) -> int:
        """Number of elements (samples or bins) per frame."""
        return self.data.shape[ELEMENT_AXIS]

    @property
    def n_frames(self) -> int:
        """Number of frames (1 when unframed)."""
        return self.data.shape[FRAME_AXIS]

    @property
    def is_framed(self) -> bool:
        """Whether the signal has been decomposed into frames."""
        return self.frame_rate > 0

    @property
    def times(self) -> np.ndarray:
        """Time of each sample of an unframed signal, in seconds."""
        if self.is_framed:
            raise ValueError("times is only defined for unframed signals; use frame_centers")
        return self.start + np.arange(self.n_elements) / self.sr

    @property
    def frame_ends(self) -> np.ndarray | None:
        """Time of the last sample of each frame, in seconds.

        MIRtoolbox's convention (``'FramePos'``, checked against the 1.8.2
        fixtures): a 2205-sample frame starting at 0 s ends at 2204 / sr,
        not at 2205 / sr.
        """
        if self.frame_starts is None:
            return None
        n_samples = math.floor(self.frame_length * self.sr + SAMPLE_TOLERANCE)
        return self.frame_starts + (n_samples - 1) / self.sr

    @property
    def frame_centers(self) -> np.ndarray | None:
        """Centre time of each frame, in seconds."""
        if self.frame_starts is None:
            return None
        return self.frame_starts + self.frame_length / 2

    def plot(self, ax=None):
        """Plot the signal; see :func:`mirpy.core.display.plot`."""
        from mirpy.core.display import plot

        return plot(self, ax)


@dataclass(frozen=True, eq=False)
class Audio(Signal):
    """An audio waveform.

    Build one from an array with :meth:`from_array` or from a file with
    :meth:`from_file`.
    """

    name: str = "Audio"
    x_name: str = "time"
    x_unit: str = "s"

    @classmethod
    def from_array(cls, y, sr: float, *, mono: bool = True, **kwargs) -> "Audio":
        """Create an :class:`Audio` from a waveform array.

        Parameters
        ----------
        y : array_like, shape (n_samples,) or (n_channels, n_samples)
            Waveform, channels first (librosa convention).
        sr : float
            Sampling rate in Hz.
        mono : bool
            Average the channels into one, as MIRtoolbox does by default.
        **kwargs
            Other :class:`Signal` fields, e.g. ``source``.

        Returns
        -------
        Audio

        Examples
        --------
        >>> import numpy as np
        >>> Audio.from_array(np.zeros((2, 100)), sr=8000).data.shape
        (1, 100, 1)
        """
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        if y.ndim != 2:
            raise ValueError(
                f"y must have shape (n_samples,) or (n_channels, n_samples), got {y.shape}"
            )
        if mono:
            y = _mix_channels(y)
        return cls(data=y[:, :, np.newaxis], sr=sr, **kwargs)

    @classmethod
    def from_file(cls, path, *, mono: bool = True) -> "Audio":
        """Read an audio file with soundfile.

        Parameters
        ----------
        path : str or path-like
            Any format libsndfile reads (WAV, FLAC, AIFF, OGG, MP3, ...).
        mono : bool
            Average the channels into one, as MIRtoolbox does by default.

        Returns
        -------
        Audio
        """
        import soundfile

        y, sr = soundfile.read(path, dtype="float64", always_2d=True)
        return cls.from_array(y.T, sr, mono=mono, source=str(path))

    def play(self):
        """Return an audio player for notebooks; see :func:`mirpy.core.display.play`."""
        from mirpy.core.display import play

        return play(self)


def _mix_channels(y: np.ndarray) -> np.ndarray:
    # MiningSuite evaleach.m mixes with data.mean('channel'). To be confirmed
    # against the MIRtoolbox 1.8.2 fixtures.
    return y.mean(axis=0, keepdims=True)


def get_data(x: Signal) -> np.ndarray:
    """Return the values of `x` with singleton axes removed.

    Equivalent to MIRtoolbox ``mirgetdata``.

    Examples
    --------
    >>> import numpy as np
    >>> get_data(Audio.from_array(np.zeros(100), sr=8000)).shape
    (100,)
    """
    return np.squeeze(x.data)
