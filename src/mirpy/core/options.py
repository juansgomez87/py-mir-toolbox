"""Frame decomposition options.

Translated from the ``'FrameSize'``/``'FrameHop'`` option specification in
MiningSuite ``+sig/@Signal/Signal.m`` and ``+sig/frame.m``, and from
``+sig/getfrate.m`` and ``+sig/getflength.m`` (BSD-3, Olivier Lartillot).
"""

import math
from dataclasses import dataclass

SIZE_UNITS = ("s", "sp")
HOP_UNITS = ("/1", "%", "s", "sp", "Hz")

DEFAULT_FRAME_SIZE = 0.05
DEFAULT_FRAME_HOP = 0.5

PERCENT = 0.01


@dataclass(frozen=True)
class FrameSpec:
    """Frame length and hop, with their units.

    Equivalent to MIRtoolbox ``'Frame', size, size_unit, hop, hop_unit``.

    Parameters
    ----------
    size : float
        Frame length. Default 0.05 (50 ms).
    hop : float
        Hop between frame starts. Default 0.5 (half the frame length).
    size_unit : {"s", "sp"}
        Unit of `size`: seconds or samples.
    hop_unit : {"/1", "%", "s", "sp", "Hz"}
        Unit of `hop`: ratio of the frame length, percentage of the frame
        length, seconds, samples, or frame rate in Hz.

    Examples
    --------
    >>> FrameSpec(size=2048, size_unit="sp", hop=0.25).hop_samples(44100)
    512.0
    """

    size: float = DEFAULT_FRAME_SIZE
    hop: float = DEFAULT_FRAME_HOP
    size_unit: str = "s"
    hop_unit: str = "/1"

    def __post_init__(self):
        if self.size_unit not in SIZE_UNITS:
            raise ValueError(f"size_unit must be one of {SIZE_UNITS}, got {self.size_unit!r}")
        if self.hop_unit not in HOP_UNITS:
            raise ValueError(f"hop_unit must be one of {HOP_UNITS}, got {self.hop_unit!r}")
        if self.size <= 0 or self.hop <= 0:
            raise ValueError("Frame size and hop must be positive")

    def _size_samples_unfloored(self, sr: float) -> float:
        return self.size * sr if self.size_unit == "s" else self.size

    def length_samples(self, sr: float) -> int:
        """Return the frame length in samples, rounded down as in MiningSuite."""
        return math.floor(self._size_samples_unfloored(sr))

    def hop_samples(self, sr: float) -> float:
        """Return the hop in samples, which may be fractional.

        Relative hops use the frame length *before* it is rounded down,
        as in MiningSuite ``frame.m``.
        """
        size = self._size_samples_unfloored(sr)
        match self.hop_unit:
            case "/1":
                return self.hop * size
            case "%":
                return self.hop * size * PERCENT
            case "s":
                return self.hop * sr
            case "sp":
                return float(self.hop)
            case "Hz":
                return sr / self.hop

    def frame_rate(self, sr: float) -> float:
        """Return the number of frames per second (``getfrate.m``)."""
        size = self._size_samples_unfloored(sr)
        match self.hop_unit:
            case "/1":
                return sr / self.hop / size
            case "%":
                return sr / self.hop / size / PERCENT
            case "s":
                return 1 / self.hop
            case "sp":
                return sr / self.hop
            case "Hz":
                return float(self.hop)

    def length_seconds(self, sr: float) -> float:
        """Return the frame length in seconds, not rounded to samples (``getflength.m``)."""
        return self.size if self.size_unit == "s" else self.size / sr


FrameArg = bool | FrameSpec | tuple | None


def as_frame_spec(frame: FrameArg, default: FrameSpec) -> FrameSpec | None:
    """Normalise an operator's ``frame`` argument.

    Parameters
    ----------
    frame : bool, FrameSpec, tuple or None
        ``False``/``None``: no framing. ``True``: use `default`.
        A tuple is passed to :class:`FrameSpec` positionally, e.g.
        ``(0.05, 0.5)`` or ``(2048, 0.25, "sp")``.
    default : FrameSpec
        The operator's default frame settings.

    Returns
    -------
    FrameSpec or None
    """
    if frame is None or frame is False:
        return None
    if frame is True:
        return default
    if isinstance(frame, FrameSpec):
        return frame
    if isinstance(frame, tuple):
        return FrameSpec(*frame)
    raise TypeError(f"frame must be a bool, FrameSpec or tuple, got {type(frame).__name__}")
