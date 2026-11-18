"""mirpy: a Python port of MIRtoolbox for music feature extraction.

py-mir-toolbox reproduces the functions and defaults of MIRtoolbox
(Lartillot & Toiviainen, 2007) in Python. Its code is translated from
MiningSuite, the BSD-licensed successor of MIRtoolbox, and validated
numerically against MIRtoolbox 1.8.2.

Use :func:`mirpy.cite` to print the references to cite.
"""

from mirpy._cite import cite
from mirpy.core.data import Audio, Signal, get_data
from mirpy.core.export import file_table, frame_table, to_arff, to_csv, to_dataframe
from mirpy.core.framing import frame
from mirpy.core.options import FrameSpec

try:
    from mirpy._version import __version__
except ImportError:  # running from a source tree without a build
    __version__ = "0.0.0"

__all__ = [
    "Audio",
    "FrameSpec",
    "Signal",
    "__version__",
    "cite",
    "file_table",
    "frame",
    "frame_table",
    "get_data",
    "to_arff",
    "to_csv",
    "to_dataframe",
]
