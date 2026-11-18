"""mirpy: a Python port of MIRtoolbox for music feature extraction.

py-mir-toolbox reproduces the functions and defaults of MIRtoolbox
(Lartillot & Toiviainen, 2007) in Python. Its code is translated from
MiningSuite, the BSD-licensed successor of MIRtoolbox, and validated
numerically against MIRtoolbox 1.8.2.

Use :func:`mirpy.cite` to print the references to cite.
"""

from mirpy._cite import cite
from mirpy.core.audio_ops import audio
from mirpy.core.data import Audio, Peaks, Signal, get_data
from mirpy.core.export import file_table, frame_table, to_arff, to_csv, to_dataframe
from mirpy.core.framing import frame
from mirpy.core.options import FrameSpec
from mirpy.core.peaks import peaks
from mirpy.features.dynamics import lowenergy, rms
from mirpy.features.mfcc import Mfcc, mfcc
from mirpy.features.moments import centroid, entropy, flatness, kurtosis, skewness, spread
from mirpy.features.spectral import brightness, rolloff
from mirpy.features.zerocross import zerocross
from mirpy.signal.autocor import Autocor, autocor
from mirpy.signal.cepstrum import Cepstrum, cepstrum
from mirpy.signal.filterbank import filterbank
from mirpy.signal.spectrum import Spectrum, spectrum
from mirpy.signal.summation import sum

try:
    from mirpy._version import __version__
except ImportError:  # running from a source tree without a build
    __version__ = "0.0.0"

__all__ = [
    "Audio",
    "Autocor",
    "Cepstrum",
    "FrameSpec",
    "Mfcc",
    "Peaks",
    "Signal",
    "Spectrum",
    "__version__",
    "audio",
    "autocor",
    "brightness",
    "centroid",
    "cepstrum",
    "cite",
    "entropy",
    "file_table",
    "filterbank",
    "flatness",
    "frame",
    "frame_table",
    "get_data",
    "kurtosis",
    "lowenergy",
    "mfcc",
    "peaks",
    "rms",
    "rolloff",
    "skewness",
    "spectrum",
    "spread",
    "sum",
    "to_arff",
    "to_csv",
    "to_dataframe",
    "zerocross",
]
