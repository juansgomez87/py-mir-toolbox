"""Phase 1 parity: framing, FFT spectrum, Mel bands and log scale.

pymirtoolbox does not wrap mirspectrum. mirmfcc's second output is, per
the MIRtoolbox manual, "the output of mirspectrum, in mel-band and
log-scale" (40 bands, 50 ms frames, half overlap), which MiningSuite
computes as ``aud.spectrum(x, 'Mel', 'log', 'Bands', 40)``.
"""

import pytest

import mirpy as mir
from parity.harness import Stage, fixture_stems, from_matlab_axes

NO_FIXTURE = pytest.param("", marks=pytest.mark.skip(reason="no MIRtoolbox fixture available"))


def stems(feature: str) -> list:
    return fixture_stems(feature) or [NO_FIXTURE]


@pytest.mark.parametrize("stem", stems("mfcc_frame"))
def test_mel_log_spectrum(stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref = mirtoolbox_fixture(stem, "mfcc_frame")
    x = mir.audio(fixture_audio(ref))
    mel = mir.spectrum(x, frame=True, band="mel", log=True)
    expected = from_matlab_axes(ref.arrays["melspec"])
    assert_parity(mel.data, expected, Stage.LINEAR, f"{stem}: mel log spectrum")
