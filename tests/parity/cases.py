"""Shared steps of the MATLAB-case parity tests (``test_phase1``, ``test_phase2``)."""

import pytest

from parity.harness import Stage, fixture_stems, from_matlab_axes

NO_FIXTURE = pytest.param("", "", marks=pytest.mark.skip(reason="no MIRtoolbox fixture available"))


def cases_with_fixtures(features) -> list:
    """Return ``(feature, stem)`` parameters for every available fixture."""
    params = [(feature, stem) for feature in features for stem in fixture_stems(feature)]
    return params or [NO_FIXTURE]


def load_matlab_case(feature, stem, calls, mirtoolbox_fixture, fixture_audio):
    """Load a MATLAB fixture, check it was made by the listed call, and find its audio."""
    ref = mirtoolbox_fixture(stem, feature)
    assert ref.metadata.get("call") == calls[feature], "fixture made by another call"
    return ref, fixture_audio(ref)


# How MIRtoolbox lays out some outputs, compared with mirpy's (channel, element, frame)
STANDARD = "standard"
BANDS_AS_CHANNELS = "bands_as_channels"  # Mel/Bark spectra: bands on MIRtoolbox's 3rd axis
BINS_ALONG_FRAMES = "bins_along_frames"  # mirhisto: bins on the 2nd axis, Pos = (min, max)


def check_signal(
    result, ref, stage: Stage, label: str, assert_parity, layout: str = STANDARD
) -> None:
    """Compare values, frame positions and element positions with a MATLAB fixture."""
    expected = from_matlab_axes(ref.arrays["data"])
    if layout == BANDS_AS_CHANNELS:
        expected = expected.transpose(1, 0, 2)
    elif layout == BINS_ALONG_FRAMES:
        expected = expected.transpose(0, 2, 1)
    assert_parity(result.data, expected, stage, label)
    if layout == BINS_ALONG_FRAMES and "pos" in ref.arrays:
        centres = ref.arrays["pos"].reshape(-1, 2).mean(axis=1)
        assert_parity(result.x_values, centres, Stage.LINEAR, f"{label} bin centres")
        return
    if result.is_framed and "framepos" in ref.arrays:
        starts, ends = ref.arrays["framepos"]
        assert_parity(result.frame_starts, starts, Stage.LINEAR, f"{label} frame starts")
        assert_parity(result.frame_ends, ends, Stage.LINEAR, f"{label} frame ends")
    # MIRtoolbox keeps no band positions in 'Pos' for Mel/Bark spectra
    if layout == STANDARD and result.x_values is not None and "pos" in ref.arrays:
        pos = ref.arrays["pos"]
        positions = pos.reshape(len(pos), -1)[:, 0]  # first frame's column
        assert_parity(result.x_values, positions, Stage.LINEAR, f"{label} positions")
