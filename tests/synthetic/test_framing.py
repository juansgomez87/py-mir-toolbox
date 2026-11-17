import math

import numpy as np
import pytest

from mirpy import Audio, FrameSpec, frame
from mirpy.core.framing import frame_start_samples

SR = 44100


def loop_frame_starts(n_samples, length, hop, start, sr):
    """Line-by-line transcription of the loop in MiningSuite frame.m (1-based)."""
    start = math.floor(start * sr) + 1
    sf = math.ceil((start - 1) / hop) + 1
    fstart = math.floor((sf - 1) * hop) + 1
    offset = fstart - start
    nfr = math.floor((n_samples - length - offset) / hop) + 1
    return [math.floor((i - 1) * hop + 1) + offset - 1 for i in range(1, nfr + 1)]


@pytest.fixture
def ramp():
    return Audio.from_array(np.arange(SR, dtype=float), sr=SR)


@pytest.mark.parametrize(
    ("length", "hop", "start"),
    [(2205, 1102.5, 0.0), (2048, 512.0, 0.0), (1000, 333.3, 0.0), (2205, 1102.5, 0.37)],
)
def test_frame_starts_match_frame_m_loop(length, hop, start):
    expected = loop_frame_starts(SR, length, hop, start, SR)
    actual = frame_start_samples(SR, length, hop, start, SR)
    np.testing.assert_array_equal(actual, expected)


def test_default_frame_shape_and_rate(ramp):
    framed = frame(ramp)
    length, hop = 2205, 1102.5
    assert framed.data.shape == (1, length, math.floor((SR - length) / hop) + 1)
    assert framed.frame_rate == pytest.approx(SR / hop)
    assert framed.frame_length == pytest.approx(0.05)
    assert framed.is_framed


def test_frame_content_is_the_signal_slice(ramp):
    framed = frame(ramp, FrameSpec(size=100, size_unit="sp", hop=30, hop_unit="sp"))
    np.testing.assert_array_equal(framed.data[0, :, 2], np.arange(60, 160))
    np.testing.assert_allclose(framed.frame_starts[:3], np.array([0, 30, 60]) / SR)
    np.testing.assert_allclose(framed.frame_ends[0], 100 / SR)


def test_fractional_hop_uses_floor(ramp):
    framed = frame(ramp, FrameSpec(size=10, size_unit="sp", hop=2.5, hop_unit="sp"))
    np.testing.assert_array_equal(framed.data[0, 0, :5], [0, 2, 5, 7, 10])


def test_multichannel_keeps_channel_axis():
    stereo = Audio.from_array(np.vstack([np.zeros(1000), np.ones(1000)]), sr=1000, mono=False)
    framed = frame(stereo, (0.1, 0.5))
    assert framed.data.shape == (2, 100, 19)
    assert np.all(framed.data[1] == 1)


@pytest.mark.parametrize(
    ("spec", "hop_samples", "frame_rate"),
    [
        (FrameSpec(0.05, 0.5, "s", "/1"), 1102.5, SR / 1102.5),
        (FrameSpec(0.05, 25, "s", "%"), 551.25, SR / 551.25),
        (FrameSpec(0.05, 0.01, "s", "s"), 441.0, 100.0),
        (FrameSpec(2048, 512, "sp", "sp"), 512.0, SR / 512),
        (FrameSpec(0.05, 20, "s", "Hz"), 2205.0, 20.0),
    ],
)
def test_hop_units(spec, hop_samples, frame_rate):
    assert spec.hop_samples(SR) == pytest.approx(hop_samples)
    assert spec.frame_rate(SR) == pytest.approx(frame_rate)


def test_relative_hop_uses_unfloored_length():
    spec = FrameSpec(size=0.0501, hop=0.5)  # 2209.41 samples
    assert spec.length_samples(SR) == 2209
    assert spec.hop_samples(SR) == pytest.approx(0.0501 * SR / 2)


def test_frame_false_returns_input(ramp):
    assert frame(ramp, False) is ramp


def test_already_framed_warns_and_returns_input(ramp):
    framed = frame(ramp)
    with pytest.warns(UserWarning, match="already"):
        assert frame(framed) is framed


def test_frame_shorter_than_one_sample_warns(ramp):
    with pytest.warns(UserWarning, match="shorter"):
        assert frame(ramp, (1e-6, 0.5)) is ramp


def test_frame_longer_than_signal_warns(ramp):
    with pytest.warns(UserWarning, match="longer"):
        assert frame(ramp, (2.0, 0.5)) is ramp


@pytest.mark.parametrize(
    "kwargs", [{"size_unit": "ms"}, {"hop_unit": "frames"}, {"size": 0}, {"hop": -1}]
)
def test_invalid_frame_spec(kwargs):
    with pytest.raises(ValueError):
        FrameSpec(**kwargs)
