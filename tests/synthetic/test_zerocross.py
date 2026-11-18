import numpy as np
import pytest

from mirpy import Audio, zerocross

SR = 8000
PHASE = 0.1  # keeps samples off exact zeros


def tone(freq, n=SR):
    return Audio.from_array(np.sin(2 * np.pi * freq * np.arange(n) / SR + PHASE), sr=SR)


@pytest.mark.parametrize("freq", [50, 100, 440])
def test_sine_crossing_rate_is_its_frequency(freq):
    assert zerocross(tone(freq)).data.item() == pytest.approx(freq, abs=1)  # whole crossings


def test_both_directions_doubles_the_rate():
    one = zerocross(tone(100)).data.item()
    assert zerocross(tone(100), direction="both").data.item() == pytest.approx(2 * one)


def test_per_sample_divides_by_the_sampling_rate():
    per_second = zerocross(tone(100)).data.item()
    assert zerocross(tone(100), per="sample").data.item() == pytest.approx(per_second / SR)


def test_counts_are_divided_by_the_number_of_samples():
    x = Audio.from_array(np.array([1.0, -1.0, 1.0, -1.0]), sr=4)
    # 3 changes, both ways, per sample, over 4 samples
    assert zerocross(x, per="sample", direction="both").data.item() == 0.75


def test_framed():
    z = zerocross(tone(200), frame=True)
    assert z.is_framed and z.n_frames == 39
    np.testing.assert_allclose(z.data, 200, rtol=0.05)


def test_rejects_unknown_options():
    with pytest.raises(ValueError, match="per"):
        zerocross(tone(100), per="minute")
    with pytest.raises(ValueError, match="direction"):
        zerocross(tone(100), direction="up")
