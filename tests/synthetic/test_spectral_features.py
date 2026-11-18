import numpy as np
import pytest

from mirpy import Audio, Signal, Spectrum, brightness, rolloff

SR = 8192
FREQS = np.arange(8) * 1000.0


def one_frame(values):
    return Spectrum(np.asarray(values, float).reshape(1, -1, 1), sr=SR, x_values=FREQS)


def test_rolloff_is_first_bin_reaching_threshold():
    s = one_frame([1, 1, 1, 1, 1, 1, 1, 1])
    assert rolloff(s).data.item() == 6000  # cumsum 7/8 >= 0.85
    assert rolloff(s, threshold=0.5).data.item() == 3000  # 4/8 reaches 0.5
    assert rolloff(s, threshold=50).data.item() == 3000  # percentage


def test_rolloff_of_silence_is_first_bin():
    # MiningSuite: threshold 0, so find(cumsum >= 0, 1) is the first bin
    assert rolloff(one_frame(np.zeros(8)), min_rms=0).data.item() == 0


def test_rolloff_unit_is_hz():
    assert rolloff(one_frame(np.ones(8))).unit == "Hz"


def test_brightness_is_share_above_cutoff():
    s = one_frame([1, 1, 2, 0, 0, 0, 0, 4])  # bins at 0..7 kHz
    assert brightness(s).data.item() == pytest.approx(6 / 8)  # above 1500 Hz
    assert brightness(s, cutoff=3000).data.item() == pytest.approx(4 / 8)


def test_brightness_of_two_tones():
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 512 * t)
    high = np.sin(2 * np.pi * 3072 * t)
    assert brightness(Audio.from_array(low, sr=SR)).data.item() < 0.01
    assert brightness(Audio.from_array(high, sr=SR)).data.item() > 0.99


def test_min_rms_gates_quiet_frames():
    data = np.ones((1, 8, 2))
    data[:, :, 1] = 1e-4
    s = Spectrum(data, sr=SR, x_values=FREQS)
    for operator in (rolloff, brightness):
        values = operator(s).data[0, 0]
        assert not np.isnan(values[0]) and np.isnan(values[1]), operator.__name__


def test_framed_audio():
    noise = Audio.from_array(np.random.default_rng(0).standard_normal(SR), sr=SR)
    for operator in (rolloff, brightness):
        result = operator(noise, frame=True)
        assert result.is_framed and result.n_frames == 39
        assert isinstance(result, Signal)
