import numpy as np
import pytest
from scipy.signal import butter, ellip, freqz_sos, sosfilt

import mirpy as mir
from mirpy import Audio, Signal, filterbank
from mirpy._vendor import auditory_toolbox
from mirpy.signal.filterbank import usable_cutoffs

SR = 22050


def impulse(n=SR // 2, sr=SR):
    y = np.zeros(n)
    y[0] = 1
    return Audio.from_array(y, sr=sr)


def noise(n=SR // 2, sr=SR):
    return Audio.from_array(np.random.default_rng(0).standard_normal(n), sr=sr)


def peak_frequency(y, sr=SR):
    magnitude = np.abs(np.fft.rfft(y))
    return np.argmax(magnitude) * sr / len(y)


def test_gammatone_bands_low_to_high_and_peak_at_centres():
    bands = filterbank(impulse(), n_channels=8, lowest=100)
    centres = auditory_toolbox.ErbSpace(100, SR / 2, 8)[::-1]
    assert bands.n_channels == 8
    assert bands.channel_labels == tuple(f"{f:.0f} Hz" for f in centres)
    for band, centre in zip(bands.data[:, :, 0], centres, strict=True):
        assert peak_frequency(band) == pytest.approx(centre, rel=0.05, abs=5)


def test_gammatone_matches_vendored_filterbank():
    x = noise()
    bands = filterbank(x)
    expected = auditory_toolbox.ErbFilterBank(
        x.data[0, :, 0], auditory_toolbox.MakeErbFilters(SR, 10, 50)
    )[::-1]
    np.testing.assert_allclose(bands.data[:, :, 0], expected)


def test_channel_selection_is_zero_based_lowest_first():
    all_bands = filterbank(noise())
    some = filterbank(noise(), channels=[0, 3])
    np.testing.assert_array_equal(some.data, all_bands.data[[0, 3]])
    assert some.channel_labels == (all_bands.channel_labels[0], all_bands.channel_labels[3])


def test_two_channels_follows_miningsuite_order():
    sr = 16000  # below 20 kHz: the high band is high-passed at 1 kHz
    x = noise(sr=sr)
    bands = filterbank(x, "2channels")
    y = x.data[0, :, 0]
    low_sos = butter(4, np.array([70, 1000]) / (sr / 2), btype="bandpass", output="sos")
    high_sos = butter(2, 1000 / (sr / 2), btype="highpass", output="sos")
    low = sosfilt(low_sos, y)
    np.testing.assert_allclose(bands.data[0, :, 0], low)
    np.testing.assert_allclose(bands.data[1, :, 0], sosfilt(high_sos, np.maximum(low, 0)))
    assert bands.channel_labels == ("low", "high")


def test_two_channels_high_band_is_bandpass_at_high_rates():
    x = noise(sr=44100)
    high = filterbank(x, "2channels").data[1, :, 0]
    low_sos = butter(4, np.array([70, 1000]) / 22050, btype="bandpass", output="sos")
    high_sos = butter(2, np.array([1000, 10000]) / 22050, btype="bandpass", output="sos")
    expected = sosfilt(high_sos, np.maximum(sosfilt(low_sos, x.data[0, :, 0]), 0))
    np.testing.assert_allclose(high, expected)


def test_default_manual_bank_is_low_and_high_pass_at_1khz():
    x = noise()
    bands = filterbank(x, "manual")
    low = ellip(4, 3, 40, 1000 / (SR / 2), btype="lowpass", output="sos")
    np.testing.assert_allclose(bands.data[0, :, 0], sosfilt(low, x.data[0, :, 0]))
    assert bands.channel_labels == ("-inf-1000 Hz", "1000-inf Hz")


def test_elliptic_band_has_3db_ripple_at_edges():
    sos = ellip(4, 3, 40, [400 / (SR / 2), 800 / (SR / 2)], btype="bandpass", output="sos")
    _, response = freqz_sos(sos, worN=[400, 800], fs=SR)
    np.testing.assert_allclose(20 * np.log10(np.abs(response)), -3, atol=0.01)


def test_cutoffs_above_nyquist_are_dropped():
    assert usable_cutoffs([-np.inf, 1000, 20000, 30000], SR) == [-np.inf, 1000, np.inf]
    assert usable_cutoffs([100, 20000, 5000], SR) == [100, 5000]


@pytest.mark.parametrize(("preset", "n_bands"), [("scheirer", 6), ("klapuri", 20), ("mel", 40)])
def test_presets(preset, n_bands):
    assert filterbank(noise(sr=44100), preset=preset).n_channels == n_bands


def test_bark_preset_drops_bands_above_nyquist():
    # 8 kHz: edges up to 3700 Hz kept, 4400 Hz becomes the high-pass edge.
    bands = filterbank(noise(sr=8000), preset="bark")
    assert bands.n_channels == 17
    assert bands.channel_labels[-1] == "3700-inf Hz"


def test_hop_two_gives_half_overlapping_bands():
    bands = filterbank(noise(), cutoffs=[100, 200, 400, 800], hop=2)
    assert bands.channel_labels == ("100-400 Hz", "200-800 Hz")


def test_frame_true_uses_half_second_frames():
    bands = filterbank(noise(n=SR * 2), frame=True)
    assert bands.is_framed and bands.frame_length == pytest.approx(0.5)
    assert bands.frame_rate == pytest.approx(1 / 0.05)


def test_invalid_inputs_rejected():
    stereo = Audio.from_array(np.zeros((2, 100)), sr=SR, mono=False)
    with pytest.raises(ValueError, match="mono"):
        filterbank(stereo)
    with pytest.raises(ValueError, match="only used by kind='manual'"):
        filterbank(noise(), "gammatone", preset="mel")
    with pytest.raises(ValueError, match="preset"):
        filterbank(noise(), preset="erb")
    with pytest.raises(ValueError, match="no band"):
        filterbank(noise(), cutoffs=[100])


def test_sum_all_and_adjacent_channels():
    x = Signal(np.arange(5.0).reshape(5, 1, 1), sr=1, channel_labels=tuple("abcde"))
    np.testing.assert_array_equal(mir.sum(x).data.ravel(), [10])
    assert mir.sum(x).channel_labels is None
    grouped = mir.sum(x, adjacent=2)
    np.testing.assert_array_equal(grouped.data.ravel(), [1, 5, 4])
    assert grouped.channel_labels == ("a .. b", "c .. d", "e .. e")


def test_sum_mean_divides_by_total_channel_count():
    x = Signal(np.ones((4, 3, 2)), sr=1)
    np.testing.assert_array_equal(mir.sum(x, mean=True).data, np.ones((1, 3, 2)))
    np.testing.assert_array_equal(mir.sum(x, mean=True, adjacent=2).data, np.full((2, 3, 2), 0.5))


def test_sum_of_filterbank_is_mono():
    summed = mir.sum(filterbank(noise()))
    assert summed.n_channels == 1 and isinstance(summed, Audio)
