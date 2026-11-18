import numpy as np
import pytest
from scipy.signal import lfilter

from mirpy import Audio, Spectrum, frame, get_data, peaks, spectrum
from mirpy.core.matlab_compat import matlab_window
from mirpy.signal import auditory_bands
from mirpy.signal.spectrum import fft_length

SR = 8000


def tone(freq, n=SR, sr=SR, amplitude=1.0):
    return Audio.from_array(amplitude * np.sin(2 * np.pi * freq * np.arange(n) / sr), sr=sr)


def test_matches_numpy_fft_of_windowed_frame():
    x = Audio.from_array(np.random.default_rng(0).standard_normal(1000), sr=SR)
    s = spectrum(x)
    expected = np.abs(np.fft.fft(x.data[0, :, 0] * matlab_window("hamming", 1000), 1024))[:513]
    np.testing.assert_allclose(s.data[0, :, 0], expected, rtol=1e-10)
    np.testing.assert_allclose(s.x_values, np.arange(513) * SR / 1024)
    assert isinstance(s, Spectrum) and s.sr == SR


@pytest.mark.parametrize(
    ("n", "kwargs", "expected"),
    [
        (1000, {}, 1024),
        (1024, {}, 1024),
        (1000, {"zero_pad": True}, 2048),
        (1000, {"zero_pad": 100}, 2048),
        (1000, {"length": 3000}, 3000),
        (1000, {"res": 3.0}, 2667),
        (100, {"min_res": 10.0}, 1024),
    ],
)
def test_fft_length(n, kwargs, expected):
    options = {"res": None, "min_res": 0.0, "length": None, "zero_pad": 0, **kwargs}
    assert fft_length(n, SR, **options) == expected


def test_odd_length_keeps_floor_half_plus_one_bins():
    s = spectrum(tone(100, n=999), length=999)
    assert s.n_elements == 500


def test_tone_peak_at_its_frequency():
    s = peaks(spectrum(tone(1234.5)), total=1)
    assert s.peaks.pos[0] == pytest.approx(1234.5, abs=1.0)


def test_framed_spectrum_keeps_frame_metadata():
    x = frame(tone(440), (0.05, 0.5))
    s = spectrum(x)
    assert s.n_frames == x.n_frames and s.is_framed
    np.testing.assert_array_equal(s.frame_starts, x.frame_starts)
    assert s.n_elements == 512 // 2 + 1  # 400-sample frames -> 512-point FFT
    np.testing.assert_array_equal(spectrum(tone(440), frame=(0.05, 0.5)).data, s.data)


def test_rectangular_window_and_phase():
    x = Audio.from_array(np.r_[1.0, np.zeros(7)], sr=SR)
    s = spectrum(x, window=None, phase=True)
    np.testing.assert_allclose(get_data(s), 1.0)
    np.testing.assert_allclose(s.phase[0, :, 0], 0.0, atol=1e-12)


def test_band_cut_keeps_bins_and_phase_aligned():
    s = spectrum(tone(440), min_freq=100, max_freq=1000, phase=True)
    assert s.x_values[0] >= 100 and s.x_values[-1] <= 1000
    assert s.phase.shape == s.data.shape
    with pytest.raises(ValueError, match="No spectrum bin"):
        spectrum(tone(440), min_freq=5000)


def test_power_normal_and_db():
    base = spectrum(tone(440))
    power = spectrum(tone(440), power=True)
    np.testing.assert_allclose(power.data, base.data**2)
    assert power.power == 2 and power.name == "Power Spectrum"
    normal = spectrum(tone(440), normal=True)
    np.testing.assert_allclose(np.linalg.norm(normal.data, axis=1), 1.0)
    db = spectrum(tone(440), db=True)
    np.testing.assert_allclose(db.data, 20 * np.log10(base.data + 1e-16))
    assert db.log_scale == 10 and db.power == 2
    np.testing.assert_allclose(spectrum(power, db=True).data, 10 * np.log10(power.data + 1e-16))


def test_db_crop_keeps_range_above_zero():
    s = spectrum(tone(440), db=40)
    assert s.data.max() == pytest.approx(40) and s.data.min() == pytest.approx(0)


def test_log_then_db_on_existing_spectrum():
    logged = spectrum(tone(440), log=True)
    assert logged.log_scale == 1
    np.testing.assert_allclose(spectrum(logged, db=True).data, spectrum(tone(440), db=True).data)


def test_moving_average_smoothing():
    base = spectrum(tone(440))
    s = spectrum(tone(440), smooth=5)
    np.testing.assert_allclose(s.data, lfilter(np.ones(5) / 5, 1, base.data, axis=1))


def test_time_smooth_averages_frames():
    x = frame(tone(440), (0.05, 0.5))
    base, s = spectrum(x), spectrum(x, time_smooth=3)
    np.testing.assert_allclose(s.data[:, :, 5], base.data[:, :, 3:6].mean(axis=2))


def test_gauss_smoothing_matches_after2_transcription():
    y = np.random.default_rng(2).random(50)
    sigma = 2
    g = 1 / sigma / 2 / np.pi * np.exp(-(np.arange(-8, 9) ** 2) / 2 / sigma**2)
    expected = lfilter(g, 1, np.r_[y, np.zeros(8)])[7:][:50]  # MATLAB y(4*sigma:end)
    s = Spectrum(y[None, :, None], sr=SR, x_values=np.arange(50.0))
    np.testing.assert_allclose(spectrum(s, gauss=sigma).data[0, :, 0], expected)


def test_mel_weights_are_unit_area_triangles():
    freqs = np.linspace(0, 11025, 8193)  # 40 bands reach about 6.9 kHz
    weights = auditory_bands.mel_weights(freqs)
    assert weights.shape == (40, 8193)
    step = freqs[1] - freqs[0]
    np.testing.assert_allclose(weights.sum(axis=1) * step, 1.0, rtol=0.02)
    assert not auditory_bands.has_undefined_bands(weights)


def test_mel_spectrum_shape_and_axis():
    s = spectrum(frame(tone(440), (0.1, 0.5)), band="mel")
    assert s.n_elements == 40 and s.is_band_spectrum
    np.testing.assert_array_equal(s.x_values, np.arange(1, 41))
    assert s.name == "Mel-Spectrum" and s.x_name == "Mel bands"
    assert spectrum(tone(440), band="mel", n_bands=20).n_elements == 20


def test_mel_on_coarse_spectrum_warns():
    with pytest.warns(UserWarning, match="Mel"):
        spectrum(tone(440, n=16), length=16, band="mel")  # 500 Hz bins


def test_bark_bands_partition_the_bins():
    freqs = np.arange(0, 4001, 10.0)
    weights = auditory_bands.bark_weights(freqs)
    assert weights.shape[0] == 18  # 4400 Hz is the first upper edge above 4000 Hz
    np.testing.assert_array_equal(weights.sum(axis=0), 1)


def test_terhardt_zeroes_first_bin_and_peaks_near_3_5_khz():
    freqs = np.array([200.0, 1000, 3300, 10000])
    weights = auditory_bands.terhardt_weights(freqs)
    assert weights[0] == 0
    assert np.argmax(weights) == 2


def test_mask_needs_band_spectrum():
    with pytest.warns(UserWarning, match="mask"):
        spectrum(tone(440), mask=True)
    masked = spectrum(tone(440), band="bark", mask=True)
    bark = spectrum(tone(440), band="bark")
    spread = auditory_bands.mask_spread(bark.n_elements)
    np.testing.assert_allclose(masked.data[0, :, 0], spread @ bark.data[0, :, 0] + 1e-16)


def test_unknown_band_rejected():
    with pytest.raises(ValueError, match="band"):
        spectrum(tone(440), band="erb")
