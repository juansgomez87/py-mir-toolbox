import math

import numpy as np
import pytest

from mirpy import Audio, Autocor, autocor, frame, peaks, spectrum
from mirpy.core.matlab_compat import matlab_window, xcorr
from mirpy.signal.autocor import hanning_autocor

SR = 44100


def noise(n=4410, sr=SR, seed=0, channels=1):
    y = np.random.default_rng(seed).standard_normal((channels, n))
    return Audio.from_array(y, sr=sr, mono=False)


def continuous_hann_autocor(tau):
    return (
        (1 - tau) * (2 + np.cos(2 * np.pi * tau)) + 3 / (2 * np.pi) * np.sin(2 * np.pi * tau)
    ) / 3


def reference_autocor(y, window, last):
    """Centre, window, xcorr 'coeff', divide by the continuous Hann autocor (MIRtoolbox 1.8.2)."""
    y = (y - y.mean()) * window
    c = xcorr(y, last, "coeff")[last:]
    return c / continuous_hann_autocor((np.arange(last + 1) + 1) / (len(y) + 1))


def test_default_matches_reference_steps():
    x = noise()
    a = autocor(x)
    last = min(math.ceil(4410 / 2) - 1, math.ceil(0.05 * SR))
    expected = reference_autocor(x.data[0, :, 0], matlab_window("hanning", 4410), last)
    np.testing.assert_allclose(a.data[0, :, 0], expected, rtol=1e-10)
    np.testing.assert_allclose(a.x_values, np.arange(last + 1) / SR)
    assert isinstance(a, Autocor) and a.normalized and a.x_unit == "s"


def test_default_max_lag_is_50_ms():
    a = autocor(noise(n=SR))
    assert a.x_values[-1] == pytest.approx(math.ceil(0.05 * SR) / SR)


def test_low_rate_signal_has_no_lag_limit_and_rectangular_window():
    x = noise(n=200, sr=100)
    a = autocor(x)
    assert a.n_elements == 100  # lags 0 .. ceil(200 / 2) - 1
    # rectangular window: divided by its sampled autocorrelation (no fixture covers this case)
    y = x.data[0, :, 0] - x.data[0, :, 0].mean()
    expected = xcorr(y, 99, "coeff")[99:] / xcorr(np.ones(200), 99, "coeff")[99:]
    np.testing.assert_allclose(a.data[0, :, 0], expected, rtol=1e-10)


@pytest.mark.parametrize(
    ("sr", "first", "last"),
    [(44100, 53, 441), (22050, 27, 221), (11025, 14, 111)],  # MIRtoolbox 1.8.2 fixtures
)
def test_explicit_bounds_round_up(sr, first, last):
    a = autocor(noise(sr=sr), min_lag=0.0012, max_lag=0.0100)
    assert round(a.x_values[0] * sr) == first
    assert round(a.x_values[-1] * sr) == last


def test_frequency_bounds_convert_to_lags():
    by_freq = autocor(noise(), min_freq=100, max_freq=1000)
    by_lag = autocor(noise(), min_lag=1 / 1000, max_lag=1 / 100)
    assert by_freq.x_values[0] == pytest.approx(45 / SR)  # ceil(44.1)
    np.testing.assert_array_equal(by_freq.data, by_lag.data)


def test_window_normalisation_restarts_at_the_first_returned_lag():
    # MIRtoolbox 1.8.2: lag k of a range starting at `first` is divided by
    # R((k - first + 1) / (n + 1)), not by R((k + 1) / (n + 1))
    cut = autocor(noise(), min_lag=0.002)
    full = autocor(noise())
    first = round(cut.x_values[0] * SR)
    n = 4410
    expected = (
        full.data[0, first : first + cut.n_elements, 0]
        * continuous_hann_autocor((np.arange(first, first + cut.n_elements) + 1) / (n + 1))
        / continuous_hann_autocor((np.arange(cut.n_elements) + 1) / (n + 1))
    )
    np.testing.assert_allclose(cut.data[0, :, 0], expected, rtol=1e-10)
    with pytest.raises(ValueError, match="either in seconds or in Hz"):
        autocor(noise(), min_lag=0.001, max_freq=1000)


def test_pulse_train_peaks_at_its_period():
    y = np.zeros(SR // 10)
    y[::147] = 1  # 300 Hz
    a = autocor(Audio.from_array(y, sr=SR))
    first = peaks(a, no_begin=True, order="abscissa").peaks
    assert first.index[0] == 147


def test_window_normalisation_off():
    x = noise()
    a = autocor(x, normal_window=False)
    assert not a.normalized
    y = x.data[0, :, 0] - x.data[0, :, 0].mean()
    np.testing.assert_allclose(
        a.data[0, :, 0], xcorr(y, a.n_elements - 1, "coeff")[a.n_elements - 1 :]
    )


def test_scale_options_are_passed_to_xcorr():
    x = noise()
    biased = autocor(x, scale="biased", normal_window=False)
    raw = autocor(x, scale="none", normal_window=False)
    np.testing.assert_allclose(biased.data, raw.data / 4410)


def test_coeff_x_channels_divides_by_energy_of_channel_sum():
    x = noise(channels=2)
    a = autocor(x, scale="coeff_x_channels", normal_window=False)
    centred = x.data - x.data.mean(axis=1, keepdims=True)
    energy = np.sum(centred.sum(axis=0) ** 2)
    np.testing.assert_allclose(a.data[:, 0, 0], np.sum(centred[:, :, 0] ** 2, axis=1) / energy)
    assert a.n_channels == 2


def test_generalized_autocorrelation():
    x = noise(n=1000)
    a = autocor(x, generalized=0.67)
    y = x.data[0, :, 0] - x.data[0, :, 0].mean()
    y = y * matlab_window("hanning", 1000)
    expected = np.fft.ifft(np.abs(np.fft.fft(y)) ** 0.67).real[: a.n_elements]
    expected = expected / hanning_autocor(np.arange(a.n_elements), 1000)
    np.testing.assert_allclose(a.data[0, :, 0], expected, atol=1e-12)
    assert a.normalized


def test_framed_input():
    a = autocor(noise(n=SR // 2), frame=(0.05, 0.5))
    framed = frame(noise(n=SR // 2), (0.05, 0.5))
    assert a.n_frames == framed.n_frames and a.is_framed
    np.testing.assert_array_equal(a.frame_starts, framed.frame_starts)


def test_autocor_of_spectrum_has_hz_lags_and_no_window():
    s = spectrum(noise())
    a = autocor(s)
    assert a.of_spectrum and a.x_unit == "Hz" and not a.normalized
    step = s.x_values[1] - s.x_values[0]
    np.testing.assert_allclose(a.x_values[:3], np.arange(3) * step)
    assert a.n_elements == math.ceil(s.n_elements / 2)


def test_freq_axis_is_inverse_lag_in_increasing_order():
    a = autocor(noise(), freq=True)
    lags = autocor(noise())
    # MIRtoolbox drops lag 0 and lists frequencies from low to high
    np.testing.assert_allclose(a.x_values, 1 / lags.x_values[1:][::-1])
    np.testing.assert_array_equal(a.data, lags.data[:, 1:][:, ::-1])
    assert a.x_name == "frequency" and a.x_unit == "Hz"


def test_halfwave_on_new_and_existing_autocor():
    a = autocor(noise())
    assert autocor(noise(), halfwave=True).data.min() == 0
    np.testing.assert_array_equal(autocor(a, halfwave=True).data, np.maximum(a.data, 0))
    assert autocor(a) is a


def test_lag_range_beyond_signal_warns():
    with pytest.warns(UserWarning, match="minimum lag set to 0"):
        autocor(noise(n=100), min_lag=1.0)


def test_unknown_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        autocor(noise(), scale="normalized")
