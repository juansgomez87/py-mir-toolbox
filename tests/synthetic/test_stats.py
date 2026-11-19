import numpy as np
import pytest

from mirpy import Audio, Histogram, Signal, histo, mean, median, mfcc, rms, stat, std

SR = 8000
FRAME_RATE = 40.0


def framed(values, name="RMS"):
    """A framed feature: values of shape (n_frames,) or (n_elements, n_frames)."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    n = values.shape[-1]
    return Signal(
        values[np.newaxis],
        sr=SR,
        name=name,
        frame_rate=FRAME_RATE,
        frame_length=0.05,
        frame_starts=np.arange(n) / FRAME_RATE,
    )


def test_histo_of_waveform_matches_numpy():
    y = np.random.default_rng(0).standard_normal(1000)
    h = histo(Audio.from_array(y, sr=SR))
    counts, edges = np.histogram(y, bins=10)
    np.testing.assert_array_equal(h.data.ravel(), counts)
    np.testing.assert_allclose(h.edges, edges)
    np.testing.assert_allclose(h.x_values, (edges[:-1] + edges[1:]) / 2)
    assert isinstance(h, Histogram) and not h.is_framed


def test_histo_of_framed_feature_counts_over_frames():
    h = histo(framed([0, 0, 0.5, 2, np.nan]), number=2)
    np.testing.assert_array_equal(h.data.ravel(), [3, 1])
    assert not h.is_framed


def test_histo_counts_a_value_on_an_inner_edge_in_both_bins():
    # MIRtoolbox 1.8.2 (pianoA4 fixture): bins are closed intervals
    h = histo(framed([0, 0, 1, 2]), number=2)
    np.testing.assert_array_equal(h.data.ravel(), [3, 2])
    ampli = histo(framed([0, 0, 1, 2]), number=2, ampli=True)
    np.testing.assert_array_equal(ampli.data.ravel(), [1, 3])


def test_histo_ampli_sums_values():
    h = histo(framed([1, 1, 4]), number=2, ampli=True)
    np.testing.assert_array_equal(h.data.ravel(), [2, 4])


def test_histo_of_spectrum_frames_keeps_frames():
    x = framed(np.arange(12.0).reshape(4, 3), name="Spectrum")
    h = histo(x, number=4)
    assert h.data.shape == (1, 4, 3) and h.is_framed
    np.testing.assert_array_equal(h.data.sum(axis=1), 4)


def test_mean_median_std_reduce_frames_ignoring_nan():
    x = framed([[1, 2, np.nan, 6], [0, 0, 0, 0]])
    np.testing.assert_allclose(mean(x).data[0, :, 0], [3, 0])
    np.testing.assert_allclose(median(x).data[0, :, 0], [2, 0])
    np.testing.assert_allclose(std(x).data[0, :, 0], [np.std([1, 2, 6], ddof=1), 0])
    assert mean(x).name == "RMS_Mean" and not mean(x).is_framed


def test_std_of_single_value_is_zero_like_matlab():
    assert std(framed([5.0])).data.item() == 0
    assert np.isnan(std(framed([np.nan, np.nan])).data.item())


def test_mean_keeps_mfcc_ranks():
    c = mfcc(Audio.from_array(np.random.default_rng(1).standard_normal(SR), sr=SR), frame=True)
    m = mean(c)
    assert m.data.shape == (1, 13, 1)
    np.testing.assert_array_equal(m.x_values, c.x_values)
    np.testing.assert_allclose(m.data[0, :, 0], c.data[0].mean(axis=1))


def test_stat_fields_and_names():
    s = stat(framed(np.arange(10.0)))
    assert list(s) == ["Mean", "Std", "Slope", "PeriodFreq", "PeriodAmp", "PeriodEntropy"]
    assert all(v.name == f"RMS_{k}" and not v.is_framed for k, v in s.items())


def test_stat_slope_of_line():
    n = 50
    up = stat(framed(np.linspace(0, 10, n)))["Slope"].data.item()
    # z-scored line against T in [-0.5, 0.5]: slope = 1 / std(T, ddof=1)
    assert up == pytest.approx(1 / np.linspace(-0.5, 0.5, n).std(ddof=1))
    assert stat(framed(np.linspace(10, 0, n)))["Slope"].data.item() == pytest.approx(-up)


def test_stat_period_of_sinusoidal_curve():
    frames = np.arange(200)
    curve = 1 + np.sin(2 * np.pi * 2.0 * frames / FRAME_RATE)  # 2 Hz at 40 frames/s
    s = stat(framed(curve))
    assert s["PeriodFreq"].data.item() == pytest.approx(2.0, rel=0.05)
    # plain autocorrelation (no centring, no window): overlap shrinks with the lag
    period = round(FRAME_RATE / 2.0)
    expected = np.sum(curve[:-period] * curve[period:]) / np.sum(curve**2)
    assert s["PeriodAmp"].data.item() == pytest.approx(expected)
    assert 0 < s["PeriodEntropy"].data.item() < 1


def test_stat_period_is_nan_without_peak():
    s = stat(framed(np.linspace(0, 1, 20)))
    assert np.isnan(s["PeriodFreq"].data.item()) and np.isnan(s["PeriodAmp"].data.item())


def test_stat_period_peak_can_be_the_last_lag():
    from mirpy.stats.summary import _local_maxima

    r = np.array([1.0, 0.5, -0.2, 0.1, 0.3])
    np.testing.assert_array_equal(_local_maxima(r), [4])


def test_stat_period_amp_is_zero_when_no_maximum_is_positive():
    from mirpy.stats.summary import _periodicity

    curve = np.array([3.0, 2, 1, 0, -1, -2, -3, -2])  # autocorrelation rises only to -0.19
    freq, amp, _ = _periodicity(curve, FRAME_RATE)
    assert np.isnan(freq) and amp == 0.0


def test_stat_discards_nan_frames():
    with_nan = stat(framed([1, 2, np.nan, 4, 5]))
    assert with_nan["Mean"].data.item() == 3
    assert np.isfinite(with_nan["Slope"].data.item())


def test_stat_of_unframed_feature_is_the_value():
    x = rms(Audio.from_array(np.ones(100), sr=SR))
    assert stat(x) == {"Mean": x}
