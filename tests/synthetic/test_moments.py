import numpy as np
import pytest
import scipy.stats

from mirpy import (
    Audio,
    Signal,
    Spectrum,
    centroid,
    entropy,
    flatness,
    kurtosis,
    skewness,
    spectrum,
    spread,
)

SR = 8192


def curve(values, positions=None):
    """A one-frame distribution with explicit positions (no gates apply)."""
    values = np.asarray(values, dtype=float)
    positions = np.arange(values.size, dtype=float) if positions is None else positions
    return Signal(values.reshape(1, -1, 1), sr=SR, x_values=positions)


def flat_spectrum(n_bins=64, n_frames=1, level=1.0):
    data = np.full((1, n_bins, n_frames), level)
    return Spectrum(data, sr=SR, x_values=np.arange(n_bins) * 10.0)


def tone(freq, n=SR):
    return Audio.from_array(np.sin(2 * np.pi * freq * np.arange(n) / SR), sr=SR)


def test_moments_match_weighted_statistics():
    rng = np.random.default_rng(0)
    weights = rng.random(50)
    f = np.linspace(0, 1000, 50)
    shifted = weights + weights.min()  # MIRtoolbox adds each curve's minimum
    p = shifted / shifted.sum()
    mean = np.sum(f * p)
    sd = np.sqrt(np.sum((f - mean) ** 2 * p))
    x = curve(weights, f)
    assert centroid(x, max_entropy=None).data.item() == pytest.approx(mean)
    assert spread(x).data.item() == pytest.approx(sd)
    assert skewness(x).data.item() == pytest.approx(np.sum((f - mean) ** 3 * p) / sd**3)
    assert kurtosis(x).data.item() == pytest.approx(np.sum((f - mean) ** 4 * p) / sd**4)


def test_kurtosis_is_not_excess_kurtosis():
    f = np.linspace(-5, 5, 2001)
    gaussian = curve(scipy.stats.norm.pdf(f), f)
    assert kurtosis(gaussian).data.item() == pytest.approx(3, abs=1e-3)  # sampled, truncated
    assert skewness(gaussian).data.item() == pytest.approx(0, abs=1e-9)


def test_centroid_of_tone_is_its_frequency():
    assert centroid(tone(1024)).data.item() == pytest.approx(1024, rel=1e-4)


def test_audio_input_uses_its_spectrum():
    x = tone(500)
    np.testing.assert_array_equal(spread(x).data, spread(spectrum(x)).data)
    assert centroid(x).name == "Spectral centroid"
    assert centroid(curve([1, 2])).name == "Centroid"


def test_framed_values_sit_on_frames():
    c = centroid(tone(1024), frame=True)
    assert c.is_framed and c.data.shape == (1, 1, 39)
    assert c.unit == "Hz"


def test_flat_distribution_entropy_and_flatness_are_one():
    s = flat_spectrum()
    assert entropy(s).data.item() == pytest.approx(1, abs=1e-9)
    assert flatness(s).data.item() == pytest.approx(1)


def test_single_peak_entropy_and_flatness_are_zero():
    peak = curve(np.eye(1, 64, 10).ravel())
    assert entropy(peak).data.item() == pytest.approx(0, abs=1e-9)
    assert flatness(peak).data.item() == 0


def test_entropy_rectifies_and_centers():
    x = curve([-1.0, 1.0, 1.0, -1.0])
    assert entropy(x).data.item() == pytest.approx(np.log(2) / np.log(4))
    y = curve([3.0, 1.0, 1.0, 3.0])  # centred: [1, -1, -1, 1]
    assert entropy(y, center=True).data.item() == pytest.approx(np.log(2) / np.log(4))


def test_max_entropy_gives_nan_on_flat_spectrum():
    s = flat_spectrum()
    assert np.isnan(centroid(s).data.item())
    assert centroid(s, max_entropy=None).data.item() == pytest.approx(315)
    assert centroid(s, max_entropy=1).data.item() == pytest.approx(315)


def test_min_rms_gives_nan_on_quiet_frames():
    data = np.ones((1, 64, 3))
    data[0, 10, :] = 50  # peaky enough to pass the entropy gate
    data[:, :, 1] *= 1e-4
    s = Spectrum(data, sr=SR, x_values=np.arange(64) * 10.0)
    for operator in (centroid, spread, flatness, entropy):
        values = operator(s).data[0, 0]
        assert np.isnan(values[1]) and not np.isnan(values[[0, 2]]).any(), operator.__name__
        assert not np.isnan(operator(s, min_rms=0).data).any(), operator.__name__
