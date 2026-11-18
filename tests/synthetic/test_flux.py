import numpy as np
import pytest

from mirpy import Audio, Signal, Spectrum, flux, mfcc, spectrum

SR = 8000


def tone(freq=440.0, n=SR):
    return Audio.from_array(np.sin(2 * np.pi * freq * np.arange(n) / SR), sr=SR)


def frames(columns, cls=Signal):
    """Framed curve with one frame per column."""
    data = np.asarray(columns, dtype=float)[np.newaxis]
    n = data.shape[-1]
    return cls(
        data,
        sr=SR,
        x_values=np.arange(data.shape[1], dtype=float),
        frame_rate=10.0,
        frame_length=0.1,
        frame_starts=np.arange(n) * 0.1,
    )


def test_distances_between_successive_frames():
    x = frames([[0, 3, 3], [0, 4, 0]])
    np.testing.assert_allclose(flux(x).data.ravel(), [5, 4])
    np.testing.assert_allclose(flux(x, dist="cityblock").data.ravel(), [7, 4])
    np.testing.assert_allclose(flux(x, dist="City").data.ravel(), [7, 4])
    cosine = flux(frames([[1, 0, 1], [0, 1, 1]]), dist="cosine").data.ravel()
    np.testing.assert_allclose(cosine, [1, 1 - 1 / np.sqrt(2)])


def test_increase_only_sums_positive_differences():
    x = frames([[0, 3, 0], [0, 4, 8]])
    np.testing.assert_allclose(flux(x, inc=True).data.ravel(), [5, 4])
    np.testing.assert_allclose(flux(x, inc=True, dist="cityblock").data.ravel(), [7, 4])
    with pytest.raises(ValueError, match="dist"):
        flux(x, inc=True, dist="cosine")


def test_value_sits_on_the_second_frame_of_each_pair():
    x = frames([[0, 1, 2, 3]])
    f = flux(x)
    np.testing.assert_allclose(f.frame_starts, [0.1, 0.2, 0.3])
    assert f.frame_length == x.frame_length and f.frame_rate == x.frame_rate


def test_audio_is_framed_and_turned_into_a_spectrum():
    x = tone()
    f = flux(x)
    expected = flux(spectrum(x, frame=True))
    np.testing.assert_array_equal(f.data, expected.data)
    assert f.name == "Spectral flux" and f.n_frames == 38


def test_stationary_tone_has_little_flux_and_onset_has_a_peak():
    t = np.arange(SR) / SR
    step = Audio.from_array(np.where(t < 0.5, 0, np.sin(2 * np.pi * 440 * t)), sr=SR)
    f = flux(step).data[0, 0]
    assert f.max() > 100 * np.median(f[20:])


def test_flux_of_other_curves():
    c = mfcc(tone(), frame=True)
    f = flux(c)
    assert f.name == "MFCC flux" and f.n_frames == c.n_frames - 1


def test_unframed_curve_is_rejected():
    with pytest.raises(ValueError, match="framed"):
        flux(spectrum(tone()))


def test_complex_flux_drops_one_more_frame_and_peaks_at_onset():
    t = np.arange(SR) / SR
    step = Audio.from_array(np.where(t < 0.5, 0, np.sin(2 * np.pi * 440 * t)), sr=SR)
    f = flux(step, complex=True)
    assert f.n_frames == flux(step).n_frames - 1
    assert np.all(np.isfinite(f.data))
    np.testing.assert_allclose(f.frame_starts, flux(step).frame_starts[:-1])
    # value k compares frames k..k+2 but sits at frame k+1 (MiningSuite): one frame early
    assert 0.425 < f.frame_starts[np.argmax(f.data[0, 0])] < 0.475


def test_complex_needs_a_waveform():
    with pytest.raises(ValueError, match="complex"):
        flux(frames([[0, 1, 2]]), complex=True)


def test_median_and_halfwave():
    x = frames([[0, 1, 1, 1, 5, 5, 5]])  # flux: 1 0 0 4 0 0
    f = flux(x, median=(0.1, 1.0)).data.ravel()  # 1 frame each side at 10 Hz
    np.testing.assert_allclose(f, [1 - 0.5, 0, 0, 4, 0, 0])
    rectified = flux(x, median=(0.1, 2.0), halfwave=True).data.ravel()
    assert rectified.min() == 0 and rectified[3] == 4
    np.testing.assert_array_equal(flux(x, median=True).data, flux(x, median=(0.2, 1.0)).data)


def test_sub_band_gives_one_flux_per_band():
    sr = 44100
    x = Audio.from_array(np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr), sr=sr)
    assert flux(x, sub_band="manual").n_channels == 10
    assert flux(x, sub_band="gammatone").n_channels == 10


def test_rejects_unknown_options():
    with pytest.raises(ValueError, match="dist"):
        flux(tone(), dist="hamming")
    with pytest.raises(ValueError, match="sub_band"):
        flux(tone(), sub_band="octave")


def test_spectrum_class_kept_for_name():
    s = frames([[0, 1]], cls=Spectrum)
    assert flux(s).name == "Spectral flux"
