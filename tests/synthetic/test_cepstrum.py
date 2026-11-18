import math

import numpy as np
import pytest

from mirpy import Audio, Cepstrum, cepstrum, frame, spectrum

SR = 22050


def harmonic_tone(f0=200.0, n=2048, sr=SR):
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 20) if f0 * h < sr / 2)
    return Audio.from_array(y, sr=sr)


def reference_cepstrum(magnitude, sr, low, high):
    """Transcription of real_cepstrum in cepstrum.m (1-based ranges)."""
    mirrored = np.r_[magnitude[:-1], magnitude[::-1]]
    y = np.abs(np.fft.fft(np.log(mirrored + 1e-12)))
    start = math.ceil(low * sr) + 1
    length = min(math.ceil(high * sr), len(y) // 2)
    return y[start - 1 : length]


def test_matches_cepstrum_m_transcription():
    x = harmonic_tone()
    c = cepstrum(x)
    magnitude = spectrum(x).data[0, :, 0]
    np.testing.assert_allclose(c.data[0, :, 0], reference_cepstrum(magnitude, SR, 0.0002, 0.05))
    assert isinstance(c, Cepstrum) and c.x_unit == "s"
    assert c.x_values[0] == pytest.approx(math.ceil(0.0002 * SR) / SR)


def test_peak_at_fundamental_period():
    c = cepstrum(harmonic_tone(f0=250), min_quefrency=0.002)
    assert 1 / c.x_values[np.argmax(c.data[0, :, 0])] == pytest.approx(250, rel=0.02)


def test_range_capped_at_half_the_mirrored_length():
    c = cepstrum(harmonic_tone(n=512), max_quefrency=1.0)
    assert c.n_elements == (2 * (512 // 2 + 1) - 1) // 2 - math.ceil(0.0002 * SR)


def test_frequency_bounds():
    by_freq = cepstrum(harmonic_tone(), min_freq=100, max_freq=1000)
    by_quef = cepstrum(harmonic_tone(), min_quefrency=1 / 1000, max_quefrency=1 / 100)
    np.testing.assert_array_equal(by_freq.data, by_quef.data)


def test_freq_axis_is_inverse_quefrency():
    c = cepstrum(harmonic_tone(), freq=True)
    q = cepstrum(harmonic_tone())
    np.testing.assert_allclose(c.x_values, 1 / q.x_values)
    assert c.x_name == "frequency" and c.x_unit == "Hz"


def test_spectrum_and_framed_inputs():
    x = harmonic_tone(n=SR // 2)
    framed = cepstrum(x, frame=(0.05, 0.5))
    assert framed.n_frames == frame(x, (0.05, 0.5)).n_frames
    np.testing.assert_array_equal(cepstrum(spectrum(x)).data, cepstrum(x).data)
    assert cepstrum(framed) is framed
