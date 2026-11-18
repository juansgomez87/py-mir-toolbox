import numpy as np
import pytest

from mirpy import Audio, Signal, frame, lowenergy, rms

SR = 8000


def tone(freq=100.0, n=SR, amplitude=1.0):
    return Audio.from_array(amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR), sr=SR)


def energy_curve(values):
    values = np.asarray(values, dtype=float)
    return Signal(
        values.reshape(1, 1, -1),
        sr=SR,
        name="RMS",
        frame_rate=40.0,
        frame_length=0.05,
        frame_starts=np.arange(values.size) * 0.025,
    )


def test_rms_of_sine_is_amplitude_over_root_two():
    assert rms(tone(amplitude=0.5)).data.item() == pytest.approx(0.5 / np.sqrt(2), rel=1e-6)


def test_rms_divides_by_samples_not_seconds():
    x = Audio.from_array(np.full(SR // 2, 0.3), sr=SR)
    assert rms(x).data.item() == pytest.approx(0.3)


def test_framed_rms_keeps_frame_positions():
    x = tone()
    framed = frame(x, (0.05, 0.5))
    r = rms(x, frame=(0.05, 0.5))
    assert r.data.shape == (1, 1, framed.n_frames)
    np.testing.assert_array_equal(r.frame_starts, framed.frame_starts)
    np.testing.assert_allclose(r.data[0, 0], np.sqrt(np.mean(framed.data[0] ** 2, axis=0)))


def test_rms_median():
    x = Audio.from_array(np.array([1.0, 1.0, 1.0, 10.0]), sr=SR)
    assert rms(x, median=True).data.item() == pytest.approx(1.0)


def test_rms_per_channel():
    y = np.vstack([np.full(100, 1.0), np.full(100, 2.0)])
    r = rms(Audio.from_array(y, sr=SR, mono=False))
    np.testing.assert_allclose(r.data.ravel(), [1.0, 2.0])


def test_lowenergy_counts_frames_below_average():
    assert lowenergy(energy_curve([1, 1, 1, 9])).data.item() == 0.75
    assert lowenergy(energy_curve([9, 9, 9, 1])).data.item() == 0.25


def test_lowenergy_threshold_and_asr():
    curve = energy_curve([1, 2, 3, 4])  # mean 2.5; squares 1, 4, 9, 16 (mean 7.5)
    assert lowenergy(curve, threshold=0.5).data.item() == 0.25  # below 1.25
    assert lowenergy(curve, asr=True).data.item() == 0.25  # below 3.75
    assert lowenergy(curve, asr=True, threshold=1).data.item() == 0.5  # below 7.5


def test_lowenergy_frames_a_waveform():
    loud_then_quiet = np.concatenate([np.ones(SR // 4), 0.01 * np.ones(3 * SR // 4)])
    result = lowenergy(Audio.from_array(loud_then_quiet, sr=SR))
    assert not result.is_framed
    assert 0.7 < result.data.item() < 0.8


def test_lowenergy_of_audio_equals_lowenergy_of_its_rms():
    x = Audio.from_array(np.random.default_rng(1).standard_normal(SR), sr=SR)
    expected = lowenergy(rms(x, frame=True)).data
    np.testing.assert_array_equal(lowenergy(x).data, expected)
