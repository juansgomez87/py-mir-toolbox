import numpy as np
import pytest
import soundfile

from mirpy import Audio, Signal, get_data

SR = 8000


def test_signal_requires_three_axes():
    with pytest.raises(ValueError, match="shape"):
        Signal(data=np.zeros(10), sr=SR)


def test_signal_is_immutable():
    x = Signal(data=np.zeros((1, 10, 1)), sr=SR)
    with pytest.raises(AttributeError):
        x.sr = 1
    with pytest.raises(TypeError):
        x.params["frame"] = True


def test_mismatched_x_values_rejected():
    with pytest.raises(ValueError, match="x_values"):
        Signal(data=np.zeros((1, 10, 1)), sr=SR, x_values=np.arange(5))


def test_mono_sums_channels():
    stereo = np.vstack([np.full(100, 0.25), np.full(100, 0.5)])
    x = Audio.from_array(stereo, sr=SR)
    assert x.data.shape == (1, 100, 1)
    np.testing.assert_allclose(get_data(x), 0.75)


def test_mono_false_keeps_channels():
    x = Audio.from_array(np.zeros((2, 100)), sr=SR, mono=False)
    assert x.n_channels == 2


def test_times_of_unframed_signal():
    x = Audio.from_array(np.zeros(4), sr=4, start=1.0)
    np.testing.assert_allclose(x.times, [1.0, 1.25, 1.5, 1.75])


def test_from_file_reads_channels_first(tmp_path):
    path = tmp_path / "stereo.wav"
    y = np.vstack([np.linspace(-0.5, 0.5, 200), np.zeros(200)])
    soundfile.write(path, y.T, SR, subtype="FLOAT")

    x = Audio.from_file(path, mono=False)
    assert x.sr == SR
    assert x.source == str(path)
    np.testing.assert_allclose(x.data[:, :, 0], y, atol=1e-7)
