import matplotlib
import numpy as np
import pytest

from mirpy import Audio, Signal, frame

matplotlib.use("Agg")

SR = 1000


@pytest.fixture
def audio():
    return Audio.from_array(np.sin(np.arange(SR) / 10), sr=SR)


def test_waveform_plotted_against_time(audio):
    (ax,) = audio.plot()
    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), audio.times)
    assert ax.get_xlabel() == "time (s)"


def test_framed_scalar_plotted_against_frame_centres():
    x = Signal(
        data=np.arange(3.0).reshape(1, 1, 3),
        sr=SR,
        name="RMS",
        frame_rate=2,
        frame_length=0.5,
        frame_starts=[0, 0.5, 1],
    )
    (ax,) = x.plot()
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), [0.25, 0.75, 1.25])


def test_framed_multi_element_drawn_as_image(audio):
    (ax,) = frame(audio, (0.1, 0.5)).plot()
    assert ax.collections[0].get_array().shape == (100, 19)


def test_one_subplot_per_channel():
    stereo = Audio.from_array(np.zeros((2, SR)), sr=SR, mono=False)
    assert len(stereo.plot()) == 2


def test_play_returns_player(audio):
    display = pytest.importorskip("IPython.display")
    assert isinstance(audio.play(), display.Audio)


def test_play_rejects_framed(audio):
    pytest.importorskip("IPython")
    with pytest.raises(ValueError, match="unframed"):
        frame(audio).play()
