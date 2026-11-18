import numpy as np
import pytest
import scipy.fft

from mirpy import Audio, Mfcc, mfcc, spectrum
from mirpy.features.mfcc import _delta, dct_matrix

SR = 8000


def noise(n=SR, seed=0):
    return Audio.from_array(np.random.default_rng(seed).standard_normal(n), sr=SR)


def test_dct_matrix_is_orthonormal_dct_ii():
    n = 40
    identity = np.eye(n)
    expected = scipy.fft.dct(identity, type=2, norm="ortho", axis=0)
    np.testing.assert_allclose(dct_matrix(n, np.arange(n)), expected, atol=1e-12)


def test_mfcc_is_dct_of_mel_log_spectrum():
    x = noise()
    mel = spectrum(x, frame=True, band="mel", log=True)
    expected = scipy.fft.dct(mel.data, type=2, norm="ortho", axis=1)[:, 1:14]
    c = mfcc(x, frame=True)
    np.testing.assert_allclose(c.data, expected, atol=1e-10)
    np.testing.assert_array_equal(c.x_values, np.arange(1, 14))
    np.testing.assert_array_equal(c.frame_starts, mel.frame_starts)
    assert isinstance(c, Mfcc) and c.delta == 0


def test_rank_zero_and_bands():
    c = mfcc(noise(), frame=True, rank=range(13), bands=20)
    assert c.data.shape == (1, 13, 39)
    np.testing.assert_array_equal(c.x_values, np.arange(13))


def test_return_spectrum():
    c, mel = mfcc(noise(), frame=True, return_spectrum=True)
    assert mel.n_elements == 40 and mel.n_frames == c.n_frames


def test_delta_of_linear_ramp_is_its_slope():
    c = mfcc(noise(), frame=True)
    ramp = np.arange(c.n_frames, dtype=float)
    linear = Mfcc(
        data=np.broadcast_to(ramp, c.data.shape).copy(),
        sr=SR,
        x_values=c.x_values,
        frame_rate=c.frame_rate,
        frame_length=c.frame_length,
        frame_starts=c.frame_starts,
    )
    d = _delta(linear, radius=2)
    np.testing.assert_allclose(d.data, 1.0)
    assert d.n_frames == c.n_frames - 4
    np.testing.assert_array_equal(d.frame_starts, c.frame_starts[2:-2])
    assert d.delta == 1 and d.name == "Delta-MFCC"


def test_delta_orders_drop_frames():
    base = mfcc(noise(), frame=True)
    assert mfcc(noise(), frame=True, delta=1).n_frames == base.n_frames - 4
    second = mfcc(noise(), frame=True, delta=2, radius=1)
    assert second.n_frames == base.n_frames - 4 and second.delta == 2


def test_delta_needs_enough_frames():
    with pytest.raises(ValueError, match="frames"):
        mfcc(noise(), delta=1)
