import numpy as np
import pytest

from mirpy.core.matlab_compat import matlab_round, matlab_window, nextpow2, xcorr


def test_matlab_round_rounds_half_away_from_zero():
    np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -0.5, -2.5]), [1, 2, 3, -1, -3])


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 0), (1, 0), (2, 1), (3, 2), (1024, 10), (1025, 11), (0.5, -1), (0.3, -1)],
)
def test_nextpow2(n, expected):
    assert nextpow2(n) == expected


def test_hamming_is_symmetric_matlab_form():
    n = 8
    expected = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    np.testing.assert_allclose(matlab_window("hamming", n), expected, atol=1e-15)


def test_hanning_has_no_zero_end_points():
    w = matlab_window("hanning", 4)
    np.testing.assert_allclose(w, 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, 5) / 5)))
    assert w[0] > 0 and w[-1] > 0


def test_rectangular_window():
    np.testing.assert_array_equal(matlab_window("rectangular", 3), [1, 1, 1])


def direct_xcorr(x, maxlag):
    n = len(x)
    return np.array(
        [
            sum(x[i + k] * x[i] for i in range(n) if 0 <= i + k < n)
            for k in range(-maxlag, maxlag + 1)
        ]
    )


@pytest.mark.parametrize("maxlag", [0, 3, 9, 15])
def test_xcorr_matches_direct_sum(maxlag):
    x = np.random.default_rng(0).standard_normal(10)
    np.testing.assert_allclose(xcorr(x, maxlag), direct_xcorr(x, maxlag), atol=1e-12)


def test_xcorr_scales():
    x = np.array([1.0, 2.0, 3.0])
    raw = np.array([3.0, 8.0, 14.0, 8.0, 3.0])
    np.testing.assert_allclose(xcorr(x, 2, "biased"), raw / 3)
    np.testing.assert_allclose(xcorr(x, 2, "unbiased"), raw / [1, 2, 3, 2, 1])
    np.testing.assert_allclose(xcorr(x, 2, "coeff"), raw / 14)


def test_xcorr_along_axis():
    x = np.random.default_rng(1).standard_normal((2, 12, 3))
    c = xcorr(x, 4, axis=1)
    assert c.shape == (2, 9, 3)
    np.testing.assert_allclose(c[1, :, 2], direct_xcorr(x[1, :, 2], 4), atol=1e-12)


def test_xcorr_rejects_unknown_scale():
    with pytest.raises(ValueError, match="scale"):
        xcorr([1.0], 0, "normalised")
