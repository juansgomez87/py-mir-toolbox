import math

import matplotlib
import numpy as np
import pytest

from mirpy import Audio, Peaks, Signal, frame, peaks

matplotlib.use("Agg")

SR = 100


def curve_signal(values, x_values=None):
    values = np.asarray(values, dtype=float)
    return Signal(values[None, :, None], sr=SR, x_values=x_values)


def matlab_search(y, x, cthr, first, thr, reso, resofirst, m, order, nobegin, noend):
    """Literal 1-based transcription of ``search`` in MiningSuite peaks.m.

    Uses index 0 as an unused slot so that ``ys[i]`` reads like MATLAB
    ``y(i)``. Returns 1-based peak indices and the precise positions and
    values.
    """
    ys = [None, math.inf if nobegin else -math.inf, *y, math.inf if noend else -math.inf]
    xs = [None, *x]
    n = len(ys) - 1  # length(y) after padding
    dy = [None] + [ys[i + 1] - ys[i] for i in range(1, n)]
    lim = max(cthr, thr)
    p = [i for i in range(1, n - 1) if ys[i + 1] >= lim and dy[i] > 0 and dy[i + 1] <= 0]
    if not p:
        return [], [], []

    def mn(lo, hi):  # min(y(lo:hi))
        return min(ys[lo : hi + 1]) if hi >= lo else math.inf

    if cthr:
        finalm = []
        j = 1
        mj = p[j - 1]
        jj = j + 1
        bufmin = math.inf
        bufmax = ys[mj + 1]
        oldbufmin = mn(1, p[0])
        while jj <= len(p):
            bufmin = min(bufmin, mn(p[jj - 2] + 2, p[jj - 1]))
            new = ys[p[jj - 1] + 1]
            if bufmax - bufmin < cthr:
                if new > bufmax and (new - bufmax > first or bufmax - oldbufmin < cthr):
                    j = jj
                    mj = p[j - 1]
                    bufmax = ys[mj + 1]
                    oldbufmin = min(oldbufmin, bufmin)
                    bufmin = math.inf
                elif new - bufmax <= first:
                    bufmax = max(bufmax, new)
                    oldbufmin = min(oldbufmin, bufmin)
            else:
                if bufmax - oldbufmin < cthr:
                    oldbufmin = min(oldbufmin, bufmin)
                else:
                    finalm.append(mj)
                    oldbufmin = bufmin
                bufmax = new
                j = jj
                mj = p[j - 1]
                bufmin = math.inf
            jj += 1
        if bufmax - oldbufmin >= cthr and bufmax - mn(p[j - 1] + 2, n) >= cthr:
            finalm.append(p[j - 1])
        p = finalm

    if reso:
        p = sorted(p, key=lambda i: -ys[i + 1])
        delete = []
        for j in range(1, len(p)):
            for jj in range(j + 1, len(p) + 1):
                if abs(xs[p[jj - 1]] - xs[p[j - 1]]) < reso:
                    delete.append(j if resofirst and p[j - 1] > p[jj - 1] else jj)
        p = [v for i, v in enumerate(p, start=1) if i not in delete]

    truncated = False
    if len(p) > m:
        p = sorted(p, key=lambda i: -ys[i + 1])[:m]
        truncated = True
    if order == "abscissa":
        p = sorted(p)
    elif not truncated:
        p = sorted(p, key=lambda i: -ys[i + 1])

    pp, pv = [], []
    for i in p:
        if 1 < i < n:  # MIRtoolbox: both neighbours exist (MiningSuite: i > 2)
            y0, ym, yp = ys[i + 1], ys[i], ys[i + 2]
            r = (yp - ym) / (2 * (2 * y0 - yp - ym))
            pv.append(y0 - 0.25 * (ym - yp) * r)
            pp.append(
                (1 - r) * xs[i] + r * xs[i + 1] if r >= 0 else (1 + r) * xs[i] - r * xs[i - 1]
            )
        else:
            pv.append(ys[i + 1])
            pp.append(xs[i])
    return p, pp, pv


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"contrast": 0.0},
        {"contrast": 0.3, "select_first": 0.15},
        {"threshold": 0.5},
        {"reso": 3.0},
        {"reso": 3.0, "reso_first": True},
        {"total": 3},
        {"total": 3, "order": "abscissa"},
        {"no_begin": True, "no_end": True},
    ],
)
def test_matches_literal_peaks_m(seed, options):
    rng = np.random.default_rng(seed)
    y = rng.random(60)
    y = (y - y.min()) / np.ptp(y)  # global normalisation is then the identity
    x = np.arange(60, dtype=float) * 0.5
    expected_index, expected_pos, expected_val = matlab_search(
        list(y),
        list(x),
        cthr=options.get("contrast", 0.1),
        first=options.get("select_first", 0.0),
        thr=options.get("threshold", 0.0),
        reso=options.get("reso"),
        resofirst=options.get("reso_first", False),
        m=options.get("total", math.inf),
        order=options.get("order", "amplitude"),
        nobegin=options.get("no_begin", False),
        noend=options.get("no_end", False),
    )
    found = peaks(curve_signal(y, x), **options).peaks
    np.testing.assert_array_equal(found.index, np.array(expected_index, dtype=int) - 1)
    np.testing.assert_allclose(found.pos, expected_pos)
    np.testing.assert_allclose(found.val, expected_val)


def test_docstring_curve_ordering():
    found = peaks(curve_signal([0, 1, 0, 0, 3, 0, 2, 0]), interpolate=False).peaks
    np.testing.assert_array_equal(found.index, [4, 6, 1])
    np.testing.assert_array_equal(found.rank, [0, 1, 2])
    found = peaks(curve_signal([0, 1, 0, 0, 3, 0, 2, 0]), order="abscissa").peaks
    np.testing.assert_array_equal(found.index, [1, 4, 6])
    np.testing.assert_array_equal(found.rank, [2, 0, 1])


def test_contrast_drops_shallow_bumps():
    y = [0, 1.0, 0.95, 0.97, 0.0, 0.5, 0.0]
    found = peaks(curve_signal(y), contrast=0.1, interpolate=False).peaks
    np.testing.assert_array_equal(np.sort(found.index), [1, 5])
    found = peaks(curve_signal(y), contrast=0.0, interpolate=False).peaks
    np.testing.assert_array_equal(np.sort(found.index), [1, 3, 5])


def test_plateau_peak_is_its_first_point():
    found = peaks(curve_signal([0, 1, 1, 1, 0]), interpolate=False).peaks
    np.testing.assert_array_equal(found.index, [1])


def test_edges_are_peaks_unless_excluded():
    y = [1.0, 0, 0.5, 0, 1.0]
    assert set(peaks(curve_signal(y), interpolate=False).peaks.index) == {0, 2, 4}
    found = peaks(curve_signal(y), no_begin=True, no_end=True).peaks
    np.testing.assert_array_equal(found.index, [2])


def test_valleys_find_minima_in_original_units():
    y = np.array([5.0, 2.0, 5.0, 3.0, 5.0])
    found = peaks(curve_signal(y), valleys=True, interpolate=False).peaks
    np.testing.assert_array_equal(found.index, [1, 3])
    np.testing.assert_allclose(found.val_original, [2.0, 3.0])


def test_interpolation_is_exact_on_a_parabola():
    x = np.linspace(-2, 2, 41)
    y = 4 - (x - 0.237) ** 2
    found = peaks(curve_signal(y, x)).peaks
    assert found.pos[0] == pytest.approx(0.237)
    assert found.val_original[0] == pytest.approx(4.0)
    assert found.val[0] == pytest.approx((4 - y.min()) / (y.max() - y.min()))


def test_val_original_undoes_global_normalisation():
    y = 10 + 3 * np.sin(np.linspace(0, 6 * np.pi, 200))
    found = peaks(curve_signal(y), interpolate=False).peaks
    np.testing.assert_allclose(found.val_original, y[found.index])
    assert found.val.max() <= 1.0


def test_local_normalisation_val_original():
    data = np.stack([np.sin(np.linspace(0, 4 * np.pi, 50)) * a for a in (1.0, 5.0)], axis=-1)
    x = Signal(data[None], sr=SR, frame_rate=1.0, frame_length=1.0, frame_starts=[0.0, 1.0])
    found = peaks(x, normalize="local", interpolate=False).peaks
    for f in (0, 1):
        sub = found.at(0, f)
        np.testing.assert_allclose(sub.val_original, data[sub.index, f], atol=1e-12)


def test_peak_on_last_point_is_not_interpolated():
    # MIRtoolbox 1.8.2 fixtures: the Nyquist-bin valley sits at 22050 Hz with its own value
    found = peaks(curve_signal([0, 0.2, 0.4, 0.6, 1.0], x_values=np.arange(5.0))).peaks
    np.testing.assert_array_equal(found.index, [4])
    assert found.pos[0] == 4.0 and found.val[0] == 1.0


def test_peak_on_second_point_is_interpolated():
    # MIRtoolbox interpolates index 1 (MiningSuite skipped it)
    found = peaks(curve_signal([0, 1.0, 0.2, 0, 0], x_values=np.arange(5.0))).peaks
    assert 1.0 < found.pos[0] < 1.5 and found.val[0] > 1.0  # pulled toward the higher side


def test_framed_one_element_feature_searches_frames():
    values = np.array([0, 1.0, 0, 2.0, 0])
    starts = np.arange(5) * 0.5
    x = Signal(values[None, None, :], sr=SR, frame_rate=2.0, frame_length=1.0, frame_starts=starts)
    found = peaks(x, interpolate=False).peaks
    assert found.axis == "frame"
    np.testing.assert_array_equal(found.index, [3, 1])
    np.testing.assert_array_equal(found.frame, found.index)
    np.testing.assert_allclose(found.pos, [2.0, 1.0])  # frame centres, as MiningSuite deframe


def test_framed_multi_element_signal_searches_each_frame():
    audio = Audio.from_array(np.sin(2 * np.pi * 5 * np.arange(SR) / SR), sr=SR)
    found = peaks(frame(audio, (0.5, 0.5))).peaks
    assert found.axis == "element"
    assert set(found.frame) == {0, 1, 2}
    for f in range(3):
        assert len(found.at(0, f)) >= 2


def test_peaks_kept_in_params_and_signal_unchanged():
    x = curve_signal([0, 1.0, 0])
    y = peaks(x, contrast=0.2)
    np.testing.assert_array_equal(y.data, x.data)
    assert x.peaks is None
    assert y.params["peaks"]["contrast"] == 0.2


def test_no_peaks_gives_empty_columns():
    found = peaks(curve_signal(np.zeros(10))).peaks
    assert len(found) == 0 and found.pos.dtype == np.float64


def test_invalid_options_rejected():
    with pytest.raises(ValueError, match="order"):
        peaks(curve_signal([0, 1, 0]), order="height")
    with pytest.raises(ValueError, match="normalize"):
        peaks(curve_signal([0, 1, 0]), normalize="No")


def test_peaks_columns_must_match():
    with pytest.raises(ValueError, match="same length"):
        Peaks("element", [0], [0], [0, 1], [0.0], [0.0], [0.0], [0])


def test_plot_marks_peaks():
    x = peaks(curve_signal([0, 1.0, 0, 2.0, 0], x_values=np.arange(5.0)), interpolate=False)
    (ax,) = x.plot()
    markers = ax.get_lines()[1]
    np.testing.assert_allclose(markers.get_xdata(), [3, 1])
    np.testing.assert_allclose(markers.get_ydata(), [2, 1])
