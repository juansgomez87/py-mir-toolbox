import math

import numpy as np
import pytest
import soundfile

from mirpy import audio, get_data
from mirpy.core.matlab_compat import matlab_round

SR = 1000


def loop_trim_bounds(y, where, threshold):
    """Line-by-line transcription of MiningSuite trim.m; returns 1-based [n1, n2]."""
    trimframe, trimhop = 100, 10
    nframes = math.floor((len(y) - trimframe) / trimhop) + 1
    rms = []
    for j in range(1, nframes + 1):
        st = math.floor((j - 1) * trimhop) + 1
        d0 = y[st - 1 : st + trimframe - 1]
        rms.append(np.linalg.norm(d0) / math.sqrt(trimframe))
    rms = np.array(rms)
    rms = (rms - rms.min()) / (rms.max() - rms.min())
    nosil = [i + 1 for i in np.flatnonzero(rms > threshold)]
    n1, n2 = 1, len(y)
    if where in ("start", "both"):
        nosil1 = min(nosil)
        if nosil1 > 1:
            nosil1 -= 1
        n1 = math.floor((nosil1 - 1) * trimhop) + 1
    if where in ("end", "both"):
        nosil2 = max(nosil)
        if nosil2 < len(rms):
            nosil2 += 1
        n2 = math.floor((nosil2 - 1) * trimhop) + 1
    return n1, n2


def padded_tone(n_before=1234, n_tone=2000, n_after=987):
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 50 * np.arange(n_tone) / SR)
    quiet = 1e-4 * rng.standard_normal(n_before + n_after)
    return np.concatenate([quiet[:n_before], tone, quiet[n_before:]])


def test_matlab_round_rounds_half_away_from_zero():
    np.testing.assert_array_equal(matlab_round([0.5, 1.5, 2.5, -0.5, -2.5]), [1, 2, 3, -1, -3])


def test_array_needs_sr_and_file_rejects_it(tmp_path):
    with pytest.raises(ValueError, match="sr is required"):
        audio(np.zeros(10))
    path = tmp_path / "x.wav"
    soundfile.write(path, np.zeros(100), SR)
    with pytest.raises(ValueError, match="only accepted with an array"):
        audio(path, sr=SR)


def test_file_is_read_and_mixed(tmp_path):
    path = tmp_path / "stereo.wav"
    soundfile.write(path, np.column_stack([np.full(100, 0.25), np.full(100, 0.75)]), SR)
    x = audio(path)
    assert x.sr == SR and x.source == str(path)
    np.testing.assert_allclose(get_data(x), 0.5, atol=1e-4)


def test_extract_seconds_includes_both_ends():
    x = audio(np.arange(SR, dtype=float), sr=SR, extract=(0.1, 0.2))
    np.testing.assert_array_equal(get_data(x), np.arange(100, 201))
    assert x.start == pytest.approx(0.1)
    np.testing.assert_allclose(x.times[0], 0.1)


def test_extract_seconds_uses_matlab_rounding():
    x = audio(np.arange(10, dtype=float), sr=2, extract=(1.25, 2.25))  # 2.5 -> 3, 4.5 -> 5
    np.testing.assert_array_equal(get_data(x), [3, 4, 5])


def test_extract_seconds_clips_to_signal():
    x = audio(np.arange(SR, dtype=float), sr=SR, extract=(-1, np.inf))
    assert x.n_elements == SR


def test_extract_samples_is_python_slice():
    y = np.arange(SR, dtype=float)
    x = audio(y, sr=SR, extract=(10, 20), extract_unit="sp")
    np.testing.assert_array_equal(get_data(x), y[10:20])


def test_empty_extract_rejected():
    with pytest.raises(ValueError, match="selects no samples"):
        audio(np.zeros(SR), sr=SR, extract=(20, 10), extract_unit="sp")


def test_channels_selected_before_mixing():
    y = np.vstack([np.full(100, 1.0), np.full(100, 2.0), np.full(100, 4.0)])
    np.testing.assert_allclose(get_data(audio(y, sr=SR, channels=2)), 4.0)
    np.testing.assert_allclose(get_data(audio(y, sr=SR, channels=[0, 1])), 1.5)
    assert audio(y, sr=SR, mono=False, channels=[0, 2]).n_channels == 2


def test_center_removes_mean_per_channel():
    y = np.vstack([np.arange(100.0), 5 + np.arange(100.0)])
    x = audio(y, sr=SR, mono=False, center=True)
    np.testing.assert_allclose(x.data.mean(axis=1), 0, atol=1e-12)


def test_center_uses_excerpt_mean():
    y = np.concatenate([np.zeros(500), np.ones(500)])
    x = audio(y, sr=SR, extract=(500, 1000), extract_unit="sp", center=True)
    np.testing.assert_allclose(get_data(x), 0)


def test_resample_changes_rate_and_length():
    x = audio(np.sin(np.arange(SR) / 10), sr=SR, sampling=500)
    assert x.sr == 500
    assert x.n_elements == SR // 2


def test_resample_needs_whole_rates():
    with pytest.raises(ValueError, match="whole-number"):
        audio(np.zeros(SR), sr=SR, sampling=440.5)


def test_fwr_is_absolute_value():
    y = np.array([-1.0, 2.0, -3.0])
    np.testing.assert_array_equal(get_data(audio(y, sr=SR, fwr=True)), [1, 2, 3])


@pytest.mark.parametrize("where", ["both", "start", "end"])
def test_trim_matches_trim_m_loop(where):
    y = padded_tone()
    n1, n2 = loop_trim_bounds(y, where, 0.06)
    x = audio(y, sr=SR, trim=where)
    np.testing.assert_array_equal(get_data(x), y[n1 - 1 : n2])
    assert x.start == pytest.approx((n1 - 1) / SR)


def test_trim_true_means_both():
    y = padded_tone()
    np.testing.assert_array_equal(
        get_data(audio(y, sr=SR, trim=True)), get_data(audio(y, sr=SR, trim="both"))
    )


def test_trim_on_constant_signal_warns_and_keeps_signal():
    with pytest.warns(UserWarning, match="not trimmed"):
        x = audio(np.ones(SR), sr=SR, trim=True)
    assert x.n_elements == SR


def test_frame_option_frames_the_result():
    x = audio(np.zeros(SR), sr=SR, frame=(0.1, 0.5))
    assert x.is_framed and x.data.shape == (1, 100, 19)
    assert x.params["center"] is False and "frame" in x.params
