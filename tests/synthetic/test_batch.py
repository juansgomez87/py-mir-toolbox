from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import mirpy as mir
from mirpy import batch, list_audio

SR = 8000


def write_tone(path, freq, seconds=0.5):
    t = np.arange(int(SR * seconds)) / SR
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), SR)
    return path


@pytest.fixture
def songs(tmp_path):
    write_tone(tmp_path / "b.wav", 440)
    write_tone(tmp_path / "a.wav", 220)
    write_tone(tmp_path / "sub" / "c.wav", 880)
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


def file_names(rows):
    return [Path(next(iter(row.values())).source).name for row in rows]


def test_folder_files_are_sorted_and_filtered(songs):
    assert [p.name for p in list_audio(songs)] == ["a.wav", "b.wav"]
    assert [p.name for p in list_audio(songs, recursive=True)] == ["a.wav", "b.wav", "c.wav"]


def test_explicit_files_keep_their_order(songs):
    files = [songs / "b.wav", songs / "a.wav"]
    assert list_audio(files) == files


def test_missing_or_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_audio(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="No files"):
        list_audio(tmp_path)


def test_one_row_per_file_matches_direct_calls(songs):
    rows = batch(songs, {"rms": mir.rms})
    assert file_names(rows) == ["a.wav", "b.wav"]
    expected = mir.rms(mir.audio(songs / "a.wav")).data
    np.testing.assert_array_equal(rows[0]["rms"].data, expected)


def test_framed_results_are_summarised_as_mirexport(songs):
    rows = batch(songs, {"centroid": lambda a: mir.centroid(a, frame=True)})
    assert sorted(rows[0]) == ["centroid_Mean", "centroid_Std"]
    framed = mir.centroid(mir.audio(songs / "a.wav"), frame=True)
    np.testing.assert_array_equal(rows[0]["centroid_Mean"].data, mir.mean(framed).data)
    table = mir.file_table(rows)
    assert table["file"] == ["a.wav", "b.wav"]


def test_summary_false_keeps_frames(songs):
    rows = batch(songs, {"rms": lambda a: mir.rms(a, frame=True)}, summary=False)
    assert rows[0]["rms"].is_framed


def test_dict_results_are_flattened(songs):
    rows = batch(songs, {"rms": lambda a: mir.stat(mir.rms(a, frame=True))})
    assert sorted(rows[0]) == sorted(
        f"rms_{field}" for field in mir.stat(mir.rms(mir.audio(songs / "a.wav"), frame=True))
    )


def test_audio_options_are_passed(songs):
    rows = batch(songs, {"rms": mir.rms}, audio_options={"extract": (0.0, 0.1)})
    expected = mir.rms(mir.audio(songs / "a.wav", extract=(0.0, 0.1))).data
    np.testing.assert_array_equal(rows[0]["rms"].data, expected)
    assert not np.array_equal(expected, mir.rms(mir.audio(songs / "a.wav")).data)


def failing(a):
    if "b.wav" in a.source:
        raise ValueError("broken file")
    return mir.rms(a)


def test_errors_raise_or_skip(songs):
    with pytest.raises(RuntimeError, match="b.wav"):
        batch(songs, {"rms": failing})
    with pytest.warns(UserWarning, match="b.wav"):
        rows = batch(songs, {"rms": failing}, on_error="skip")
    assert file_names(rows) == ["a.wav"]
    with pytest.raises(ValueError, match="on_error"):
        batch(songs, {"rms": mir.rms}, on_error="ignore")


def test_parallel_gives_the_same_rows(songs):
    pytest.importorskip("joblib")
    serial = batch(songs, {"rms": mir.rms}, recursive=True)
    parallel = batch(songs, {"rms": mir.rms}, recursive=True, n_jobs=2)
    assert file_names(parallel) == file_names(serial)
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a["rms"].data, b["rms"].data)
