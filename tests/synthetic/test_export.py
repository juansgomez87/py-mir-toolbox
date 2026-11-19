import numpy as np
import pytest

from mirpy import Signal
from mirpy.core.export import file_table, frame_table, to_arff, to_csv, to_dataframe

SR = 8000


def scalar(value, name="RMS", source=None):
    return Signal(data=np.full((1, 1, 1), value), sr=SR, name=name, source=source)


def framed(values, name, starts=(0.0, 0.5, 1.0)):
    values = np.asarray(values, dtype=float).reshape(1, -1, len(starts))
    return Signal(
        data=values, sr=SR, name=name, frame_rate=2.0, frame_length=0.5, frame_starts=starts
    )


def test_file_table_one_row_per_file():
    files = [
        {"rms": scalar(0.1, source="a/one.wav")},
        {"rms": scalar(0.2, source="a/two.wav")},
    ]
    assert file_table(files) == {"file": ["one.wav", "two.wav"], "RMS": [0.1, 0.2]}


def test_multi_element_and_channel_columns():
    x = Signal(data=np.arange(4.0).reshape(2, 2, 1), sr=SR, name="Mel Spectrum")
    assert list(file_table([{"mel": x}])) == [
        "file",
        "MelSpectrum_1_ch1",
        "MelSpectrum_2_ch1",
        "MelSpectrum_1_ch2",
        "MelSpectrum_2_ch2",
    ]


def test_file_table_rejects_framed_features():
    with pytest.raises(ValueError, match="frame_table"):
        file_table([{"c": framed([1, 2, 3], "Centroid")}])


def test_file_table_rejects_missing_features():
    with pytest.raises(ValueError, match="missing"):
        file_table([{"rms": scalar(0.1)}, {"zcr": scalar(5.0, name="ZCR")}])


def test_frame_table_has_frame_positions():
    table = frame_table({"c": framed([1, 2, 3], "Centroid"), "r": framed([4, 5, 6], "Rolloff")})
    assert table["frame_start"] == [0.0, 0.5, 1.0]
    last_sample = (0.5 * SR - 1) / SR  # MIRtoolbox: end = time of the last sample
    assert table["frame_end"] == pytest.approx([last_sample, 0.5 + last_sample, 1.0 + last_sample])
    assert table["Centroid"] == [1, 2, 3]


def test_frame_table_rejects_misaligned_frames():
    with pytest.raises(ValueError, match="frame positions"):
        frame_table({"c": framed([1, 2, 3], "C"), "r": framed([1, 2, 3], "R", starts=(0, 1, 2))})


def test_csv_writes_nan_and_tab_layout(tmp_path):
    path = tmp_path / "features.txt"
    to_csv({"file": ["a.wav"], "RMS": [float("nan")]}, path, sep="\t")
    assert path.read_text().splitlines() == ["file\tRMS", "a.wav\tNaN"]


def test_arff_layout(tmp_path):
    path = tmp_path / "features.arff"
    to_arff({"file": ["it's.wav"], "RMS": [0.5], "ZCR": [float("nan")]}, path)
    assert path.read_text().splitlines() == [
        "@RELATION 'mirpy'",
        "",
        "@ATTRIBUTE 'file' STRING",
        "@ATTRIBUTE 'RMS' NUMERIC",
        "@ATTRIBUTE 'ZCR' NUMERIC",
        "",
        "@DATA",
        "'it\\'s.wav',0.5,?",
    ]


def test_to_dataframe():
    pd = pytest.importorskip("pandas")
    df = to_dataframe({"file": ["a.wav"], "RMS": [0.1]})
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["file", "RMS"]
