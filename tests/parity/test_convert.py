import importlib.util
from pathlib import Path

import numpy as np
from scipy.io import savemat

from parity.harness import PHASE1_CASES, load_fixture, matlab_peaks, read_cases

CONVERTER = Path(__file__).parents[2] / "validation" / "convert_matlab_fixtures.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_matlab_fixtures", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_matlab_like_fixture(path):
    """Mimic what phase1_fixtures.m saves for a framed spectrum with peaks."""
    savemat(
        path,
        {
            "data": np.arange(12.0).reshape(3, 4),  # (element, frame); channel dropped
            "pos": np.tile([[0.0], [10.0], [20.0]], (1, 4)),
            "framepos": np.array([[0.0, 0.1, 0.2, 0.3], [0.2, 0.3, 0.4, 0.5]]),
            "sampling": 8000.0,
            "peak_row": np.ones((3, 1)),
            "peak_frame": np.array([[1.0], [1.0], [4.0]]),
            "peak_channel": np.ones((3, 1)),
            "peak_index": np.array([[2.0], [3.0], [1.0]]),
            "peak_pos": np.array([[10.0], [20.0], [0.0]]),
            "peak_precise_pos": np.array([[11.0], [19.0], [np.nan]]),
            "peak_val": np.array([[1.0], [0.5], [0.2]]),
            "peak_precise_val": np.array([[1.1], [0.6], [np.nan]]),
            "pos_unused_empty": np.zeros((0, 0)),
            "feature": "spectrum_frame",
            "call": "mirspectrum(f,'Frame')",
            "audio": "ragtime.wav",
            "matlab_version": "25.2.0 (R2025b)",
            "mirtoolbox_version": "",
            "generated": "2026-11-18T12:00:00",
        },
    )


def test_converted_fixture_loads_with_metadata_and_arrays(tmp_path):
    mat = tmp_path / "ragtime__spectrum_frame.mat"
    write_matlab_like_fixture(mat)
    out = load_converter().convert_file(mat, tmp_path)
    assert out.name == "ragtime__spectrum_frame.npz"

    fixture = load_fixture(out)
    assert fixture.metadata["call"] == "mirspectrum(f,'Frame')"
    assert fixture.metadata["audio"] == "ragtime.wav"
    assert fixture.metadata["source"] == "matlab"
    assert fixture.metadata["mirtoolbox_version"] == ""
    np.testing.assert_array_equal(fixture.arrays["data"], np.arange(12.0).reshape(3, 4))
    assert fixture.arrays["sampling"].item() == 8000.0
    assert "pos_unused_empty" not in fixture.arrays


def test_peak_table_is_zero_based(tmp_path):
    mat = tmp_path / "ragtime__spectrum_frame.mat"
    write_matlab_like_fixture(mat)
    peaks = matlab_peaks(load_fixture(load_converter().convert_file(mat, tmp_path)))
    np.testing.assert_array_equal(peaks["channel"], [0, 0, 0])
    np.testing.assert_array_equal(peaks["frame"], [0, 0, 3])
    np.testing.assert_array_equal(peaks["index"], [1, 2, 0])
    np.testing.assert_array_equal(peaks["precise_pos"], [11.0, 19.0, np.nan])


def test_case_list_parses():
    cases = read_cases(PHASE1_CASES)
    assert cases["spectrum_frame"] == "mirspectrum(f,'Frame')"
    assert len(cases) == len(set(cases))
    assert all(call.count("(") == call.count(")") for call in cases.values())
