import importlib.util
from pathlib import Path

import numpy as np
from scipy.io import savemat

from parity.harness import PHASE1_CASES, PHASE2_CASES, load_fixture, matlab_peaks, read_cases

VALIDATION = Path(__file__).parents[2] / "validation"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, VALIDATION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_converter():
    return load_script("convert_matlab_fixtures")


def write_matlab_like_fixture(path):
    """Mimic what mirtoolbox_fixtures.m saves for a framed spectrum with peaks."""
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


def test_stat_fields_are_converted(tmp_path):
    mat = tmp_path / "ragtime__stat_rms.mat"
    savemat(mat, {"stat_Mean": 0.5, "stat_Slope": -0.1, "call": "mirstat(mirrms(f,'Frame'))"})
    fixture = load_fixture(load_converter().convert_file(mat, tmp_path))
    assert fixture.arrays["stat_Mean"].item() == 0.5
    assert fixture.arrays["stat_Slope"].item() == -0.1


def test_case_lists_do_not_share_names():
    """The MATLAB script runs all case lists into one folder."""
    phase1, phase2 = read_cases(PHASE1_CASES), read_cases(PHASE2_CASES)
    assert not set(phase1) & set(phase2)
    pymirtoolbox = set(load_script("generate_fixtures").FEATURE_CALLS)
    assert not (set(phase1) | set(phase2)) & pymirtoolbox
    assert all(call.count("(") == call.count(")") for call in phase2.values())
