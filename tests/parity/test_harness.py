import json

import numpy as np
import pytest

from parity.harness import Stage, compare, from_matlab_axes, load_fixture, write_report


def test_from_matlab_axes_restores_trailing_singletons():
    framed_scalar = np.arange(5.0).reshape(1, 5)  # MATLAB (element=1, frame=5)
    assert from_matlab_axes(framed_scalar).shape == (1, 1, 5)
    spectrum = np.zeros((513, 10, 2))
    assert from_matlab_axes(spectrum).shape == (2, 513, 10)


def test_from_matlab_axes_keeps_values_in_place():
    values = np.arange(24.0).reshape(4, 3, 2)  # (element, frame, channel)
    converted = from_matlab_axes(values)
    assert converted[1, 2, 0] == values[2, 0, 1]


def test_linear_stage_scales_with_signal_magnitude():
    expected = np.array([0.0, 1000.0, -500.0])
    assert compare(expected + 1e-4, expected, Stage.LINEAR, "x").passed
    assert not compare(expected + 1e-2, expected, Stage.LINEAR, "x").passed


def test_discrete_stage_allows_one_bin():
    expected = np.array([10, 20, 30])
    result = compare(expected + [1, -1, 0], expected, Stage.DISCRETE, "peaks")
    assert result.passed
    result = compare(expected + [2, 0, 0], expected, Stage.DISCRETE, "peaks")
    assert not result.passed
    assert result.agreement == pytest.approx(2 / 3)


def test_nans_must_match():
    expected = np.array([1.0, np.nan, 3.0])
    assert compare(expected.copy(), expected, Stage.EXACT, "x").passed
    result = compare(np.array([1.0, 2.0, 3.0]), expected, Stage.LINEAR, "x")
    assert not result.nan_ok
    assert not result.passed


def test_shape_mismatch_fails():
    assert not compare(np.zeros(3), np.zeros(4), Stage.LINEAR, "x").shape_ok


def test_fixture_and_report_round_trip(tmp_path):
    path = tmp_path / "sine__rms.npz"
    np.savez(path, data=np.ones((1, 3)), metadata=json.dumps({"function": "mirrms"}))
    fixture = load_fixture(path)
    assert fixture.metadata == {"function": "mirrms"}
    np.testing.assert_array_equal(fixture.arrays["data"], np.ones((1, 3)))

    report = tmp_path / "out" / "parity.json"
    write_report([compare(np.ones(3), np.ones(3), Stage.LINEAR, "rms")], report)
    assert json.loads(report.read_text())[0]["passed"] is True
