"""The Parity report page's renderer (``docs/parity_report.py``), on a hand-made snapshot."""

import importlib.util
import json
from pathlib import Path

import pytest

RENDERER = Path(__file__).parents[2] / "docs" / "parity_report.py"


@pytest.fixture(scope="module")
def page():
    spec = importlib.util.spec_from_file_location("parity_report", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(case, check, passed, *, stem="ragtime", phase=1, max_rel=0.0, r=1.0):
    return {
        "feature": f"{stem}: {case} {check}".strip(), "stage": "LINEAR", "shape_ok": True,
        "nan_ok": True, "max_abs": max_rel, "max_rel": max_rel, "r": r, "agreement": 1.0,
        "passed": passed, "stem": stem, "case": case, "check": check, "phase": phase,
        "call": None if phase is None else f"mir{case}(f)",
    }  # fmt: skip


RUN = {
    "date": "2026-11-19T15:00:00+00:00", "mirpy_version": "0.1.0", "mirpy_commit": "abc1234",
    "mirtoolbox_versions": ["1.8.1"], "matlab_versions": ["R2026b"], "audio": ["ragtime.wav"],
}  # fmt: skip


@pytest.fixture
def snapshot(tmp_path):
    results = [
        result("spectrum", "", True),
        result("spectrum", "frame starts", False, max_rel=2e-3, r=0.99),
        result("spectrum", "", True, stem="Cmaj"),
        result("rms", "", True, phase=2),
        result("brightness_runtime", "", True, phase=None, r=float("nan")),
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"run": RUN, "results": results}))
    return path


def test_one_row_per_case_failing_if_any_check_fails(page, snapshot):
    _, results = page.load_report(snapshot)
    rows = page.case_rows(results)
    assert [row.case for row in rows] == ["spectrum", "rms", "brightness_runtime"]
    spectrum = rows[0]
    assert spectrum.files == ("Cmaj", "ragtime") and spectrum.n_checks == 3
    assert not spectrum.passed and spectrum.max_rel == 2e-3 and spectrum.min_r == 0.99
    assert spectrum.function == "mirspectrum"
    assert rows[2].phase == page.UNLISTED_PHASE and rows[2].function == "brightness_runtime"


def test_markdown_tables(page, snapshot):
    text = page.report_markdown(snapshot)
    assert "MIRtoolbox 1.8.2 (it reports itself as 1.8.1)" in text
    assert "| Phase 1: signal primitives | 0 of 1 | 3 |" in text
    assert (
        "| `spectrum` | `mirspectrum(f)` | Cmaj, ragtime | LINEAR | 2.0e-03 | 2.0e-03 | 0.99 | ✗ |"
        in text
    )
    assert "### Other cases (pymirtoolbox)" in text
    assert "| `brightness_runtime` | `—` |" in text and "| — | ✓ |" in text
