"""Compare mirpy outputs with MIRtoolbox 1.8.2 reference fixtures.

Fixtures are ``<stem>__<feature>.npz`` files written by
``validation/generate_fixtures.py``. Their directory is ``$MIRPY_FIXTURES``,
or ``tests/data`` (the future ``py-mir-toolbox-test-data`` submodule).
"""

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np

FIXTURES_ENV = "MIRPY_FIXTURES"
DEFAULT_FIXTURES_DIR = Path(__file__).parents[1] / "data"
REPORT_ENV = "MIRPY_PARITY_REPORT"
DEFAULT_REPORT_PATH = Path(__file__).parents[2] / "build" / "parity_results.json"
MATLAB_AXES = 3  # MIRtoolbox data are (element, frame, channel)


class Stage(Enum):
    """Tolerance per processing stage (plan.md §4): (rtol, atol in bins)."""

    LINEAR = (1e-6, 0)  # framing, FFT spectrum, autocorrelation, RMS
    FILTER = (1e-4, 0)  # IIR filters, filterbanks, envelopes
    DISCRETE = (0, 1)  # peak indices, key, tempo bins: within one bin
    EXACT = (0, 0)

    @property
    def rtol(self) -> float:
        return self.value[0]

    @property
    def atol(self) -> float:
        return self.value[1]


@dataclass(frozen=True)
class ParityResult:
    """Agreement between a mirpy output and its MIRtoolbox reference."""

    feature: str
    stage: str
    shape_ok: bool
    nan_ok: bool
    max_abs: float
    max_rel: float
    r: float
    agreement: float
    passed: bool


@dataclass(frozen=True)
class Fixture:
    """Arrays and metadata of one reference fixture, as saved (MATLAB order)."""

    arrays: dict[str, np.ndarray]
    metadata: dict


def fixtures_dir() -> Path:
    return Path(os.environ.get(FIXTURES_ENV, DEFAULT_FIXTURES_DIR))


def report_path() -> Path:
    return Path(os.environ.get(REPORT_ENV, DEFAULT_REPORT_PATH))


def load_fixture(path: Path) -> Fixture:
    with np.load(path) as npz:
        arrays = {key: npz[key] for key in npz.files if key != "metadata"}
        metadata = json.loads(str(npz["metadata"])) if "metadata" in npz.files else {}
    return Fixture(arrays, metadata)


def from_matlab_axes(values) -> np.ndarray:
    """Convert MIRtoolbox ``(element, frame, channel)`` data to mirpy order.

    MATLAB drops trailing singleton dimensions, so missing axes are
    restored as size 1 before reordering to ``(channel, element, frame)``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim > MATLAB_AXES:
        raise ValueError(f"Expected at most {MATLAB_AXES} axes, got shape {values.shape}")
    values = values.reshape(values.shape + (1,) * (MATLAB_AXES - values.ndim))
    return values.transpose(2, 0, 1)


def compare(actual, expected, stage: Stage, feature: str) -> ParityResult:
    """Measure agreement; NaNs must sit at the same positions in both."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        nan = float("nan")
        return ParityResult(feature, stage.name, False, False, nan, nan, nan, 0.0, False)

    nan_ok = bool(np.array_equal(np.isnan(actual), np.isnan(expected)))
    finite = np.isfinite(actual) & np.isfinite(expected)
    diff = np.abs(actual[finite] - expected[finite])
    scale = np.max(np.abs(expected[finite]), initial=0.0)
    within = np.isclose(
        actual, expected, rtol=0, atol=stage.atol + stage.rtol * scale, equal_nan=True
    )
    return ParityResult(
        feature=feature,
        stage=stage.name,
        shape_ok=True,
        nan_ok=nan_ok,
        max_abs=float(np.max(diff, initial=0.0)),
        max_rel=float(np.max(diff, initial=0.0) / scale) if scale else 0.0,
        r=_correlation(actual[finite], expected[finite]),
        agreement=float(np.mean(within)) if within.size else 1.0,
        passed=nan_ok and bool(np.all(within)),
    )


def failure_message(result: ParityResult) -> str:
    if not result.shape_ok:
        return f"{result.feature}: shape mismatch"
    return (
        f"{result.feature} ({result.stage}): max_abs={result.max_abs:.3g}, "
        f"max_rel={result.max_rel:.3g}, r={result.r:.6f}, "
        f"agreement={result.agreement:.1%}, NaNs match={result.nan_ok}"
    )


def write_report(results: list[ParityResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
