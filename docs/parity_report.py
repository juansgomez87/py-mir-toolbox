"""Render the parity snapshot (``docs/parity/results.json``) as Markdown tables.

The snapshot is written by ``make parity-report`` from a local run of
``tests/parity`` against the MIRtoolbox fixtures. Each comparison in it is
one check of one case on one file (data, frame starts, positions...); the
page shows one row per case, which passes only if all its checks pass.
"""

import json
from dataclasses import dataclass
from pathlib import Path

UNLISTED_PHASE = "other"  # cases outside the phase*_cases.tsv lists (pymirtoolbox)
PHASE_TITLES = {1: "Phase 1: signal primitives", 2: "Phase 2: features and statistics"}
PASS_MARK, FAIL_MARK = "✓", "✗"


@dataclass(frozen=True)
class CaseRow:
    """All the checks of one case, over every file it was compared on."""

    phase: int | str
    case: str
    call: str
    files: tuple[str, ...]
    stages: tuple[str, ...]
    max_abs: float
    max_rel: float
    min_r: float
    passed: bool
    n_checks: int

    @property
    def function(self) -> str:
        """MIRtoolbox function name, e.g. ``mirspectrum``; the case name when unlisted."""
        return self.call.split("(", 1)[0] if self.call else self.case


def load_report(path) -> tuple[dict, list[dict]]:
    """Return the run description and the list of comparisons."""
    report = json.loads(Path(path).read_text())
    return report["run"], report["results"]


def case_rows(results: list[dict]) -> list[CaseRow]:
    """Group the comparisons by case, in case-list order (phase, then first appearance)."""
    groups: dict[str, list[dict]] = {}
    for result in results:
        groups.setdefault(result["case"], []).append(result)
    rows = [_case_row(case, checks) for case, checks in groups.items()]
    return sorted(rows, key=lambda row: _phase_key(row.phase))


def run_markdown(run: dict) -> str:
    """Describe the run: date, mirpy version, reference versions, audio files."""
    versions = ", ".join(run["mirtoolbox_versions"]) or "unknown"
    lines = [
        f"- **Run:** {run['date'][:10]}, mirpy {run['mirpy_version']}"
        + (f" (commit `{run['mirpy_commit']}`)" if run.get("mirpy_commit") else ""),
        f"- **Reference:** MIRtoolbox 1.8.2 (it reports itself as {versions}), "
        f"MATLAB {', '.join(run['matlab_versions']) or 'unknown'}",
        f"- **Audio:** {', '.join(run['audio']) or 'none recorded'}",
    ]
    return "\n".join(lines)


def summary_markdown(rows: list[CaseRow]) -> str:
    """Tabulate the passing cases and number of checks per phase."""
    lines = ["| Phase | Cases passing | Checks |", "|---|---|---|"]
    for phase in _phases(rows):
        in_phase = [row for row in rows if row.phase == phase]
        passing = sum(row.passed for row in in_phase)
        checks = sum(row.n_checks for row in in_phase)
        lines.append(f"| {_phase_title(phase)} | {passing} of {len(in_phase)} | {checks} |")
    return "\n".join(lines)


def phase_markdown(rows: list[CaseRow], phase) -> str:
    """Tabulate one phase, one row per case."""
    lines = [
        f"### {_phase_title(phase)}",
        "",
        "| Case | MATLAB call | Files | Tolerance | Max abs | Max rel | Min r | Pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in (row for row in rows if row.phase == phase):
        lines.append(
            f"| `{row.case}` | `{row.call or '—'}` | {', '.join(row.files)} | "
            f"{', '.join(row.stages)} | {_number(row.max_abs)} | {_number(row.max_rel)} | "
            f"{_number(row.min_r, digits=6)} | {PASS_MARK if row.passed else FAIL_MARK} |"
        )
    return "\n".join(lines)


def report_markdown(path) -> str:
    """Render the generated part of the page: run, summary, one table per phase."""
    run, results = load_report(path)
    rows = case_rows(results)
    parts = [run_markdown(run), summary_markdown(rows)]
    parts += [phase_markdown(rows, phase) for phase in _phases(rows)]
    return "\n\n".join(parts)


def _case_row(case: str, checks: list[dict]) -> CaseRow:
    first = checks[0]
    phase = first["phase"] if first["phase"] is not None else UNLISTED_PHASE
    return CaseRow(
        phase=phase,
        case=case,
        call=first["call"] or "",
        files=tuple(sorted({c["stem"] for c in checks})),
        stages=tuple(dict.fromkeys(c["stage"] for c in checks)),
        max_abs=_worst(checks, "max_abs", max),
        max_rel=_worst(checks, "max_rel", max),
        min_r=_worst(checks, "r", min),
        passed=all(c["passed"] for c in checks),
        n_checks=len(checks),
    )


def _worst(checks: list[dict], key: str, pick) -> float:
    values = [c[key] for c in checks if c[key] is not None and c[key] == c[key]]  # drop NaN
    return pick(values) if values else float("nan")


def _phases(rows: list[CaseRow]) -> list:
    return sorted({row.phase for row in rows}, key=_phase_key)


def _phase_key(phase) -> tuple[int, int]:
    return (1, 0) if phase == UNLISTED_PHASE else (0, phase)


def _phase_title(phase) -> str:
    if phase == UNLISTED_PHASE:
        return "Other cases (pymirtoolbox)"
    return PHASE_TITLES.get(phase, f"Phase {phase}")


def _number(value: float, digits: int = 2) -> str:
    if value != value:  # NaN: no finite values to compare (e.g. constant curves)
        return "—"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}" if digits > 2 else f"{value:.{digits - 1}e}"
