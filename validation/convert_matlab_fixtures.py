"""Convert MATLAB fixtures from ``mirtoolbox_fixtures.m`` to the parity-test format.

Each ``<stem>__<feature>.mat`` file becomes ``<stem>__<feature>.npz``, the
format the parity tests read (see ``tests/parity/harness.py``). Arrays stay
in MIRtoolbox's order and 1-based indexing; the tests convert them.

    python validation/convert_matlab_fixtures.py --in-dir MAT_DIR --out-dir validation/fixtures
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ARRAY_FIELDS = (
    "data",
    "pos",
    "framepos",
    "sampling",
    "peak_row",
    "peak_frame",
    "peak_channel",
    "peak_index",
    "peak_pos",
    "peak_precise_pos",
    "peak_val",
    "peak_precise_val",
)
STAT_PREFIX = "stat_"  # mirstat fields, e.g. stat_Mean
TEXT_FIELDS = (
    "feature",
    "call",
    "audio",
    "matlab_version",
    "mirtoolbox_version",
    "mirtoolbox_path",
    "generated",
)
FIXTURE_SEPARATOR = "__"


def read_mat_fixture(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return the arrays and the metadata stored in one MATLAB fixture."""
    raw = loadmat(path, squeeze_me=False, chars_as_strings=True)
    names = [*ARRAY_FIELDS, *(key for key in raw if key.startswith(STAT_PREFIX))]
    arrays = {
        name: np.asarray(raw[name], dtype=np.float64)
        for name in names
        if name in raw and np.size(raw[name])
    }
    metadata = {name: _text(raw[name]) for name in TEXT_FIELDS if name in raw}
    metadata["source"] = "matlab"
    return arrays, metadata


def convert_file(path: Path, out_dir: Path) -> Path:
    """Write ``<stem>__<feature>.npz`` for one ``.mat`` fixture; return its path."""
    arrays, metadata = read_mat_fixture(path)
    out_path = out_dir / f"{path.stem}.npz"
    np.savez_compressed(out_path, **arrays, metadata=json.dumps(metadata))
    return out_path


def _text(value) -> str:
    """MATLAB char arrays load as arrays of strings; empty ones as empty arrays."""
    value = np.asarray(value)
    return str(value.ravel()[0]) if value.size else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.in_dir.glob(f"*{FIXTURE_SEPARATOR}*.mat")):
        print(f"{path.name} -> {convert_file(path, args.out_dir).name}")


if __name__ == "__main__":
    main()
