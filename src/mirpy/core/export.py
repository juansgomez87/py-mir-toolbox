"""Export features as tables (replaces MIRtoolbox ``mirexport``).

Tables are plain column dictionaries, ``{column_name: list_of_values}``,
so CSV and ARFF export need no extra dependency. :func:`to_dataframe`
converts a table to pandas (optional ``export`` extra).

Column names follow ``mirexport``: spaces are removed from the feature
name, and multi-element features get a 1-based element index, e.g.
``Mel-Spectrum_1``. Multichannel features get a 1-based channel index,
e.g. ``RMS_ch2``. Summary statistics columns (``_Mean``, ``_Std``) arrive
with ``mirstat`` in a later release.
"""

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from mirpy.core.data import Signal

FeatureSet = Mapping[str, Signal]
Table = dict[str, list]

FILE_COLUMN = "file"
FRAME_START_COLUMN = "frame_start"
FRAME_END_COLUMN = "frame_end"
ARFF_MISSING = "?"
CSV_MISSING = "NaN"


def file_table(files: Sequence[FeatureSet]) -> Table:
    """Build a table with one row per audio file.

    Parameters
    ----------
    files : sequence of mapping
        One ``{feature_name: Signal}`` mapping per file. All mappings must
        have the same features, and every feature must be unframed.

    Returns
    -------
    dict
        Column name to list of values. The first column, ``"file"``, holds
        each file's name (from ``Signal.source``) or its index.

    Examples
    --------
    >>> import numpy as np
    >>> from mirpy.core.data import Signal
    >>> rms = Signal(data=np.full((1, 1, 1), 0.1), sr=8000, name="RMS")
    >>> file_table([{"rms": rms}])
    {'file': ['0'], 'RMS': [0.1]}
    """
    table: Table = {FILE_COLUMN: []}
    for index, features in enumerate(files):
        table[FILE_COLUMN].append(_file_label(features, index))
        for name, x in features.items():
            if x.is_framed:
                raise ValueError(f"Feature {name!r} is framed; export it with frame_table")
            for column, values in _feature_columns(x).items():
                table.setdefault(column, []).append(float(values[0]))
    _check_rectangular(table)
    return table


def frame_table(features: FeatureSet) -> Table:
    """Build a table with one row per frame, for one audio file.

    Parameters
    ----------
    features : mapping
        ``{feature_name: Signal}``; all features must be framed with the
        same frame positions.

    Returns
    -------
    dict
        Column name to list of values, starting with the frame start and
        end times in seconds.
    """
    reference = next(iter(features.values()))
    if not all(x.is_framed for x in features.values()):
        raise ValueError("All features must be framed; export unframed ones with file_table")
    for name, x in features.items():
        same_frames = x.n_frames == reference.n_frames and np.allclose(
            x.frame_starts, reference.frame_starts
        )
        if not same_frames:
            raise ValueError(f"Feature {name!r} has different frame positions")
    table: Table = {
        FRAME_START_COLUMN: reference.frame_starts.tolist(),
        FRAME_END_COLUMN: reference.frame_ends.tolist(),
    }
    for x in features.values():
        for column, values in _feature_columns(x).items():
            table[column] = values.tolist()
    return table


def to_dataframe(table: Table):
    """Convert a table to a :class:`pandas.DataFrame`.

    Requires pandas (``pip install py-mir-toolbox[export]``).
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "to_dataframe needs pandas: pip install 'py-mir-toolbox[export]'"
        ) from error
    return pd.DataFrame(table)


def to_csv(table: Table, path, *, sep: str = ",") -> None:
    """Write a table as CSV.

    Parameters
    ----------
    table : dict
        Output of :func:`file_table` or :func:`frame_table`.
    path : str or path-like
        Output file.
    sep : str
        Field separator. Use a tab for the layout of MIRtoolbox
        ``mirexport`` text files.
    """
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(table)
        for row in _rows(table):
            writer.writerow(CSV_MISSING if _is_missing(v) else v for v in row)


def to_arff(table: Table, path, *, relation: str = "mirpy") -> None:
    """Write a table as a Weka ARFF file.

    Parameters
    ----------
    table : dict
        Output of :func:`file_table` or :func:`frame_table`.
    path : str or path-like
        Output file.
    relation : str
        ARFF relation name.
    """
    lines = [f"@RELATION {_arff_quote(relation)}", ""]
    for column, values in table.items():
        kind = "STRING" if isinstance(values[0], str) else "NUMERIC"
        lines.append(f"@ATTRIBUTE {_arff_quote(column)} {kind}")
    lines += ["", "@DATA"]
    for row in _rows(table):
        lines.append(",".join(_arff_value(v) for v in row))
    Path(path).write_text("\n".join(lines) + "\n")


def _feature_columns(x: Signal) -> dict[str, np.ndarray]:
    """Return one array of per-frame values for each channel and element."""
    base = x.name.replace(" ", "")
    columns = {}
    for channel in range(x.n_channels):
        channel_suffix = f"_ch{channel + 1}" if x.n_channels > 1 else ""
        for element in range(x.n_elements):
            element_suffix = f"_{element + 1}" if x.n_elements > 1 else ""
            columns[base + element_suffix + channel_suffix] = x.data[channel, element, :]
    return columns


def _file_label(features: FeatureSet, index: int) -> str:
    sources = {x.source for x in features.values() if x.source}
    return Path(sources.pop()).name if len(sources) == 1 else str(index)


def _check_rectangular(table: Table) -> None:
    n_rows = len(table[FILE_COLUMN])
    uneven = [column for column, values in table.items() if len(values) != n_rows]
    if uneven:
        raise ValueError(f"Features missing from some files: {uneven}")


def _rows(table: Table):
    return zip(*table.values(), strict=True)


def _is_missing(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _arff_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _arff_value(value) -> str:
    if isinstance(value, str):
        return _arff_quote(value)
    return ARFF_MISSING if _is_missing(value) else repr(value)
