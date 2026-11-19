"""Feature extraction over many audio files.

Replaces the MIRtoolbox ``'Folder'`` / ``'Folders'`` input and ``mireval``.
MIRtoolbox evaluates a lazy design on every file; mirpy is eager, so
:func:`batch` simply calls the feature functions on each file in turn.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mirpy.core.audio_ops import audio
from mirpy.core.data import Audio, Signal
from mirpy.stats.summary import mean, std

DEFAULT_PATTERN = "*.wav"
ON_ERROR = ("raise", "skip")
SUMMARY_FIELDS = {"Mean": mean, "Std": std}  # mirexport columns for framed features

FeatureFunction = Callable[[Audio], Signal | Mapping[str, Signal]]


def list_audio(source, *, pattern: str = DEFAULT_PATTERN, recursive: bool = False) -> list[Path]:
    """List the audio files of a folder, sorted by path.

    Parameters
    ----------
    source : str, path-like, or sequence of them
        A folder, or the files themselves (kept in the given order).
    pattern : str
        Glob pattern for the files of a folder.
    recursive : bool
        Also search the sub-folders. MATLAB ``'Folders'``.

    Returns
    -------
    list of Path
    """
    if isinstance(source, (str, Path)):
        folder = Path(source)
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a folder: {folder}")
        files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
        if not files:
            raise ValueError(f"No files matching {pattern!r} in {folder}")
        return files
    return [Path(path) for path in source]


def batch(
    source,
    features: Mapping[str, FeatureFunction],
    *,
    pattern: str = DEFAULT_PATTERN,
    recursive: bool = False,
    summary: bool = True,
    audio_options: Mapping | None = None,
    n_jobs: int = 1,
    on_error: str = "raise",
) -> list[dict[str, Signal]]:
    """Compute the same features on every audio file of a folder.

    Equivalent to calling MIRtoolbox operators with ``'Folder'``.

    Parameters
    ----------
    source : str, path-like, or sequence of them
        A folder (files sorted by path), or a list of files.
    features : mapping of str to callable
        ``{name: function}``; each function takes an :class:`~mirpy.Audio`
        and returns a :class:`~mirpy.Signal`, or a dict of them such as
        :func:`mirpy.stat`, whose entries are named ``"<name>_<field>"``.
    pattern : str
        Glob pattern for the files of a folder.
    recursive : bool
        Also search the sub-folders. MATLAB ``'Folders'``.
    summary : bool
        Replace each framed result by its :func:`~mirpy.mean` and
        :func:`~mirpy.std` (``"<name>_Mean"``, ``"<name>_Std"``), as
        ``mirexport`` does, so the output fits :func:`~mirpy.file_table`.
        ``False`` keeps the frames, for :func:`~mirpy.frame_table`.
    audio_options : mapping, optional
        Keyword arguments for :func:`mirpy.audio`, e.g. ``{"sampling": 11025}``.
    n_jobs : int
        Number of files processed in parallel; needs joblib
        (``pip install 'py-mir-toolbox[parallel]'``). The output keeps the
        file order.
    on_error : {"raise", "skip"}
        What to do with a file that cannot be read or analysed: stop, or
        leave it out with a warning.

    Returns
    -------
    list of dict
        One ``{name: Signal}`` mapping per file, in file order.

    Examples
    --------
    >>> import mirpy as mir
    >>> features = {"rms": mir.rms, "centroid": lambda a: mir.centroid(a, frame=True)}
    >>> rows = mir.batch("songs/", features)  # doctest: +SKIP
    >>> mir.to_csv(mir.file_table(rows), "features.csv")  # doctest: +SKIP
    """
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}, not {on_error!r}")
    files = list_audio(source, pattern=pattern, recursive=recursive)
    options = dict(audio_options or {})
    jobs = [(path, features, summary, options) for path in files]
    outcomes = _run(jobs, n_jobs)
    results = []
    for path, (result, error) in zip(files, outcomes, strict=True):
        if error is None:
            results.append(result)
        elif on_error == "skip":
            warnings.warn(f"Skipped {path}: {error!r}", stacklevel=2)
        else:
            raise RuntimeError(f"Feature extraction failed on {path}") from error
    return results


def _run(jobs: Sequence[tuple], n_jobs: int) -> list[tuple]:
    if n_jobs == 1:
        return [_safe_extract(*job) for job in jobs]
    try:
        from joblib import Parallel, delayed
    except ImportError as error:
        raise ImportError(
            "batch with n_jobs > 1 needs joblib: pip install 'py-mir-toolbox[parallel]'"
        ) from error
    return Parallel(n_jobs=n_jobs)(delayed(_safe_extract)(*job) for job in jobs)


def _safe_extract(path, features, summary, options) -> tuple[dict | None, Exception | None]:
    """Return ``(result, None)``, or ``(None, error)`` so one file cannot stop the others."""
    try:
        return _extract(path, features, summary, options), None
    except Exception as error:  # noqa: BLE001 - reported per file by batch()
        return None, error


def _extract(path, features, summary, options) -> dict[str, Signal]:
    a = audio(path, **options)
    result: dict[str, Signal] = {}
    for name, function in features.items():
        for key, x in _named_outputs(name, function(a)).items():
            result.update(_summarise(key, x) if summary and x.is_framed else {key: x})
    return result


def _named_outputs(name: str, output) -> dict[str, Signal]:
    if isinstance(output, Mapping):
        return {f"{name}_{field}": x for field, x in output.items()}
    return {name: output}


def _summarise(name: str, x: Signal) -> dict[str, Signal]:
    return {f"{name}_{field}": reduce(x) for field, reduce in SUMMARY_FIELDS.items()}
