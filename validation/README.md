# Validation against MIRtoolbox 1.8.2

The parity tests compare mirpy with reference outputs ("fixtures") computed by MIRtoolbox 1.8.2 in MATLAB.

## Requirements

- MATLAB with the **Signal Processing** and **Statistics and Machine Learning** toolboxes (Octave does not work: MIRtoolbox fails to parse there).
- MIRtoolbox 1.8.2 in `reference/MIRtoolbox1.8.2/`.

## 1. Compute the fixtures in MATLAB

From the repository root:

```matlab
addpath(genpath('reference/MIRtoolbox1.8.2'))
cd validation
demos = '../reference/MIRtoolbox1.8.2/MIRToolboxDemos';

% Quick check on one file
mirtoolbox_fixtures(demos, 'mirtoolbox_mat', [], {'ragtime.wav'})

% Full run: ragtime, Cmaj, pianoA4 and trumpet (slow: tempo, pulse clarity, emotion)
mirtoolbox_fixtures(demos, 'mirtoolbox_mat')
```

This runs every call listed in `phase*_cases.tsv` and writes one `<file>__<feature>.mat` per call into `validation/mirtoolbox_mat/`. Calls that fail are skipped and listed in `mirtoolbox_mat/failures.txt`.

## 2. Convert them for Python

```sh
uv run python validation/convert_matlab_fixtures.py --in-dir validation/mirtoolbox_mat --out-dir validation/fixtures
```

## 3. Run the parity tests

```sh
MIRPY_FIXTURES=validation/fixtures uv run --group test pytest tests/parity
```

A test passes when mirpy matches MIRtoolbox within the tolerance of its stage: `rtol=1e-6` for FFT-based operators, `1e-4` for filters, and within one bin for peak positions. NaNs must match. Tests without a fixture are skipped. Every comparison (max error, correlation, pass/fail) is written to `build/parity_results.json`.

`validation/mirtoolbox_mat/` and `validation/fixtures/` are git-ignored.

## Files

| File | Purpose |
|---|---|
| `phase1_cases.tsv` … `phase7_cases.tsv` | MIRtoolbox calls, one per line: `feature <TAB> call [<TAB> output number]` |
| `mirtoolbox_fixtures.m` | Runs the calls and saves the results |
| `convert_matlab_fixtures.py` | `.mat` → `.npz` |
| `generate_fixtures.py` | Alternative without a MATLAB licence: [pymirtoolbox](https://github.com/frshdjfry/pymirtoolbox) on the free MATLAB Runtime (Linux), for the 22 functions it wraps |
