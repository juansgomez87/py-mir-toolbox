# py-mir-toolbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24380735.svg)](https://doi.org/10.5281/zenodo.24380735)

**A Python port of [MIRtoolbox](https://github.com/olivierlar/mirtoolbox) for music feature extraction, designed as a bridge from music psychology to the Python music information retrieval (MIR) ecosystem.**

MIRtoolbox (Lartillot & Toiviainen, 2007) is the MATLAB toolbox most music cognition and music psychology students learn first. py-mir-toolbox:

1. keeps the MIRtoolbox function names and default parameters, so existing analyses carry over;
2. is validated numerically against MIRtoolbox 1.8.2 (see the [Parity report](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/parity.html)); and
3. comes with course notebooks that compute each feature with py-mir-toolbox, [librosa](https://librosa.org) and [Essentia](https://essentia.upf.edu) side by side, runnable in Google Colab with no installation.

## Install

py-mir-toolbox needs Python 3.12 or newer.

```bash
pip install py-mir-toolbox
```

Optional extras (keep the quotes; zsh, the default macOS shell, needs them):

```bash
pip install "py-mir-toolbox[export]"    # pandas, for mir.to_dataframe
pip install "py-mir-toolbox[parallel]"  # joblib, for mir.batch(..., n_jobs=4)
pip install "py-mir-toolbox[course]"    # librosa, pandas and JupyterLab, to run the course notebooks
```

## Example

```python
import mirpy as mir

a = mir.audio("song.wav")  # a = miraudio('song.wav')
rms = mir.rms(a, frame=True)  # mirrms(a, 'Frame')
mir.stat(mir.centroid(a, frame=True))  # mirstat(mircentroid(a, 'Frame'))

# One row per WAV file of a folder (pattern="*.flac" for others), saved for R, SPSS or JASP
features = {"rms": mir.rms, "centroid": lambda a: mir.centroid(a, frame=True)}
mir.to_csv(mir.file_table(mir.batch("songs/", features)), "features.csv")
```

## What release 0.1 covers

- **Audio and framing:** `audio`, `frame`.
- **Signal processing:** `spectrum`, `cepstrum`, `autocor`, `filterbank`, `sum`, `peaks`.
- **Dynamics:** `rms`, `lowenergy`.
- **Timbre:** `zerocross`, `centroid`, `spread`, `skewness`, `kurtosis`, `flatness`, `entropy`, `rolloff`, `brightness`, `mfcc`, `flux`.
- **Statistics and export:** `mean`, `std`, `median`, `stat`, `histo`, `batch`, `list_audio`, `get_data`, `file_table`, `frame_table`, `to_csv`, `to_arff`, `to_dataframe`.

Pitch, tonality, rhythm and structure follow in later releases.

## Documentation and course

- Website: https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox
- [API reference](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/api.html) and [Parity report](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/parity.html)
- Course chapters (open in Colab from each page): [Basics](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/chapters/B_basics.html), [Dynamics](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/chapters/C3_dynamics.html), [Spectral shape](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/chapters/C4a_spectral_shape.html), [MFCCs, flux and statistics](https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/docs/chapters/C4b_mfcc_flux_statistics.html)

## Credits and citing

py-mir-toolbox is translated from [MiningSuite](https://github.com/olivierlar/miningsuite) (BSD-3-Clause, © Olivier Lartillot and the MiningSuite contributors), the successor of MIRtoolbox. MIRtoolbox 1.8.2 (GPL-2.0) is used only as a numerical reference; its source code is not included or translated. See [`NOTICE`](https://github.com/juansgomez87/py-mir-toolbox/blob/main/NOTICE).

If you use py-mir-toolbox, please cite it (see [`CITATION.cff`](https://github.com/juansgomez87/py-mir-toolbox/blob/main/CITATION.cff); metadata also in [`codemeta.json`](https://github.com/juansgomez87/py-mir-toolbox/blob/main/codemeta.json)) **and** the original MIRtoolbox papers:

- Lartillot, O., & Toiviainen, P. (2007). A Matlab toolbox for musical feature extraction from audio. *Proceedings of the 10th International Conference on Digital Audio Effects (DAFx-07)*, Bordeaux, France.
- Lartillot, O., Toiviainen, P., & Eerola, T. (2008). A Matlab toolbox for music information retrieval. In C. Preisach, H. Burkhardt, L. Schmidt-Thieme, & R. Decker (Eds.), *Data Analysis, Machine Learning and Applications*. Springer.

`mirpy.cite()` prints these entries as BibTeX.

## Repository layout

| Path | Contents |
|---|---|
| `src/mirpy/` | The package |
| `tests/` | Unit and parity tests |
| `validation/` | Scripts that generate MIRtoolbox 1.8.2 reference outputs |
| `docs/` | Sphinx documentation and course (served under `/docs/`) |
| `docs/chapters/` | Course notebooks (Colab-ready) |
| `reference/miningsuite/` | MiningSuite `+sig`, `+aud`, `+mus` (BSD-3) |

## License

Code: BSD-3-Clause. Course text: CC-BY 4.0.
