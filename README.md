# py-mir-toolbox

**A Python port of [MIRtoolbox](https://github.com/olivierlar/mirtoolbox) for music feature extraction, designed as a bridge from music psychology to the Python music information retrieval (MIR) ecosystem.**

MIRtoolbox (Lartillot & Toiviainen, 2007) is the MATLAB toolbox most music cognition and music psychology students learn first. py-mir-toolbox:

1. keeps the MIRtoolbox function names and default parameters, so existing analyses carry over;
2. is validated numerically against MIRtoolbox 1.8.2 (see the parity report in the documentation); and
3. comes with course notebooks that compute each feature with py-mir-toolbox, [librosa](https://librosa.org) and [Essentia](https://essentia.upf.edu) side by side, runnable in Google Colab with no installation.

## Install

```bash
pip install py-mir-toolbox
```

```python
import mirpy
```

## Documentation and course

https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox

## Credits and citing

py-mir-toolbox is translated from [MiningSuite](https://github.com/olivierlar/miningsuite) (BSD-3-Clause, © Olivier Lartillot and the MiningSuite contributors), the successor of MIRtoolbox. MIRtoolbox 1.8.2 (GPL-2.0) is used only as a numerical reference; its source code is not included or translated. See [`NOTICE`](NOTICE).

If you use py-mir-toolbox, please cite it (see [`CITATION.cff`](CITATION.cff)) **and** the original MIRtoolbox papers:

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
| `reference/miningsuite/` | MiningSuite `+sig`, `+aud`, `+mus` (BSD-3), the translation source |

## License

Code: BSD-3-Clause. Course text: CC-BY 4.0.
