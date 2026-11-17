# Coming from MATLAB MIRtoolbox

py-mir-toolbox keeps the MIRtoolbox function names without the `mir` prefix, and turns MATLAB string options into Python keyword arguments.

| MIRtoolbox (MATLAB) | py-mir-toolbox (Python) |
|---|---|
| `a = miraudio('song.wav');` | `a = mir.audio("song.wav")` |
| `mirrms(a, 'Frame', 0.05, 0.5)` | `mir.rms(a, frame=(0.05, 0.5))` |
| `mirgetdata(r)` | `r.data` |
| `mirexport('result.txt', r)` | `mir.export("result.csv", r)` |
| Indexing starts at 1 | Indexing starts at 0 |

A complete table, including the librosa and Essentia equivalents of every function, will be added as features are ported.
