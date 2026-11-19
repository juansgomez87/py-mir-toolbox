# Coming from MATLAB MIRtoolbox

py-mir-toolbox keeps the MIRtoolbox function names without the `mir` prefix, and turns MATLAB string options into Python keyword arguments.

| MIRtoolbox (MATLAB) | py-mir-toolbox (Python) |
|---|---|
| `a = miraudio('song.wav');` | `a = mir.audio("song.wav")` |
| `mirrms(a, 'Frame')` | `mir.rms(a, frame=True)` (50 ms frames, half overlap) |
| `mirrms(a, 'Frame', 0.05, 0.5)` | `mir.rms(a, frame=(0.05, 0.5))` |
| `mirstat(r)` | `mir.stat(r)` (a dictionary: `"Mean"`, `"Std"`, `"Slope"`, ...) |
| `mirgetdata(r)` | `mir.get_data(r)` (squeezed, like `mirgetdata`) or `r.data` (always 3-D) |
| `mirexport('result.txt', r)` | `mir.to_csv(mir.file_table([{"r": r}]), "result.csv")` |
| `mirrms('Folder')` | `mir.batch("folder/", {"rms": mir.rms})` |
| Indexing starts at 1 | Indexing starts at 0 |

`mir.batch` reads `*.wav` files by default; pass `pattern="*.flac"` for other formats. To export a framed feature frame by frame, use `mir.to_csv(mir.frame_table({"r": r}), "frames.csv")`.

Each course chapter ends with a table of the MIRtoolbox functions it covers and their mirpy, librosa and Essentia equivalents: {doc}`chapters/C3_dynamics`, {doc}`chapters/C4a_spectral_shape`, {doc}`chapters/C4b_mfcc_flux_statistics`.
