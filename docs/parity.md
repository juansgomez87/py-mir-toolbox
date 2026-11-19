---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Parity report

py-mir-toolbox aims to give the same numbers as MIRtoolbox 1.8.2. Every ported function is run on the MIRtoolbox demo recordings with the options listed in `validation/phase*_cases.tsv`, and compared with the output of the same call in MATLAB, stage by stage (audio → frames → spectrum → peaks → feature), so a difference can be traced to the step where it starts.

The MATLAB outputs (fixtures) are generated locally and are not part of the repository. This page shows the last local comparison, saved in `docs/parity/results.json` by `make parity-report`.

## How to read the tables

- **One row per case**: a MATLAB call such as `mirspectrum(f,'Frame')` and its Python equivalent. A case has several checks (values, frame positions, bin positions, peak indices...); it passes only if all of them pass.
- **Tolerance**: `LINEAR` (relative error below 1e-6 of the largest value: FFT, framing, statistics), `FILTER` (1e-4: recursive filters), `DISCRETE` (within one bin: peak indices), `EXACT` (identical: peak counts, channels, frames). NaN values must sit at the same places.
- **Max abs / Max rel**: the largest absolute error, and that error divided by the largest reference value, over all checks of the case. **Min r**: the lowest correlation between the two outputs ("—" when a check has a single value).

## Results

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown

from parity_report import report_markdown

Markdown(report_markdown("parity/results.json"))
```

## Known differences

**Accepted failure.** `mirfilterbank(f,'2Channels')` differs by about 1e-4 of the signal's peak. Both use the same Butterworth filters (coefficients agree to 1e-15); the difference is floating-point round-off in the 70–1000 Hz band-pass, whose poles are very close to 0 Hz at 44.1 kHz. Changing the filter's coefficients by the smallest possible amount (one unit in the last place) already changes its output by 1.3e-4, so no implementation can match MATLAB more closely. py-mir-toolbox filters in second-order sections, the most accurate form.

**MIRtoolbox behaviour that differs from its manual or from MiningSuite.** py-mir-toolbox follows MIRtoolbox in each case. Each behaviour was found by comparing outputs, and confirmed on synthetic signals where needed:

- `mircentroid`, `mirspread`, `mirskewness`, `mirkurtosis` weight the spectrum *plus its own minimum*, not the spectrum itself. This probably explains the "slight overestimate" of the centroid reported by Kazazis et al. (2017).
- `mirkurtosis` is the plain kurtosis (3 for a Gaussian), not the excess kurtosis described in the manual.
- `mirrms` divides by the number of samples minus one on a whole file, and by the number of samples on frames.
- `mirautocor` divides by the autocorrelation of a continuous Hann window, also for the generalized (`'Compres'`) autocorrelation.
- `mirpeaks` with `'Reso'` decides which of two close peaks is higher from the values of the first frame, in every frame. Peak heights are reported in the signal's units.
- `mirstat`'s `PeriodFreq`, `PeriodAmp` and `PeriodEntropy` use a plain autocorrelation of the feature curve (no centring, no window), convert lags with the first frame step (44100/1102 Hz rather than 40 Hz for 50 ms frames with half overlap), and take the entropy of its absolute value.
- `mirflux` with `'Median'` leaves the last frame out of every median window.
- Frame end times are the time of the frame's last sample.

## Not covered yet

- Only the files listed under *Audio* above have been compared.
- The `'MinRMS'` and `'MaxEntropy'` gates of the spectral statistics never fire on these recordings, so they are not checked yet.
- Multichannel input (e.g. `'Reso'` on several channels) has no fixture.

## Updating this page

With the fixtures in `validation/fixtures/` (see `validation/README.md`):

```sh
make parity-report
```

This runs `tests/parity` and writes `docs/parity/results.json`; commit it when the results change.
