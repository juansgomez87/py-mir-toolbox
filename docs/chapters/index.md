# Course

Chapter 0 covers the Python you need as a MIRtoolbox user. Each chapter after that introduces a musical or perceptual concept and computes it in four ways:

1. from scratch with NumPy, so the maths is visible;
2. with py-mir-toolbox, which reproduces MIRtoolbox;
3. with librosa; and
4. with Essentia.

The chapter ends by explaining where the numbers differ and why. Each chapter has an **Open in Colab** button; to run it on your computer, see {doc}`../getting_started`.

The chapter layout follows the [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html) by Meinard Müller. The references at the end of each chapter are copied from the MIRtoolbox 1.8.1 User's Manual; the numbers are compared with MIRtoolbox 1.8.2 (see the {doc}`../parity`).

```{toctree}
:maxdepth: 1

00_basics
01_dynamics
02_spectral_shape
03_mfcc_flux
04_statistics_export
```
