# py-mir-toolbox

A Python port of [MIRtoolbox](https://github.com/olivierlar/mirtoolbox) for music feature extraction, written for music cognition and music psychology students who want to move into the Python music information retrieval (MIR) ecosystem.

```python
import mirpy as mir

a = mir.audio("song.wav")
mir.rms(a, frame=(0.05, 0.5))
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc
Install the package, or open the course in Google Colab with nothing to install.
:::

:::{grid-item-card} Coming from MATLAB
:link: coming_from_matlab
:link-type: doc
MIRtoolbox commands and their Python equivalents, side by side.
:::

:::{grid-item-card} Course
:link: chapters/index
:link-type: doc
Chapters in the style of the FMP notebooks. Each concept is computed from scratch in NumPy, then with mirpy, librosa and Essentia.
:::

:::{grid-item-card} Parity report
:link: parity
:link-type: doc
How closely each function matches MIRtoolbox 1.8.2, and the known differences.
:::

:::{grid-item-card} Credits & citing
:link: credits
:link-type: doc
How to cite MIRtoolbox, MiningSuite and py-mir-toolbox.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
coming_from_matlab
chapters/index
api
parity
credits
```
