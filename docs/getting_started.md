# Getting started

## Without installing anything

Every chapter of the course has an **Open in Colab** button. Google Colab runs the notebook in your browser on a free Google server. The first cell installs py-mir-toolbox, librosa and Essentia.

## On your own computer

py-mir-toolbox needs Python 3.12 or newer.

```bash
pip install py-mir-toolbox
```

To follow the comparisons in the course, also install librosa:

```bash
pip install "py-mir-toolbox[librosa]"
```

Check the installation:

```python
import mirpy

print(mirpy.__version__)
```

```{note}
WAV, FLAC and OGG files are read directly. Convert MP3 or M4A files to WAV first, for example with Audacity.
```
