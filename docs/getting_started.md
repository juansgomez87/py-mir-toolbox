# Getting started

## Without installing anything

Every chapter of the course has an **Open in Colab** button. Google Colab runs the notebook in your browser on a free Google server. The first cell installs py-mir-toolbox, librosa and Essentia.

## On your own computer

py-mir-toolbox needs Python 3.12 or newer.

```bash
pip install py-mir-toolbox
```

Check the installation:

```python
import mirpy

print(mirpy.__version__)
```

```{note}
Keep the quotes around names with square brackets, such as `"py-mir-toolbox[course]"`. Without them, zsh (the default shell on macOS) answers `no matches found`.
```

## Running the course notebooks on your computer

1. Install py-mir-toolbox with everything the notebooks use (librosa, pandas and JupyterLab):

   ```bash
   pip install "py-mir-toolbox[course]"
   ```

2. Download a chapter: open it in the [course folder on GitHub](https://github.com/juansgomez87/py-mir-toolbox/tree/main/docs/chapters) and click *Download raw file*. (From Colab: *File → Download → Download .ipynb*.)
3. Start JupyterLab in the folder where you saved it, and open the file:

   ```bash
   jupyter lab
   ```

   JupyterLab opens in your browser. Run a cell with Shift+Enter.

### Essentia (optional)

Essentia is not part of the `course` extra, because it cannot be installed everywhere. Without it, the notebooks skip the Essentia cells and run everything else.

| Your computer | `pip install essentia` |
|---|---|
| macOS 15 or later, or Linux on Intel/AMD | works |
| macOS 13 or 14 | may fail, depending on your Python version; use Colab if it does |
| Windows | not available; use Colab, or WSL (Linux inside Windows) |
| Linux on ARM (e.g. Raspberry Pi) | not available; use Colab |

## Audio files

`mir.audio` reads WAV, FLAC, OGG and MP3 files. MP3 is a lossy format, so its values differ slightly from those of the original recording; use WAV or FLAC when you need to compare with MIRtoolbox.

`mir.batch` and `mir.list_audio` look for `*.wav` files by default. For other formats, pass a pattern, for example `mir.batch("songs/", features, pattern="*.flac")`.
