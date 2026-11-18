"""Plotting and playback (replaces MiningSuite ``@Signal/display.m``).

Only the main display cases are covered: waveform or curve over time,
curve over frames, and an image (time x element) for framed multi-element
data such as spectrograms. matplotlib and IPython are imported only
when used.
"""

import numpy as np

from mirpy.core.data import Signal


def plot(x: Signal, ax=None):
    """Plot a signal, one subplot per channel.

    Parameters
    ----------
    x : Signal
        Signal to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; only for single-channel signals. A new figure
        is created when omitted.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        One axes per channel.
    """
    axes = _make_axes(x.n_channels, ax)
    for channel, channel_ax in enumerate(axes):
        _plot_channel(x, channel, channel_ax)
        if x.channel_labels is not None:
            channel_ax.set_title(x.channel_labels[channel], loc="right", fontsize="small")
    axes[0].set_title(x.name)
    axes[-1].set_xlabel("time (s)" if x.is_framed or x.x_values is None else _x_label(x))
    return axes


def play(x: Signal):
    """Return an IPython audio player for an unframed signal.

    The waveform is normalised to its peak for playback.

    Returns
    -------
    IPython.display.Audio
    """
    try:
        from IPython.display import Audio as AudioPlayer
    except ImportError as error:
        raise ImportError("play() needs IPython; run it in Jupyter or Colab") from error
    if x.is_framed:
        raise ValueError("play() needs an unframed signal")
    return AudioPlayer(data=x.data[:, :, 0], rate=x.sr)


def _make_axes(n_channels: int, ax):
    if ax is not None:
        if n_channels > 1:
            raise ValueError("ax can only be given for single-channel signals")
        return np.array([ax])
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(n_channels, 1, sharex=True, squeeze=False)
    return axes[:, 0]


def _plot_channel(x: Signal, channel: int, ax) -> None:
    values = x.data[channel]
    if not x.is_framed:
        positions = x.times if x.x_values is None else x.x_values
        ax.plot(positions, values[:, 0])
        ax.set_ylabel(_y_label(x))
    elif x.n_elements == 1:
        ax.plot(x.frame_centers, values[0])
        ax.set_ylabel(_y_label(x))
    else:
        elements = np.arange(x.n_elements) if x.x_values is None else x.x_values
        ax.pcolormesh(x.frame_centers, elements, values, shading="nearest")
        ax.set_ylabel(_x_label(x))


def _y_label(x: Signal) -> str:
    return f"{x.name} ({x.unit})" if x.unit else x.name


def _x_label(x: Signal) -> str:
    name = x.x_name or "element"
    return f"{name} ({x.x_unit})" if x.x_unit else name
