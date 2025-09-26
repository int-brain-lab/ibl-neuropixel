import numpy as np
import matplotlib.pyplot as plt

AP_RANGE_UV = 75
LF_RANGE_UV = 250


def show_channels_labels(
    raw,
    fs,
    channel_labels,
    xfeats,
    similarity_threshold=(-0.5, 1),
    psd_hf_threshold=0.02,
):
    """
    Shows the features side by side a snippet of raw data
    :param sr:
    :return:
    """
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = np.minimum(ns, 3000)
    fig, ax = plt.subplots(
        1, 5, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 8, 0.2]}
    )
    ax[0].plot(xfeats["xcor_hf"], np.arange(nc))
    ax[0].plot(  # plot channel below the similarity threshold as dead in black
        xfeats["xcor_hf"][(iko := channel_labels == 1)], np.arange(nc)[iko], "k*"
    )
    ax[0].plot(  # plot the values above the similarity threshold as noisy in red
        xfeats["xcor_hf"][
            (iko := np.where(xfeats["xcor_hf"] > similarity_threshold[1]))
        ],
        np.arange(nc)[iko],
        "r*",
    )
    ax[0].plot(similarity_threshold[0] * np.ones(2), [0, nc], "k--")
    ax[0].plot(similarity_threshold[1] * np.ones(2), [0, nc], "r--")
    ax[0].set(
        ylabel="channel #",
        xlabel="high coherence",
        ylim=[0, nc],
        title="a) dead channel",
    )
    ax[1].plot(xfeats["psd_hf"], np.arange(nc))
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "r*",
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "r--")
    ax[1].set(yticklabels=[], xlabel="PSD", ylim=[0, nc], title="b) noisy channel")
    ax[1].sharey(ax[0])
    ax[2].plot(xfeats["xcor_lf"], np.arange(nc))
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)], np.arange(nc)[iko], "y*"
    )
    ax[2].plot([-0.75, -0.75], [0, nc], "y--")
    ax[2].set(yticklabels=[], xlabel="LF coherence", ylim=[0, nc], title="c) outside")
    ax[2].sharey(ax[0])
    voltageshow(raw[:, :ns_plot], fs, ax=ax[3], cax=ax[4])
    ax[3].sharey(ax[0])
    fig.tight_layout()
    return fig, ax


def voltageshow(
    raw,
    fs,
    cmap="PuOr",
    ax=None,
    cax=None,
    cbar_label="Voltage (uV)",
    scaling=1e6,
    vrange=None,
    **axis_kwargs,
):
    """
    Visualizes electrophysiological voltage data as a heatmap.

    This function displays raw voltage data as a color-coded image with appropriate
    scaling based on the sampling frequency. It automatically selects voltage range
    based on whether the data is low-frequency (LF) or action potential (AP) data.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw voltage data array with shape (channels, samples), in Volts
    fs : float
        Sampling frequency in Hz, used to determine time axis scaling and voltage range.
    cmap : str, optional
        Matplotlib colormap name for the heatmap. Default is 'PuOr'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    cax : matplotlib.axes.Axes, optional
        Axes object for the colorbar. If None and ax is None, a new colorbar axes is created.
    cbar_label : str, optional
        Label for the colorbar. Default is 'Voltage (uV)'.
    vrange: float, optional
        Voltage range for the colorbar. Defaults to +/- 75 uV for AP and +/- 250 uV for LF.
    scaling: float, optional
        Unit transform: default is 1e6: we expect Volts but plot uV.
    **axis_kwargs: optional
        Additional keyword arguments for the axis properties, fed to the ax.set() method.
    Returns
    -------
    matplotlib.image.AxesImage
        The image object created by imshow, which can be used for further customization.
    """
    if ax is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
        ax, cax = axs
    nc, ns = raw.shape
    default_vrange = LF_RANGE_UV if fs < 2600 else AP_RANGE_UV
    vrange = vrange if vrange is not None else default_vrange
    im = ax.imshow(
        raw * scaling,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=-vrange,
        vmax=vrange,
        extent=[0, ns / fs, 0, nc],
    )
    # set the axis properties: we use defaults values that can be overridden by user-provided ones
    axis_kwargs = (
        dict(ylim=[0, nc], xlabel="Time (s)", ylabel="Depth (μm)") | axis_kwargs
    )
    ax.set(**axis_kwargs)
    ax.grid(False)
    if cax is not None:
        plt.colorbar(im, cax=cax, shrink=0.8).ax.set(ylabel=cbar_label)

    return im
