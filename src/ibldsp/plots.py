import numpy as np
import matplotlib.pyplot as plt


def show_channels_labels(
    raw, fs, channel_labels, xfeats, similarity_threshold, psd_hf_threshold=0.02
):
    """
    Shows the features side by side a snippet of raw data
    :param sr:
    :return:
    """
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = np.minimum(ns, 3000)
    vaxis_uv = 250 if fs < 2600 else 75
    fig, ax = plt.subplots(
        1, 5, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 8, 0.2]}
    )
    ax[0].plot(xfeats["xcor_hf"], np.arange(nc))
    ax[0].plot(
        xfeats["xcor_hf"][(iko := channel_labels == 1)], np.arange(nc)[iko], "k*"
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
    ax[1].plot(xfeats["psd_hf"][(iko := channel_labels == 2)], np.arange(nc)[iko], "r*")
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
    im = ax[3].imshow(
        raw[:, :ns_plot] * 1e6,
        origin="lower",
        cmap="PuOr",
        aspect="auto",
        vmin=-vaxis_uv,
        vmax=vaxis_uv,
        extent=[0, ns_plot / fs * 1e3, 0, nc],
    )
    ax[3].set(yticklabels=[], title="d) Raw data", xlabel="time (ms)", ylim=[0, nc])
    ax[3].grid(False)
    ax[3].sharey(ax[0])
    plt.colorbar(im, cax=ax[4], shrink=0.8).ax.set(ylabel="(uV)")
    fig.tight_layout()
    return fig, ax
