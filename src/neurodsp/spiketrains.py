import numpy as np
from brainbox.processing import bincount2D


def spikes_venn3(
    samples_tuple,
    channels_tuple,
    samples_binsize=None,
    channels_binsize=4,
    fs=30000,
    num_channels=384,
):
    """
    Given a set of spikes found by different spike sorters over the same snippet,
    return the venn diagram counts as a dictionary suitable for the `subsets` arg of
    matplotlib_venn.venn3().

    :param samples_tuple: A tuple of N sample times of spikes (each a 1D NumPy array)
    :param channels_tuple: A tuple of N channel locations of spikes (each a 1D NumPy array)
    :param samples_binsize: Size of sample bins in number of samples. Defaults to 0.4 ms.
    :param channels_binsize: Size of channel bins in number of channels. Defaults to 4.
    :param fs: Sampling rate (Hz). Defaults to 30000.
    :param num_channels: Total number of channels where spikes appear. Defaults to 384.
    :return: dict containing venn diagram spike counts for the spike sorters.
    """
    if not samples_binsize:
        # Default: 0.4 ms
        samples_binsize = int(0.4 * fs / 1000)

    max_samples = max([np.max(samples) for samples in samples_tuple])

    bin_counts = np.array(
        [
            bincount2D(
                samples_tuple[i],
                channels_tuple[i],
                samples_binsize,
                channels_binsize,
                [0, int(max_samples)],
                [0, num_channels],
            )[0].flatten()
            for i in range(3)
        ]
    )

    cond_names = ["100", "010", "110", "001", "101", "011", "111"]
    pre_result = np.zeros(7, int)

    max_per_spike = np.amax(bin_counts, axis=0)
    overall_max = np.max(max_per_spike)

    vec = np.array([1, 2, 4])

    for i in range(0, overall_max):
        ind = max_per_spike - i > 0
        venn_info = bin_counts[:, ind] >= (max_per_spike - i)[ind]
        venn_info_int = vec @ venn_info
        conds, counts = np.unique(venn_info_int, return_counts=True)
        pre_result[conds - 1] += counts

    return dict(zip(cond_names, pre_result))
