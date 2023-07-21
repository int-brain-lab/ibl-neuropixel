import numpy as np
from iblutil.numerical import bincount2D
import tqdm
import logging

_logger = logging.getLogger("ibllib")
_logger.setLevel("INFO")


def spikes_venn3(
    samples_tuple,
    channels_tuple,
    samples_binsize=None,
    channels_binsize=4,
    fs=30000,
    num_channels=384,
    chunk_size=None,
):
    """
    Given a set of spikes found by different spike sorters over the same snippet,
    return the venn diagram counts as a dictionary suitable for the `subsets` arg of
    `matplotlib_venn.venn3()`. "100" represents the number of spikes found by sorter 1
    but not the other two, "110" represents the number of spikes found by sorters 1 and 2
    but not 3, etc. The algorithm works by binning in the time and channel dimensions and
    counting spikes found by different sorters within the same bins.

    :param samples_tuple: A tuple of 3 sample times of spikes (each a 1D NumPy array).
    :param channels_tuple: A tuple of 3 channel locations of spikes (each a 1D NumPy array).
    :param samples_binsize: Size of sample bins in number of samples. Defaults to 0.4 ms.
    :param channels_binsize: Size of channel bins in number of channels. Defaults to 4.
    :param fs: Sampling rate (Hz). Defaults to 30000.
    :param num_channels: Total number of channels where spikes appear. Defaults to 384.
    :param chunk_size: Chunk size to process spike data (in samples). Defaults to 20 seconds.
    :return: dict containing venn diagram spike counts for the spike sorters.
    """
    assert len(samples_tuple) == 3, "Must have 3 sets of samples."
    assert len(channels_tuple) == 3, "Must have 3 sets of channels."
    assert all(
        samples_tuple[i].shape == channels_tuple[i].shape for i in range(3)
    ), "Samples and channels must match for each sorter."

    if not samples_binsize:
        # set default: 0.4 ms
        samples_binsize = int(0.4 * fs / 1000)

    if not chunk_size:
        # set default: 20 s
        chunk_size = 20 * fs

    # find the timestamp of the last spike detected by any of the sorters
    # to calibrate chunking
    max_samples = max([np.max(samples) for samples in samples_tuple])
    num_chunks = int((max_samples // chunk_size) + 1)

    # each spike falls into one of 7 conditions based on whether it was found
    # by different sortings
    cond_names = ["100", "010", "110", "001", "101", "011", "111"]
    pre_result = np.zeros(7, int)
    vec = np.array([1, 2, 4])

    _logger.info(f"Running spike venning routine with {num_chunks} chunks.")
    for ch in tqdm.tqdm(range(num_chunks)):
        # select spikes within this chunk's time snippet
        sample_offset = ch * chunk_size
        spike_indices = [
            slice(
                *np.searchsorted(samples, [sample_offset, sample_offset + chunk_size])
            )
            for samples in samples_tuple
        ]
        # get corresponding spike sample times and channels
        samples_chunks = [
            samples[spike_indices[i]].astype(int) - sample_offset
            for i, samples in enumerate(samples_tuple)
        ]
        channels_chunks = [
            channels[spike_indices[i]].astype(int)
            for i, channels in enumerate(channels_tuple)
        ]

        # compute fast 2D bin count for each sorter, resulting in an (3, num_bins)
        # array where the (i, j) number is the number of spikes found by sorter i
        # in (linearized) bin j.
        bin_counts = np.array(
            [
                bincount2D(
                    samples_chunks[i],
                    channels_chunks[i],
                    samples_binsize,
                    channels_binsize,
                    [0, chunk_size],
                    [0, num_channels],
                )[0].flatten()
                for i in range(3)
            ]
        )

        # this process iteratively counts the number of spikes falling into each
        # of the 7 conditions by separating out which spikes must have been found
        # by each spike sorter within each bin, and updates the master `pre_result`
        # count array for this chunk
        max_per_spike = np.amax(bin_counts, axis=0)
        overall_max = np.max(max_per_spike)

        for i in range(0, overall_max):
            ind = max_per_spike - i > 0
            venn_info = bin_counts[:, ind] >= (max_per_spike - i)[ind]
            venn_info_int = vec @ venn_info
            conds, counts = np.unique(venn_info_int, return_counts=True)
            pre_result[conds - 1] += counts

    return dict(zip(cond_names, pre_result))
