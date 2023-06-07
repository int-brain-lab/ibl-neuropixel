import numpy as np
from functools import reduce


def spiketrain_intersect(
    samples1,
    channels1,
    samples2,
    channels2,
    samples_binsize=None,
    channels_binsize=4,
    fs=30000,
    num_channels=384,
):
    """
    Given two spike trains and respective channel assignments from different spike sorters,
    identifies spikes identified by both sorters via binning the spike trains in the time
    and channel dimensions.

    :param samples1: A 1-D NumPy array containing sample times of spikes.
    :param channels1: A 1-D NumPy array containing channel assignments of spikes in `sample1`.
    Must have same length as `sample1`.
    :param samples2: A 1-D NumPy array containing sample times of another spike train.
    :param channels2: A 1-D NumPy array containing channel assignments of spikes in `sample2`.
    Must have same length as `sample2`.
    :param samples_binsize: Size of sample bins in number of samples. Defaults to 0.4 ms.
    :param channels_binsize: Size of channel bins in number of channels. Defaults to 4.
    :param fs: Sampling rate (Hz). Defaults to 30000.
    :param num_channels: Total number of channels where spikes appear. Defaults to 384.
    :return: common_spike_bins, indices1_foundby2, indices2_foundby1
    """

    assert (
        samples1.shape == channels1.shape
    ), "samples1 and channels1 must have the same shape."
    assert (
        samples2.shape == channels2.shape
    ), "samples2 and channels2 must have the same shape."

    if not samples_binsize:
        # Default: 0.4 ms
        samples_binsize = int(0.4 * fs / 1000)

    max_samples = max(np.max(samples1), np.max(samples2))
    bins2d_shape = (
        int(max_samples // samples_binsize) + 2,
        int(num_channels // channels_binsize) + 2,
    )

    # array of bin shifts in time and channel directions
    # applied to catch spikes on either side of bin borders
    shifts = np.array(
        [
            [0, 0],
            [samples_binsize // 2, channels_binsize // 2],
            [samples_binsize // 2, 0],
            [0, channels_binsize // 2],
        ]
    )

    # we'll divide by this factor and take the floor to find the sample and channel bin
    # for each spike
    factor = np.array([[samples_binsize, channels_binsize] * 4], dtype=np.float64)

    # tile data over the 4 possible shift conditions
    # What we end up with is an array that looks like:
    # [sample_bins_shift(0,0)]
    # [channel_bins_shift(0,0)]
    # [sample_bins_shift(1,0)]
    # ...
    data1 = np.tile(np.vstack((samples1, channels1)), (4, 1))
    bins1 = np.floor((data1 + shifts.flatten()[:, None]) / factor.T).astype(int)

    # store the ravelled (sample bin, channel bin) indices for each shift
    linear_indices1 = []
    for i in range(4):
        idx = slice(i * 2, (i + 1) * 2)
        linear_indices1.append(np.ravel_multi_index(bins1[idx], bins2d_shape))

    data2 = np.tile(np.vstack((samples2, channels2)), (4, 1))
    bins2 = np.floor((data2 + shifts.flatten()[:, None]) / factor.T).astype(int)

    linear_indices2 = []
    for i in range(4):
        idx = slice(i * 2, (i + 1) * 2)
        linear_indices2.append(np.ravel_multi_index(bins2[idx], bins2d_shape))

    # find the intersection of the two spike trains' bin indices for each shift
    bin_linear_indices = []
    indices1 = []
    indices2 = []
    for i in range(4):
        bin_linear_idx, idx1, idx2 = np.intersect1d(
            linear_indices1[i], linear_indices2[i], return_indices=True
        )
        bin_linear_indices.append(bin_linear_idx)
        indices1.append(idx1)
        indices2.append(idx2)

    # take the union over shifts
    common_spike_bins = list(
        zip(*np.unravel_index(reduce(np.union1d, bin_linear_indices), bins2d_shape))
    )
    indices1_foundby2 = reduce(np.union1d, indices1)
    indices2_foundby1 = reduce(np.union1d, indices2)

    return common_spike_bins, indices1_foundby2, indices2_foundby1
