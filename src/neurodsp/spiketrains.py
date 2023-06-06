import numpy as np

def spiketrain_intersect(samples1, channels1, samples2, channels2, samples_binsize=None, 
                         channels_binsize=4, fs=30000, num_channels=384):
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
    :return: A list of tuples containing the bin assignments of the spikes found in common. 
    """

    max_samples = max(np.max(samples1), np.max(samples2))
    bins2d_shape = (int(num_channels // channels_binsize), int(max_samples // channels_binsize))

    if not samples_binsize:
        # SpikeInterface default: 0.4 ms
        samples_binsize = int(0.4 * fs / 1000)

    # Get sample bins
    sample_bins1 = np.floor(samples1 / samples_binsize).astype(np.int64)
    sample_bins2 = np.floor(samples2 / samples_binsize).astype(np.int64)

    # Get channel bins
    channel_bins1 = np.floor(channels1 / channels_binsize).astype(np.int64)
    channel_bins2 = np.floor(channels2 / channels_binsize).astype(np.int64)
    # get linear indices
    linear_ind1 = np.ravel_multi_index(
        np.array([channel_bins1, sample_bins1]), 
        bins2d_shape,
    )
    linear_ind2 = np.ravel_multi_index(
        np.array([channel_bins2, sample_bins2]), 
        bins2d_shape,
    )

    common_spikes_linearized = np.intersect1d(linear_ind1, linear_ind2).astype(int)
    common_spikes = np.unravel_index(
        common_spikes_linearized, 
        bins2d_shape
    )

    return list(zip(*common_spikes))

