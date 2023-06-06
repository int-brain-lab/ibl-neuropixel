import numpy as np

def spiketrain_intersect(samples1, channels1, samples2, channels2, samples_binsize=None, 
                         channels_binsize=4, FS=30000, num_channels=384):
    """
    
    """

    max_samples = max(np.max(samples1), np.max(samples2))
    bins2d_shape = (int(num_channels // channels_binsize), int(max_samples // channels_binsize))

    if not samples_binsize:
        # SpikeInterface default: 0.4 ms
        samples_binsize = int(0.4 * FS / 1000)

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

