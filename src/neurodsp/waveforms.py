"""
This module is to compute features on spike waveforms.
Throughout the code, the convention is to have 2D waveforms in the (time, traces)
For efficiency, several wavforms are fed in a memory contiguous manner: (iwaveform, time, traces)
"""
import numpy as np
import pandas as pd


def _validate_arr_in(arr_in):
    # expand array if 2d
    if arr_in.ndim == 2:
        arr_in = arr_in[np.newaxis, :, :]

    # Init remove nan vals in entry array
    arr_in[np.isnan(arr_in)] = 0
    return arr_in


def get_array_peak(arr_in, df):
    '''
    Create matrix of just NxT (spikes x time) of the peak waveforms channel (=1 channel)

    :param arr_in: NxTxC waveform matrix (spikes x time x channel) ; expands to 1xTxC if TxC as input
    :param df: dataframe of waveform features
    :return: NxT waveform matrix : spikes x time, only the peak channel
    '''
    arr_in = _validate_arr_in(arr_in)
    arr_peak = arr_in[np.arange(arr_in.shape[0]), :, df['peak_trace_idx'].to_numpy()]
    return arr_peak


def arr_pre_post(arr_peak, indx_peak):
    '''
    :param arr_peak: NxT waveform matrix : spikes x time, only the peak channel
    :param indx_peak: Nx1 matrix : indices of the peak for each channel
    :return:
    '''
    # Create zero mask with 1 at peak, cumsum
    arr_mask = np.zeros(arr_peak.shape)
    arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 1
    arr_mask = np.cumsum(arr_mask, axis=1)
    # arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 2  # to keep peak in both cases
    # Commented code above not needed: We want to keep peak = nan when keeping pre-values

    # Pad with Nans (as cannot slice since each waveform will have different length from peak)
    indx_prepeak = np.where(arr_mask == 0)
    indx_postpeak = np.where(arr_mask == 1)
    del arr_mask

    arr_pre = arr_peak.copy()
    arr_pre = arr_pre.astype('float')
    arr_pre[indx_postpeak] = np.nan  # Array with values pre-, nans post- peak (from peak to end)

    arr_post = arr_peak.copy()
    arr_post = arr_post.astype('float')
    arr_post[indx_prepeak] = np.nan  # Array with values post-, nans pre- peak (from start to peak-1)
    return arr_pre, arr_post


def pick_maxima(arr_in):
    """
    From one or several single or multi-trace waveforms, extract the absolute maxima for all traces
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace)
    :return: indices of time peaks, values of maxima, each of shape (nwav, ntraces)
    """
    arr_in = _validate_arr_in(arr_in)
    max_vals = np.max(np.abs(arr_in[:, :]), axis=1)
    indx_maxs = np.argmax(np.abs(arr_in[:, :]), axis=1)
    return indx_maxs, max_vals


def pick_maximum(arr_in):
    """
    From one or several single or multi-trace waveforms, extract the maximum for each wavelet.
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace)
    :return: sample index of maximum, trace index of maximum, values of maximum, length of N wav
    """
    arr_in = _validate_arr_in(arr_in)
    indx_maxs, max_vals = pick_maxima(arr_in)
    indx_trace = np.argmax(max_vals, axis=1)
    # indx_peak = indx_maxs[np.arange(0, indx_maxs.shape[0], 1), indx_trace]
    indx_peak = np.argmin(arr_in[np.arange(arr_in.shape[0]), :, indx_trace], axis=1)
    val_peak = arr_in[np.arange(0, arr_in.shape[0], 1), indx_peak, indx_trace]

    return indx_trace, indx_peak, val_peak


def peak_trough_tip(arr_in, return_peak_trace=False):
    """
    From one or several single or multi-trace waveforms, extract the times and associated
     values of the peak, through and tip of the peak channel
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace)
    :return: indices of traces and peaks, length of N wav
    """
    arr_in = _validate_arr_in(arr_in)

    # 1. Find max peak (absolute deviation in STD units)

    indx_trace, indx_peak, val_peak = pick_maximum(arr_in)
    # 2. Find trough and tip (at peak waveform)
    # Per waveform, keep only trace that contains the peak
    arr_out = arr_in[np.arange(0, arr_in.shape[0], 1), :, indx_trace]

    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_out, indx_peak)

    # Find trough
    indx_trough = np.nanargmin(arr_post * np.sign(val_peak)[:, np.newaxis], axis=1)
    val_trough = arr_out[np.arange(0, arr_out.shape[0], 1), indx_trough]
    del arr_post

    # Find tip
    y_dif1 = np.diff(arr_pre, axis=1)
    indx_posit = np.where(y_dif1 > 0)
    del arr_pre

    arr_cs = np.zeros(y_dif1.shape)
    arr_cs[indx_posit] = 1
    indx_tip = np.argmax(np.cumsum(arr_cs, axis=1), axis=1) + 1
    val_tip = arr_out[np.arange(0, arr_out.shape[0], 1), indx_tip]
    del arr_cs

    # Create dict / pd df
    d_out = pd.DataFrame()
    d_out['peak_trace_idx'] = indx_trace
    d_out['peak_time_idx'] = indx_peak
    d_out['peak_val'] = val_peak

    d_out['trough_time_idx'] = indx_trough
    d_out['trough_val'] = val_trough

    d_out['tip_time_idx'] = indx_tip
    d_out['tip_val'] = val_tip

    if return_peak_trace:
        return d_out, arr_out
    else:
        return d_out


def plot_peaktiptrough(df, arr, ax, nth_wav=0):
    ax.plot(arr[nth_wav], c='gray', alpha=0.5)
    ax.plot(arr[nth_wav][:, int(df.iloc[nth_wav].peak_trace_idx)], marker=".", c='blue')
    ax.plot(df.iloc[nth_wav].peak_time_idx, df.iloc[nth_wav].peak_val, 'r*')
    ax.plot(df.iloc[nth_wav].trough_time_idx, df.iloc[nth_wav].trough_val, 'g*')
    ax.plot(df.iloc[nth_wav].tip_time_idx, df.iloc[nth_wav].tip_val, 'k*')


def half_peak_point(arr_peak, df):
    '''
    Compute the two intersection points at halp-maximum peak
    :param: arr_in: NxT waveform matrix : spikes x time, only the peak channel
    :return: df with columns containing indices of intersection points and values, length of N wav
    '''
    # TODO Review: is df.to_numpy() necessary ?
    # Get the sign of the peak
    indx_pos = np.where(df['peak_val'].to_numpy() > 0)[0]  # todo Check [0] necessary
    # Flip positive wavs so all are negative
    if len(indx_pos) > 0:
        arr_peak[indx_pos, :] = -1 * arr_peak[indx_pos, :]
    # Compute half max value, repmat and substract it
    half_max = df['peak_val'].to_numpy() / 2
    half_max_rep = np.tile(half_max, (arr_peak.shape[1], 1)).transpose()
    # Note on the above: using np.tile because np.repeat does not work with axis=1
    # todo rewrite with np.repeat and np.newaxis
    arr_sub = arr_peak - half_max_rep
    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_sub, df['peak_time_idx'].to_numpy())
    # POST: Find first time it crosses 0 (from negative -> positive values)
    indx_post = np.argmax(arr_post > 0, axis=1)
    val_post = arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_post]
    # PRE: Find first time it crosses 0 (from positive -> negative values)
    indx_pre = np.argmax(arr_pre < 0, axis=1)
    val_pre = arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_pre]
    # Todo this algorithm does not deal if there are no points between 0 and the peak (no points found for half)

    # Add columns to DF and return
    df['half_peak_post_time_idx'] = indx_post
    df['half_peak_pre_time_idx'] = indx_pre
    df['half_peak_post_val'] = val_post
    df['half_peak_pre_val'] = val_pre

    return df


def half_peak_duration(df, fs=30000):
    '''
    Compute the half peak duration (in second)
    :param df: dataframe of waveforms features, with the half peak intersection points computed
    :param fs:  sampling rate (Hz)
    :return: dataframe wirth added column
    '''
    df['half_peak_duration'] = (df['half_peak_post_time_idx'] - df['half_peak_pre_time_idx']) / fs
    return df


def peak_to_trough(df, fs=30000):
    '''
    Compute the duration (second) and ratio of the peak-to-trough
    :param df: dataframe of waveforms features
    :param fs: sampling rate (Hz)
    :return:
    '''
    # Duration
    df['peak_to_trough_duration'] = (df['trough_time_idx'] - df['peak_time_idx']) / fs
    # Ratio
    df['peak_to_trough_ratio'] = np.log(np.abs(df['peak_val'] / df['trough_val']))
    return df


def polarisation_slopes(df, fs=30000):
    '''
    Computes the depolarisation and repolarisation slopes as the difference between tip-peak
    and peak-trough respectively.
    :param df: dataframe of waveforms features
    :param fs: sampling frequency (Hz)
    :return: dataframe with added columns
    '''
    # Depolarisation: slope before the peak (between tip and peak)
    depolarise_duration = (df['peak_time_idx'] - df['tip_time_idx']) / fs
    depolarise_volt = df['peak_val'] - df['tip_val']
    df['depolarisation_slope'] = depolarise_volt / depolarise_duration
    # Repolarisation: slope after the peak (between peak and trough)
    repolarise_duration = (df['trough_time_idx'] - df['peak_time_idx']) / fs
    repolarise_volt = df['trough_val'] - df['peak_val']
    df['repolarisation_slope'] = repolarise_volt / repolarise_duration
    return df


def recovery_point(arr_peak, df, idx_from_trough=5):
    '''
    Compute the single recovery secondary point (selected by a fixed increment
    from the trough). If the fixed increment from the trough is outside the matrix boundary, the
    last value of the waveform is used.
    :param arr_peak: NxT waveform matrix : spikes x time, only the peak channel
    :param df: dataframe of waveforms features
    :param idx_from_trough: sample index to be taken into account for the second point ; index from the trough
    :return: dataframe with added columns
    '''
    # Check range is not outside of matrix boundary)
    if idx_from_trough >= (arr_peak.shape[1]):
        raise ValueError('Index out of bound: Index larger than waveform array shape')

    # Check df['peak_time_idx'] + pt_idx is not out of bound
    idx_all = df['trough_time_idx'].to_numpy() + idx_from_trough
    # Find waveform(s) for which the second point is outside matrix boundary range
    idx_over = np.where(idx_all > arr_peak.shape[1])[0]
    if len(idx_over) > 0:
        # Todo should this raise a warning ?
        idx_all[idx_over] = arr_peak.shape[1] - 1  # Take the last value of the waveform

    df['recovery_time_idx'] = idx_all
    df['recovery_val'] = arr_peak[np.arange(0, arr_peak.shape[0], 1), idx_all]
    return df


def recovery_slope(df, fs=30000):
    '''
    Compute the recovery slope, from the trough to the single secondary point.
    :param df: dataframe of waveforms features
    :param fs: sampling frequency (Hz)
    :return: dataframe with added columns
    '''
    # Note: this could be lumped in with the polarisation_slopes
    # Time, volt and slope values
    recovery_duration = (df['recovery_time_idx'] - df['trough_time_idx']) / fs  # Diff between second point and peak
    recovery_volt = df['recovery_val'] - df['trough_val']
    df['recovery_slope'] = recovery_volt / recovery_duration
    return df


def dist_chanel_from_peak(channel_geometry, peak_trace_idx):
    '''
    Compute distance for each channel from the peak channel, for each spike
    :param channel_geometry: Matrix N(spikes) * N(channels) * 3 (spatial coordinates x,y,z)
    # Note: computing this to provide it as input will be a pain
    :param peak_trace_idx: index of the highest amplitude channel in the multi-channel waveform
    :return: eu_dist : N(spikes) * N(channels): the euclidian distance between each channel and the peak channel,
    for each waveform
    '''
    # Note: It deals with Nan in entry coordinate (fake padding channels); returns Nan as Eu dist
    # Get peak coordinates (x,y,z)
    peak_coord = channel_geometry[np.arange(0, channel_geometry.shape[0], 1), peak_trace_idx, :]

    # repmat peak coordinates (x,y,z) [Nspikes x Ncoordinates] across channels
    peak_coord_rep = np.repeat(peak_coord[:, :, np.newaxis], channel_geometry.shape[1], axis=2)  # Todo -1
    peak_coord_rep = np.swapaxes(peak_coord_rep, 1, 2)  # N spikes x channel x coordinates

    # Difference
    diff_ch = peak_coord_rep - channel_geometry
    # Square
    square_ch = np.square(diff_ch)
    # Sum
    sum_ch = np.sum(square_ch, axis=2)
    # Sqrt
    eu_dist = np.sqrt(sum_ch)
    return eu_dist


def spatial_spread_weighted(eu_dist, weights):
    '''
    Returns the spatial spread defined by the sum(w_i * dist_i) / sum(w_i).
    The weight is a given value per channel (e.g. the absolute peak voltage value)
    :param eu_dist: N(spikes) * N(channels): the euclidian distance between each channel and the peak channel,
    for each waveform
    :param weights: N(spikes) * N(channels): the weights per channel per spikes
    :return: spatial_spread : N(spikes) * 1 vector
    '''
    # Note: possible to have nan entries in eu_dist
    spatial_spread = np.nansum(np.multiply(eu_dist, weights), axis=1) / np.sum(weights, axis=1)
    return spatial_spread


def compute_spike_features(waveforms, fs=30000, recovery_duration_ms=0.16, return_peak_channel=False):
    """
    This is the main function to compute spike features from a set of waveforms
    Current features:
    Index(['peak_trace_idx', 'peak_time_idx', 'peak_val', 'trough_time_idx',
       'trough_val', 'tip_time_idx', 'tip_val', 'half_peak_post_time_idx',
       'half_peak_pre_time_idx', 'half_peak_post_val', 'half_peak_pre_val',
       'half_peak_duration', 'recovery_time_idx', 'recovery_val',
       'depolarisation_slope', 'repolarisation_slope', 'recovery_slope'],
    :param waveforms: npsikes * nsamples * nchannels 3D np.array containing multi-channel waveforms
    :param fs: sampling frequency (Hz)
    :recovery_duration_ms: in ms, the duration from the trough to the recovery point
    :return: dataframe of spikes with all features,
    Returns:
    """
    df = peak_trough_tip(waveforms)
    # Array peak
    arr_peak = get_array_peak(waveforms, df)
    # Half peak points
    df = half_peak_point(arr_peak, df)
    # Half peak duration
    df = half_peak_duration(df, fs=fs)
    # Recovery point
    df = recovery_point(arr_peak, df, idx_from_trough=int(round(recovery_duration_ms * fs / 1000)))
    # Slopes (this was not checked by eye but saved for future testing)
    df = polarisation_slopes(df, fs=fs)
    df = recovery_slope(df, fs=fs)
    if return_peak_channel:
        return df, arr_peak
    else:
        return df
