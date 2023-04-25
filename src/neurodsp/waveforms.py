"""
This module is to compute features on spike waveforms.
Throughout the code, the convention is to have 2D waveforms in the (time, traces)
For efficiency, several wavforms are fed in a memory contiguous manner: (iwaveform, time, traces)
"""
import numpy as np
import pandas as pd


def arr_pre_post(arr_in, indx_peak):
    '''
    :param arr_in: NxC waveform matrix : spikes x channel, only the peak channel
    :param indx_peak: Nx1 matrix : indices of the peak for each channel
    :return:
    '''
    # Create zero mask with 1 at peak, cumsum
    arr_mask = np.zeros(arr_in.shape)
    arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 1
    arr_mask = np.cumsum(arr_mask, axis=1)
    # arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 2  # to keep peak in both cases
    # Commented code above not needed: We want to keep peak = nan when keeping pre-values

    # Pad with Nans (as cannot slice since each waveform will have different length from peak)
    indx_prepeak = np.where(arr_mask == 0)
    indx_postpeak = np.where(arr_mask == 1)
    del arr_mask

    arr_pre = arr_in.copy()
    arr_pre = arr_pre.astype('float')
    arr_pre[indx_postpeak] = np.nan  # Array with values pre-, nans post- peak (from peak to end)

    arr_post = arr_in.copy()
    arr_post = arr_post.astype('float')
    arr_post[indx_prepeak] = np.nan  # Array with values post-, nans pre- peak (from start to peak-1)
    return arr_pre, arr_post

def _validate_arr_in(arr_in):
    # expand array if 2d
    if arr_in.ndim == 2:
        arr_in = arr_in[np.newaxis, :, :]

    # Init remove nan vals in entry array
    arr_in[np.isnan(arr_in)] = 0
    return arr_in


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
    ax.plot(arr[nth_wav][:, int(df.iloc[nth_wav].peak_trace_idx)], c='blue')
    ax.plot(df.iloc[nth_wav].peak_time_idx, df.iloc[nth_wav].peak_val, 'r*')
    ax.plot(df.iloc[nth_wav].trough_time_idx, df.iloc[nth_wav].trough_val, 'g*')
    ax.plot(df.iloc[nth_wav].tip_time_idx, df.iloc[nth_wav].tip_val, 'k*')


def half_peak(arr_in, df=None):
    '''
    Compute the two intersection points at halp-maximum peak
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace) = (spikes, time, channels)
    :return: indices of intersection points and values, length of N wav
    '''
    # Recompute peak-tip-trough if not passed in
    if df is None:  # Recompute
        df = peak_trough_tip(arr_in)
    # TODO Review: is df.to_numpy() necessary ?
    # Create matrix of just NxC (spikes x time) of the peak waveforms channel (=1 channel)
    arr_peak = arr_in[np.arange(arr_in.shape[0]), :, df['peak_trace_idx'].to_numpy()]
    # Get the sign of the peak
    indx_pos = np.where(df['peak_val'].to_numpy() > 0)
    # Flip positive wavs so all are negative
    if len(indx_pos) > 0:
        arr_peak[indx_pos, :] = -1 * arr_peak[indx_pos, :]
    # Compute half max value, repmat and substract it
    half_max = df['peak_val'].to_numpy()/2
    half_max_rep = np.tile(half_max, (arr_peak.shape[1], 1)).transpose()
    # Note on the above: using np.tile because np. repeat does not work with axis=1
    arr_sub = arr_peak - half_max_rep
    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_sub, df['peak_time_idx'].to_numpy())
    # POST: Find first time it crosses 0 (from negative -> positive values)
    indx_post = np.argmax(arr_post > 0, axis=1)
    val_post = arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_post]
    # PRE: Find first time it crosses 0 (from positive -> negative values)
    indx_pre = np.argmax(arr_pre < 0, axis=1)
    val_pre = arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_pre]

    return indx_post, val_post, indx_pre, val_pre