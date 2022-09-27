"""
This module is to compute features on spike waveforms.
Throughout the code, the convention is to have 2D waveforms in the (time, traces)
For efficiency, several wavforms are fed in a memory contiguous manner: (iwaveform, time, traces)
"""
import numpy as np
import pandas as pd


def peak_trough_tip(arr_in, return_peak_trace=False):
    """
    From one or several single or multi-trace waveforms, extract the times and associated
     values of the peak, through and tip of the peak channel
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace)
    :return: indices of traces and peaks, length of N wav
    """
    # expand array if 2d
    if arr_in.ndim == 2:
        arr_in = arr_in[np.newaxis, :, :]

    # Init remove nan vals in entry array
    arr_in[np.isnan(arr_in)] = 0

    # 1. Find max peak (absolute deviation in STD units)
    max_vals = np.max(np.abs(arr_in[:, :]), axis=1)
    indx_maxs = np.argmax(np.abs(arr_in[:, :]), axis=1)
    indx_trace = np.argmax(max_vals, axis=1)
    indx_peak = indx_maxs[np.arange(0, indx_maxs.shape[0], 1), indx_trace]
    val_peak = arr_in[np.arange(0, arr_in.shape[0], 1), indx_peak, indx_trace]
    del (max_vals, indx_maxs)

    # 2. Find trough and tip (at peak waveform)
    # Per waveform, keep only trace that contains the peak
    arr_out = arr_in[np.arange(0, arr_in.shape[0], 1), :, indx_trace]

    # Create zero mask with 1 at peak, cumsum
    arr_mask = np.zeros(arr_out.shape)
    arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 1
    arr_mask = np.cumsum(arr_mask, axis=1)
    # arr_mask[np.arange(0, arr_mask.shape[0], 1), indx_peak] = 2  # to keep peak in both cases
    # Commented code above not needed: We want to keep peak = nan when keeping pre-values

    # Pad with Nans (as cannot slice since each waveform will have different length from peak)
    indx_prepeak = np.where(arr_mask == 0)
    indx_postpeak = np.where(arr_mask == 1)
    del arr_mask

    arr_pre = arr_out.copy()
    arr_pre = arr_pre.astype('float')
    arr_pre[indx_postpeak] = np.nan  # Array with values pre-, nans post- peak (from peak to end)

    arr_post = arr_out.copy()
    arr_post = arr_post.astype('float')
    arr_post[indx_prepeak] = np.nan  # Array with values post-, nans pre- peak (from start to peak-1)

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
    d_out = dict()
    d_out['peak_trace_idx'] = indx_trace
    d_out['peak_time_idx'] = indx_peak
    d_out['peak_val'] = val_peak

    d_out['trough_time_idx'] = indx_trough
    d_out['trough_val'] = val_trough

    d_out['tip_time_idx'] = indx_tip
    d_out['tip_val'] = val_tip

    df = pd.DataFrame(data=d_out)

    if return_peak_trace:
        return df, arr_out
    else:
        return df
