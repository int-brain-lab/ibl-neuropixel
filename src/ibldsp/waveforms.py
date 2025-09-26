"""
This module is to compute features on spike waveforms.
Throughout the code, the convention is to have 2D waveforms in the (time, traces)
For efficiency, several wavforms are fed in a memory contiguous manner: (iwaveform, time, traces)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

import neuropixel
from ibldsp.utils import parabolic_max, make_channel_index
from ibldsp.fourier import fshift

EXTRACT_RADIUS_UM = 200


def _validate_arr_in(arr_in):
    # expand array if 2d
    if arr_in.ndim == 2:
        arr_in = arr_in[np.newaxis, :, :]

    # Init remove nan vals in entry array
    arr_in[np.isnan(arr_in)] = 0
    return arr_in


def get_array_peak(arr_in, df):
    """
    Create matrix of just NxT (spikes x time) of the peak waveforms channel (=1 channel)

    :param arr_in: NxTxC waveform matrix (spikes x time x channel) ; expands to 1xTxC if TxC as input
    :param df: dataframe of waveform features
    :return: NxT waveform matrix : spikes x time, only the peak channel
    """
    arr_in = _validate_arr_in(arr_in)
    arr_peak = arr_in[np.arange(arr_in.shape[0]), :, df["peak_trace_idx"].to_numpy()]
    return arr_peak


def invert_peak_waveform(arr_peak, df):
    # Get the sign of the peak
    indx_pos = np.where(df["peak_val"].to_numpy() > 0)[0]
    # Flip positive wavs so all are negative
    if len(indx_pos) > 0:
        arr_peak[indx_pos, :] = -1 * arr_peak[indx_pos, :]

    df["invert_sign_peak"] = (
        np.sign(df["peak_val"]) * -1
    )  # Inverted signe peak to multiply point values by
    return arr_peak, df


def arr_pre_post(arr_peak, indx_peak):
    """
    :param arr_peak: NxT waveform matrix : spikes x time, only the peak channel
    :param indx_peak: Nx1 matrix : indices of the peak for each channel
    :return:
    """
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
    arr_pre = arr_pre.astype("float")
    arr_pre[indx_postpeak] = (
        np.nan
    )  # Array with values pre-, nans post- peak (from peak to end)

    arr_post = arr_peak.copy()
    arr_post = arr_post.astype("float")
    arr_post[indx_prepeak] = (
        np.nan
    )  # Array with values post-, nans pre- peak (from start to peak-1)
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
    # Select maximum of absolute value as peak
    indx_peak = indx_maxs[np.arange(0, indx_maxs.shape[0], 1), indx_trace]
    # Select minimum as peak (disregarded on 02-06-2023)
    # indx_peak = np.argmin(arr_in[np.arange(arr_in.shape[0]), :, indx_trace], axis=1)
    val_peak = arr_in[np.arange(0, arr_in.shape[0], 1), indx_peak, indx_trace]

    return indx_trace, indx_peak, val_peak


def find_peak(arr_in):
    """
    From one or several single or multi-trace waveforms, extract the times and associated
     values of the peak, through and tip of the peak channel
    :param: arr_in: array of waveforms; 3D dimension have to be (wav, time, trace)
    :return: indices of traces and peaks, length of N wav
    """
    arr_in = _validate_arr_in(arr_in)

    # 1. Find max peak (absolute deviation in STD units)
    indx_trace, indx_peak, val_peak = pick_maximum(arr_in)

    # Create dict / pd df
    df = pd.DataFrame()
    df["peak_trace_idx"] = indx_trace
    df["peak_time_idx"] = indx_peak
    df["peak_val"] = val_peak
    return df


def find_trough(arr_peak, df):
    # Find tip (at peak waveform)

    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_peak, df["peak_time_idx"].to_numpy())

    # Find trough
    # indx_trough = np.nanargmin(arr_post * np.sign(val_peak)[:, np.newaxis], axis=1)
    indx_trough = np.nanargmax(arr_post, axis=1)
    val_trough = (
        arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_trough]
        * df["invert_sign_peak"].to_numpy()
    )

    # Put values into df
    df["trough_time_idx"] = indx_trough
    df["trough_val"] = val_trough

    return df


def find_tip(arr_peak, df):
    # Find tip (at peak waveform)

    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_peak, df["peak_time_idx"].to_numpy())

    # Find tip
    """
    # 02-06-2023 ; Decided not to use the inflection point but rather maximum
    # Leaving code for now commented as legacy example

    # Inflection point
    y_dif1 = np.diff(arr_pre, axis=1)
    indx_posit = np.where(y_dif1 > 0)
    del arr_pre
    arr_cs = np.zeros(y_dif1.shape)
    arr_cs[indx_posit] = 1
    indx_tip = np.argmax(np.cumsum(arr_cs, axis=1), axis=1) + 1
    val_tip = arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_tip] * df['invert_sign_peak'].to_numpy()
    del arr_cs
    """
    # Maximum
    indx_tip = np.nanargmax(arr_pre, axis=1)
    val_tip = (
        arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_tip]
        * df["invert_sign_peak"].to_numpy()
    )

    # Put values into df
    df["tip_time_idx"] = indx_tip
    df["tip_val"] = val_tip

    return df


def find_tip_trough(arr_peak, arr_peak_real, df):
    """
    :param arr_in: inverted
    :param df:
    :return:
    """
    # 2. Find trough and tip (at peak waveform)

    # Find trough
    df = find_trough(arr_peak, df)
    df = peak_to_trough_ratio(df)
    # If ratio of peak/trough is near 1, and peak is positive :
    # Assign trough as peak on same waveform channel
    # Call the function again to compute trough etc. with new peak assigned

    # Find df rows to be changed
    df_index = df.index[(df["peak_val"] > 0) & (df["peak_to_trough_ratio"] <= 1.5)]
    df_rows = df.iloc[df_index]
    if len(df_index) > 0:
        # New peak - Swap peak for trough values
        df_rows = df_rows.drop(["peak_val", "peak_time_idx"], axis=1)
        df_rows["peak_val"] = df_rows["trough_val"]
        df_rows["peak_time_idx"] = df_rows["trough_time_idx"]

        # df_trials.loc[iss, f] = predicted[f].values

        # Drop trough columns
        df_rows = df_rows.drop(["trough_time_idx", "trough_val"], axis=1)
        # Create mini arr_peak for those rows uniquely (take the real waveforms value in, not inverted ones)
        arr_peak_rows = arr_peak_real[df_index, :]
        # Place into "inverted" array peak for return
        arr_peak[df_index, :] = arr_peak_rows
        # Get new sign for the peak
        arr_peak_rows, df_rows = invert_peak_waveform(arr_peak_rows, df_rows)
        # New trough
        df_rows = find_trough(arr_peak_rows, df_rows)
        # New peak-trough ratio
        df_rows = peak_to_trough_ratio(df_rows)
        # Assign back into the dataframe
        df.loc[df_index] = df_rows
    # Find tip
    df = find_tip(arr_peak, df)

    return df, arr_peak


def plot_wiggle(
    wav,
    fs=1,
    ax=None,
    scale=0.3,
    clip=10,
    fill_sign=-1,
    plot_kwargs=None,
    fill_kwargs=None,
):
    """
    Displays a multi-trace waveform in a wiggle traces with negative
    amplitudes filled
    :param wav: (nchannels, nsamples)
    :param fs: sampling rate
    :param ax: axis to plot on
    :param scale: waveform amplitude that will be displayed as one inter-trace: if scale = 20e-6 one intertrace will be 20uV
    :param clip: maximum value for the traces
    :param fill_sign: -1 for negative (default for spikes), 1 for positive
    :param plot_kwargs: kwargs for the line plot
    :param fill_kwargs: kwargs for the fill
    :return: axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    plot_kwargs = {"color": "k", "linewidth": 0.5} | (plot_kwargs or {})
    fill_kwargs = {"color": "k", "aa": True} | (fill_kwargs or {})
    nc, ns = wav.shape
    vals = np.c_[wav, wav[:, :1] * np.nan].ravel()  # flat view of the 2d array.
    vect = np.arange(vals.size).astype(
        np.float32
    )  # flat index array, for correctly locating zero crossings in the flat view
    crossing = np.where(np.diff(np.signbit(vals)))[0]  # index before zero crossing
    # use linear interpolation to find the zero crossing, i.e. y = mx + c.
    x1 = vals[crossing]
    x2 = vals[crossing + 1]
    y1 = vect[crossing]
    y2 = vect[crossing + 1]
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    # tack these values onto the end of the existing data
    x = np.hstack([vals, np.zeros_like(c)]) / scale
    x = np.maximum(np.minimum(x, clip), -clip)
    y = np.hstack([vect, c])
    # resort the data
    order = np.argsort(y)
    # shift from amplitudes to plotting coordinates
    x_shift, y = y[order].__divmod__(ns + 1)
    ax.plot(y / fs, x[order] + x_shift + 1, **plot_kwargs)
    if fill_sign < 0:
        x[x > 0] = np.nan
    else:
        x[x < 0] = np.nan
    x = x[order] + x_shift + 1
    ax.fill(y / fs, x, **fill_kwargs)
    ax.set(xlim=[0, ns / fs], ylim=[-1, nc + 1])
    plt.tight_layout()
    return ax


def double_wiggle(wav, fs=1, ax=None, colors=None, **kwargs):
    """
    Double trouble: this wiggle colours both the negative and the postive values
    :param wav: (nchannels, nsamples)
    :param fs: sampling rate
    :param ax: axis to plot on
    :param scale: scale factor for the traces
    :param clip: maximum value for the traces
    :param fill_sign: -1 for negative (default for spikes), 1 for positive
    :param plot_kwargs: kwargs for the line plot
    :param fill_kwargs: kwargs for the fill
    :return:
    """
    if colors is None:
        cmap = "PuOr"
        _cmap = mpl.colormaps.get_cmap(cmap)
        colors = _cmap(np.linspace(0, 1, 256))
        colors = [colors[50], colors[-50]]
    if ax is None:
        fig, ax = plt.subplots()
    plot_wiggle(
        wav,
        fs=fs / 1e3,
        ax=ax,
        plot_kwargs={"linewidth": 0},
        fill_kwargs={"color": colors[0]},
        **kwargs,
    )
    plot_wiggle(
        wav,
        fs=fs / 1e3,
        ax=ax,
        fill_sign=1,
        plot_kwargs={"linewidth": 0.5},
        fill_kwargs={"color": colors[1]},
        **kwargs,
    )
    return ax


def plot_peaktiptrough(df, arr, ax, nth_wav=0, plot_grey=True, fs=30000):
    # Time axix
    nech, ntr = arr[nth_wav].shape
    tscale = np.array([0, nech - 1]) / fs * 1e3

    if ax is None:
        fig, ax = plt.subplots()
    if plot_grey:
        ax.plot(tscale, arr[nth_wav], c="gray", alpha=0.5)
    # Peak channel
    ax.plot(
        tscale,
        arr[nth_wav][:, int(df.iloc[nth_wav].peak_trace_idx)],
        marker=".",
        c="blue",
    )
    # Peak point
    ax.plot(tscale[df.iloc[nth_wav].peak_time_idx], df.iloc[nth_wav].peak_val, "r*")
    # Trough point
    ax.plot(tscale[df.iloc[nth_wav].trough_time_idx], df.iloc[nth_wav].trough_val, "g*")
    # Tip point
    ax.plot(tscale[df.iloc[nth_wav].tip_time_idx], df.iloc[nth_wav].tip_val, "k*")
    # Half peak points
    ax.plot(
        tscale[df.iloc[nth_wav].half_peak_post_time_idx],
        df.iloc[nth_wav].half_peak_post_val,
        "c*",
    )
    ax.plot(
        tscale[df.iloc[nth_wav].half_peak_pre_time_idx],
        df.iloc[nth_wav].half_peak_pre_val,
        "c*",
    )
    # Line for half peak boundary
    # ax.plot((0, arr.shape[1]), np.array((1, 1)) * df.iloc[nth_wav].peak_val / 2, '-k')
    ax.plot(
        (tscale[0], tscale[-1]), np.array((1, 1)) * df.iloc[nth_wav].peak_val / 2, "-k"
    )
    # Recovery point
    ax.plot(
        tscale[df.iloc[nth_wav].recovery_time_idx], df.iloc[nth_wav].recovery_val, "y*"
    )
    # Axis labels
    ax.set_ylabel("(Volt)")
    ax.set_xlabel("Time (ms)")


def half_peak_point(arr_peak, df):
    """
    Compute the two intersection points at halp-maximum peak
    :param: arr_peak: NxT waveform matrix : spikes x time, only the peak channel (inverted for positive wavs)
    :return: df with columns containing indices of intersection points and values, length of N wav
    """
    # TODO Review: is df.to_numpy() necessary ?
    # Compute half max value, repmat and substract it
    half_max = (df["peak_val"].to_numpy() / 2) * df["invert_sign_peak"].to_numpy()
    half_max_rep = np.tile(half_max, (arr_peak.shape[1], 1)).transpose()
    # Note on the above: using np.tile because np.repeat does not work with axis=1
    # todo rewrite with np.repeat and np.newaxis
    arr_sub = arr_peak - half_max_rep
    # Create masks pre/post
    arr_pre, arr_post = arr_pre_post(arr_sub, df["peak_time_idx"].to_numpy())
    # POST: Find first time it crosses 0 (from negative -> positive values)
    indx_post = np.argmax(arr_post > 0, axis=1)
    val_post = (
        arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_post]
        * df["invert_sign_peak"].to_numpy()
    )
    # PRE:
    # Invert matrix (flip L-R) to find first point crossing threshold before peak
    arr_pre_flip = np.fliplr(arr_pre)
    # Find first time it crosses 0 (from negative -> positive values)
    indx_pre_flip = np.argmax(arr_pre_flip > 0, axis=1)
    # Fill a matrix of 0 with 1 at index, flip, then find index
    arr_zeros = np.zeros(arr_pre_flip.shape)
    arr_zeros[np.arange(0, arr_pre_flip.shape[0], 1), indx_pre_flip] = 1
    arr_pre_ones = np.fliplr(arr_zeros)
    # Find index where there are 1
    indx_pre = np.argmax(arr_pre_ones > 0, axis=1)
    val_pre = (
        arr_peak[np.arange(0, arr_peak.shape[0], 1), indx_pre]
        * df["invert_sign_peak"].to_numpy()
    )

    # Add columns to DF and return
    df["half_peak_post_time_idx"] = indx_post
    df["half_peak_pre_time_idx"] = indx_pre
    df["half_peak_post_val"] = val_post
    df["half_peak_pre_val"] = val_pre

    return df


def half_peak_duration(df, fs=30000):
    """
    Compute the half peak duration (in second)
    :param df: dataframe of waveforms features, with the half peak intersection points computed
    :param fs:  sampling rate (Hz)
    :return: dataframe wirth added column
    """
    df["half_peak_duration"] = (
        df["half_peak_post_time_idx"] - df["half_peak_pre_time_idx"]
    ) / fs
    return df


def peak_to_trough_duration(df, fs=30000):
    """
    Compute the duration (second) of the peak-to-trough
    :param df: dataframe of waveforms features
    :param fs: sampling rate (Hz)
    :return: df
    """
    # Duration
    df["peak_to_trough_duration"] = (df["trough_time_idx"] - df["peak_time_idx"]) / fs
    return df


def peak_to_trough_ratio(df):
    """
    Compute the ratio of the peak-to-trough
    :param df: dataframe of waveforms features
    :param fs: sampling rate (Hz)
    :return:
    """
    # Ratio
    df["peak_to_trough_ratio"] = np.abs(
        df["peak_val"] / df["trough_val"]
    )  # Division by 0 returns NaN
    # Ratio log-scale
    df["peak_to_trough_ratio_log"] = np.log(df["peak_to_trough_ratio"])
    return df


def polarisation_slopes(df, fs=30000):
    """
    Computes the depolarisation and repolarisation slopes as the difference between tip-peak
    and peak-trough respectively.
    :param df: dataframe of waveforms features
    :param fs: sampling frequency (Hz)
    :return: dataframe with added columns
    """
    # Depolarisation: slope before the peak (between tip and peak)
    depolarise_duration = (df["peak_time_idx"] - df["tip_time_idx"]) / fs
    depolarise_volt = df["peak_val"] - df["tip_val"]
    df["depolarisation_slope"] = depolarise_volt / depolarise_duration
    # Repolarisation: slope after the peak (between peak and trough)
    repolarise_duration = (df["trough_time_idx"] - df["peak_time_idx"]) / fs
    repolarise_volt = df["trough_val"] - df["peak_val"]
    df["repolarisation_slope"] = repolarise_volt / repolarise_duration
    return df


def recovery_point(arr_peak, df, idx_from_trough=5):
    """
    Compute the single recovery secondary point (selected by a fixed increment
    from the trough). If the fixed increment from the trough is outside the matrix boundary, the
    last value of the waveform is used.
    :param arr_peak: NxT waveform matrix : spikes x time, only the peak channel
    :param df: dataframe of waveforms features
    :param idx_from_trough: sample index to be taken into account for the second point ; index from the trough
    :return: dataframe with added columns
    """
    # Check range is not outside of matrix boundary)
    if idx_from_trough >= (arr_peak.shape[1]):
        raise ValueError("Index out of bound: Index larger than waveform array shape")

    # Check df['peak_time_idx'] + pt_idx is not out of bound
    idx_all = df["trough_time_idx"].to_numpy() + idx_from_trough
    # Find waveform(s) for which the second point is outside matrix boundary range
    idx_over = np.where(idx_all > arr_peak.shape[1])[0]
    if len(idx_over) > 0:
        # Todo should this raise a warning ?
        idx_all[idx_over] = arr_peak.shape[1] - 1  # Take the last value of the waveform

    df["recovery_time_idx"] = idx_all
    df["recovery_val"] = (
        arr_peak[np.arange(0, arr_peak.shape[0], 1), idx_all]
        * df["invert_sign_peak"].to_numpy()
    )
    return df


def recovery_slope(df, fs=30000):
    """
    Compute the recovery slope, from the trough to the single secondary point.
    :param df: dataframe of waveforms features
    :param fs: sampling frequency (Hz)
    :return: dataframe with added columns
    """
    # Note: this could be lumped in with the polarisation_slopes
    # Time, volt and slope values
    recovery_duration = (
        df["recovery_time_idx"] - df["trough_time_idx"]
    ) / fs  # Diff between second point and peak
    recovery_volt = df["recovery_val"] - df["trough_val"]
    df["recovery_slope"] = recovery_volt / recovery_duration
    return df


def dist_chanel_from_peak(channel_geometry, peak_trace_idx):
    """
    Compute distance for each channel from the peak channel, for each spike
    :param channel_geometry: Matrix N(spikes) * N(channels) * 3 (spatial coordinates x,y,z)
    # Note: computing this to provide it as input will be a pain
    :param peak_trace_idx: index of the highest amplitude channel in the multi-channel waveform
    :return: eu_dist : N(spikes) * N(channels): the euclidian distance between each channel and the peak channel,
    for each waveform
    """
    # Note: It deals with Nan in entry coordinate (fake padding channels); returns Nan as Eu dist
    # Get peak coordinates (x,y,z)
    peak_coord = channel_geometry[
        np.arange(0, channel_geometry.shape[0], 1), peak_trace_idx, :
    ]

    # repmat peak coordinates (x,y,z) [Nspikes x Ncoordinates] across channels
    peak_coord_rep = np.repeat(
        peak_coord[:, :, np.newaxis], channel_geometry.shape[1], axis=2
    )  # Todo -1
    peak_coord_rep = np.swapaxes(
        peak_coord_rep, 1, 2
    )  # N spikes x channel x coordinates

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
    """
    Returns the spatial spread defined by the sum(w_i * dist_i) / sum(w_i).
    The weight is a given value per channel (e.g. the absolute peak voltage value)
    :param eu_dist: N(spikes) * N(channels): the euclidian distance between each channel and the peak channel,
    for each waveform
    :param weights: N(spikes) * N(channels): the weights per channel per spikes
    :return: spatial_spread : N(spikes) * 1 vector
    """
    # Note: possible to have nan entries in eu_dist
    spatial_spread = np.nansum(np.multiply(eu_dist, weights), axis=1) / np.sum(
        weights, axis=1
    )
    return spatial_spread


def reshape_wav_one_channel(arr):
    """
    Reshape matrix so instead of being like waveforms: (wav, time, trace) i.e. (npsikes x nsamples x nchannels)
    it is of size (npsikes * nchannels) x nsamples
    :param waveforms: 3D np.array containing multi-channel waveforms, 3D dimension have to be (wav, time, trace)
    :return:
    """
    # Swap axis so the matrix is now: wav x channel x time
    arr_ax = np.swapaxes(arr, 1, 2)
    # reshape using the first 2 dimension (multiplied) x time
    arr_resh = arr_ax.reshape(-1, arr_ax.shape[-1])
    # add a new axis for computation
    arr_out = arr_resh[:, :, np.newaxis]
    return arr_out


def weights_spk_ch(arr, weight_type="peak"):
    """
    Compute a value on all channels of a waveform matrix, and return as weights (to be used in spatial spread).
    :param arr: 3D np.array containing multi-channel waveforms, 3D dimension have to be (wav, time, trace)
    :param weight_type: value to be returned as weight (implemented: peak)
    :return: weights: N(spikes) * N(channels): the weights per channel per spikes
    """
    # Reshape
    arr_resh = reshape_wav_one_channel(arr)
    # Peak
    df = find_peak(arr_resh)
    if weight_type == "peak":
        weights_flat = df["peak_val"].to_numpy()
    else:
        raise ValueError("weight_type: unknown value attributed")
    # Reshape
    # Order in DF: #1-2-3 channel of spike #1, then #1-2-3 channel spike #2 etc
    weights = np.reshape(weights_flat, (arr.shape[0], arr.shape[2]))
    return weights


def compute_spatial_spread(arr, df, channel_geometry, weight_type="peak"):
    eu_dist = dist_chanel_from_peak(channel_geometry, df)
    weights = weights_spk_ch(arr, weight_type)
    df["spatial_spread"] = spatial_spread_weighted(eu_dist, weights)
    return df


def compute_spike_features(
    arr_in, fs=30000, recovery_duration_ms=0.16, return_peak_channel=False
):
    """
    This is the main function to compute spike features from a set of waveforms
    Current features:
    Index(['peak_trace_idx', 'peak_time_idx', 'peak_val', 'trough_time_idx',
       'trough_val', 'tip_time_idx', 'tip_val', 'half_peak_post_time_idx',
       'half_peak_pre_time_idx', 'half_peak_post_val', 'half_peak_pre_val',
       'half_peak_duration', 'recovery_time_idx', 'recovery_val',
       'depolarisation_slope', 'repolarisation_slope', 'recovery_slope'],
    :param arr_in: 3D np.array containing multi-channel waveforms; 3D dimension have to be (wav, time, trace)
    :param fs: sampling frequency (Hz)
    :recovery_duration_ms: in ms, the duration from the trough to the recovery point
    :param return_peak_channel: if True, return the peak channel traces
    :return: dataframe of spikes with all features,
    Returns:
    """
    df = find_peak(arr_in)
    # Per waveform, keep only trace that contains the peak
    arr_peak_real = get_array_peak(arr_in, df)
    # Invert positive spikes
    arr_peak, df = invert_peak_waveform(
        arr_peak_real.copy(), df
    )  # Copy otherwise overwrite the variable in memory
    # Tip-trough (this also computes the peak_to_trough_ratio)
    df, arr_peak = find_tip_trough(arr_peak, arr_peak_real, df)
    # Peak to trough duration
    df = peak_to_trough_duration(df, fs=fs)
    # Half peak points
    df = half_peak_point(arr_peak, df)
    # Half peak duration
    df = half_peak_duration(df, fs=fs)
    # Recovery point
    df = recovery_point(
        arr_peak, df, idx_from_trough=int(round(recovery_duration_ms * fs / 1000))
    )
    # Slopes
    df = polarisation_slopes(df, fs=fs)
    df = recovery_slope(df, fs=fs)

    if return_peak_channel:
        return df, arr_peak_real
    else:
        return df


def wave_shift_corrmax(spike, spike2):
    """
    Shift in time (sub-sample) the spike2 onto the spike
    (For residual subtraction, typically, the spike2 would be the template)
    :param spike: 1D array of float (e.g. on peak channel); same size as spike2
    :param spike2: 1D array of float
    :return: spike_resync: 1D array of float, shift_computed: in time sample (e.g. -4.03)
    """
    # Numpy implementation of correlation centers it in the middle at np.floor(len_sample/2)
    assert spike.shape[0] == spike2.shape[0]
    sig_len = spike.shape[0]
    c = scipy.signal.correlate(spike, spike2, mode="same")
    ipeak, maxi = parabolic_max(c)
    shift_computed = (ipeak - np.floor(sig_len / 2)) * -1
    spike_resync = fshift(spike2, -shift_computed)
    return spike_resync, shift_computed


# -------------------------------------------------------------
# Functions to fit the phase slope, and find the relationship between phase slope and sample shift


def line_fit(x, a, b):  # function to fit a line and get the slope out
    return a * x + b


def get_apf_from2spikes(spike, spike2, fs):
    fscale = np.fft.rfftfreq(spike.size, 1 / fs)
    C = np.fft.rfft(spike) * np.conj(np.fft.rfft(spike2))

    # Take the phase for freq at high amplitude, and compute slope
    amp = np.abs(C)
    phase = np.unwrap(np.angle(C))
    return amp, phase, fscale


def get_phase_slope(amp, phase, fscale, q=90):
    # Take 90 percentile of distribution to find high amplitude freq
    thresh_amp = np.percentile(amp, q)
    indx_highamp = np.where(amp >= thresh_amp)[0]
    # Perform linear fit to get the slope
    popt, _ = scipy.optimize.curve_fit(
        line_fit, xdata=fscale[indx_highamp], ydata=phase[indx_highamp]
    )
    a, b = popt
    return a, b


def fit_phaseshift(phase_slopes, sample_shifts):
    # Get parameters for the phase slope / sample shift curve
    popt, _ = scipy.optimize.curve_fit(
        line_fit, xdata=sample_shifts, ydata=phase_slopes
    )
    a, b = popt
    return a, b


def get_phase_from_fit(sample_shifts, a, b):
    # phases = line_fit(np.abs(sample_shifts), a, b) * np.sign(sample_shifts)
    phases = line_fit(sample_shifts, a, b)
    return phases


def get_shift_from_fit(phases, a, b):
    # Invert the line function: x = (y-b)/a
    sample_shifts = (phases - b) / a
    return sample_shifts


def get_spike_slopeparams(spike, fs, num_estim=50):
    sample_shifts = np.linspace(-1, 1, num=num_estim)
    phase_slopes = np.empty(shape=sample_shifts.shape)

    for i_shift, sample_shift in enumerate(sample_shifts):
        spike2 = fshift(spike, sample_shift)
        # Get amplitude, phase, fscale
        amp, phase, fscale = get_apf_from2spikes(spike, spike2, fs)
        # Perform linear fit to get the slope
        a, b = get_phase_slope(amp, phase, fscale)
        phase_slopes[i_shift] = a

    a_pslope, b_pslope = fit_phaseshift(phase_slopes, sample_shifts)
    return a_pslope, b_pslope, sample_shifts, phase_slopes


def wave_shift_phase(spike, spike2, fs, a_pos=None, b_pos=None):
    """
    Resynch spike2 onto spike using the phase spectrum's slope
    (this work perfectly in theory, but does not work well with raw daw sampled at 30kHz!)
    """
    # Get template parameters if not passed in
    if a_pos is None or b_pos is None:
        a_pos, b_pos, _, _ = get_spike_slopeparams(spike, fs)
    # Get amplitude, phase, fscale
    amp, phase, fscale = get_apf_from2spikes(spike, spike2, fs)
    # Perform linear fit to get the slope
    a, b = get_phase_slope(amp, phase, fscale)
    phase_slope = a
    # Get sample shift
    sample_shift = get_shift_from_fit(phase_slope, a_pos, b_pos)

    # Resynch in time given phase slope
    spike_resync = fshift(spike2, -sample_shift)  # Use negative to re-synch
    return spike_resync, sample_shift


# End of functions
# -------------------------------------------------------------


def shift_waveform(wf_cluster):
    """
    :param wf_cluster: # A matrix of spike waveforms per cluster (N spike, trace, time)
    :return: wf_out (same shape as waveform cluster): A matrix with the waveforms shifted in time
    """
    # Take first the average as template to compute shift on
    wfs_avg = np.nanmedian(wf_cluster, axis=0)
    # Find the peak channel from template
    template = np.transpose(
        wfs_avg.copy()
    )  # wfs_avg is 2D (trace, time) -> transpose: (time, trace)
    arr_temp = np.expand_dims(
        template, axis=0
    )  # 3D dimension have to be (wav, time, trace) -> add 1 dimension (ax=0)
    df_temp = find_peak(arr_temp)
    spike_template = arr_temp[
        :, :, df_temp["peak_trace_idx"][0]
    ]  # Take template at peak trace
    spike_template = np.squeeze(spike_template)

    # Take the raw spikes at that channel
    # Create df for all spikes
    """
    Note: took the party here to NOT recompute the peak channel of each waveform, but to reuse the one from the
    template — this is because the function to find the peak assumes the waveform has been denoised
    and uses the maximum amplitude value --> which here would lead to failures in the case of collision
    """
    df = pd.DataFrame()
    df["peak_trace_idx"] = [df_temp["peak_trace_idx"][0]] * wf_cluster.shape[0]

    # Per waveform, keep only trace that contains the peak
    arr_in = np.swapaxes(
        wf_cluster, axis1=1, axis2=2
    )  # wfs size (wav, trace, time) -> swap (wav, time, trace)
    arr_peak_real = get_array_peak(arr_in, df)

    # Resynch 1 spike with 1 template (using only peak channel) ; Apply shift to all wav traces
    wf_out = np.zeros(wf_cluster.shape)
    shift_applied = np.zeros(wf_cluster.shape[0])
    for i_spike in range(0, wf_cluster.shape[0]):
        # Raw spike at peak channel
        spike_raw = arr_peak_real[i_spike, :]
        # Resynch
        spike_template_resynch, shift_computed = wave_shift_corrmax(
            spike_raw, spike_template
        )
        # Apply shift to all traces at once
        wfs_avg_resync = fshift(wf_cluster[i_spike, :, :], shift_computed)
        wf_out[i_spike, :, :] = wfs_avg_resync
        shift_applied[i_spike] = shift_computed

    return wf_out, shift_applied


def iradon(w, sr, offset, fmin=0, fmax=None, slowness=None, N=1, mu=0.1):
    """
    Given a waveform w, computes the plane wave decomposition
    :param w:
    :param sr:
    :param offset:
    :param slowness:
    :param fmin: minimum frequency to invert for, defaults to 0
    :param fmax: maximum frequency to invert for, defaults to Nyquist
    :param N: ray parameter: 1: linear Tau-P, 2: parabolic Tau-P
    :param mu:
    :return:
    """

    fmax = 1 / (sr * 2) if fmax is None else fmax
    if slowness is None:
        ds = 1 / fmax / (np.max(offset) - np.min(offset))
        dx = np.median(np.abs(np.diff(offset)))
        smax = 1 / dx / fmax
        slowness = np.linspace(0, smax, int(np.ceil(smax / ds)))

    # Decompose les donnees sous la forme d'une somme d'ondes de vitesse 1./slowness
    ns, nc = w.shape
    nq = slowness.size

    W = np.fft.rfft(w, axis=0)
    freqs = np.fft.rfftfreq(ns, d=sr)
    R = np.zeros((W.shape[0], nq), dtype=complex)

    # propagation matrix M[nc, nq] corresponds to time shifts (secs)
    M = (offset[:, np.newaxis] ** N) * slowness[np.newaxis, :]

    for k in np.where(np.logical_and(fmin <= freqs, freqs <= fmax))[0]:
        L = np.exp(2j * np.pi * freqs[k] * M)
        y = np.conj(W[k, :]).T
        MATRIX = np.conj(L).T @ L
        tr = np.abs(np.trace(MATRIX))
        Q = mu * tr * np.eye(nq) / nq
        x, _, _, _ = np.linalg.lstsq(MATRIX + Q, np.conj(L).T @ y, rcond=None)
        R[k, :] = np.conj(x).T

    return np.fft.irfft(R, ns, axis=0)


def radon(r, sr, offset, slowness, N=1, fmin=0, fmax=None):
    """
    Given a plane-wave decomposition, returns
    :param r:
    :param sr:
    :param offset:
    :param slowness:
    :param N:
    :param fmin:
    :param fmax:
    :return:
    """
    fmax = 1 / (sr * 2) if fmax is None else fmax
    ns, nq = r.shape
    nc = offset.size
    R = np.fft.rfft(r, axis=0)
    W = np.zeros((r.shape[0], nc), dtype=complex)
    freqs = np.fft.rfftfreq(ns, d=sr)
    for k in np.where(np.logical_and(fmin <= freqs, freqs <= fmax))[0]:
        propagation = (offset[:, np.newaxis] ** N) * slowness[np.newaxis, :]  # nc * nq
        L = np.exp(2j * np.pi * freqs[k] * propagation)  # nc * nq
        W[k, :] = np.conj(L @ np.conj(R[k, :]).T).T

    return np.fft.irfft(W, ns, axis=0)


def get_waveforms_coordinates(
    trace_indices,
    xy=None,
    extract_radius_um=EXTRACT_RADIUS_UM,
    return_indices=False,
):
    """
    Get coordinates of channels within a specified radius around spike detection channels.

    This function reproduces the localization code channel selection when extracting
    waveforms from raw data. For each spike detection channel, it identifies all channels
    within a specified radius and returns their coordinates.

    Parameters
    ----------
    trace_indices : numpy.ndarray
        Array of shape (nspikes,) containing indices of the traces where spikes were detected.
    xy : numpy.ndarray, optional
        Array of shape (nchannels, 2) containing x and y coordinates of all channels.
        If None, default Neuropixel probe geometry will be used.
    extract_radius_um : float, default=EXTRACT_RADIUS_UM
        Radius in micrometers around each spike detection channel. All channels within
        this radius will be included in the output.
    return_indices : bool, default=False
        If True, also returns the indices of the channels within the radius.

    Returns
    -------
    wxy : numpy.ndarray
        Array of shape (nspikes, ntraces, 2) containing x and y coordinates of channels
        within the specified radius for each spike.
    inds : numpy.ndarray, optional
        Array of shape (nspikes, ntraces) containing indices of channels within the
        specified radius for each spike. Only returned if return_indices is True.

    Notes
    -----
    Channels outside the specified radius are represented with NaN coordinates.
    """
    if xy is None:
        th = neuropixel.trace_header(version=1)
        xy = np.c_[th["x"], th["y"]]
    channel_lookups = make_channel_index(
        xy, radius=extract_radius_um, pad_val=xy.shape[0]
    )
    inds = channel_lookups[trace_indices.astype(np.int32)]
    # add a dummy channel to have nans in the coordinates
    inds[np.isnan(inds)] = xy.shape[0] - 1
    wxy = np.r_[(xy, xy[0, np.newaxis] * np.nan)][inds.astype(np.int32)]
    if return_indices:
        return wxy, inds.astype(int)
    else:
        return wxy
