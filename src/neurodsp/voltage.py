"""
Module to work with raw voltage traces. Spike sorting pre-processing functions.
"""
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd
from joblib import Parallel, delayed, cpu_count

from iblutil.numerical import rcoeff
import spikeglx
import neuropixel

import neurodsp.fourier as fourier
import neurodsp.utils as utils

import logging
import traceback


def agc(x, wl=0.5, si=0.002, epsilon=1e-8, gpu=False):
    """
    Automatic gain control
    w_agc, gain = agc(w, wl=.5, si=.002, epsilon=1e-8)
    such as w_agc * gain = w
    :param x: seismic array (sample last dimension)
    :param wl: window length (secs)
    :param si: sampling interval (secs)
    :param epsilon: whitening (useful mainly for synthetic data)
    :param gpu: bool
    :return: AGC data array, gain applied to data
    """
    if gpu:
        import cupy as gp
    else:
        gp = np
    ns_win = int(gp.round(wl / si / 2) * 2 + 1)
    w = gp.hanning(ns_win)
    w /= gp.sum(w)
    gain = fourier.convolve(gp.abs(x), w, mode="same", gpu=gpu)
    gain += (gp.sum(gain, axis=1) * epsilon / x.shape[-1])[:, gp.newaxis]
    dead_channels = np.sum(gain, axis=1) == 0
    x[~dead_channels, :] = x[~dead_channels, :] / gain[~dead_channels, :]
    if gpu:
        return (x * gain).astype("float32"), gain.astype("float32")

    return x, gain


def fk(
    x,
    si=0.002,
    dx=1,
    vbounds=None,
    btype="highpass",
    ntr_pad=0,
    ntr_tap=None,
    lagc=0.5,
    collection=None,
    kfilt=None,
):
    """Frequency-wavenumber filter: filters apparent plane-waves velocity
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param si: sampling interval (secs)
    :param dx: spatial interval (usually meters)
    :param vbounds: velocity high pass [v1, v2], cosine taper from 0 to 1 between v1 and v2
    :param btype: {‘lowpass’, ‘highpass’}, velocity filter : defaults to highpass
    :param ntr_pad: padding will add ntr_padd mirrored traces to each side
    :param ntr_tap: taper (if None, set to ntr_pad)
    :param lagc: length of agc in seconds. If set to None or 0, no agc
    :param kfilt: optional (None) if kfilter is applied, parameters as dict (bounds are in m-1
    according to the dx parameter) kfilt = {'bounds': [0.05, 0.1], 'btype', 'highpass'}
    :param collection: vector length ntraces. Each unique value set of traces is a collection
    on which the FK filter will run separately (shot gaters, receiver gathers)
    :return:
    """
    if collection is not None:
        xout = np.zeros_like(x)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = fk(
                x[sel, :],
                si=si,
                dx=dx,
                vbounds=vbounds,
                ntr_pad=ntr_pad,
                ntr_tap=ntr_tap,
                lagc=lagc,
                collection=None,
            )
        return xout

    assert vbounds
    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # compute frequency wavenumber scales and deduce the velocity filter
    fscale = fourier.fscale(nt, si)
    kscale = fourier.fscale(nxp, dx)
    kscale[0] = 1e-6
    v = fscale[np.newaxis, :] / kscale[:, np.newaxis]
    if btype.lower() in ["highpass", "hp"]:
        fk_att = fourier.fcn_cosine(vbounds)(np.abs(v))
    elif btype.lower() in ["lowpass", "lp"]:
        fk_att = 1 - fourier.fcn_cosine(vbounds)(np.abs(v))

    # if a k-filter is also provided, apply it
    if kfilt is not None:
        katt = fourier._freq_vector(np.abs(kscale), kfilt["bounds"], typ=kfilt["btype"])
        fk_att *= katt[:, np.newaxis]

    # import matplotlib.pyplot as plt
    # plt.imshow(np.fft.fftshift(np.abs(v), axes=0).T, aspect='auto', vmin=0, vmax=1e5,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])
    # plt.imshow(np.fft.fftshift(np.abs(fk_att), axes=0).T, aspect='auto', vmin=0, vmax=1,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])

    # apply the attenuation in fk-domain
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=si)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = np.r_[np.flipud(xf[:ntr_pad]), xf, np.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fourier.fcn_cosine([0, ntr_tap])(np.arange(nxp))  # taper up
        taper *= 1 - fourier.fcn_cosine([nxp - ntr_tap, nxp])(
            np.arange(nxp)
        )  # taper down
        xf = xf * taper[:, np.newaxis]
    xf = np.real(np.fft.ifft2(fk_att * np.fft.fft2(xf)))

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf * gain


def car(x, collection=None, lagc=300, butter_kwargs=None, **kwargs):
    """
    Applies common average referencing with optional automatic gain control
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param collection:
    :param lagc: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering parameters: defaults: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    :return:
    """
    if butter_kwargs is None:
        butter_kwargs = {"N": 3, "Wn": 0.1, "btype": "highpass"}
    if collection is not None:
        xout = np.zeros_like(x)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = kfilt(
                x=x[sel, :],
                ntr_pad=0,
                ntr_tap=None,
                collection=None,
                butter_kwargs=butter_kwargs,
            )
        return xout

    # apply agc and keep the gain in handy
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=1.0)
    # apply CAR and then un-apply the gain
    xf = xf - np.median(xf, axis=0)
    return xf * gain


def kfilt(
    x, collection=None, ntr_pad=0, ntr_tap=None, lagc=300, butter_kwargs=None, gpu=False
):
    """
    Applies a butterworth filter on the 0-axis with tapering / padding
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param collection:
    :param ntr_pad: traces added to each side (mirrored)
    :param ntr_tap: n traces for apodizatin on each side
    :param lagc: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering parameters: defaults: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    :param gpu: bool
    :return:
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    if butter_kwargs is None:
        butter_kwargs = {"N": 3, "Wn": 0.1, "btype": "highpass"}
    if collection is not None:
        xout = gp.zeros_like(x)
        for c in gp.unique(collection):
            sel = collection == c
            xout[sel, :] = kfilt(
                x=x[sel, :],
                ntr_pad=0,
                ntr_tap=None,
                collection=None,
                butter_kwargs=butter_kwargs,
            )
        return xout
    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # apply agc and keep the gain in handy
    if not lagc:
        xf = gp.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=1.0, gpu=gpu)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = gp.r_[gp.flipud(xf[:ntr_pad]), xf, gp.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fourier.fcn_cosine([0, ntr_tap], gpu=gpu)(gp.arange(nxp))  # taper up
        taper *= 1 - fourier.fcn_cosine([nxp - ntr_tap, nxp], gpu=gpu)(
            gp.arange(nxp)
        )  # taper down
        xf = xf * taper[:, gp.newaxis]
    sos = scipy.signal.butter(**butter_kwargs, output="sos")
    if gpu:
        from .filter_gpu import sosfiltfilt_gpu

        xf = sosfiltfilt_gpu(sos, xf, axis=0)
    else:
        xf = scipy.signal.sosfiltfilt(sos, xf, axis=0)

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf * gain


def interpolate_bad_channels(
    data, channel_labels=None, x=None, y=None, p=1.3, kriging_distance_um=20, gpu=False
):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    The weights applied to neighbouring channels come from an exponential decay function
    :param data: (nc, ns) np.ndarray
    :param channel_labels; (nc) np.ndarray: 0: channel is good, 1: dead, 2:noisy, 3: out of the brain
    :param x: channel x-coordinates, np.ndarray
    :param y: channel y-coordinates, np.ndarray
    :param p:
    :param kriging_distance_um:
    :param gpu: bool
    :return:
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    # from ibllib.plots.figures import ephys_bad_channels
    # ephys_bad_channels(x, 30000, channel_labels[0], channel_labels[1])

    # we interpolate only noisy channels or dead channels (0: good), out of the brain channels are left
    bad_channels = gp.where(np.logical_or(channel_labels == 1, channel_labels == 2))[0]
    for i in bad_channels:
        # compute the weights to apply to neighbouring traces
        offset = gp.abs(x - x[i] + 1j * (y - y[i]))
        weights = gp.exp(-((offset / kriging_distance_um) ** p))
        weights[bad_channels] = 0
        weights[weights < 0.005] = 0
        weights = weights / gp.sum(weights)
        imult = gp.where(weights > 0.005)[0]
        if imult.size == 0:
            data[i, :] = 0
            continue
        data[i, :] = gp.matmul(weights[imult], data[imult, :])
    # from viewephys.gui import viewephys
    # f = viewephys(data.T, fs=1/30, h=h, title='interp2')
    return data


def _get_destripe_parameters(fs, butter_kwargs, k_kwargs, k_filter):
    """gets the default params for destripe. This is used for both the destripe fcn on a
    numpy array and the function that actuates on a cbin file"""
    if butter_kwargs is None:
        butter_kwargs = {"N": 3, "Wn": 300 / fs * 2, "btype": "highpass"}
    if k_kwargs is None:
        lagc = None if fs < 3000 else int(fs / 10)
        k_kwargs = {
            "ntr_pad": 60,
            "ntr_tap": 0,
            "lagc": lagc,
            "butter_kwargs": {"N": 3, "Wn": 0.01, "btype": "highpass"},
        }
    if k_filter:
        spatial_fcn = lambda dat: kfilt(dat, **k_kwargs)  # noqa
    else:
        spatial_fcn = lambda dat: car(dat, **k_kwargs)  # noqa
    return butter_kwargs, k_kwargs, spatial_fcn


def destripe(
    x,
    fs,
    h=None,
    neuropixel_version=1,
    butter_kwargs=None,
    k_kwargs=None,
    channel_labels=None,
    k_filter=True,
):
    """Super Car (super slow also...) - far from being set in stone but a good workflow example
    :param x: demultiplexed array (nc, ns)
    :param fs: sampling frequency
    :param neuropixel_version (optional): 1 or 2. Useful for the ADC shift correction. If None,
     no correction is applied
    :param channel_labels:
      None: (default) keep all channels
     OR (recommended to pre-compute)
        index array for the first axis of x indicating the selected traces.
     On a full workflow, one should scan sparingly the full file to get a robust estimate of the
     selection. If None, and estimation is done using only the current batch is provided for
     convenience but should be avoided in production.
      OR (only for quick display or as an example)
       True: deduces the bad channels from the data provided
    :param butter_kwargs: (optional, None) butterworth params, see the code for the defaults dict
    :param k_kwargs: (optional, None) K-filter params, see the code for the defaults dict
        can also be set to 'car', in which case the median accross channels will be subtracted
    :param k_filter (True): applies k-filter by default, otherwise, apply CAR.
    :return: x, filtered array
    """
    butter_kwargs, k_kwargs, spatial_fcn = _get_destripe_parameters(
        fs, butter_kwargs, k_kwargs, k_filter
    )
    if h is None:
        h = neuropixel.trace_header(version=neuropixel_version)
    if channel_labels is True:
        channel_labels, _ = detect_bad_channels(x, fs)
    # butterworth
    sos = scipy.signal.butter(**butter_kwargs, output="sos")
    x = scipy.signal.sosfiltfilt(sos, x)
    # channel interpolation
    # apply ADC shift
    if neuropixel_version is not None:
        x = fourier.fshift(x, h["sample_shift"], axis=1)
    # apply spatial filter only on channels that are inside of the brain
    if channel_labels is not None:
        x = interpolate_bad_channels(x, channel_labels, h["x"], h["y"])
        inside_brain = np.where(channel_labels != 3)[0]
        x[inside_brain, :] = spatial_fcn(x[inside_brain, :])  # apply the k-filter
    else:
        x = spatial_fcn(x)
    return x


def destripe_lfp(x, fs, channel_labels=None, **kwargs):
    """
    Wrapper around the destipe function with some default parameters to destripe the LFP band
    See help destripe function for documentation
    :param x:
    :param fs:
    :return:
    """
    kwargs["butter_kwargs"] = {"N": 3, "Wn": 2 / fs * 2, "btype": "highpass"}
    kwargs["k_filter"] = False
    if channel_labels is True:
        kwargs["channel_labels"], _ = detect_bad_channels(
            x, fs=fs, psd_hf_threshold=1.4
        )
    return destripe(x, fs, **kwargs)


def decompress_destripe_cbin(
    sr_file,
    output_file=None,
    h=None,
    wrot=None,
    append=False,
    save_sync_channels=False,
    butter_kwargs=None,
    dtype=np.float32,
    ns2add=0,
    batch_size=None,
    nprocesses=None,
    compute_rms=True,
    reject_channels=True,
    k_kwargs=None,
    k_filter=True,
    reader_kwargs=None,
    output_qc_path=None,
):
    """
    From a spikeglx Reader object, decompresses and apply ADC.
    Saves output as a flat binary file in float32 by default.
    Production version with optimized FFTs - requires pyfftw.

    :param sr_file: Path to a mtscomp-compressed .cbin file.
    :param output_file: (optional, defaults to .bin extension of the compressed bin file)
    :param h: (optional) neuropixel trace header. Dictionary with key 'sample_shift'
    :param wrot: (optional) whitening matrix [nc x nc] or amplitude scalar to apply to the output
    :param append: (optional, False) for chronic recordings, append to end of file
    :param save_sync_channels: (optional, False) saves non selected channels (synchronisation trace) in output
    :param butter_kwargs: Keyword arguments to scipy.signal.butter.
    By default, {'N': 3, 'Wn': 300 / sr.fs * 2, 'btype': 'highpass'}.
    :param dtype: (optional, np.float32) Datatype to cast output.
    :param ns2add: (optional) for kilosort, adds padding samples at the end of the file so the total
    number of samples is a multiple of the batchsize
    :param batch_size: (optional, 65536) batch size
    :param nprocesses: (optional) number of parallel processes to run, defaults to number or processes detected with joblib
     interp 3:outside of brain and discard
    :param reject_channels: (True) detects noisy or bad channels and interpolate them. Channels outside of the brain are left
     untouched
    :param k_kwargs: (None) arguments for the kfilter function
    :param reader_kwargs: (None) optional arguments for the spikeglx Reader instance
    :param k_filter: (True) Performs a k-filter - if False will do median common average referencing
    :param output_qc_path: (None) if specified, will save the QC rms in a different location than the output
    :return:
    """
    # confirm pyfftw installed.
    try:
        import pyfftw
    except ImportError:
        raise RuntimeError(
            "Failed to import pyfftw. decompress_destripe_cbin() requires pyfftw."
        )

    # create spikeglx Reader object with provided kwargs
    reader_kwargs = {} if reader_kwargs is None else reader_kwargs
    sr = spikeglx.Reader(sr_file, open=True, **reader_kwargs)

    # file I/O
    assert isinstance(sr_file, str) or isinstance(
        sr_file, Path
    ), "Input file must be str or Path."
    output_file = (
        sr.file_bin.with_suffix(".bin") if output_file is None else Path(output_file)
    )
    assert output_file != sr_file, "Output filename is identical to input filename."
    if append:
        assert (
            output_file.exists()
        ), f"Append mode specified but output file {output_file} does not exist."

    # important information for writing the output binary file
    nbytes = dtype(1).nbytes  # 4 for float32
    if append:
        # need to find the end of the file and the offset
        offset = Path(output_file).stat().st_size
    else:
        offset = 0
        # create empty output file
        open(output_file, "wb").close()

    # detect & interpolate bad channels and account for sr sync channel
    if reject_channels:
        channel_labels = detect_bad_channels_cbin(sr)
        channel_labels_file = output_file.parent.joinpath("channel_labels.npy")
        np.save(channel_labels_file, channel_labels.astype(int))
    # remove sync channel(s) by default
    nc_out = sr.nc if save_sync_channels else sr.nc - sr.nsync

    # get channel geometry from Reader
    h = sr.geometry if h is None else h
    ncv = h["sample_shift"].size  # number of channels

    # parallel processing parameters
    BATCH_SIZE = batch_size or 65536
    # max number of processes
    nprocesses = nprocesses or int(cpu_count() - cpu_count() / 4)
    CHUNK_SIZE = int(sr.ns / nprocesses)

    # Create samples taper
    SAMPLES_TAPER = 1024
    taper = np.r_[0, scipy.signal.windows.cosine((SAMPLES_TAPER - 1) * 2), 0]

    # Create Butterworth filter
    butter_kwargs, k_kwargs, spatial_fcn = _get_destripe_parameters(
        sr.fs, butter_kwargs, k_kwargs, k_filter
    )
    sos = scipy.signal.butter(**butter_kwargs, output="sos")

    # Create PyFFTW object
    win = pyfftw.empty_aligned((ncv, BATCH_SIZE), dtype="float32")
    WIN = pyfftw.empty_aligned((ncv, int(BATCH_SIZE / 2 + 1)), dtype="complex64")
    fft_object = pyfftw.FFTW(win, WIN, axes=(1,), direction="FFTW_FORWARD", threads=4)

    # Create dephasing array
    dephas = np.zeros((ncv, BATCH_SIZE), dtype=np.float32)
    dephas[:, 1] = 1.0
    DEPHAS = np.exp(
        1j * np.angle(fft_object(dephas)) * h["sample_shift"][:, np.newaxis]
    )

    # handle rms files, create new or append to existing
    if compute_rms:
        ap_rms_file = output_file.parent.joinpath("ap_rms.bin")
        ap_time_file = output_file.parent.joinpath("ap_time.bin")
        rms_nbytes = np.float32(1).nbytes

        if append:
            # compute offsets for appending to RMS files
            assert (
                ap_rms_file.exists()
            ), f"compute_rms and append flags were set but RMS file {ap_rms_file} does not exist."
            assert (
                ap_time_file.exists()
            ), f"compute_rms and append flags were set but time file {ap_time_file} does not exist."
            rms_offset = Path(ap_rms_file).stat().st_size
            time_offset = Path(ap_time_file).stat().st_size
            with open(ap_time_file, "rb") as tid:
                t = tid.read()
            time_data = np.frombuffer(t, dtype=np.float32)
            t0 = time_data[-1]
        else:
            # create empty RMS files
            rms_offset = 0
            time_offset = 0
            t0 = 0
            open(ap_rms_file, "wb").close()
            open(ap_time_file, "wb").close()

    def decompress_destripe_chunk(i_chunk):
        _sr = spikeglx.Reader(sr_file, **reader_kwargs)

        n_batch = int(np.ceil(i_chunk * CHUNK_SIZE / BATCH_SIZE))
        first_s = (BATCH_SIZE - SAMPLES_TAPER * 2) * n_batch

        # Find the maximum sample for each chunk
        max_s = _sr.ns if i_chunk == nprocesses - 1 else (i_chunk + 1) * CHUNK_SIZE
        # need to redefine this here to avoid 4 byte boundary error
        win = pyfftw.empty_aligned((ncv, BATCH_SIZE), dtype="float32")
        WIN = pyfftw.empty_aligned((ncv, int(BATCH_SIZE / 2 + 1)), dtype="complex64")
        fft_object = pyfftw.FFTW(
            win, WIN, axes=(1,), direction="FFTW_FORWARD", threads=4
        )
        ifft_object = pyfftw.FFTW(
            WIN, win, axes=(1,), direction="FFTW_BACKWARD", threads=4
        )

        fid = open(output_file, "r+b")
        if i_chunk == 0:
            fid.seek(offset)
        else:
            fid.seek(offset + ((first_s + SAMPLES_TAPER) * nc_out * nbytes))

        if compute_rms:
            aid = open(ap_rms_file, "r+b")
            tid = open(ap_time_file, "r+b")
            if i_chunk == 0:
                aid.seek(rms_offset)
                tid.seek(time_offset)
            else:
                aid.seek(rms_offset + (n_batch * ncv * rms_nbytes))
                tid.seek(time_offset + (n_batch * rms_nbytes))

        while True:
            last_s = np.minimum(BATCH_SIZE + first_s, _sr.ns)
            # Apply tapers
            chunk = _sr[first_s:last_s, :ncv].T
            chunk[:, :SAMPLES_TAPER] *= taper[:SAMPLES_TAPER]
            chunk[:, -SAMPLES_TAPER:] *= taper[SAMPLES_TAPER:]
            # Apply filters
            chunk = scipy.signal.sosfiltfilt(sos, chunk)
            # Find the indices to save
            ind2save = [SAMPLES_TAPER, BATCH_SIZE - SAMPLES_TAPER]
            if last_s == _sr.ns:
                # for the last batch just use the normal fft as the stencil doesn't fit
                chunk = fourier.fshift(chunk, s=h["sample_shift"])
                ind2save[1] = BATCH_SIZE
            else:
                # apply precomputed fshift of the proper length
                chunk = ifft_object(fft_object(chunk) * DEPHAS)
            if first_s == 0:
                # for the first batch save the start with taper applied
                ind2save[0] = 0
            # interpolate missing traces after the low-cut filter it's important to leave the
            # channels outside of the brain outside of the computation
            if reject_channels:
                chunk = interpolate_bad_channels(chunk, channel_labels, h["x"], h["y"])
                inside_brain = np.where(channel_labels != 3)[0]
                chunk[inside_brain, :] = spatial_fcn(
                    chunk[inside_brain, :]
                )  # apply the k-filter / CAR
            else:
                chunk = spatial_fcn(chunk)  # apply the k-filter / CAR

            # add back sync trace and save
            chunk = np.r_[chunk, _sr[first_s:last_s, ncv:].T].T

            # Compute rms - we get it before applying the whitening
            if compute_rms:
                ap_rms = utils.rms(chunk[:, :ncv], axis=0)
                ap_t = t0 + (first_s + (last_s - first_s - 1) / 2) / _sr.fs
                ap_rms.astype(np.float32).tofile(aid)
                ap_t.astype(np.float32).tofile(tid)

            # convert to normalised
            intnorm = 1 / _sr.sample2volts
            chunk = chunk[slice(*ind2save), :] * intnorm
            # apply the whitening matrix if necessary
            if wrot is not None:
                chunk[:, :ncv] = np.dot(chunk[:, :ncv], wrot)
            chunk[:, :nc_out].astype(dtype).tofile(fid)
            first_s += BATCH_SIZE - SAMPLES_TAPER * 2

            if last_s >= max_s:
                if last_s == _sr.ns:
                    if ns2add > 0:
                        np.tile(chunk[-1, :nc_out].astype(dtype), (ns2add, 1)).tofile(
                            fid
                        )
                fid.close()
                if compute_rms:
                    aid.close()
                    tid.close()
                break

    _ = Parallel(n_jobs=nprocesses)(
        delayed(decompress_destripe_chunk)(i) for i in range(nprocesses)
    )

    sr.close()

    # Here convert the ap_rms bin files to the ibl format and save
    if compute_rms:
        with open(ap_rms_file, "rb") as aid, open(ap_time_file, "rb") as tid:
            rms_data = aid.read()
            time_data = tid.read()
        time_data = np.frombuffer(time_data, dtype=np.float32)
        rms_data = np.frombuffer(rms_data, dtype=np.float32)
        assert rms_data.shape[0] == time_data.shape[0] * ncv
        rms_data = rms_data.reshape(time_data.shape[0], ncv)
        output_qc_path = output_qc_path or output_file.parent
        np.save(output_qc_path.joinpath("_iblqc_ephysTimeRmsAP.rms.npy"), rms_data)
        np.save(
            output_qc_path.joinpath("_iblqc_ephysTimeRmsAP.timestamps.npy"), time_data
        )


def decompress_destripe_cbin_snippet(
    sr_file,
    output_file,
    s_start,
    s_end,
    log=False,
):
    """
    Pared-down version of decompress_destripe_cbin suitable for extracting
    and destriping 1-10 minutes of data from an mtscomp compressed cbin file
    and saving to flat float32 binary. Optimized to run on a single 48-core node and not
    tested for > 10 minute snippets. Does not provide the customizable options of
    decompress_destripe_cbin.

    Uses a default Butterworth low-cut filter, applies ADC shifts, detects and interpolates
    bad channels, and applies a spatial filter.

    Runs in parallel on batches with a hardcoded size of 65536 samples. If `s_start`
    is greater than 1024 samples into the file, an extra batch is run to fill in the
    beginning of the traces using the 1024 samples prior to `s_start`.

    Recommended SLURM parameters:

    ```
    #!/bin/bash
    #SBATCH --job-name="destripe"
    #SBATCH --cpus-per-task=48
    #SBATCH --time=00:10:00
    #SBATCH --nodes=1
    ```

    :param sr_file: Path to a mtscomp compressed cbin file.
    :param output_file: Path to an output binary file.
    :s_start: Starting time in samples.
    :s_end: Ending time in samples.
    :log: (optional, False) Create logs for each job spawned under `destripe_logs`
        in the output file's directory.
    """
    # confirm pyfftw installed.
    try:
        import pyfftw
    except ImportError:
        raise RuntimeError(
            "Failed to import pyfftw. decompress_destripe_cbin() requires pyfftw."
        )

    # Setup dtype for writing to binary
    dtype = np.float32
    nbytes = dtype(1).itemsize

    # Create output file
    open(output_file, "wb").close()

    with spikeglx.Reader(sr_file) as sr:
        h = sr.geometry
        fs = sr.fs
        # Detect bad channels for interpolation
        channel_labels = detect_bad_channels_cbin(sr)
    num_channels = h["sample_shift"].size

    if log:
        log_dir = output_file.parent / "destripe_log"
        log_dir.mkdir(exist_ok=True)

    # Length of snippet
    s_length = s_end - s_start

    if s_length > sr.fs * 600:  # 10 minutes
        print(
            "WARNING: You are trying to destripe a snippet greater than 10 minutes in"
            "length, which may be inefficient and/or create too many jobs"
            "for your CPU."
        )

    #### PARALLEL PROCESSING
    SAMPLES_TAPER = 1024
    batch_size = 65536 - SAMPLES_TAPER * 2
    num_batches = s_length // batch_size + 1

    print(f"num_batches: {num_batches}")

    # assign batches per job
    # Timing / CPU heuristics
    if s_length > sr.fs * 60 * 5:  # >5 minutes
        num_batches_per_job = 6  # Will spawn 48 jobs for 10 minutes
    elif s_length > sr.fs * 60:  # >1 minute
        num_batches_per_job = 4  # Will spawn 36 jobs for 5 minutes
    elif s_length > batch_size:  # <= 1 minute
        num_batches_per_job = 1  # Will spawn 29 jobs for 1 minute
    else:
        raise RuntimeError("Cannot destripe less than batch size")

    num_jobs = num_batches // num_batches_per_job + 1

    print(f"num_jobs: {num_jobs}")

    ## assign batches per job
    batches_per_job = [[] for i in range(num_jobs)]
    for i in range(num_jobs):
        batches = [
            b
            for b in range(num_batches)
            if b // num_batches_per_job < i + 1 and b // num_batches_per_job >= i
        ]
        batches_per_job[i] = batches
        print(f"Job {i}: batches {batches}")

    ## assign start and end samples per batch
    ranges_per_batch = [[] for i in range(num_batches)]
    for i in range(num_batches):
        start = s_start + i * batch_size + SAMPLES_TAPER
        end = min(s_end, s_start + (i + 1) * batch_size + SAMPLES_TAPER)
        ranges_per_batch[i] = [start, end]
        print(f"Batch {i}: [{start}, {end-1}]. Length: {end - start + 1}")

    # Edge taper for each batch
    SAMPLES_TAPER = 1024
    taper = np.r_[0, scipy.signal.windows.cosine((SAMPLES_TAPER - 1) * 2), 0]

    # Create Butterworth filter
    butter_kwargs = None
    k_kwargs = None
    k_filter = True
    butter_kwargs, k_kwargs, spatial_fcn = _get_destripe_parameters(
        fs, butter_kwargs, k_kwargs, k_filter
    )
    sos = scipy.signal.butter(**butter_kwargs, output="sos")

    def run_job(job_id):
        print(f"Running job {job_id}")
        _sr = spikeglx.Reader(sr_file, open=True)
        fout = open(output_file, "r+b")

        batches = batches_per_job[job_id]
        if not batches:
            return
        seek_sample = ranges_per_batch[batches[0]][0]

        if log:
            logger = logging.getLogger(f"logger_{job_id}")
            logger.addHandler(
                logging.FileHandler(log_dir / f"log_{job_id}.txt", mode="w")
            )
            logger.setLevel("DEBUG")
            logger.debug(f"Job {job_id}")
            logger.debug(f"Working on batches {batches}")

        win = pyfftw.empty_aligned(
            (num_channels, batch_size + 2 * SAMPLES_TAPER), dtype="float32"
        )
        WIN = pyfftw.empty_aligned(
            (num_channels, int((batch_size + 2 * SAMPLES_TAPER) / 2 + 1)),
            dtype="complex64",
        )
        fft_object = pyfftw.FFTW(
            win, WIN, axes=(1,), direction="FFTW_FORWARD", threads=4
        )
        ifft_object = pyfftw.FFTW(
            WIN, win, axes=(1,), direction="FFTW_BACKWARD", threads=4
        )

        # Create dephasing array
        dephas = np.zeros(
            (num_channels, batch_size + 2 * SAMPLES_TAPER), dtype=np.float32
        )
        dephas[:, 1] = 1.0
        DEPHAS = np.exp(
            1j * np.angle(fft_object(dephas)) * h["sample_shift"][:, np.newaxis]
        )

        if job_id == 0:
            fout.seek(0)
        else:
            fout.seek((seek_sample - s_start) * num_channels * nbytes)

        def run_segment(start, end, batch_number):
            chunk = _sr[start - SAMPLES_TAPER: end + SAMPLES_TAPER, :num_channels].T
            # apply taper to edges
            chunk[:, :SAMPLES_TAPER] *= taper[:SAMPLES_TAPER]
            chunk[:, -SAMPLES_TAPER:] *= taper[SAMPLES_TAPER:]

            # Highpass temporal filter
            chunk = scipy.signal.sosfiltfilt(sos, chunk)

            # Fourier shift
            if batch_number == num_batches - 1:
                chunk = fourier.fshift(chunk, s=h["sample_shift"])
            else:
                chunk = ifft_object(fft_object(chunk) * DEPHAS)

            # interpolate bad channels
            chunk = interpolate_bad_channels(
                chunk, channel_labels, h["x"], h["y"]
            )
            inside_brain = np.where(channel_labels != 3)[0]

            # apply highpass spatial filter ignoring channels outside brain
            chunk[inside_brain, :] = spatial_fcn(chunk[inside_brain, :])

            # chop off tapers
            if batch_number == 0:
                # save tapered beginning
                ind2save = [0, batch_size + SAMPLES_TAPER]
            elif batch_number == num_batches - 1:
                # cut off the taper that was tacked on to the end
                # of the last batch when loaded
                ind2save = [SAMPLES_TAPER, -SAMPLES_TAPER]
            else:
                ind2save = [SAMPLES_TAPER, batch_size + SAMPLES_TAPER]

            # add back in sync
            chunk = np.r_[
                chunk, _sr[start - SAMPLES_TAPER: end + SAMPLES_TAPER, num_channels:].T
            ].T
            # normalize
            chunk = chunk[slice(*ind2save), :] * (1 / _sr.sample2volts)
            chunk[:, :num_channels].astype(np.float32).tofile(fout)

        for i, batch_number in enumerate(batches):
            try:
                start, end = ranges_per_batch[batch_number]
                run_segment(start, end, batch_number)
            except Exception:
                if log:
                    logger.exception(traceback.format_exc())
        # fill in beginning patch to avoid taper
        if job_id == 0 and s_start > SAMPLES_TAPER:
            # backfills the first 1024 samples by shifting window
            fout.seek(0)
            run_segment(s_start, s_start + batch_size, -1)

        fout.close()

    _ = Parallel(n_jobs=num_jobs)(delayed(run_job)(i) for i in range(num_jobs))


def detect_bad_channels(raw, fs, similarity_threshold=(-0.5, 1), psd_hf_threshold=None):
    """
    Bad channels detection for Neuropixel probes
    Labels channels
     0: all clear
     1: dead low coherence / amplitude
     2: noisy
     3: outside of the brain
    :param raw: [nc, ns]
    :param fs: sampling frequency
    :param similarity_threshold:
    :param psd_hf_threshold:
    :return: labels (numpy vector [nc]), xfeats: dictionary of features [nc]
    """

    def rneighbours(raw, n=1):  # noqa
        """
        Computes Pearson correlation with the sum of neighbouring traces
        :param raw: nc, ns
        :param n:
        :return:
        """
        nc = raw.shape[0]
        mixer = np.triu(np.ones((nc, nc)), 1) - np.triu(np.ones((nc, nc)), 1 + n)
        mixer += np.tril(np.ones((nc, nc)), -1) - np.tril(np.ones((nc, nc)), -n - 1)
        r = rcoeff(raw, np.matmul(raw.T, mixer).T)
        r[np.isnan(r)] = 0
        return r

    def detrend(x, nmed):
        """
        Subtract the trend from a vector
        The trend is a median filtered version of the said vector with tapering
        :param x: input vector
        :param nmed: number of points of the median filter
        :return: np.array
        """
        ntap = int(np.ceil(nmed / 2))
        xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]
        # assert np.all(xcorf[ntap:-ntap] == xcor)
        xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
        return x - xf

    def channels_similarity(raw, nmed=0):
        """
        Computes the similarity based on zero-lag crosscorrelation of each channel with the median
        trace referencing
        :param raw: [nc, ns]
        :param nmed:
        :return:
        """

        def fxcor(x, y):
            return scipy.fft.irfft(
                scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1]
            )

        def nxcor(x, ref):
            ref = ref - np.mean(ref)
            apeak = fxcor(ref, ref)[0]
            x = x - np.mean(x, axis=-1)[:, np.newaxis]  # remove DC component
            return fxcor(x, ref)[:, 0] / apeak

        ref = np.median(raw, axis=0)
        xcor = nxcor(raw, ref)

        if nmed > 0:
            xcor = detrend(xcor, nmed) + 1
        return xcor

    nc, _ = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    xcor = channels_similarity(raw)
    fscale, psd = scipy.signal.welch(raw * 1e6, fs=fs)  # units; uV ** 2 / Hz
    if psd_hf_threshold is None:
        # the LFP band data is obviously much stronger so auto-adjust the default threshold
        psd_hf_threshold = 1.4 if fs < 5000 else 0.02
    sos_hp = scipy.signal.butter(
        **{"N": 3, "Wn": 300 / fs * 2, "btype": "highpass"}, output="sos"
    )
    hf = scipy.signal.sosfiltfilt(sos_hp, raw)
    xcorf = channels_similarity(hf)

    xfeats = {
        "ind": np.arange(nc),
        "rms_raw": utils.rms(raw),  # very similar to the rms avfter butterworth filter
        "xcor_hf": detrend(xcor, 11),
        "xcor_lf": xcorf - detrend(xcorf, 11) - 1,
        "psd_hf": np.mean(psd[:, fscale > (fs / 2 * 0.8)], axis=-1),  # 80% nyquists
    }

    # make recommendation
    ichannels = np.zeros(nc)
    idead = np.where(similarity_threshold[0] > xfeats["xcor_hf"])[0]
    inoisy = np.where(
        np.logical_or(
            xfeats["psd_hf"] > psd_hf_threshold,
            xfeats["xcor_hf"] > similarity_threshold[1],
        )
    )[0]
    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency
    ioutside = np.where(xfeats["xcor_lf"] < -0.75)[0]
    if ioutside.size > 0 and ioutside[-1] == (nc - 1):
        a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
        ioutside = ioutside[a == np.max(a)]
        ichannels[ioutside] = 3

    # indices
    ichannels[idead] = 1
    ichannels[inoisy] = 2
    # from ibllib.plots.figures import ephys_bad_channels
    # ephys_bad_channels(x, 30000, ichannels, xfeats)
    return ichannels, xfeats


def detect_bad_channels_cbin(bin_file, n_batches=10, batch_duration=0.3, display=False):
    """
    Runs a ap-binary file scan to automatically detect faulty channels
    :param bin_file: full file path to the binary or compressed binary file from spikeglx
    :param n_batches: number of batches throughout the file (defaults to 10)
    :param batch_duration: batch length in seconds, defaults to 0.3
    :param display: if True will return a figure with features and an excerpt of the raw data
    :return: channel_labels: nc int array with 0:ok, 1:dead, 2:high noise, 3:outside of the brain
    """
    sr = (
        bin_file if isinstance(bin_file, spikeglx.Reader) else spikeglx.Reader(bin_file)
    )
    nc = sr.nc - sr.nsync
    channel_labels = np.zeros((nc, n_batches))
    # loop over the file and take the mode of detections
    for i, t0 in enumerate(np.linspace(0, sr.rl - batch_duration, n_batches)):
        sl = slice(int(t0 * sr.fs), int((t0 + batch_duration) * sr.fs))
        channel_labels[:, i], _xfeats = detect_bad_channels(sr[sl, :nc].T, fs=sr.fs)
        if i == 0:  # init the features dictionary if necessary
            xfeats = {k: np.zeros((nc, n_batches)) for k in _xfeats}
        for k in xfeats:
            xfeats[k][:, i] = _xfeats[k]
    # the features are averaged  so there may be a discrepancy between the mode and applying
    # the thresholds to the average of the features - the goal of those features is for display only
    xfeats_med = {k: np.median(xfeats[k], axis=-1) for k in xfeats}
    channel_flags, _ = scipy.stats.mode(channel_labels, axis=1)
    if display:
        raw = sr[sl, :nc].TO
        from ibllib.plots.figures import ephys_bad_channels

        ephys_bad_channels(raw, sr.fs, channel_flags, xfeats_med)
    return channel_flags


def resample_denoise_lfp_cbin(lf_file, RESAMPLE_FACTOR=10, output=None):
    """
    Downsamples an LFP file and apply dstriping
    ```
    nc = 384
    ns = int(lf_file_out.stat().st_size / nc / 4)
    sr_ = spikeglx.Reader(lf_file_out, nc=nc, fs=sr.fs / RESAMPLE_FACTOR, ns=ns,  dtype=np.float32)
    ```
    :param lf_file:
    :param RESAMPLE_FACTOR:
    :param output: Path
    :return: None
    """

    output = output or Path(lf_file).parent.joinpath("lf_resampled.bin")
    sr = spikeglx.Reader(lf_file)
    wg = utils.WindowGenerator(ns=sr.ns, nswin=65536, overlap=1024)
    cflags = detect_bad_channels_cbin(lf_file)

    c = 0
    with open(output, "wb") as f:
        for first, last in wg.firstlast:
            butter_kwargs = {
                "N": 3,
                "Wn": np.array([2, 200]) / sr.fs * 2,
                "btype": "bandpass",
            }
            sos = scipy.signal.butter(**butter_kwargs, output="sos")
            raw = sr[first:last, : -sr.nsync]
            raw = scipy.signal.sosfiltfilt(sos, raw, axis=0)
            destripe = destripe_lfp(raw.T, fs=sr.fs, channel_labels=cflags)
            # viewephys(raw.T, fs=sr.fs, title='raw')
            # viewephys(destripe, fs=sr.fs, title='destripe')
            rsamp = scipy.signal.decimate(
                destripe, RESAMPLE_FACTOR, axis=1, ftype="fir"
            ).T
            # viewephys(rsamp, fs=sr.fs / RESAMPLE_FACTOR, title='rsamp')
            first_valid = 0 if first == 0 else int(wg.overlap / 2 / RESAMPLE_FACTOR)
            last_valid = (
                rsamp.shape[0]
                if last == sr.ns
                else int(rsamp.shape[0] - wg.overlap / 2 / RESAMPLE_FACTOR)
            )
            rsamp = rsamp[first_valid:last_valid, :]
            c += rsamp.shape[0]
            print(first, last, last - first, first_valid, last_valid, c)
            rsamp.astype(np.float32).tofile(f)
    # first, last = (500, 550)
    # viewephys(sr[int(first * sr.fs) : int(last * sr.fs), :-sr.nsync].T, sr.fs, title='orig')
    # viewephys(sr_[int(first * sr_.fs):int(last * sr_.fs), :].T, sr_.fs, title='rsamp')


def stack(data, word, fcn_agg=np.nanmean, header=None):
    """
    Stack numpy array traces according to the word vector
    :param data: (ntr, ns) numpy array of sample values
    :param word: (ntr) label according to which the traces will be aggregated (usually cdp)
    :param header: dictionary of vectors (ntr): header labels, will be aggregated as average
    :param fcn_agg: function, defaults to np.mean but could be np.sum or np.median
    :return: stack (ntr_stack, ns): aggregated numpy array
             header ( ntr_stack): aggregated header. If no header is provided, fold of coverage
    """
    (ntr, ns) = data.shape
    group, uinds, fold = np.unique(word, return_inverse=True, return_counts=True)
    ntrs = group.size

    stack = np.zeros((ntrs, ns), dtype=data.dtype)
    for sind in np.arange(ntrs):
        i2stack = sind == uinds
        stack[sind, :] = fcn_agg(data[i2stack, :], axis=0)

    # aggregate the header using pandas
    if header is None:
        hstack = fold
    else:
        header["stack_word"] = word
        dfh = pd.DataFrame(header).groupby("stack_word")
        hstack = dfh.aggregate("mean").to_dict(orient="series")
        hstack = {k: hstack[k].values for k in hstack.keys()}
        hstack["fold"] = fold

    return stack, hstack


def current_source_density(lfp, h, method="diff", sigma=1 / 3):
    """
    Compute the current source density (CSD) of a given LFP signal recorded on neuropixel 1 or 2
    :param data: LFP signal (n_channels, n_samples)
    :param h: trace header dictionary
    :param method: diff (straight double difference) or kernel CSD (needs the KCSD python package)
    :param sigma: conductivity, defaults to 1/3 S.m-1
    :return:
    """
    csd = np.zeros(lfp.shape, dtype=np.float64) * np.NAN
    xy = h["x"] + 1j * h["y"]
    for col in np.unique(h["col"]):
        ind = np.where(h["col"] == col)[0]
        isort = np.argsort(h["row"][ind])
        itr = ind[isort]
        dx = np.median(np.diff(np.abs(xy[itr])))
        if method == "diff":
            csd[itr[1:-1], :] = (
                np.diff(lfp[itr, :].astype(np.float64), n=2, axis=0) / dx**2 * sigma
            )
        elif method == "kcsd":
            from kcsd import KCSD1D

            # here we could eventually expose the KCSD kwargs
            csd[itr, :] = KCSD1D(
                h["y"][itr, np.newaxis],
                lfp[itr, :],
                h=np.median(
                    np.diff(h["y"][ind])
                ),  # this seems to work well with the current intertrace
                sigma=sigma,
                xmin=np.min(h["y"][itr]),
                xmax=np.max(h["y"][itr]),
                gdx=np.ceil((np.max(h["y"][itr]) - np.min(h["y"][itr])) / itr.size),
                lambd=0.0,
                R_init=5.0,
                n_src_init=10000,
                src_type="gauss",
            ).values("CSD")
    return csd
