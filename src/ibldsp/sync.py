from pathlib import Path
import uuid

import numpy as np
import scipy.interpolate

import one.alf.io as alfio

import spikeglx
import ibldsp.utils


def extract_spikeglx_sync(raw_ephys_apfile, output_path=None, save=False, parts=""):
    """
    Extracts sync.times, sync.channels and sync.polarities from binary ephys dataset

    :param raw_ephys_apfile: bin file containing ephys data or spike
    :param output_path: output directory
    :param save: bool write to disk only if True
    :param parts: string or list of strings that will be appended to the filename before extension
    :return:
    """
    # handles input argument: support ibllib.io.spikeglx.Reader, str and pathlib.Path
    if isinstance(raw_ephys_apfile, spikeglx.Reader):
        sr = raw_ephys_apfile
    else:
        raw_ephys_apfile = Path(raw_ephys_apfile)
        sr = spikeglx.Reader(raw_ephys_apfile)

    SYNC_BATCH_SIZE_SECS = 100

    if not (opened := sr.is_open):
        sr.open()
    # if no output, need a temp folder to swap for big files
    if output_path is None:
        output_path = sr.file_bin.parent
    file_ftcp = Path(output_path).joinpath(
        f"fronts_times_channel_polarity{uuid.uuid4()}.bin"
    )

    # loop over chunks of the raw ephys file
    wg = ibldsp.utils.WindowGenerator(
        sr.ns, int(SYNC_BATCH_SIZE_SECS * sr.fs), overlap=1
    )
    fid_ftcp = open(file_ftcp, "wb")
    for sl in wg.slice:
        ss = sr.read_sync(sl)
        ind, fronts = ibldsp.utils.fronts(ss, axis=0)
        # a = sr.read_sync_analog(sl)
        sav = np.c_[(ind[0, :] + sl.start) / sr.fs, ind[1, :], fronts.astype(np.double)]
        sav.tofile(fid_ftcp)
    # close temp file, read from it and delete
    fid_ftcp.close()
    tim_chan_pol = np.fromfile(str(file_ftcp))
    tim_chan_pol = tim_chan_pol.reshape((int(tim_chan_pol.size / 3), 3))
    file_ftcp.unlink()
    sync = {
        "times": tim_chan_pol[:, 0],
        "channels": tim_chan_pol[:, 1],
        "polarities": tim_chan_pol[:, 2],
    }
    # If opened Reader was passed into function, leave open
    if not opened:
        sr.close()
    if save:
        out_files = alfio.save_object_npy(
            output_path, sync, "sync", namespace="spikeglx", parts=parts
        )
        return sync, out_files
    else:
        return sync


def sync_timestamps(tsa, tsb, tbin=0.1, return_indices=False, linear=False):
    """
    Sync two arrays of time stamps
    :param tsa: vector of timestamps
    :param tsb: vector of timestamps
    :param tbin: time bin length
    :param return_indices (bool), if True returns 2 sets of indices for tsa and tsb with
    :param linear: (bool) if True, restricts the fit to linear
    identified matches
    :return:
     function: interpolation function such as fnc(tsa) = tsb
     float: drift in ppm
     numpy array: of indices ia
     numpy array: of indices ib
    """

    def _interp_fcn(tsa, tsb, ib, linear=linear):
        # now compute the bpod/fpga drift and precise time shift
        ab = np.polyfit(tsa[ib >= 0], tsb[ib[ib >= 0]] - tsa[ib >= 0], 1)
        drift_ppm = ab[0] * 1e6
        if linear:
            fcn_a2b = lambda x: x * (1 + ab[0]) + ab[1]  # noqa
        else:
            fcn_a2b = scipy.interpolate.interp1d(
                tsa[ib >= 0], tsb[ib[ib >= 0]], fill_value="extrapolate"
            )
        return fcn_a2b, drift_ppm

    # assert sorted inputs
    tmin = np.min([np.min(tsa), np.min(tsb)])
    tmax = np.max([np.max(tsa), np.max(tsb)])
    # brute force correlation to get an estimate of the delta_t between series
    x = np.zeros(int(np.ceil(tmax - tmin) / tbin))
    y = np.zeros_like(x)
    x[np.int32(np.floor((tsa - tmin) / tbin))] = 1
    y[np.int32(np.floor((tsb - tmin) / tbin))] = 1
    delta_t = (
        ibldsp.utils.parabolic_max(scipy.signal.correlate(x, y, mode="full"))[0]
        - x.shape[0]
        + 1
    ) * tbin
    # do a first assignment at a DT threshold
    ib = np.zeros(tsa.shape, dtype=np.int32) - 1
    threshold = tbin
    for m in np.arange(tsa.shape[0]):
        dt = np.abs(tsa[m] - delta_t - tsb)
        inds = np.where(dt < threshold)[0]
        if inds.size == 1:
            ib[m] = inds[0]
        elif inds.size > 1:
            candidates = inds[~np.isin(inds, ib[:m])]
            if candidates.size == 1:
                ib[m] = candidates[0]
            elif candidates.size > 1:
                ib[m] = inds[np.argmin(dt[inds])]

    fcn_a2b, _ = _interp_fcn(tsa, tsb, ib)
    # do a second assignment - this time a full matrix of candidate matches is computed
    # the most obvious matches are assigned first and then one by one
    iamiss = np.where(ib < 0)[0]
    ibmiss = np.setxor1d(np.arange(tsb.size), ib[ib >= 0])
    dt = np.abs(fcn_a2b(tsa[iamiss]) - tsb[ibmiss][:, np.newaxis])
    dt[dt > tbin] = np.nan
    while ~np.all(np.isnan(dt)):
        _b, _a = np.unravel_index(np.nanargmin(dt), dt.shape)
        ib[iamiss[_a]] = ibmiss[_b]
        dt[:, _a] = np.nan
        dt[_b, :] = np.nan
    fcn_a2b, drift_ppm = _interp_fcn(tsa, tsb, ib, linear=linear)

    if return_indices:
        return fcn_a2b, drift_ppm, np.where(ib >= 0)[0], ib[ib >= 0]
    else:
        return fcn_a2b, drift_ppm
