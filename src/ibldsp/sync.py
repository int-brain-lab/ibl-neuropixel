import logging
from pathlib import Path
import uuid

import numpy as np
import scipy.interpolate

import one.alf.io as alfio

import spikeglx
import ibldsp.utils

_logger = logging.getLogger("ibllib")

SYNC_QC_CRITERIA = (
    "sync_present",
    "same_duration",
    "same_n_edges",
    "regular",
    "complete",
)


def spikeglx_sync_qc(
    md_nidq, md_probe, t_nidq, t_probe, half_period=None, tol=0.01, label=""
):
    """
    Check whether the imec sync edges of a probe and of the nidq can be paired one to one.

    The imec sync is a periodic square wave recorded on both streams. Pairing edge `i` of the probe with
    edge `i` of the nidq is only safe if both files hold the same, complete, regular train of edges. Five
    criteria are evaluated, each one logs a warning when it fails:

    0. `sync_present`: at least two edges on both streams.
    1. `same_duration`: `|probe.fileTimeSecs - nidq.fileTimeSecs| < h / 2`.
    2. `same_n_edges`: both streams hold the same number of edges.
    3. `regular`: every interval between consecutive edges is within `[h / 2, 3 h / 2]`, on both streams.
    4. `complete`: the first edge is within `h + tol` of the file start and the last edge within `h + tol`
       of the file end, on both streams: no edge is missing at either end of the file.

    `h` is the sync half-period, `syncSourcePeriod / 2` in the meta-data.

    Parameters
    ----------
    md_nidq : dict
        nidq meta-data, as returned by `spikeglx.read_meta_data`.
    md_probe : dict
        Probe ap meta-data, as returned by `spikeglx.read_meta_data`.
    t_nidq : numpy.ndarray
        Times (s) of the imec sync edges (rising and falling) on the nidq, from the start of the nidq file.
    t_probe : numpy.ndarray
        Times (s) of the imec sync edges (rising and falling) on the probe, from the start of the probe file.
    half_period : float, optional
        Sync half-period `h` (s). Defaults to half the nidq `syncSourcePeriod`, or 0.5 s if absent.
    tol : float, optional
        Tolerance (s) added to `h` for the `complete` criterion, default 0.01.
    label : str, optional
        Prefix of the warning messages, for example the session path and probe name.

    Returns
    -------
    dict
        One boolean per criterion in `SYNC_QC_CRITERIA`, `passed` (all criteria pass) and the measured values:
        `half_period`, `dT` (probe minus nidq `fileTimeSecs`, s), `dn_edges` (probe minus nidq edge count),
        `k` (`dT / h` rounded: number of half-periods by which the durations differ), and per stream
        (`<s>` in `nidq`, `probe`): `<s>_n_edges`, `<s>_min_interval`, `<s>_max_interval`, `<s>_head`
        (time of the first edge) and `<s>_tail` (`fileTimeSecs` minus time of the last edge), in s.
    """
    t = {
        "nidq": np.asarray(t_nidq, dtype=float),
        "probe": np.asarray(t_probe, dtype=float),
    }
    md = {"nidq": md_nidq, "probe": md_probe}
    h = half_period or float(md_nidq.get("syncSourcePeriod", 1.0)) / 2
    out = {"half_period": h}
    for s, ts in t.items():
        dt = np.diff(ts)
        out[f"{s}_n_edges"] = ts.size
        out[f"{s}_min_interval"] = dt.min() if dt.size else np.nan
        out[f"{s}_max_interval"] = dt.max() if dt.size else np.nan
        out[f"{s}_head"] = ts[0] if ts.size else np.nan
        out[f"{s}_tail"] = float(md[s]["fileTimeSecs"]) - ts[-1] if ts.size else np.nan
    out["dT"] = float(md_probe["fileTimeSecs"]) - float(md_nidq["fileTimeSecs"])
    out["k"] = int(np.round(out["dT"] / h))
    out["dn_edges"] = out["probe_n_edges"] - out["nidq_n_edges"]
    # NaN comparisons are False: criteria 3 and 4 fail when the sync is absent
    out["sync_present"] = out["nidq_n_edges"] >= 2 and out["probe_n_edges"] >= 2
    out["same_duration"] = abs(out["dT"]) < h / 2
    out["same_n_edges"] = out["dn_edges"] == 0
    out["regular"] = all(
        out[f"{s}_min_interval"] >= h / 2 and out[f"{s}_max_interval"] <= 3 * h / 2
        for s in t
    )
    out["complete"] = all(
        out[f"{s}_head"] <= h + tol and out[f"{s}_tail"] <= h + tol for s in t
    )
    out["passed"] = all(out[c] for c in SYNC_QC_CRITERIA)

    messages = {
        "sync_present": f"imec sync edges missing: nidq {out['nidq_n_edges']}, probe {out['probe_n_edges']}",
        "same_duration": f"probe and nidq file durations differ by {out['dT']:.3f} s ({out['k']} half-periods)",
        "same_n_edges": f"probe and nidq edge counts differ by {out['dn_edges']} "
        f"(nidq {out['nidq_n_edges']}, probe {out['probe_n_edges']})",
        "regular": "irregular sync edge intervals (s): "
        + ", ".join(
            f"{s} [{out[f'{s}_min_interval']:.3f}, {out[f'{s}_max_interval']:.3f}]"
            for s in t
        ),
        "complete": "sync edges missing at the file start or end (s): "
        + ", ".join(
            f"{s} head {out[f'{s}_head']:.3f} tail {out[f'{s}_tail']:.3f}" for s in t
        ),
    }
    for c in SYNC_QC_CRITERIA:
        if not out[c]:
            _logger.warning(f"{label} {messages[c]}".strip())
    return out


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
