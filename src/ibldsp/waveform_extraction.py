import logging

import scipy
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
from joblib import Parallel, delayed, cpu_count

import neuropixel
import spikeglx
from ibldsp.voltage import detect_bad_channels, interpolate_bad_channels, car
from ibldsp.fourier import fshift
from ibldsp.utils import make_channel_index

logger = logging.getLogger(__name__)


def extract_wfs_array(
    arr,
    df,
    channel_neighbors,
    trough_offset=42,
    spike_length_samples=128,
    add_nan_trace=False,
    verbose=False,
):
    """
    Extract waveforms at specified samples and peak channels
    as a stack.

    :param arr: Array of traces. (samples, channels). The last trace of the array should be a
        row of non-data NaNs. If this has not been added set the `add_nan_trace` flag.
    :param df: df containing "sample" and "peak_channel" columns.
    :param channel_neighbors: Channel neighbor matrix (384x384)
    :param trough_offset: Number of samples to include before peak.
    (defaults to 42)
    :param spike_length_samples: Total length of wf in samples.
    (defaults to 128)
    :param add_nan_trace: Whether to add a row of `NaN`s as the last trace.
        (If False, code assumes this has already been added)
    """
    # This is to do fast index assignment to assign missing channels (out of the probe) to NaN
    if add_nan_trace:
        newcol = np.empty((arr.shape[0], 1))
        newcol[:] = np.nan
        arr = np.hstack([arr, newcol])

    # check that the spike window is included in the recording:
    last_idx = df["sample"].iloc[-1]
    assert (
        last_idx + (spike_length_samples - trough_offset) < arr.shape[0]
    ), f"Spike index {last_idx} extends past end of recording ({arr.shape[0]} samples)."

    nwf = len(df)

    # Get channel indices
    cind = channel_neighbors[df["peak_channel"].to_numpy()]

    # Get sample indices
    sind = df["sample"].to_numpy()[:, np.newaxis] + (
        np.arange(spike_length_samples) - trough_offset
    )
    nchan = cind.shape[1]

    wfs = np.zeros((nwf, spike_length_samples, nchan), arr.dtype)
    fun = range
    if verbose:
        try:
            from tqdm import trange

            fun = trange
        except ImportError:
            pass
    for i in fun(nwf):
        wfs[i, :, :] = arr[sind[i], :][:, cind[i]]

    return wfs.swapaxes(1, 2), cind, trough_offset


def _get_channel_labels(sr, num_snippets=20, verbose=True):
    """
    Given a spikeglx Reader object, samples `num_snippets` 1-second
    segments of the recording and returns the median channel labels
    across the segments as an array of size (nc,).
    """
    if verbose:
        from tqdm import trange

    # for most of recordings we take 100 secs left and right but account for recordings smaller
    buffer_left_right = np.minimum(100, sr.rl * 0.03)
    start = (
        np.linspace(buffer_left_right, int(sr.rl) - buffer_left_right, num_snippets)
        * sr.fs
    ).astype(int)
    end = start + int(sr.fs)

    _channel_labels = np.zeros((384, num_snippets), int)

    for i in trange(num_snippets):
        s0 = start[i]
        s1 = end[i]
        arr = sr[s0:s1, : -sr.nsync].T
        _channel_labels[:, i] = detect_bad_channels(arr, fs=30_000)[0]

    channel_labels = scipy.stats.mode(_channel_labels, axis=1, keepdims=True)[0].T

    return channel_labels


def _make_wfs_table(
    sr,
    spike_times,
    spike_clusters,
    spike_channels,
    max_wf=256,
    trough_offset=42,
    spike_length_samples=128,
):
    """
    Given a recording `sr` and spike detections, pick up to `max_wf`
    waveforms uniformly for each unit and return their times, peak channels,
    and unit assignments.

    :return: wf_flat, unit_ids Dataframe of waveform information and unit ids.
    """
    # exclude spikes without a buffer on either end
    # of recording
    allowed_idx = (spike_times > trough_offset) & (
        spike_times < sr.ns - (spike_length_samples - trough_offset)
    )

    rng = np.random.default_rng(seed=2024)  # numpy 1.23.5

    unit_ids = np.unique(spike_clusters)
    nu = unit_ids.shape[0]

    # this array contains the (up to) max_wf *indices* of the wfs
    # we are going to extract for that unit
    unit_wf_idx = np.zeros((nu, max_wf), int)
    unit_nspikes = np.zeros(nu, int)
    for i, u in enumerate(unit_ids):
        u_spikeidx = np.where((spike_clusters == u) & allowed_idx)[0]
        nspikes = u_spikeidx.shape[0]
        unit_nspikes[i] = nspikes
        # uniformly select up to 500 spikes
        u_wf_idx = rng.choice(u_spikeidx, min(max_wf, nspikes))
        unit_wf_idx[u, : min(max_wf, nspikes)] = u_wf_idx

    # all wf indices in order
    wf_idx = np.sort(unit_wf_idx.flatten())
    # remove initial zeros
    wf_idx = wf_idx[np.nonzero(wf_idx)[0][0]:]

    # get sample times, clusters, channels

    wf_flat = pd.DataFrame(
        {
            "index": np.arange(wf_idx.shape[0]),
            "sample": spike_times[wf_idx].astype(int),
            "cluster": spike_clusters[wf_idx].astype(int),
            "peak_channel": spike_channels[wf_idx].astype(int),
        }
    )

    return wf_flat, unit_ids


def write_wfs_chunk(
    i_chunk,
    cbin,
    wfs_fn,
    mmap_shape,
    geom_dict,
    channel_labels,
    channel_neighbors,
    wf_flat,
    sr_sl,
    chunksize_samples,
    trough_offset,
    spike_length_samples,
):
    """
    Parallel job to extract waveforms from chunk `i_chunk` of a recording `sr` and
    write them to the correct spot in the output .npy file `wfs_fn`.
    """
    if len(wf_flat) == 0:
        return

    my_sr = spikeglx.Reader(cbin)
    s0, s1 = sr_sl

    wfs_mmap = open_memmap(wfs_fn, shape=mmap_shape, mode="r+", dtype=np.float32)

    # create filters
    butter_kwargs = {"N": 3, "Wn": 300 / my_sr.fs * 2, "btype": "highpass"}
    sos = scipy.signal.butter(**butter_kwargs, output="sos")
    k_kwargs = {
        "ntr_pad": 60,
        "ntr_tap": 0,
        "lagc": int(my_sr.fs / 10),
        "butter_kwargs": {"N": 3, "Wn": 0.01, "btype": "highpass"},
    }
    car_func = lambda dat: car(dat, **k_kwargs)  # noqa: E731

    if i_chunk == 0:
        offset = 0
    else:
        offset = trough_offset

    sample = wf_flat["sample"].astype(int) + offset - i_chunk * chunksize_samples
    peak_channel = wf_flat["peak_channel"]

    df = pd.DataFrame({"sample": sample, "peak_channel": peak_channel})

    snip = my_sr[
        s0 - offset: s1 + spike_length_samples - trough_offset, : -my_sr.nsync
    ]
    snip0 = interpolate_bad_channels(
        fshift(
            scipy.signal.sosfiltfilt(sos, snip.T), geom_dict["sample_shift"], axis=1
        ),
        channel_labels,
        geom_dict["x"],
        geom_dict["y"],
    )
    # car
    snip1 = np.full((my_sr.nc, snip0.shape[1]), np.nan)
    snip1[:-1, :] = car_func(snip0)
    wfs_mmap[wf_flat["index"], :, :] = extract_wfs_array(
        snip1.T, df, channel_neighbors
    )[0]
    wfs_mmap.flush()


def extract_wfs_cbin(
    cbin_file,
    output_dir,
    spike_times,
    spike_clusters,
    spike_channels,
    h=None,
    max_wf=256,
    trough_offset=42,
    spike_length_samples=128,
    chunksize_t=0.1,
    nprocesses=None,
):
    """
    Given a cbin file and locations of spikes, extract waveforms for each unit, compute
    the templates, and save the results in `output_path`. The waveforms come from chunks
    of raw data which are phase-corrected to account for the ADC, high-pass filtered in
    time with an order 3 Butterworth filter with a 300Hz cutoff, and a common-average
    reference procedure is applied in the spatial dimension.

    The following files will be generated:
    - waveforms.traces.npy: `(num_units, max_wf, nc, spike_length_samples)`
        This file contains the lightly processed waveforms indexed by cluster in the first
        dimension. By default `max_wf=256, nc=40, spike_length_samples=128`.

    - waveforms.templates.npy: `(num_units, nc, spike_length_samples)`
        This file contains the median across individual waveforms for each unit.

    - waveforms.channels.npz: `(num_units * max_wf, nc)`
        The i'th row contains the ordered indices of the `nc`-channel neighborhood used
        to extract the i'th waveform. A NaN means the waveform is missing because the
        unit it was supposed to come from has less than `max_wf` spikes total in the
        recording.

    - waveforms.table.pqt: `num_units * max_wf` rows
        For each waveform, gives the absolute sample number from the recording (i.e.
        where to find it in `spikes.samples`), peak channel, cluster, and linear index.
        A row of NaN's implies that the waveform is missing because the unit is was supposed
        to come from has less than `max_wf` spikes total.
    """
    if h is None:
        h = neuropixel.trace_header()

    nprocesses = nprocesses or int(cpu_count() / 2)

    sr = spikeglx.Reader(cbin_file)

    chunksize_samples = int(chunksize_t * 30_000)
    s0_arr = np.arange(0, sr.ns, chunksize_samples)
    s1_arr = s0_arr + chunksize_samples
    s1_arr[-1] = sr.ns

    wf_flat, unit_ids = _make_wfs_table(
        sr,
        spike_times,
        spike_clusters,
        spike_channels,
        max_wf,
        trough_offset,
        spike_length_samples,
    )
    num_chunks = s0_arr.shape[0]

    logger.info(f"Chunk size: {chunksize_t}")
    logger.info(f"Num chunks: {num_chunks}")

    logger.info("Running channel detection")
    channel_labels = _get_channel_labels(sr)

    nwf = len(wf_flat)
    nu = unit_ids.shape[0]
    logger.info(f"Extracting {nwf} waveforms from {nu} units")

    #  get channel geometry
    geom = np.c_[h["x"], h["y"]]
    channel_neighbors = make_channel_index(geom)
    nc = channel_neighbors.shape[1]

    int_fn = output_dir.joinpath("_wf_extract_intermediate.npy")
    wfs = open_memmap(
        int_fn, mode="w+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
    )

    slices = [
        slice(*(np.searchsorted(wf_flat["sample"], [s0_arr[i], s1_arr[i]]).astype(int)))
        for i in range(num_chunks)
    ]

    _ = Parallel(n_jobs=nprocesses)(
        delayed(write_wfs_chunk)(
            i,
            cbin_file,
            int_fn,
            wfs.shape,
            h,
            channel_labels,
            channel_neighbors,
            wf_flat.iloc[slices[i]],
            (s0_arr[i], s1_arr[i]),
            chunksize_samples,
            trough_offset,
            spike_length_samples,
        )
        for i in range(num_chunks)
    )

    traces_fn = output_dir.joinpath("waveforms.traces.npy")
    templates_fn = output_dir.joinpath("waveforms.templates.npy")
    table_fn = output_dir.joinpath("waveforms.table.pqt")
    channels_fn = output_dir.joinpath("waveforms.channels.npz")

    # rearrange and save traces by unit
    wfs_templates = np.full((nu, nc, spike_length_samples), np.nan, dtype=np.float32)
    print("Computing templates")
    for i, u in enumerate(unit_ids):
        idx = np.where(wf_flat["cluster"] == u)[0]
        nwf_u = idx.shape[0]
        wfs = open_memmap(
            int_fn, mode="r+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
        )
        traces_by_unit = open_memmap(
            traces_fn,
            mode="w+",
            shape=(nu, max_wf, nc, spike_length_samples),
            dtype=np.float16,
        )
        traces_by_unit[i, : min(max_wf, nwf_u), :, :] = wfs[idx].astype(np.float16)
        traces_by_unit.flush()
        wfs_templates[i, :, :] = np.nanmedian(wfs[idx], axis=0)

    # cleanup intermediate file
    int_fn.unlink()

    # save waveforms and templates
    np.save(templates_fn, wfs_templates)  # waveforms.templates.npy

    # save dataframe
    wf_flat.to_parquet(table_fn)

    # save channel map for each waveform
    peak_channels = wf_flat["peak_channel"].to_numpy()
    chan_map = channel_neighbors[peak_channels, :]
    np.savez(channels_fn, chan_map)
