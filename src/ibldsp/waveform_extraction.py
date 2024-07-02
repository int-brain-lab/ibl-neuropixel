import logging

import scipy
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
from joblib import Parallel, delayed, cpu_count

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
    spike_samples,
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
    allowed_idx = (spike_samples > trough_offset) & (
        spike_samples < sr.ns - (spike_length_samples - trough_offset)
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
        u_wf_idx = rng.choice(u_spikeidx, min(max_wf, nspikes), replace=False)
        unit_wf_idx[i, : min(max_wf, nspikes)] = u_wf_idx

    # all wf indices in order
    wf_idx = np.sort(unit_wf_idx.flatten())
    # remove initial zeros
    wf_idx = wf_idx[np.nonzero(wf_idx)[0][0]:]

    # get sample times, clusters, channels
    wf_flat = pd.DataFrame(
        {
            "index": np.arange(wf_idx.shape[0]),
            "sample": spike_samples[wf_idx].astype(int),
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
    reader_kwargs,
    preprocess,
):
    """
    Parallel job to extract waveforms from chunk `i_chunk` of a recording `sr` and
    write them to the correct spot in the output .npy file `wfs_fn`.
    """
    if len(wf_flat) == 0:
        return

    my_sr = spikeglx.Reader(cbin, **reader_kwargs)
    s0, s1 = sr_sl

    wfs_mmap = open_memmap(wfs_fn, shape=mmap_shape, mode="r+", dtype=np.float32)

    if i_chunk == 0:
        offset = 0
    else:
        offset = trough_offset

    sample = wf_flat["sample"].astype(int) + offset - i_chunk * chunksize_samples
    peak_channel = wf_flat["peak_channel"]

    df = pd.DataFrame({"sample": sample, "peak_channel": peak_channel})

    snip = my_sr[
        s0 - offset:s1 + spike_length_samples - trough_offset, :-my_sr.nsync
    ]

    if not preprocess:
        wfs_mmap[wf_flat["index"], :, :] = extract_wfs_array(
            snip, df, channel_neighbors, add_nan_trace=True
        )[0]
        return

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


def extract_waveforms_cbin(
    bin_file,
    output_dir,
    spike_samples,
    spike_clusters,
    spike_channels,
    h=None,
    channel_labels=None,
    max_wf=256,
    trough_offset=42,
    spike_length_samples=128,
    chunksize_samples=int(3000),
    reader_kwargs={},
    n_jobs=None,
    wfs_dtype=np.float32,
    preprocess=False,
):
    """
    Given a bin file and locations of spikes, extract waveforms for each unit, compute
    the templates, and save the results in `output_path`. If preprocess=True, the waveforms
    come from chunks of raw data which are phase-corrected to account for the ADC, high-pass
    filtered in time with an order 3 Butterworth filter with a 300Hz cutoff, and a common-average
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
        A row of -1s implies that the waveform is missing because the unit is was supposed
        to come from has less than `max_wf` spikes total.

    Parameters:
    :param bin_file: Path to cbin or bin file to be read by spikeglx.Reader
    :param output_dir: Folder where waveform extraction files will be saved
    :param spike_samples: Spike times in samples
    :param spike_clusters: Spike cluster labels
    :param spike_channels: Peak channel around which to extract waveform for each spike
    :param h: Geometry header file for probe (default: NP1)
    :param channel_labels: Array of channel labels used for bad channel interpolation
        (0: good, 1: dead, 2: noisy, 3: out of brain). If not set and preprocess=True,
        channel detection will be run in this function.
    :param max_wf: Max number of waveforms to extract per cluster (default: 256)
    :param trough_offset: Location of peak in spike, in samples (default: 42)
    :param spike_length_samples: Number of samples to extract per spike (default: 128)
    :param chunksize_samples: Length of chunk to process at a time in samples (default: 3000)
    :param reader_kwargs: Kwargs to pass to spikeglx.Reader()
    :param n_jobs: Number of parallel jobs to run. By default it will use 3/4 of available CPUs.
    :param wfs_dtype: Data type of raw waveforms saved (default np.float32)
    :param preprocess: Whether to preprocess the data
    """
    n_jobs = n_jobs or int(cpu_count() / 2)

    sr = spikeglx.Reader(bin_file, **reader_kwargs)
    if h is None:
        h = sr.geometry

    s0_arr = np.arange(0, sr.ns, chunksize_samples)
    s1_arr = s0_arr + chunksize_samples
    s1_arr[-1] = sr.ns

    # selects spikes from throughout the recording for each unit
    wf_flat, unit_ids = _make_wfs_table(
        sr,
        spike_samples,
        spike_clusters,
        spike_channels,
        max_wf,
        trough_offset,
        spike_length_samples,
    )
    num_chunks = s0_arr.shape[0]

    logger.info(f"Chunk size samples: {chunksize_samples}")
    logger.info(f"Num chunks: {num_chunks}")

    logger.info("Running channel detection")
    if channel_labels is None:
        channel_labels = _get_channel_labels(sr)

    nwf = len(wf_flat)
    nu = unit_ids.shape[0]
    logger.info(f"Extracting {nwf} waveforms from {nu} units")

    #  get channel geometry
    geom = np.c_[h["x"], h["y"]]
    channel_neighbors = make_channel_index(geom)
    nc = channel_neighbors.shape[1]

    # this intermediate memmap is written to in parallel
    # the waveforms are ordered only by their chronological position
    # in the recording, as we are reading them in time chunks
    int_fn = output_dir.joinpath("_wf_extract_intermediate.npy")
    wfs = open_memmap(
        int_fn, mode="w+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
    )

    slices = [
        slice(*(np.searchsorted(wf_flat["sample"], [s0_arr[i], s1_arr[i]]).astype(int)))
        for i in range(num_chunks)
    ]

    _ = Parallel(n_jobs=n_jobs)(
        delayed(write_wfs_chunk)(
            i,
            bin_file,
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
            reader_kwargs,
            preprocess
        )
        for i in range(num_chunks)
    )

    # output files
    traces_fn = output_dir.joinpath("waveforms.traces.npy")
    templates_fn = output_dir.joinpath("waveforms.templates.npy")
    table_fn = output_dir.joinpath("waveforms.table.pqt")
    channels_fn = output_dir.joinpath("waveforms.channels.npz")

    ## rearrange and save traces by unit
    # store medians across waveforms
    wfs_templates = np.full((nu, nc, spike_length_samples), np.nan, dtype=np.float32)
    # create waveform output file (~2-3 GB)
    traces_by_unit = open_memmap(
        traces_fn,
        mode="w+",
        shape=(nu, max_wf, nc, spike_length_samples),
        dtype=wfs_dtype,
    )
    logger.info("Writing to output files")

    for i, u in enumerate(unit_ids):
        idx = np.where(wf_flat["cluster"] == u)[0]
        nwf_u = idx.shape[0]
        # reopening these memmaps on each iteration
        # forces Python to clean up each large array it loads
        # and prevent a memory leak
        wfs = open_memmap(
            int_fn, mode="r+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
        )
        traces_by_unit = open_memmap(
            traces_fn,
            mode="r+",
            shape=(nu, max_wf, nc, spike_length_samples),
            dtype=wfs_dtype,
        )
        # write up to 256 waveforms and leave the rest of dimensions 1-3 as NaNs
        traces_by_unit[i, : min(max_wf, nwf_u), :, :] = wfs[idx].astype(wfs_dtype)
        traces_by_unit.flush()
        # populate this array in memory as it's 256x smaller
        wfs_templates[i, :, :] = np.nanmedian(wfs[idx], axis=0)

    # cleanup intermediate file
    int_fn.unlink()

    # save templates
    np.save(templates_fn, wfs_templates)

    # add in dummy rows and order by unit, and then sample
    unit_counts = wf_flat.groupby("cluster")["sample"].count().reset_index(name="count")
    unit_counts["missing"] = max_wf - unit_counts["count"]
    missing_wf = unit_counts[unit_counts["missing"] > 0]
    total_missing = sum(missing_wf.missing)
    extra_rows = pd.DataFrame(
        {
            "sample": [np.nan] * total_missing,
            "peak_channel": [np.nan] * total_missing,
            "index": [np.nan] * total_missing,
            "cluster": sum(
                [[row["cluster"]] * row["missing"] for _, row in missing_wf.iterrows()],
                [],
            ),
        }
    )
    save_df = pd.concat([wf_flat, extra_rows])
    # now the waveforms are arranged by cluster, and then in time
    # these match dimensions 0 and 1 of waveforms.traces.npy
    save_df.sort_values(["cluster", "sample"], inplace=True)
    save_df.to_parquet(table_fn)

    # save channel map for each waveform
    # these values are now reordered so that they match the pqt
    # and the traces file
    peak_channel = np.nan_to_num(save_df["peak_channel"].to_numpy(), nan=-1).astype(
        np.int16
    )
    dummy_idx = np.where(peak_channel >= 0)[0]
    # leave "missing" waveforms as -1 since we can't have NaN with int dtype
    chan_map = np.ones((max_wf * nu, nc), np.int16) * -1
    chan_map[dummy_idx] = channel_neighbors[peak_channel[dummy_idx].astype(int)]
    np.savez(channels_fn, channels=chan_map)
