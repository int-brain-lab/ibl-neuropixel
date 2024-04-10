import scipy
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
import neuropixel
import spikeglx

from joblib import Parallel, delayed, cpu_count

from ibldsp.voltage import detect_bad_channels, interpolate_bad_channels, car
from ibldsp.fourier import fshift
from ibldsp.utils import make_channel_index


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

    start = (np.linspace(100, int(sr.rl) - 100, num_snippets) * sr.fs).astype(int)
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
    chunksize_t=10,
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
            "indices": np.arange(wf_idx.shape[0]),
            "samples": spike_times[wf_idx].astype(int),
            "clusters": spike_clusters[wf_idx].astype(int),
            "channels": spike_channels[wf_idx].astype(int),
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

    sample = wf_flat["samples"].astype(int) + offset - i_chunk * chunksize_samples
    peak_channel = wf_flat["channels"]

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
    wfs_mmap[wf_flat["indices"], :, :] = extract_wfs_array(
        snip1.T, df, channel_neighbors
    )[0]
    wfs_mmap.flush()


def extract_wfs_cbin(
    cbin_file,
    output_file,
    spike_times,
    spike_clusters,
    spike_channels,
    h=None,
    wf_extract_params=None,
    nprocesses=None,
):
    """
    Given a cbin file and locations of spikes, extract waveforms for each unit, compute
    the templates, and save to `output_file`.

    If `output_file=Path("/path/to/example_clusters.npy")`, this array will be of shape
    `(num_units, max_wf, nc, spike_length_samples)` where by default `max_wf=256, nc=40,
    spike_length_samples=128`.

    The file "path/to/example_clusters_templates.npy" will also be generated, of shape
    `(num_units, nc, spike_length_samples)`, where the median across waveforms is taken
    for each unit.

    The parquet file "path/to/example_clusters.pqt" contains the samples and max channels
    of each waveform, indexed by unit.
    """
    if h is None:
        h = neuropixel.trace_header()

    if wf_extract_params is None:
        wf_extract_params = {
            "max_wf": 256,
            "trough_offset": 42,
            "spike_length_samples": 128,
            "chunksize_t": 10,
        }

    output_path = output_file.parent

    max_wf = wf_extract_params["max_wf"]
    trough_offset = wf_extract_params["trough_offset"]
    spike_length_samples = wf_extract_params["spike_length_samples"]
    chunksize_t = wf_extract_params["chunksize_t"]

    sr = spikeglx.Reader(cbin_file)
    chunksize_samples = chunksize_t * 30_000
    s0_arr = np.arange(0, sr.ns, chunksize_samples)
    s1_arr = s0_arr + chunksize_samples
    s1_arr[-1] = sr.ns

    wf_flat, unit_ids = _make_wfs_table(
        sr, spike_times, spike_clusters, spike_channels, **wf_extract_params
    )
    num_chunks = s0_arr.shape[0]
    print(f"Chunk size: {chunksize_t}")
    print(f"Num chunks: {num_chunks}")

    print("Running channel detection")
    channel_labels = _get_channel_labels(sr)

    nwf = wf_flat["samples"].shape[0]
    nu = unit_ids.shape[0]
    print(f"Extracting {nwf} waveforms from {nu} units")

    #  get channel geometry
    geom = np.c_[h["x"], h["y"]]
    channel_neighbors = make_channel_index(geom)
    nc = channel_neighbors.shape[1]

    fn = output_path.joinpath("_wf_extract_intermediate.npy")
    wfs = open_memmap(
        fn, mode="w+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
    )

    slices = [
        slice(
            *(np.searchsorted(wf_flat["samples"], [s0_arr[i], s1_arr[i]]).astype(int))
        )
        for i in range(num_chunks)
    ]

    nprocesses = nprocesses or int(cpu_count() - cpu_count() / 4)
    _ = Parallel(n_jobs=nprocesses)(
        delayed(write_wfs_chunk)(
            i,
            cbin_file,
            fn,
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

    wfs = open_memmap(
        fn, mode="r+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
    )
    # bookkeeping
    wfs_by_unit = np.full(
        (nu, max_wf, nc, spike_length_samples), np.nan, dtype=np.float16
    )
    wfs_medians = np.full((nu, nc, spike_length_samples), np.nan, dtype=np.float32)
    print("Computing templates")
    for i, u in enumerate(unit_ids):
        _wfs_unit = wfs[wf_flat["clusters"] == u]
        nwf_u = _wfs_unit.shape[0]
        wfs_by_unit[i, : min(max_wf, nwf_u), :, :] = _wfs_unit.astype(np.float16)
        wfs_medians[i, :, :] = np.nanmedian(_wfs_unit, axis=0)

    df = pd.DataFrame(
        {
            "sample": wf_flat["samples"],
            "peak_channel": wf_flat["channels"],
            "cluster": wf_flat["clusters"],
        }
    )
    df = df.sort_values(["cluster", "sample"]).set_index(["cluster", "sample"])

    np.save(output_file, wfs_by_unit)
    # medians
    avg_file = output_file.parent.joinpath(output_file.stem + "_templates.npy")
    np.save(avg_file, wfs_medians)
    df.to_parquet(output_file.with_suffix(".pqt"))

    fn.unlink()
