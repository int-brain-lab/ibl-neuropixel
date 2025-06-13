import logging
from pathlib import Path

import scipy
import pandas as pd
import numpy as np
from numpy.lib.format import open_memmap
from joblib import Parallel, delayed, cpu_count

import spikeglx
from ibldsp.voltage import detect_bad_channels, interpolate_bad_channels, car, kfilt
from ibldsp.fourier import fshift
from ibldsp.utils import make_channel_index
from iblutil.numerical import ismember

logger = logging.getLogger(__name__)


def aggregate_by_clusters(df_wavs):
    """
    Group by the waveform dataframe by clusters
    :param df_wavs:
    :return:
    """
    df_clusters = (
        df_wavs.loc[df_wavs["sample"] >= 0, :]
        .groupby("cluster")
        .aggregate(
            count=pd.NamedAgg(column="cluster", aggfunc="count"),
            first_index=pd.NamedAgg(column="waveform_index", aggfunc="min"),
            last_index=pd.NamedAgg(column="waveform_index", aggfunc="max"),
        )
    )
    return df_clusters


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

    :param arr: Array of traces. (nc, ns). The last trace of the array should be a
        row of non-data NaNs. If this has not been added set the `add_nan_trace` flag.
    :param df: df containing "sample" and "peak_channel" columns.
    :param channel_neighbors: Channel neighbor matrix (nc, nx)
    :param trough_offset: Number of samples to include before peak.
    (defaults to 42)
    :param spike_length_samples: Total length of wf in samples.
    (defaults to 128)
    :param add_nan_trace: Whether to add a row of nan's as the last trace.
        (If False, code assumes this has already been added)
    """
    # This is to do fast index assignment to assign missing channels (out of the probe) to nan
    if add_nan_trace:
        newcol = np.empty((1, arr.shape[1]))
        newcol[:] = np.nan
        arr = np.vstack([arr, newcol])

    # check that the spike window is included in the recording:
    last_idx = df["sample"].iloc[-1]
    assert last_idx + (spike_length_samples - trough_offset) < arr.shape[1], (
        f"Spike index {last_idx} extends past end of recording ({arr.shape[1]} samples)."
    )

    nwf = len(df)

    # Get channel indices
    cind = channel_neighbors[df["peak_channel"].to_numpy()]

    # Get sample indices
    sind = df["sample"].to_numpy()[:, np.newaxis] + (
        np.arange(spike_length_samples) - trough_offset
    )
    nchan = cind.shape[1]

    wfs = np.zeros((nwf, nchan, spike_length_samples), arr.dtype)
    fun = range
    if verbose:
        try:
            from tqdm import trange

            fun = trange
        except ImportError:
            pass
    for i in fun(nwf):
        wfs[i, :, :] = arr[:, sind[i]][cind[i], :]

    return wfs, cind, trough_offset


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

    _channel_labels = np.zeros((sr.nc - sr.nsync, num_snippets), int)

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
    seed=None,
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
    rng = np.random.default_rng(seed=seed)  # numpy 1.23.5

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
    wf_idx = wf_idx[np.nonzero(wf_idx)[0][0] :]

    # get sample times, clusters, channels
    wf_flat = pd.DataFrame(
        {
            "index": np.arange(wf_idx.shape[0]),
            "sample": spike_samples[wf_idx].astype(np.int64),
            "cluster": spike_clusters[wf_idx].astype(int),
            "peak_channel": spike_channels[wf_idx].astype(int),
            "waveform_index": np.zeros(wf_idx.shape[0], int),
        }
    )

    # we pre-compute the final absolute indices of each waveform
    unique_clusters, cluster_index, cluster_counts = np.unique(
        wf_flat["cluster"], return_inverse=True, return_counts=True
    )
    index_order_clusters = np.argsort(cluster_index, kind="stable")
    wf_flat.loc[index_order_clusters, "waveform_index"] = np.arange(
        wf_flat.shape[0]
    )  # 3d "flat" version
    return wf_flat, unit_ids


def write_wfs_chunk(
    i_chunk,
    cbin,
    wfs_mmap,
    geom_dict,
    channel_labels,
    channel_neighbors,
    wf_flat,
    sr_sl,
    chunksize_samples,
    trough_offset,
    spike_length_samples,
    reader_kwargs,
    preprocess_steps,
):
    """
    Parallel job to extract waveforms from chunk `i_chunk` of a recording `sr` and
    write them to the correct spot in the output .npy file `wfs_fn`.
    """
    if len(wf_flat) == 0:
        return

    my_sr = spikeglx.Reader(cbin, **reader_kwargs)
    s0, s1 = sr_sl

    if i_chunk == 0:
        offset = 0
    else:
        offset = trough_offset

    sample = wf_flat["sample"].astype(int) + offset - i_chunk * chunksize_samples
    peak_channel = wf_flat["peak_channel"]

    df = pd.DataFrame({"sample": sample, "peak_channel": peak_channel})

    snip = my_sr[
        s0 - offset : s1 + spike_length_samples - trough_offset, : -my_sr.nsync
    ].T

    if "butterworth" in preprocess_steps:
        butter_kwargs = {"N": 3, "Wn": 300 / my_sr.fs * 2, "btype": "highpass"}
        sos = scipy.signal.butter(**butter_kwargs, output="sos")
        snip = scipy.signal.sosfiltfilt(sos, snip)

    if "phase_shift" in preprocess_steps:
        snip = fshift(snip, geom_dict["sample_shift"], axis=-1)

    if "bad_channel_interpolation" in preprocess_steps:
        snip = interpolate_bad_channels(
            snip,
            channel_labels,
            geom_dict["x"],
            geom_dict["y"],
        )

    k_kwargs = {
        "ntr_pad": 60,
        "ntr_tap": 0,
        "lagc": 0,  # no agc for the median estimator of common reference channel
        "butter_kwargs": {"N": 3, "Wn": 0.01, "btype": "highpass"},
    }
    if "car" in preprocess_steps:
        car_func = lambda dat: car(dat, **k_kwargs)  # noqa: E731
        snip = car_func(snip)

    if "kfilt" in preprocess_steps:
        kfilt_func = lambda dat: kfilt(dat, **k_kwargs)  # noqa: E731
        snip = kfilt_func(snip)
    iw = wf_flat["waveform_index"].values
    wfs_mmap[iw, :, :] = extract_wfs_array(
        snip, df, channel_neighbors, add_nan_trace=True
    )[0]


def extract_wfs_cbin(
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
    chunksize_samples=int(30_000),
    reader_kwargs=None,
    n_jobs=None,
    preprocess_steps=None,
    seed=None,
    scratch_dir=None,
):
    """
    Given a bin file and locations of spikes, extract waveforms for each unit, compute
    the templates, and save the results in `output_path`. If preprocess=True, the waveforms
    come from chunks of raw data which are phase-corrected to account for the ADC, high-pass
    filtered in time with an order 3 Butterworth filter with a 300Hz cutoff, and a common-average
    reference procedure is applied in the spatial dimension.

    The following files will be generated:
    - waveforms.traces.npy: `(total_waveforms, nc, spike_length_samples)`
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
    :param chunksize_samples: Length of chunk to process at a time in samples (default: 30_000)
    :param reader_kwargs: Kwargs to pass to spikeglx.Reader()
    :param n_jobs: Number of parallel jobs to run. By default it will use 3/4 of available CPUs.
    :param wfs_dtype: Data type of raw waveforms saved (default np.float32)
    :param preprocess: Preprocessing options to apply, list which must be a subset of
        ["phase_shift", "bad_channel_interpolation", "butterworth", "car", "kfilt"]
        By default a butterworth 300Hz high-pass and the rephasing of the channels is performed
    """
    n_jobs = n_jobs or int(cpu_count() / 2)
    preprocess_steps = (
        ["butterworth", "phase_shift"] if preprocess_steps is None else preprocess_steps
    )
    reader_kwargs = {} if reader_kwargs is None else reader_kwargs

    assert set(preprocess_steps).issubset(
        {"phase_shift", "bad_channel_interpolation", "butterworth", "car", "kfilt"}
    )

    if "car" in preprocess_steps and "kfilt" in preprocess_steps:
        raise ValueError("Must choose car or kfilt spatial filter")

    sr = spikeglx.Reader(bin_file, **reader_kwargs)
    if h is None:
        h = sr.geometry

    if sr.is_mtscomp:
        bin_file = sr.decompress_to_scratch(scratch_dir=scratch_dir)
        sr = spikeglx.Reader(bin_file, **reader_kwargs)
        file_to_unlink = bin_file
    else:
        file_to_unlink = None

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
        seed,
    )
    num_chunks = s0_arr.shape[0]

    logger.info(f"Chunk size samples: {chunksize_samples}")
    logger.info(f"Num chunks: {num_chunks}")

    if channel_labels is None and "bad_channel_interpolation" in preprocess_steps:
        logger.info("Running channel detection")
        channel_labels = _get_channel_labels(sr)
    elif channel_labels is None:
        channel_labels = np.zeros(sr.nc - sr.nsync)

    nwf = wf_flat.shape[0]
    nu = unit_ids.shape[0]
    logger.info(f"Extracting {nwf} waveforms from {nu} units")

    #  get channel geometry
    geom = np.c_[h["x"], h["y"]]
    channel_neighbors = make_channel_index(geom)
    nc = channel_neighbors.shape[1]

    # this intermediate memmap is written to in parallel
    # the waveforms are ordered only by their chronological position
    # in the recording, as we are reading them in time chunks
    traces_fn = output_dir.joinpath("waveforms.traces.npy")
    wfs = open_memmap(
        traces_fn, mode="w+", shape=(nwf, nc, spike_length_samples), dtype=np.float32
    )

    slices = [
        slice(*(np.searchsorted(wf_flat["sample"], [s0_arr[i], s1_arr[i]]).astype(int)))
        for i in range(num_chunks)
    ]

    _ = Parallel(n_jobs=n_jobs)(
        delayed(write_wfs_chunk)(
            i,
            bin_file,
            wfs,
            h,
            channel_labels,
            channel_neighbors,
            wf_flat.iloc[slices[i]],
            (s0_arr[i], s1_arr[i]),
            chunksize_samples,
            trough_offset,
            spike_length_samples,
            reader_kwargs,
            preprocess_steps,
        )
        for i in range(num_chunks)
    )

    # output files
    templates_fn = output_dir.joinpath("waveforms.templates.npy")
    table_fn = output_dir.joinpath("waveforms.table.pqt")
    channels_fn = output_dir.joinpath("waveforms.channels.npz")

    ## rearrange dataframe: sort waveforms by cluster and aggregate by cluster
    wf_flat.sort_values(by=["cluster", "sample"], inplace=True)
    df_clusters = aggregate_by_clusters(wf_flat)

    # we want to store the index of the waveform within each cluster to facilitate loading later
    wf_flat["index_within_clusters"] = np.ones(wf_flat.shape[0])
    inewc = (
        np.diff(wf_flat["cluster"].values, prepend=wf_flat["cluster"].values[0]) != 0
    )
    wf_flat.loc[inewc, "index_within_clusters"] = -df_clusters["count"].values[:-1] + 1
    wf_flat["index_within_clusters"] = (
        np.cumsum(wf_flat["index_within_clusters"].values).astype(int) - 1
    )

    # store medians across waveforms
    wfs_templates = np.full((nu, nc, spike_length_samples), np.nan, dtype=np.float32)
    logger.info("Writing to output files")
    wfs = open_memmap(traces_fn)
    for i, rec in enumerate(df_clusters.itertuples()):
        wfs_templates[i] = np.nanmedian(
            wfs[rec.first_index : rec.last_index + 1], axis=0
        )
    # save templates
    np.save(templates_fn, wfs_templates)
    # save the waveform table

    wf_flat.to_parquet(table_fn)

    # save channel map for each waveform
    # these values are now reordered so that they match the pqt
    # and the traces file
    peak_channel = np.nan_to_num(wf_flat["peak_channel"].to_numpy(), nan=-1).astype(
        np.int16
    )
    chan_map = channel_neighbors[peak_channel.astype(int)]
    np.savez(channels_fn, channels=chan_map)
    # clean up the cached bin file
    if file_to_unlink is not None:
        file_to_unlink.with_suffix(".meta").unlink()
        file_to_unlink.unlink()
    return sorted([templates_fn, table_fn, channels_fn, traces_fn])


class WaveformsLoader:
    data_version = None

    """
    Interface to the output of `extract_wfs_cbin`. Requires the following four files to
    exist in `data_dir`:

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

    WaveformsLoader.load_waveforms() and random_waveforms() allow selection of a subset of
        waveforms.
    """

    def __init__(
        self,
        data_dir,
        trough_offset=42,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.trough_offset = trough_offset
        self.traces_fp = self.data_dir.joinpath("waveforms.traces.npy")
        self.templates_fp = self.data_dir.joinpath("waveforms.templates.npy")
        self.table_fp = self.data_dir.joinpath("waveforms.table.pqt")
        self.channels_fp = self.data_dir.joinpath("waveforms.channels.npz")

        assert self.traces_fp.exists(), "waveforms.traces.npy file missing!"
        assert self.templates_fp.exists(), "waveforms.templates.npy file missing!"
        assert self.table_fp.exists(), "waveforms.table.pqt file missing!"
        assert self.channels_fp.exists(), "waveforms.channels.npz file missing!"

        self.traces = np.lib.format.open_memmap(self.traces_fp)
        self.df_wav = (
            pd.read_parquet(self.table_fp)
            .reset_index(drop=True)
            .drop(columns=["index"])
        )
        if len(self.traces.shape) == 4:
            self.data_version = 1
            self.df_wav["sample"] = self.df_wav["sample"].astype("Int64")
            self.df_wav["peak_channel"] = self.df_wav["peak_channel"].astype("Int64")
            self.df_wav["waveform_index"] = np.arange(
                self.df_wav.shape[0], dtype=np.int64
            )
            self.df_wav["index_within_cluster"] = np.tile(
                np.arange(self.traces.shape[1]), self.traces.shape[0]
            )
            self.total_wfs = sum(~self.df_wav["peak_channel"].isna())
        else:
            self.data_version = 2

        self.df_clusters = aggregate_by_clusters(self.df_wav)
        self.templates = np.lib.format.open_memmap(self.templates_fp, dtype=np.float32)
        self.channels = np.load(self.channels_fp)["channels"]

    def __repr__(self):
        return f"""
        WaveformsLoader data version {self.data_version}
        {self.nw:_} total waveforms {self.ns} samples, {self.nc} channels
        {self.nu:_} units, {self.max_wf:_} max waveforms per label
        dtype: {self.wfs_dtype}
        data path: {self.data_dir}
        """

    @property
    def max_wf(self):
        return self.df_clusters["count"].max()

    @property
    def wfs_dtype(self):
        return self.traces.dtype

    @property
    def nu(self):
        return self.df_clusters.shape[0]

    @property
    def ns(self):
        return self.traces.shape[-1]

    @property
    def nc(self):
        return self.traces.shape[-2]

    @property
    def nw(self):
        return self.df_wav.shape[0]

    def load_waveforms(
        self, labels=None, indices=None, return_info=True, flatten=False
    ):
        """
        Returns a specified subset of waveforms from the dataset.

        :param labels: (list, NumPy array) Label ids (usually clusters) from which to get waveforms.
        :param indices: (list, NumPy array) Waveform indices to grab for each waveform 1D.
        :param return_info: If True, returns waveforms, table, channels, where table is a DF containing
            information about the waveforms returned, and channels is the channel map for each waveform.
        :param flatten: If True, returns all waveforms stacked along dimension zero, otherwise returns
            array of shape (num_labels, num_indices_per_label, num_channels, spike_length_samples)
        """
        labels = np.array(self.df_clusters.index if labels is None else labels)
        iw, _ = ismember(self.df_wav["cluster"], labels)
        if self.data_version == 1:
            indices = np.array(np.arange(self.max_wf) if indices is None else indices)
            indices = (
                np.tile(indices, (labels.size, 1)) if indices.ndim < 2 else indices
            )
            assert indices.shape[0] == labels.size, (
                "If indices is a 2D-array, the second dimension must match the number of clusters."
            )
            _, iu, _ = np.intersect1d(
                self.df_clusters.index, labels, return_indices=True
            )
            assert iu.size == labels.size, "Not all labels found in dataset."
            wfs = self.traces[iu[:, np.newaxis], indices].astype(np.float32)
            if flatten:
                wfs = wfs.reshape(-1, self.nc, self.ns)
        elif self.data_version == 2:
            if indices is not None:
                iw = np.where(iw)[0]
                iw = iw[
                    self.df_wav.loc[iw, "index_within_clusters"].isin(
                        np.atleast_1d(np.array(indices))
                    )
                ]
            wfs = self.traces[iw].astype(np.float32)
        info = self.df_wav.loc[iw, :].copy()
        channels = self.channels[iw].astype(int)
        n_nan = sum(info["sample"].isna())
        if n_nan > 0:
            logger.info(f"{n_nan} NaN waveforms included in result.")
        if return_info:
            return wfs, info, channels
        else:
            return wfs

    def random_waveforms(
        self,
        labels=None,
        num_random_labels=None,
        num_random_waveforms=None,
        return_info=True,
        seed=None,
        flatten=False,
    ):
        """
        Returns a random subset of waveforms from the dataset.

        :param labels: (list, NumPy array) Label ids (usually clusters) from which to get waveforms.
            If None, 10 random labels are chosen.
        :param num_random_labels: If set, this number of random labels are chosen
        :param num_random_waveforms: If set, this number of random waveforms are chosen for each label.
            If None, 10 random waveforms are chosen for each label.
        :param return_info: If True, returns waveforms, table, channels, where table is a DF containing
            information about the waveforms returned, and channels is the channel map for each waveform.
        :param flatten: If True, returns all waveforms stacked along dimension zero, otherwise returns
            array of shape (num_labels, num_indices_per_label, num_channels, spike_length_samples)

        """
        rg = np.random.default_rng(seed=seed)

        if labels is None:
            if num_random_labels is None:
                labels = rg.choice(self.labels, 10)
            else:
                labels = rg.choice(self.labels, num_random_labels, replace=False)
        else:
            assert num_random_labels is None, (
                "labels and num_random_labels cannot both be set"
            )

        labels = np.array(labels)
        label_idx = np.array([np.where(self.labels == label)[0][0] for label in labels])

        num_labels = labels.shape[0]

        if num_random_waveforms is None:
            num_random_waveforms = 10

        # now select random non-NaN indices for each label
        indices = np.zeros((num_labels, num_random_waveforms), int)
        for u, label in enumerate(labels):
            _t = self.table[self.table["cluster"] == label]
            nanidx = _t["sample"].isna()
            valid = _t[~nanidx]

            num_valid_waveforms = len(valid)
            if num_valid_waveforms >= num_random_waveforms:
                indices[u, :] = rg.choice(
                    valid.wf_number.to_numpy(), num_random_waveforms, replace=False
                )
                continue

            num_nan_waveforms = num_random_waveforms - num_valid_waveforms
            indices[u, :num_valid_waveforms] = rg.choice(
                valid.wf_number.to_numpy(), num_valid_waveforms, replace=False
            )

            indices[u, num_valid_waveforms:] = np.arange(
                num_valid_waveforms, num_valid_waveforms + num_nan_waveforms
            )

        wfs = self.traces[label_idx[:, None], indices].astype(np.float32)

        if flatten:
            wfs = wfs.reshape(-1, self.num_channels, self.spike_length_samples)

        info = self.table[self.table["cluster"].isin(labels)].copy()
        dfs = []
        for i, lab in enumerate(labels):
            _idx = indices[i]
            dfs.append(info[(info["wf_number"].isin(_idx)) & (info["cluster"] == lab)])
        info = pd.concat(dfs).reset_index(drop=True)

        channels = self.channels[info["linear_index"].to_numpy()].astype(int)

        n_nan = sum(info["sample"].isna())
        if n_nan > 0:
            logger.warning(f"{n_nan} NaN waveforms included in result.")
        if return_info:
            return wfs, info, channels

        logger.info("Use return_info=True and check the table for details.")

        return wfs
