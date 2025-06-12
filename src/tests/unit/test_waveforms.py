from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import ibldsp.utils as utils
import ibldsp.waveforms as waveforms
import ibldsp.waveform_extraction as waveform_extraction
from neurowaveforms.model import generate_waveform
from neuropixel import trace_header
from ibldsp.fourier import fshift

TEST_PATH = Path(__file__).parents[1].joinpath("fixtures")


def _dummy_spike(ns):
    ns = 100
    w = np.zeros(ns)
    w[0] = 1
    w = np.fft.fftshift(w)
    sos = scipy.signal.butter(3, 0.2, "low", output="sos")
    w = scipy.signal.sosfiltfilt(sos, w)
    return w


def make_array_peak_through_tip():
    arr = np.array(
        [
            [
                [1, 1, np.nan],
                [2, 2, np.nan],
                [3, 5, np.nan],
                [4, 4, np.nan],
                [4, -5, np.nan],
                [4, -6, np.nan],
                [4, 5, np.nan],
            ],
            [
                [-8, 7, 7],
                [-7, 7, 7],
                [7, 7, 7],
                [7, 7, 7],
                [4, 5, 4],
                [4, 5, 4],
                [4, 5, 4],
            ],
        ]
    )
    return arr


def make_array_peak_through_tip_v2():
    # Duplicating as above array throws error due to Nans when computing arr_pre
    arr = np.array(
        [
            [
                [1, 1, np.nan],
                [2, 2, np.nan],
                [3, 5, np.nan],
                [4, 4, np.nan],
                [4, -5, np.nan],
                [4, -6, np.nan],
                [4, 5, np.nan],
            ],
            [
                [1, 1, 0],
                [2, 2, 0],
                [3, 5, 0],
                [4, 4, 0],
                [4, -5, 0],
                [4, -8, 1],
                [4, 5, 0],
            ],
        ]
    )
    return arr


def test_peak_through_tip_2d():
    arr = make_array_peak_through_tip()
    df = waveforms.compute_spike_features(arr[0, :, :])
    np.testing.assert_equal(df.shape[0], 1)


def test_peak_through_tip_3d():
    arr = make_array_peak_through_tip_v2()
    df = waveforms.compute_spike_features(arr)
    arr_out = waveforms.get_array_peak(arr, df)

    np.testing.assert_equal(
        arr_out, np.array([[1, 2, 5, 4, -5, -6, 5], [1, 2, 5, 4, -5, -8, 5]])
    )

    np.testing.assert_equal(df.peak_trace_idx, np.array([1, 1]))
    np.testing.assert_equal(df.peak_time_idx, np.array([5, 5]))
    np.testing.assert_equal(df.peak_val, np.array([-6, -8]))

    np.testing.assert_equal(df.trough_time_idx, np.array([6, 6]))
    np.testing.assert_equal(df.trough_val, np.array([5, 5]))

    np.testing.assert_equal(df.tip_time_idx, np.array([2, 2]))
    np.testing.assert_equal(df.tip_val, np.array([5, 5]))


def test_halfpeak_slopes():
    # Load fixtures
    folder_save = TEST_PATH.joinpath("waveform_sample")
    arr_in = np.load(folder_save.joinpath("test_arr_in.npy"))
    test_arr_peak = np.load(folder_save.joinpath("test_arr_peak.npy"))
    test_df = pd.read_csv(folder_save.joinpath("test_df.csv"))
    test_df = test_df.drop("Unnamed: 0", axis=1)  # Dropping the "Unnamed: 0" column
    df = waveforms.compute_spike_features(arr_in, fs=30000, recovery_duration_ms=0.16)
    # Array peak testing
    arr_peak = waveforms.get_array_peak(arr_in, df)
    np.testing.assert_equal(arr_peak, test_arr_peak)
    # Df testing
    pd.testing.assert_frame_equal(df.astype(float), test_df.astype(float))


def test_dist_chanel_from_peak():
    # Distance test
    xyz_testd = np.stack(
        (
            np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]),
            np.array([[4, 0, 0], [2, 0, 0], [np.nan, np.nan, np.nan], [1, 0, 0]]),
        ),
        axis=2,
    )
    xyz_testd = np.swapaxes(xyz_testd, axis1=0, axis2=2)
    xyz_testd = np.swapaxes(xyz_testd, axis1=1, axis2=2)

    eu_dist = waveforms.dist_chanel_from_peak(xyz_testd, np.array([0, 1]))
    weights = np.zeros(eu_dist.shape)
    weights[0][1:3] = 1
    weights[1][2:4] = 1
    sp_spread = waveforms.spatial_spread_weighted(eu_dist, weights)

    np.testing.assert_almost_equal(eu_dist[0, :], np.array([0, 1, 1, np.sqrt(2)]))
    np.testing.assert_equal(eu_dist[1, :], np.array([2, 0, np.nan, 1]))
    np.testing.assert_equal(sp_spread, np.array([1, 0.5]))


def test_reshape_wav_one_channel():
    # Test reshape into 1 channel
    arr = make_array_peak_through_tip()
    arr_out = waveforms.reshape_wav_one_channel(arr)
    # Check against test
    arr_tested = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0], [4.0], [4.0], [4.0]],
            [[1.0], [2.0], [5.0], [4.0], [-5.0], [-6.0], [5.0]],
            [[np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]],
            [[-8.0], [-7.0], [7.0], [7.0], [4.0], [4.0], [4.0]],
            [[7.0], [7.0], [7.0], [7.0], [5.0], [5.0], [5.0]],
            [[7.0], [7.0], [7.0], [7.0], [4.0], [4.0], [4.0]],
        ]
    )
    np.testing.assert_equal(arr_out, arr_tested)


def test_weights_all_channels():
    arr = make_array_peak_through_tip()
    weight = waveforms.weights_spk_ch(arr)
    weight_tested = np.array([[4.0, -6.0, 0.0], [-8.0, 7.0, 7.0]])

    np.testing.assert_equal(weight, weight_tested)


def test_generate_waveforms():
    wav = generate_waveform()
    assert wav.shape == (40, 121)


class TestWaveformExtractorArray(unittest.TestCase):
    # create sample array with 10 point wfs at different
    # channel locations
    trough_offset = 42
    ns = 1000
    nc = 384
    samples = np.arange(100, ns, 100)
    channels = np.arange(12, 384, 45)

    arr = np.zeros((ns, nc + 1), np.float32)
    for i in range(9):
        s, c = samples[i], channels[i]
        arr[s, c] = float(i + 1)
    arr[:, -1] = np.nan

    df = pd.DataFrame({"sample": samples, "peak_channel": channels})
    # generate channel neighbor matrix for NP1, default radius 200um
    geom_dict = trace_header(version=1)
    geom = np.c_[geom_dict["x"], geom_dict["y"]]
    channel_neighbors = utils.make_channel_index(geom, radius=200.0)
    # radius = 200um, 38 chans
    num_channels = 40
    arr = arr.T

    def test_extract_waveforms_array(self):
        wfs, _, _ = waveform_extraction.extract_wfs_array(
            self.arr, self.df, self.channel_neighbors
        )

        # first wf is a special case: it's at the top of the probe so the center
        # index is the actual channel index, and the rest of the wf has been padded
        # with NaNs
        assert wfs[0, self.channels[0], self.trough_offset] == 1.0
        assert np.all(
            np.isnan(wfs[0, self.num_channels // 2 + self.channels[0] + 1 :, :])
        )

        for i in range(1, 9):
            print(i)
            # center channel depends on odd/even of channel
            if self.channels[i] % 2 == 0:
                centered_channel_idx = 19
            else:
                centered_channel_idx = 20
            assert wfs[i, centered_channel_idx, self.trough_offset] == float(i + 1)

        # last wf is a special case analogous to the first wf, but at the bottom
        # of the probe
        if self.channels[-1] % 2 == 0:
            centered_channel_idx = 19
        else:
            centered_channel_idx = 20
        assert wfs[-1, centered_channel_idx, self.trough_offset] == 9.0

    def test_spike_window(self):
        # check that we have an error when the last spike window
        # extends past end of recording
        df = self.df.copy()
        df["sample"].iloc[-1] = 996
        with self.assertRaisesRegex(AssertionError, "extends"):
            _ = waveform_extraction.extract_wfs_array(
                self.arr, df, self.channel_neighbors
            )

    def test_nan_channel(self):
        # test that if user does not fill last column with NaNs
        # the user can set the flag and the result will be the same
        arr = self.arr.copy()[:, :-1]
        wfs = waveform_extraction.extract_wfs_array(
            self.arr, self.df, self.channel_neighbors
        )
        wfs_nan = waveform_extraction.extract_wfs_array(
            arr, self.df, self.channel_neighbors, add_nan_trace=True
        )
        np.testing.assert_equal(wfs, wfs_nan)

    def test_wave_shift_corrmax(self):
        sample_shifts = [4.43, -1.0]
        sig_lens = [100, 101]
        for sample_shift in sample_shifts:
            for sig_len in sig_lens:
                spike = _dummy_spike(sig_len)
                spike = -np.fft.irfft(
                    np.fft.rfft(np.real(spike)) * np.exp(1j * 45 / 180 * np.pi)
                )
                spike2 = fshift(spike, sample_shift)
                spike3, shift_computed = waveforms.wave_shift_corrmax(spike, spike2)

                np.testing.assert_equal(
                    sample_shift, np.around(shift_computed, decimals=2)
                )

    def test_wave_shift_phase(self):
        fs = 30000
        # Resynch in time spike2 onto spike
        sample_shift_original = 0.323
        spike = _dummy_spike(100)
        spike = -np.fft.irfft(
            np.fft.rfft(np.real(spike)) * np.exp(1j * 45 / 180 * np.pi)
        )
        spike = np.append(spike, np.zeros((1, 25)))
        spike2 = fshift(spike, sample_shift_original)
        # Resynch
        spike_resync, sample_shift_applied = waveforms.wave_shift_phase(
            spike, spike2, fs
        )
        np.testing.assert_equal(
            sample_shift_original, np.around(sample_shift_applied, decimals=3)
        )

    def test_wave_shift_waveform(self):
        sample_shift_original = 15.32
        # Create peak channel spike
        spike_peak = _dummy_spike(100)  # 100 time samples
        spike_peak = -np.fft.irfft(
            np.fft.rfft(np.real(spike_peak)) * np.exp(1j * 45 / 180 * np.pi)
        )
        # Create other channel spikes
        spike_oth = spike_peak * 0.3
        # Create shifted spike
        spike_peak2 = fshift(spike_peak, sample_shift_original)
        spike_oth2 = fshift(spike_oth, sample_shift_original)

        # Create matrix N=512 wavs: 511 spikes the same, 1 with shifted (one spike will have 2 channels)
        wav_normal = np.stack([spike_peak, spike_oth])  # size (trace, time) : (2, 100)
        wav_shifted = np.stack([spike_peak2, spike_oth2])

        n_wav = 511
        wav_rep = np.repeat(wav_normal[:, :, np.newaxis], n_wav, axis=2)
        wav_all = np.dstack(
            (wav_rep, wav_shifted)
        )  # size (trace, time, N spike) : (2, 100, 512)

        # Change axis to (N spike, trace, time) : (512, 2, 100)
        wav_cluster = np.swapaxes(wav_all, axis1=1, axis2=2)  # (2, 512, 100)
        wav_cluster = np.swapaxes(wav_cluster, axis1=1, axis2=0)
        # The last wav (-1) has the shift after all this swapping - checked visually by plotting below
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(-np.flipud(wav_cluster[0, :, :]), cmap="Grays")
        axs[1].imshow(-np.flipud(wav_cluster[-1, :, :]), cmap="Grays")
        """
        wav_out, shift_applied = waveforms.shift_waveform(wav_cluster)
        # Test last waveform shift applied is minus the original shift, and the rest 511 waveforms are 0
        np.testing.assert_equal(
            -sample_shift_original, np.around(shift_applied[-1], decimals=2)
        )
        np.testing.assert_equal(
            np.zeros(n_wav), np.abs(np.around(shift_applied[0:-1], decimals=2))
        )


class TestWaveformExtractorBin(unittest.TestCase):
    ns = 38502
    nc = 385
    n_clusters = 2
    ns_extract = 128
    max_wf = 25

    # 2 clusters
    spike_samples = np.repeat(
        np.arange(0, ns, 1600), 2
    )  # 50 spikes, but 2 of them are on 0 sample
    spike_channels = np.tile(np.array([100, 368]), 25)
    spike_clusters = np.tile(np.array([1, 2]), 25)

    def setUp(self):
        self.workdir = TEST_PATH
        self.tmpdir = Path(tempfile.gettempdir()) / "test_wfs"
        self.tmpdir.mkdir(exist_ok=True)
        self.bin_file = self.tmpdir.joinpath("wfs_test.bin")
        data = np.tile(np.arange(0, 385), (1, self.ns)).astype(np.float32)
        data.tofile(self.bin_file)

        h = trace_header()
        self.geom = np.c_[h["x"], h["y"]]
        self.chan_map = utils.make_channel_index(self.geom)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _ground_truth_values(self):
        # here we have to hard-code 48 and 24 because the 2 first spikes are rejected since on sample 0
        nc_extract = self.chan_map.shape[1]
        gt_templates = (
            np.ones((self.n_clusters, nc_extract, self.ns_extract), np.float32) * np.nan
        )
        gt_waveforms = np.ones((48, nc_extract, self.ns_extract), np.float32) * np.nan

        c0_chans = self.chan_map[100].astype(np.float32)
        gt_templates[0, :, :] = np.tile(c0_chans, (self.ns_extract, 1)).T

        gt_waveforms[:24, :, :] = gt_templates[0]

        c1_chans = self.chan_map[368].astype(np.float32)
        c1_chans[c1_chans == 384] = np.nan
        gt_templates[1, :, :] = np.tile(c1_chans, (self.ns_extract, 1)).T
        gt_waveforms[24:, :, :] = gt_templates[1]

        return gt_templates, gt_waveforms

    def test_extract_waveforms_bin(self):
        output_files = waveform_extraction.extract_wfs_cbin(
            self.bin_file,
            self.tmpdir,
            self.spike_samples,
            self.spike_clusters,
            self.spike_channels,
            reader_kwargs={
                "ns": self.ns,
                "nc": self.nc,
                "nsync": 1,
                "dtype": "float32",
            },
            max_wf=self.max_wf,
            h=trace_header(),
            preprocess_steps=[],
        )
        assert len(output_files) == 4
        templates = np.load(self.tmpdir.joinpath("waveforms.templates.npy"))
        waveforms = np.load(self.tmpdir.joinpath("waveforms.traces.npy"))
        table = pd.read_parquet(self.tmpdir.joinpath("waveforms.table.pqt"))

        cluster_ids = table.cluster.unique()

        for i, u in enumerate(cluster_ids):
            inds = table[table.cluster == u].waveform_index.to_numpy()
            assert np.allclose(
                templates[i], np.nanmedian(waveforms[inds], axis=0), equal_nan=True
            )

        gt_templates, gt_waveforms = self._ground_truth_values()

        assert np.allclose(np.nan_to_num(gt_templates), np.nan_to_num(templates))
        assert np.allclose(np.nan_to_num(gt_waveforms), np.nan_to_num(waveforms))

        wfl = waveform_extraction.WaveformsLoader(self.tmpdir)

        wfs = wfl.load_waveforms(return_info=False)
        assert np.allclose(np.nan_to_num(waveforms), np.nan_to_num(wfs))

        labels = np.array([1, 2])
        indices = np.arange(10)

        # test the waveform loader
        wfs, info, channels = wfl.load_waveforms(labels=labels, indices=indices)

        # right waveforms
        assert np.allclose(
            np.nan_to_num(waveforms[:10, :]),
            np.nan_to_num(wfs[info["cluster"] == 1, :, :]),
        )
        assert np.allclose(
            np.nan_to_num(waveforms[25:35, :]),
            np.nan_to_num(wfs[info["cluster"] == 2, :, :]),
        )
        # right channels
        assert np.all(
            channels == self.chan_map[info.peak_channel.astype(int).to_numpy()]
        )


def test_wiggle():
    wav = generate_waveform()
    wav = wav / np.max(np.abs(wav)) * 120 * 1e-6
    fig, ax = plt.subplots(1, 2)
    waveforms.plot_wiggle(wav, scale=40 * 1e-6, ax=ax[0])
    waveforms.double_wiggle(wav, scale=40 * 1e-6, fs=30_000, ax=ax[1])
    plt.close("all")


def test_waveform_indices():
    wxy, inds = waveforms.get_waveforms_coordinates(
        trace_indices=np.array([22, 150, 370]), return_indices=True
    )
    np.testing.assert_array_equal(
        wxy.flatten()[::19],
        np.array(
            [
                59.0,
                140.0,
                59.0,
                320.0,
                11.0,
                1400.0,
                43.0,
                1580.0,
                27.0,
                3580.0,
                59.0,
                3760.0,
                np.nan,
            ]
        ),
    )
