from pathlib import Path

import numpy as np
import pandas as pd

import ibldsp.utils as utils
import ibldsp.waveforms as waveforms
from neurowaveforms.model import generate_waveform
from neuropixel import trace_header
from ibldsp.fourier import fshift
import scipy

import unittest


def make_array_peak_through_tip():
    arr = np. array([[[1, 1, np.nan],
                      [2, 2, np.nan],
                      [3, 5, np.nan],
                      [4, 4, np.nan],
                      [4, -5, np.nan],
                      [4, -6, np.nan],
                      [4, 5, np.nan]],

                     [[-8, 7, 7],
                      [-7, 7, 7],
                      [7, 7, 7],
                      [7, 7, 7],
                      [4, 5, 4],
                      [4, 5, 4],
                      [4, 5, 4]]])
    return arr


def make_array_peak_through_tip_v2():
    # Duplicating as above array throws error due to Nans when computing arr_pre
    arr = np. array([[[1, 1, np.nan],
                      [2, 2, np.nan],
                      [3, 5, np.nan],
                      [4, 4, np.nan],
                      [4, -5, np.nan],
                      [4, -6, np.nan],
                      [4, 5, np.nan]],

                    [[1, 1, 0],
                     [2, 2, 0],
                     [3, 5, 0],
                     [4, 4, 0],
                     [4, -5, 0],
                     [4, -8, 1],
                     [4, 5, 0]]])
    return arr


def test_peak_through_tip_2d():
    arr = make_array_peak_through_tip()
    df = waveforms.compute_spike_features(arr[0, :, :])
    np.testing.assert_equal(df.shape[0], 1)


def test_peak_through_tip_3d():
    arr = make_array_peak_through_tip_v2()
    df = waveforms.compute_spike_features(arr)
    arr_out = waveforms.get_array_peak(arr, df)

    np.testing.assert_equal(arr_out, np.array([[1, 2, 5, 4, -5, -6, 5],
                                               [1, 2, 5, 4, -5, -8, 5]]))

    np.testing.assert_equal(df.peak_trace_idx, np.array([1, 1]))
    np.testing.assert_equal(df.peak_time_idx, np.array([5, 5]))
    np.testing.assert_equal(df.peak_val, np.array([-6, -8]))

    np.testing.assert_equal(df.trough_time_idx, np.array([6, 6]))
    np.testing.assert_equal(df.trough_val, np.array([5, 5]))

    np.testing.assert_equal(df.tip_time_idx, np.array([2, 2]))
    np.testing.assert_equal(df.tip_val, np.array([5, 5]))


def test_halfpeak_slopes():
    # Load fixtures
    folder_save = Path(waveforms.__file__).parents[1].joinpath('tests', 'unit', 'cpu', 'fixtures', 'waveform_sample')
    arr_in = np.load(folder_save.joinpath('test_arr_in.npy'))
    test_arr_peak = np.load(folder_save.joinpath('test_arr_peak.npy'))
    test_df = pd.read_csv(folder_save.joinpath('test_df.csv'))
    test_df = test_df.drop("Unnamed: 0", axis=1)  # Dropping the "Unnamed: 0" column
    df = waveforms.compute_spike_features(arr_in, fs=30000, recovery_duration_ms=.16)
    # Array peak testing
    arr_peak = waveforms.get_array_peak(arr_in, df)
    np.testing.assert_equal(arr_peak, test_arr_peak)
    # Df testing
    pd.testing.assert_frame_equal(df.astype(float), test_df.astype(float))


def test_dist_chanel_from_peak():
    # Distance test
    xyz_testd = np.stack((np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]),
                          np.array([[4, 0, 0], [2, 0, 0], [np.NaN, np.NaN, np.NaN], [1, 0, 0]])), axis=2)
    xyz_testd = np.swapaxes(xyz_testd, axis1=0, axis2=2)
    xyz_testd = np.swapaxes(xyz_testd, axis1=1, axis2=2)

    eu_dist = waveforms.dist_chanel_from_peak(xyz_testd, np.array([0, 1]))
    weights = np.zeros(eu_dist.shape)
    weights[0][1:3] = 1
    weights[1][2:4] = 1
    sp_spread = waveforms.spatial_spread_weighted(eu_dist, weights)

    np.testing.assert_almost_equal(eu_dist[0, :], np.array([0, 1, 1, np.sqrt(2)]))
    np.testing.assert_equal(eu_dist[1, :], np.array([2, 0, np.NaN, 1]))
    np.testing.assert_equal(sp_spread, np.array([1, 0.5]))


def test_reshape_wav_one_channel():
    # Test reshape into 1 channel
    arr = make_array_peak_through_tip()
    arr_out = waveforms.reshape_wav_one_channel(arr)
    # Check against test
    arr_tested = np.array([[[1.],
                            [2.],
                            [3.],
                            [4.],
                            [4.],
                            [4.],
                            [4.]],
                           [[1.],
                            [2.],
                            [5.],
                            [4.],
                            [-5.],
                            [-6.],
                            [5.]],
                           [[np.nan],
                            [np.nan],
                            [np.nan],
                            [np.nan],
                            [np.nan],
                            [np.nan],
                            [np.nan]],
                           [[-8.],
                            [-7.],
                            [7.],
                            [7.],
                            [4.],
                            [4.],
                            [4.]],
                           [[7.],
                            [7.],
                            [7.],
                            [7.],
                            [5.],
                            [5.],
                            [5.]],
                           [[7.],
                            [7.],
                            [7.],
                            [7.],
                            [4.],
                            [4.],
                            [4.]]])
    np.testing.assert_equal(arr_out, arr_tested)


def test_weights_all_channels():
    arr = make_array_peak_through_tip()
    weight = waveforms.weights_spk_ch(arr)
    weight_tested = np.array([[4., -6., 0.],
                              [-8., 7., 7.]])

    np.testing.assert_equal(weight, weight_tested)


def test_generate_waveforms():
    wav = generate_waveform()
    assert wav.shape == (121, 40)


class TestWaveformExtractor(unittest.TestCase):
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
    channel_neighbors = utils.make_channel_index(geom, radius=200.)
    # radius = 200um, 38 chans
    num_channels = 38

    def test_extract_waveforms(self):
        wfs, _, _ = waveforms.extract_wfs_array(self.arr, self.df, self.channel_neighbors)

        # first wf is a special case: it's at the top of the probe so the center
        # index is the actual channel index, and the rest of the wf has been padded
        # with NaNs
        assert wfs[0, self.channels[0], self.trough_offset] == 1.
        assert np.all(np.isnan(wfs[0, self.num_channels // 2 + self.channels[0] + 1:, :]))

        for i in range(1, 8):
            # center channel depends on odd/even of channel
            if self.channels[i] % 2 == 0:
                centered_channel_idx = 18
            else:
                centered_channel_idx = 19
            assert wfs[i, centered_channel_idx, self.trough_offset] == float(i + 1)

        # last wf is a special case analogous to the first wf, but at the bottom
        # of the probe
        if self.channels[-1] % 2 == 0:
            centered_channel_idx = 18
        else:
            centered_channel_idx = 19
        assert wfs[-1, centered_channel_idx, self.trough_offset] == 9.

    def test_spike_window(self):
        # check that we have an error when the last spike window
        # extends past end of recording
        df = self.df.copy()
        df["sample"].iloc[-1] = 996
        with self.assertRaisesRegex(AssertionError, "extends"):
            _ = waveforms.extract_wfs_array(self.arr, df, self.channel_neighbors)

    def test_nan_channel(self):
        # test that if user does not fill last column with NaNs
        # the user can set the flag and the result will be the same
        arr = self.arr.copy()[:, :-1]
        wfs = waveforms.extract_wfs_array(self.arr, self.df, self.channel_neighbors)
        wfs_nan = waveforms.extract_wfs_array(arr, self.df, self.channel_neighbors, add_nan_trace=True)
        np.testing.assert_equal(wfs, wfs_nan)

    def test_wave_shift_corrmax(self):
        sample_shifts = [4.43, -1.0]
        sig_lens = [100, 101]
        for sample_shift in sample_shifts:
            for sig_len in sig_lens:
                spike = scipy.signal.morlet2(sig_len, 8.0, 2.0)
                spike = -np.fft.irfft(np.fft.rfft(spike) * np.exp(1j * 45 / 180 * np.pi))

                spike2 = fshift(spike, sample_shift)
                spike3, shift_computed = waveforms.wave_shift_corrmax(spike, spike2)

                np.testing.assert_equal(sample_shift, np.around(shift_computed, decimals=2))

    def test_wave_shift_phase(self):
        fs = 30000
        # Resynch in time spike2 onto spike
        sample_shift_original = 0.323
        spike = scipy.signal.morlet2(100, 8.5, 2.0)
        spike = -np.fft.irfft(np.fft.rfft(spike) * np.exp(1j * 45 / 180 * np.pi))
        spike = np.append(spike, np.zeros((1, 25)))
        spike2 = fshift(spike, sample_shift_original)
        # Resynch
        spike_resync, sample_shift_applied = waveforms.wave_shift_phase(spike, spike2, fs)
        np.testing.assert_equal(sample_shift_original, np.around(sample_shift_applied, decimals=3))
