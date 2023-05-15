import numpy as np
from neurodsp.waveforms import peak_trough_tip
from neurodsp.waveforms import get_array_peak, half_peak_duration, half_peak_point, recovery_point, recovery_slope, \
                               polarisation_slopes, dist_chanel_from_peak, spatial_spread_weighted
import pandas as pd
from pathlib import Path


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


def test_peak_through_tip_2d():
    arr = make_array_peak_through_tip()
    df = peak_trough_tip(arr[0, :, :])
    np.testing.assert_equal(df.shape[0], 1)


def test_peak_through_tip_3d():
    arr = make_array_peak_through_tip()
    df, arr_out = peak_trough_tip(arr, return_peak_trace=True)
    np.testing.assert_equal(arr_out, np.array([[1, 2, 5, 4, -5, -6, 5],
                                               [-8, -7, 7, 7, 4, 4, 4]]))

    np.testing.assert_equal(df.peak_trace_idx, np.array([1, 0]))
    np.testing.assert_equal(df.peak_time_idx, np.array([5, 0]))
    np.testing.assert_equal(df.peak_val, np.array([-6, -8]))

    np.testing.assert_equal(df.trough_time_idx, np.array([6, 2]))
    np.testing.assert_equal(df.trough_val, np.array([5, 7]))

    np.testing.assert_equal(df.tip_time_idx, np.array([2, 1]))
    np.testing.assert_equal(df.tip_val, np.array([5, -7]))


def compute_df_arr_peak(arr_in):
    # ----- Compute -----
    df = peak_trough_tip(arr_in)
    # Array peak
    arr_peak = get_array_peak(arr_in, df)  # this output is correct (manually inspected) ; should be saved for test
    # Half peak points
    df = half_peak_point(arr_peak, df)
    # Half peak duration
    df = half_peak_duration(df, fs=30000)
    # Recovery point
    df = recovery_point(arr_peak, df, idx_from_trough=5)
    # Slopes (this was not checked by eye but saved for future testing)
    df = polarisation_slopes(df, fs=30000)
    df = recovery_slope(df, fs=30000)
    return df, arr_peak


def test_halfpeak_slopes():
    # Load fixtures
    folder_save = Path('../int-brain-lab/ibl-neuropixel/src/tests/unit/cpu/fixtures/waveform_sample')
    arr_in = np.load(folder_save.joinpath('test_arr_in.npy'))
    test_arr_peak = np.load(folder_save.joinpath('test_arr_peak.npy'))
    test_df = pd.read_csv(folder_save.joinpath('test_df.csv'))
    test_df = test_df.drop("Unnamed: 0", axis=1)  # Dropping the "Unnamed: 0" column

    df, arr_peak = compute_df_arr_peak(arr_in)

    # Array peak testing
    np.testing.assert_equal(arr_peak, test_arr_peak)
    # Df testing
    pd.testing.assert_frame_equal(df.astype(float), test_df.astype(float))


def test_dist_chanel_from_peak():
    # Distance test
    xyz_testd = np.stack((np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]),
                          np.array([[4, 0, 0], [2, 0, 0], [np.NaN, np.NaN, np.NaN], [1, 0, 0]])), axis=2)

    xyz_testd = np.swapaxes(xyz_testd, axis1=0, axis2=2)
    xyz_testd = np.swapaxes(xyz_testd, axis1=1, axis2=2)
    df_testd = pd.DataFrame.from_dict({"peak_trace_idx": np.array([0, 1])})
    eu_dist = dist_chanel_from_peak(xyz_testd, df_testd)
    weights = np.zeros(eu_dist.shape)
    weights[0][1:3] = 1
    weights[1][2:4] = 1
    sp_spread = spatial_spread_weighted(eu_dist, weights)

    np.testing.assert_almost_equal(eu_dist[0, :], np.array([0, 1, 1, np.sqrt(2)]))
    np.testing.assert_equal(eu_dist[1, :], np.array([2, 0, np.NaN, 1]))
    np.testing.assert_equal(sp_spread, np.array([1, 0.5]))
