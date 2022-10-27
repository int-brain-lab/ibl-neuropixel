import numpy as np
from neurodsp.waveforms import peak_trough_tip


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
