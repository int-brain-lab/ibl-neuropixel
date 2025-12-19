import unittest

import numpy as np

import ibldsp.cadzow
import neuropixel


class TestCadzow(unittest.TestCase):
    def test_trajectory_matrixes_indices(self):
        assert np.all(
            ibldsp.cadzow.traj_matrix_indices(4) == np.array([[1, 0], [2, 1], [3, 2]])
        )
        assert np.all(
            ibldsp.cadzow.traj_matrix_indices(3) == np.array([[1, 0], [2, 1]])
        )

    def test_trajectory(self):
        th = neuropixel.trace_header(version=1)
        tm, it, ic, trcount = ibldsp.cadzow.trajectory(
            th["x"][:8], th["y"][:8], dtype=np.int32
        )
        # make sure tm can be indexed
        np.testing.assert_array_equal(tm[it], 0)
