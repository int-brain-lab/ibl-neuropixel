import numpy as np
import cupy as cp
from scipy.signal import butter, sosfiltfilt
import unittest

from neurodsp.filter_gpu import sosfiltfilt_gpu


class TestFilterGpuCpuParity(unittest.TestCase):

    def test_parity(self):

        GPU_TOL = 1e-3
        BUTTER_KWARGS = {'N': 3, 'Wn': 300 / 30000 * 2, 'btype': 'highpass'}
        N_SIGNALS = 300
        N_SAMPLES = 60000

        sos = butter(**BUTTER_KWARGS, output='sos')
        array_cpu = np.cumsum(np.random.randn(N_SIGNALS, N_SAMPLES), axis=1)
        array_gpu = cp.array(array_cpu, dtype='float32')

        output_cpu = sosfiltfilt(sos, array_cpu)
        output_gpu = sosfiltfilt_gpu(sos, array_gpu)

        assert cp.allclose(output_cpu, output_gpu, atol=GPU_TOL)
