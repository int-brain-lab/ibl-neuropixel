import numpy as np
import cupy as cp
import unittest

from neurodsp.fourier import fshift, channel_shift


class TestFourierAlignmentGpuCpuParity(unittest.TestCase):

    def test_parity(self):
        N_TIMES = 65600
        N_CHANNELS = 384

        data = np.random.randn(N_CHANNELS, N_TIMES) * 100
        sample_shifts = np.linspace(0, 0.9, N_CHANNELS)

        data_shifted_cpu = fshift(data, sample_shifts)
        data_shifted_gpu = channel_shift(cp.array(data), cp.array(sample_shifts))

        assert cp.allclose(data_shifted_gpu, data_shifted_cpu, atol=1e-5)
