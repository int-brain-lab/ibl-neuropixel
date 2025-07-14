import unittest

import numpy as np

import ibldsp.plots
import ibldsp.voltage


class TestPlots(unittest.TestCase):
    def test_voltage(self):
        ibldsp.plots.voltageshow(
            (np.random.rand(384, 2000) - 0.5) / 1e6 * 20, fs=30_000
        )

    def test_bad_channels(self):
        np.random.seed(0)
        raw = np.random.randn(384, 2000) / 1e6 * 15
        raw += np.random.randn(1, 2000) / 1e6 * 2
        raw[66] *= 2
        raw[166] = 0
        fs = 30_000
        labels, features = ibldsp.voltage.detect_bad_channels(raw, fs)
        ibldsp.plots.show_channels_labels(
            raw=raw,
            fs=30_000,
            channel_labels=labels,
            xfeats=features,
        )
        np.testing.assert_array_equal(np.argwhere(labels == 2), 66)
        np.testing.assert_array_equal(np.argwhere(labels == 1), 166)
