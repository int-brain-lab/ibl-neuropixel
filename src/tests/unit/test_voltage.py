import numpy as np
import tempfile
from pathlib import Path
import unittest

import pandas as pd

import spikeglx
import ibldsp.voltage
import ibldsp.fourier
import ibldsp.utils
import ibldsp.cadzow


class TestDestripe(unittest.TestCase):
    def test_destripe_parameters(self):
        import inspect

        _, _, spatial_fcn = ibldsp.voltage._get_destripe_parameters(
            30_000, None, None, k_filter=True
        )
        assert "kfilt" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = ibldsp.voltage._get_destripe_parameters(
            2_500, None, None, k_filter=False
        )
        assert "car" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = ibldsp.voltage._get_destripe_parameters(
            2_500, None, None, k_filter=None
        )
        assert "dat: dat" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = ibldsp.voltage._get_destripe_parameters(
            2_500, None, None, k_filter=lambda dat: 3 * dat
        )
        assert "lambda dat: 3 * dat" in inspect.getsource(spatial_fcn)

    def test_fk(self):
        """
        creates a couple of plane waves and separate them using the velocity HP filter
        """
        ntr, ns, sr, dx, v1, v2 = (500, 2000, 0.002, 5, 2000, 1000)
        data = np.zeros((ntr, ns), np.float32)
        data[:, :100] = ibldsp.utils.ricker(100, 4)
        offset = np.arange(ntr) * dx
        offset = np.abs(offset - np.mean(offset))
        data_v1 = ibldsp.fourier.fshift(data, offset / v1 / sr)
        data_v2 = ibldsp.fourier.fshift(data, offset / v2 / sr)

        noise = np.random.randn(ntr, ns) / 60
        fk = ibldsp.voltage.fk(
            data_v1 + data_v2 + noise,
            si=sr,
            dx=dx,
            vbounds=[1200, 1500],
            ntr_pad=10,
            ntr_tap=15,
            lagc=0.25,
        )
        fknoise = ibldsp.voltage.fk(
            noise, si=sr, dx=dx, vbounds=[1200, 1500], ntr_pad=10, ntr_tap=15, lagc=0.25
        )
        # at least 90% of the traces should be below 50dB and 98% below 40 dB
        assert (
            np.mean(20 * np.log10(ibldsp.utils.rms(fk - data_v1 - fknoise)) < -50) > 0.9
        )
        assert (
            np.mean(20 * np.log10(ibldsp.utils.rms(fk - data_v1 - fknoise)) < -40)
            > 0.98
        )
        # test the K option
        kbands = np.sin(np.arange(ns) / ns * 8 * np.pi) / 10
        fkk = ibldsp.voltage.fk(
            data_v1 + data_v2 + kbands,
            si=sr,
            dx=dx,
            vbounds=[1200, 1500],
            ntr_pad=40,
            ntr_tap=15,
            lagc=0.25,
            kfilt={"bounds": [0, 0.01], "btype": "hp"},
        )
        assert np.mean(20 * np.log10(ibldsp.utils.rms(fkk - data_v1)) < -40) > 0.9
        # from easyqc.gui import viewseis
        # a = viewseis(data_v1 + data_v2 + kbands, .002, title='input')
        # b = viewseis(fkk, .002, title='output')
        # c = viewseis(data_v1 - fkk, .002, title='test')


class TestSaturation(unittest.TestCase):
    def test_saturation_cbin(self):
        nsat = 252
        ns, nc = (350_072, 384)
        s2v = np.float32(2.34375e-06)
        sat = ibldsp.utils.fcn_cosine([0, 100])(
            np.arange(nsat)
        ) - ibldsp.utils.fcn_cosine([150, 250])(np.arange(nsat))
        range_volt = 0.0012
        sat = (sat / s2v * 0.0012).astype(np.int16)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_bin = Path(temp_dir) / "binary.bin"
            data = np.memmap(file_bin, dtype=np.int16, mode="w+", shape=(ns, nc))
            data[50_000 : 50_000 + nsat, :] = sat[:, np.newaxis]

            _sr = spikeglx.Reader(
                file_bin, fs=30_000, dtype=np.int16, nc=nc, nsync=0, s2v=s2v
            )
            file_saturation = ibldsp.voltage.saturation_cbin(
                _sr, max_voltage=range_volt, n_jobs=1
            )
            df_sat = pd.read_parquet(file_saturation)
            assert np.sum(df_sat["stop_sample"] - df_sat["start_sample"]) == 67

    def test_saturation(self):
        np.random.seed(7654)
        data = (np.random.randn(384, 30_000).astype(np.float32) + 20) * 1e-6
        saturated, mute = ibldsp.voltage.saturation(data, max_voltage=1200)
        np.testing.assert_array_equal(saturated, 0)
        np.testing.assert_array_equal(mute, 1.0)
        # now we stick a big waveform in the middle of the recorder and expect some saturation
        w = ibldsp.utils.ricker(100, 4)
        w = np.minimum(1200, w / w.max() * 1400)
        data[:, 13_600:13700] = data[0, 13_600:13700] + w * 1e-6
        saturated, mute = ibldsp.voltage.saturation(
            data,
            max_voltage=np.ones(
                384,
            )
            * 1200
            * 1e-6,
        )
        self.assertGreater(np.sum(saturated), 5)
        self.assertGreater(np.sum(mute == 0), np.sum(saturated))


class TestCadzow(unittest.TestCase):
    def test_trajectory_matrixes(self):
        assert np.all(
            ibldsp.cadzow.traj_matrix_indices(4) == np.array([[1, 0], [2, 1], [3, 2]])
        )
        assert np.all(
            ibldsp.cadzow.traj_matrix_indices(3) == np.array([[1, 0], [2, 1]])
        )
