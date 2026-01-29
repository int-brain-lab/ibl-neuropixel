from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np
import pandas as pd

import spikeglx
import ibldsp.voltage
import ibldsp.fourier
import ibldsp.utils
import ibldsp.cadzow


class TestDestripe(unittest.TestCase):
    def test_destripe_parameters(self):
        # ibldsp.voltage.apply_spatial_filter(x, k_filter, **kwargs)
        x = np.random.randn(100, 1000)

        # K-filter = True calls the spatial filter function
        _, k_kwargs = ibldsp.voltage._get_destripe_parameters(
            30_000, None, None, k_filter=True
        )
        with unittest.mock.patch("ibldsp.voltage.kfilt") as mock_fcn_spatial_filter:
            ibldsp.voltage.apply_spatial_filter(x, k_filter=True, **k_kwargs)
            # Assert the function was called
            mock_fcn_spatial_filter.assert_called_once()

        # K-filter = False calls the CAR function
        with unittest.mock.patch("ibldsp.voltage.car") as mock_fcn_spatial_filter:
            ibldsp.voltage.apply_spatial_filter(x, k_filter=False, **k_kwargs)
            # Assert the function was called
            mock_fcn_spatial_filter.assert_called_once()

        # K-filter = None does not apply any filtering
        xx = ibldsp.voltage.apply_spatial_filter(x, k_filter=None, **k_kwargs)
        np.testing.assert_array_equal(xx, x)

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
        n_expected_samples = np.sum(sat > 0.96)
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
            assert (
                np.sum(df_sat["stop_sample"] - df_sat["start_sample"])
                == n_expected_samples
            )

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

    def test_saturation_intervals_output(self):
        saturation = np.zeros(50_000, dtype=bool)
        # we test empty files, make sure we can read/write from empty parquet
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file path within the temporary directory
            temp_file = Path(temp_dir).joinpath("saturation.pqt")
            df_nothing = ibldsp.voltage.saturation_samples_to_intervals(
                saturation, output_file=Path(temp_dir).joinpath("saturation.pqt")
            )
            df_nothing2 = pd.read_parquet(temp_file)
        self.assertEqual(df_nothing.shape[0], 0)
        self.assertEqual(df_nothing2.shape[0], 0)
        # for the case with saturation intervals, we simply test the number of rows correspond to the events
        saturation[3441:3509] = True
        saturation[45852:45865] = True
        df_sat = ibldsp.voltage.saturation_samples_to_intervals(saturation)
        self.assertEqual(81, np.sum(df_sat["stop_sample"] - df_sat["start_sample"]))


class TestLFP(unittest.TestCase):
    def test_rsamp_cbin(self):
        """
        Resamples a binary file by a factor of 5
        :return:
        """
        ns = int(125.83948 * 2500) + 3
        nc = 12
        fs = 2500
        resamp_factor_q = 5
        with tempfile.TemporaryDirectory() as temp_dir:
            testfile = Path(temp_dir).joinpath("test.dat")
            out_file = Path(temp_dir).joinpath("test_rs.dat")
            # testfile = Path.home().joinpath('lfp', 'test.dat')
            # out_file = Path.home().joinpath('lfp', 'test_rs.dat')
            testfile.parent.mkdir(exist_ok=True, parents=True)
            d = np.zeros((ns, nc), dtype=np.float32)
            for ic in range(nc):
                freq = 10 + ic * 4
                print(freq)
                d[:, ic] = np.sin(2 * np.pi * freq * np.arange(ns) / fs) * 1000

            with open(testfile, "wb+") as f:
                d.tofile(f)

            sr = spikeglx.Reader(testfile, ns=ns, nc=nc, fs=2500, dtype=np.float32)
            ibldsp.voltage.resample_denoise_lfp_cbin(sr, output=out_file)
            za = spikeglx.Reader(out_file, ns=ns // 5, nc=nc, fs=500, dtype=np.float32)

            diff = d[0:-1:resamp_factor_q, :] - za[:]
            np.testing.assert_array_less(np.abs(diff[1024:-1024] / 1000), 1e-3)
