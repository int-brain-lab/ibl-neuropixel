from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np
import pandas as pd
import scipy.signal

import neuropixel
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

    def test_saturation_intervals_boundary_edges(self):
        # recordings that start and/or end saturated must not raise (diff-based fronts
        # misses boundary edges): a leading rising / trailing falling edge is inserted.
        n = 1000
        sat = np.zeros(n, dtype=bool)
        sat[:50] = True  # starts saturated
        sat[900:] = True  # ends saturated
        df = ibldsp.voltage.saturation_samples_to_intervals(sat)
        self.assertEqual(df.shape[0], 2)
        np.testing.assert_array_equal(df["start_sample"].to_numpy(), [0, 900])
        np.testing.assert_array_equal(df["stop_sample"].to_numpy(), [50, n - 1])
        # fully saturated → a single interval spanning the recording
        df_full = ibldsp.voltage.saturation_samples_to_intervals(np.ones(n, dtype=bool))
        self.assertEqual(df_full.shape[0], 1)
        np.testing.assert_array_equal(
            df_full[["start_sample", "stop_sample"]].to_numpy(), [[0, n - 1]]
        )


class TestInterpolateBadChannels(unittest.TestCase):
    """groupby_y=True: exact per-column linear interpolation along y, ignoring
    geometrically-closer cross-column neighbours. Motivated by current_source_density,
    which differentiates each column independently and is disrupted by a channel filled
    in from a different column's trend."""

    def setUp(self):
        self.h = neuropixel.trace_header(version=1)
        self.nc = self.h["x"].size
        self.x, self.y = self.h["x"], self.h["y"]
        # a function that is linear in y: exact two-point linear interpolation must
        # reproduce it exactly at any interior missing sample.
        f = 3.0 * self.y + 7.0
        self.data = np.tile(f[:, np.newaxis], (1, 5)).astype(np.float64)

    def test_isolated_bad_channel_exact_recovery(self):
        labels = np.zeros(self.nc, dtype=int)
        labels[193] = 1
        out = ibldsp.voltage.interpolate_bad_channels(
            self.data.copy(), labels, self.x, self.y, groupby_y=True
        )
        np.testing.assert_allclose(out[193], self.data[193])

    def test_adjacent_same_column_bad_channels_exact_recovery(self):
        # 250 and 254 are adjacent within the same column (x[250] == x[254]).
        assert self.x[250] == self.x[254]
        labels = np.zeros(self.nc, dtype=int)
        labels[[250, 254]] = 1
        out = ibldsp.voltage.interpolate_bad_channels(
            self.data.copy(), labels, self.x, self.y, groupby_y=True
        )
        np.testing.assert_allclose(out[[250, 254]], self.data[[250, 254]])

    def test_ignores_closer_cross_column_neighbour(self):
        # channel 191 is geometrically closer to 193 than 193's same-column
        # neighbours (189, 195, 197) are - groupby_y=True must not use it.
        labels = np.zeros(self.nc, dtype=int)
        labels[193] = 1
        corrupted = self.data.copy()
        corrupted[191] = 1e6  # would dominate a distance-weighted kriging fit
        out = ibldsp.voltage.interpolate_bad_channels(
            corrupted, labels, self.x, self.y, groupby_y=True
        )
        np.testing.assert_allclose(out[193], self.data[193])

    def test_edge_channel_falls_back_to_single_neighbour(self):
        # the top-most channel of a column has no neighbour above it.
        col0 = np.where(self.x == self.x[0])[0]
        top = col0[np.argmin(self.y[col0])]
        below = col0[self.y[col0] == np.sort(self.y[col0])[1]][0]
        labels = np.zeros(self.nc, dtype=int)
        labels[top] = 1
        out = ibldsp.voltage.interpolate_bad_channels(
            self.data.copy(), labels, self.x, self.y, groupby_y=True
        )
        np.testing.assert_allclose(out[top], self.data[below])

    def test_default_kriging_path_unaffected(self):
        # groupby_y defaults to False: behaviour must match the original
        # multi-column kriging interpolation, including the all-neighbours-bad
        # fallback to zero (previously reached via a NaN division, now explicit).
        labels = np.zeros(self.nc, dtype=int)
        labels[:5] = 1  # first few channels: neighbours within range are also bad
        out = ibldsp.voltage.interpolate_bad_channels(
            self.data.copy(), labels, self.x, self.y
        )
        self.assertTrue(np.all(np.isfinite(out)))


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
            out_file = Path(temp_dir).joinpath("test_rs.npy")
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
            ibldsp.voltage.resample_denoise_lfp_cbin(
                sr,
                output=out_file,
                dtype=np.float32,
                highpass_cutoff=None,
                car=False,
            )
            za = spikeglx.Reader(out_file, ns=ns // 5, nc=nc, fs=500, dtype=np.float32)
            diff = d[0:-1:resamp_factor_q, :] - za[:]
            np.testing.assert_array_less(np.abs(diff[1024:-1024] / 1000), 1e-3)

    def test_rsamp_cbin_saturation_mute(self):
        """A saturation_file mutes the flagged raw samples before filtering: the decimated
        output over the (tapered) saturated span is attenuated to ~zero, while an
        unmuted run leaves that span at full amplitude."""
        ns, nc, fs, q = int(20 * 2500), 8, 2500, 5
        with tempfile.TemporaryDirectory() as temp_dir:
            testfile = Path(temp_dir).joinpath("test.dat")
            sat_file = Path(temp_dir).joinpath("sat.npy")
            d = np.zeros((ns, nc), dtype=np.float32)
            for ic in range(nc):
                d[:, ic] = np.sin(2 * np.pi * (10 + ic * 4) * np.arange(ns) / fs) * 1000
            d.tofile(testfile)
            # flag a saturated span well inside the recording (avoid chunk/filter edges)
            sat = np.zeros(ns, dtype=bool)
            sat[10 * 2500 : 11 * 2500] = True
            np.save(sat_file, sat)

            sr = spikeglx.Reader(testfile, ns=ns, nc=nc, fs=fs, dtype=np.float32)
            out_plain = Path(temp_dir).joinpath("plain.npy")
            out_muted = Path(temp_dir).joinpath("muted.npy")
            for out, sf in [(out_plain, None), (out_muted, sat_file)]:
                ibldsp.voltage.resample_denoise_lfp_cbin(
                    sr,
                    output=out,
                    dtype=np.float32,
                    highpass_cutoff=None,
                    car=False,
                    saturation_file=sf,
                )
            za_plain = spikeglx.Reader(
                out_plain, ns=ns // q, nc=nc, fs=fs / q, dtype=np.float32
            )[:]
            za_muted = spikeglx.Reader(
                out_muted, ns=ns // q, nc=nc, fs=fs / q, dtype=np.float32
            )[:]
            # decimated indices of the muted span core (inside the taper)
            core = slice(int(10.2 * 500), int(10.8 * 500))
            self.assertLess(np.abs(za_muted[core]).max(), 1.0)  # muted ≈ 0
            self.assertGreater(
                np.abs(za_plain[core]).max(), 100.0
            )  # unmuted full amplitude
            # outside the saturated span the two runs agree
            outside = slice(0, int(9.0 * 500))
            np.testing.assert_allclose(za_muted[outside], za_plain[outside], atol=1e-3)

    def test_rsamp_cbin_cadzow_reinterpolation(self):
        """channel_labels + cadzow_kwargs together: the bad channels Cadzow re-estimates
        must be overwritten by an exact per-column linear interpolation of their
        Cadzow-denoised neighbours, so the pipeline output is a fixed point of
        interpolate_bad_channels(groupby_y=True)."""
        ns, nc, fs = int(10 * 2500), 16, 2500
        with tempfile.TemporaryDirectory() as temp_dir:
            testfile = Path(temp_dir).joinpath("test.dat")
            out_file = Path(temp_dir).joinpath("test_rs.npy")
            d = np.zeros((ns, nc), dtype=np.float32)
            for ic in range(nc):
                d[:, ic] = np.sin(2 * np.pi * (10 + ic * 4) * np.arange(ns) / fs) * 1000
            d.tofile(testfile)

            sr = spikeglx.Reader(testfile, ns=ns, nc=nc, fs=fs, dtype=np.float32)
            h = neuropixel.trace_header(version=1)
            x, y = h["x"][:nc], h["y"][:nc]
            self.assertEqual(x[4], x[8])  # 4 & 8 adjacent within the same column
            channel_labels = np.zeros(nc, dtype=int)
            channel_labels[[4, 8, 9]] = 1

            ibldsp.voltage.resample_denoise_lfp_cbin(
                sr,
                output=out_file,
                dtype=np.float32,
                highpass_cutoff=None,
                car=False,
                channel_labels=channel_labels,
                cadzow_kwargs=dict(rank=3, fmax=None, nswx=8, ovx=4),
            )
            za = spikeglx.Reader(
                out_file, ns=ns // 5, nc=nc, fs=fs / 5, dtype=np.float32
            )
            out = za[:].T.astype(np.float64)
            self.assertTrue(np.all(np.isfinite(out)))
            fixed_point = ibldsp.voltage.interpolate_bad_channels(
                out.copy(), channel_labels, x, y, groupby_y=True
            )
            np.testing.assert_allclose(out, fixed_point, rtol=1e-6)


class TestDetectBadChannels(unittest.TestCase):
    @staticmethod
    def a_little_spike(nsw=121, nc=1):
        # creates a kind of waveform that resembles a spike
        wav = np.zeros(nsw)
        wav[0] = 1
        wav[5] = -0.1
        wav[10] = -0.3
        wav[15] = -0.1
        sos = scipy.signal.butter(N=3, Wn=0.15, output="sos")
        spike = scipy.signal.sosfilt(sos, wav)
        spike = -spike / np.max(spike)
        if nc > 1:
            spike = spike[:, np.newaxis] * scipy.signal.hamming(nc)[np.newaxis, :]
        return spike

    @staticmethod
    def make_synthetic_data(
        ns=10000, nc=384, nss=121, ncs=21, nspikes=1200, tr=None, sample=None
    ):
        if tr is None:
            tr = np.random.randint(np.ceil(ncs / 2), nc - np.ceil(ncs / 2), nspikes)
        if sample is None:
            sample = np.random.randint(np.ceil(nss / 2), ns - np.ceil(nss / 2), nspikes)
        h = neuropixel.trace_header(1)
        icsmid = int(np.floor(ncs / 2))
        issmid = int(np.floor(nss / 2))
        template = TestDetectBadChannels.a_little_spike(121)
        data = np.zeros((ns, nc))
        for m in np.arange(tr.size):
            itr = np.arange(tr[m] - icsmid, tr[m] + icsmid + 1)
            iss = np.arange(sample[m] - issmid, sample[m] + issmid + 1)
            offset = np.abs(
                h["x"][itr[icsmid]]
                + 1j * h["y"][itr[icsmid]]
                - h["x"][itr]
                - 1j * h["y"][itr]
            )
            ampfac = 1 / (offset + 10) ** 1.3
            ampfac = ampfac / np.max(ampfac)
            tmp = template[:, np.newaxis] * ampfac[np.newaxis, :]
            data[slice(iss[0], iss[-1] + 1), slice(itr[0], itr[-1] + 1)] += tmp
        return data

    @staticmethod
    def synthetic_with_bad_channels():
        np.random.seed(12345)
        ns, nc, fs = (30000, 384, 30000)
        data = TestDetectBadChannels.make_synthetic_data(ns=ns, nc=nc) * 1e-6 * 50

        st = np.round(
            np.cumsum(-np.log(np.random.rand(int(ns / fs * 50 * 1.5))) / 50) * fs
        )
        st = st[st < ns].astype(np.int32)
        stripes = np.zeros(ns)
        stripes[st] = 1
        stripes = (
            scipy.signal.convolve(stripes, ibldsp.utils.ricker(1200, 40), "same")
            * 1e-6
            * 2500
        )

        data = data + stripes[:, np.newaxis]
        noise = np.random.randn(*data.shape) * 1e-6 * 10

        channels = {
            "idead": [29, 36, 39, 40, 191],
            "inoisy": [133, 235],
            "ioutside": np.arange(275, 384),
        }

        data[:, channels["idead"]] = data[:, channels["idead"]] / 20
        noise[:, channels["inoisy"]] = noise[:, channels["inoisy"]] * 200
        data[:, channels["idead"]] = data[:, channels["idead"]] / 20
        data[:, channels["ioutside"]] = 0
        data += noise
        return data, channels

    def test_channel_detections(self):
        """
        This test creates a synthetic dataset with voltage stripes and 3 types of bad channels
        1) dead channels or low amplitude
        2) noisy
        3) out of the brain
        """
        data, channels = self.synthetic_with_bad_channels()
        labels, xfeats = ibldsp.voltage.detect_bad_channels(data.T, fs=30000)
        assert np.all(np.where(labels == 1)[0] == np.array(channels["idead"]))
        assert np.all(np.where(labels == 2)[0] == np.array(channels["inoisy"]))
        assert np.all(np.where(labels == 3)[0] == np.array(channels["ioutside"]))
        # from easyqc.gui import viewseis
        # eqc = viewseis(data, si=1 / 30000 * 1e3, h=h, title='synth', taxis=0)
        # from ibllib.plots.figures import ephys_bad_channels
        # ephys_bad_channels(data.T, 30000, labels, xfeats)
