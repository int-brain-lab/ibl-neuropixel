import unittest
import numpy as np
import scipy.signal
import scipy.fft
import spikeglx

import ibldsp.fourier as fourier
import ibldsp.utils as utils
import ibldsp.voltage as voltage
import ibldsp.cadzow as cadzow
import ibldsp.smooth as smooth
import ibldsp.spiketrains as spiketrains
import ibldsp.raw_metrics as raw_metrics

from pathlib import Path
import tempfile
import shutil

FIXTURE_PATH = Path(__file__).parents[1].joinpath("fixtures")


class TestSyncTimestamps(unittest.TestCase):
    def test_sync_timestamps_linear(self):
        ta = np.cumsum(np.abs(np.random.randn(100))) * 10
        tb = ta * 1.0001 + 100
        fcn, drif, ia, ib = utils.sync_timestamps(
            ta, tb, return_indices=True, linear=True
        )
        np.testing.assert_almost_equal(drif, 100)
        np.testing.assert_almost_equal(tb, fcn(ta))

    def test_timestamps(self):
        np.random.seed(4132)
        n = 50
        drift = 17.14
        offset = 34.323
        tsa = np.cumsum(np.random.random(n) * 10)
        tsb = tsa * (1 + drift / 1e6) + offset

        # test linear drift
        _fcn, _drift = utils.sync_timestamps(tsa, tsb)
        assert np.all(np.isclose(_fcn(tsa), tsb))
        assert np.isclose(drift, _drift)

        # test missing indices on a
        imiss = np.setxor1d(np.arange(n), [1, 2, 34, 35])
        _fcn, _drift, _ia, _ib = utils.sync_timestamps(
            tsa[imiss], tsb, return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[_ib]))

        # test missing indices on b
        _fcn, _drift, _ia, _ib = utils.sync_timestamps(
            tsa, tsb[imiss], return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[_ia]), tsb[imiss[_ib]]))

        # test missing indices on both
        imiss2 = np.setxor1d(np.arange(n), [14, 17])
        _fcn, _drift, _ia, _ib = utils.sync_timestamps(
            tsa[imiss], tsb[imiss2], return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[imiss2[_ib]]))

        # test timestamps with huge offset (previously caused ArrayMemoryError)
        # tsb -= 1e15
        # _fcn, _drift = utils.sync_timestamps(tsa, tsb)
        # assert np.all(np.isclose(_fcn(tsa), tsb))


class TestParabolicMax(unittest.TestCase):
    # expected values
    maxi = np.array([np.nan, 0, 3.04166667, 3.04166667, 5, 5])
    ipeak = np.array([np.nan, 0, 5.166667, 2.166667, 0, 7])
    # input
    x = np.array(
        [
            [0, 0, 0, 0, 0, np.nan, 0, 0],  # some nans
            [0, 0, 0, 0, 0, 0, 0, 0],  # all flat
            [0, 0, 0, 0, 1, 3, 2, 0],
            [0, 1, 3, 2, 0, 0, 0, 0],
            [5, 1, 3, 2, 0, 0, 0, 0],  # test first sample
            [0, 1, 3, 2, 0, 0, 0, 5],  # test last sample
        ]
    )

    def test_2d(self):
        ipeak_, maxi_ = utils.parabolic_max(self.x)
        self.assertTrue(np.all(np.isclose(self.maxi, maxi_, equal_nan=True)))
        self.assertTrue(np.all(np.isclose(self.ipeak, ipeak_, equal_nan=True)))

    def test_1d(self):
        # look over the 2D array as 1D chunks
        for i, x in enumerate(self.x):
            ipeak_, maxi_ = utils.parabolic_max(x)
            self.assertTrue(np.all(np.isclose(self.ipeak[i], ipeak_, equal_nan=True)))
            self.assertTrue(np.all(np.isclose(self.maxi[i], maxi_, equal_nan=True)))


class TestDspMisc(unittest.TestCase):
    def test_dsp_cosine_func(self):
        x = np.linspace(0, 40)
        fcn = utils.fcn_cosine(bounds=[20, 30])
        y = fcn(x)
        self.assertTrue(y[0] == 0 and y[-1] == 1 and np.all(np.diff(y) >= 0))


class TestPhaseRegression(unittest.TestCase):
    def test_fit_phase1d(self):
        w = np.zeros(500)
        w[1] = 1
        self.assertTrue(np.isclose(fourier.fit_phase(w, 0.002), 0.002))

    def test_fit_phase2d(self):
        w = np.zeros((500, 2))
        w[1, 0], w[2, 1] = (1, 1)
        self.assertTrue(
            np.all(
                np.isclose(
                    fourier.fit_phase(w, 0.002, axis=0), np.array([0.002, 0.004])
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    fourier.fit_phase(w.transpose(), 0.002), np.array([0.002, 0.004])
                )
            )
        )


class TestShift(unittest.TestCase):
    def test_shift_already_fft(self):
        for ns in [500, 501]:
            w = utils.ricker(ns, 10)
            W = scipy.fft.rfft(w)
            ws = np.real(scipy.fft.irfft(fourier.fshift(W, 1, ns=np.shape(w)[0]), n=ns))
            self.assertTrue(np.all(np.isclose(ws, np.roll(w, 1))))

    def test_shift_floats(self):
        ns = 500
        w = utils.ricker(ns, 10)
        w_ = fourier.fshift(w.astype(np.float32), 1)
        assert w_.dtype == np.float32

    def test_shift_1d(self):
        ns = 500
        w = utils.ricker(ns, 10)
        self.assertTrue(np.all(np.isclose(fourier.fshift(w, 1), np.roll(w, 1))))

    def test_shift_2d(self):
        ns = 500
        w = utils.ricker(ns, 10)
        w = np.tile(w, (100, 1)).transpose()
        self.assertTrue(
            np.all(np.isclose(fourier.fshift(w, 1, axis=0), np.roll(w, 1, axis=0)))
        )
        self.assertTrue(
            np.all(np.isclose(fourier.fshift(w, 1, axis=1), np.roll(w, 1, axis=1)))
        )
        # # test with individual shifts for each trace
        self.assertTrue(
            np.all(
                np.isclose(
                    fourier.fshift(w, np.ones(w.shape[1]), axis=0),
                    np.roll(w, 1, axis=0),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    fourier.fshift(w, np.ones(w.shape[0]), axis=1),
                    np.roll(w, 1, axis=1),
                )
            )
        )


class TestSmooth(unittest.TestCase):
    def test_smooth_lp(self):
        np.random.seed(458)
        a = np.random.rand(
            500,
        )
        a_ = smooth.lp(a, [0.1, 0.15])
        res = fourier.hp(np.pad(a_, 100, mode="edge"), 1, [0.1, 0.15])[100:-100]
        self.assertTrue((utils.rms(a) / utils.rms(res)) > 500)


class TestFFT(unittest.TestCase):
    def test_spectral_convolution(self):
        sig = np.random.randn(20, 500)
        w = np.hanning(25)
        c = fourier.convolve(sig, w)
        s = np.convolve(sig[0, :], w)
        self.assertTrue(np.all(np.isclose(s, c[0, :-1])))

        c = fourier.convolve(sig, w, mode="same")
        s = np.convolve(sig[0, :], w, mode="same")
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

        c = fourier.convolve(sig, w[:-1], mode="same")
        s = np.convolve(sig[0, :], w[:-1], mode="same")
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

    def test_nech_optim(self):
        self.assertTrue(fourier.ns_optim_fft(2048) == 2048)
        self.assertTrue(fourier.ns_optim_fft(65532) == 65536)

    def test_freduce(self):
        # test with 1D arrays
        fs = np.fft.fftfreq(5)
        self.assertTrue(np.all(fourier.freduce(fs) == fs[:-2]))
        fs = np.fft.fftfreq(6)
        self.assertTrue(np.all(fourier.freduce(fs) == fs[:-2]))

        # test 2D arrays along both dimensions
        fs = np.tile(fourier.fscale(500, 0.001), (4, 1))
        self.assertTrue(fourier.freduce(fs).shape == (4, 251))
        self.assertTrue(fourier.freduce(np.transpose(fs), axis=0).shape == (251, 4))

    def test_fexpand(self):
        # test odd input
        res = np.random.rand(11)
        X = fourier.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(fourier.fexpand(X, 11)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test even input
        res = np.random.rand(12)
        X = fourier.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(fourier.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 2 dimensional input along last dimension
        res = np.random.rand(2, 12)
        X = fourier.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(fourier.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 3 dimensional input along last dimension
        res = np.random.rand(3, 5, 12)
        X = fourier.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(fourier.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with 2 dimensional input along first dimension
        fs = np.transpose(np.tile(fourier.fscale(500, 0.001, one_sided=True), (4, 1)))
        self.assertTrue(fourier.fexpand(fs, 500, axis=0).shape == (500, 4))

    def test_fscale(self):
        # test for an even number of samples
        res = ([0, 100, 200, 300, 400, 500, -400, -300, -200, -100],)
        self.assertTrue(np.all(np.abs(fourier.fscale(10, 0.001) - res) < 1e-6))
        # test for an odd number of samples
        res = (
            [
                0,
                90.9090909090909,
                181.818181818182,
                272.727272727273,
                363.636363636364,
                454.545454545455,
                -454.545454545455,
                -363.636363636364,
                -272.727272727273,
                -181.818181818182,
                -90.9090909090909,
            ],
        )
        self.assertTrue(np.all(np.abs(fourier.fscale(11, 0.001) - res) < 1e-6))

    def test_filter_lp_hp(self):
        # test 1D time serie: subtracting lp filter removes DC
        ts1 = np.random.rand(500)
        out1 = fourier.lp(ts1, 1, [0.1, 0.2])
        self.assertTrue(np.mean(ts1 - out1) < 0.001)
        # test 2D case along the last dimension
        ts = np.tile(ts1, (11, 1))
        out = fourier.lp(ts, 1, [0.1, 0.2])
        self.assertTrue(np.allclose(out, out1))
        # test 2D case along the first dimension
        ts = np.tile(ts1[:, np.newaxis], (1, 11))
        out = fourier.lp(ts, 1, [0.1, 0.2], axis=0)
        self.assertTrue(np.allclose(np.transpose(out), out1))
        # test 1D time serie: subtracting lp filter removes DC
        out2 = fourier.hp(ts1, 1, [0.1, 0.2])
        self.assertTrue(np.allclose(out1, ts1 - out2))

    def test_dft(self):
        # test 1D complex
        x = np.array([1, 2 - 1j, -1j, -1 + 2j])
        X = fourier.dft(x)
        assert np.all(np.isclose(X, np.fft.fft(x)))
        # test 1D real
        x = np.random.randn(7)
        X = fourier.dft(x)
        assert np.all(np.isclose(X, np.fft.rfft(x)))
        # test along the 3 dimensions of a 3D array
        x = np.random.rand(10, 11, 12)
        for axis in np.arange(3):
            X_ = np.fft.rfft(x, axis=axis)
            assert np.all(np.isclose(X_, fourier.dft(x, axis=axis)))
        # test 2D irregular grid
        _n0, _n1, nt = (10, 11, 30)
        x = np.random.rand(_n0 * _n1, nt)
        X_ = np.fft.fft(np.fft.fft(x.reshape(_n0, _n1, nt), axis=0), axis=1)
        r, c = [
            v.flatten()
            for v in np.meshgrid(
                np.arange(_n0) / _n0, np.arange(_n1) / _n1, indexing="ij"
            )
        ]
        nk, nl = (_n0, _n1)
        X = fourier.dft2(x, r, c, nk, nl)
        assert np.all(np.isclose(X, X_))


class TestWindowGenerator(unittest.TestCase):
    def test_window_simple(self):
        wg = utils.WindowGenerator(ns=500, nswin=100, overlap=50)
        sl = list(wg.firstlast)
        self.assertTrue(wg.nwin == len(sl) == 9)
        self.assertTrue(
            np.all(np.array([s[0] for s in sl]) == np.arange(0, wg.nwin) * 50)
        )
        self.assertTrue(
            np.all(np.array([s[1] for s in sl]) == np.arange(0, wg.nwin) * 50 + 100)
        )

        wg = utils.WindowGenerator(ns=500, nswin=100, overlap=10)
        sl = list(wg.firstlast)
        first = np.array([0, 90, 180, 270, 360, 450])
        last = np.array([100, 190, 280, 370, 460, 500])
        self.assertTrue(wg.nwin == len(sl) == 6)
        self.assertTrue(np.all(np.array([s[0] for s in sl]) == first))
        self.assertTrue(np.all(np.array([s[1] for s in sl]) == last))

    def test_nwindows_computation(self):
        for m in np.arange(0, 100):
            wg = utils.WindowGenerator(ns=500 + m, nswin=87 + m, overlap=11 + m)
            sl = list(wg.firstlast)
            self.assertTrue(wg.nwin == len(sl))

    def test_firstlast_slices(self):
        # test also the indexing versus direct slicing
        my_sig = np.random.rand(
            500,
        )
        wg = utils.WindowGenerator(ns=500, nswin=100, overlap=50)
        # 1) get the window by
        my_rms = np.zeros((wg.nwin,))
        for first, last in wg.firstlast:
            my_rms[wg.iw] = utils.rms(my_sig[first:last])
        # test with slice_array method
        my_rms_ = np.zeros((wg.nwin,))
        for wsig in wg.slice_array(my_sig):
            my_rms_[wg.iw] = utils.rms(wsig)
        self.assertTrue(np.all(my_rms_ == my_rms))
        # test with the slice output
        my_rms_ = np.zeros((wg.nwin,))
        for sl in wg.slice:
            my_rms_[wg.iw] = utils.rms(my_sig[sl])
        self.assertTrue(np.all(my_rms_ == my_rms))

    def test_firstlast_splicing(self):
        sig_in = np.random.randn(600)
        sig_out = np.zeros_like(sig_in)
        wg = utils.WindowGenerator(ns=600, nswin=100, overlap=20)
        for first, last, amp in wg.firstlast_splicing:
            sig_out[first:last] = sig_out[first:last] + amp * sig_in[first:last]
        np.testing.assert_allclose(sig_out, sig_in)

    def test_firstlast_valid(self):
        sig_in = np.random.randn(600)
        sig_out = np.zeros_like(sig_in)
        wg = utils.WindowGenerator(ns=600, nswin=100, overlap=20)
        for first, last, first_valid, last_valid in wg.firstlast_valid:
            sig_out[first_valid:last_valid] = sig_in[first_valid:last_valid]
        np.testing.assert_array_equal(sig_out, sig_in)

    def test_tscale(self):
        wg = utils.WindowGenerator(ns=500, nswin=100, overlap=50)
        ts = wg.tscale(fs=1000)
        self.assertTrue(ts[0] == (100 - 1) / 2 / 1000)
        self.assertTrue((np.allclose(np.diff(ts), 0.05)))


class TestFrontDetection(unittest.TestCase):
    def test_rises_falls(self):
        # test 1D case with a long pulse and a dirac
        a = np.zeros(
            500,
        )
        a[80:120] = 1
        a[200] = 1
        # rising fronts
        self.assertTrue(all(utils.rises(a) == np.array([80, 200])))
        # falling fronts
        self.assertTrue(all(utils.falls(a) == np.array([120, 201])))
        # both
        ind, val = utils.fronts(a)
        self.assertTrue(all(ind == np.array([80, 120, 200, 201])))
        self.assertTrue(all(val == np.array([1, -1, 1, -1])))

        # test a 2D case with 2 long pulses and a dirac
        a = np.zeros((2, 500))
        a[0, 80:120] = 1
        a[0, 200] = 1
        a[1, 280:320] = 1
        a[1, 400] = 1
        # rising fronts
        self.assertTrue(
            np.all(utils.rises(a) == np.array([[0, 0, 1, 1], [80, 200, 280, 400]]))
        )
        # falling fronts
        self.assertTrue(
            np.all(utils.falls(a) == np.array([[0, 0, 1, 1], [120, 201, 320, 401]]))
        )
        # both
        ind, val = utils.fronts(a)
        self.assertTrue(all(ind[0] == np.array([0, 0, 0, 0, 1, 1, 1, 1])))
        self.assertTrue(
            all(ind[1] == np.array([80, 120, 200, 201, 280, 320, 400, 401]))
        )
        self.assertTrue(all(val == np.array([1, -1, 1, -1, 1, -1, 1, -1])))

    def test_rises_analog(self):
        a = utils.fcn_cosine([0, 1])(np.linspace(-5, 5, 500))
        a = np.r_[a, np.flipud(a)] * 4
        np.testing.assert_array_equal(utils.falls(a, step=3, analog=True), 717)
        np.testing.assert_array_equal(utils.rises(a, step=3, analog=True), 283)


class TestVoltage(unittest.TestCase):
    def test_destripe_parameters(self):
        import inspect

        _, _, spatial_fcn = voltage._get_destripe_parameters(
            30_000, None, None, k_filter=True
        )
        assert "kfilt" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = voltage._get_destripe_parameters(
            2_500, None, None, k_filter=False
        )
        assert "car" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = voltage._get_destripe_parameters(
            2_500, None, None, k_filter=None
        )
        assert "dat: dat" in inspect.getsource(spatial_fcn)
        _, _, spatial_fcn = voltage._get_destripe_parameters(
            2_500, None, None, k_filter=lambda dat: 3 * dat
        )
        assert "lambda dat: 3 * dat" in inspect.getsource(spatial_fcn)

    def test_fk(self):
        """
        creates a couple of plane waves and separate them using the velocity HP filter
        """
        ntr, ns, sr, dx, v1, v2 = (500, 2000, 0.002, 5, 2000, 1000)
        data = np.zeros((ntr, ns), np.float32)
        data[:, :100] = utils.ricker(100, 4)
        offset = np.arange(ntr) * dx
        offset = np.abs(offset - np.mean(offset))
        data_v1 = fourier.fshift(data, offset / v1 / sr)
        data_v2 = fourier.fshift(data, offset / v2 / sr)

        noise = np.random.randn(ntr, ns) / 60
        fk = voltage.fk(
            data_v1 + data_v2 + noise,
            si=sr,
            dx=dx,
            vbounds=[1200, 1500],
            ntr_pad=10,
            ntr_tap=15,
            lagc=0.25,
        )
        fknoise = voltage.fk(
            noise, si=sr, dx=dx, vbounds=[1200, 1500], ntr_pad=10, ntr_tap=15, lagc=0.25
        )
        # at least 90% of the traces should be below 50dB and 98% below 40 dB
        assert np.mean(20 * np.log10(utils.rms(fk - data_v1 - fknoise)) < -50) > 0.9
        assert np.mean(20 * np.log10(utils.rms(fk - data_v1 - fknoise)) < -40) > 0.98
        # test the K option
        kbands = np.sin(np.arange(ns) / ns * 8 * np.pi) / 10
        fkk = voltage.fk(
            data_v1 + data_v2 + kbands,
            si=sr,
            dx=dx,
            vbounds=[1200, 1500],
            ntr_pad=40,
            ntr_tap=15,
            lagc=0.25,
            kfilt={"bounds": [0, 0.01], "btype": "hp"},
        )
        assert np.mean(20 * np.log10(utils.rms(fkk - data_v1)) < -40) > 0.9
        # from easyqc.gui import viewseis
        # a = viewseis(data_v1 + data_v2 + kbands, .002, title='input')
        # b = viewseis(fkk, .002, title='output')
        # c = viewseis(data_v1 - fkk, .002, title='test')

    def test_saturation(self):
        np.random.seed(7654)
        data = (np.random.randn(384, 30_000).astype(np.float32) + 20) * 1e-6
        saturated, mute = voltage.saturation(data, max_voltage=1200)
        np.testing.assert_array_equal(saturated, 0)
        np.testing.assert_array_equal(mute, 1.0)
        # now we stick a big waveform in the middle of the recorder and expect some saturation
        w = utils.ricker(100, 4)
        w = np.minimum(1200, w / w.max() * 1400)
        data[:, 13_600:13700] = data[0, 13_600:13700] + w * 1e-6
        saturated, mute = voltage.saturation(
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
            cadzow.traj_matrix_indices(4) == np.array([[1, 0], [2, 1], [3, 2]])
        )
        assert np.all(cadzow.traj_matrix_indices(3) == np.array([[1, 0], [2, 1]]))


class TestStack(unittest.TestCase):
    def test_simple_stack(self):
        ntr, ns = (24, 400)
        data = np.zeros((ntr, ns), dtype=np.float32)
        word = np.flipud(np.floor(np.arange(ntr) / 3))
        data += word[:, np.newaxis] * 10
        stack, fold = voltage.stack(data, word=word)
        assert np.all(fold == 3)
        assert np.all(np.squeeze(np.unique(stack, axis=-1)) == np.arange(8) * 10)
        # test with a header
        header = {"toto": np.random.randn(ntr)}
        stack, hstack = voltage.stack(data, word=word, header=header)
        assert list(hstack.keys()) == ["toto", "fold"]
        assert np.all(hstack["fold"] == 3)


class TestSpikeTrains(unittest.TestCase):
    def test_spikes_venn3(self):
        rng = np.random.default_rng()
        # make sure each 'spiketrain' goes up to around 30000 samples
        samples_tuple = (
            np.cumsum(rng.poisson(30, 1000)),
            np.cumsum(rng.poisson(20, 1500)),
            np.cumsum(rng.poisson(15, 2000)),
        )
        # used reduced number of channels for speed and to increase collision chance
        channels_tuple = (
            rng.integers(20, size=(1000,)),
            rng.integers(20, size=(1500,)),
            rng.integers(20, size=(2000,)),
        )
        # check that spikes have been accounted for exactly once
        venn_info = spiketrains.spikes_venn3(
            samples_tuple, channels_tuple, num_channels=20
        )
        assert (
            venn_info["100"] + venn_info["101"] + venn_info["110"] + venn_info["111"]
            == 1000
        )
        assert (
            venn_info["010"] + venn_info["011"] + venn_info["110"] + venn_info["111"]
            == 1500
        )
        assert (
            venn_info["001"] + venn_info["101"] + venn_info["011"] + venn_info["111"]
            == 2000
        )

    def test_spikes_venn2(self):
        rng = np.random.default_rng()

        # make sure each 'spiketrain' goes up to around 30000 samples
        samples_tuple = (
            np.cumsum(rng.poisson(30, 1000)),
            np.cumsum(rng.poisson(20, 1500)),
        )
        # used reduced number of channels for speed and to increase collision chance
        channels_tuple = (
            rng.integers(20, size=(1000,)),
            rng.integers(20, size=(1500,)),
        )

        venn_info = spiketrains.spikes_venn2(
            samples_tuple, channels_tuple, num_channels=20
        )

        assert venn_info["10"] + venn_info["11"] == 1000
        assert venn_info["01"] + venn_info["11"] == 1500


class TestRawDataFeatures(unittest.TestCase):
    def setUp(self):
        self.fixtures_path = FIXTURE_PATH
        self.tmpdir = Path(tempfile.gettempdir()) / "rawdata"
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.tmpdir.mkdir()
        self.ns_ap = 38502
        self.nc = 385
        self.features = [
            "ap_dc_offset",
            "ap_raw_rms",
            "ap_butter_rms",
            "ap_destripe_rms",
            "ap_striping_rms",
            "ap_channel_labels",
            "ap_xcor_hf_raw",
            "ap_xcor_lf_raw",
            "ap_psd_hf_raw",
            "ap_xcor_hf_destripe",
            "ap_xcor_lf_destripe",
            "ap_psd_hf_destripe",
            "lf_dc_offset",
            "lf_raw_rms",
            "lf_butter_rms",
            "lf_destripe_rms",
            "lf_striping_rms",
            "lf_channel_labels",
            "lf_xcor_hf_raw",
            "lf_xcor_lf_raw",
            "lf_psd_hf_raw",
            "lf_xcor_hf_destripe",
            "lf_xcor_lf_destripe",
            "lf_psd_hf_destripe",
        ]
        self.ap_meta = self.fixtures_path.joinpath("sample3B_g0_t0.imec1.ap.meta")
        self.lf_meta = self.fixtures_path.joinpath("sample3B_g0_t0.imec1.lf.meta")
        self.ap_cbin = spikeglx._mock_spikeglx_file(
            self.tmpdir.joinpath("test_ap.bin"),
            self.ap_meta,
            self.ns_ap,
            self.nc,
            sync_depth=16,
        )["bin_file"]
        self.lf_cbin = spikeglx._mock_spikeglx_file(
            self.tmpdir.joinpath("test_lf.bin"),
            self.lf_meta,
            self.ns_ap * 12,
            self.nc,
            sync_depth=16,
        )["bin_file"]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_compute_features_snip(self):
        """
        Test create features table on one snip.
        """
        t0 = 0.2
        t1 = 0.7
        sr_ap = spikeglx.Reader(self.ap_cbin)
        sr_lf = spikeglx.Reader(self.lf_cbin)
        df = raw_metrics.compute_raw_features_snippet(sr_ap, sr_lf, t0, t1)
        self.assertEqual(self.nc - 1, len(df))
        self.assertEqual(set(self.features), set(list(df.columns)))

    def test_compute_features(self):
        """
        Test create features table from several snips.
        """
        num_snippets = 2
        t_start = [0.1, 0.5]
        t_end = [0.4, 0.8]
        df = raw_metrics.raw_data_features(self.ap_cbin, self.lf_cbin, t_start, t_end)
        multi_index = [(i, j) for i in range(num_snippets) for j in range(self.nc - 1)]
        self.assertEqual(multi_index, list(df.index))
        self.assertEqual(["snippet_id", "channel_id"], list(df.index.names))
        self.assertEqual(num_snippets * (self.nc - 1), len(df))


if __name__ == "__main__":
    unittest.main()
