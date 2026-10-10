import unittest
import warnings
from unittest.mock import patch

import numpy as np

import ibldsp.cadzow
import neuropixel


def _plane_wave(nc=32, ns=500, fs=250.0, freq=5.0, noise_sigma=0.1, seed=0):
    """Rank-1 plane wave on the first nc channels of NP1, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    th = neuropixel.trace_header(version=1)
    y = th["y"][:nc]
    t = np.arange(ns) / fs
    signal = np.sin(2 * np.pi * freq * t[None, :] + 0.01 * y[:, None]).astype(
        np.float32
    )
    noise = (noise_sigma * rng.standard_normal((nc, ns))).astype(np.float32)
    return signal + noise, signal


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

    def test_fmax_none(self):
        """fmax=None must process all bins up to Nyquist without error."""
        wav, _ = _plane_wave(nc=128)
        out_new = ibldsp.cadzow.cadzow_denoiser(wav, fmax=None)
        self.assertEqual(out_new.shape, wav.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out_old = ibldsp.cadzow.cadzow_np1(wav, fs=250.0, fmax=None)
        self.assertEqual(out_old.shape, wav.shape)

    def test_apply_rank_threshold_fixed(self):
        """With gap_threshold=None, columns >= r are zeroed."""
        s = np.array([[10.0, 5.0, 1.0, 0.1]])
        ibldsp.cadzow._apply_rank_threshold(s, r=2)
        np.testing.assert_array_equal(s[0, 2:], 0.0)
        self.assertGreater(s[0, 1], 0.0)

    def test_apply_rank_threshold_adaptive(self):
        """With gap_threshold, rank is set at the largest ratio >= threshold."""
        # ratios: [10/5, 5/1, 1/0.1] = [2.0, 5.0, 10.0] — largest is index 2 → rank=3
        s = np.array([[10.0, 5.0, 1.0, 0.1]])
        ibldsp.cadzow._apply_rank_threshold(s.copy(), r=4, gap_threshold=1.5)
        s_copy = np.array([[10.0, 5.0, 1.0, 0.1]])
        ibldsp.cadzow._apply_rank_threshold(s_copy, r=4, gap_threshold=1.5)
        # rank should be 3 (gap at index 2, ratio 10.0 >= 1.5), so s[3] zeroed
        self.assertEqual(s_copy[0, 3], 0.0)
        self.assertGreater(s_copy[0, 2], 0.0)

    def test_denoise_fxy_reduces_noise(self):
        """denoise_fxy on a plane wave + noise should reduce RMS error vs clean signal."""
        wav, signal = _plane_wave(nc=32, ns=500, noise_sigma=0.2)
        th = neuropixel.trace_header(version=1)
        x, y = th["x"][:32], th["y"][:32]
        import scipy.fft

        WAV = scipy.fft.rfft(wav)
        imax = WAV.shape[1]  # all bins
        WAV_out = ibldsp.cadzow.denoise_fxy(WAV, x=x, y=y, r=3, imax=imax)
        out = scipy.fft.irfft(WAV_out, n=wav.shape[1]).astype(np.float32)
        rms_before = float(np.sqrt(np.mean((wav - signal) ** 2)))
        rms_after = float(np.sqrt(np.mean((out - signal) ** 2)))
        self.assertLess(
            rms_after,
            rms_before,
            msg=f"Denoiser did not reduce noise: {rms_before:.4f} → {rms_after:.4f}",
        )

    def test_denoise_fxy_ppca_k(self):
        """ppca_k should reduce the residual on an outlier channel."""
        wav, signal = _plane_wave(nc=32, ns=500, noise_sigma=0.05)
        # inject a single-channel amplitude outlier
        wav_outlier = wav.copy()
        wav_outlier[10] *= 8.0
        th = neuropixel.trace_header(version=1)
        x, y = th["x"][:32], th["y"][:32]
        import scipy.fft

        WAV = scipy.fft.rfft(wav_outlier)
        imax = WAV.shape[1]
        ns = wav.shape[1]
        out_no_ppca = scipy.fft.irfft(
            ibldsp.cadzow.denoise_fxy(WAV, x=x, y=y, r=3, imax=imax), n=ns
        ).astype(np.float32)
        out_ppca = scipy.fft.irfft(
            ibldsp.cadzow.denoise_fxy(WAV, x=x, y=y, r=3, imax=imax, ppca_k=2.0), n=ns
        ).astype(np.float32)
        err_no_ppca = float(np.sqrt(np.mean((out_no_ppca[10] - signal[10]) ** 2)))
        err_ppca = float(np.sqrt(np.mean((out_ppca[10] - signal[10]) ** 2)))
        self.assertLess(
            err_ppca,
            err_no_ppca,
            msg=f"ppca_k did not improve outlier channel: {err_no_ppca:.4f} → {err_ppca:.4f}",
        )

    def test_cadzow_denoiser_regression(self):
        """cadzow_np1 and cadzow_denoiser should agree to within 5 % RMS on clean data."""
        wav, _ = _plane_wave(nc=32, ns=500, noise_sigma=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out_old = ibldsp.cadzow.cadzow_np1(wav, fs=250.0, rank=3, fmax=None)
        # use the same windowing as cadzow_np1 defaults for a fair regression comparison
        out_new = ibldsp.cadzow.cadzow_denoiser(
            wav, fs=250.0, rank=3, fmax=None, nswx=32, ovx=16
        )
        rms_old = float(np.sqrt(np.mean(out_old**2)))
        rms_diff = float(np.sqrt(np.mean((out_old - out_new) ** 2)))
        rel = rms_diff / rms_old
        self.assertLess(
            rel,
            0.05,
            msg=f"cadzow_np1 vs cadzow_denoiser relative RMS diff = {rel:.3f} (threshold 0.05)",
        )

    def test_cadzow_denoiser_ppca_k_end_to_end(self):
        """cadzow_denoiser with ppca_k: correct shape, no NaN/Inf, differs from no-ppca run."""
        wav, _ = _plane_wave(nc=128, ns=500, noise_sigma=0.1)
        out_base = ibldsp.cadzow.cadzow_denoiser(wav, fs=250.0, rank=3, fmax=None)
        out_ppca = ibldsp.cadzow.cadzow_denoiser(
            wav, fs=250.0, rank=3, fmax=None, ppca_k=2.0
        )
        self.assertEqual(out_ppca.shape, wav.shape)
        self.assertFalse(np.any(np.isnan(out_ppca)), "NaN in ppca_k output")
        self.assertFalse(np.any(np.isinf(out_ppca)), "Inf in ppca_k output")
        # ppca should change the output (it's active on this data)
        self.assertFalse(
            np.allclose(out_base, out_ppca),
            "ppca_k=2.0 produced identical output to ppca_k=None",
        )

    def test_cadzow_denoiser_zero_matrix(self):
        """All-zero input must complete without error and return finite values."""
        wav = np.zeros((128, 500), dtype=np.float32)
        out = ibldsp.cadzow.cadzow_denoiser(wav, n_jobs=1)
        self.assertEqual(out.shape, wav.shape)
        self.assertFalse(np.any(~np.isfinite(out)))

    def test_cadzow_svd_nonconvergence_fallback(self):
        """Simulate LAPACK gesdd non-convergence: _safe_svd must fall back to scipy gesvd.

        Before the fix, _process_window called np.linalg.svd directly so a
        LinAlgError propagated up (reproduces the supercomputer crash).  After
        the fix, _process_window calls _safe_svd which catches the error and
        retries slice-by-slice with scipy gesvd (lapack_driver='gesvd').
        """
        wav, _ = _plane_wave(nc=128, ns=500)

        def always_fail_svd(a, *args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        with patch.object(np.linalg, "svd", always_fail_svd):
            out = ibldsp.cadzow.cadzow_denoiser(wav, n_jobs=1)

        self.assertEqual(out.shape, wav.shape)
        self.assertFalse(np.any(np.isnan(out)))


def _laminar_field(ns=512, fs=250.0, n_sources=6, seed=0):
    """Smooth laminar field on NP1: Gaussian depth profiles with sinusoidal time courses, constant laterally."""
    rng = np.random.default_rng(seed)
    y = neuropixel.trace_header(version=1)["y"].astype(float)
    t = np.arange(ns) / fs
    out = np.zeros((y.size, ns))
    for _ in range(n_sources):
        z0, width, freq = (
            rng.uniform(0, 3840),
            rng.uniform(200, 600),
            rng.uniform(2, 30),
        )
        phase = rng.uniform(0, 2 * np.pi)
        profile = np.exp(-0.5 * ((y - z0) / width) ** 2)
        out += profile[:, None] * np.sin(2 * np.pi * freq * t + phase)[None, :]
    return out


class TestCadzowFillGridShrinkage(unittest.TestCase):
    H = neuropixel.trace_header(version=1)
    HXY = {"x": H["x"], "y": H["y"]}
    SCALE = 2.25  # shrinkage noise scale for NP1, full grid, nswx=64, ovx=32

    def csd_error(self, out, clean):
        """CSD error relative to the CSD RMS of the clean field."""
        from ibldsp.voltage import current_source_density

        csd = current_source_density(clean, self.H)
        err = current_source_density(out.astype(np.float64), self.H) - csd
        return np.sqrt(np.mean(err**2) / np.mean(csd**2))

    def test_fill_grid(self):
        """NP1 checkerboard: twice the positions, channels kept as is, a depth-linear field exact away from the ends."""
        wav = np.tile(self.H["y"][:, None].astype(float), (1, 3))
        g, gx, gy, ireal = ibldsp.cadzow._fill_grid(wav, self.H["x"], self.H["y"])
        self.assertEqual(gx.size, 768)
        np.testing.assert_array_equal(gx[ireal], self.H["x"])
        np.testing.assert_array_equal(gy[ireal], self.H["y"])
        np.testing.assert_array_equal(g[ireal], wav)
        interior = (gy > gy.min()) & (gy < gy.max())
        np.testing.assert_allclose(g[interior, 0], gy[interior])

    def test_fill_grid_np2_is_identity(self):
        """NP2 has no empty grid positions: fill_grid must not change the output."""
        h2 = neuropixel.trace_header(version=2)
        wav = np.random.default_rng(3).standard_normal((384, 256))
        kwargs = dict(h={"x": h2["x"], "y": h2["y"]}, rank=3, fmax=None)
        np.testing.assert_array_equal(
            ibldsp.cadzow.cadzow_denoiser(wav, **kwargs),
            ibldsp.cadzow.cadzow_denoiser(wav, fill_grid=True, **kwargs),
        )

    def test_apply_shrinkage(self):
        """Noise-level singular values are zeroed, a strong one is shrunk but kept, rank is capped."""
        shape = (51, 32)
        rng = np.random.default_rng(0)
        T = rng.standard_normal((4, *shape)) + 1j * rng.standard_normal((4, *shape))
        s = np.linalg.svd(T, compute_uv=False)
        s[:, :2] *= np.array([50.0, 30.0])  # two strong components
        s_ = s.copy()
        ibldsp.cadzow._apply_shrinkage(s_, r=1, shape=shape, scale=1.0)
        self.assertTrue(np.all(s_[:, 0] > 0))
        self.assertTrue(np.all(s_[:, 0] < s[:, 0]))
        np.testing.assert_array_equal(s_[:, 1:], 0.0)  # capped at rank 1
        s_ = s.copy()
        ibldsp.cadzow._apply_shrinkage(s_, r=32, shape=shape, scale=1.0)
        self.assertTrue(np.all(s_[:, :2] > 0))
        np.testing.assert_array_equal(s_[:, 5:], 0.0)  # noise bulk removed

    def test_shrinkage_white_noise(self):
        """Pure white noise is almost entirely removed."""
        wav = np.random.default_rng(2).standard_normal((384, 512))
        out = ibldsp.cadzow.cadzow_denoiser(
            wav,
            h=self.HXY,
            fmax=None,
            fill_grid=True,
            shrinkage=self.SCALE,
        )
        self.assertLess(out.std() / wav.std(), 0.05)

    def test_fill_grid_shrinkage_csd(self):
        """On a smooth laminar field, full grid + shrinkage beats the lfpack v04 settings in CSD, clean and noisy."""
        clean = _laminar_field()
        noisy = (
            clean
            + np.random.default_rng(1).standard_normal(clean.shape) * 0.05 * clean.std()
        )
        production = dict(rank=5, fmax=None, gap_threshold=2.0, ppca_k=2.0)
        new = dict(
            rank=5,
            fmax=None,
            ppca_k=2.0,
            fill_grid=True,
            shrinkage=self.SCALE,
        )
        for wav, tol in ((clean, 0.05), (noisy, 0.4)):
            err_prod = self.csd_error(
                ibldsp.cadzow.cadzow_denoiser(wav, h=self.HXY, **production), clean
            )
            err_new = self.csd_error(
                ibldsp.cadzow.cadzow_denoiser(wav, h=self.HXY, **new), clean
            )
            self.assertLess(err_new, tol)
            self.assertLess(err_new, err_prod / 3)
