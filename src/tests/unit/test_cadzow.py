import unittest
import warnings

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
        out_new = ibldsp.cadzow.cadzow_denoiser(wav, fs=250.0, rank=3, fmax=None, nswx=32, ovx=16)
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
