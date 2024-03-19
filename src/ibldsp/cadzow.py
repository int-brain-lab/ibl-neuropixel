"""
Module to apply cadzow denoising to N-dimensional arrays, specifically Neuropixel probes

This paper gives a good overview of the techniques used, notably the Hankel matrix building
for N-spatial dimensions.

@article{sternfels2015multidimensional,
  title={Multidimensional simultaneous random plus erratic noise attenuation and interpolation for seismic data by
  joint low-rank and sparse inversion},
  author={Sternfels, Raphael and Viguier, Ghislain and Gondoin, Regis and Le Meur, David},
  journal={Geophysics},
  volume={80},
  number={6},
  pages={WD129--WD141},
  year={2015},
  publisher={Society of Exploration Geophysicists}
}
"""

import numpy as np
import scipy.fft
from iblutil.numerical import ismember2d

import neuropixel


def derank(T, r):
    u, s, v = np.linalg.svd(T)
    # try non-integer rank as a proportion of singular values ?
    # ik = np.searchsorted(np.cumsum(s) / np.sum(s), KEEP)
    T_ = np.zeros_like(T)
    for i in np.arange(r):
        T_ += s[i] * np.outer(u.T[i], v[i])
    return T_


def traj_matrix_indices(n):
    """
    Computes the single spatial dimension Toeplitz-like indices from a number of spatial traces
    :param n: number of dimensions
    :return: 2-D int matrix whose elements are indices of the spatial dimension
    """
    nrows = int(np.floor(n / 2 + 1))
    ncols = int(np.ceil(n / 2))
    itraj = np.tile(np.arange(nrows), (ncols, 1)).T + np.flipud(np.arange(ncols))
    return itraj


def trajectory(x, y):
    """
    Computes the 2 spatial dimensions block-Toeplitz indices from x and y traces coordinates
    Coordinates are assumed to be regularly spaced
    :param x: trace spatial coordinate (np.array)
    :param y: trace spatial coordinate (np.array)
    :return: T: 2-D complex matrix whose elements are the spatial dimensions
    :return: it, itr: (tuple, ndarray) indices such that T[it] = data[itr]
    :return: trcount: count of traces in the trajectory matrix
    """
    xu, ix = np.unique(x, return_inverse=True)
    yu, iy = np.unique(y, return_inverse=True)
    nx, ny = (np.size(xu), np.size(yu))

    tiy_ = traj_matrix_indices(ny)
    tix_ = traj_matrix_indices(nx)
    tiy = np.tile(tiy_, tix_.shape)
    tix = np.repeat(np.repeat(tix_, tiy_.shape[0], axis=0), tiy_.shape[1], axis=1)

    it, itr = ismember2d(np.c_[tix.flatten(), tiy.flatten()], np.c_[ix, iy])
    it = np.unravel_index(np.where(it)[0], tiy.shape)

    T = np.zeros(tix.shape, dtype=np.complex128)

    trcount = np.bincount(itr)
    return T, it, itr, trcount


def denoise(WAV, x, y, r, imax=None, niter=1):
    """
    Applies cadzow denoising by de-ranking spatial matrices in frequency domain
    :param WAV: np array nc / ns in frequency domain
    :param x: trace spatial coordinate (np.array)
    :param y: trace spatial coordinate (np.array)
    :param r: rank
    :param imax: index of the maximum frequency to keep, all frequencies are de-ranked if None (None)
    :param niter: number of iterations (1)
    :return: WAV_: np array nc / ns in frequency domain
    """
    WAV_ = np.copy(WAV)
    imax = np.minimum(WAV.shape[-1], imax) if imax else WAV.shape[-1]
    T, it, itr, trcount = trajectory(x, y)
    for ind_f in np.arange(imax):
        for _ in np.arange(niter):
            T[it] = WAV_[itr, ind_f]
            T_ = derank(T, r)
            WAV_[:, ind_f] = np.bincount(itr, weights=np.real(T_[it]))
            WAV_[:, ind_f] += 1j * np.bincount(itr, weights=np.imag(T_[it]))
            WAV_[:, ind_f] /= trcount

    return WAV_


def cadzow_np1(
    wav,
    fs=30000,
    rank=5,
    niter=1,
    fmax=7500,
    h=None,
    ovx=int(16),
    nswx=int(32),
    npad=int(0),
):
    """
    Apply Fxy rank-denoiser to a full recording of Neuropixel 1 probe geometry
    ntr - nswx has to be a multiple of (nswx - ovx)
    Examples of working set of parameters:
        ovx = int(5); nswx = int(33); ovx = int(6))
        ovx = int(16); nswx = int(32); ovx = int(0))
        ovx = int(32); nswx = int(64); ovx = int(0))
        ovx = int(24); nswx = int(64); ovx = int(0))
        ovx = int(8); nswx = int(16); ovx = int(0))
    :param wav: ntr, ns
    :param fs:
    :param ovx is the overlap in x
    :param nswx is the size of the window in x
    :param npad is the padding
    :return:
    """
    #
    ntr, ns = wav.shape
    h = h or neuropixel.trace_header(version=1)
    nwinx = int(np.ceil((ntr + npad * 2 - ovx) / (nswx - ovx)))
    fscale = scipy.fft.rfftfreq(ns, d=1 / fs)
    imax = np.searchsorted(fscale, fmax)
    WAV = scipy.fft.rfft(wav[:, :])
    padgain = scipy.signal.windows.hann(npad * 2)[:npad]
    WAV = np.r_[
        np.flipud(WAV[1: npad + 1, :]) * padgain[:, np.newaxis],
        WAV,
        np.flipud(WAV[-npad - 2: -1, :]) * np.flipud(np.r_[padgain, 1])[:, np.newaxis],
    ]  # apply padding
    x = np.r_[np.flipud(h["x"][1: npad + 1]), h["x"], np.flipud(h["x"][-npad - 2: -1])]
    y = np.r_[
        np.flipud(h["y"][1: npad + 1]) - 120,
        h["y"],
        np.flipud(h["y"][-npad - 2: -1]) + 120,
    ]
    WAV_ = np.zeros_like(WAV)
    gain = np.zeros(ntr + npad * 2 + 1)
    hanning = scipy.signal.windows.hann(ovx * 2 - 1)[0:ovx]
    assert np.all(np.isclose(hanning + np.flipud(hanning), 1))
    gain_window = np.r_[hanning, np.ones(nswx - ovx * 2), np.flipud(hanning)]
    for firstx in np.arange(nwinx) * (nswx - ovx):
        lastx = int(firstx + nswx)
        if firstx == 0:
            gw = np.r_[hanning * 0 + 1, np.ones(nswx - ovx * 2), np.flipud(hanning)]
        elif lastx == ntr:
            gw = np.r_[hanning, np.ones(nswx - ovx * 2), hanning * 0 + 1]
        else:
            gw = gain_window
        gain[firstx:lastx] += gw
        array = WAV[firstx:lastx, :]
        array = denoise(
            array, x=x[firstx:lastx], y=y[firstx:lastx], r=rank, imax=imax, niter=niter
        )
        WAV_[firstx:lastx, :] += array * gw[:, np.newaxis]

    WAV_ = WAV_[npad: -npad - 1]  # remove padding
    wav_ = scipy.fft.irfft(WAV_)
    return wav_
