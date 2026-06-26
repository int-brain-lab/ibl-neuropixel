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

import warnings

import numpy as np
import scipy.fft
import scipy.signal
from iblutil.numerical import ismember2d

import neuropixel


def _apply_rank_threshold(s, r, gap_threshold=None):
    """Zero singular values beyond the adaptive rank for each frequency bin.

    Parameters
    ----------
    s : ndarray (nbins, k)
        Singular values sorted descending per row, modified in-place.
    r : int
        Hard upper bound on rank; fallback when no gap qualifies.
    gap_threshold : float or None
        Minimum s[i]/s[i+1] ratio to count as a dominant spectral gap.
        The rank per bin is the position of the largest such ratio, clamped
        to [1, r].  None disables adaptive selection (fixed rank r).
    """
    if gap_threshold is None:
        s[:, r:] = 0.0
        return
    ratios = s[:, :-1] / (s[:, 1:] + 1e-10)
    has_gap = ratios.max(axis=1) >= gap_threshold
    r_adapt = np.where(has_gap, np.argmax(ratios, axis=1) + 1, r)
    r_adapt = np.clip(r_adapt, 1, r)
    idx = np.arange(s.shape[1])[np.newaxis, :]
    s[idx >= r_adapt[:, np.newaxis]] = 0.0


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


def trajectory(x, y, dtype=np.complex128):
    """
    Computes the 2 spatial dimensions block-Toeplitz indices from x and y trace coordinates.

    This function creates a trajectory matrix and associated indices for use in
    2D spatial denoising algorithms, such as Cadzow denoising. It assumes the input
    coordinates are regularly spaced.

    Parameters:
    -----------
    x : array_like
        1D array of x-coordinates for each trace.
    y : array_like
        1D array of y-coordinates for each trace.
    dtype : numpy.dtype, optional
        Data type for the trajectory matrix. Default is np.complex128.

    Returns:
    --------
    T : ndarray
        2D empyt matrix representing the trajectory matrix. Its shape is determined
        by the unique values in x and y.
    it : tuple of ndarrays
        A tuple of index arrays that can be used to index into T.
    ic : ndarray
        1D array of channel indices that map elements of T to the original data.
    trcount : ndarray
        1D array containing the count of traces for each unique (x, y) coordinate.

    Notes:
    ------
    - The function creates a block-Toeplitz structure that captures spatial relationships
      in both x and y dimensions.
    - The returned indices (it and ic) allow for efficient mapping between the trajectory
      matrix and the original data space.
    - This function is particularly useful in multi-dimensional signal processing
      applications, such as seismic data analysis or image processing.

    Example:
    --------
    >>> x = np.array([0, 1, 2, 0, 1, 2])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> T, it, ic, trcount = trajectory(x, y)
    """
    xu, ix = np.unique(x, return_inverse=True)
    yu, iy = np.unique(y, return_inverse=True)
    nx, ny = (np.size(xu), np.size(yu))

    tiy_ = traj_matrix_indices(ny)
    tix_ = traj_matrix_indices(nx)
    tiy = np.tile(tiy_, tix_.shape)
    tix = np.repeat(np.repeat(tix_, tiy_.shape[0], axis=0), tiy_.shape[1], axis=1)

    it, ic = ismember2d(np.c_[tix.flatten(), tiy.flatten()], np.c_[ix, iy])
    it = np.unravel_index(np.where(it)[0], tiy.shape)

    T = np.zeros(tix.shape, dtype=dtype)

    trcount = np.bincount(ic)
    return T, it, ic, trcount


def denoise(WAV, x, y, r, imax=None, niter=1):
    """
    Applies cadzow denoising by de-ranking spatial matrices in frequency domain.

    .. deprecated::
        Use `denoise_fxy` instead — it replaces the per-bin Python loop with a
        single batched SVD call, giving a large speed improvement.

    :param WAV: np array (nc, ns) in frequency domain
    :param x: trace spatial coordinate np.array (nc)
    :param y: trace spatial coordinate np.array (nc)
    :param r: rank
    :param imax: index of the maximum frequency to keep, all frequencies are de-ranked if None (None)
    :param niter: number of iterations (1)
    :return: WAV_: np array nc / ns in frequency domain
    """
    warnings.warn(
        "denoise() is deprecated; use denoise_fxy() for the same result with batched SVD.",
        DeprecationWarning,
        stacklevel=2,
    )
    WAV_ = np.zeros_like(WAV)
    WAV0 = np.copy(WAV)
    imax = np.minimum(WAV.shape[-1], imax) if imax else WAV.shape[-1]
    T, it, itr, trcount = trajectory(x, y)
    for _ in np.arange(niter):
        for ind_f in np.arange(imax):
            T[it] = WAV0[itr, ind_f]
            T_ = derank(T, r)
            WAV_[:, ind_f] = np.bincount(itr, weights=np.real(T_[it]))
            WAV_[:, ind_f] += 1j * np.bincount(itr, weights=np.imag(T_[it]))
            WAV_[:, ind_f] /= trcount
        WAV0 = WAV_.copy()
    return WAV_


def _safe_svd(T_batch):
    """Batched SVD with NaN guard and gesdd→gesvd fallback.

    ``np.linalg.svd`` uses LAPACK gesdd (divide-and-conquer) which fails on
    ill-conditioned complex matrices from artefact-contaminated recordings.
    Non-finite values are zeroed first; if gesdd still does not converge we
    fall back to gesvd (QR iteration) one slice at a time via scipy.

    Parameters
    ----------
    T_batch : ndarray (nf, nrows, ncols), complex
        Stacked trajectory matrices (one per frequency bin).

    Returns
    -------
    U, s, Vh : ndarrays
    """
    if not np.all(np.isfinite(T_batch)):
        T_batch = np.nan_to_num(T_batch)
    try:
        return np.linalg.svd(T_batch, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _scipy_svd

        results = [
            _scipy_svd(T_batch[i], full_matrices=False, lapack_driver="gesvd")
            for i in range(T_batch.shape[0])
        ]
        return (
            np.stack([r[0] for r in results]),
            np.stack([r[1] for r in results]),
            np.stack([r[2] for r in results]),
        )


def _process_window(
    WAV_sl, it, ic, T_shape, scatter, r, imax, niter, gap_threshold, ppca_k
):
    """SVD rank reduction for one spatial window with precomputed trajectory geometry.

    Parameters
    ----------
    WAV_sl : ndarray (nc_w, nf), complex
        Frequency-domain data for this window.
    it : tuple of ndarray
        Row/col indices into the trajectory matrix (from ``trajectory()``).
    ic : ndarray
        Channel indices mapping trajectory entries to channels.
    T_shape : tuple
        Shape of the trajectory matrix ``(nrows, ncols)``.
    scatter : ndarray (n_entries, nc_w)
        Precomputed scatter matrix: maps trajectory entries back to channels.
    r, imax, niter, gap_threshold, ppca_k
        Algorithm parameters (see ``denoise_fxy``).

    Returns
    -------
    WAV_ : ndarray (nc_w, nf), complex
    """
    WAV_ = WAV_sl.copy()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(niter):
            T_batch = np.zeros((imax, *T_shape), dtype=complex)
            T_batch[:, it[0], it[1]] = WAV_[ic, :imax].T
            U, s, Vh = _safe_svd(T_batch)
            _apply_rank_threshold(s, r, gap_threshold)
            T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

            if ppca_k is not None:
                WAV_rec = (T_batch_[:, it[0], it[1]] @ scatter).T
                residual = np.abs(WAV_[:, :imax] - WAV_rec)
                med = np.median(residual, axis=0)
                mad = np.median(np.abs(residual - med[np.newaxis, :]), axis=0)
                mask = residual > (med + ppca_k * mad)
                WAV_clean = WAV_[:, :imax].copy()
                WAV_clean[mask] = WAV_rec[mask]
                T_batch[:, it[0], it[1]] = WAV_clean[ic, :].T
                U, s, Vh = _safe_svd(T_batch)
                _apply_rank_threshold(s, r, gap_threshold)
                T_batch_ = (U * s[:, np.newaxis, :]) @ Vh

            vals = T_batch_[:, it[0], it[1]]
            WAV_new = WAV_.copy()
            WAV_new[:, :imax] = (vals @ scatter).T
            WAV_ = WAV_new
    return WAV_


def denoise_fxy(WAV, x, y, r, imax=None, niter=1, gap_threshold=None, ppca_k=None):
    """
    F-X Cadzow denoiser using a single batched SVD over all frequency bins.

    Replaces the per-frequency-bin Python loop in `denoise` with a single
    ``np.linalg.svd`` call on the stacked ``(imax, nrows, ncols)`` tensor,
    giving a large speed improvement via LAPACK batched paths.

    Parameters
    ----------
    WAV : ndarray (nc, nf), complex
        Channels in the frequency domain (rfft output).
    x : ndarray (nc,)
        Lateral channel coordinates [µm].
    y : ndarray (nc,)
        Depth channel coordinates [µm].
    r : int
        Maximum SVD rank (number of plane waves retained).
    imax : int, optional
        Process only bins ``[:imax]``; higher bins pass through unchanged.
        Defaults to all bins.
    niter : int
        Number of Cadzow iterations.  Default 1.
    gap_threshold : float, optional
        If set, enables adaptive per-bin rank selection: the rank for each
        bin is the position of the largest s[i]/s[i+1] ratio, clamped to
        [1, r].  Falls back to ``r`` when no ratio exceeds ``gap_threshold``.
        A value around 1.5–2.0 is a reasonable starting point.  None uses
        fixed rank ``r`` (default).
    ppca_k : float, optional
        If set, enables a PPCA-style outlier correction each iteration: after
        the first rank reduction, channels whose per-frequency amplitude
        deviates from the model by more than ``median + ppca_k * MAD`` are
        replaced by the model prediction, then the SVD is repeated on the
        cleaned data.  Suppresses impedance-mismatch artefacts without
        altering channels consistent with the spatial model.  Typical values:
        2–5.  None disables (default).

    Returns
    -------
    WAV_ : ndarray (nc, nf), complex
    """
    nc, nf = WAV.shape
    imax = int(min(imax if imax is not None else nf, nf))
    T, it, ic, trcount = trajectory(x, y)
    scatter = np.zeros((len(ic), nc))
    scatter[np.arange(len(ic)), ic] = 1.0 / trcount[ic]
    return _process_window(
        WAV, it, ic, T.shape, scatter, r, imax, niter, gap_threshold, ppca_k
    )


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
    Apply Fxy rank-denoiser to a full recording of Neuropixel 1 probe geometry.

    .. deprecated::
        Use `cadzow_denoiser` instead — it accepts any probe geometry and uses
        batched SVD (via `denoise_fxy`) for a large speed improvement.

    :param wav: ntr, ns
    :param fs:
    :param ovx is the overlap in x
    :param nswx is the size of the window in x
    :param npad is the padding
    :return:
    """
    warnings.warn(
        "cadzow_np1() is deprecated; use cadzow_denoiser() for any probe geometry "
        "and faster batched SVD.",
        DeprecationWarning,
        stacklevel=2,
    )
    ntr, ns = wav.shape
    h = h or neuropixel.trace_header(version=1)
    nwinx = int(np.ceil((ntr + npad * 2 - ovx) / (nswx - ovx)))
    fscale = scipy.fft.rfftfreq(ns, d=1 / fs)
    WAV = scipy.fft.rfft(wav[:, :])
    imax = WAV.shape[-1] if fmax is None else int(np.searchsorted(fscale, fmax))
    padgain = scipy.signal.windows.hann(npad * 2)[:npad]
    WAV = np.r_[
        np.flipud(WAV[1 : npad + 1, :]) * padgain[:, np.newaxis],
        WAV,
        np.flipud(WAV[-npad - 2 : -1, :]) * np.flipud(np.r_[padgain, 1])[:, np.newaxis],
    ]  # apply padding
    x = np.r_[
        np.flipud(h["x"][1 : npad + 1]), h["x"], np.flipud(h["x"][-npad - 2 : -1])
    ]
    y = np.r_[
        np.flipud(h["y"][1 : npad + 1]) - 120,
        h["y"],
        np.flipud(h["y"][-npad - 2 : -1]) + 120,
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

    WAV_ = WAV_[npad : -npad - 1]  # remove padding
    wav_ = scipy.fft.irfft(WAV_)
    return wav_


def cadzow_denoiser(
    wav,
    h=None,
    fs=250.0,
    rank=5,
    niter=1,
    fmax=100.0,
    nswx=64,
    ovx=32,
    npad=0,
    gap_threshold=None,
    ppca_k=None,
    n_jobs=1,
):
    """
    F-X Cadzow denoiser for any Neuropixel probe geometry.

    Geometry-agnostic replacement for `cadzow_np1`: accepts an explicit probe
    header ``h`` (NP1, NP2, or any custom geometry) and uses batched SVD via
    `_process_window` instead of a per-bin Python loop.  Window trajectories are
    precomputed once and dispatched to parallel workers via joblib when
    ``n_jobs != 1``.

    Parameters
    ----------
    wav : ndarray (nc, ns), float
        Raw LFP, channels × samples.
    h : dict or None
        Probe header with keys ``'x'`` and ``'y'`` (channel coordinates [µm]).
        Must have at least ``nc`` elements.  Defaults to the NP1 geometry.
        Pass ``neuropixel.trace_header(version=2)`` for NP2 probes.
    fs : float
        Sampling rate [Hz].  Default 250.
    rank : int
        Maximum SVD rank (plane waves retained per frequency bin).  Default 5.
    niter : int
        Number of Cadzow iterations.  Default 1.
    fmax : float or None
        Maximum frequency processed [Hz]; higher bins pass through unchanged.
        None processes all bins up to Nyquist.  Default 100.
    nswx : int
        Channel-window width (number of channels per spatial window).
        Default 64.
    ovx : int
        Channel-window overlap in channels.  Any value in ``[1, nswx - 1]`` is
        valid; overlaps above 50% (``ovx > nswx // 2``) use a Hann synthesis
        window with running normalisation instead of the partition-of-unity gain.
        Default 16 (kept for backward compatibility; recommended value is
        ``nswx // 2``, e.g. 32 for the default ``nswx=64``).
    npad : int
        Reflective channel padding on each side.  Default 0.
    gap_threshold : float, optional
        Adaptive rank: use the largest singular-value gap as the per-bin rank,
        clamped to [1, rank].  Falls back to fixed rank when the maximum ratio
        is below this value.  None disables adaptive selection.
    ppca_k : float, optional
        PPCA-style outlier correction threshold in MAD units.  After an initial
        rank reduction, channels deviating from the model by more than
        ``median + ppca_k * MAD`` (per frequency bin) are replaced by the model
        prediction and the SVD is repeated.  Suppresses impedance-mismatch
        artefacts.  Typical values: 2–5.  None disables (default).
    n_jobs : int
        Number of parallel workers for the spatial-window loop.  ``np.linalg.svd``
        releases the GIL, so threads (``prefer='threads'``) are used.  Default 1
        (serial).  Use ``-1`` for all available cores.

    Returns
    -------
    wav_ : ndarray (nc, ns), float32
    """
    from joblib import Parallel, delayed

    ntr, ns = wav.shape
    if h is None:
        _h = neuropixel.trace_header(version=1)
        h = {k: v[:ntr] for k, v in _h.items()}

    nwinx = int(np.ceil((ntr + npad * 2 - ovx) / (nswx - ovx)))
    WAV = scipy.fft.rfft(wav)
    fscale = scipy.fft.rfftfreq(ns, d=1.0 / fs)
    imax = WAV.shape[1] if fmax is None else int(np.searchsorted(fscale, fmax))
    padgain = scipy.signal.windows.hann(npad * 2)[:npad]
    WAV = np.r_[
        np.flipud(WAV[1 : npad + 1, :]) * padgain[:, np.newaxis],
        WAV,
        np.flipud(WAV[-npad - 2 : -1, :]) * np.flipud(np.r_[padgain, 1])[:, np.newaxis],
    ]
    x = np.r_[
        np.flipud(h["x"][1 : npad + 1]), h["x"], np.flipud(h["x"][-npad - 2 : -1])
    ]
    y = np.r_[
        np.flipud(h["y"][1 : npad + 1]) - 120,
        h["y"],
        np.flipud(h["y"][-npad - 2 : -1]) + 120,
    ]

    # Synthesis windowing strategy depends on overlap fraction:
    #   ovx ≤ nswx//2  (≤50%): exact partition-of-unity gain window — backward-compatible,
    #                           no normalisation required.
    #   ovx > nswx//2  (>50%): Hann synthesis window + running normalisation sum.
    #                           Supports any ovx in [1, nswx-1]; the accumulation step
    #                           divides each channel by the total Hann weight it received,
    #                           so blending is always correct regardless of overlap ratio.
    step = nswx - ovx
    # Backward-compatible branch: for ovx ≤ nswx//2 the original partition-of-unity
    # gain window is used unchanged; the WOLA path only activates for higher overlaps.
    high_overlap = ovx > nswx // 2
    if not high_overlap:
        hanning = scipy.signal.windows.hann(ovx * 2 - 1)[:ovx]
        gain_window = np.r_[hanning, np.ones(nswx - ovx * 2), np.flipud(hanning)]
    else:
        hann_full = scipy.signal.windows.hann(nswx)

    # Precompute per-window geometry (trajectory + scatter + gain) once before dispatch
    windows = []
    for i, firstx in enumerate(np.arange(nwinx) * step):
        lastx = int(firstx + nswx)
        sl = slice(firstx, lastx)
        nc_w = len(x[sl])
        if high_overlap:
            gw = hann_full.copy()
            if i == 0:  # replace fade-in with ones at probe start
                gw[:step] = 1.0
            if i == nwinx - 1:  # replace fade-out with ones at probe end
                gw[max(0, nswx - step) :] = 1.0
            gw = gw[:nc_w]
        else:
            if firstx == 0:
                gw = np.r_[np.ones(nswx - ovx), np.flipud(hanning)][:nc_w]
            elif lastx >= ntr:
                gw = np.r_[hanning, np.ones(nswx - ovx)][:nc_w]
            else:
                gw = gain_window[:nc_w]
        T, it, ic, trcount = trajectory(x[sl], y[sl])
        scatter = np.zeros((len(ic), nc_w))
        scatter[np.arange(len(ic)), ic] = 1.0 / trcount[ic]
        windows.append((sl, gw, it, ic, T.shape, scatter))

    def _worker(sl, gw, it, ic, T_shape, scatter):
        return (
            sl,
            gw,
            _process_window(
                WAV[sl],
                it,
                ic,
                T_shape,
                scatter,
                rank,
                imax,
                niter,
                gap_threshold,
                ppca_k,
            ),
        )

    if n_jobs == 1:
        results = [_worker(*w) for w in windows]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker)(*w) for w in windows
        )

    WAV_ = np.zeros_like(WAV)
    if high_overlap:
        WAV_norm = np.zeros(WAV.shape[0])
        for sl, gw, WAV_w in results:
            WAV_[sl] += WAV_w * gw[:, np.newaxis]
            WAV_norm[sl] += gw
        WAV_ /= np.maximum(WAV_norm, 1e-6)[:, np.newaxis]
    else:
        for sl, gw, WAV_w in results:
            WAV_[sl] += WAV_w * gw[:, np.newaxis]

    WAV_ = WAV_[
        npad : -npad - 1
    ]  # remove channel padding (npad=0 trims the phantom tail row)
    return scipy.fft.irfft(WAV_).astype(np.float32)
