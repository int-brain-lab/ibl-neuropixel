"""
Window generator, front detections, rms
"""

import numpy as np
import scipy


def sync_timestamps(tsa, tsb, tbin=0.1, return_indices=False, linear=False):
    """
    Sync two arrays of time stamps
    :param tsa: vector of timestamps
    :param tsb: vector of timestamps
    :param tbin: time bin length
    :param return_indices (bool), if True returns 2 sets of indices for tsa and tsb with
    :param linear: (bool) if True, restricts the fit to linear
    identified matches
    :return:
     function: interpolation function such as fnc(tsa) = tsb
     float: drift in ppm
     numpy array: of indices ia
     numpy array: of indices ib
    """

    def _interp_fcn(tsa, tsb, ib, linear=linear):
        # now compute the bpod/fpga drift and precise time shift
        ab = np.polyfit(tsa[ib >= 0], tsb[ib[ib >= 0]] - tsa[ib >= 0], 1)
        drift_ppm = ab[0] * 1e6
        if linear:
            fcn_a2b = lambda x: x * (1 + ab[0]) + ab[1]  # noqa
        else:
            fcn_a2b = scipy.interpolate.interp1d(
                tsa[ib >= 0], tsb[ib[ib >= 0]], fill_value="extrapolate"
            )
        return fcn_a2b, drift_ppm

    # assert sorted inputs
    tmin = np.min([np.min(tsa), np.min(tsb)])
    tmax = np.max([np.max(tsa), np.max(tsb)])
    # brute force correlation to get an estimate of the delta_t between series
    x = np.zeros(int(np.ceil(tmax - tmin) / tbin))
    y = np.zeros_like(x)
    x[np.int32(np.floor((tsa - tmin) / tbin))] = 1
    y[np.int32(np.floor((tsb - tmin) / tbin))] = 1
    delta_t = (
        parabolic_max(scipy.signal.correlate(x, y, mode="full"))[0] - x.shape[0] + 1
    ) * tbin
    # do a first assignment at a DT threshold
    ib = np.zeros(tsa.shape, dtype=np.int32) - 1
    threshold = tbin
    for m in np.arange(tsa.shape[0]):
        dt = np.abs(tsa[m] - delta_t - tsb)
        inds = np.where(dt < threshold)[0]
        if inds.size == 1:
            ib[m] = inds[0]
        elif inds.size > 1:
            candidates = inds[~np.isin(inds, ib[:m])]
            if candidates.size == 1:
                ib[m] = candidates[0]
            elif candidates.size > 1:
                ib[m] = inds[np.argmin(dt[inds])]

    fcn_a2b, _ = _interp_fcn(tsa, tsb, ib)
    # do a second assignment - this time a full matrix of candidate matches is computed
    # the most obvious matches are assigned first and then one by one
    iamiss = np.where(ib < 0)[0]
    ibmiss = np.setxor1d(np.arange(tsb.size), ib[ib >= 0])
    dt = np.abs(fcn_a2b(tsa[iamiss]) - tsb[ibmiss][:, np.newaxis])
    dt[dt > tbin] = np.nan
    while ~np.all(np.isnan(dt)):
        _b, _a = np.unravel_index(np.nanargmin(dt), dt.shape)
        ib[iamiss[_a]] = ibmiss[_b]
        dt[:, _a] = np.nan
        dt[_b, :] = np.nan
    fcn_a2b, drift_ppm = _interp_fcn(tsa, tsb, ib, linear=linear)

    if return_indices:
        return fcn_a2b, drift_ppm, np.where(ib >= 0)[0], ib[ib >= 0]
    else:
        return fcn_a2b, drift_ppm


def parabolic_max(x):
    """
    Maximum picking with parabolic interpolation around the maxima
    :param x: 1d or 2d array
    :return: interpolated max index, interpolated max
    """
    # for 2D arrays, operate along the last dimension
    ns = x.shape[-1]
    axis = -1
    imax = np.argmax(x, axis=axis)

    if x.ndim == 1:
        v010 = x[np.maximum(np.minimum(imax + np.array([-1, 0, 1]), ns - 1), 0)]
        v010 = v010[:, np.newaxis]
    else:
        v010 = np.vstack(
            (
                x[..., np.arange(x.shape[0]), np.maximum(imax - 1, 0)],
                x[..., np.arange(x.shape[0]), imax],
                x[..., np.arange(x.shape[0]), np.minimum(imax + 1, ns - 1)],
            )
        )
    poly = np.matmul(0.5 * np.array([[1, -2, 1], [-1, 0, 1], [0, 2, 0]]), v010)
    ipeak = -poly[1] / (poly[0] + np.double(poly[0] == 0)) / 2
    maxi = poly[2] + ipeak * poly[1] + ipeak**2.0 * poly[0]
    ipeak += imax
    # handle edges
    iedges = np.logical_or(imax == 0, imax == ns - 1)
    if x.ndim == 1:
        maxi = v010[1, 0] if iedges else maxi[0]
        ipeak = imax if iedges else ipeak[0]
    else:
        maxi[iedges] = v010[1, iedges]
        ipeak[iedges] = imax[iedges]
    return ipeak, maxi


def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds, gpu=False):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :param gpu: bool
    :return: lambda function
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    def _cos(x):
        return (1 - gp.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * gp.pi)) / 2

    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def fronts(x, axis=-1, step=1):
    """
    Detects Rising and Falling edges of a voltage signal, returns indices and

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, 1) value of the step to detect
    :return: numpy array of indices, numpy array of rises (1) and falls (-1)
    """
    d = np.diff(x, axis=axis)
    ind = np.array(np.where(np.abs(d) >= step))
    sign = d[tuple(ind)]
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0], sign
    else:
        return ind, sign


def falls(x, axis=-1, step=-1, analog=False):
    """
    Detects Falling edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, -1) value of the step to detect
    :param analog: (optional, False) in case the signal is analog, converts the voltage to boolean (> step) before
     detecting edges
    :return: numpy array
    """
    return rises(-x, axis=axis, step=-step, analog=analog)


def rises(x, axis=-1, step=1, analog=False):
    """
    Detect Rising edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :param step: (optional, 1) amplitude of the step to detect
    :param analog: (optional, False) in case the signal is analog, converts the voltage to boolean (> step) before
     detecting edges
    :return: numpy array
    """
    if analog:
        x = (x > step).astype(np.float64)
        step = 1
    ind = np.array(np.where(np.diff(x, axis=axis) >= step))
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0]
    else:
        return ind


def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x**2, axis=axis))


def make_channel_index(geom, radius=200.0, pad_val=None):
    """
    Create a channel index array for a Neuropixels probe based on geometry and proximity.

    This function generates an array where each row represents a channel and contains
    the IDs of neighboring channels within a specified radius. It's useful for
    operations that require knowledge of nearby channels on a Neuropixels probe.

    Parameters:
    -----------
    geom : array-like or dict
        Either:
        - A 2D array representing the geometry of the Neuropixels probe.
          Each row should contain the (x, y) coordinates of a channel.
        - A dictionary with keys 'x' and 'y', each containing a 1D array of

    radius : float, optional
        The maximum distance (in micrometers) within which channels are considered
        neighbors. Default is 200.0 μm.

    pad_val : int, optional
        The value used to pad rows for channels with fewer neighbors than the
        maximum. If None, it defaults to the total number of channels.

    Returns:
    --------
    channel_idx : numpy.ndarray
        A 2D integer array where each row corresponds to a channel and contains
        the IDs of its neighboring channels. Rows are padded with `pad_val` if
        a channel has fewer neighbors than the maximum possible.
    """
    if isinstance(geom, dict):
        geom = np.c_[geom["x"], geom["y"]]
    neighbors = (
        scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(geom)) <= radius
    )
    n_nbors = np.max(np.sum(neighbors, 0))
    nc = geom.shape[0]
    if pad_val is None:
        pad_val = nc
    channel_idx = np.full((nc, n_nbors), pad_val, dtype=int)
    for c in range(nc):
        ch_idx = np.flatnonzero(neighbors[c, :])
        channel_idx[c, : ch_idx.shape[0]] = ch_idx

    return channel_idx


class WindowGenerator(object):
    """
    A utility class for generating sliding windows for signal processing applications.

    WindowGenerator provides various methods to iterate through windows of a signal
    with configurable window size and overlap. It's particularly useful for operations
    like spectrograms, filtering, or any processing that requires windowed analysis.

    Parameters
    ----------
    ns : int
        Total number of samples in the signal to be windowed.
    nswin : int
        Number of samples in each window.
    overlap : int
        Number of samples that overlap between consecutive windows.

    Attributes
    ----------
    ns : int
        Total number of samples in the signal.
    nswin : int
        Number of samples in each window.
    overlap : int
        Number of samples that overlap between consecutive windows.
    nwin : int
        Total number of windows.
    iw : int or None
        Current window index during iteration.

    Notes
    -----
    For straightforward spectrogram or periodogram implementation,
    scipy methods are recommended over this class.

    Examples
    --------
    # straight windowing without overlap
    >>> wg = WindowGenerator(ns=1000, nwin=111)
    >>> signal = np.random.randn(1000)
    >>> for window_slice in wg.slice:
    ...     window_data = signal[window_slice]
    ...     # Process window_data

    # windowing with overlap (ie. buffers for apodization)
    >>> for win_slice, valid_slice, win_valid_slice in wg.slices_valid:
    ...     window = signal[win_slice]
    ...     # Process window
    ...     processed = some_function_with_edge_effect(window)
    ...     # Only use the valid portion for reconstruction
    ...     recons[valid_slice] = processed[win_valid_slice]

    # splicing add a fade-in / fade-out in the overlap so that reconstruction has unit amplitude
    >>> recons = np.zeros_like(signal)
    >>> for win_slice, amplitude in wg.splice:
    ...     window = signal[win_slice]
    ...     # Process window
    ...     processed = some_function(window)
    ...     # The processed windows is weighted with the amplitude and added to the reconstructed signal
    ...     recons[win_slice] = recons[win_slice] + processed * amplitude
    """

    def __init__(self, ns, nswin, overlap):
        """
        :param ns: number of sample of the signal along the direction to be windowed
        :param nswin: number of samples of the window
        :return: dsp.WindowGenerator object:
        """
        self.ns = int(ns)
        self.nswin = int(nswin)
        self.overlap = int(overlap)
        self.nwin = int(np.ceil(float(ns - nswin) / float(nswin - overlap))) + 1
        self.iw = None

    @property
    def splice(self):
        """
        Generator that yields slices and amplitude arrays for windowed signal processing with splicing.

        This property provides a convenient way to iterate through all windows with their
        corresponding amplitude arrays for proper signal reconstruction. The amplitude arrays
        contain tapering values (from a Hann window) at the overlapping regions to ensure
        unit amplitude of all samples of the original signal

        Yields
        ------
        tuple
            A tuple containing:
            - slice: A Python slice object representing the current window
            - amp: A numpy array containing amplitude values for proper splicing/tapering
              at overlap regions

        Notes
        -----
        This is particularly useful for overlap-add methods where windows need to be
        properly weighted before being combined in the reconstruction process.
        """
        for first, last, amp in self.firstlast_splicing:
            yield slice(first, last), amp

    @property
    def firstlast_splicing(self):
        """
        cf. self.splice
        """
        w = scipy.signal.windows.hann((self.overlap + 1) * 2 + 1, sym=True)[
            1 : self.overlap + 1
        ]
        assert np.all(np.isclose(w + np.flipud(w), 1))

        for first, last in self.firstlast:
            amp = np.ones(last - first)
            amp[: self.overlap] = 1 if first == 0 else w
            amp[-self.overlap :] = 1 if last == self.ns else np.flipud(w)
            yield (first, last, amp)

    @property
    def firstlast_valid(self):
        """
        Generator that yields a tuple of first, last, first_valid, last_valid index of windows
        The valid indices span up to half of the overlap
        :return:
        """
        assert self.overlap % 2 == 0, "Overlap must be even"
        for first, last in self.firstlast:
            first_valid = 0 if first == 0 else first + self.overlap // 2
            last_valid = last if last == self.ns else last - self.overlap // 2
            yield (first, last, first_valid, last_valid)

    @property
    def firstlast(self):
        """
        Generator that yields first and last index of windows

        :return: tuple of [first_index, last_index] of the window
        """
        self.iw = 0
        first = 0
        while True:
            last = first + self.nswin
            last = min(last, self.ns)
            yield (first, last)
            if last == self.ns:
                break
            first += self.nswin - self.overlap
            self.iw += 1

    @property
    def slice(self):
        """
        Generator that yields slice objects for each window in the signal.

        This property provides a convenient way to iterate through all windows
        defined by the WindowGenerator parameters. Each yielded slice can be
        used directly to index into the original signal array.

        Yields
        ------
        slice
            A Python slice object representing the current window, defined by
            its first and last indices. The slice can be used to extract the
            corresponding window from the original signal.
        """
        for first, last in self.firstlast:
            yield slice(first, last)

    @property
    def slices_valid(self):
        """
        Generator that yields slices for windowed signal processing with valid regions.

        This method generates tuples of slice objects that can be used to extract windows
        from a signal and identify the valid (non-overlapping) portions within each window.
        It's particularly useful for reconstruction operations where overlapping regions
        need special handling.

        Yields
        ------
        tuple
            A tuple containing three slice objects:
            - slice(first, last): The full window slice
            - slice(first_valid, last_valid): The valid portion of the signal in absolute indices
            - slice_window_valid: The valid portion relative to the window (for use within the window)

        Notes
        -----
        This generator relies on the firstlast_valid property which provides the
        indices for both the full windows and their valid regions.
        """
        for first, last, first_valid, last_valid in self.firstlast_valid:
            slice_window_valid = slice(
                first_valid - first, None if (lv := -(last - last_valid)) == 0 else lv
            )
            yield slice(first, last), slice(first_valid, last_valid), slice_window_valid

    def slice_array(self, sig, axis=-1):
        """
        Provided an array or sliceable object, generator that yields
        slices corresponding to windows. Especially useful when working on memmpaps

        :param sig: array
        :param axis: (optional, -1) dimension along which to provide the slice
        :return: array slice Generator
        """
        for first, last in self.firstlast:
            yield np.take(sig, np.arange(first, last), axis=axis)

    def tscale(self, fs):
        """
        Returns the time scale associated with Window slicing (middle of window)
        :param fs: sampling frequency (Hz)
        :return: time axis scale
        """
        return np.array(
            [(first + (last - first - 1) / 2) / fs for first, last in self.firstlast]
        )


def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".
    scipy.signal.ricker was removed in SciPy 1.15

    It models the function:

        ``A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)``,

    where ``A = 2/(sqrt(3*a)*(pi**0.25))``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.
    """
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total
