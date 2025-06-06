import json
import logging
from pathlib import Path
import re
import shutil
import time

import numpy as np

import mtscomp
from iblutil.io import hashfile
from iblutil.util import Bunch
import one.alf.path

import neuropixel

SAMPLE_SIZE = 2  # int16
DEFAULT_BATCH_SIZE = 1e6
_logger = logging.getLogger("ibllib")


def _get_companion_file(sglx_file, pattern=".meta"):
    # on SDSC there is a possibility that there is an UUID string in the filename
    sglx_file = Path(sglx_file)
    companion_file = sglx_file.with_suffix(pattern)
    if not companion_file.exists():
        search_pattern = f"{one.alf.path.remove_uuid_string(sglx_file).stem}*{pattern}"
        companion_file = next(sglx_file.parent.glob(search_pattern), companion_file)
    return companion_file


class Reader:
    """
    Class for SpikeGLX reading purposes
    Some format description was found looking at the Matlab SDK here
    https://github.com/billkarsh/SpikeGLX/blob/master/MATLAB-SDK/DemoReadSGLXData.m

    To open a spikeglx file that has an associated meta-data file:
    sr = spikeglx.Reader(bin_file_path)

    To open a flat binary file:

    sr = spikeglx.Reader(bin_file_path, nc=385, ns=nsamples, fs=30000)
    one can provide more options to the reader:
    sr = spikeglx.Reader(..., dtype='int16, s2mv=2.34375e-06)

    usual sample 2 mv conversion factors (cf. neuropixel module):
        s2mv = 2.34375e-06 (NP1 ap banc) : default value used
        s2mv = 4.6875e-06 (NP1 lfp band)

    Note: To release system resources the close method must be called
    """

    def __init__(
        self,
        sglx_file,
        open=True,
        nc=None,
        ns=None,
        fs=None,
        dtype="int16",
        s2v=None,
        nsync=None,
        ignore_warnings=False,
        meta_file=None,
        ch_file=None,
        sort=True,
    ):
        """
        An interface for reading data from a SpikeGLX file
        :param sglx_file: Path to a SpikeGLX file (compressed or otherwise), or to a meta-data file
        :param open: when True the file is opened
        :param sort: (True) by default always return channels sorted by shank, row and column. If set to false,
        the data will be returned as written on disk, for NP2 versions this may result in interleaved shanks
        """
        self.geometry = None
        self.ignore_warnings = ignore_warnings
        sglx_file = Path(sglx_file)
        meta_file = meta_file or _get_companion_file(sglx_file, ".meta")
        # only used if MTSCOMP compressed
        self.ch_file = ch_file

        if meta_file == sglx_file:
            # if a meta-data file is provided, try to get the binary file
            self.file_bin = (
                sglx_file.with_suffix(".cbin")
                if sglx_file.with_suffix(".cbin").exists()
                else None
            )
            self.file_bin = (
                sglx_file.with_suffix(".bin")
                if sglx_file.with_suffix(".bin").exists()
                else None
            )
        else:
            self.file_bin = sglx_file
        self.nbytes = self.file_bin.stat().st_size if self.file_bin else None
        self.dtype = np.dtype(dtype)

        if not meta_file.exists():
            # if no meta-data file is provided, try to get critical info from the binary file
            # by seeing if filesize checks out with neuropixel 384 channels
            if self.file_bin.stat().st_size / 384 % 2 == 0:
                nc = nc or 384
                ns = ns or self.file_bin.stat().st_size / 2 / 384
                fs = fs or 30000
            elif self.file_bin.stat().st_size / 385 % 2 == 0:
                nc = nc or 385
                ns = ns or self.file_bin.stat().st_size / 2 / 385
                fs = fs or 30000
                nsync = nsync or 1

            err_str = "Instantiating an Reader without meta data requires providing nc, fs and nc parameters"
            assert nc is not None and fs is not None and nc is not None, err_str
            self.file_meta_data = None
            self.meta = None
            self._nc, self._fs, self._ns = (int(nc), int(fs), int(ns))
            # handles default parameters: if int16 we assume it's a raw recording, we've checked the
            # multiple of the file size above to determine if there is a sync or not
            self._nsync = nsync or 0
            if s2v is None:
                s2v = neuropixel.S2V_AP if self.dtype == np.dtype("int16") else 1.0
            self.channel_conversion_sample2v = {"samples": np.ones(nc) * s2v}
            if self._nsync > 0:
                self.channel_conversion_sample2v["samples"][-nsync:] = 1
            self.geometry = neuropixel.trace_header(version=1)
        else:
            # normal case we continue reading and interpreting the metadata file
            self.file_meta_data = meta_file
            self.meta = read_meta_data(meta_file)
            self.channel_conversion_sample2v = _conversion_sample2v_from_meta(self.meta)
            self._raw = None
            self.geometry, order = geometry_from_meta(
                self.meta, return_index=True, sort=sort
            )
            self.raw_channel_order = np.arange(self.nc)
            if self.geometry is not None:  # nidq files won't return any geometry here
                self.raw_channel_order[: order.size] = order
        if open and self.file_bin:
            self.open()

    def open(self):
        # if we are not looking at a compressed file, use a memmap, otherwise instantiate mtscomp
        sglx_file = str(self.file_bin)
        if self.is_mtscomp:
            self._raw = mtscomp.Reader()
            ch_file = self.ch_file or _get_companion_file(sglx_file, ".ch")
            self._raw.open(self.file_bin, ch_file)
            if self._raw.shape != (self.ns, self.nc):
                ftsec = self._raw.shape[0] / self.fs
                if not self.ignore_warnings:  # avoid the checks for streaming data
                    _logger.warning(
                        f"{sglx_file} : meta data and compressed chunks dont checkout\n"
                        f"File duration: expected {self.meta['fileTimeSecs']},"
                        f" actual {ftsec}\n"
                        f"Will attempt to fudge the meta-data information."
                    )
                self.meta["fileTimeSecs"] = ftsec
        else:
            if self.nc * self.ns * self.dtype.itemsize != self.nbytes:
                ftsec = (
                    self.file_bin.stat().st_size
                    / self.dtype.itemsize
                    / self.nc
                    / self.fs
                )
                if self.meta is not None:
                    if not self.ignore_warnings:
                        _logger.warning(
                            f"{sglx_file} : meta data and filesize do not checkout\n"
                            f"File size: expected {self.meta['fileSizeBytes']},"
                            f" actual {self.file_bin.stat().st_size}\n"
                            f"File duration: expected {self.meta['fileTimeSecs']},"
                            f" actual {ftsec}\n"
                            f"Will attempt to fudge the meta-data information."
                        )
                    self.meta["fileTimeSecs"] = ftsec
            self._raw = np.memmap(
                sglx_file, dtype=self.dtype, mode="r", shape=(self.ns, self.nc)
            )

    def close(self):
        if self.is_open:
            getattr(self._raw, "_mmap", self._raw).close()

    def __enter__(self):
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.read(nsel=item, sync=False)
        elif len(item) == 2:
            return self.read(nsel=item[0], csel=item[1], sync=False)

    @property
    def sample2volts(self):
        return self.channel_conversion_sample2v[self.type]

    @property
    def shape(self):
        return self.ns, self.nc

    @property
    def is_open(self):
        return self._raw is not None

    @property
    def is_mtscomp(self):
        return "cbin" in self.file_bin.suffix

    @property
    def version(self):
        """Gets the version string: '3A', '3B2', '3B1', 'NP2.1', 'NP2.4'"""
        return (
            None if self.meta is None else _get_neuropixel_version_from_meta(self.meta)
        )

    @property
    def major_version(self):
        """Gets the the major version int: 1 or 2"""
        return (
            None
            if self.meta is None
            else _get_neuropixel_major_version_from_meta(self.meta)
        )

    @property
    def rl(self):
        return self.ns / self.fs

    @property
    def type(self):
        """:return: ap, lf or nidq. Useful to index dictionaries"""
        if not self.meta:
            return "samples"
        return _get_type_from_meta(self.meta)

    @property
    def fs(self):
        """:return: sampling frequency (Hz)"""
        return self._fs if self.meta is None else _get_fs_from_meta(self.meta)

    @property
    def nc(self):
        """:return: number of channels"""
        return self._nc if self.meta is None else _get_nchannels_from_meta(self.meta)

    @property
    def nsync(self):
        """:return: number of sync channels"""
        return (
            self._nsync
            if self.meta is None
            else len(_get_sync_trace_indices_from_meta(self.meta))
        )

    @property
    def ns(self):
        """:return: number of samples"""
        if self.meta is None:
            return self._ns
        return int(np.round(self.meta.get("fileTimeSecs") * self.fs))

    @property
    def range_volts(self):
        """
        Returns the maximum voltage that can be recorded before saturation
        :return: [nc, ] array of float32 (V)
        """
        if not self.meta:
            return self.sample2volts * np.nan
        maxint = _get_max_int_from_meta(self.meta)
        return self.sample2volts * maxint

    def read(self, nsel=slice(0, 10000), csel=slice(None), sync=True):
        """
        Read from slices or indexes
        :param slice_n: slice or sample indices
        :param slice_c: slice or channel indices
        :return: float32 array
        """
        if not self.is_open:
            raise IOError("Reader not open; call `open` before `read`")
        if hasattr(self, "raw_channel_order"):
            csel = self.raw_channel_order[csel]
        darray = self._raw[nsel, :].astype(np.float32, copy=True)[..., csel]
        darray *= self.channel_conversion_sample2v[self.type][csel]
        if sync:
            return darray, self.read_sync(nsel)
        else:
            return darray

    def read_samples(self, first_sample=0, last_sample=10000, channels=None):
        """
        reads all channels from first_sample to last_sample, following numpy slicing convention
        sglx.read_samples(first=0, last=100) would be equivalent to slicing the array D
        D[:,0:100] where the last axis represent time and the first channels.

         :param first_sample: first sample to be read, python slice-wise
         :param last_sample:  last sample to be read, python slice-wise
         :param channels: slice or numpy array of indices
         :return: numpy array of int16
        """
        if channels is None:
            channels = slice(None)
        return self.read(slice(first_sample, last_sample), channels)

    def read_sync_digital(self, _slice=slice(0, 10000)):
        """
        Reads only the digital sync trace at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_digital(slice(0,10000))
        """
        if not self.is_open:
            raise IOError("Reader not open; call `open` before `read`")
        if not self.meta:
            _logger.warning("Sync trace not labeled in metadata. Assuming last trace")
        return split_sync(
            self._raw[_slice, _get_sync_trace_indices_from_meta(self.meta)]
        )

    def read_sync_analog(self, _slice=slice(0, 10000)):
        """
        Reads only the analog sync traces at specified samples using slicing syntax
        >>> sync_samples = sr.read_sync_analog(slice(0,10000))
        """
        if not self.meta:
            return
        csel = _get_analog_sync_trace_indices_from_meta(self.meta)
        if not csel:
            return
        else:
            return self.read(nsel=_slice, csel=csel, sync=False)

    def read_sync(self, _slice=slice(0, 10000), threshold=1.2, floor_percentile=10):
        """
        Reads all sync trace. Convert analog to digital with selected threshold and append to array
        :param _slice: samples slice
        :param threshold: (V) threshold for front detection, defaults to 1.2 V
        :param floor_percentile: 10% removes the percentile value of the analog trace before
         thresholding. This is to avoid DC offset drift
        :return: int8 array
        """
        digital = self.read_sync_digital(_slice)
        analog = self.read_sync_analog(_slice)
        if analog is not None and floor_percentile:
            analog -= np.percentile(analog, 10, axis=0)
        if analog is None:
            return digital
        analog[np.where(analog < threshold)] = 0
        analog[np.where(analog >= threshold)] = 1
        return np.concatenate((digital, np.int8(analog)), axis=1)

    def compress_file(self, keep_original=True, **kwargs):
        """
        Compresses
        :param keep_original: defaults True. If False, the original uncompressed file is deleted
         and the current spikeglx.Reader object is modified in place
        :param kwargs:
        :return: pathlib.Path of the compressed *.cbin file
        """
        file_tmp = self.file_bin.with_suffix(".cbin_tmp")
        assert not self.is_mtscomp
        mtscomp.compress(
            self.file_bin,
            out=file_tmp,
            outmeta=self.file_bin.with_suffix(".ch"),
            sample_rate=self.fs,
            n_channels=self.nc,
            dtype=self.dtype,
            **kwargs,
        )
        file_out = file_tmp.with_suffix(".cbin")
        file_tmp.rename(file_out)
        if not keep_original:
            self.file_bin.unlink()
            self.file_bin = file_out
        return file_out

    def decompress_file(self, keep_original=True, file_ch=None, **kwargs):
        """
        Decompresses a mtscomp file
        :param keep_original: defaults True. If False, the original compressed file (input)
        is deleted and the current spikeglx.Reader object is modified in place
        NB: This is not equivalent to overwrite (which replaces the output file)
        :return: pathlib.Path of the decompressed *.bin file
        """
        if "out" not in kwargs:
            kwargs["out"] = self.file_bin.with_suffix(".bin")
        assert self.is_mtscomp
        if file_ch is None:
            file_ch = self.file_bin.with_suffix(".ch")

        r = mtscomp.decompress(self.file_bin, file_ch, **kwargs)
        r.close()
        if not keep_original:
            self.close()
            self.file_bin.unlink()
            self.file_bin.with_suffix(".ch").unlink()
            self.file_bin = kwargs["out"]
        return kwargs["out"]

    def decompress_to_scratch(self, file_meta=None, file_ch=None, scratch_dir=None):
        """
        Decompresses the file to a temporary directory
        Copy over the metadata file
        """
        if file_meta is None:
            file_meta = Path(self.file_bin).with_suffix(".meta")

        if scratch_dir is None:
            bin_file = Path(self.file_bin).with_suffix(".bin")
        else:
            scratch_dir.mkdir(exist_ok=True, parents=True)
            bin_file = scratch_dir / Path(self.file_bin).with_suffix(".bin").name
            file_meta_scratch = scratch_dir / file_meta.name
            shutil.copy(self.file_meta_data, file_meta_scratch)
        if not bin_file.exists():
            t0 = time.time()
            _logger.info("File is compressed, decompressing to a temporary file...")
            self.decompress_file(
                keep_original=True,
                file_ch=file_ch,
                out=bin_file.with_suffix(".bin_temp"),
                check_after_decompress=False,
                overwrite=True,
            )
            shutil.move(bin_file.with_suffix(".bin_temp"), bin_file)
            _logger.info(f"Decompression complete: {time.time() - t0:.2f}s")
        return bin_file

    def verify_hash(self):
        """
        Computes SHA-1 hash and returns True if it matches metadata, False otherwise
        :return: boolean
        """
        if self.is_mtscomp:
            with open(self.file_bin.with_suffix(".ch")) as fid:
                mtscomp_params = json.load(fid)
            sm = mtscomp_params.get("sha1_compressed", None)
            if sm is None:
                _logger.warning(
                    "SHA1 hash is not implemented for compressed ephys. To check "
                    "the spikeglx acquisition hash, uncompress the file first !"
                )
                return True
            sm = sm.upper()
        else:
            sm = self.meta.fileSHA1
        sc = hashfile.sha1(self.file_bin).upper()
        if sm == sc:
            log_func = _logger.info
        else:
            log_func = _logger.error
        log_func(f"SHA1 metadata: {sm}")
        log_func(f"SHA1 computed: {sc}")
        return sm == sc


class OnlineReader(Reader):
    @property
    def ns(self):
        return int(self.file_bin.stat().st_size / self.dtype.itemsize / self.nc)


def read(sglx_file, first_sample=0, last_sample=10000):
    """
    Function to read from a spikeglx binary file without instantiating the class.
    Gets the meta-data as well.

    >>> ibllib.io.spikeglx.read('/path/to/file.bin', first_sample=0, last_sample=1000)

    :param sglx_file: full path the the binary file to read
    :param first_sample: first sample to be read, python slice-wise
    :param last_sample: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with Reader(sglx_file) as sglxr:
        D, sync = sglxr.read_samples(first_sample=first_sample, last_sample=last_sample)
    return D, sync, sglxr.meta


def read_meta_data(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values

    :param md_file: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    with open(md_file) as fid:
        md = fid.read()
    d = {}
    for a in md.splitlines():
        k, v = a.split("=", maxsplit=1)
        # if all numbers, try to interpret the string
        if v and re.fullmatch("[0-9,.]*", v) and v.count(".") < 2:
            v = [float(val) for val in v.split(",")]
            # scalars should not be nested
            if len(v) == 1:
                v = v[0]
        # tildes in keynames removed
        d[k.replace("~", "")] = v
    d["neuropixelVersion"] = _get_neuropixel_version_from_meta(d)
    d["serial"] = _get_serial_number_from_meta(d)
    return Bunch(d)


def write_meta_data(md, md_file):
    """
    Parses a dict into a spikeglx meta data file
    :param meta: meta data dict
    :param md_file: file to save meta data to
    :return:
    """
    with open(md_file, "w") as fid:
        for key, val in md.items():
            if isinstance(val, list):
                val = ",".join([str(int(v)) for v in val])
            if isinstance(val, float):
                if val.is_integer():
                    val = int(val)
            fid.write(f"{key}={val}\n")


def _get_savedChans_subset(chns):
    """
    Get the subset of the original channels that are saved per shank
    :param chns:
    :return:
    """
    chn_grps = np.r_[0, np.where(np.diff(chns) != 1)[0] + 1, len(chns)]
    chn_subset = [
        f"{chns[chn_grps[i]]}:{chns[chn_grps[i + 1] - 1]}"
        if chn_grps[i] < len(chns) - 1
        else f"{chns[chn_grps[i]]}"
        for i in range(len(chn_grps) - 1)
    ]

    return ",".join([sub for sub in chn_subset])


def _get_serial_number_from_meta(md):
    """
    Get neuropixel serial number from the metadata dictionary
    """
    # imProbeSN for 3A, imDatPrb_sn for 3B2, None for nidq 3B2
    serial = md.get("imProbeSN") or md.get("imDatPrb_sn")
    if serial:
        return int(serial)


def _get_neuropixel_major_version_from_meta(md):
    MAJOR_VERSION = {
        "3A": 1,
        "3B2": 1,
        "3B1": 1,
        "NP2.1": 2,
        "NP2.4": 2.4,
        "NPultra": "NPultra",
    }
    version = _get_neuropixel_version_from_meta(md)
    if version is not None:
        return MAJOR_VERSION[version]


def _get_max_int_from_meta(md, neuropixel_version=None):
    """
    Gets the int value corresponding to the maximum voltage (range max)
    :param md:
    :param neuropixel_version:
    :return:
    """
    # if this is an imec probe, this is electrophysiology and we assert the imMaxInt presence in NP2
    if md.get("typeThis", None) == "imec":
        neuropixel_version = neuropixel_version or _get_neuropixel_version_from_meta(md)
        if "NP2" in neuropixel_version:
            return int(md["imMaxInt"])  # usually 8192 but could be different
        else:  # in case of NP1 it may not be in the header, but it has always been 512
            return int(md.get("imMaxInt", 512))
    else:  # this is a nidq device
        return int(md.get("imMaxInt", 32768))


def _get_neuropixel_version_from_meta(md):
    """
    Get neuropixel version tag (3A, 3B1, 3B2) from the metadata dictionary
    A useful resource is the following link:
    https://billkarsh.github.io/SpikeGLX/help/parsing/
    """
    if "typeEnabled" in md.keys():
        return "3A"
    prb_type = md.get("imDatPrb_type")
    # Neuropixel 1.0 either 3B1 or 3B2 (ask Olivier about 3B1)
    if prb_type == 0:
        if "imDatPrb_port" in md.keys() and "imDatPrb_slot" in md.keys():
            return "3B2"
        else:
            return "3B1"
    # Neuropixel 2.0 single shank
    elif prb_type == 21 or prb_type == 1030:
        return "NP2.1"
    # Neuropixel 2.0 four shank
    elif prb_type == 24 or prb_type == 2013:
        return "NP2.4"
    elif prb_type == 1100:
        return "NPultra"


def _get_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    ntr = int(_get_nchannels_from_meta(md))
    if typ == "nidq":
        nsync = int(md.get("snsMnMaXaDw")[-1])
    elif typ in ["lf", "ap"]:
        nsync = int(md.get("snsApLfSy")[2])
    return list(range(ntr - nsync, ntr))


def _get_analog_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    if typ != "nidq":
        return []
    tr = md.get("snsMnMaXaDw")
    nsa = int(tr[-2])
    return list(range(int(sum(tr[0:2])), int(sum(tr[0:2])) + nsa))


def _get_nchannels_from_meta(md):
    return int(md.get("nSavedChans"))


def _get_nshanks_from_meta(md):
    th = geometry_from_meta(md)
    return len(np.unique(th["shank"]))


def _get_fs_from_meta(md):
    if md.get("typeThis") == "imec":
        return md.get("imSampRate")
    else:
        return md.get("niSampRate")


def _get_type_from_meta(md):
    """
    Get neuropixel data type (ap, lf or nidq) from metadata
    """
    snsApLfSy = md.get("snsApLfSy", [-1, -1, -1])
    if snsApLfSy[0] == 0 and snsApLfSy[1] != 0:
        return "lf"
    elif snsApLfSy[0] != 0 and snsApLfSy[1] == 0:
        return "ap"
    elif snsApLfSy == [-1, -1, -1] and md.get("typeThis", None) == "nidq":
        return "nidq"


def _split_geometry_into_shanks(th, meta_data):
    """
    Reduces the geometry information to that pertaining to specific shank
    :param th:
    :param meta_data:
    :return:
    """
    if "NP2.4_shank" in meta_data.keys():
        shank_idx = np.where(th["shank"] == int(meta_data["NP2.4_shank"]))[0]
        th = {key: th[key][shank_idx] for key in th.keys()}

    return th


def geometry_from_meta(meta_data, return_index=False, nc=384, sort=True):
    """
    Gets the geometry, ie. the full trace header for the recording
    :param meta_data: meta_data dictionary as read by ibllib.io.spikeglx.read_meta_data
    :param return_index: (False): flag to optionally return the sorted indices
    :param sort: (True) sort the geometry by shank row col
    :param nc: number of channels if geometry is not in the metadata file
    :return: dictionary with keys 'row', 'col', 'ind', 'shank', 'adc', 'x', 'y', 'sample_shift'
    """
    cm = _map_channels_from_meta(meta_data)
    major_version = _get_neuropixel_major_version_from_meta(meta_data)
    if cm is None or all(map(lambda x: x is None, cm.values())):
        _logger.warning(
            "Meta data doesn't have geometry (snsShankMap/snsGeomMap field), returning defaults"
        )
        if major_version is None:
            if return_index:
                return None, None
            else:
                return None
        th = neuropixel.trace_header(version=major_version)
        th["flag"] = th["x"] * 0 + 1.0
        if return_index:
            return th, np.arange(nc)
        else:
            return th
    th = cm.copy()
    # as of 2023-04 spikeglx stores only x, y coordinates of sites in UM and no col / row. Here
    # we convert to col / row for consistency with previous versions
    if "x" in cm.keys():
        # the spike sorting channel maps have a flipped version of the channel map
        # there is a 20um offset between the probe tip and the first site in the coordinate conversion
        if major_version == 1:
            th["x"] = 70 - (th["x"])
        th["y"] += 20
        th.update(neuropixel.xy2rc(th["x"], th["y"], version=major_version))
    else:
        # the spike sorting channel maps have a flipped version of the channel map
        if major_version == 1:
            th["col"] = -cm["col"] * 2 + 2 + np.mod(cm["row"], 2)
        th.update(neuropixel.rc2xy(th["row"], th["col"], version=major_version))
    th["sample_shift"], th["adc"] = neuropixel.adc_shifts(
        version=major_version, nc=th["col"].size
    )
    th = _split_geometry_into_shanks(th, meta_data)
    th["ind"] = np.arange(th["col"].size)
    if sort:
        # here we sort the channels by shank, row and -col, this preserves the original NP1
        # order while still allowing to deal with creative imro tables in NP2
        sort_keys = np.c_[-th["col"], th["row"], th["shank"]]
        inds = np.lexsort(sort_keys.T)
        th = {k: v[inds] for k, v in th.items()}
    else:
        inds = np.arange(th["col"].size)
    if return_index:
        return th, inds
    else:
        return th


def read_geometry(meta_file):
    """
    Reads the geometry
    :param meta_file:
    :return: dictionary with keys 'shank', 'col', 'row', 'flag', 'x', 'y', 'sample_shift', 'adc', 'ind'
    """
    return geometry_from_meta(read_meta_data(meta_file))


def _map_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of channel positions along the shank

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: dictionary of arrays 'shank', 'col', 'row', 'flag', one value per active site
    """
    if "snsShankMap" in meta_data.keys():
        chmap = re.findall(r"([0-9]*:[0-9]*:[0-9]*:[0-9]*)", meta_data["snsShankMap"])
        key_names = {"shank": 0, "col": 1, "row": 2, "flag": 3}
    elif "snsGeomMap" in meta_data.keys():
        chmap = re.findall(r"([0-9]*:[0-9]*:[0-9]*:[0-9]*)", meta_data["snsGeomMap"])
        key_names = {"shank": 0, "x": 1, "y": 2, "flag": 3}
    else:
        return None
    # for digital nidq types, the key exists but does not contain any information
    if not chmap:
        return {"shank": None, "col": None, "row": None, "flag": None}
    # shank#, col#, row#, drawflag
    # (nb: drawflag is one should be drawn and considered spatial average)
    chmap = np.array([np.float32(cm.split(":")) for cm in chmap])
    return {k: chmap[:, v] for (k, v) in key_names.items()}


def _conversion_sample2v_from_meta(meta_data):
    """
    Interpret the meta data to extract an array of conversion factors for each channel
    so the output data is in Volts
    Conversion factor is: int2volt / channelGain
    For Lf/Ap interpret the gain string from metadata
    For Nidq, repmat the gains from the trace counts in `snsMnMaXaDw`

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: numpy array with one gain value per channel
    """

    def int2volts(md):
        """:return: Conversion scalar to Volts. Needs to be combined with channel gains"""
        maxint = _get_max_int_from_meta(md)
        if md.get("typeThis", None) == "imec":
            return md.get("imAiRangeMax") / maxint
        else:
            return md.get("niAiRangeMax") / maxint

    int2volt = int2volts(meta_data)
    version = _get_neuropixel_version_from_meta(meta_data)
    # interprets the gain value from the metadata header:
    if "imroTbl" in meta_data.keys():  # binary from the probes: ap or lf
        sy_gain = np.ones(int(meta_data["snsApLfSy"][-1]), dtype=np.float32)
        # imroTbl has 384 entries regardless of no of channels saved, so need to index by n_ch
        n_chn = _get_nchannels_from_meta(meta_data) - len(
            _get_sync_trace_indices_from_meta(meta_data)
        )
        if "NP2" in version:
            # NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            out = {
                "lf": np.hstack(
                    (int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain)
                ),
                "ap": np.hstack(
                    (int2volt / 80 * np.ones(n_chn).astype(np.float32), sy_gain)
                ),
            }
        else:
            # the sync traces are not included in the gain values, so are included for
            # broadcast ops
            gain = re.findall(
                r"([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)", meta_data["imroTbl"]
            )[:n_chn]
            out = {
                "lf": np.hstack(
                    (
                        np.array([1 / np.float32(g.split(" ")[-1]) for g in gain])
                        * int2volt,
                        sy_gain,
                    )
                ),
                "ap": np.hstack(
                    (
                        np.array([1 / np.float32(g.split(" ")[-2]) for g in gain])
                        * int2volt,
                        sy_gain,
                    )
                ),
            }

    # nidaq gain can be read in the same way regardless of NP1.0 or NP2.0
    elif "niMNGain" in meta_data.keys():  # binary from nidq
        gain = np.r_[
            np.ones(
                int(
                    meta_data["snsMnMaXaDw"][0],
                )
            )
            / meta_data["niMNGain"]
            * int2volt,
            np.ones(
                int(
                    meta_data["snsMnMaXaDw"][1],
                )
            )
            / meta_data["niMAGain"]
            * int2volt,
            np.ones(
                int(
                    meta_data["snsMnMaXaDw"][2],
                )
            )
            * int2volt,  # no gain for analog sync
            np.ones(
                int(
                    np.sum(meta_data["snsMnMaXaDw"][3]),
                )
            ),
        ]  # no unit for digital sync
        out = {"nidq": gain}

    return out


def split_sync(sync_tr):
    """
    The synchronization channels are stored as single bits, this will split the int16 original
    channel into 16 single bits channels

    :param sync_tr: numpy vector: samples of synchronisation trace
    :return: int8 numpy array of 16 channels, 1 column per sync trace
    """
    sync_tr = np.int16(np.copy(sync_tr))
    out = np.unpackbits(sync_tr.view(np.uint8)).reshape(sync_tr.size, 16)
    out = np.flip(np.roll(out, 8, axis=1), axis=1)
    return np.int8(out)


def get_neuropixel_version_from_folder(session_path):
    ephys_files = glob_ephys_files(session_path, ext="meta")
    return get_neuropixel_version_from_files(ephys_files)


def get_neuropixel_version_from_files(ephys_files):
    if any([ef.get("nidq") for ef in ephys_files]):
        return "3B"
    else:
        return "3A"


def get_probes_from_folder(session_path):
    # should glob the ephys files and get out the labels
    # This assumes the meta files exist on the server (this is the case for now but should it be?)
    ephys_files = glob_ephys_files(session_path, ext="meta")
    probes = []
    for files in ephys_files:
        if files["label"]:
            probes.append(files["label"])

    return probes


def glob_ephys_files(
    session_path, suffix=".meta", ext="bin", recursive=True, bin_exists=True
):
    """
    From an arbitrary folder (usually session folder) gets the ap and lf files and labels
    Associated to the subfolders where they are
    the expected folder tree is:
    ├── 3A
    │   ├── imec0
    │   ├── sync_testing_g0_t0.imec0.ap.bin
    │   │   └── sync_testing_g0_t0.imec0.lf.bin
    │   └── imec1
    │      ├── sync_testing_g0_t0.imec1.ap.bin
    │      └── sync_testing_g0_t0.imec1.lf.bin
    └── 3B
        ├── sync_testing_g0_t0.nidq.bin
        ├── imec0
        │   ├── sync_testing_g0_t0.imec0.ap.bin
        │   └── sync_testing_g0_t0.imec0.lf.bin
        └── imec1
            ├── sync_testing_g0_t0.imec1.ap.bin
            └── sync_testing_g0_t0.imec1.lf.bin

    :param bin_exists:
    :param suffix:
    :param ext: file extension to look for, default 'bin' but could also be 'meta' or 'ch'
    :param recursive:
    :param session_path: folder, string or pathlib.Path
    :returns: a list of dictionaries with keys 'ap': apfile, 'lf': lffile and 'label'
    """

    def get_label(raw_ephys_apfile):
        if raw_ephys_apfile.parts[-2] != "raw_ephys_data":
            return raw_ephys_apfile.parts[-2]
        else:
            return ""

    recurse = "**/" if recursive else ""
    ephys_files = []
    for raw_ephys_file in Path(session_path).glob(f"{recurse}*.ap*{suffix}"):
        raw_ephys_apfile = next(
            raw_ephys_file.parent.glob(raw_ephys_file.stem + f".*{ext}"), None
        )
        if not raw_ephys_apfile and bin_exists:
            continue
        elif not raw_ephys_apfile and ext != "bin":
            continue
        elif not bin_exists and ext == "bin":
            raw_ephys_apfile = raw_ephys_file.with_suffix(".bin")
        # first get the ap file
        ephys_files.extend(
            [Bunch({"label": None, "ap": None, "lf": None, "path": None})]
        )
        ephys_files[-1].ap = raw_ephys_apfile
        # then get the corresponding lf file if it exists
        lf_file = raw_ephys_apfile.parent / raw_ephys_apfile.name.replace(
            ".ap.", ".lf."
        )
        ephys_files[-1].lf = next(lf_file.parent.glob(lf_file.stem + f".*{ext}"), None)
        # finally, the label is the current directory except if it is bare in raw_ephys_data
        ephys_files[-1].label = get_label(raw_ephys_apfile)
        ephys_files[-1].path = raw_ephys_apfile.parent
    # for 3b probes, need also to get the nidq dataset type
    for raw_ephys_file in Path(session_path).rglob(f"{recurse}*.nidq*{suffix}"):
        raw_ephys_nidqfile = next(
            raw_ephys_file.parent.glob(raw_ephys_file.stem + f".*{ext}"), None
        )
        if not bin_exists and ext == "bin":
            raw_ephys_nidqfile = raw_ephys_file.with_suffix(".bin")
        ephys_files.extend(
            [
                Bunch(
                    {
                        "label": get_label(raw_ephys_file),
                        "nidq": raw_ephys_nidqfile,
                        "path": raw_ephys_file.parent,
                    }
                )
            ]
        )
    return ephys_files


def _mock_spikeglx_file(
    mock_bin_file,
    meta_file,
    ns,
    nc,
    sync_depth,
    random=False,
    int2volts=0.6 / 32768,
    corrupt=False,
):
    """
    For testing purposes, create a binary file with sync pulses to test reading and extraction
    """
    meta_file = Path(meta_file)
    mock_path_bin = Path(mock_bin_file)
    mock_path_meta = mock_path_bin.with_suffix(".meta")
    md = read_meta_data(meta_file)
    assert meta_file != mock_path_meta
    fs = _get_fs_from_meta(md)
    fid_source = open(meta_file)
    fid_target = open(mock_path_meta, "w+")
    line = fid_source.readline()
    while line:
        line = fid_source.readline()
        if line.startswith("fileSizeBytes"):
            line = f"fileSizeBytes={ns * nc * 2}\n"
        if line.startswith("fileTimeSecs"):
            if corrupt:
                line = f"fileTimeSecs={ns / fs + 1.8324}\n"
            else:
                line = f"fileTimeSecs={ns / fs}\n"
        fid_target.write(line)
    fid_source.close()
    fid_target.close()
    if random:
        D = np.random.randint(-32767, 32767, size=(ns, nc), dtype=np.int16)
    else:  # each channel as an int of chn + 1
        D = np.tile(np.int16((np.arange(nc) + 1) / int2volts), (ns, 1))
        D[0:16, :] = 0
    # the last channel is the sync that we fill with
    sync = np.int16(2 ** np.float32(np.arange(-1, sync_depth)))
    D[:, -1] = 0
    D[: sync.size, -1] = sync
    with open(mock_path_bin, "w+") as fid:
        D.tofile(fid)
    return {
        "bin_file": mock_path_bin,
        "ns": ns,
        "nc": nc,
        "sync_depth": sync_depth,
        "D": D,
    }


def get_hardware_config(config_file):
    """
    Reads the neuropixel_wirings.json file containing sync mapping and parameters
    :param config_file: folder or json file
    :return: dictionary or None
    """
    config_file = Path(config_file)
    if config_file.is_dir():
        config_file = list(config_file.glob("*.wiring*.json"))
        if config_file:
            config_file = config_file[0]
    if not config_file or not config_file.exists():
        return
    with open(config_file) as fid:
        par = json.loads(fid.read())
    return par


def _sync_map_from_hardware_config(hardware_config):
    """
    :param hardware_config: dictonary from json read of neuropixel_wirings.json
    :return: dictionary where key names refer to object and values to sync channel index
    """
    if hardware_config["SYSTEM"] == "3A" or hardware_config["SYSTEM"] == "3B":
        pin_out = neuropixel.SYNC_PIN_OUT[hardware_config["SYSTEM"]]
        sync_map = {
            hardware_config["SYNC_WIRING_DIGITAL"][pin]: pin_out[pin]
            for pin in hardware_config["SYNC_WIRING_DIGITAL"]
            if pin_out[pin] is not None
        }
    else:
        digital = hardware_config.get("SYNC_WIRING_DIGITAL")
        sync_map = {digital[pin]: int(pin[3:]) for pin in digital}

    analog = hardware_config.get("SYNC_WIRING_ANALOG")
    if analog:
        sync_map.update({analog[pin]: int(pin[2:]) + 16 for pin in analog})
    return sync_map


def get_sync_map(folder_ephys):
    hc = get_hardware_config(folder_ephys)
    if not hc:
        _logger.warning(f"No channel map for {str(folder_ephys)}")
        return None
    else:
        return _sync_map_from_hardware_config(hc)
