import time
import numpy as np
import logging
import shutil
import unittest

from neuropixel import trace_header
import spikeglx
from neurodsp import voltage, utils

_logger = logging.getLogger(__name__)

from pathlib import Path
DATA_PATH = Path("/datadisk/Data/tests-ibl-neuropixel")


class TestEphysSpikeSortingPreProc(unittest.TestCase):

    def test_pre_proc(self):
        cbin_file = DATA_PATH.joinpath('adc', 'adc_test.ap.cbin')
        sr = spikeglx.Reader(cbin_file, open=True)
        bin_file = cbin_file.with_suffix('.bin')

        ts = time.time()
        voltage.decompress_destripe_cbin(cbin_file, nprocesses=4)
        te = time.time() - ts
        _logger.info(f'Time elapsed: {te}, length file (secs): {sr.ns / sr.fs}')
        sr_out = spikeglx.Reader(bin_file)

        assert sr.shape == sr_out.shape

        sel_comp = slice(int(65536 * 0.4), int(65536 * 1.6))
        h = trace_header(version=1)
        # create the FFT stencils
        ncv = h['x'].size  # number of channels
        expected = voltage.destripe(sr[sel_comp, :ncv].T, fs=sr.fs, channel_labels=True).T
        diff = expected - sr_out[sel_comp, :ncv]
        assert np.min(20 * np.log10(utils.rms(diff[10000:-10000, :], axis=0)
                                    / utils.rms(sr_out[sel_comp, :ncv], axis=0))) < 35
        sr_out.close()
        bin_file.unlink()


class TestEphysSpikeSortingMultiProcess(unittest.TestCase):
    def setUp(self) -> None:

        file_path = DATA_PATH.joinpath('np2', '_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        self.file_path = file_path.parent.parent.joinpath('probe00_temp', file_path.name)
        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        meta_file = file_path.parent.joinpath('NP1_meta', '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        self.meta_file = self.file_path.parent.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        shutil.copy(file_path, self.file_path)
        shutil.copy(meta_file, self.meta_file)

        self.sglx_instances = []

    def tearDown(self):
        _ = [sglx.close() for sglx in self.sglx_instances]
        # For case where we have deleted already as part of test
        if self.file_path.parent.exists():
            shutil.rmtree(self.file_path.parent)

    def test_parallel_computation(self):

        out_file = self.file_path.parent.joinpath('one_process.bin')
        shutil.copy(self.meta_file, out_file.with_suffix('.meta'))
        voltage.decompress_destripe_cbin(self.file_path, out_file, nprocesses=1, nbatch=6556)
        sr_one = spikeglx.Reader(out_file)
        self.sglx_instances.append(sr_one)

        out_file = self.file_path.parent.joinpath('four_process.bin')
        shutil.copy(self.meta_file, out_file.with_suffix('.meta'))
        voltage.decompress_destripe_cbin(self.file_path, out_file, nprocesses=4, nbatch=6556)
        sr_four = spikeglx.Reader(out_file)
        self.sglx_instances.append(sr_four)
        assert np.array_equal(sr_one[:, :], sr_four[:, :])

        # Now test the extra samples at the end
        out_file = self.file_path.parent.joinpath('four_process_extra.bin')
        shutil.copy(self.meta_file, out_file.with_suffix('.meta'))
        ns2add = 100
        voltage.decompress_destripe_cbin(self.file_path, out_file, nprocesses=4, nbatch=6556, ns2add=ns2add)
        sr_four_extra = spikeglx.Reader(out_file)
        self.sglx_instances.append(sr_four_extra)
        assert sr_four_extra.ns == sr_four.ns + ns2add
        assert np.array_equal(sr_four[:, :], sr_four_extra[:-ns2add, :])
        assert all((sr_four_extra[-ns2add:, :] == sr_four[-1, :]).ravel())

        # Now test the whitening matrix
        wm = np.identity(sr_one.nc - 1)
        out_file = self.file_path.parent.joinpath('four_process_whiten.bin')
        shutil.copy(self.meta_file, out_file.with_suffix('.meta'))
        voltage.decompress_destripe_cbin(self.file_path, out_file, nprocesses=4, nbatch=6556, wrot=wm)
        sr_four_whiten = spikeglx.Reader(out_file)
        self.sglx_instances.append(sr_four_whiten)
        assert np.array_equal(sr_four_whiten._raw[:, :-1], sr_four._raw[:, :-1])

        # Now test appending on the the end of an existing file
        out_file = self.file_path.parent.joinpath('four_process.bin')
        shutil.copy(self.meta_file, out_file.with_suffix('.meta'))
        voltage.decompress_destripe_cbin(self.file_path, out_file, nprocesses=4, nbatch=6556, append=True)
        sr_four_append = spikeglx.Reader(out_file)
        self.sglx_instances.append(sr_four_append)
        assert sr_four_append.ns == 2 * sr_four.ns
        assert np.array_equal(sr_four_append[sr_four.ns:, :], sr_four_append[:sr_four.ns, :])
        assert np.array_equal(sr_four_append[:sr_four.ns, :], sr_four[:, :])
        assert np.array_equal(sr_four_append[sr_four.ns:, :], sr_four[:, :])
