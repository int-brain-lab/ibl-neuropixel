import datetime
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import spikeglx
import ibldsp.sync

TEST_PATH = Path(__file__).parents[1].joinpath("fixtures")


class TestSyncTimestamps(unittest.TestCase):
    def test_deprecation(self):
        if datetime.datetime.now() > datetime.datetime(2026, 10, 12):
            raise NotImplementedError(
                "Time to deprecate ibldsp.utils.sync_timestamps()"
            )

    def test_sync_timestamps_linear(self):
        ta = np.cumsum(np.abs(np.random.randn(100))) * 10
        tb = ta * 1.0001 + 100
        fcn, drif, ia, ib = ibldsp.sync.sync_timestamps(
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
        _fcn, _drift = ibldsp.sync.sync_timestamps(tsa, tsb)
        assert np.all(np.isclose(_fcn(tsa), tsb))
        assert np.isclose(drift, _drift)

        # test missing indices on a
        imiss = np.setxor1d(np.arange(n), [1, 2, 34, 35])
        _fcn, _drift, _ia, _ib = ibldsp.sync.sync_timestamps(
            tsa[imiss], tsb, return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[_ib]))

        # test missing indices on b
        _fcn, _drift, _ia, _ib = ibldsp.sync.sync_timestamps(
            tsa, tsb[imiss], return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[_ia]), tsb[imiss[_ib]]))

        # test missing indices on both
        imiss2 = np.setxor1d(np.arange(n), [14, 17])
        _fcn, _drift, _ia, _ib = ibldsp.sync.sync_timestamps(
            tsa[imiss], tsb[imiss2], return_indices=True
        )
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[imiss2[_ib]]))

        # test timestamps with huge offset (previously caused ArrayMemoryError)
        # tsb -= 1e15
        # _fcn, _drift = utils.sync_timestamps(tsa, tsb)
        # assert np.all(np.isclose(_fcn(tsa), tsb))


class TestSyncSpikeGlx:
    # def setUp(self):
    #     self.workdir = Path(__file__).parents[1] / 'fixtures' / 'sync_ephys_fpga'
    #     self.meta_files = list(Path.glob(self.workdir, '*.meta'))

    def test_sync_nidq(self):
        self.sync_gen(fn="sample3B_g0_t0.nidq.meta", ns=32, nc=2, sync_depth=8)

    def test_sync_NP1(self):
        self.sync_gen(fn="sample3B_g0_t0.imec1.ap.meta", ns=32, nc=385, sync_depth=16)

    def sync_gen(self, fn, ns, nc, sync_depth):
        # nidq has 1 analog and 1 digital sync channels
        with tempfile.TemporaryDirectory() as tdir:
            ses_path = Path(tdir).joinpath("raw_ephys_data")
            ses_path.mkdir(parents=True, exist_ok=True)
            meta_file = ses_path.joinpath(fn)
            bin_file = meta_file.with_suffix(".bin")
            import shutil

            shutil.copy(TEST_PATH.joinpath(fn), meta_file)
            _ = spikeglx._mock_spikeglx_file(
                bin_file,
                meta_file=TEST_PATH.joinpath(fn),
                ns=ns,
                nc=nc,
                sync_depth=sync_depth,
            )
            sr = spikeglx.Reader(bin_file)
            # for a nidq file, there can be additional analog sync channels shown in the sync
            csel = spikeglx._get_analog_sync_trace_indices_from_meta(sr.meta)
            df_sync = pd.DataFrame(ibldsp.sync.extract_spikeglx_sync(bin_file))
            np.testing.assert_equal(
                len(df_sync["channels"].unique()), sync_depth + len(csel)
            )
