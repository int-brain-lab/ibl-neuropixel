import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import spikeglx
import ibldsp.sync

TEST_PATH = Path(__file__).parents[1].joinpath("fixtures")


class TestSyncTimestamps(unittest.TestCase):
    def test_removed_from_utils(self):
        """`sync_timestamps` lives in `ibldsp.sync` only: the `ibldsp.utils` alias was removed in 1.15.0."""
        import ibldsp.utils

        self.assertFalse(hasattr(ibldsp.utils, "sync_timestamps"))
        self.assertTrue(callable(ibldsp.sync.sync_timestamps))

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


class TestSpikeglxSyncQC(unittest.TestCase):
    """spikeglx_sync_qc on synthetic 1 Hz sync trains (one edge every 0.5 s)."""

    def setUp(self):
        self.t = 0.145 + np.arange(200) * 0.5  # last edge at 99.645 s
        self.md = {"fileTimeSecs": 100.0, "syncSourcePeriod": 1.0}

    def qc(self, t_probe, md_probe=None):
        return ibldsp.sync.spikeglx_sync_qc(
            self.md, md_probe or self.md, self.t, t_probe
        )

    def assert_failures(self, out, failures):
        self.assertEqual(
            {c for c in ibldsp.sync.SYNC_QC_CRITERIA if not out[c]}, set(failures)
        )
        self.assertEqual(out["passed"], len(failures) == 0)

    def test_good_probe(self):
        with self.assertNoLogs("ibllib", level="WARNING"):
            out = self.qc(self.t + 0.003)
        self.assert_failures(out, [])
        self.assertEqual(out["half_period"], 0.5)

    def test_missing_edges_at_start(self):
        # the nidq file starts one second late: its first two edges are missing, the durations differ by k = 2
        out = ibldsp.sync.spikeglx_sync_qc(
            self.md, {"fileTimeSecs": 101.0}, self.t, np.r_[0.145, 0.645, self.t + 1]
        )
        self.assert_failures(out, ["same_duration", "same_n_edges"])
        self.assertEqual((out["k"], out["dn_edges"]), (2, 2))

    def test_incomplete_start_and_end(self):
        with self.assertLogs("ibllib", level="WARNING") as log:
            self.assert_failures(self.qc(self.t[2:]), ["same_n_edges", "complete"])
        self.assertIn("missing at the file start or end", log.output[-1])
        self.assert_failures(self.qc(self.t[:-2]), ["same_n_edges", "complete"])

    def test_irregular(self):
        self.assert_failures(
            self.qc(np.delete(self.t, 100)), ["same_n_edges", "regular"]
        )
        self.assert_failures(
            self.qc(self.t + np.where(np.arange(self.t.size) == 100, 0.3, 0)),
            ["regular"],
        )

    def test_no_sync(self):
        self.assert_failures(
            self.qc(np.array([])),
            ["sync_present", "same_n_edges", "regular", "complete"],
        )
