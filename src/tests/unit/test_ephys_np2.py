import random
import shutil
import unittest
import tempfile
from pathlib import Path

import numpy as np
from neuropixel import NP2Converter, NP2Reconstructor
import spikeglx

FIXTURE_PATH = Path(__file__).parents[1].joinpath("fixtures", "np2split")


class BaseEphysNP2(unittest.TestCase):
    nc = None
    folder_test_case = None

    @classmethod
    def setUpClass(cls):
        # create a temporary directory for the test
        cls._temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_np2_")
        cls.data_path = Path(cls._temp_dir_obj.name)
        # we create a small dummy spikeglx file for testing
        cls.orig_file = cls.data_path.joinpath("_spikeglx_ephysData_g0_t0.imec0.ap.bin")
        # the metadata file is copied from fixtures
        cls.orig_meta_file = cls.orig_file.with_suffix(".meta")
        shutil.copy(
            FIXTURE_PATH.joinpath(
                cls.folder_test_case, "_spikeglx_ephysData_g0_t0.imec0.ap.meta"
            ),
            cls.orig_meta_file,
        )
        # generation of a dummy binary file for testing
        dat = np.tile(np.arange(cls.nc)[np.newaxis, :] + 10000, [30000, 1]).astype(
            np.int16
        )
        with open(cls.orig_file, "bw+") as fid:
            dat.tofile(fid)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir_obj.cleanup()

    def setUp(self):
        """
        :param folder_test_case: NP1_meta | NP21_meta | NP24_meta
        :return:
        """
        self._temp_dir_obj_case = tempfile.TemporaryDirectory(prefix="test_np2_case_")
        current_case_path = Path(self._temp_dir_obj_case.name).joinpath("probe00")
        current_case_path.mkdir(parents=True, exist_ok=True)
        self.file_path = current_case_path.joinpath(self.orig_file.name)
        self.meta_file = self.file_path.with_suffix(".meta")
        shutil.copy(self.orig_file, self.file_path)
        shutil.copy(self.orig_meta_file, self.meta_file)
        self.sglx_instances = []

    def tearDown(self):
        self._temp_dir_obj_case.cleanup()
        _ = [sglx.close() for sglx in self.sglx_instances]


class TestNeuropixel2ConverterNP24(BaseEphysNP2):
    """
    Check NP2 converter with NP2.4 type probes
    """

    nc = 385
    folder_test_case = "NP24_meta"

    def testDecimate(self):
        """
        Check integrity of windowing and downsampling by comparing results when using different
        window lengths for iterating through data
        :return:
        """
        FS = 30000
        np_a = NP2Converter(self.file_path, post_check=False, compress=False)
        np_a.init_params(nwindow=0.3 * FS, extra="_0_5s_test", nshank=[0])
        np_a.process()

        np_b = NP2Converter(self.file_path, post_check=False, compress=False)
        np_b.init_params(nwindow=0.5 * FS, extra="_1s_test", nshank=[0])
        np_b.process()

        np_c = NP2Converter(self.file_path, post_check=False, compress=False)
        np_c.init_params(nwindow=1 * FS, extra="_2s_test", nshank=[0])
        np_c.process()

        sr = spikeglx.Reader(self.file_path, sort=False)
        self.sglx_instances.append(sr)
        sr_a_ap = spikeglx.Reader(np_a.shank_info["shank0"]["ap_file"], sort=False)
        self.sglx_instances.append(sr_a_ap)
        sr_b_ap = spikeglx.Reader(np_b.shank_info["shank0"]["ap_file"], sort=False)
        self.sglx_instances.append(sr_b_ap)
        sr_c_ap = spikeglx.Reader(np_c.shank_info["shank0"]["ap_file"], sort=False)
        self.sglx_instances.append(sr_c_ap)

        # Make sure all the aps are the same regardless of window size we used
        np.testing.assert_array_equal(sr_a_ap[:, :], sr_b_ap[:, :])
        np.testing.assert_array_equal(sr_a_ap[:, :], sr_c_ap[:, :])
        np.testing.assert_array_equal(sr_b_ap[:, :], sr_c_ap[:, :])

        # For AP also check that all values are the same as the original file
        # sr_a_ap[0, :-1].T / 7.6293946e-07
        # sr[0, np_a.shank_info['shank0']['chns'][:-1]] / 7.6293946e-07
        # sr[:, np_a.shank_info['shank0']['chns']]
        np.testing.assert_array_equal(
            sr_a_ap[:, :], sr[:, np_a.shank_info["shank0"]["chns"]]
        )
        np.testing.assert_array_equal(
            sr_b_ap[:, :], sr[:, np_b.shank_info["shank0"]["chns"]]
        )
        np.testing.assert_array_equal(
            sr_c_ap[:, :], sr[:, np_c.shank_info["shank0"]["chns"]]
        )

        sr_a_lf = spikeglx.Reader(np_a.shank_info["shank0"]["lf_file"], sort=False)
        self.sglx_instances.append(sr_a_lf)
        sr_b_lf = spikeglx.Reader(np_b.shank_info["shank0"]["lf_file"], sort=False)
        self.sglx_instances.append(sr_b_lf)
        sr_c_lf = spikeglx.Reader(np_c.shank_info["shank0"]["lf_file"], sort=False)
        self.sglx_instances.append(sr_c_lf)

        # Make sure all the lfps are the same regardless of window size we used
        np.testing.assert_array_equal(sr_a_lf[:, :], sr_b_lf[:, :])
        np.testing.assert_array_equal(sr_a_lf[:, :], sr_c_lf[:, :])
        np.testing.assert_array_equal(sr_b_lf[:, :], sr_c_lf[:, :])

    def testProcessNP24(self):
        """
        Check normal workflow of splittig data into individual shanks
        :return:
        """
        # Make sure normal workflow runs without problems
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Test a random ap metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_ap = spikeglx.Reader(
            np_conv.shank_info[f"shank{shank_n}"]["ap_file"], sort=False
        )
        np.testing.assert_array_equal(sr_ap.meta["acqApLfSy"], [96, 0, 1])
        np.testing.assert_array_equal(sr_ap.meta["snsApLfSy"], [96, 0, 1])
        self.assertEqual(sr_ap.meta["nSavedChans"], 97)
        self.assertEqual(sr_ap.meta["snsSaveChanSubset"], "0:96")
        self.assertEqual(sr_ap.meta["NP2.4_shank"], shank_n)
        self.assertEqual(sr_ap.meta["original_meta"], "False")
        sr_ap.close()

        # Test a random lf metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_lf = spikeglx.Reader(
            np_conv.shank_info[f"shank{shank_n}"]["lf_file"], sort=False
        )
        np.testing.assert_array_equal(sr_lf.meta["acqApLfSy"], [0, 96, 1])
        np.testing.assert_array_equal(sr_lf.meta["snsApLfSy"], [0, 96, 1])
        self.assertEqual(sr_lf.meta["nSavedChans"], 97)
        self.assertEqual(sr_lf.meta["snsSaveChanSubset"], "0:96")
        self.assertEqual(sr_lf.meta["NP2.4_shank"], shank_n)
        self.assertEqual(sr_lf.meta["original_meta"], "False")
        self.assertEqual(sr_lf.meta["imSampRate"], 2500)
        sr_lf.close()

        # Rerun again and make sure that nothing happens because folders already exists
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertTrue(np_conv.already_exists)
        self.assertFalse(status)

        # But if we set the overwrite flag to True we force rerunning
        # here we also test deleting of the original folder
        np_conv = NP2Converter(self.file_path, delete_original=True)
        np_conv.init_params(extra="_test")
        status = np_conv.process(overwrite=True)
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)
        np_conv.sr.close()

        # test that original has been deleted
        self.assertFalse(self.file_path.exists())

        # Finally test that we cannot process a file that has already been split
        shank_n = random.randint(0, 3)
        ap_file = np_conv.shank_info[f"shank{shank_n}"]["ap_file"]
        np_conv = NP2Converter(ap_file)
        status = np_conv.process()
        self.assertTrue(np_conv.already_processed)
        self.assertFalse(status)

        np_conv.sr.close()

    def testIncorrectSplitting(self):
        """
        Check that if the data has been incorrectly split we get a warning error
        :return:
        """
        np_conv = NP2Converter(self.file_path, compress=False)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Change some of the data and make sure the checking function is working as expected
        ap_file = np_conv.shank_info["shank0"]["ap_file"]
        with open(ap_file, "r+b") as f:
            f.write((chr(10) + chr(20) + chr(30) + chr(40)).encode())

        # Now that we have changed the file we expect an assertion error when we do the check
        with self.assertRaises(AssertionError) as context:
            np_conv.check_NP24()
        self.assertTrue("Arrays are not equal" in str(context.exception))

    def testFromCompressed(self):
        """
        Check that processing works even if ap file has already been compressed
        :return:
        """
        sr_ap = spikeglx.Reader(self.file_path)
        cbin_file = sr_ap.compress_file(check_after_compress=False)
        sr_ap.close()
        self.file_path.unlink()

        np_conv = NP2Converter(cbin_file)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)
        np_conv.sr.close()


class TestNeuropixel2ConverterNP21(BaseEphysNP2):
    """
    Check NP2 converter with NP2.1 type probes
    """

    nc = 385
    folder_test_case = "NP21_meta"

    def testProcessNP21(self):
        """
        Check normal workflow of getting LFP data out and storing in main probe folder
        :return:
        """

        # make sure it runs smoothly
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # test the meta file
        sr_ap = spikeglx.Reader(np_conv.shank_info["shank0"]["lf_file"])
        np.testing.assert_array_equal(sr_ap.meta["acqApLfSy"], [0, 384, 1])
        np.testing.assert_array_equal(sr_ap.meta["snsApLfSy"], [0, 384, 1])
        self.assertEqual(sr_ap.meta["nSavedChans"], 385)
        self.assertEqual(sr_ap.meta["snsSaveChanSubset"], "0:384")
        self.assertEqual(sr_ap.meta["NP2.1_shank"], 0)
        self.assertEqual(sr_ap.meta["original_meta"], "False")
        sr_ap.close()

        np_conv.sr.close()

        # now run again and make sure that it doesn't run
        np_conv = NP2Converter(self.file_path.with_suffix(".cbin"))
        status = np_conv.process()
        self.assertTrue(np_conv.already_exists)
        self.assertFalse(status)
        np_conv.sr.close()

        # Now try with the overwrite flag and make sure it runs, this also tests running from
        # a compressed file
        np_conv = NP2Converter(self.file_path.with_suffix(".cbin"))
        status = np_conv.process(overwrite=True)
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)
        np_conv.sr.close()


class TestNeuropixel2ConverterNP1(BaseEphysNP2):
    """
    Check NP2 converter with NP1 type probes
    """

    nc = 385
    folder_test_case = "NP1_meta"

    def testProcessNP1(self):
        """
        Check normal workflow -> nothing should happen!
        """
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process()
        self.assertEqual(status, -1)


class TestNeuropixelReconstructor(BaseEphysNP2):
    nc = 385
    folder_test_case = "NP24_meta"

    def test_reconstruction(self):
        # First split the probes
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertTrue(status)
        np_conv.sr.close()

        # Delete the original file
        self.file_path.unlink()
        self.meta_file.unlink()

        # Now reconstruct
        np_recon = NP2Reconstructor(
            self.file_path.parents[1], pname="probe00", compress=True
        )
        status = np_recon.process()

        self.assertEqual(status, 1)

        recon_file = self.file_path.with_suffix(".cbin")
        sr_recon = spikeglx.Reader(recon_file, sort=False)
        self.sglx_instances.append(sr_recon)
        sr_orig = spikeglx.Reader(self.orig_file, sort=False)
        self.sglx_instances.append(sr_orig)

        np.testing.assert_array_equal(sr_recon._raw[:, :], sr_orig._raw[:, :])

        orig_meta = spikeglx.read_meta_data(self.orig_meta_file)
        recon_meta = spikeglx.read_meta_data(recon_file.with_suffix(".meta"))
        recon_meta.pop("original_meta")

        self.assertEqual(orig_meta, recon_meta)


class TestNeuropixel2ConverterNP2QB(BaseEphysNP2):
    nc = 385 * 4
    folder_test_case = "NP2QB_meta"

    def testProcessNP2QB(self):
        # Make sure normal workflow runs without problems
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra="_test")
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Test a random ap metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_ap = spikeglx.Reader(
            np_conv.shank_info[f"shank{shank_n}"]["ap_file"], sort=False
        )
        np.testing.assert_array_equal(sr_ap.meta["acqApLfSy"], [384, 0, 1])
        np.testing.assert_array_equal(sr_ap.meta["snsApLfSy"], [384, 0, 1])
        self.assertEqual(sr_ap.meta["nSavedChans"], 385)
        self.assertEqual(sr_ap.meta["snsSaveChanSubset"], "0:384")
        self.assertEqual(sr_ap.meta["NP2.4_shank"], shank_n)
        self.assertEqual(sr_ap.meta["original_meta"], "False")
        sr_ap.close()
