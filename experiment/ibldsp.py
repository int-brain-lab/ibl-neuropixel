from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import detect_bad_channels
from one.api import ONE
import numpy as np
import json
from pathlib import Path
import os


class TestBadChannelDetection():

    def __init__(self):
        # Get the directory where the script is located
        try:
            # Script mode
            self.filepath_save = Path(__file__).parent
        except NameError:
            # Interactive mode fallback
            self.filepath_save = Path(os.getcwd()).joinpath('ibl-neuropixel').joinpath('experiment')
            if not self.filepath_save.is_dir():
                self.filepath_save.mkdir()

        self.filename_save = 'bad_ch_annotations'
        self.annotations_path = self.filepath_save.joinpath(f"{self.filename_save}.json")

        if self.annotations_path.is_file():
            with open(self.annotations_path, "r") as f:
                self.annotations  = json.load(f)
        else:
            self.annotations = {
                '1d547041-230a-4af3-ba6a-7287de2bdec3': {'idead': [191], 'inoisy': [], 'ioutside': []},
                # eid:e5c75b62-6871-4135-b3d0-f6464c2d90c0, probe01, ('KS043', '2020-12-07', '001')
                '4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe': {'idead': [16, 29, 191, 357], 'inoisy': [],
                                                         'ioutside': [358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
                                                                      369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,
                                                                      380, 381, 382, 383]},
                'ce0dc660-f19e-46a3-94f9-646bebae6805': {'idead': [29, 36, 39, 40, 191], 'inoisy': [133, 235],
                                                         'ioutside': [274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
                                                                      285, 286,
                                                                      287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297,
                                                                      298, 299,
                                                                      300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                                                                      311, 312,
                                                                      313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
                                                                      324, 325,
                                                                      326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
                                                                      337, 338,
                                                                      339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
                                                                      350, 351,
                                                                      352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
                                                                      363, 364,
                                                                      365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
                                                                      376, 377,
                                                                      378, 379, 380, 381, 382, 383]},
                # eid:8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8, probe01, ('NYU-40', '2021-04-14', '001')
                '3b729602-20d5-4be8-a10e-24bde8fc3092': {'idead': [191], 'inoisy': [],
                                                         'ioutside': [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
                                                                      367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
                                                                      378, 379, 380, 381, 382, 383]},
                # eid:fc43390d-457e-463a-9fd4-b94a0a8b48f5, probe00, ('NYU-47', '2021-06-25', '001')
                'a9c9df46-85f3-46ad-848d-c6b8da4ae67c': {'idead': [191], 'inoisy': [],
                                                         'ioutside': [348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358,
                                                                      359, 360,
                                                                      361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
                                                                      372, 373,
                                                                      374, 375, 376, 377, 378, 379, 380, 381, 382, 383]},
                # eid:41431f53-69fd-4e3b-80ce-ea62e03bf9c7, probe01, ('CSH_ZAD_022', '2020-05-21', '001')
                '03d2d8d1-a116-4763-8425-4ef7b1c1bd35': {'idead': [131, 191, 235], 'inoisy': [], 'ioutside': []},
                # eid:81a78eac-9d36-4f90-a73a-7eb3ad7f770b, probe01, ('CSH_ZAD_026', '2020-08-17', '001')
                'fe380793-8035-414e-b000-09bfe5ece92a': {'idead': [131, 191], 'inoisy': [], 'ioutside': []},
                # eid:ff48aa1d-ef30-4903-ac34-8c41b738c1b9, probe01, ('CSH_ZAD_025', '2020-08-03', '001')
                '19c9caea-2df8-4097-92f8-0a2bad055948': {'idead': [], 'inoisy': [191], 'ioutside': []},
                # eid:aa20388b-9ea3-4506-92f1-3c2be84b85db, probe01, ('DY_016', '2020-09-14', '001')
                '25a9182c-4795-4768-af47-98975d2d2a8a': {'idead': [], 'inoisy': [191], 'ioutside': []},
                # eid:26aa51ff-968c-42e4-85c8-8ff47d19254d, probe01, ('DY_020', '2020-10-03', '001')
            }


    def save_dict(self, filepath=None, filename=None):
        if filepath is None:
            filepath = self.filepath_save
        if filename is None:
            filename = self.filename_save
        self.annotations_path = filepath.joinpath(f"{filename}.json")
        # Save to file
        with open(self.annotations_path, "w") as f:
            json.dump(self.annotations, f, indent=4)


    def test_bad_channel_detection(self):

        def detect_bad_chan(pid, one=None, t0=100, duration=1.0, display=False):
            '''
            t0: int (seconds) timepoint in recording to stream
            duration: float (seconds) duration of the snippet, typically set to 1 second of data
            '''
            if one is None:
                one = ONE()
            ssl = SpikeSortingLoader(pid=pid, one=one)
            # Get AP and LFP spikeglx.Reader objects
            sr_ap = ssl.raw_electrophysiology(band="ap", stream=True)
            s_event = int(ssl.samples2times(t0, direction='reverse'))
            # get the AP data surrounding samples
            window_secs_ap = [0, duration]
            first, last = (int(window_secs_ap[0] * sr_ap.fs) + s_event, int(window_secs_ap[1] * sr_ap.fs + s_event))
            raw_ap = sr_ap[first:last, :-sr_ap.nsync].T
            # Detect bad channels
            channel_labels, _ = detect_bad_channels(raw_ap, fs=sr_ap.fs, display=display)
            return channel_labels

        one = ONE()
        for pid in self.annotations.keys():
            channel_labels = detect_bad_chan(pid, one=one, display=False)
            idead = np.where(channel_labels == 1)[0]
            inoisy = np.where(channel_labels == 2)[0]
            ioutside = np.where(channel_labels == 3)[0]

            np.testing.assert_equal(idead, self.annotations[pid]['idead'])
            np.testing.assert_equal(inoisy, self.annotations[pid]['inoisy'])
            np.testing.assert_equal(ioutside, self.annotations[pid]['ioutside'])
