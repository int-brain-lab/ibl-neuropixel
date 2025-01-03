import neuropixel
import numpy as np


def test_sites_coordinates_deprecated():
    assert neuropixel.SITES_COORDINATES.shape == (374, 2)


def test_adc_shifts():
    # test ADC shifts version 1
    h1 = neuropixel.trace_header(version=1)
    np.testing.assert_equal(np.unique(h1["sample_shift"] * 13), np.arange(12))
    # test ADC shifts version 2
    h21 = neuropixel.trace_header(version=2.1)
    h24 = neuropixel.trace_header(version=2.4)
    np.testing.assert_equal(h24["sample_shift"], h21["sample_shift"])
    np.testing.assert_equal(np.unique(h21["sample_shift"] * 16), np.arange(16))
    # test ADC shifts uhd
    hUHD = neuropixel.trace_header(version="NPultra")
    np.testing.assert_equal(hUHD["sample_shift"], h1["sample_shift"])


def test_geom_np1():
    gt = dict(
        ind=np.arange(384),
        shank=np.zeros(384),
        row=np.repeat(np.arange(192), 2),
        col=np.tile(np.array([2, 0, 3, 1]), 96),
        x=np.tile(np.array([43, 11, 59, 27]), 96),
        y=np.repeat(np.arange(0, 3840, 20), 2) + 20,
    )

    h = neuropixel.trace_header(1)
    for k, v in gt.items():
        np.testing.assert_equal(v, h[k])


def test_geom_np2_1shank():
    gt = dict(
        ind=np.arange(384),
        shank=np.zeros(384),
        row=np.repeat(np.arange(192), 2),
        col=np.tile(np.array([0, 1]), 192),
        x=np.tile(np.array([27, 59]), 192),
        y=np.repeat(np.arange(0, 2880, 15), 2) + 20,
    )

    h = neuropixel.trace_header(2, 1)
    for k, v in gt.items():
        np.testing.assert_equal(v, h[k])


def test_geom_np2_4shank():
    depth_blocks = np.vstack(
        [np.repeat(np.arange(24), 2), np.repeat(np.arange(24, 48), 2)]
    )
    row_ind = np.concatenate([depth_blocks[i] for i in [0, 0, 1, 1, 0, 0, 1, 1]])
    gt = dict(
        ind=np.arange(384),
        shank=np.repeat(np.array([0, 1, 0, 1, 2, 3, 2, 3]), 48),
        row=row_ind,
        col=np.tile(np.array([0, 1]), 192),
        x=np.tile(np.array([27, 59]), 192),
        y=row_ind * 15 + 20,
    )

    h = neuropixel.trace_header(2, 4)
    for k, v in gt.items():
        np.testing.assert_equal(v, h[k])


def test_geom_npultra():
    gt = dict(
        ind=np.arange(384),
        shank=np.zeros(384),
        row=np.repeat(np.arange(48), 8),
        col=np.tile(np.arange(8), 48),
        x=np.tile(np.arange(0, 48, 6), 48),
        y=np.repeat(np.arange(0, 288, 6), 8),
    )

    h = neuropixel.trace_header("NPultra")
    for k, v in gt.items():
        np.testing.assert_equal(v, h[k])
