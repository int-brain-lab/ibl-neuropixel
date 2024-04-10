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
