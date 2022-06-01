import neuropixel


def test_sites_coordinates_deprecated():
    assert neuropixel.SITES_COORDINATES.shape == (374, 2)
