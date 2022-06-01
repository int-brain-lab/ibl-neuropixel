from typing import Any
import warnings

import numpy as np

# sample to volt conversion factors
S2V_AP = 2.34375e-06
S2V_LFP = 4.6875e-06
TIP_SIZE_UM = 200
NC = 384
SITES_COORDINATES: np.array


def _deprecated_sites_coordinates() -> np.array:
    # global SITES_COORDINATES
    # this is used in legacy code, where the reference channels are removed
    warnings.warn("the SITES_COORDINATES module attribute reflects only 374 channels and is only applicable to old"
                  "deprecated 3A probes \n Use `neuropixel.trace_header() to get the canonical probe geometries."
                  " see help(neuropixel.trace_header) \n or directly read the probe geometry"
                  " from the metadata using spigeglx.Reader")
    refch_3a = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379])
    th = trace_header(version=1)
    SITES_COORDINATES = np.delete(np.c_[th['x'], th['y']], refch_3a, axis=0)
    return SITES_COORDINATES


def __getattr__(name: str) -> Any:
    if name == "SITES_COORDINATES":
        return _deprecated_sites_coordinates()


def rc2xy(row, col, version=1):
    """
    converts the row/col indices to um coordinates.
    :param row: row index on the probe
    :param col: col index on the probe
    :param version: neuropixel major version 1 or 2
    :return: dictionary with keys x and y
    """
    if version == 1:
        x = col * 16 + 11
        y = (row * 20) + 20
    elif np.floor(version) == 2:
        x = col * 32
        y = row * 15
    return {'x': x, 'y': y}


def dense_layout(version=1, nshank=1):
    """
    Returns a dense layout indices map for neuropixel, as used at IBL
    :param version: major version number: 1 or 2 or 2.4
    :return: dictionary with keys 'ind', 'col', 'row', 'x', 'y'
    """
    ch = {'ind': np.arange(NC),
          'row': np.floor(np.arange(NC) / 2),
          'shank': np.zeros(NC)}

    if version == 1:  # version 1 has a dense layout, checkerboard pattern
        ch.update({'col': np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
    elif np.floor(version) == 2 and nshank == 1:  # single shank NP1 has 2 columns in a dense patter
        ch.update({'col': np.tile(np.array([0, 1]), int(NC / 2))})
    elif np.floor(version) == 2 and nshank == 4:  # the 4 shank version default is rather complicated
        shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
        shank_row = np.tile(shank_row, 8)
        shank_row += np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))).flatten() * 24
        ch.update({
            'col': np.tile(np.array([0, 1]), int(NC / 2)),
            'shank': np.tile(np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))).flatten(),
            'row': shank_row})
    # for all, get coordinates
    ch.update(rc2xy(ch['row'], ch['col'], version=version))
    return ch


def adc_shifts(version=1):
    """
    The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
    The ADC to channel mapping is done per odd and even channels:
    ADC1: ch1, ch3, ch5, ch7...
    ADC2: ch2, ch4, ch6....
    ADC3: ch33, ch35, ch37...
    ADC4: ch34, ch36, ch38...
    Therefore, channels 1, 2, 33, 34 get sample at the same time. I hope this is more or
    less clear. In 1.0, it is similar, but there we have 32 ADC that sample each 12 channels."
    - Nick on Slack after talking to Carolina - ;-)
    """
    if version == 1:
        adc_channels = 12
        # version 1 uses 32 ADC that sample 12 channels each
    elif np.floor(version) == 2:
        # version 2 uses 24 ADC that sample 16 channels each
        adc_channels = 16
    adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
    return sample_shift, adc


def trace_header(version=1, nshank=1):
    """
    Returns the channel map for the dense layouts used at IBL. The following pairs are commonly used:
    version=1: NP1: returns single shank dense layout with 4 columns in checkerboard pattern
    version=2, nshank=1: NP2: returns single shank dense layout with 2 columns in-line
    version=2, nshank=4: NP2: returns 4 shanks dense layout with columns in-line
    Whenever possible, it is recommended to read the geometry using `spikeglx.Reader.geometry()` method to
     ensure the channel maps corresponds the actually read data.`
    :param version: major version number: 1 or 2
    :param nshank: (defaults 1) number of shanks for NP2
    :return: , returns a dictionary with keys
    x, y, row, col, ind, adc and sampleshift vectors corresponding to each site
    """
    h = dense_layout(version=version, nshank=nshank)
    h['sample_shift'], h['adc'] = adc_shifts(version=version)
    return h
