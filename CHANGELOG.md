# Changelog

## [1.9.0] - Not released 

### added
- `ibldsp.plots.voltageshow`: displays raw data snippets for LFP / AP / CSD with a matplotlib backend
- new methods and documentation for the `ibldsp.util.WindowGenerator`: 
  - `wg.slice` returns a straight slice to index the window
  - `wg.slices_valid` returns 3 slices to index the full window, the valid window, and the valid window relative to the full window
  - `wg.splice`: splicing add a fade-in / fade-out in the overlap so that reconstruction has unit amplitude
- `ibldsp.voltage.saturation_cbin`: stand alone tool to compute the `_iblqc_ephysSaturation.samples.npy` file giving saturation intervals from a bin or cbin file

### modified
- `ibldsp.voltage.csd`: computations in SI to provide current flux in A.m-3 

### fixed
- `ibldsp.plots.show_channels_labels`: noisy channels ambiguity resolved: offending channels are displayed with
their respective features


## [1.8.1] - 2025-06-13

### fixed
- ibldsp.voltage.decompress_destripe_cbin: new option to provide a set of custom bad channel labels for the destriping 


## [1.8.0] - 2025-06-13

### changed
- ibldsp.voltage.destripe: `k_filter` argument: feeding None to the kfilter does not apply any spatial filter. It is also possible give a function to apply to the data.
- ibldsp.utils.make_channel_index: allow a dictionary as an input to compute neighbour distances 
- ibldsp.cadzow.trajectory: allows defining dtype of trajectory for time domain applications

### Added
- waveforms.get_waveforms_coordinates(): allows finding the indices of raw traces from waveform datasets

## [1.7.1] - 2025-05-22
### fixed
- remove the offset introduced to sync_timestamps that causes errors downstream due to in place assignment

## [1.7.0] - 2025-05-20

### changed
- spikeglx.decompress_to_scratch: supports a custom ch file with a different naming convention (for SDSC datasets)

### fixed
- neuropixel.NP2Converter: keeps the original NP2 AP band samples in int16 instead of doing a round-trip via float32
- ibldsp.voltage.destripe_lfp: accepts spurious arguments for back-compatibility

## [1.6.3] - 2025-01-15

### changed
- ibldsp.utils.sync_timestamps performance no longer impacted by timestamp offset


## [1.6.2] - 2025-01-06

### added
- scipy.signal.ricker has been removed and is now available at ibldsp.utils.ricker


## [1.6.1] - 2025-01-03

### fixed
- waveforms extractor returns a list of files for registration

### changed
- moved the repo contribution guide to automatic ruff formatting


## [1.6.0] - 2024-12-06

### added
- single derivative option for CSD for RIGOR metrics
- radon forward and inverse transforms
- support headerless binary files (open ephys)
-
### fixed
- cadzow iterations
- numpy 2.0 support
- destripe with channels=False

## [1.5.0] - 2024-10-24
- Automatic sorting per shank / row / col
- Minimum Python version is 3.10

## [1.4.0] - 2024-10-05
- Waveform extraction:
  - Optimization of the waveform extractor, outputs flattened waveforms
  - Refactoring ot the waveform loader with back compability
- Bad channel detector:
  - The bad channel detector has a plot option to visualize the bad channels and thresholds
  - The default low-cut filters are set to 300Hz for AP band and 2 Hz for LF band

## 1.3.2 2024-09-18

- Hotfix for WaveformsLoader label ids

## [1.3.1] 2024-09-05

- Hotfix for running tests with PyPI install

## [1.3.0] 2024-09-05

- Add support for NPultra high-density probes
- NumPy and SciPy version floors

## 1.2.1 2024-08-20
- bugfix waveform extraction: fix logic when channel labels is not None

## 1.2.0 2024-08-01
- Adds `ibldsp.waveform_extraction.WaveformsLoader`, an interface for waveforms extracted by `extract_wfs_cbin`.

## 1.1.3 2024-07-11
- Add features and tests for `extract_wfs_cbin`, including various preprocessing options.

## 1.1.2 2024-07-03
-  bugfix waveform extraction: reverting refactoring of the function to maintain compatibility with current ibllib

## 1.1.1 2024-06-07
-  Add support for NP2.0 prototype probe with probetype 1030

## 1.0.1 2024-05-29: support for waveform extraction on non-standard electrode layouts
  - bugfix waveform extraction: the probe channel layout is inferred from the spikeglx metadata by default
  - bugfix waveform extraction: the channel neighnourhood fill value is the last channel index + 1 by default instead of 384
## 1.0.0 2024-04-22
- Functionalities to check and mitigate saturation in neuropixel recordings
  - `spikeglx.Reader` has a `range_volts` method to get the saturating voltage value for a given type of probe
  - `ibldsp.voltage.saturation()` is a function that returns a boolean array indicating which samples are saturated, and a mute function
  - `ibldsp.voltage.decompress_destripe_cbin` saves a `_iblqc_ephysSaturation.samples.npy` and applies the mute function post-destriping

## 0.10.3 2024-04-18
-  Patch fixing memory leaks for `waveform_extraction` module.
## 0.10.2 2024-04-10
-  Add `waveform_extraction` module to `ibldsp`. This includes the `extract_wfs_array` and `extract_wfs_cbin` methods.
-  Add code for performing subsample shifts of waveforms.
## 0.10.1 2024-03-19
-  ensure compatibility with spikeglx 202309 metadata coordinates
## 0.10.0 2024-03-14
-  add support for online spikeglx reader

## 0.9.2 2024-02-08
-   `neurodsp` is now `ibldsp`. Drop-in replacement of the package name is all that is required to update. The `neurodsp` name will disappear on 01-Sep-2024; until then both names will work.
## 0.9.0 2024-01-17
-   `neurodsp.utils.sync_timestamps`: uses FFT based correlation to speed up large arrays alignments
-   `waveforms`: new wiggle plot for multi-trace waveforms

## 0.8.1 2023-09-21
- revert to reading channel info from private methods in shank splitting NP2.4 code to get the original channel layout from shank metadata file
## 0.8.0 2023-09-01
- add compatibility with spikeglx metadata version 2023-04 to get probe geometry

## 0.7.0 2023-06-29
- Add function `spike_venn3` in new module `neurodsp.spiketrains`
- Update `iblutil` dependency to 1.7.0 to use `iblutil.numerical.bincount2D`

## 0.6.2 2023-06-19
- add option to specify meta-data file for spikeglx.Reader

## 0.6.1 2023-06-06
- Fix bug in ADC cycles sampling for Neuropixel 1.0 probes
-
## 0.6.0 2023-05-15
- Add waveforms utilities for spike features computations

## 0.5.3 2023-04-24
- Fix for cases where channel map keys are not present

## 0.5.2 2023-04-19
- Geometry now supports both snsShankMap and snsGeomMap fields
- Option to compute LFP without splitting shanks

## 0.5.1 2023-03-07
- Enforce iblutil dependency

## 0.5.0 2023-03-06
- KCSD option for LFP current source density
- Cadzow for NP1: expose spatial parameters for LFP

## 0.4.1 2022-11-29
- CAT GT command in meta file

## 0.4.0 2022-10-28
- current source density simple double diff with denoising from raw LFP
-
## 0.3.2 2022-10-27
- spikeglx geometry chops the ADC sample shift to the number of channels to accomodate legacy 3A probes with 276 channels
- agc: gain refers to the inverse of applied gains for agc - done to handle dead channels in destriping
-
## 0.3.1
- neurodsp.utils.rises / falls: detects rising and falling edges of oversampled analog signals
- neuropixel: add functions to reconstruct original files from split NP2.4 files

## minor changes
- support for returning number of shanks from metadata

## 0.2.2
- BUGFIX change getattr of `neuropixel` to allow for stared imports

## 0.2.1
- BUGFIX constant SYNC_PIN_OUT re-introduced in `ibl-neuropixel`

## 0.2.0
- destripe_decompress_cbin: add a parameter to output QC in a different location to accomodate pykilosort scratch dir
- support for NP2.4 and NP2.1 geometries on spikeGlx

## minor changes
-   deprecated SITES_COORDINATES that would default on Neuropixel 1.0 early generations (3A probes with 374 channels)
-   spikeglx compress: pass dtype to mtscomp
-
