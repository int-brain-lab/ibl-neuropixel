# 0.6.0
## 0.6.2 2023-06-19
- add option to specify meta-data file for spikeglx.Reader

## 0.6.1 2023-06-06
- Fix bug in ADC cycles sampling for Neuropixel 1.0 probes
- 
## 0.6.0 2023-05-15
- Add waveforms utilities for spike features computations

# 0.5.0
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

# 0.4.0
## 0.4.1 2022-11-29
- CAT GT command in meta file

## 0.4.0 2022-10-28
- current source density simple double diff with denoising from raw LFP

# 0.3.0
## 0.3.2 2022-10-27
- spikeglx geometry chops the ADC sample shift to the number of channels to accomodate legacy 3A probes with 276 channels
- agc: gain refers to the inverse of applied gains for agc - done to handle dead channels in destriping
## 0.3.1
- neurodsp.utils.rises / falls: detects rising and falling edges of oversampled analog signals


## 0.3.0
- neuropixel: add functions to reconstruct original files from split NP2.4 files

## minor changes
- support for returning number of shanks from metadata

# 0.2.0
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
