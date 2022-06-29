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
