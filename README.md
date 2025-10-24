# ibl-neuropixel
Collection of tools to handle Neuropixel 1.0 and 2.0 data
(documentation coming soon...)

## Installation
Minimum Python version supported is 3.10
`pip install ibl-neuropixel`


## Destriping
### Getting started

#### Compress a binary file losslessly using `mtscomp`

The mtscomp util implements fast chunked compression for neurophysiology data in a single shard.
Package repository is [here](https://github.com/int-brain-lab/mtscomp).


```python
from pathlib import Path
import spikeglx
file_spikeglx = Path('/datadisk/neuropixel/file.imec0.ap.bin')
sr = spikeglx.Reader(file_spikeglx)
sr.compress_file()
# note: you can use sr.compress_file(keep_original=False) to also remove the orginal bin file
```

#### Reading raw spikeglx file and manipulating arrays

The mtscomp util implements fast chunked compression for neurophysiology data in a single shard.
Package repository is [here](https://github.com/int-brain-lab/mtscomp).

```python
from pathlib import Path
import spikeglx

import ibldsp.voltage

file_spikeglx = Path('/datadisk/Data/neuropixel/human/Pt01.imec0.ap.bin')
sr = spikeglx.Reader(file_spikeglx)

# reads in 300ms of data
raw = sr[10_300_000:10_310_000, :sr.nc - sr.nsync].T
destripe = ibldsp.voltage.destripe(raw, fs=sr.fs, neuropixel_version=1)

# display with matplotlib backend
import ibldsp.plots
ibldsp.plots.voltageshow(raw, fs=sr.fs, title='raw')
ibldsp.plots.voltageshow(destripe, fs=sr.fs, title='destripe')

# display with QT backend
from viewephys.gui import viewephys
eqc = {}
eqc['raw'] = viewephys(raw, fs=sr.fs, title='raw')
eqc['destripe'] = viewephys(destripe, fs=sr.fs, title='destripe')
```

#### Destripe a binary file
This relies on a fast fourier transform external library: `pip install pyfftw`.

Minimal working example to destripe a neuropixel binary file. 
```python
from pathlib import Path
from ibldsp.voltage import decompress_destripe_cbin
sr_file = Path('/datadisk/Data/spike_sorting/pykilosort_tests/imec_385_100s.ap.bin')
out_file = Path('/datadisk/scratch/imec_385_100s.ap.bin')

decompress_destripe_cbin(sr_file=sr_file, output_file=out_file, nprocesses=8)
```

### Viewer

The best way to look at the results is to use [viewephys](https://github.com/oliche/viewephys),
open an ephys viewer on the raw data.

- tick the destripe box.
- move to a desired location in the file
- ctr+P will make the gain and axis the same on both windows

![alt text](./docs/raw_bin_viewer_destripe.png "Ephys viewer")

You can then move within the raw data file.

### White Paper
The following describes the methods implemented in this repository.
https://doi.org/10.6084/m9.figshare.19705522

## Contribution
Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to contribute to this project.
