# ibl-neuropixel
Collection of tools to handle Neuropixel 1.0 and 2.0 data
(documentation coming soon...)

## Installation
`pip install ibl-neuropixel`

## Destriping
Minimal working example to destripe a neuropixel binary file. 
```
from pathlib import Path
from neurodsp.voltage import decompress_destripe_cbin
sr_file = Path('/datadisk/Data/spike_sorting/pykilosort_tests/imec_385_100s.ap.bin')
out_file = Path('/datadisk/scratch/imec_385_100s.ap.bin')

decompress_destripe_cbin(sr_file=sr_file, output_file=out_file, nprocesses=8)
```

## Contribution

Pypi Release checklist:
```shell
flake8
rm -fR dist
rm -fR build
python setup.py sdist bdist_wheel
twine upload dist/*
#twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### White Paper
The following describes the methods implemented in this repository.
https://doi.org/10.6084/m9.figshare.19705522
