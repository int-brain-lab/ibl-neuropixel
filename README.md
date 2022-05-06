# ibl-neuropixel
Collection of tools to handle Neuropixel 1.0 and 2.0 data
(documentation coming soon...)

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
