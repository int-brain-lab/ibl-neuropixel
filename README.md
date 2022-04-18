# ibl-neuropixel
Collection of tools to handle Neuropixel 1.0 and 2.0 data


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

### Main contributors
Olivier Winter, Kush Banga, Mayo Faulkner
