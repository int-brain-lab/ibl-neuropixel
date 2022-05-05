import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setuptools.setup(
    name="ibl-neuropixel",
    version="0.1.0",
    author="The International Brain Laboratory",
    description="Collection of tools for Neuropixel 1.0 and 2.0 probes data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/int-brain-lab/ibl-neuropixel",
    project_urls={
        "Bug Tracker": "https://github.com/int-brain-lab/ibl-neuropixel/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=require,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src", exclude=['tests']),
    include_package_data=True,
    py_modules=['spikeglx', 'neuropixel'],
    python_requires=">=3.8",
)
