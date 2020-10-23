#!/usr/bin/env python

from setuptools import setup

setup(
    name="h5writer",
    version="0.7.0",
    description="Tool for writing HDF5 files",
    author="Max F. Hantke, Benedikt Daurer, Filipe R. N. C. Maia",
    author_email="maxhantke@gmail.com",
    url="https://github.com/mhantke/h5writer",
    install_requires=["h5py>=2.2"],
    extras_require={"mpi": "mpi4py>=1.3.1"},
    packages=["h5writer"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    license="MIT License",
)

