#!/usr/bin/env python

from pathlib import Path
import re
from setuptools import setup

NAME_PACKAGE = "lw8"

PATH_CWD = Path()
PATH_PACKAGE = PATH_CWD / NAME_PACKAGE
PATH_VERSION = PATH_PACKAGE / "_version.py"


def get_version_tag(path_version: Path) -> str:
    verstrline = open(str(path_version), "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo is not None:
        return mo.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {str(path_version)}")


setup(
    name="h5writer",
    version=get_version_tag(PATH_VERSION),
    description="Tool for writing HDF5 files",
    author="Max F. Hantke, Benedikt Daurer, Filipe R. N. C. Maia, Jamie van der Sanden",
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

