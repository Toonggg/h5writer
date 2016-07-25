#!/usr/bin/env python

from setuptools import setup

setup(name='h5writer',
      version='0.5.0',
      description='Tool for writing HDF5 files',
      author='Max F. Hantke, Benedikt Daurer, Filipe R. N. C. Maia',
      author_email='maxhantke@gmail.com',
      url='https://github.com/mhantke/h5writer',
      install_requires=['h5py>=1.2.1'],
      extra_requires=['mpi4py>=1.0'],
      packages = ['h5writer'],
)

