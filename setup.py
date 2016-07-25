#!/usr/bin/env python

from setuptools import setup

setup(name='h5writer',
      version='0.4.9',
      description='Writing HDF5 files with openMPI.',
      author='Max F. Hantke, Benedikt Daurer',
      author_email='maxhantke@gmail.com',
      url='https://github.com/mhantke/h5writer',
      install_requires=['numpy', 'h5py', 'mpi4py>=2.0.0'],
      packages = ['h5writer'],
)

