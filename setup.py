#!/usr/bin/env python

from setuptools import setup

setup(name='h5writer',
      version='0.1.5',
      description='Writing HDF5 files with openMPI.',
      author='Max F. Hantke, Benedikt Daurer',
      author_email='maxhantke@gmail.com',
      url='https://github.com/mhantke/h5writer',
      install_requires=['numpy', 'h5py', 'mpi4py'],
      packages = ['h5writer'],
      package_dir={'h5writer':'src'},
     )
