#!/usr/bin/env python

from warnings import warn

try:
    import h5py
    h5py_av = True
    try:
        h5py_mpi_av = h5py.h5.get_config().mpi
    except AttributeError:
        h5py_mpi_av = False
except ImportError:
    h5py_av = False
    h5py_mpi_av = False
        
if h5py_av:
        
    from setuptools import setup

    setup(name='h5writer',
          version='0.4.7',
          description='Writing HDF5 files with openMPI.',
          author='Max F. Hantke, Benedikt Daurer',
          author_email='maxhantke@gmail.com',
          url='https://github.com/mhantke/h5writer',
          #install_requires=['numpy', 'h5py', 'mpi4py>=2.0.0'],
          packages = ['h5writer'],
          #package_dir={'h5writer':'src'},
    )

    if not h5py_mpi_av:
        print "WARNING: Currently installed version of h5py has not support for parallalisation. Certain features of h5writer will not be available."
        print "\tFor installation instructions for parallel h5py visit for example:"
        print "\thttp://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5"

        
    
else:
    
    print 100*"*"

    print "\th5py cannot be found! Please install h5py."
    print "\tFor installation instructions for parallel h5py visit for example:"
    print "\thttp://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5"

    print 100*"*"

