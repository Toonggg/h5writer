#!/usr/bin/env python

try:
    import h5py
    h5py_av = True
    h5py_mpi_av = h5py.h5.get_config().mpi
except ImportError:
    h5py_av = False
    h5py_mpi_av = False

if h5py_av and h5py_mpi_av:
    from setuptools import setup

    setup(name='h5writer',
          version='0.4.0',
          description='Writing HDF5 files with openMPI.',
          author='Max F. Hantke, Benedikt Daurer',
          author_email='maxhantke@gmail.com',
          url='https://github.com/mhantke/h5writer',
          #install_requires=['numpy', 'mpi4py>=2.0.0'],
          packages = ['h5writer'],
          #package_dir={'h5writer':'src'},
    )
    
else:
    print 100*"*"
    if not h5py_av:
        print "ERROR: Error cannot import h5py."
    if h5py_av and not h5py_mpi_av:
        print "ERROR: Currently installed version of h5py has no openMPI support."    
    print "\tPlease install h5py with openMPI support. For installation instructions visit for example:"
    print "\thttp://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5"
    print 100*"*"
