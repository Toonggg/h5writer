#!/usr/bin/env python

import logging
logging.basicConfig()

import numpy

from mpi4py import MPI
import h5writer
from h5writer import H5WriterMPI, H5Writer

h5writer.logger.setLevel("INFO")

Ws = [H5Writer("./test_nompi_%i.h5" % MPI.COMM_WORLD.rank),
      H5WriterMPI("./test_mpi.h5", comm=MPI.COMM_WORLD)]


for W in Ws:
    O = {}
    O["parameters"] = {}
    O["parameters"]["foo"] = [4234,4234,5435,6354]
    W.write_solo(O)

    if isinstance(W, H5WriterMPI):
        O = {}
        O["parameters"] = {}
        O["parameters"]["rank_min"] = MPI.COMM_WORLD.rank
        W.write_solo_mpi_reduce(O, MPI.MIN)
        O = {}
        O["parameters"] = {}       
        O["parameters"]["rank_max"] = MPI.COMM_WORLD.rank
        W.write_solo_mpi_reduce(O, MPI.MAX)
        O = {}
        O["parameters"] = {}       
        O["parameters"]["rank_mean"] = MPI.COMM_WORLD.rank / float(MPI.COMM_WORLD.size)
        W.write_solo_mpi_reduce(O, MPI.SUM)
        O = {}
        O["parameters"] = {}
        O["parameters"]["rank_index"] = numpy.zeros(MPI.COMM_WORLD.size)
        O["parameters"]["rank_index"][MPI.COMM_WORLD.rank] = MPI.COMM_WORLD.rank
        W.write_solo_mpi_reduce(O, MPI.SUM)
        
    for i in range(10):
        O = {}
        O["entry_1"] = {}
        O["entry_1"]["data_1"] = {}
        O["entry_1"]["data_1"]["data"] = numpy.random.rand(10,10)
        W.write_slice(O)

    W.close()
