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
    O["parameters"]["foo"] = [1,2,3,4]
    O["parameters"]["rank_min"] = MPI.COMM_WORLD.rank
    O["parameters"]["rank_max"] = MPI.COMM_WORLD.rank
    W.write_solo(O)

    for i in range(10):
        O = {}
        O["entry_1"] = {}
        O["entry_1"]["data_1"] = {}
        O["entry_1"]["data_1"]["data"] = numpy.random.rand(10,10)
        W.write_slice(O)

    W.close()
