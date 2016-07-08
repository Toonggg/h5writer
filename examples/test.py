#!/usr/bin/env python

import logging
logging.basicConfig()

import time
import numpy

from mpi4py import MPI
import h5writer
from h5writer import H5Writer, H5WriterMPISW

#h5writer.logger.setLevel("INFO")
h5writer.logger.setLevel("DEBUG")

outdir = "."

import sys
if len(sys.argv) >= 2:
    i = int(sys.argv[1])
    filename_no_mpi = "%s/test_nompi_%s.h5" % (outdir, i)
    filename_mpisw = "%s/test_mpisw_%i.h5" % (outdir, i)
else:
    filename_no_mpi = "%s/test_nompi.h5" % outdir
    filename_mpisw = "%s/test_mpisw.h5" % outdir
    
    
def main():
    
    Ws = []

    if MPI.COMM_WORLD.size > 1:
        Ws.append(H5WriterMPISW(filename_mpisw, comm=MPI.COMM_WORLD, chunksize=3))
    else:
        print "*"*100
        print "!!! WARNING: MPI COMMUNICATOR HAS SIZE 1. CANNOT PERFORM MPI TESTS WITH THIS CONFIGURATION."
        print "TRY FOR EXAMPLE:"
        print "\t $ mpirun -n 4 python test.py"
        print "*"*100

    if MPI.COMM_WORLD.rank == 0:
        Ws.append(H5Writer(filename_no_mpi))
    
    for W in Ws:
        O = {}
        O["parameters"] = {}
        O["parameters"]["foo"] = [4234,4234,5435,6354]
        W.write_solo(O)

        if MPI.COMM_WORLD.rank == 1:
            time.sleep(1)
            
        for i in range(10):
            O = {}
            O["entry_1"] = {}
            O["entry_1"]["data_1"] = {}
            O["entry_1"]["data_1"]["data"] = numpy.random.rand(10,10)
            W.write_slice(O)

        W.close()


#if MPI.COMM_WORLD.rank == 0:
#    import cProfile
#    cProfile.run('main()')
#else:
#    main()

main()
