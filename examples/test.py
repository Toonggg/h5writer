#!/usr/bin/env python

import logging
logging.basicConfig()

import time
import numpy
import h5py

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
    

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
master = comm.rank == 0
    
def main():
    
    Ws = []

    if MPI.COMM_WORLD.size > 1:
        tmp = H5WriterMPISW(filename=filename_mpisw, comm=MPI.COMM_WORLD, chunksize=3)
        Ws.append(tmp)
    else:
        print "*"*100
        print "!!! WARNING: MPI COMMUNICATOR HAS SIZE 1. CANNOT PERFORM MPI TESTS WITH THIS CONFIGURATION."
        print "TRY FOR EXAMPLE:"
        print "\t $ mpirun -n 4 python test.py"
        print "*"*100

    if master:
        Ws.append(H5Writer(filename_no_mpi))
    
    for W in Ws:

        if isinstance(W, H5Writer) or (isinstance(W, H5WriterMPISW) and not master):
            O = {}
            grp = "parameters%i" % rank
            O[grp] = {}
            O[grp]["rank_and_size"] = [rank, size]
            W.write_solo(O)
            
            for i in range(20):
                O = {}
                O["entry_1"] = {}
                O["entry_1"]["data_1"] = {}
                O["entry_1"]["data_1"]["ones"] = numpy.ones(shape=(10,10))
                O["entry_1"]["data_2"] = {}
                O["entry_1"]["data_2"]["zeros"] = numpy.zeros(shape=(10,10))
                W.write_slice(O)

        W.close()

    # Verify correctness of written data
    if master:
        with h5py.File(filename_no_mpi, "r") as f:
            assert "entry_1" in f
            assert "data_1" in f["/entry_1"]
            assert "data_2" in f["/entry_1"]
            assert "ones" in f["/entry_1/data_1"]
            assert "zeros" in f["/entry_1/data_2"]
            assert f["/entry_1/data_1/ones"].shape[0] == 20
            assert f["/entry_1/data_1/ones"].shape[1] == 10
            assert f["/entry_1/data_1/ones"].shape[2] == 10
            assert f["/entry_1/data_2/zeros"].shape[0] == 20
            assert f["/entry_1/data_2/zeros"].shape[1] == 10
            assert f["/entry_1/data_2/zeros"].shape[2] == 10
            assert (numpy.array(f["/entry_1/data_1/ones"]) == 1.).all()
            assert (numpy.array(f["/entry_1/data_2/zeros"]) == 0.).all()
            assert "parameters0" in f
            assert "rank_and_size" in f["/parameters0"]
            assert f["/parameters0/rank_and_size"][0] == 0
            assert f["/parameters0/rank_and_size"][1] == size
        with h5py.File(filename_mpisw, "r") as f:
            assert "entry_1" in f
            assert "data_1" in f["/entry_1"]
            assert "data_2" in f["/entry_1"]
            assert "ones" in f["/entry_1/data_1"]
            assert "zeros" in f["/entry_1/data_2"]
            assert f["/entry_1/data_1/ones"].shape[0] == 20*(size-1)
            assert f["/entry_1/data_1/ones"].shape[1] == 10
            assert f["/entry_1/data_1/ones"].shape[2] == 10
            assert f["/entry_1/data_2/zeros"].shape[0] == 20*(size-1)
            assert f["/entry_1/data_2/zeros"].shape[1] == 10
            assert f["/entry_1/data_2/zeros"].shape[2] == 10
            assert (numpy.array(f["/entry_1/data_1/ones"]) == 1.).all()
            assert (numpy.array(f["/entry_1/data_2/zeros"]) == 0.).all()
            for i in range(1,size):
                assert ("parameters%i" % i) in f
                assert "rank_and_size" in f["/parameters%i" % i]
                assert f["/parameters%i/rank_and_size" % i][0] == i
                assert f["/parameters%i/rank_and_size" % i][1] == size
            

#if MPI.COMM_WORLD.rank == 0:
#    import cProfile
#    cProfile.run('main()')
#else:
#    main()

main()
