#!/usr/bin/env python

import logging
logging.basicConfig()

import time
import numpy
import h5py

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import h5writer
from h5writer import H5Writer, H5WriterMPISW

h5writer.logger.setLevel("WARNING")
#h5writer.logger.setLevel("INFO")
#h5writer.logger.setLevel("DEBUG")

outdir = "."

import sys
if len(sys.argv) >= 2:
    i = int(sys.argv[1])
    filename_no_mpi = "%s/test_nompi_%s.h5" % (outdir, i)
    filename_mpisw = "%s/test_mpisw_%i.h5" % (outdir, i)
else:
    filename_no_mpi = "%s/test_nompi.h5" % outdir
    filename_mpisw = "%s/test_mpisw.h5" % outdir
    

if MPI is None:
    comm = None
    rank = 0
    size = 1
    master = True
else:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    master = comm.rank == 0
    
def main():

    N_frames = 21
    N_ones = 1000
    N_zeros = 10
    
    Ws = []

    if MPI is not None and MPI.COMM_WORLD.size > 1:
        Ws.append(H5WriterMPISW(filename=filename_mpisw, comm=MPI.COMM_WORLD, chunksize=3))

    if master:
        Ws.append(H5Writer(filename_no_mpi, chunksize=2))
    
    for W in Ws:

        if isinstance(W, H5Writer) or (isinstance(W, H5WriterMPISW) and not master):
            O = {}
            grp = "parameters%i" % rank
            O[grp] = {}
            O[grp]["rank_and_size"] = [rank, size]
            W.write_solo(O)
            
            for i in range(N_frames):
                O = {}
                O["entry_1"] = {}
                O["entry_1"]["data_1"] = {}
                O["entry_1"]["data_1"]["ones"] = numpy.ones(shape=(N_ones,N_ones), dtype=numpy.float16)
                O["entry_1"]["data_2"] = {}
                O["entry_1"]["data_2"]["zeros"] = numpy.zeros(shape=(N_zeros,N_zeros))
                O["rank"] = rank
                O["size"] = size
                O["some_string"] = "bla"*i
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
            assert "rank" in f
            assert "size" in f
            assert f["/entry_1/data_1/ones"].shape[0] == N_frames
            assert f["/entry_1/data_1/ones"].shape[1] == N_ones
            assert f["/entry_1/data_1/ones"].shape[2] == N_ones
            assert f["/entry_1/data_2/zeros"].shape[0] == N_frames
            assert f["/entry_1/data_2/zeros"].shape[1] == N_zeros
            assert f["/entry_1/data_2/zeros"].shape[2] == N_zeros
            assert (numpy.array(f["/entry_1/data_1/ones"]) == 1.).all()
            assert (numpy.array(f["/entry_1/data_2/zeros"]) == 0.).all()
            #assert f["/rank"][0] == rank
            assert f["/size"][0] == size
            assert "parameters0" in f
            assert "rank_and_size" in f["/parameters0"]
            assert f["/parameters0/rank_and_size"][0] == 0
            assert f["/parameters0/rank_and_size"][1] == size
        if MPI is not None and MPI.COMM_WORLD.size > 1:
            with h5py.File(filename_mpisw, "r") as f:
                assert "entry_1" in f
                assert "data_1" in f["/entry_1"]
                assert "data_2" in f["/entry_1"]
                assert "ones" in f["/entry_1/data_1"]
                assert "zeros" in f["/entry_1/data_2"]
                assert f["/entry_1/data_1/ones"].shape[0] == N_frames*(size-1)
                assert f["/entry_1/data_1/ones"].shape[1] == N_ones
                assert f["/entry_1/data_1/ones"].shape[2] == N_ones
                assert f["/entry_1/data_2/zeros"].shape[0] == N_frames*(size-1)
                assert f["/entry_1/data_2/zeros"].shape[1] == N_zeros
                assert f["/entry_1/data_2/zeros"].shape[2] == N_zeros
                assert (numpy.array(f["/entry_1/data_1/ones"]) == 1.).all()
                assert (numpy.array(f["/entry_1/data_2/zeros"]) == 0.).all()
                for i in range(1,size):
                    assert ("parameters%i" % i) in f
                    assert "rank_and_size" in f["/parameters%i" % i]
                    assert f["/parameters%i/rank_and_size" % i][0] == i
                    assert f["/parameters%i/rank_and_size" % i][1] == size
                    #assert f["/rank"][i] == i
                    assert f["/size"][i] == size                    

    return 0
            

#if MPI.COMM_WORLD.rank == 0:
#    import cProfile
#    cProfile.run('main()')
#else:
#    main()

err = main()

if err == 0:
    if master:
        if MPI is None or MPI.COMM_WORLD.size == 1:
            print "Tested only serial version."
            print "(For testing MPI execute for example: mpirun -n 4 python test.py)"
        else:
            print "Tested both serial and MPI version."
        print "Tests successful!"
else:
    print "Failed!"
    
