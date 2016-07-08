import sys
import numpy, os, time
import h5py

from log import log_and_raise_error, log_warning, log_info, log_debug

try:
    from mpi4py import MPI
except:
    log_warning(logger, "Cannot import mpi4py!")

MPI_TAG_INIT   = 1
MPI_TAG_EXPAND = 2
MPI_TAG_READY  = 3
MPI_TAG_CLOSE  = 4

from h5writer import AbstractH5Writer,logger

class H5WriterMPIPW(AbstractH5Writer):
    """
    HDF5 writer class for MPI and parallel writing.
    """
    def __init__(self, filename, comm, chunksize=100, compression=None):
        # Initialisation of base class
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        # MPI communicator
        self.comm = comm
        # This "if" avoids that processes that are not in the communicator (like the master process of hummingbird) interact with the file and block
        if not self._is_in_communicator():
            log_warning(logger, "This process cannot write.")
            return
        # Logging
        self._log_prefix = "(%03i) " %  self.comm.rank
        # Index
        self._i = -self.comm.size + self.comm.rank
        self._i_max = -1
        # Expansion flag
        self._expand_flag = False
        # Chache
        self._solocache = {}
        # Status
        self._ready = False
        # Open file
        if os.path.exists(self._filename):
            log_warning(logger, self._log_prefix + "File %s exists and is being overwritten" % (self._filename))
        self.comm.Barrier()
        self._f = h5py.File(self._filename, "w", driver='mpio', comm=self.comm)
        self._f.atomic = True
        self.comm.Barrier()
        log_debug(logger, self._log_prefix + "File successfully opened for parallel writing.")
        
    def write_slice(self, data_dict):
        """
        Call this function for writing all data in data_dict as a stack of slices (first dimension = stack dimension).
        Dictionaries within data_dict are represented as HDF5 groups. The slice index is either the next one.
        """

        # Initialise of tree (groups and datasets)
        if not self._initialised:
            self.comm.Barrier()
            self._initialise_tree(data_dict)
            if self.comm.size > self._stack_length:
                self.comm.Barrier()
                self._expand_stacks_mpi(i_max=self.comm.size-1)  
            self._initialised = True
            self.comm.Barrier()
            log_debug(logger, self._log_prefix + "Tree initialised")
        self._i += self.comm.size
        # Update of maximum index
        if self._i > self._i_max:
            self._i_max = self._i
        # Enter poll for other processes that might need stack expansion?
        self._expand_poll()
        log_debug(logger, self._log_prefix + "Write")
        # Write data
        self._write_group(data_dict)
        log_debug(logger, self._log_prefix + "Written")
        sys.stdout.flush()

    def write_solo(self, data_dict):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        """
        if self._is_in_communicator():
            return self._to_solocache(data_dict, target=self._solocache)
        
    def write_solo_mpi_reduce(self, data_dict, op):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        Data will be reduced between processes using the given open MPI operator (see for example https://pythonhosted.org/mpi4py/apiref/mpi4py-module.html).
        """
        if self._is_in_communicator():
            return self._to_solocache(data_dict, target=self._solocache, op=op)

    def close(self):
        """
        Close file.
        """
        # This "if" avoids that processes that are not in the communicator (like the master process of hummingbird) interact with the file and block
        if not self._is_in_communicator():
            return

        if not self._initialised:
            log_and_raise_error(logger, "Cannot close uninitialised file. Every worker has to write at least one frame to file. Reduce your number of workers and try again.")
            exit(1)
        self._close_signal()
        log_debug(logger, self._log_prefix + "Enter closing loop")
        while True:
            self._expand_poll()
            self._update_ready()
            if self._ready:
                break
            time.sleep(0.1)
        log_debug(logger, self._log_prefix + "Leave closing loop")

        self.comm.Barrier()
        log_debug(logger, self._log_prefix + "Sync stack length")
        self._sync_i_max()

        self.comm.Barrier()
        log_debug(logger, self._log_prefix + "Closing file %s for parallel writing." % (self._filename))
        self._f.close()
        log_debug(logger, self._log_prefix + "File %s closed for parallel writing." % (self._filename))

        if self._is_master():
            log_debug(logger, self._log_prefix + "Opening file %s for single-process writing." % (self._filename))
            self._f = h5py.File(self._filename, "r+")
            log_debug(logger, self._log_prefix + "Shrink stacks")
            self._resize_stacks(self._i_max + 1)
            
        self._write_solocache_group_to_file(self._solocache)

        if self._is_master():
            log_debug(logger, self._log_prefix + "Closing file %s for single-process writing." % (self._filename))
            self._f.close()
            log_info(logger, self._log_prefix + "HDF5 parallel writer instance for file %s closed." % (self._filename))
    
    def _is_in_communicator(self):
        try:
            out = self.comm.rank != MPI.UNDEFINED
        except MPI.Exception:
            out = False
        return out
            
    def _is_master(self):
        return (self._is_in_communicator() and self.comm.rank == 0)
        
    def _to_solocache(self, data_dict, target, op=None):
        keys = data_dict.keys()
        keys.sort()
        for k in keys:
            if isinstance(data_dict[k], dict):
                if k not in target:
                    target[k] = {}
                self._to_solocache(data_dict[k], target=target[k], op=op)
            else:
                target[k] = (data_dict[k], op)
            
    def _write_solocache_group_to_file(self, data_dict, group_prefix="/"):
        if self._is_master() and group_prefix != "/" and group_prefix not in self._f:
            self._f.create_group(group_prefix)
        keys = data_dict.keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(data_dict[k], dict):
                self._write_solocache_group_to_file(data_dict[k], group_prefix=name+"/")
            else:
                (data, op) = data_dict[k]
                if op is not None:
                    if numpy.isscalar(data):
                        sendobj = numpy.array(data)
                    else:
                        sendobj = data
                    recvobj = numpy.empty_like(data)
                    log_debug(logger, self._log_prefix + "Reducing data %s" % (name))
                    self.comm.Reduce(
                        [sendobj, MPI.DOUBLE],
                        [recvobj, MPI.DOUBLE],
                        op = op,
                        root = 0
                    )
                    data = recvobj
                if self._is_master():
                    log_debug(logger, self._log_prefix + "Writing data %s" % (name))
                    self._f[name] = data
        
    def _expand_poll(self):
        #log_debug(logger, self._log_prefix + "Polling for stack expansion")
        expand = False
        if self.comm.Iprobe(source=(self.comm.rank-1) % self.comm.size, tag=MPI_TAG_EXPAND):
            buf1 = numpy.empty(1, dtype='i')
            self.comm.Recv([buf1, MPI.INT], source=(self.comm.rank-1) %  self.comm.size, tag=MPI_TAG_EXPAND)
            buf2 = numpy.array(1, dtype='i')
            self.comm.Send([buf2, MPI.INT], dest=(self.comm.rank+1) %  self.comm.size, tag=MPI_TAG_EXPAND)
            expand= True
        # Sending of expansion signal needed?
        elif self._i >= self._stack_length:
            buf1 = numpy.array(1, dtype='i')
            req_s = self.comm.Isend([buf1, MPI.INT], dest=(self.comm.rank+1) %  self.comm.size, tag=MPI_TAG_EXPAND)
            buf2 = numpy.empty(1, dtype='i')
            req_r = self.comm.Irecv([buf2, MPI.INT], source=(self.comm.rank-1) %  self.comm.size, tag=MPI_TAG_EXPAND)
            while True:
                sent     = req_s.Test()
                received = req_r.Test()
                if sent and received:
                    break
                time.sleep(0.01)
            expand = True
        if expand:
            log_debug(logger, self._log_prefix + "Do stack expansion")
            self.comm.Barrier()
            self._sync_i_max()
            self._expand_stacks_mpi()
        else:
            pass
            #log_debug(logger, self._log_prefix + "No stack expansion")

    def _expand_stacks_mpi(self, i_max=None):
        self._f.close()
        log_debug(logger, self._log_prefix + "File closed for stack expansion")
        self.comm.Barrier()
        if i_max is None:
            i_max = self._i_max
        stack_length_new = self._stack_length
        while i_max >= stack_length_new:
            stack_length_new *= 2
        if self.comm.rank == 0:
            self._f = h5py.File(self._filename, "r+")
            log_debug(logger, self._log_prefix + "Start stack expansion (%i >= %i) - new stack length will be %i" % (i_max, self._stack_length, stack_length_new))
            self._resize_stacks(stack_length_new)
            self._f.close()
        self._stack_length = stack_length_new
        self.comm.Barrier()
        self._f = h5py.File(self._filename, "r+", driver='mpio', comm=self.comm)
        self._f.atomic = True
        self.comm.Barrier()
        log_debug(logger, self._log_prefix + "File reopened after stack expansion")
        
    def _close_signal(self):
        if self.comm.rank == 0:
            self._closing_ranks = [0]
        else:
            buf = numpy.array(self.comm.rank, dtype="i")
            self.comm.Send([buf, MPI.INT], dest=0, tag=MPI_TAG_READY)
            log_debug(logger, self._log_prefix + "Rank %i sent closing signal to master" % (self.comm.rank))
            
    def _update_ready(self):
        if self.comm.rank == 0:
            while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI_TAG_READY):
                buf = numpy.empty(1, dtype='i')
                self.comm.Recv([buf, MPI.INT], source=MPI.ANY_SOURCE, tag=MPI_TAG_READY)
                rank = buf[0]
                self._closing_ranks.append(rank)
                log_debug(logger, self._log_prefix + "Received closing signal from rank %i (%i/%i)" % (rank,len(self._closing_ranks),self.comm.size))
            if len(self._closing_ranks) == self.comm.size:
                for i in range(1, self.comm.size):
                    send_buf = numpy.array(1, dtype='i')
                    self.comm.Send([send_buf, MPI.INT], dest=i, tag=MPI_TAG_READY)
                    recv_buf = numpy.empty(1, dtype='i')
                    self.comm.Recv([recv_buf, MPI.INT], source=i, tag=MPI_TAG_READY)
                self._ready = True
                log_debug(logger, self._log_prefix + "Master sent out ready signals to slaves")
        else:
            if self.comm.Iprobe(source=0, tag=MPI_TAG_READY):
                recv_buf = numpy.empty(1, dtype='i')
                self.comm.Recv([recv_buf, MPI.INT], source=0, tag=MPI_TAG_READY)
                send_buf = numpy.array(1, dtype='i')
                self.comm.Send([send_buf, MPI.INT], dest=0, tag=MPI_TAG_READY)
                self._ready = True
                log_debug(logger, self._log_prefix + "Slave rank %i received ready signals from master" %  self.comm.rank)
                
    def _sync_i_max(self):
        sendbuf = numpy.array(self._i_max, dtype='i')
        recvbuf = numpy.empty(1, dtype='i')
        log_debug(logger, self._log_prefix + "Entering allreduce with maximum index %i" % (self._i_max))
        self.comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        self._i_max = recvbuf[0]
        log_debug(logger, self._log_prefix + "After reduce: i_max = %i" % self._i_max)

        
    
