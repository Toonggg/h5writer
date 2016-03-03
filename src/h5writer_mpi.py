import numpy, os, time
import h5py

from log import log_and_raise_error, log_warning, log_info, log_debug

try:
    from mpi4py import MPI
except:
    log_warning(logger, "Cannot import mpi4py!")

MPI_TAG_INIT   = 1# + 4353
MPI_TAG_EXPAND = 2# + 4353
MPI_TAG_READY  = 3# + 4353
MPI_TAG_CLOSE  = 4# + 4353

from h5writer import AbstractH5Writer,logger

class H5WriterMPI(AbstractH5Writer):
    """
    HDF5 writer class for multi-process writing.
    """
    def __init__(self, filename, comm, chunksize=100, compression=None):
        # Initialisation of base class
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        # MPI communicator
        self.comm = comm
        # This "if" avoids that processes that are not in the communicator (like the master process of hummingbird) interact with the file and block
        if not self._is_in_communicator():
            return
        # Logging
        self._log_prefix = "(%i) " %  self.comm.rank
        # Index
        self._i += self.comm.rank
        self._i_max = -1
        # Status
        self._ready = False
        # Chache
        self._solocache = {}
        # Open file
        if os.path.exists(self._filename):
            log_warning(logger, self._log_prefix + "File %s exists and is being overwritten" % (self._filename))
        self._f = h5py.File(self._filename, "w", driver='mpio', comm=self.comm)        
        
    def write_slice(self, data_dict, i=None):
        """
        Call this function for writing all data in data_dict as a slice of stacks (first dimension = stack dimension).
        Dictionaries within data_dict are represented as HDF5 groups. The slice index is either the next one (i=None) or a given integer i.
        """
        # Initialise of tree (groups and datasets)
        if not self._initialised:
            self._initialise_tree(data_dict)
            self._initialised = True
        self._i = self._i + self.comm.size if i is None else i
        if self._i >= self._stack_length:
            # Expand stacks if needded
            while self._i >= self._stack_length:
                self._expand_signal()
                self._expand_poll()
                if self._i >= self._stack_length:
                    time.sleep(1)
        else:
            self._expand_poll()
        # Write data
        self._write_group(data_dict)
        # Update of maximum index
        self._i_max = self._i if self._i > self._i_max else self._i_max

    def write_solo(self, data_dict):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        """
        return self._to_solocache(data_dict, target=self._solocache)

    def write_solo_mpi_reduce(self, data_dict, op):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        Data will be reduced between processes using the given open MPI operator (see for example https://pythonhosted.org/mpi4py/apiref/mpi4py-module.html).
        """
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
        while True:
            log_debug(logger, self._log_prefix + "Closing loop")
            self._expand_poll()
            self._update_ready()
            if self._ready:
                break
            time.sleep(1.)

        self.comm.Barrier()
        log_debug(logger, self._log_prefix + "Sync stack length")
        self._sync_i_max()

        log_debug(logger, self._log_prefix + "Shrink stacks")
        self.comm.Barrier()
        self._shrink_stacks()
        self.comm.Barrier()

        log_debug(logger, self._log_prefix + "Closing file %s for parallel writing." % (self._filename))
        self._f.close()
        log_debug(logger, self._log_prefix + "File %s closed for parallel writing." % (self._filename))

        log_debug(logger, self._log_prefix + "Write solo cache to file %s" % (self._filename))
        self._write_solocache_to_file()
        log_debug(logger, self._log_prefix + "Solo cache written to file %s" % (self._filename))

        log_info(logger, self._log_prefix + "HDF5 parallel writer instance for file %s closed." % (self._filename))
    
    def _is_in_communicator(self):
        try:
            out = self.comm.rank != MPI.UNDEFINED
        except MPI.Exception:
            out = False
        if not out:
            log_warning(logger, "This process cannot write.")
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

    def _write_solocache_to_file(self):
        if self._is_master():
            self._f = h5py.File(self._filename, "r+")
        self._write_solocache_group_to_file(self._solocache)
        if self._is_master():
            self._f.close()
            
    def _write_solocache_group_to_file(self, data_dict, group_prefix="/"):
        if self._is_master() and group_prefix != "/":
            if group_prefix in self._f:
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
        
    def _expand_signal(self):
        log_debug(logger, self._log_prefix + "Send expand signal")
        for i in range(self.comm.size):
            buf = numpy.array(self._i, dtype="i")
            self.comm.Send([buf, MPI.INT], dest=i, tag=MPI_TAG_EXPAND)
            #self.comm.send(buf, dest=i, tag=MPI_TAG_EXPAND)

    def _expand_poll(self):
        i_max = None
        for i in range(self.comm.size): 
            #buf = numpy.empty(1, dtype="i")
            #req = self.comm.Irecv([buf, MPI.INT], source=i, tag=MPI_TAG_EXPAND)
            #if req.test():
            if self.comm.Iprobe(source=i, tag=MPI_TAG_EXPAND):
                buf = numpy.empty(1, dtype="i")
                self.comm.Recv([buf, MPI.INT], source=i, tag=MPI_TAG_EXPAND)
                if i_max is None:
                    i_max = buf[0]
                else:
                    i_max = i_max if i_max > buf[0] else buf[0]
        if i_max is not None:
            # Is expansion still needed or is the signal outdated?
            if i_max < self._stack_length:
                log_debug(logger, self._log_prefix + "Expansion signal no longer needed (%i < %i)" % (i_max, self._stack_length))
                return
            while i_max >= self._stack_length:
                # OK - There is a process that needs longer stacks, so we'll actually expand the stacks
                stack_length_new = self._stack_length * 2
                log_debug(logger, self._log_prefix + "Start stack expansion (%i >= %i) - new stack length will be %i" % (i_max, self._stack_length, stack_length_new))
                self._expand_stacks(stack_length_new)

    def _close_signal(self):
        if self.comm.rank == 0:
            self._busy_clients    = [i for i in range(self.comm.size) if i != self.comm.rank]
            self._closing_clients = [i for i in range(self.comm.size) if i != self.comm.rank]
            self._signal_sent     = False
        else:
            buf = numpy.array(self.comm.rank, dtype="i")
            self.comm.Send([buf, MPI.INT], dest=0, tag=MPI_TAG_READY)
            
    def _update_ready(self):
        if self.comm.rank == 0:
            if (len(self._busy_clients) > 0):
                for i in self._busy_clients:
                    if self.comm.Iprobe(source=i, tag=MPI_TAG_READY):
                        self._busy_clients.remove(i)
            else:
                if not self._signal_sent:
                    for i in self._closing_clients:
                        # Send out signal
                        #self.comm.Isend([numpy.array(-1, dtype="i"), MPI.INT], dest=i, tag=MPI_TAG_CLOSE)
                        buf = numpy.array(-1, dtype="i")
                        self.comm.Send([buf, MPI.INT], dest=i, tag=MPI_TAG_CLOSE)
                    self._signal_sent = True
                # Collect more confirmations
                for i in self._closing_clients:
                    if self.comm.Iprobe(source=i, tag=MPI_TAG_CLOSE):
                        self._closing_clients.remove(i)
            self._ready = len(self._closing_clients) == 0
        else:
            if self.comm.Iprobe(source=0, tag=MPI_TAG_CLOSE):
                #self.comm.Isend([numpy.array(1, dtype="i"), MPI.INT], dest=0, tag=MPI_TAG_CLOSE)
                buf = numpy.array(1, dtype="i")
                self.comm.Send([buf, MPI.INT], dest=0, tag=MPI_TAG_CLOSE)
                self._ready = True
        log_debug(logger, self._log_prefix + "Ready status updated: %i" % (self._ready))
        
    
    def _sync_i_max(self):
        sendbuf = numpy.array(self._i_max, dtype='i')
        recvbuf = numpy.empty(1, dtype='i')
        log_debug(logger, self._log_prefix + "Entering allreduce with maximum index %i" % (self._i_max))
        self.comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        self._i_max = recvbuf[0]
        
    
