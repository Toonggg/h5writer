import numpy, os, time
import h5py

try:
    from mpi4py import MPI
except:
    log_warning(logger, "Cannot import mpi4py!")    
    MPI = None
    
from log import log_and_raise_error, log_warning, log_info, log_debug

from h5writer import AbstractH5Writer,logger

class H5WriterMPISW(AbstractH5Writer):
    """
    HDF5 writer class for MPI with a master process (rank=0) reserved for writing and inter-process-communication. Slave processes (rank>0) can call write methods and data is sent to the master process for serial writing.
    """
    def __init__(self, filename, comm, chunksize=100, compression=None):
        if MPI is None:
            log_and_raise_error(logger, "Could not import MPI and hence cannot initialise H5WriterMPI instance.")
            return
        self.comm = comm
        if not isinstance(self.comm, MPI.Comm):
            log_and_raise_error(logger, "Cannot initialise H5WriterMPI instance. \'%s\' is not an mpi4py.MPI.Comm instance." % str(self.comm))
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        if self._is_master():
            if os.path.exists(self._filename):
                log_warning(logger, self._log_prefix + "File %s exists and is being overwritten" % (self._filename))
            self._f = h5py.File(self._filename, "w")
            self._master_loop()
        self._closed = False
            
    def _is_in_communicator(self):
        try:
            out = self.comm.rank != MPI.UNDEFINED
        except MPI.Exception:
            out = False
        return out
            
    def _is_master(self):
        return (self._is_in_communicator() and self.comm.rank == 0)

    def _master_loop(self):
        log_debug(logger, "Master loop started")
        t_start = time.time()
        slices = numpy.zeros(self.comm.size, 'i')
        closed = numpy.zeros(self.comm.size, 'i')
        while True:
            status = MPI.Status()
            l = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
            source = status.Get_source()
            if l == "close":
                closed[source] = 1
                if closed.sum() == self.comm.size-1:
                    break
            else:
                self._transfer_numpy_arrays(l, source=source)
                if "write_slice" in l:
                    log_debug(logger, "Write slice to file")
                    t0 = time.time()
                    self._write_slice_master(l["write_slice"], i=slices[source]*(self.comm.size-1)+source-1)
                    slices[source] += 1
                    t1 = time.time()
                    t_log = t1-t0
                    log_info(logger, "Writing rate %.1f Hz; slice %i; logging %.2f sec" % (slices.sum()/(time.time()-t_start),slices.sum(),t_log))
                if "write_solo" in l:
                    log_debug(logger, "Write solo to file")
                    self.write_solo(l["write_solo"])

        log_debug(logger, "Master writer is closing.")
        self._resize_stacks(self._i_max + 1)
        self._f.close()
        log_debug(logger, "File %s closed." % self._filename)
                        
    def _send_for_writing(self, data_dict):
        """
        Sending data dictionaries to master process without pickeling numpy arrays.
        """
        skeleton = _make_skeleton(data_dict)
        self.comm.send(skeleton, dest=0, tag=0)
        self._transfer_numpy_arrays(skeleton, data_dict)

    def _transfer_numpy_arrays(self, skeleton, data_dict=None, source=None):
        mode = 'master' if data_dict is None else 'slave' 
        keys = skeleton.keys()
        keys.sort()
        for k in keys:
            if isinstance(skeleton[k], dict):
                self._transfer_numpy_arrays(skeleton[k], None if mode == 'master' else data_dict[k], source=source)
            elif isinstance(skeleton[k], _ArrayDescriptor):
                if mode == 'master':
                    skeleton[k] = numpy.empty(shape=skeleton[k].shape, dtype=skeleton[k].dtype)
                    self.comm.Recv(skeleton[k], source=source, tag=0)
                else:
                    self.comm.Send(data_dict[k], dest=0, tag=0)

    def write_slice(self, data_dict):
        """
        Call this function for writing all data in data_dict as a stack of slices (first dimension = stack dimension).
        Dictionaries within data_dict are represented as HDF5 groups. The slice index is either the next one.
        """
        self._send_for_writing({"write_slice": data_dict})
        
    def _write_slice_master(self, data_dict, i):
        if not self._initialised:
            # Initialise of tree (groups and datasets)
            self._initialise_tree(data_dict)
            self._initialised = True        
        self._i = i
        # Expand stacks if needed
        if self._i >= (self._stack_length-1):
            self._resize_stacks(self._stack_length * 2)
        # Write data
        self._write_group(data_dict)
        # Update of maximum index
        self._i_max = self._i if self._i > self._i_max else self._i_max

    def write_solo(self, data_dict):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        """
        self._send_for_writing({"write_solo": data_dict})
        
    def _write_solo_master(self, data_dict):
        self._write_solo_group(data_dict)

    def _write_solo_group(self, data_dict, group_prefix="/"):
        if group_prefix != "/" and group_prefix not in self._f:
            self._f.create_group(group_prefix)
        keys = data_dict.keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(data_dict[k], dict):
                self._write_solo_group(data_dict[k], group_prefix=name+"/")
            else:
                if name in self._f:
                    log_warning(logger, "Dataset %s already exists! Overwriting with new data." % name)
                self._f[name] = data_dict[k]
                
    def close(self):
        """
        Close file.
        """
        if self._closed:
            log_debug("Instance already closed.")
        if not self._is_master():
            self.comm.send("close", dest=0, tag=0)
        else:
            log_info(logger, self._log_prefix + "HDF5 writer master process for file %s closed." % (self._filename))
        self._closed = True
        
def _make_skeleton(data_dict):
    skeleton = {}
    keys = data_dict.keys()
    keys.sort()
    for k in keys:
        v = data_dict[k]
        if isinstance(v, dict):
            skeleton[k] = _make_skeleton(v)
        else:
            if isinstance(v, numpy.ndarray):
                skeleton[k] = _ArrayDescriptor(v)
            else:
                skeleton[k] = v
    return skeleton

class _ArrayDescriptor:
    def __init__(self, ndarray):
        self.dtype = ndarray.dtype
        self.shape = ndarray.shape
