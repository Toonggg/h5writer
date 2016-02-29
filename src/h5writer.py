# Adapted code from condor (https://github.com/mhantke/condor)
import numpy, os, time
import h5py

import logging
from log import log_and_raise_error, log_warning, log_info, log_debug
logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except:
    log_warning(logger, "Cannot import mpi4py!")

MPI_TAG_INIT   = 1# + 4353
MPI_TAG_EXPAND = 2# + 4353
MPI_TAG_READY  = 3# + 4353
MPI_TAG_CLOSE  = 4# + 4353

class AbstractH5Writer:
    def __init__(self, filename, chunksize, compression):
        self._filename = os.path.expandvars(filename)        
        self._chunksize = chunksize
        self._stack_length = chunksize
        self._i = -1
        self._i_max = -1
        self._create_dataset_kwargs = {}
        self._log_prefix = ""
        if compression is not None:
            self._create_dataset_kwargs["compression"] = compression
        self._initialised = False
            
    def _initialise_tree(self, D, group_prefix="/"):
        keys = D.keys()
        keys.sort()
        for k in keys:
            if isinstance(D[k],dict):
                group_prefix_new = group_prefix + k + "/"
                log_debug(logger, self._log_prefix + "Creating group %s" % (group_prefix_new))
                self._f.create_group(group_prefix_new)
                self._initialise_tree(D[k], group_prefix=group_prefix_new)
            else:
                name = group_prefix + k
                log_debug(logger, self._log_prefix + "Creating dataset %s" % (name))
                data = D[k]
                self._create_dataset(data, name)
                    
    def _write_group(self, D, group_prefix="/"):
        keys = D.keys()
        keys.sort()
        for k in keys:
            if isinstance(D[k],dict):
                group_prefix_new = group_prefix + k + "/"
                self._write_group(D[k], group_prefix_new)
            else:
                name = group_prefix + k
                data = D[k]
                log_debug(logger, self._log_prefix + "Write to dataset %s at stack position %i" % (name, self._i))
                if numpy.isscalar(data):
                    self._f[name][self._i] = data
                else:
                    self._f[name][self._i,:] = data[:]
                
    def _create_dataset(self, data, name):
        if numpy.isscalar(data):
            maxshape = (None,)
            shape = (self._chunksize,)
            dtype = numpy.dtype(type(data))
            if dtype == "S":
                dtype = h5py.new_vlen(str)
            axes = "experiment_identifier:value"
        else:
            data = numpy.asarray(data)
            try:
                h5py.h5t.py_create(data.dtype, logical=1)
            except TypeError:
                log_warning(logger, self._log_prefix + "Could not save dataset %s. Conversion to numpy array failed" % (name))
                return 1
            maxshape = tuple([None]+list(data.shape))
            shape = tuple([self._chunksize]+list(data.shape))
            dtype = data.dtype
            ndim = data.ndim
            axes = "experiment_identifier"
            if ndim == 1: axes = axes + ":x"
            elif ndim == 2: axes = axes + ":y:x"
            elif ndim == 3: axes = axes + ":z:y:x"
        log_debug(logger, self._log_prefix + "Create dataset %s [shape=%s, dtype=%s]" % (name, str(shape), str(dtype)))
        self._f.create_dataset(name, shape, maxshape=maxshape, dtype=dtype, **self._create_dataset_kwargs)
        self._f[name].attrs.modify("axes",[axes])
        return 0
                    
    def _expand_stacks(self, stack_length, group_prefix="/"):
        keys = self._f[group_prefix].keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(self._f[name], h5py.Dataset):
                if not (name[1:].startswith("__") and name.endswith("__")):
                    self._expand_stack(stack_length, name)
            else:
                self._expand_stacks(stack_length, name + "/")
            
    def _expand_stack(self, stack_length, name):
        new_shape = list(self._f[name].shape)
        new_shape[0] = stack_length
        new_shape = tuple(new_shape)
        log_info(logger, self._log_prefix + "Expand dataset %s [old shape: %s, new shape: %s]" % (name, str(self._f[name].shape), str(new_shape)))
        self._f[name].resize(new_shape)
        self._stack_length = stack_length
            
    def _shrink_stacks(self, group_prefix="/"):
        stack_length = self._i_max + 1
        if stack_length == 0:
            log_warning(logger, self._log_prefix + "Cannot shrink stacks to length 0. Skip shrinking stacks.")
            return
        keys = self._f[group_prefix].keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(self._f[name], h5py.Dataset):
                if not (name[1:].startswith("__") and name.endswith("__")):
                    if stack_length < 1:
                        log_warning(logger, self._log_prefix + "Cannot reduce dataset %s to length %i" % (name, stack_length))
                        return
                    log_debug(logger, self._log_prefix + "Shrinking dataset %s to stack length %i" % (name, stack_length))
                    s = list(self._f[name].shape)
                    s.pop(0)
                    s.insert(0, self._i_max+1)
                    s = tuple(s)
                    self._f[name].resize(s)
            else:
                self._shrink_stacks(name + "/")


class H5Writer(AbstractH5Writer):
    def __init__(self, filename, chunksize=100, compression=None):
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        if os.path.exists(self._filename):
            log_warning(logger, self._log_prefix + "File %s exists and is being overwritten" % (self._filename))
        self._f = h5py.File(self._filename, "w")

    def close(self):
        self._f.close()
        log_info(logger, self._log_prefix + "HDF5 writer instance for file %s closed." % (self._filename))
        
    def write_slice(self, data_dict, i=None):
        if not self._initialised:
            # Initialise of tree (groups and datasets)
            self._initialise_tree(data_dict)
            self._initialised = True        
        self._i = self._i + 1 if i is None else i
        if self._i >= (self._stack_length-1):
            # Expand stacks if needed
            self._expand_stacks(self._stack_length * 2)
        # Write data
        self._write_group(data_dict)
        # Update of maximum index
        self._i_max = self._i if self._i > self._i_max else self._i_max

    def write_solo(self, data_dict): 
        self._write_solo_group(data_dict)
            
    def _write_solo_group(self, data_dict, group_prefix="/"):
        if group_prefix != "/" and not group_prefix in self._f:
            self._f.create_group(group_prefix)
        for k,v in data_dict.items():
            name = group_prefix + k
            if isinstance(v, dict):
                self._write_solo_group(v, group_prefix=name+"/")
            else:
                self._f[name] = v                
        
class H5WriterMPI(AbstractH5Writer):
    def __init__(self, filename, comm, chunksize=100, compression=None):
        # Initialisation of base class
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        # MPI communicator
        self.comm = comm
        # Logging
        self._log_prefix = "(%i) " %  self.comm.rank
        # This "if" avoids that processes that are not in the communicator (like the master process of hummingbird) interact with the file and block
        if not self._is_in_communicator():
            return
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
        
    def _is_in_communicator(self):
        try:
            out = self.comm.rank != MPI.UNDEFINED
        except MPI.Exception:
            out = False
        if not out:
            log_warning(logger, self._log_prefix + "This process cannot write.")
        return out
            
    def _is_master(self):
        return (self._is_in_communicator() and self.comm.rank == 0)
        
    def write_slice(self, data_dict, i=None):
        # Initialise of tree (groups and datasets)
        if not self._initialised:
            self._initialise_tree(data_dict)
            self._initialised = True
        self._i = (self._i + 1) if i is None else i
        if self._i > (self._stack_length-1):
            # Expand stacks if needde
            while self._i > (self._stack_length-1):
                self._expand_signal()
                self._expand_poll()
                if self._i > (self._stack_length-1):
                    time.sleep(1)
        else:
            self._expand_poll()
        # Write data
        self._write_group(data_dict)
        # Update of maximum index
        self._i_max = self._i if self._i > self._i_max else self._i_max

    def write_solo(self, data_dict):
        return self._to_solocache(data_dict, target=self._solocache)

    def write_solo_mpi_reduce(self, data_dict, op):
        return self._to_solocache(data_dict, target=self._solocache, op=op)

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
            self.comm.Send([numpy.array(self._i, dtype="i"), MPI.INT], dest=i, tag=MPI_TAG_EXPAND)

    def _expand_poll(self):
        L = []
        for i in range(self.comm.size): 
            if self.comm.Iprobe(source=i, tag=MPI_TAG_EXPAND):
                buf = numpy.empty(1, dtype="i")
                self.comm.Recv([buf, MPI.INT], source=i, tag=MPI_TAG_EXPAND)
                L.append(buf[0])
        if len(L) > 0:
            i_max = max(L)
            # Is expansion still needed or is the signal outdated?
            if i_max < self._stack_length:
                log_debug(logger, self._log_prefix + "Expansion signal no longer needed (%i < %i)" % (i_max, self._stack_length))
                return
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
            self.comm.Isend([numpy.array(self.comm.rank, dtype="i"), MPI.INT], dest=0, tag=MPI_TAG_READY)
            
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
                        self.comm.Isend([numpy.array(-1, dtype="i"), MPI.INT], dest=i, tag=MPI_TAG_CLOSE)
                    self._signal_sent = True
                # Collect more confirmations
                for i in self._closing_clients:
                    if self.comm.Iprobe(source=i, tag=MPI_TAG_CLOSE):
                        self._closing_clients.remove(i)
            self._ready = len(self._closing_clients) == 0
        else:
            if self.comm.Iprobe(source=0, tag=MPI_TAG_CLOSE):
                self.comm.Isend([numpy.array(1, dtype="i"), MPI.INT], dest=0, tag=MPI_TAG_CLOSE)
                self._ready = True
        log_debug(logger, self._log_prefix + "Ready status updated: %i" % (self._ready))
        
    
    def _sync_i_max(self):
        sendbuf = numpy.array(self._i_max, dtype='i')
        recvbuf = numpy.empty(1, dtype='i')
        log_debug(logger, self._log_prefix + "Entering allreduce with maximum index %i" % (self._i_max))
        self.comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.MAX)
        self._i_max = recvbuf[0]
        
    def close(self):
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
