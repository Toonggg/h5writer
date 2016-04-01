import numpy, os, time
import h5py

import logging
logger = logging.getLogger(__name__)

from log import log_and_raise_error, log_warning, log_info, log_debug

class AbstractH5Writer:
    def __init__(self, filename, chunksize, compression):
        self._filename = os.path.expandvars(filename)        
        self._chunksize = chunksize
        self._stack_length = chunksize
        self._i = None
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
                if group_prefix_new != "/" and group_prefix_new not in self._f:
                    self._f.create_group(group_prefix_new)
                self._initialise_tree(D[k], group_prefix=group_prefix_new)
            else:
                name = group_prefix + k
                log_debug(logger, self._log_prefix + "Creating dataset %s" % (name))
                if name not in self._f:
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
                if name not in self._f:
                    log_and_raise_error(logger, self._log_prefix + "Write to dataset %s at stack position %i failed because it does not exist. Note that all datasets are initialised from the data of the first write call." % (name, self._i))
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

    def _is_stack(self, name):
        a = self._f[name].attrs
        return ("axes" in a.keys() and a["axes"][0].startswith('experiment_identifier'))
    
    def _resize_stacks(self, stack_length, group_prefix="/"):
        if group_prefix == "/":
            log_info(logger, self._log_prefix + "Resize datasets to new length: %i" % stack_length)
        if stack_length == 0:
            log_warning(logger, self._log_prefix + "Cannot resize stacks to length 0. Skip resize stacks.")
            return
        keys = self._f[group_prefix].keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(self._f[name], h5py.Dataset):
                if self._is_stack(name):
                    self._resize_stack(stack_length, name)
            else:
                self._resize_stacks(stack_length, name + "/")
        self._stack_length = stack_length
            
    def _resize_stack(self, stack_length, name):
        new_shape = list(self._f[name].shape)
        new_shape[0] = stack_length
        new_shape = tuple(new_shape)
        log_debug(logger, self._log_prefix + "Resize dataset %s [old shape: %s, new shape: %s]" % (name, str(self._f[name].shape), str(new_shape)))
        t0 = time.time()
        self._f[name].resize(new_shape)
        t1 = time.time()
        log_debug(logger, self._log_prefix + "Resizing time: %f sec (HDF5)" % (t1-t0))
