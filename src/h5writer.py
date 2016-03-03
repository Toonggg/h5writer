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
        self._i = -1
        self._i_max = -1
        self._create_dataset_kwargs = {}
        self._log_prefix = ""
        if compression is not None:
            self._create_dataset_kwargs["compression"] = compression
        self._initialised = False
        # Chache
        self._solocache = {}
            
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

    def write_solo(self, data_dict):
        """
        Call this function for writing datasets that have no stack dimension (i.e. no slices).
        """
        return self._to_solocache(data_dict, target=self._solocache)
                    
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
