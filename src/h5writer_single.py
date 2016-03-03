import numpy, os, time
import h5py

from log import log_and_raise_error, log_warning, log_info, log_debug

from h5writer import AbstractH5Writer,logger

class H5Writer(AbstractH5Writer):
    """
    HDF5 writer class for single-process writing.
    """
    def __init__(self, filename, chunksize=100, compression=None):
        AbstractH5Writer.__init__(self, filename, chunksize=chunksize, compression=compression)
        if os.path.exists(self._filename):
            log_warning(logger, self._log_prefix + "File %s exists and is being overwritten" % (self._filename))
        self._f = h5py.File(self._filename, "w")

    def write_slice(self, data_dict, i=None):
        """
        Call this function for writing all data in data_dict as a slice of stacks (first dimension = stack dimension).
        Dictionaries within data_dict are represented as HDF5 groups. The slice index is either the next one (i=None) or a given integer i.
        """
        if not self._initialised:
            # Initialise of tree (groups and datasets)
            self._initialise_tree(data_dict)
            self._intialised = True        
        self._i = self._i + 1 if i is None else i
        if self._i >= (self._stack_length-1):
            # Expand stacks if needed
            self._expand_stacks(self._stack_length * 2)
        # Write data
        self._write_group(data_dict)
        # Update of maximum index
        self._i_max = self._i if self._i > self._i_max else self._i_max

    def _to_solocache(self, data_dict, target):
        keys = data_dict.keys()
        keys.sort()
        for k in keys:
            if isinstance(data_dict[k], dict):
                if k not in target:
                    target[k] = {}
                self._to_solocache(data_dict[k], target=target[k])
            else:
                target[k] = data_dict[k]

    def _write_solocache_group_to_file(self, data_dict, group_prefix="/"):
        if group_prefix != "/" and group_prefix not in self._f:
            self._f.create_group(group_prefix)
        keys = data_dict.keys()
        keys.sort()
        for k in keys:
            name = group_prefix + k
            if isinstance(data_dict[k], dict):
                self._write_solocache_group_to_file(data_dict[k], group_prefix=name+"/")
            else:
                data = data_dict[k]
                self._f[name] = data

    def close(self):
        """
        Close file.
        """
        self._shrink_stacks()
        self._write_solocache_group_to_file(self._solocache)
        self._f.close()
        log_info(logger, self._log_prefix + "HDF5 writer instance for file %s closed." % (self._filename))
        
