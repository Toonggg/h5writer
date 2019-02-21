# h5writer
HDF5 write tool for writing data slices with *numpy* objects

### Basic usage

#### Writing to file:
```
import numpy as np
from h5writer import H5Writer

W = H5Writer("file.h5")

for i in range(10):
    slice_dict = {
        "i" : i,
        "data1d" : np.random.rand(10),
        "data2d" : np.random.rand(10*20).reshape((10, 20)),
        "data3d" : np.random.rand(10*20*30).reshape((10, 20, 30)),
    }
    W.write_slice(slice_dict)

solo_dict = {
    "general_string" : np.string_("abc"),
    "general_data1d" : np.random.rand(10),
}

W.write_solo(solo_dict)
W.close()
```
#### Reading file:
```
with h5py.File("file.h5", "r") as f:
    for ds in f.keys():
        print(f[ds])
```
#### *Output:*
```
<HDF5 dataset "data1d": shape (10, 10), type "<f8">
<HDF5 dataset "data2d": shape (10, 10, 20), type "<f8">
<HDF5 dataset "data3d": shape (10, 10, 20, 30), type "<f8">
<HDF5 dataset "general_data1d": shape (10,), type "<f8">
<HDF5 dataset "general_string": shape (), type "|O">
<HDF5 dataset "i": shape (10,), type "<i4">
```
