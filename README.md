# h5writer
HDF5 write tool for writing data slices

Basic usage

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
    "general_string" : "abc",
    "general_data1d" : np.random.rand(10),
}

W.write_solo(solo_dict)
W.close()

with h5py.File("file.h5", "r") as f:
    for ds in f.keys():
        print(f[ds])
```
