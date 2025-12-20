Using the new MDIO storage backend

Quick example:

from synthoseis.storage import StorageClient
import numpy as np

# create/open a store
client = StorageClient.open("/tmp/synth_store", mode="a")

# write a dataset
arr = np.random.random((100,100,50))
client.create_dataset("vp", arr)

# read it back
vp = client.get_dataset("vp")

client.close()

Notes
-----
- This backend uses mdio (zarr) under the hood. Install with `pip install mdio zarr` or add to your environment.
- The StorageClient is intentionally small; extend for attributes, chunking, and dask-backed reading.
