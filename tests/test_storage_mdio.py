import os
import tempfile
import numpy as np

try:
    from synthoseis.storage import StorageClient
    has_mdio = True
except Exception:
    has_mdio = False


@pytest.mark.skipif(not has_mdio, reason="MDIO not available")
def test_storage_create_and_read():
    if not has_mdio:
        print("mdio not installed; skipping storage test")
        return
    td = tempfile.mkdtemp()
    client = StorageClient.open(td, mode="a")
    data = np.arange(24).reshape((2,3,4))
    client.create_dataset("testarray", data)
    out = client.get_dataset("testarray")
    assert np.array_equal(out, data)
    # Test dask if available
    try:
        dask_arr = client.get_dataset("testarray", use_dask=True)
        assert hasattr(dask_arr, 'compute')  # dask array
        assert np.array_equal(dask_arr.compute(), data)
    except ImportError:
        pass  # dask not available
    client.close()