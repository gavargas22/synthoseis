# Zarr v3 Migration - Implementation Plan
## Complete HDF5 Deprecation & Zarr v3 Finalization

**Branch:** `claude/migrate-hdf5-storage-u9pgu` (current working branch)
**Base Branch:** `feat/gavargas22/mdio`
**Target:** Complete HDF5 removal, production-ready Zarr v3 backend
**Estimated Effort:** 8-12 hours
**Status:** Ready to Implement

---

## Executive Summary

The `feat/gavargas22/mdio` branch has **95% complete** migration to Zarr v3. This plan covers the remaining 5% to achieve full HDF5 deprecation:

- ✅ Zarr v3.1.3 is installed and working
- ✅ All core modules updated
- ✅ HDF5 dependencies removed from pyproject.toml
- ⚠️ **5% Remaining:** Cleanup legacy references, remove MDIO dep, enhance tests, add docs

**Key Finding:** Despite the branch name "mdio", the implementation uses **pure Zarr v3 API**, not MDIO library. The MDIO dependency can be safely removed.

---

## Current Branch Status

### Working Branch Decision

**Option 1: Continue on `claude/migrate-hdf5-storage-u9pgu`**
- ✅ Already checked out
- ✅ Clean starting point
- ✅ Follows naming convention (claude/*)
- ✅ Can pull from `feat/gavargas22/mdio` as base
- ⚠️ Need to ensure we're based on mdio branch

**Option 2: Create new branch from `feat/gavargas22/mdio`**
- Could create `claude/zarr-v3-finalize-u9pgu`
- Cleaner git history
- More explicit branch name

**Recommendation:** Continue on `claude/migrate-hdf5-storage-u9pgu` since it's already set up. First, we'll ensure we have the latest from `feat/gavargas22/mdio`.

### Git Strategy

```bash
# Ensure we're on the right branch
git checkout claude/migrate-hdf5-storage-u9pgu

# Merge latest from mdio branch
git merge origin/feat/gavargas22/mdio --no-ff

# Complete implementation tasks
# ... work ...

# Commit changes
git add .
git commit -m "Complete Zarr v3 migration: remove HDF5 remnants, enhance tests, add docs"

# Push to origin
git push -u origin claude/migrate-hdf5-storage-u9pgu
```

---

## Implementation Tasks

### Phase 1: Code Cleanup (Priority: HIGH)

**Estimated Time:** 1 hour

#### Task 1.1: Remove Legacy HDF5 References

**Files to Modify:**
1. `datagenerator/Parameters.py`
2. `datagenerator/Salt.py`

**Changes:**

**File:** `datagenerator/Parameters.py`
```python
# REMOVE these lines (around line 492-494):
self.hdf_master = os.path.join(
    self.work_subfolder, f"seismicCube__{self.date_stamp}.hdf"
)

# REMOVE this line (around line 677):
self.hdf_store = d["write_to_hdf"]
```

**File:** `datagenerator/Salt.py`
```python
# REMOVE these commented lines (around 449, 487-488):
#    self.cfg.h5file.root.ModelData.faulted_depth.shape[2] - 1,
# self.cfg.h5file.root.ModelData.faulted_depth_maps[:] = depth_maps
# self.cfg.h5file.root.ModelData.faulted_depth_maps_gaps[:] = depth_maps_gaps_salt
```

**Validation:**
```bash
# Verify no HDF5 references remain
grep -r "h5py\|hdf5\|\.hdf\|\.h5\|h5file" --include="*.py" datagenerator/ synthoseis/ rockphysics/
# Should return no results (except maybe docs)
```

#### Task 1.2: Remove MDIO Dependency

**File:** `pyproject.toml`

**Change:**
```toml
# REMOVE this line:
"multidimio[distributed]",
```

**Rationale:**
- Current implementation uses pure `zarr` API, not MDIO
- `mdio_backend.py` only imports mdio in try/except but never uses it
- Storage backend uses `zarr.open()`, not `mdio.*` functions
- Reduces dependencies and install size

**Validation:**
```bash
# Test that code still works without mdio
uv remove multidimio
uv sync
uv run python -c "from synthoseis.storage import StorageClient; print('OK')"
```

#### Task 1.3: Rename Backend File (Optional)

**Optional Change:**
```bash
git mv synthoseis/storage/mdio_backend.py synthoseis/storage/zarr_backend.py
```

**If renamed, update:**
- `synthoseis/storage/__init__.py`:
  ```python
  from .zarr_backend import StorageClient, DatasetNotFound
  ```
- Any tests importing from mdio_backend

**Decision:** Optional - can keep current name for now, rename later if desired.

---

### Phase 2: Enhanced Configuration (Priority: MEDIUM)

**Estimated Time:** 2-3 hours

#### Task 2.1: Add Chunking Strategies

**File:** `synthoseis/storage/zarr_backend.py` (or mdio_backend.py)

**Add module-level constants:**
```python
# Chunking strategies for different volume types
CHUNK_STRATEGIES = {
    "3d_volume_balanced": (128, 128, 128),     # Default - general purpose
    "3d_volume_z_major": (64, 64, 256),        # Vertical operations
    "3d_volume_xy_major": (128, 128, 64),      # Horizon slicing
    "horizon_maps": (128, 128, 10),             # 2D maps
    "seismic_4d": (1, 64, 64, 256),            # Multi-angle seismic
    "test_mode": (32, 32, 32),                  # Small for testing
}

def get_chunk_shape(strategy: str, array_shape: tuple) -> tuple:
    """Get chunk shape for given strategy and array shape."""
    base_chunks = CHUNK_STRATEGIES.get(strategy, (128, 128, 128))
    # Ensure chunks don't exceed array dimensions
    return tuple(min(chunk, dim) for chunk, dim in zip(base_chunks, array_shape))
```

**Update `create_dataset()` method:**
```python
def create_dataset(
    self,
    name: str,
    data: Optional[np.ndarray] = None,
    shape: Optional[Tuple[int, ...]] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    chunk_strategy: Optional[str] = None,  # NEW
    dtype: Optional[Any] = None,
    compressor: Optional[Any] = None
) -> Optional[np.ndarray]:
    # ... existing code ...

    # Determine chunks
    if chunks is None:
        if chunk_strategy:
            chunks = get_chunk_shape(chunk_strategy, actual_shape)
        else:
            # Default chunking
            chunks = tuple(min(128, s) for s in actual_shape)

    # ... rest of method ...
```

#### Task 2.2: Add Compression Configurations

**File:** `synthoseis/storage/zarr_backend.py`

**Add compression presets:**
```python
from numcodecs import Blosc, Zstd, GZip

COMPRESSOR_PRESETS = {
    "none": None,
    "blosc_fast": Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE),
    "blosc_default": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
    "blosc_max": Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE),
    "zstd": Zstd(level=5),
    "gzip": GZip(level=6),
}

def get_compressor(preset: str):
    """Get compressor from preset name."""
    if preset in COMPRESSOR_PRESETS:
        return COMPRESSOR_PRESETS[preset]
    return COMPRESSOR_PRESETS["blosc_default"]
```

**Update `create_dataset()` to accept preset name:**
```python
def create_dataset(
    self,
    name: str,
    data: Optional[np.ndarray] = None,
    shape: Optional[Tuple[int, ...]] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    chunk_strategy: Optional[str] = None,
    dtype: Optional[Any] = None,
    compressor: Optional[Any] = None,
    compressor_preset: Optional[str] = None,  # NEW
) -> Optional[np.ndarray]:
    # ... existing code ...

    # Determine compressor
    if compressor is None and compressor_preset:
        compressor = get_compressor(compressor_preset)
    elif compressor is None:
        compressor = COMPRESSOR_PRESETS["blosc_default"]  # Default compression

    # ... rest of method ...
```

#### Task 2.3: Update Parameters Class

**File:** `datagenerator/Parameters.py`

**Add configuration attributes:**
```python
def __init__(self, inputJSON, runid=None, test_mode=None):
    # ... existing code ...

    # Zarr configuration
    zarr_config = d.get("zarr_config", {})
    self.zarr_chunk_strategy = zarr_config.get("chunk_strategy", "3d_volume_balanced")
    self.zarr_compressor = zarr_config.get("compressor", "blosc_default")
    self.zarr_cache_size_mb = zarr_config.get("cache_size_mb", 500)
```

**Update example config:**

**File:** `config/example.json`

**Add new section:**
```json
{
  "model_name": "example_model",

  "zarr_config": {
    "chunk_strategy": "3d_volume_balanced",
    "compressor": "blosc_default",
    "cache_size_mb": 500
  }
}
```

#### Task 2.4: Use Configuration in Storage Calls

**Update calls in modules to pass configuration:**

**Example in `datagenerator/Horizons.py`:**
```python
# Before:
cfg.storage.create_dataset("depth_maps", depth_maps)

# After:
cfg.storage.create_dataset(
    "depth_maps",
    depth_maps,
    chunk_strategy="horizon_maps",
    compressor_preset=cfg.zarr_compressor
)
```

**Apply similar updates in:**
- `Geomodels.py` - use "3d_volume_balanced"
- `Seismic.py` - use "seismic_4d" for multi-angle volumes
- `Faults.py` - use "3d_volume_balanced"
- `Closures.py` - use "3d_volume_balanced"

---

### Phase 3: Testing (Priority: HIGH)

**Estimated Time:** 3-4 hours

#### Task 3.1: Rename and Expand Unit Tests

**File:** Rename `tests/test_storage_mdio.py` → `tests/test_zarr_backend.py`

**Expand tests:**
```python
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from synthoseis.storage import StorageClient, DatasetNotFound


@pytest.fixture
def temp_store():
    """Create temporary zarr store."""
    path = tempfile.mkdtemp(suffix=".zarr")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_storage_client_creation(temp_store):
    """Test StorageClient initialization."""
    client = StorageClient.open(temp_store, mode="a")
    assert client.root_path == str(Path(temp_store).absolute())
    assert client.store is not None
    client.close()


def test_create_dataset_with_data(temp_store):
    """Test dataset creation with numpy array."""
    client = StorageClient.open(temp_store, mode="a")
    data = np.random.random((100, 100, 50)).astype(np.float32)

    result = client.create_dataset("test_data", data)
    assert "test_data" in client.list_datasets()

    retrieved = client.get_dataset("test_data")
    np.testing.assert_array_equal(data, retrieved)
    client.close()


def test_create_dataset_with_shape(temp_store):
    """Test dataset creation with shape only."""
    client = StorageClient.open(temp_store, mode="a")

    result = client.create_dataset(
        "test_empty",
        shape=(50, 50, 25),
        dtype=np.float32
    )

    assert "test_empty" in client.list_datasets()
    retrieved = client.get_dataset("test_empty")
    assert retrieved.shape == (50, 50, 25)
    client.close()


def test_get_dataset_not_found(temp_store):
    """Test error handling for missing dataset."""
    client = StorageClient.open(temp_store, mode="a")

    with pytest.raises(DatasetNotFound):
        client.get_dataset("nonexistent")

    client.close()


def test_list_datasets(temp_store):
    """Test dataset listing."""
    client = StorageClient.open(temp_store, mode="a")

    client.create_dataset("data1", np.ones((10, 10, 10)))
    client.create_dataset("data2", np.ones((10, 10, 10)))
    client.create_dataset("other", np.ones((10, 10, 10)))

    all_datasets = client.list_datasets()
    assert len(all_datasets) == 3

    filtered = client.list_datasets(prefix="data")
    assert len(filtered) == 2
    assert "data1" in filtered
    assert "data2" in filtered

    client.close()


def test_remove_dataset(temp_store):
    """Test dataset removal."""
    client = StorageClient.open(temp_store, mode="a")

    client.create_dataset("temp_data", np.ones((10, 10, 10)))
    assert "temp_data" in client.list_datasets()

    client.remove_dataset("temp_data")
    assert "temp_data" not in client.list_datasets()

    client.close()


def test_chunking_configuration(temp_store):
    """Test custom chunking."""
    client = StorageClient.open(temp_store, mode="a")

    data = np.random.random((300, 300, 250)).astype(np.float32)
    custom_chunks = (64, 64, 128)

    client.create_dataset("chunked_data", data, chunks=custom_chunks)

    # Verify chunks in zarr metadata
    arr = client.store["chunked_data"]
    assert arr.chunks == custom_chunks

    client.close()


def test_compression_configuration(temp_store):
    """Test different compressors."""
    from numcodecs import Blosc

    client = StorageClient.open(temp_store, mode="a")
    data = np.random.random((100, 100, 50)).astype(np.float32)

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    client.create_dataset("compressed_data", data, compressor=compressor)

    # Verify compression in zarr metadata
    arr = client.store["compressed_data"]
    assert arr.compressor is not None
    assert arr.compressor.cname == "zstd"

    client.close()


def test_dask_array_retrieval(temp_store):
    """Test use_dask=True parameter."""
    pytest.importorskip("dask")  # Skip if dask not installed

    client = StorageClient.open(temp_store, mode="a")
    data = np.random.random((100, 100, 50)).astype(np.float32)
    client.create_dataset("dask_data", data)

    dask_array = client.get_dataset("dask_data", use_dask=True)

    # Check if it's a dask array
    import dask.array as da
    assert isinstance(dask_array, da.Array)

    # Verify data matches
    np.testing.assert_array_equal(data, dask_array.compute())

    client.close()


def test_context_manager(temp_store):
    """Test with statement usage."""
    data = np.ones((50, 50, 25))

    with StorageClient.open(temp_store, mode="a") as client:
        client.create_dataset("ctx_data", data)
        assert "ctx_data" in client.list_datasets()

    # Verify store closed properly
    # Re-open and verify data persisted
    with StorageClient.open(temp_store, mode="r") as client:
        retrieved = client.get_dataset("ctx_data")
        np.testing.assert_array_equal(data, retrieved)


def test_overwrite_dataset(temp_store):
    """Test overwriting existing dataset."""
    client = StorageClient.open(temp_store, mode="a")

    data1 = np.ones((50, 50, 25))
    data2 = np.zeros((50, 50, 25))

    client.create_dataset("data", data1)
    client.create_dataset("data", data2)  # Overwrite

    retrieved = client.get_dataset("data")
    np.testing.assert_array_equal(data2, retrieved)

    client.close()


def test_different_dtypes(temp_store):
    """Test various numpy dtypes."""
    client = StorageClient.open(temp_store, mode="a")

    dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8]

    for dtype in dtypes:
        data = np.random.random((10, 10, 10)).astype(dtype)
        name = f"data_{dtype.__name__}"
        client.create_dataset(name, data)
        retrieved = client.get_dataset(name)
        assert retrieved.dtype == dtype
        np.testing.assert_array_equal(data, retrieved)

    client.close()


def test_large_array_metadata(temp_store):
    """Test large array creation (metadata only, no actual data)."""
    client = StorageClient.open(temp_store, mode="a")

    # Create large array specification (would be ~8GB if materialized)
    large_shape = (1000, 1000, 1000)
    client.create_dataset("large_array", shape=large_shape, dtype=np.float64)

    # Verify it was created without errors
    assert "large_array" in client.list_datasets()
    arr = client.store["large_array"]
    assert arr.shape == large_shape

    client.close()
```

#### Task 3.2: Create Integration Test

**File:** `tests/test_zarr_integration.py`

```python
import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration file."""
    config = {
        "model_name": "test_model",
        "work_folder": str(tmp_path / "output"),
        "cube_shape": [50, 50, 50],
        "infill_factor": 1,
        "clip_edges": False,
        "verbose": False,
        "qc_plots": False,
        "write_to_segy": False,

        "zarr_config": {
            "chunk_strategy": "test_mode",
            "compressor": "blosc_fast",
            "cache_size_mb": 100
        },

        # Minimal geological config
        "max_layers": 10,
        "layer_type": "flat",
        "num_faults": 0,
        "include_salt": False,
        "include_closures": False,
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    return str(config_file)


def test_full_model_generation_with_zarr(test_config, tmp_path):
    """Test complete model generation using zarr backend."""
    from main import build_model

    # Run model generation
    output_folder = build_model(test_config, run_id="test_001", test_mode=1)

    # Verify output folder created
    assert os.path.exists(output_folder)

    # Verify zarr store was created and cleaned up
    # (temp store should be deleted after completion)
    temp_folders = list(Path("/tmp").glob("temp_folder__*"))
    # Should be cleaned up, but check recent ones

    # Verify outputs exist (depends on what gets copied to final folder)
    # This will depend on your workflow

    # Basic sanity check
    assert True  # Modify based on expected outputs


def test_zarr_store_structure(tmp_path):
    """Test zarr store has expected structure."""
    from synthoseis.storage import StorageClient
    import numpy as np

    store_path = tmp_path / "test.zarr"

    with StorageClient.open(str(store_path), mode="a") as client:
        # Create typical datasets
        client.create_dataset("depth_maps", np.ones((50, 50, 10)))
        client.create_dataset("geologic_age", np.ones((50, 50, 50), dtype=np.int32))
        client.create_dataset("lithology", np.ones((50, 50, 50), dtype=np.int32))

        # Verify datasets
        datasets = client.list_datasets()
        assert "depth_maps" in datasets
        assert "geologic_age" in datasets
        assert "lithology" in datasets

    # Verify zarr.json exists
    assert (store_path / "zarr.json").exists()

    # Verify dataset folders exist
    assert (store_path / "depth_maps").is_dir()
    assert (store_path / "geologic_age").is_dir()


def test_data_round_trip(tmp_path):
    """Test data integrity through write/read cycle."""
    from synthoseis.storage import StorageClient
    import numpy as np

    store_path = tmp_path / "roundtrip.zarr"
    original_data = np.random.random((100, 100, 50)).astype(np.float32)

    # Write
    with StorageClient.open(str(store_path), mode="a") as client:
        client.create_dataset("test", original_data, chunks=(32, 32, 32))

    # Read in new client session
    with StorageClient.open(str(store_path), mode="r") as client:
        retrieved_data = client.get_dataset("test")

    # Verify exact match
    np.testing.assert_array_equal(original_data, retrieved_data)
    assert original_data.dtype == retrieved_data.dtype
```

#### Task 3.3: Add Performance Benchmark (Optional)

**File:** `benchmarks/zarr_performance.py`

```python
import numpy as np
import time
import tempfile
import shutil
from synthoseis.storage import StorageClient


def benchmark_write_throughput():
    """Measure write speed for different array sizes."""
    sizes = [
        (50, 50, 50),
        (100, 100, 100),
        (200, 200, 200),
        (300, 300, 250),
    ]

    for shape in sizes:
        data = np.random.random(shape).astype(np.float32)
        size_gb = data.nbytes / 1e9

        store_path = tempfile.mkdtemp(suffix=".zarr")

        try:
            client = StorageClient.open(store_path, mode="a")

            start = time.time()
            client.create_dataset("test", data)
            elapsed = time.time() - start

            throughput = size_gb / elapsed
            print(f"Write {shape}: {throughput:.2f} GB/s ({elapsed:.3f}s)")

            client.close()
        finally:
            shutil.rmtree(store_path, ignore_errors=True)


def benchmark_compressors():
    """Compare different compressors."""
    from numcodecs import Blosc, Zstd, GZip

    data = np.random.random((300, 300, 250)).astype(np.float32)
    size_gb = data.nbytes / 1e9

    compressors = {
        "none": None,
        "blosc_lz4": Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE),
        "blosc_zstd": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
        "zstd": Zstd(level=5),
        "gzip": GZip(level=6),
    }

    for name, compressor in compressors.items():
        store_path = tempfile.mkdtemp(suffix=".zarr")

        try:
            client = StorageClient.open(store_path, mode="a")

            start = time.time()
            client.create_dataset("test", data, compressor=compressor)
            write_time = time.time() - start

            # Measure compressed size
            import os
            store_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, dirs, files in os.walk(store_path)
                for file in files
            )
            compression_ratio = data.nbytes / store_size

            print(f"{name:12s}: {write_time:.3f}s write, "
                  f"{compression_ratio:.2f}× compression")

            client.close()
        finally:
            shutil.rmtree(store_path, ignore_errors=True)


if __name__ == "__main__":
    print("=== Write Throughput ===")
    benchmark_write_throughput()

    print("\n=== Compressor Comparison ===")
    benchmark_compressors()
```

---

### Phase 4: Documentation (Priority: MEDIUM)

**Estimated Time:** 2-3 hours

#### Task 4.1: Create Storage README

**File:** `synthoseis/storage/README.md`

```markdown
# Synthoseis Storage Backend

## Overview

The Synthoseis storage backend provides a clean abstraction over Zarr v3 for persisting geological model data. It supports:

- Chunked storage for large-than-RAM arrays
- Configurable compression
- Dask integration for lazy loading
- Simple key-value API

## Quick Start

```python
from synthoseis.storage import StorageClient
import numpy as np

# Create or open a store
client = StorageClient.open("/path/to/store.zarr", mode="a")

# Write data
depth_maps = np.random.random((300, 300, 50))
client.create_dataset("depth_maps", depth_maps, chunks=(128, 128, 10))

# Read data
depth = client.get_dataset("depth_maps")

# List datasets
datasets = client.list_datasets()
print(f"Store contains: {datasets}")

# Close
client.close()
```

## API Reference

### StorageClient

#### `StorageClient.open(root_path, mode="a")`

Open or create a zarr store.

**Parameters:**
- `root_path` (str): Path to zarr store directory
- `mode` (str): Access mode - "r" (read), "w" (write/create), "a" (append, default)

**Returns:** StorageClient instance

#### `create_dataset(name, data=None, shape=None, chunks=None, dtype=None, compressor=None)`

Create a dataset in the store.

**Parameters:**
- `name` (str): Dataset name (unique identifier)
- `data` (ndarray, optional): Data to write (provide either data or shape)
- `shape` (tuple, optional): Shape for empty array
- `chunks` (tuple, optional): Chunk shape (default: auto)
- `dtype` (dtype, optional): Data type (default: inferred from data or float32)
- `compressor` (Codec, optional): Compression codec (default: blosc/zstd)

**Returns:** Created array data (if data provided) or None

#### `get_dataset(name, use_dask=False)`

Retrieve a dataset from the store.

**Parameters:**
- `name` (str): Dataset name
- `use_dask` (bool): Return dask array instead of numpy (default: False)

**Returns:** numpy ndarray or dask array

**Raises:** `DatasetNotFound` if dataset doesn't exist

#### `list_datasets(prefix="")`

List all datasets in the store.

**Parameters:**
- `prefix` (str, optional): Filter by name prefix

**Returns:** List of dataset names

#### `remove_dataset(name)`

Remove a dataset from the store.

**Parameters:**
- `name` (str): Dataset name to remove

## Configuration

### Chunking Strategies

Chunk size affects performance. Choose based on access patterns:

```python
# Horizon slicing (access xy planes frequently)
chunks = (128, 128, 64)

# Vertical operations (access z direction frequently)
chunks = (64, 64, 256)

# Balanced (general purpose)
chunks = (128, 128, 128)
```

### Compression

Available compressor presets:

| Preset | Ratio | Speed | Use Case |
|--------|-------|-------|----------|
| `"none"` | 1× | Fastest | Temp storage, SSD |
| `"blosc_fast"` | 2× | Very Fast | Development |
| `"blosc_default"` | 3-4× | Fast | **Recommended** |
| `"blosc_max"` | 4-5× | Medium | Archival |
| `"gzip"` | 3-4× | Slow | Compatibility |

Example:
```python
from numcodecs import Blosc

compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
client.create_dataset("data", array, compressor=compressor)
```

## Advanced Usage

### Dask Integration

For large arrays that don't fit in memory:

```python
import dask.array as da

# Read as dask array
dask_data = client.get_dataset("large_volume", use_dask=True)

# Lazy operations
result = dask_data * 2 + dask_data.mean()

# Compute when needed
final = result.compute()
```

### Context Manager

Automatically closes store:

```python
with StorageClient.open("/path/to/store.zarr") as client:
    data = client.get_dataset("depth_maps")
    # process data...
# Store closed automatically
```

## Performance Tips

1. **Chunk Size:** Target 1-10 MB per chunk
2. **Compression:** Use `blosc_default` for best balance
3. **Dask:** Use `use_dask=True` for arrays >1 GB
4. **SSDs:** Faster than HDD for random chunk access
5. **Parallel I/O:** Zarr supports concurrent reads

## Troubleshooting

**"Dataset not found" error:**
- Check dataset name spelling
- Use `list_datasets()` to see available datasets

**Memory errors:**
- Use `use_dask=True` for large arrays
- Reduce chunk cache size
- Process data in chunks

**Slow performance:**
- Adjust chunk size for your access pattern
- Use compression to reduce I/O
- Check disk speed (SSD recommended)

## Migration from HDF5

Synthoseis no longer supports HDF5 files. To use outputs from older versions:
- Re-run model generation with current version
- No conversion tool available (clean migration)

## Support

For issues or questions:
- GitHub: https://github.com/sede-open/synthoseis/issues
- Documentation: See `.agentic-docs/zarr-v3-migration-spec.md`
```

#### Task 4.2: Update Main README

**File:** `README.md`

Add section about storage backend:

```markdown
## Storage Backend

Synthoseis uses **Zarr v3** for data persistence, enabling:

- Processing models larger than available RAM
- Compressed storage (3-5× reduction)
- Cloud-native workflows (S3/GCS compatible)
- Dask integration for parallel processing

### Requirements

- Python >=3.12
- zarr >=3.1.3
- numpy >=2.0

### Storage Configuration

Configure chunking and compression in your config JSON:

```json
{
  "zarr_config": {
    "chunk_strategy": "3d_volume_balanced",
    "compressor": "blosc_default",
    "cache_size_mb": 500
  }
}
```

See `synthoseis/storage/README.md` for details.

### Important: No HDF5 Support

HDF5 files are no longer supported. Legacy models must be regenerated with the current version.
```

#### Task 4.3: Add Module Docstrings

**File:** `synthoseis/storage/zarr_backend.py` (or mdio_backend.py)

Enhance class/method docstrings:

```python
"""Zarr v3 storage backend for Synthoseis.

This module provides a clean abstraction over Zarr v3 for persisting
geological model data (depth maps, geological volumes, elastic properties,
seismic volumes, etc.).

Key Features:
- Chunked storage for processing large-than-RAM arrays
- Configurable compression codecs (blosc, zstd, gzip)
- Dask integration for lazy loading and parallel operations
- Simple key-value API for datasets

Example:
    >>> from synthoseis.storage import StorageClient
    >>> client = StorageClient.open("/tmp/model.zarr", mode="a")
    >>> client.create_dataset("depth", depth_array, chunks=(128, 128, 128))
    >>> depth = client.get_dataset("depth")
    >>> client.close()

See Also:
    - Zarr v3 documentation: https://zarr.readthedocs.io/
    - Storage README: synthoseis/storage/README.md
"""
```

---

### Phase 5: Final Validation (Priority: HIGH)

**Estimated Time:** 1-2 hours

#### Task 5.1: Run Full Test Suite

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=synthoseis.storage --cov-report=term-missing

# Target: >80% coverage for storage module
```

#### Task 5.2: Integration Test

```bash
# Run small model generation
uv run python main.py -c config/test_config.json -n 1 -t 1

# Verify:
# - Model completes successfully
# - Outputs created
# - Temp folder cleaned up
# - Memory usage reasonable
```

#### Task 5.3: Code Quality Checks

```bash
# Run ruff linting
uv run ruff check .

# Run ruff formatting check
uv run ruff format --check .

# Fix any issues
uv run ruff check --fix .
uv run ruff format .
```

---

## Deliverables Checklist

### Code Changes
- [ ] Remove `hdf_master` from Parameters.py
- [ ] Remove commented HDF5 code from Salt.py
- [ ] Remove `hdf_store` config read from Parameters.py
- [ ] Remove `multidimio[distributed]` from pyproject.toml
- [ ] Add chunking strategies to storage backend
- [ ] Add compression presets to storage backend
- [ ] Update Parameters class with zarr config
- [ ] Update example config with zarr_config section
- [ ] Update module storage calls to use configuration

### Testing
- [ ] Rename test_storage_mdio.py → test_zarr_backend.py
- [ ] Add 15+ unit tests for StorageClient
- [ ] Add integration test for full model generation
- [ ] Add performance benchmarks (optional)
- [ ] Achieve >80% test coverage for storage module
- [ ] All tests passing

### Documentation
- [ ] Create synthoseis/storage/README.md
- [ ] Update main README.md with zarr info
- [ ] Add module docstrings to storage backend
- [ ] Add migration notes
- [ ] Document configuration options

### Quality
- [ ] No HDF5 references in codebase (except docs)
- [ ] Ruff linting passes
- [ ] Ruff formatting passes
- [ ] No runtime errors
- [ ] Memory usage validated

---

## Git Workflow

### Commit Strategy

**Recommended commits:**

1. "Remove legacy HDF5 references from codebase"
   - Parameters.py: remove hdf_master, hdf_store
   - Salt.py: remove commented HDF5 code

2. "Remove unused MDIO dependency"
   - pyproject.toml: remove multidimio

3. "Add configurable chunking and compression to zarr backend"
   - storage/zarr_backend.py: add CHUNK_STRATEGIES, COMPRESSOR_PRESETS
   - storage/zarr_backend.py: update create_dataset() signature

4. "Add zarr configuration to Parameters and example config"
   - datagenerator/Parameters.py: add zarr_config attributes
   - config/example.json: add zarr_config section

5. "Expand storage backend tests"
   - Rename test file
   - Add comprehensive unit tests
   - Add integration tests

6. "Add storage backend documentation"
   - Create synthoseis/storage/README.md
   - Update main README.md
   - Add docstrings

7. "Complete Zarr v3 migration: HDF5 fully deprecated"
   - Final validation commit
   - Update CHANGELOG if exists

### Push and PR

```bash
# Stage and commit
git add .
git commit -m "Complete Zarr v3 migration: remove HDF5 remnants, enhance tests, add docs"

# Push to origin
git push -u origin claude/migrate-hdf5-storage-u9pgu

# Create PR (if using GitHub CLI)
gh pr create \
  --title "Complete Zarr v3 migration and HDF5 deprecation" \
  --body "$(cat <<EOF
## Summary
Finalizes the migration from HDF5 to Zarr v3 storage backend:

- ✅ Removed all legacy HDF5 references
- ✅ Removed unused MDIO dependency (using pure Zarr v3)
- ✅ Added configurable chunking strategies
- ✅ Added compression codec presets
- ✅ Expanded test coverage (15+ unit tests, integration tests)
- ✅ Comprehensive documentation

## Testing
- All tests passing
- Integration test with full model generation
- Coverage >80% for storage module

## Breaking Changes
- HDF5 files no longer supported (as planned)
- Config schema extended (backward compatible)

## Migration Guide
See .agentic-docs/zarr-v3-migration-spec.md

Closes #XX (if issue exists)
EOF
)"
```

---

## Timeline Summary

| Phase | Tasks | Time | Priority |
|-------|-------|------|----------|
| **Phase 1: Cleanup** | Remove HDF5 refs, MDIO dep | 1h | HIGH |
| **Phase 2: Config** | Chunking, compression config | 2-3h | MEDIUM |
| **Phase 3: Testing** | Unit tests, integration tests | 3-4h | HIGH |
| **Phase 4: Docs** | README, docstrings | 2-3h | MEDIUM |
| **Phase 5: Validation** | Test suite, QA checks | 1-2h | HIGH |
| **Total** | | **9-13h** | |

**Recommended Order:**
1. Phase 1 (cleanup) - Quick wins
2. Phase 3 (testing) - Ensure stability
3. Phase 2 (config) - Add features
4. Phase 4 (docs) - Document changes
5. Phase 5 (validation) - Final QA

---

## Success Criteria

**Must Have (MVP):**
- ✅ Zero HDF5 references in code
- ✅ MDIO dependency removed
- ✅ All tests passing
- ✅ Basic documentation
- ✅ Model generation works

**Should Have:**
- ✅ Configurable chunking
- ✅ Configurable compression
- ✅ >80% test coverage
- ✅ Comprehensive docs

**Nice to Have:**
- ⚠️ Performance benchmarks
- ⚠️ Memory profiling
- ⚠️ Cloud storage examples

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing workflows | Low | High | Thorough testing, integration tests |
| Performance regression | Low | Medium | Benchmarks, profiling |
| Configuration complexity | Medium | Low | Good defaults, documentation |
| Test coverage gaps | Medium | Medium | Comprehensive test suite |
| Dependency issues | Low | Low | Zarr v3 is stable, well-supported |

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Confirm branch strategy** (continue on current or create new)
3. **Start with Phase 1** (quick cleanup tasks)
4. **Iterate through phases** with commits
5. **Create PR** when complete
6. **Merge** after review

---

**Questions? Decisions Needed:**

1. Keep branch name `claude/migrate-hdf5-storage-u9pgu` or rename?
2. Rename `mdio_backend.py` → `zarr_backend.py` or keep current name?
3. Default compression: enable or disable? (recommend enable with blosc_default)
4. Performance benchmarks: include or defer?
5. Integration test: full model or minimal config?

**Ready to proceed!** 🚀
