Modernization plan: migrate synthoseis storage to MDIO (zarr-backed)

Overview
--------
Goal: Replace the project's legacy HDF5 / NumPy file I/O with a modern MDIO-based backend using zarr. The migration is allowed to change the public API (breaking changes OK) but should improve ergonomics, performance, and reproducibility.

High-level approach
-------------------
1. Introduce a new storage package `synthoseis.storage` with a clear MDIO-based client. This will be the only supported storage layer.
2. Replace all uses of PyTables/h5py/np.save/np.savez/pickle-based storage across `datagenerator` and other modules to use `synthoseis.storage` APIs.
3. Remove old HDF initialization helpers (`hdf_setup`, `hdf_init`, etc.) and provide new initialization and dataset APIs.
4. Add tests and docs showing how to read/write data with MDIO.

Deliverables
------------
- `pyproject.toml` updated to include `mdio` and `zarr` dependencies
- `synthoseis/storage/__init__.py` and `synthoseis/storage/mdio_backend.py` implementing the new API
- Reworked `datagenerator.Parameters` to use the new storage client (e.g., `storage = StorageClient(path)`)
- Updated modules under `datagenerator/` to use the new APIs (calls to `.h5file`, `np.save`, and `tables` removed)
- A smoke test `tests/test_mdio_storage.py` demonstrating read/write and a simple model generation run
- `.agentic-docs/modernization-plan.md` (this file)

API contract (recommended)
--------------------------
Design a minimal StorageClient that supports the following operations:

- StorageClient.open(path, mode="a") -> StorageClient
  - Opens or creates a zarr store via MDIO and returns a client
- client.create_dataset(name: str, data: np.ndarray, chunks: tuple | None = None, dtype=None, compressor=None)
  - Creates or overwrites a dataset
- client.get_dataset(name: str) -> numpy.ndarray (or dask array)
  - Read dataset (supports slicing)
- client.list_datasets(prefix: str = "") -> list[str]
- client.remove_dataset(name: str)
- client.close()

Additional features (nice-to-have):
- Lazy/dask-backed reads for large volumes
- Metadata support: client.attrs (dict-like)
- Simple concurrency guidance for parallel generation runs

Edge cases and behavior
-----------------------
- Missing dataset: raise a clear `KeyError` or custom `DatasetNotFound`
- Partial writes/crashes: write to a temporary store then atomically move/rename if possible, or write a `_complete` flag
- Large volumes: expose chunking configuration and recommend chunk layouts in code
- Platform specifics: zarr works across OS; MDIO may require additional dependencies (xarray, zarr) — list them in `pyproject.toml`.

Testing strategy
----------------
- Unit tests for `StorageClient` covering create/read/list/remove
- Integration smoke test that runs a small model generation using a temp MDIO store and asserts outputs are present
- CI: add test step installing deps and running tests (optional for now)

Rollout plan
------------
1. Add dependencies and storage module (this PR)
2. Update `Parameters` to use StorageClient (small changes)
3. Incrementally replace file writes (prefer replacing central APIs first so fewer edits downstream)
4. Run tests and fix runtime issues
5. Remove old HDF/Tables dependencies in follow-up PR once everything green

Assumptions
-----------
- The user wants to accept breaking API changes.
- MDIO and zarr are available for Python >=3.9.
- It's acceptable to introduce dask/xarray optional dependencies for performance.

Next steps (what I'll do now)
----------------------------
- Implement `synthoseis/storage/mdio_backend.py` and a minimal `StorageClient` API.
- Update `pyproject.toml` to add `mdio` and `zarr` to `dependencies`.


