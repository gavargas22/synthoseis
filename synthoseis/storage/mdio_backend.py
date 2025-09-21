"""MDIO-based storage backend for synthoseis

This provides a minimal StorageClient that wraps mdio (zarr) operations.
The implementation favors clarity and a small API surface.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple
import numpy as np

try:
    import mdio
except Exception:  # pragma: no cover - mdio may not be installed in test env
    mdio = None


class DatasetNotFound(Exception):
    pass


class StorageClient:
    """Minimal storage client backed by MDIO (zarr).

    Usage:
        client = StorageClient.open("/path/to/store", mode="a")
        client.create_dataset("vp", data)
        arr = client.get_dataset("vp")
    """

    def __init__(self, root_path: str, mode: str = "a") -> None:
        if mdio is None:
            raise RuntimeError("mdio is required for StorageClient but is not installed")
        self.root_path = os.path.abspath(root_path)
        self.mode = mode
        self.store = mdio.open(self.root_path, mode=mode)

    @classmethod
    def open(cls, root_path: str, mode: str = "a") -> "StorageClient":
        return cls(root_path, mode=mode)

    def create_dataset(self, name: str, data: np.ndarray, chunks: Optional[Tuple[int, ...]] = None, dtype: Optional[Any] = None, compressor: Optional[Any] = None) -> None:
        # mdio exposes a zarr-like API via mdio.write or mdio.create
        if dtype is None:
            dtype = data.dtype
        # Always overwrite existing dataset for now
        if name in self.store:
            del self.store[name]
        # For scalability, default to chunking if not specified
        if chunks is None:
            # Default chunking: reasonable size, e.g., 128x128x128 for 3D
            chunks = tuple(min(128, s) for s in data.shape)
        # mdio expects a path-like key
        self.store.create(name, data=data, chunks=chunks, dtype=dtype, compressor=compressor)

    def get_dataset(self, name: str, use_dask: bool = False):
        if name not in self.store:
            raise DatasetNotFound(name)
        arr = self.store[name]
        if use_dask:
            try:
                import dask.array as da
                return da.from_array(arr, chunks=arr.chunks)
            except ImportError:
                pass  # fall back to numpy
        return arr[...]

    def list_datasets(self, prefix: str = ""):
        return [k for k in list(self.store.keys()) if k.startswith(prefix)]

    def remove_dataset(self, name: str):
        if name in self.store:
            del self.store[name]

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass

    # convenience context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()