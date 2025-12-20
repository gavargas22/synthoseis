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
    import zarr
except Exception:  # pragma: no cover - mdio may not be installed in test env
    mdio = None
    zarr = None


class DatasetNotFound(Exception):
    pass


class StorageClient:
    """Minimal storage client backed by MDIO (zarr).

    Usage:
        client = StorageClient.open("/path/to/store", mode="a")
        client.create_dataset("vp", data)
        arr = client.get_dataset("vp")
    """

    def __init__(self, root_path: str, mode: str = "a", default_chunks: Optional[Tuple[int, ...]] = None) -> None:
        if zarr is None:
            raise RuntimeError("zarr is required for StorageClient but is not installed")
        self.root_path = os.path.abspath(root_path)
        self.mode = mode
        self.default_chunks = default_chunks or (128, 128, 128)
        self.store = zarr.open(self.root_path, mode=mode)

    @classmethod
    def open(cls, root_path: str, mode: str = "a", default_chunks: Optional[Tuple[int, ...]] = None) -> "StorageClient":
        return cls(root_path, mode=mode, default_chunks=default_chunks)

    def create_dataset(
        self,
        name: str,
        data: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        compressor: Optional[Any] = None
    ) -> None:
        # mdio exposes a zarr-like API via mdio.write or mdio.create
        if data is not None and shape is not None:
            raise ValueError("Cannot specify both 'data' and 'shape'")
        if data is None and shape is None:
            raise ValueError("Must specify either 'data' or 'shape'")
        
        if data is not None:
            if dtype is None:
                dtype = data.dtype
            actual_shape = data.shape
        else:
            if dtype is None:
                dtype = np.float32  # default dtype
            actual_shape = shape
        
        # Always overwrite existing dataset for now
        if name in self.store:
            del self.store[name]
        # For scalability, default to chunking if not specified
        if chunks is None:
            chunks = self.default_chunks[:len(actual_shape)] if len(self.default_chunks) >= len(actual_shape) else self.default_chunks + (min(128, actual_shape[len(self.default_chunks)]),) * (len(actual_shape) - len(self.default_chunks))
            chunks = tuple(min(c, s) for c, s in zip(chunks, actual_shape)) + tuple(actual_shape[len(chunks):])
        
        if data is not None:
            # Create with data
            self.store.create_array(
                name, data=data, chunks=chunks, compressor=compressor
            )
            return self.store[name][...]
        else:
            # Create empty array with shape
            self.store.create_array(
                name, shape=actual_shape, dtype=dtype,
                chunks=chunks, compressor=compressor
            )
            return None

    def get_dataset(self, name: str, lazy: bool = False, use_dask: bool = False):
        if name not in self.store:
            raise DatasetNotFound(name)
    arr = self.store[name]
    if use_dask:
        try:
            import dask.array as da
            return da.from_zarr(self.store.store, component=name)
        except ImportError:
            return np.asarray(arr)
    elif lazy:
        return arr
    else:
        return np.asarray(arr)    def list_datasets(self, prefix: str = ""):
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