"""xarray/zarr output writer for synthoseis final volumes.

Every final pipeline output is written as a zarr store that conforms to
the xarray/CF-conventions layout, readable with xarray.open_zarr(path).
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def write_volume_to_zarr(
    arr: np.ndarray,
    path: str,
    name: str = "data",
    dims: tuple[str, ...] = ("inline", "crossline", "time"),
    coords: dict | None = None,
    attrs: dict | None = None,
    chunks: dict | None = None,
) -> None:
    """Write a numpy array as a CF-convention xarray zarr store.

    Parameters
    ----------
    arr : np.ndarray
        Array to write (any shape, any dtype).
    path : str
        Output zarr path (directory). Created if absent.
    name : str
        Variable name inside the dataset (default "data").
    dims : tuple[str, ...]
        Dimension names, length must match arr.ndim.
    coords : dict | None
        Coordinate arrays keyed by dim name. Optional.
    attrs : dict | None
        Global dataset attributes (e.g. angle_deg, sample_rate_ms).
    chunks : dict | None
        Chunk sizes keyed by dim name. None = xarray auto-chunking.
    """
    da = xr.DataArray(arr, dims=dims, coords=coords or {}, attrs=attrs or {})
    ds = da.to_dataset(name=name)
    encoding = None
    if chunks:
        encoding = {name: {"chunks": [chunks.get(d, -1) for d in dims]}}
    ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)


def read_volume_from_zarr(path: str, name: str = "data") -> np.ndarray:
    """Read a zarr volume back as a numpy array.

    Parameters
    ----------
    path : str
        Path to the zarr store written by write_volume_to_zarr.
    name : str
        Variable name inside the dataset.
    """
    ds = xr.open_zarr(path)
    return ds[name].values
