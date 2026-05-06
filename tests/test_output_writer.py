"""Round-trip tests for output_writer.write_volume_to_zarr / read_volume_from_zarr."""
import numpy as np
import pytest
import tempfile, os
import xarray as xr
from datagenerator.output_writer import write_volume_to_zarr, read_volume_from_zarr


def test_roundtrip_values(tmp_path):
    arr = np.random.rand(10, 10, 20).astype("float32")
    path = str(tmp_path / "test.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="amplitude")
    np.testing.assert_array_equal(result, arr)


def test_roundtrip_attrs(tmp_path):
    arr = np.zeros((5, 5, 10), dtype="float32")
    path = str(tmp_path / "test_attrs.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"),
                         attrs={"angle_deg": 15, "sample_rate_ms": 4})
    ds = xr.open_zarr(path)
    assert ds.attrs.get("angle_deg") == 15 or \
           ds["amplitude"].attrs.get("angle_deg") == 15


def test_roundtrip_label_volume(tmp_path):
    arr = np.zeros((8, 8, 16), dtype="uint8")
    arr[2:5, 2:5, 4:8] = 1
    path = str(tmp_path / "labels.zarr")
    write_volume_to_zarr(arr, path, name="label",
                         dims=("inline", "crossline", "time"))
    result = read_volume_from_zarr(path, name="label")
    np.testing.assert_array_equal(result, arr)


def test_open_zarr_returns_dataset(tmp_path):
    arr = np.ones((4, 4, 8), dtype="float32")
    path = str(tmp_path / "ds.zarr")
    write_volume_to_zarr(arr, path, name="amplitude",
                         dims=("inline", "crossline", "time"))
    ds = xr.open_zarr(path)
    assert isinstance(ds, xr.Dataset)
    assert "amplitude" in ds
