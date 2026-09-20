"""Rust MDIO create/write → Python mdio open interop smoke.

Rust owns create/write (synthoseis CLI / synthoseis-io). Python `mdio`
(multidimio) is the open/interop gate only — not the long-term writer.
Preferred Python create/write/read entry is the maturin extension (`synthoseis_mdio`; see `tests/test_mdio_bindings.py`).

The Rust store is the mdio-python *create_empty*-shaped Zarr v2 hierarchy
(`metadata/` + `data/`, `live_mask`, `chunked_012`). multidimio 1.x
`open_mdio` is xarray-flat and returns root attrs without nested arrays;
we still call it as the public open API, then assert the hierarchy and
payload via zarr (and via `MDIOReader` when that symbol exists, e.g. 0.9.x).

Requires a Rust toolchain (`cargo`). Marked `slow` so default CI
(`pytest -m "not slow"`) stays fast; see `.github/workflows/mdio-interop.yml`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import zarr

mdio = pytest.importorskip("mdio")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_DIR = REPO_ROOT / "rust"


def _cargo_available() -> bool:
    return shutil.which("cargo") is not None and (RUST_DIR / "Cargo.toml").is_file()


def _create_rust_mdio_store(store: Path) -> None:
    env = os.environ.copy()
    env.setdefault("CARGO_TERM_COLOR", "never")
    cmd = [
        "cargo",
        "run",
        "-p",
        "synthoseis",
        "--quiet",
        "--",
        "run",
        "--store",
        str(store),
    ]
    subprocess.run(cmd, cwd=RUST_DIR, env=env, check=True)


@pytest.mark.slow
@pytest.mark.skipif(not _cargo_available(), reason="cargo / rust/ workspace not available")
def test_rust_mdio_store_opens_with_python_mdio(tmp_path: Path) -> None:
    store = tmp_path / "interop_smoke.mdio"
    _create_rust_mdio_store(store)

    assert store.is_dir()
    assert (store / ".zmetadata").is_file(), "Rust must write consolidated .zmetadata for mdio open"
    assert (store / "data" / "chunked_012" / ".zarray").is_file()
    assert (store / "metadata" / "live_mask" / ".zarray").is_file()
    assert (store / "metadata" / "chunked_012_trace_headers" / ".zarray").is_file()

    # Public 1.x open API — must not raise. Nested create_empty arrays are not
    # flattened into data_vars; root attrs still prove mdio accepted the store.
    open_mdio = getattr(mdio, "open_mdio", None)
    if open_mdio is not None:
        ds = open_mdio(str(store))
        assert ds is not None
        attrs = dict(ds.attrs)
        assert attrs.get("synthoseis_mdio") in (True, "true", 1)
        assert int(attrs.get("trace_count", -1)) == 4
        assert attrs.get("api_version")

    # Hierarchy + payload gate (works with zarr 3 against the Rust v2 layout).
    root = zarr.open_group(str(store), mode="r")
    samples = root["data"]["chunked_012"]
    live = root["metadata"]["live_mask"]
    assert tuple(samples.shape) == (2, 2, 4)
    assert samples.dtype == np.dtype("float32")
    assert tuple(live.shape) == (2, 2)
    live_arr = np.asarray(live[:])
    assert live_arr.dtype == np.bool_ or live_arr.dtype == bool
    assert bool(live_arr.all())

    volume = np.asarray(samples[:])
    expected = (np.arange(16, dtype=np.float32) * 0.5).reshape(2, 2, 4)
    np.testing.assert_allclose(volume, expected, rtol=0, atol=0)

    # Stronger path when MDIOReader is available (multidimio 0.9.x).
    reader_cls = getattr(mdio, "MDIOReader", None)
    if reader_cls is not None:
        reader = reader_cls(str(store))
        assert tuple(reader.shape) == (2, 2, 4)
        assert int(reader.trace_count) == 4
        assert bool(np.asarray(reader.live_mask).all())
        np.testing.assert_allclose(np.asarray(reader[:]), expected, rtol=0, atol=0)
