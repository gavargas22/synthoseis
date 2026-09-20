"""Rust-backed MDIO round-trip via maturin / PyO3 (`synthoseis_mdio`).

Preferred Python entry for create → write → read (no cargo subprocess).
Requires the extension built with maturin; see rust/synthoseis-py/README.md
and .github/workflows/maturin-bindings.yml.

Skipped automatically when the extension is not installed so default
`pytest -m "not slow"` / pure-Python CI stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

synthoseis_mdio = pytest.importorskip("synthoseis_mdio")


def test_create_write_read_round_trip(tmp_path: Path) -> None:
    store_path = tmp_path / "binding_smoke.mdio"
    shape = (2, 2, 4)
    n = shape[0] * shape[1] * shape[2]
    data = [0.5 * i for i in range(n)]

    store = synthoseis_mdio.create_empty(str(store_path), shape=shape, seed=7)
    assert store.shape == shape
    assert Path(store.path).resolve() == store_path.resolve() or store.path.endswith(
        str(store_path)
    )

    store.write_volume(data)
    back = store.read_volume()
    assert len(back) == n
    assert back == pytest.approx(data)
    mask = store.read_live_mask()
    assert mask == [True, True, True, True]

    # Hierarchy on disk (honest MDIO / Zarr v2)
    assert (store_path / ".zgroup").is_file()
    assert (store_path / ".zmetadata").is_file()
    assert (store_path / "data" / "chunked_012" / ".zarray").is_file()
    assert (store_path / "metadata" / "live_mask" / ".zarray").is_file()

    opened = synthoseis_mdio.open_store(str(store_path))
    assert opened.shape == shape
    assert synthoseis_mdio.read_volume(str(store_path)) == pytest.approx(data)
    assert synthoseis_mdio.read_live_mask(str(store_path)) == [True] * 4


def test_write_chunk_then_read(tmp_path: Path) -> None:
    store_path = tmp_path / "chunk.mdio"
    store = synthoseis_mdio.MdioStore.create_empty(
        str(store_path), shape=(2, 2, 4), chunks=(2, 2, 4)
    )
    chunk = [float(i) for i in range(16)]
    store.write_chunk((0, 0, 0), chunk)
    assert store.read_volume() == pytest.approx(chunk)
