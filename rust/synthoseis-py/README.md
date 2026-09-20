# synthoseis-py (`synthoseis_mdio`)

Minimal **maturin / PyO3** bindings over [`synthoseis-io`](../synthoseis-io) MDIO
create / write / read. Python is the public package surface; Rust owns the store.

## Prerequisites

- **Rust** stable (`rustup`)
- **Python ≥ 3.12**
- [maturin](https://www.maturin.rs/) (`pip install maturin` / `uv tool install maturin`)

## Build

From the repository root (recommended):

```bash
maturin develop --manifest-path rust/synthoseis-py/Cargo.toml
```

Or from this directory:

```bash
cd rust/synthoseis-py && maturin develop
```

With the repo uv env:

```bash
uv sync --group dev
uv pip install maturin
uv run maturin develop --manifest-path rust/synthoseis-py/Cargo.toml
```

## Python API

```python
import synthoseis_mdio as mdio_rs

store = mdio_rs.create_empty("/tmp/demo.mdio", shape=(2, 2, 4))
store.write_volume([0.5 * i for i in range(16)])
vol = store.read_volume()
mask = store.read_live_mask()  # list[bool]
assert store.shape == (2, 2, 4)
```

| Surface | Notes |
|---------|--------|
| `MdioStore.create_empty` / `open` | Class handle |
| `MdioStore.write_volume` / `write_chunk` | float32 samples |
| `MdioStore.read_volume` / `read_live_mask` | round-trip |
| `create_empty`, `open_store`, `write_volume`, `read_volume`, `read_live_mask` | Module helpers |

This is the **preferred Python entry** for Rust MDIO. The `#8` interop workflow
(`tests/test_mdio_rust_interop.py`: cargo CLI + Python `mdio` open) remains for
cross-language open checks.
