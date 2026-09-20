# Synthoseis Rust workspace

Skeleton for the Rust rewrite of [synthoseis](https://github.com/gavargas22/synthoseis).
The Python tree at the repository root stays intact; this workspace lives under `rust/`.

## Crates

| Crate | Role |
|-------|------|
| `synthoseis` | CLI binary (`synthoseis --help`, `synthoseis run`) |
| `synthoseis-core` | Config, RNG, job partition, single-worker path stubs |
| `synthoseis-io` | **Honest MDIO** (Zarr v2) create / write / read |
| `synthoseis-geo` | Future geology / horizons port |
| `synthoseis-closures` | Future closure / trap geometry port |
| `synthoseis-seismic` | Future seismic modeling port |
| `synthoseis-rpm` | Future rock-physics model port |
| `synthoseis-gpu` | Future GPU acceleration port |

## I/O: honest MDIO (Zarr v2)

Working volumes and deliverables use a real **MDIO-shaped** on-disk store.

There is no seismic MDIO crate on crates.io (`mdio` is Ethernet PHY — do not use it).
MDIO ([mdio.dev](https://mdio.dev) / mdio-python) is Zarr-backed.

`synthoseis-io` writes **Zarr format 2** to match mdio-python's `create_empty`:

```text
<root>/
  .zgroup  .zattrs          # name, api_version, created, dimension, digi, seed, units, stats
  metadata/
    .zgroup  .zattrs        # text_header / binary_header stubs
    live_mask/              # bool array, spatial shape
  data/
    .zgroup
    chunked_012/            # float32 samples, chunked
```

Public API (see `synthoseis-io`):

- `MdioStore::create_empty(path, &CreateConfig)` — empty MDIO hierarchy (dims / dtype float32 / chunks)
- `MdioStore::write_volume` / `write_chunk` — float32 samples; volume also updates `live_mask`
- `MdioStore::read_volume` / `read_live_mask` / `open` — round-trip

**Zarr v2 vs v3 / `zarrs`:** mdio-python commonly writes Zarr **v2**. We emit v2 JSON + raw
little-endian chunks directly (no Blosc) so the hierarchy stays recognizable. The maintained
`zarrs` crate is V3-first with a high MSRV; swapping the backend to `zarrs` is a follow-up.

**Python `mdio` interop gate:** Rust creates/writes the MDIO store; Python
`multidimio` (`import mdio`) is open/interop only (not the long-term writer). Maturin/PyO3
comes later. `tests/test_mdio_rust_interop.py` builds this CLI smoke store, calls
`mdio.open_mdio` (must not raise), and asserts `chunked_012` / `live_mask` via zarr.
Consolidated `.zmetadata` + stub `chunked_012_trace_headers` also let multidimio 0.9.x
`MDIOReader` open the same hierarchy.

**Known gaps vs full mdio-python:** no Blosc/ZFP compressors, stub (not SEG-Y-faithful)
trace headers, no full SEG-Y text/binary header fidelity, no cloud object-store backends.
mdio 1.x `open_mdio` does not yet flatten nested create_empty arrays into xarray
data_vars — that flattening is a later slice.

## Parity harness

`synthoseis-core` includes a fixed-seed golden comparison **placeholder**.
Python baseline vs Rust MAE/IoU metrics come later — do not treat the stub as numeric parity.

## Single-worker path

CLI / `synthoseis-core` stubs a **one local worker** run path. No cloud orchestration yet.

## Develop

```bash
cd rust
cargo check
cargo test
cargo run -p synthoseis -- --help
cargo run -p synthoseis -- run --store /tmp/smoke.mdio
```

CI: `.github/workflows/rust-ci.yml` runs `cargo check` + `cargo test` in `rust/` on push/PR.

## Next ports (out of scope here)

- Geology / closures / seismic / RPM algorithms
- Python MDIOReader parity harness wiring
- Multi-worker / cloud job partition
