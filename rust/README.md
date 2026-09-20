# Synthoseis Rust workspace

Skeleton for the Rust rewrite of [synthoseis](https://github.com/gavargas22/synthoseis).
The Python tree at the repository root stays intact; this workspace lives under `rust/`.

## Crates

| Crate | Role |
|-------|------|
| `synthoseis` | CLI binary (`synthoseis --help`, `synthoseis run`) |
| `synthoseis-core` | Config, RNG, job partition, single-worker path stubs |
| `synthoseis-io` | **Honest MDIO** (Zarr v2) create / write / read |
| `synthoseis-py` | **maturin / PyO3** Python extension (`synthoseis_mdio`) over `synthoseis-io` |
| `synthoseis-geo` | Geology / horizons kernels (plane fit, thickness clip, label fill) |
| `synthoseis-closures` | Closure / trap kernels (label sizes, flood-fill, fluid masks) |
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

**Preferred Python entry (maturin / PyO3):** `rust/synthoseis-py` builds the
`synthoseis_mdio` extension so Python can create / write / read without shelling out to
`cargo`. See `rust/synthoseis-py/README.md` and `tests/test_mdio_bindings.py`.

**Python `mdio` interop gate (#8):** Rust creates/writes the MDIO store; Python
`multidimio` (`import mdio`) remains open/interop only (not the long-term writer).
`tests/test_mdio_rust_interop.py` builds a CLI smoke store, calls `mdio.open_mdio`
(must not raise), and asserts `chunked_012` / `live_mask` via zarr. Consolidated
`.zmetadata` + stub `chunked_012_trace_headers` also let multidimio 0.9.x `MDIOReader`
open the same hierarchy.

**Known gaps vs full mdio-python:** no Blosc/ZFP compressors, stub (not SEG-Y-faithful)
trace headers, no full SEG-Y text/binary header fidelity, no cloud object-store backends.
mdio 1.x `open_mdio` does not yet flatten nested create_empty arrays into xarray
data_vars — that flattening is a later slice.

## Parity harness

`synthoseis-core::parity` compares **label volumes** and **angle-stack volumes**
(not bit-identical full seismic):

| Volume | Metrics | Default tolerances |
|--------|---------|-------------------|
| Labels (u8) | macro IoU (skip unset=255), per-voxel agreement | IoU ≥ 0.99, agreement ≥ 0.99 |
| Angle stacks (f32) | MAE, max-abs | MAE ≤ 1e-3, max-abs ≤ 5e-3 |

Fixed-seed **8³** fixtures: `tests/fixtures/parity_cubes_8.json`
(labels RLE-compressed; angle stacks synthesized from `0.1*i+0.05*j+0.02*k`).
Regenerate with `python tests/fixtures/generate_parity_cubes.py`.

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

### Python extension (maturin)

Prerequisites: Rust stable + Python ≥ 3.12 + maturin.

```bash
# from repo root
maturin develop --manifest-path rust/synthoseis-py/Cargo.toml
pytest tests/test_mdio_bindings.py -q
```

CI: `.github/workflows/rust-ci.yml` runs `cargo check` + `cargo test` in `rust/` on push/PR.
`.github/workflows/maturin-bindings.yml` builds the extension and runs the binding pytest.

## Next ports

**Landed**

- Parity harness wired (label IoU / agreement + angle-stack MAE / max-abs)
- First geo kernels in `synthoseis-geo`: `fit_plane_lsq`, `eval_plane`,
  `rotate_point`, `enforce_nonnegative_thicknesses` (from
  `datagenerator/Horizons.py`), plus `fill_layer_labels` feeding the harness
- First closure kernels in `synthoseis-closures` (from
  `datagenerator/_closures_vectorised.py` + `Closures.py`):
  `bincount_label_sizes`, `relabel_consecutive`, `filter_labels_by_min_voxels`,
  `closure_size_filter_sizes`, `parse_closure_codes`, `assign_fluid_types`,
  `get_top_of_closure`, `bbox_for_label_and_fault`, `flood_fill_heap_2d`
  — golden fixtures in `tests/fixtures/closure_cubes_8.json`

**Still out of scope / next**

- Seismic convolution / Zoeppritz / bandpass / noise (`synthoseis-seismic`)
- RPM depth trends (`synthoseis-rpm`)
- Full geology stack / faults / GPU
- Replacing Parameters Python zarr store end-to-end
- Multi-worker / cloud job partition
- Publishing wheels
