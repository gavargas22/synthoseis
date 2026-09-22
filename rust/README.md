# Synthoseis Rust workspace

Skeleton for the Rust rewrite of [synthoseis](https://github.com/gavargas22/synthoseis).
The Python tree at the repository root stays intact; this workspace lives under `rust/`.

## Crates

| Crate | Role |
|-------|------|
| `synthoseis` | CLI binary (`synthoseis --help`, `synthoseis run`) |
| `synthoseis-core` | Config, RNG, **multi-worker job partition**, parity harness, e2e pipeline |
| `synthoseis-io` | **Honest MDIO** (Zarr v2) create / write / read |
| `synthoseis-py` | **maturin / PyO3** Python extension (`synthoseis_mdio`) over `synthoseis-io` |
| `synthoseis-geo` | Geology / horizons kernels (plane fit, thickness clip, label fill) |
| `synthoseis-closures` | Closure / trap kernels (label sizes, flood-fill, fluid masks) |
| `synthoseis-seismic` | Seismic kernels (Zoeppritz RFC, wavelets, SNR) |
| `synthoseis-rpm` | RPM depth-trend kernels (example + Tagilsk) |
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
- `MdioStore::write_labels_u8` / `read_labels_u8` — uint8 label deliverable under `data/labels`
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

## Single-worker e2e path

**Landed:** one local worker runs a **tiny 8³ cube** end-to-end:

```text
MDIO create → geo (horizons/labels) → closures (relabel/filter)
  → RPM (elastic depth trends) → seismic (Zoeppritz + Ricker)
  → MDIO write (data/chunked_012 = angle stack, data/labels = u8)
  → parity (second deterministic pass + MDIO round-trip)
```

Parity = **labels + angle stacks** (IoU / agreement / MAE / max-abs), not bit-identical
full seismic. CPU only — **no** GPU. Local multi-worker partition is landed (below);
cloud execution is still later.

```bash
cd rust
cargo run -p synthoseis -- run --e2e --store /tmp/e2e.mdio
```

`synthoseis-core::pipeline::{generate_tiny_cube, run_e2e}` is the library entry;
CLI `--e2e` is the CI-friendly smoke; `--chunked` selects the fused memory-bounded path.


## Multi-worker job partition (local)

**Landed:** `JobPartition` / `partition_jobs` / `JobPartitionPlan` (serde JSON) +
`MultiWorkerRunner` in `synthoseis-core`.

- Contiguous chunks over `job_ids` `0..inline×crossline` across `--workers N`
- Union covers all jobs with no overlap; when `jobs < workers`, empty worker
  slots are **kept** (not dropped) for stable cloud handoff
- Placeholder mode fans out locally via `std::thread::scope`
- `--partition-plan path.json` writes a cloud-ready plan artifact
- Non-chunked `--e2e --workers N` still runs the full tiny cube once; use
  `--e2e --chunked --workers N` for strip-stitch (below)

```bash
cd rust
cargo run -p synthoseis -- run --workers 4
cargo run -p synthoseis -- run --workers 4 --partition-plan /tmp/plan.json
cargo run -p synthoseis -- run --e2e --workers 4 --store /tmp/e2e.mdio
```


## Memory-bounded chunked e2e (single-worker)

**Landed:** fused elastic → Zoeppritz RFC → Ricker wavelet per spatial tile so the
working set stays ≈ chunk (plus O(ni×nj) horizon maps and the u8 label deliverable).
MDIO create/write uses **sub-volume** chunk shapes (never full-array `[ni,nj,nk]` when
the grid allows). Deliverables remain labels + angle stacks only.

```bash
cd rust
# Default e2e still works (8³); MDIO now uses sub-volume chunks (e.g. 4×4×8).
cargo run -p synthoseis -- run --e2e --store /tmp/e2e.mdio

# Explicit fused chunked path + chunk sizes (strip-friendly keys for later multi-worker).
cargo run -p synthoseis -- run --e2e --chunked --chunk-i 4 --chunk-j 4 --store /tmp/e2e-chunked.mdio
```

Library entry points: `synthoseis_core::pipeline_stream::{generate_chunked, run_e2e_chunked,
run_e2e_streaming, resolve_chunk_shape}` plus
`synthoseis_core::run_e2e_streaming_overlapped`.

For single-worker stores, `--overlap` uses a one-deep `std::sync::mpsc` writer
thread so tile N+1 is fused while tile N flushes, with no async runtime:

```bash
cargo run -p synthoseis -- run --e2e --chunked --overlap \
  --store /tmp/e2e-overlap.mdio
```

**Arbitrary size** is now a time/disk bound for the Rust path, not a RAM bound for
elastic/RFC/stack temps. The Python generator is still wasteful and untouched.

## Strip-stitch multi-worker e2e (local)

**Landed:** wire `partition_inline_strips` / `JobPartition` onto the chunked fused
path so N local workers each own contiguous **inline strips** snapped to MDIO
`chunk_i`, fuse-generate their tiles, and `write_chunk` / `write_labels_chunk`
into **one shared store** (no overlapping chunk keys). After join: finalize +
full-volume parity vs a single-worker chunked reference.

```bash
cd rust
# workers=1 stays today's chunked/single path
cargo run -p synthoseis -- run --e2e --chunked --workers 1 --store /tmp/e2e-c.mdio

# strip-stitch (N>1): shared MDIO, chunk-aligned inline ownership
cargo run -p synthoseis -- run --e2e --chunked --workers 4 --chunk-i 2 --chunk-j 4 \
  --store /tmp/e2e-strip.mdio
```

Library: `MultiWorkerRunner::run_e2e_strip_stitched` /
`pipeline_stream::run_e2e_strip_stitched`.

## Multi-process JobPartitionPlan e2e

**Landed:** prove non-overlapping writers on **separate OS processes** sharing one
MDIO store via the `JobPartitionPlan` JSON artifact (no K8s/AWS).

```bash
cd rust
# Orchestrator: prepare store + chunk-aligned plan, spawn N children, finalize
cargo run -p synthoseis -- run --e2e --chunked --multiprocess --workers 4 \
  --chunk-i 2 --chunk-j 4 --store /tmp/e2e-mp.mdio
```

Library: `prepare_multiprocess_store` / `run_worker_partition` /
`finalize_multiprocess_e2e` / `run_e2e_multiprocess` / `write_strip_partition`.
CLI `--worker-id` is the child entry. See [`SCALE.md`](./SCALE.md).

See also [`SCALE.md`](./SCALE.md) for the RAM → strip-stitch → multi-process → async → GPU ladder.

## Develop

```bash
cd rust
cargo check
cargo test
cargo run -p synthoseis -- --help
cargo run -p synthoseis -- run --store /tmp/smoke.mdio
cargo run -p synthoseis -- run --workers 4 --partition-plan /tmp/plan.json
cargo run -p synthoseis -- run --e2e --store /tmp/e2e.mdio
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
- First seismic kernels in `synthoseis-seismic` (from
  `datagenerator/zoeppritz_kernel.py`, `wavelets.py`, `Seismic.py`):
  `zoeppritz_pp`, `compute_rfc_volumes`, `ricker`, `hanflat`,
  `convolve_same_1d`, `apply_wavelet_traces`, `snr_std_ratio`,
  `hilterman_noise_weights` — goldens in `tests/fixtures/seismic_kernels.json`;
  angle-stack MAE wired through `synthoseis-core::parity`
- First RPM depth-trend kernels in `synthoseis-rpm` (from
  `rockphysics/rpm_example.py` + `rpm_tagilsk_trends.py`):
  `RpmExampleTrends::*`, `tagilsk_shale_*` / `tagilsk_brine_sand_*` /
  `tagilsk_gas_sand_*`, `polyval` — goldens in `tests/fixtures/rpm_trends.json`

**Landed (e2e)**

- Single-worker tiny-cube wiring: MDIO → geo → closures → RPM → seismic → MDIO
  with parity harness checks (`synthoseis run --e2e`)

**Landed (multi-worker partition)**

- Local `partition_jobs` + `JobPartitionPlan` (serde) + `MultiWorkerRunner`
- CLI `--workers` / `--partition-plan` (cloud handoff smoke)
- Non-chunked e2e remains full-cube single pass; strip-stitch uses `--chunked`

**Landed (chunked streaming e2e)**

- Single-worker fused chunked pipeline (no full elastic/RFC/stack temps)
- MDIO sub-volume chunks + `--chunked` / `--chunk-i/j/k` CLI
- Working-set bound tests on 32×32×64 with 8×8×64 tiles

**Landed (strip-stitch multi-worker e2e)**

- Local N-worker inline strips → shared MDIO chunk writes → parity vs single-worker
- CLI `--e2e --chunked --workers N` (N>1)

**Landed (multi-process JobPartitionPlan e2e)**

- OS multi-process writers via plan artifact on shared FS
- CLI `--e2e --chunked --multiprocess --workers N` + `--worker-id` child mode

**Still out of scope / next**

- **Cloud** K8s/AWS execution consuming `JobPartitionPlan` (local multi-process proves the plan)
- Full Butterworth bandpass / lateral filter / RMO / end-to-end SeismicVolume
- Full Tagilsk oil-sand polys + EndMemberMixing / Backus moduli
- Full geology stack / faults / **GPU**
- Replacing Parameters Python zarr store / dropping the Python generator
- Publishing wheels
