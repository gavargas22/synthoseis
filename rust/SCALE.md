# Scale-out strategy (Rust path)

Product lock: **arbitrary cube size is a time/disk bound, not a RAM bound** for
elastic / RFC / angle temps. Adding workers or machines lowers wall-clock until
I/O saturates. Labels stay a compact u8 deliverable; horizon maps are O(ni×nj).

## Ladder

| # | Stage | Status | What it buys |
|---|-------|--------|--------------|
| 1 | **Chunked fused streaming** | **Landed** (#15) | Fuse elastic → Zoeppritz RFC → wavelet per spatial tile; MDIO sub-volume chunks. Peak temps ≈ one tile (plus O(ni×nj) maps + u8 labels), not three full elastic volumes. |
| 2 | **Strip-stitch multi-worker (local)** | **Landed** (#16) | N local threads own contiguous **inline strips** snapped to `chunk_i`, fuse-generate, and `write_chunk` / `write_labels_chunk` into **one shared store** with no overlapping keys. Parity vs single-worker chunked reference. |
| 3 | **JobPartitionPlan → multi-process** | **Landed** (#17) | Serialize `JobPartitionPlan` (serde JSON) to non-overlapping writers on **separate OS processes** sharing one FS store. Same chunk-key ownership; prove multi-process without K8s/AWS. |
| 4 | **Async compute / write overlap** | **Landed** (#18) | A one-deep `std::sync::mpsc::sync_channel(1)` writer overlaps CPU tile fusion with the previous chunk flush. No Tokio/io_uring; single-worker first cut. |
| 5 | **Geometry once / seismic many** | **Landed** (#19) | Generate labels (+ maps) once; fuse N incidence angles into sibling MDIO angle stacks without regenerating geology. CLI `--angles` / `--seismic-many`. |
| 6 | **GPU tile kernels (CPU software)** | **Landed** (#20) | Per-tile Zoeppritz + wavelet in `synthoseis-gpu` with **CPU software** backend (CI-safe); host owns strip partition + MDIO writes. CLI `--gpu`. |
| 6b | **WGSL / wgpu tile fuse** | **This PR** | Real wgpu adapter probe + WGSL Zoeppritz+wavelet compute dispatch; `FuseBackend::Gpu` when adapter present, else CPU fallback. GPU is f32 near-parity (≤1e-2 max-abs on tiny tiles), not bit-identical to CPU f64. |
| 7 | **Zarr sharding / compression** | Later | Blosc/ZFP (or Zarr v3 sharding) to cut disk and network; interchangeable with today’s raw LE chunks once writers stay non-overlapping. |

## Invariants to keep

- **Working set:** temps bounded by chunk/tile size (slack for wavelet support), not by full `ni×nj×nk` elastic/RFC/stack.
- **Ownership:** writers never share an MDIO chunk key (`i_chunk, j_chunk, k_chunk`). Strip-stitch / multi-process snap inline ranges to `chunk_i`. File-system ownership of distinct keys is the lock (no shared Mutex across processes).
- **Parity:** deliverables are **labels + angle stacks** (IoU / agreement / MAE / max-abs), not bit-identical full seismic.
- **Cloud handoff:** `JobPartitionPlan` is the artifact; local multi-process CLI proves the same sharding on a shared FS without K8s/AWS in this tree yet.

## CLI map

```bash
# (1) single-worker chunked
cargo run -p synthoseis -- run --e2e --chunked --store /tmp/c.mdio

# (2) strip-stitch local multi-worker (threads)
cargo run -p synthoseis -- run --e2e --chunked --workers 4 --chunk-i 2 --chunk-j 4 \
  --store /tmp/strip.mdio

# (3) multi-process JobPartitionPlan e2e (orchestrator spawns N OS children)
cargo run -p synthoseis -- run --e2e --chunked --multiprocess --workers 4 \
  --chunk-i 2 --chunk-j 4 --store /tmp/mp.mdio

# (3b) worker-only mode (invoked by orchestrator; deterministic argv)
# synthoseis run --e2e --chunked --worker-id K --partition-plan PLAN \
#   --store STORE --seed S --workers N --chunk-i CI --chunk-j CJ --chunk-k CK

# (4) single-worker compute/write overlap (one-deep std writer)
cargo run -p synthoseis -- run --e2e --chunked --overlap \
  --store /tmp/overlap.mdio

# (6b) prefer GPU/WGSL tile fuse (falls back to CPU software when no device)
cargo run -p synthoseis -- run --e2e --chunked --gpu \
  --store /tmp/gpu.mdio

# Plan artifact only (chunk-aligned when --chunked / --chunk-i set)
cargo run -p synthoseis -- run --workers 4 --chunked --chunk-i 2 \
  --partition-plan /tmp/plan.json
```

### Deterministic child argv (stage 3)

The `--multiprocess` orchestrator re-execs the same binary once per worker:

```text
<exe> run --e2e --chunked --worker-id <k> --partition-plan <PLAN> --store <STORE> \
  --seed <S> --workers <N> --chunk-i <ci> --chunk-j <cj> --chunk-k <ck>
```

Workers load the plan, open the existing store, and call `run_worker_partition`
for their strip only. Sidecar stats/samples land under `{STORE}.mp/`. Finalize
runs once in the parent (`finalize_multiprocess_e2e`).

## Explicitly deferred

Cloud K8s/AWS execution, RPM-on-GPU / multi-worker GPU, Zarr sharding,
Python generator rewrite, publishing wheels.
