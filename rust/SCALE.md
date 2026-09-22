# Scale-out strategy (Rust path)

Product lock: **arbitrary cube size is a time/disk bound, not a RAM bound** for
elastic / RFC / angle temps. Adding workers or machines lowers wall-clock until
I/O saturates. Labels stay a compact u8 deliverable; horizon maps are O(ni×nj).

## Ladder

| # | Stage | Status | What it buys |
|---|-------|--------|--------------|
| 1 | **Chunked fused streaming** | **Landed** (#15) | Fuse elastic → Zoeppritz RFC → wavelet per spatial tile; MDIO sub-volume chunks. Peak temps ≈ one tile (plus O(ni×nj) maps + u8 labels), not three full elastic volumes. |
| 2 | **Strip-stitch multi-worker (local)** | **This PR** | N local workers own contiguous **inline strips** snapped to `chunk_i`, fuse-generate, and `write_chunk` / `write_labels_chunk` into **one shared store** with no overlapping keys. Parity vs single-worker chunked reference. |
| 3 | **JobPartitionPlan → multi-process / cloud** | Next | Serialize `JobPartitionPlan` (already serde JSON) to non-overlapping writers on separate processes / pods / machines. Same chunk-key ownership; object-store or shared FS backend. |
| 4 | **Async compute / write overlap** | Later | Pipeline tile fuse on CPU while previous chunk bytes flush (io_uring / async runtime). Hides store latency without changing ownership rules. |
| 5 | **GPU tile kernels** | Later | Port per-tile Zoeppritz + wavelet (and optionally RPM trends) to GPU; host still owns strip partition + MDIO writes. |
| 6 | **Zarr sharding / compression** | Later | Blosc/ZFP (or Zarr v3 sharding) to cut disk and network; interchangeable with today’s raw LE chunks once writers stay non-overlapping. |

## Invariants to keep

- **Working set:** temps bounded by chunk/tile size (slack for wavelet support), not by full `ni×nj×nk` elastic/RFC/stack.
- **Ownership:** writers never share an MDIO chunk key (`i_chunk, j_chunk, k_chunk`). Strip-stitch snaps inline ranges to `chunk_i`.
- **Parity:** deliverables are **labels + angle stacks** (IoU / agreement / MAE / max-abs), not bit-identical full seismic.
- **Cloud handoff:** `JobPartitionPlan` is the artifact; local `MultiWorkerRunner` proves the same sharding without K8s/AWS in this tree yet.

## CLI map

```bash
# (1) single-worker chunked
cargo run -p synthoseis -- run --e2e --chunked --store /tmp/c.mdio

# (2) strip-stitch local multi-worker
cargo run -p synthoseis -- run --e2e --chunked --workers 4 --chunk-i 2 --chunk-j 4 \
  --store /tmp/strip.mdio

# (3) plan artifact for a future cloud consumer
cargo run -p synthoseis -- run --workers 4 --partition-plan /tmp/plan.json
```

## Explicitly deferred

Cloud K8s/AWS execution, GPU, Python generator rewrite, publishing wheels.
