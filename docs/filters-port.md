# Post-convolution seismic filter port (Python `datagenerator/Seismic.py` → Rust)

This is the second slice of the *geology realism* track. The Rust pipeline can now
apply the legacy post-convolution **Butterworth bandpass** (zero-phase
`filtfilt`) and the **lateral box filter** (`uniform_filter` over inline and
crossline) to every angle stack it produces.

The filters are **off by default** (`E2eConfig::filters == FilterConfig::default()`).
With them off, every existing output stays bit-identical: a test pins hashes
recorded on master `ca11a457`. When they are on, the filtered stacks match the
legacy Python code **bit for bit**, and the output is bit-identical for every
chunk shape, strip-worker count and multi-process partition.

## What the legacy code does

`SeismicVolume.build_seismic_volumes` → `postprocess_rfc_cubes` runs these
steps on the 4-D `(angles, ni, nj, nk)` reflectivity cube:

| # | Legacy step | Code | Ported? |
|---|---|---|---|
| 1 | Reflectivity per angle | `create_rfc_volumes` | yes (earlier: `compute_rfc_volumes`, fused tile path) |
| 2 | Noise | `add_weighted_noise(depth_maps)` | **deferred** (see below) |
| 3 | Bandpass | `apply_bandlimits` → `derive_butterworth_bandpass` + `apply_butterworth_bandpass` | **yes** |
| 4 | Lateral filter, only if `lateral_filter_size > 1` | `apply_lateral_filter` → `uniform_filter(size=(0, n, n, 0))` | **yes** |
| 5 | Scaling and output | `write_final_cubes_to_disk` → `_scale_seismic` (global std → 100) and rpm near/mid/far factors | deferred |
| 6 | Relative acoustic impedance deliverable | `apply_cumsum` (float32 `cumsum` + 2–100 Hz bandpass) | kernel only (`cumsum_traces_f32` + bandpass), not wired |
| 7 | Augmentations | `apply_augmentations` / `apply_rmo` | deferred |

The alternative wavelet path (`bandlimit_volumes_wavelets`, used only when
`cfg.wavelets` is set) runs `oaconvolve` with a wavelet, then the lateral
filter, then cumsum.

**Parameters.** In `Parameters.py` the corners are drawn per model as
`lowfreq ~ U(bandwidth_low)` and `highfreq ~ U(bandwidth_high)` (3–6 Hz and
20–35 Hz in the example config). `order = bandwidth_ord` (4), and
`lateral_filter_size = int(U(0, 2) + 0.5) * 2 + 1`, which gives 1, 3 or 5.
`digi` is 4 ms, so fs = 250 Hz and Nyquist = 125 Hz. The Rust port takes these
values explicitly through `FilterConfig` and does **not** port the random draws.

## Python → Rust mapping

| Python | Rust (`synthoseis-seismic::kernels::filters`, `synthoseis-core`) |
|---|---|
| `derive_butterworth_bandpass(low, high, digitisation, order)`: `fs = 1/(digitisation/1000)`, `butter(order, [low/nyq, high/nyq], "bandpass", output="ba")` | `butterworth_bandpass(low, high, digitisation_ms, order) -> IirFilter { b, a, zi }` follows the same scipy chain: `buttap` → `lp2bp_zpk` → `bilinear_zpk` → `zpk2tf`. `poly()` uses numpy's complex product order, and the complex arithmetic follows numpy (Smith division, glibc `csqrt`). |
| legacy `digitisation = (digi / 1000) * 1000` | `legacy_digitisation_ms(digi)` (same float round trip) |
| `scipy.signal.lfilter_zi(b, a)` (scipy ≥ 1.15 closed form, numpy pairwise `sum`) | `lfilter_zi(b, a)` using `np_sum`, a port of numpy's pairwise sum |
| `apply_butterworth_bandpass` = `filtfilt(b, a, x, method="pad")`: odd extension with `padlen = 3·max(len(a), len(b))`, forward and backward `lfilter` seeded with `zi·x[0]` | `IirFilter::filtfilt_f32(trace, scratch)`: the odd extension is built in f32 like numpy on a float32 array, the filtering runs in f64 (direct form II transposed), and the result is stored as f32 like the legacy float32 cube |
| `apply_bandlimits` (every trace of the 4-D cube) | `IirFilter::filtfilt_traces_f32(volume, nk)` |
| `apply_lateral_filter` = `uniform_filter(data, size=(0, n, n, 0))`, mode `reflect`, inline pass then crossline pass, each stored as f32 | `lateral_uniform_tile(src, src_i, src_j, shape, out_i, out_j, n, out)` and `lateral_uniform_volume`; `reflect_index` is the ndimage `reflect` boundary and `uniform_window(n)` gives the window `[−n/2, n−n/2−1]` |
| `apply_cumsum` step 1 (`np.cumsum(axis=-1)` on float32) | `cumsum_traces_f32` (kernel only, fixture-tested) |
| `postprocess_rfc_cubes` steps 3–4 | `FilterConfig` → `SeismicFilters::from_config` (validation plus one-time design) → `fuse_tile_filtered` in every generation path |

## Where the stage sits in the Rust pipeline

The Rust e2e pipeline has no separate reflectivity-cube stage. It fuses
Zoeppritz reflectivity and (optionally) the Ricker wavelet per tile
(`fuse_tile_local`) straight into the angle stack. The filters run right after
fusion, in the legacy order: bandpass, then lateral filter.

### Ricker skip (legacy chain: reflectivity, then bandpass)

Legacy applies the bandpass to *RFC + noise* and never convolves with a
wavelet in its default path: the Butterworth filter *is* the wavelet. The
first filter port (#25) filtered the Ricker-convolved stack, which band-limits
twice. Now, **when the bandpass is on, the Ricker convolution is skipped by
default**, so the Rust chain is reflectivity → bandpass → lateral, exactly as
in legacy.

| `FilterConfig` | Wavelet | Chain |
|---|---|---|
| filters off (default) | Ricker 40 Hz | reflectivity ⊛ Ricker (unchanged, bit-identical to master) |
| lateral only (`bandpass_hz: None`) | Ricker 40 Hz | reflectivity ⊛ Ricker → lateral |
| bandpass on (default `keep_ricker: false`) | **none** | reflectivity → bandpass [→ lateral] (**legacy**) |
| bandpass on, `keep_ricker: true` / CLI `--keep-ricker` | Ricker 40 Hz | reflectivity ⊛ Ricker → bandpass [→ lateral] (#25 behaviour, bit-identical to master `9d5d2051`) |

- `FilterConfig::skips_ricker()` is the switch. `SeismicFilters::wavelet(w)`
  and `effective_wavelet(cfg, w)` return `synthoseis_gpu::NO_WAVELET` (an empty
  slice) when it is set. Every path goes through `fuse_tile_filtered` (or the
  classic `generate_tiny_cube`), so they all honour it.
- An empty wavelet is an exact identity: `convolve_same_1d` returns the signal,
  so the fused tile is the raw f32 reflectivity (f32 → f64 → f32 round trip is
  lossless).
- **GPU (`--gpu`, wgpu/WGSL).** The WGSL kernel `fuse_tile.wgsl` copies its
  reflectivity scratch straight to the output when `wavelet_len == 0`, so the
  GPU fuse path honours the skip natively; no CPU fallback is needed.
  `synthoseis-gpu/tests/parity_fuse_tile.rs::no_wavelet_skip_respected_by_cpu_gpu_and_dispatch`
  checks the CPU adapter is bit-identical to the reference reflectivity, and
  that `fuse_tile_gpu`, `fuse_tile_auto` and `fuse_tile_dispatch` (prefer-GPU)
  with `NO_WAVELET` equal the CPU bits on CPU fallback or are within
  `GPU_CPU_MAX_ABS_TOL` (1e-2) on a real adapter. On the dev box's llvmpipe
  Vulkan adapter the skip-mode WGSL reflectivity differs from the CPU by at
  most 6.2e-8 (0°), 7.4e-4 (15°) and 2.7e-3 (30°): the shader evaluates
  Zoeppritz in f32, the same near-parity as the Ricker path. So bit-exact
  legacy parity is a CPU-path property; `--gpu` is near-parity. CI also runs
  an e2e `--gpu --bandpass 4,30` smoke.
- `generate_reflectivity(cfg, angle)` returns the raw pre-wavelet angle stack
  (legacy `rfc_raw` for one angle), for parity and QC.

## Tiling invariance: halos

- **Bandpass.** `filtfilt` works along the trace. Every generation path already
  fuses full-`nk` columns per tile, because the wavelet needs the whole trace, so
  the bandpass needs no vertical halo. MDIO `ck` chunking only affects the write.
- **Lateral filter.** An output column `(i, j)` reads the columns
  `reflect(i + d)` × `reflect(j + e)` for `d, e ∈ [−n/2, n−n/2−1]`.
  `lateral_source_range(o0, o1, n_axis, n)` returns the source range, or halo,
  for an output range on one axis. It includes reflected boundary columns, so
  a tile at the cube edge reads its mirror columns.
- `fuse_tile_filtered` works in four steps:
  1. Fuse the halo tile `[si0, si1) × [sj0, sj1)` from the full-volume labels,
     which every path already holds.
  2. Bandpass every halo trace.
  3. Evaluate the lateral filter for the tile's own columns.
  4. Write only those columns.
- Fusion and bandpass are pure per-column functions of the labels, and the
  lateral window is summed in a fixed order: f64 accumulation, `/ n`, f32 store
  per pass. A halo column recomputed by one tile, strip worker or OS process is
  therefore bit-identical to the same column computed by its owner.
- No worker-to-worker exchange is needed, and a tile's memory stays bounded by
  `(ti + n − 1)(tj + n − 1)·nk` floats plus one trace of scratch.
- The cost is recomputing up to `n − 1` extra columns per tile axis. For
  16×16 tiles and `n = 3` that is 27 % more fuse work. `WorkingSetStats`
  records the halo buffers.
- Every path uses `fuse_tile_filtered`: `generate_chunked`,
  `generate_chunked_at_angle`, `run_e2e_streaming`, `run_e2e_streaming_overlapped`,
  `run_e2e_strip_stitched` (`write_strip_partition`), `run_worker_partition` /
  `run_e2e_multiprocess`, and geometry-once. The classic whole-cube
  `generate_tiny_cube` applies the same kernels to the full volume
  (`apply_filters_to_volume`).
- **Fixed bug.** Several streaming paths passed a full `ci·cj·nk` buffer to the
  tile fuser, which asserts `ti·tj·nk`. That would panic on edge tiles when the
  chunk shape does not divide the cube. `fuse_tile_filtered` slices the
  buffer, so non-dividing chunk shapes such as 5×7 now work everywhere.

**Proof (tests in `rust/synthoseis-core/tests/filters_pipeline.rs`).** The filter
configurations are 4–30 Hz with lateral 3, 5.5–22 Hz with lateral 5, bandpass
only (order 2) — all three with the Ricker skipped — lateral only with an even
`n = 4` (Ricker kept), and 4–30 Hz lateral 3 with `keep_ricker`. The worker /
path test runs the two skip configs and the `keep_ricker` config. The cube is faulted,
24×20×64. All comparisons are exact `f32::to_bits` equality.

- `filtered_stack_equals_whole_volume_filter`: the halo-tiled output equals
  the whole-volume kernels applied to the unfiltered input (the raw
  reflectivity when the Ricker is skipped, else the Ricker stack).
- `ricker_skip_filters_raw_reflectivity`: with the skip, the filtered stack is
  exactly bandpass + lateral of `generate_reflectivity`.
- `keep_ricker_is_bit_identical_to_master_filtered_output`: with
  `keep_ricker`, three filter configs reproduce hashes recorded on master
  `9d5d2051` (chunked and classic paths), and the skip output differs.
- `filtered_stack_invariant_to_chunk_shape`: chunk shapes 24×20, 8×5, 5×7,
  1×20, 24×1 (ck 16), 7×3 (ck 32) and 2×2 all match, and so does the classic
  `generate_tiny_cube`.
- `filtered_stack_invariant_to_workers_and_paths`: all of these read back the
  same MDIO `angle_stack` as the single-worker reference:
  - streaming and overlapped streaming with chunks 8×5×16, 5×7×64 and 3×20×32;
  - strip-stitch with 2, 3 and 4 workers;
  - multi-process with 1, 2 and 3 workers;
  - geometry-once, at 15°.
- `filters_disabled_by_default_is_bit_identical_to_master`: FNV hashes of labels
  and angle stacks recorded on master `ca11a457` for three configurations,
  faulted and unfaulted.
- `invalid_filter_config_is_an_error`: `nk ≤ padlen` (27 for order 4) or a
  corner at or above Nyquist returns `Err` from every `run_*` entry point.

## Parity against legacy Python

The fixture generator is `tests/fixtures/generate_seismic_filters.py`. It calls the
**real** legacy `derive_butterworth_bandpass`, `SeismicVolume.apply_bandlimits`,
`apply_lateral_filter` and `apply_cumsum` through a stub `cfg`. `numba` and
`rockphysics` are stubbed only so the module imports. It uses numpy 2.5.3 and
scipy 1.18.1, and writes `tests/fixtures/seismic_filters.json`. The test is
`rust/synthoseis-seismic/tests/filters_parity.rs`.

| Check | Result |
|---|---|
| `butter` b/a, 10 designs (orders 1–6, dt 1/2/4 ms, 2–100 Hz) | max relative difference 1.4e-15; b and a bit-identical in 5 of 10 designs |
| `lfilter_zi` | ≤ 6.4e-12 relative: `sum(b) ≈ 0`, so `y_inf` is rounding noise, and the effect on outputs is 0 |
| `apply_bandlimits` (3–20 Hz o4, 6–35 Hz o3; 64-sample traces) | **bit-exact, 100 %** |
| `apply_lateral_filter` (n = 3, 4, 5, including n > axis length) | **bit-exact, 100 %** |
| bandpass 4.5–27.3 Hz + lateral 3 chain | **bit-exact, 100 %** |
| `apply_cumsum` (cumsum + 2–100 Hz) | **bit-exact, 100 %** |

**Realistic cubes.** `examples/filters_demo.rs` dumps the unfiltered and
filtered Rust 15° stacks. `examples/plot_filters_demo.py` filters the
*unfiltered Rust stack* with the legacy code and compares:

| Cube | Filters | max \|Rust − legacy\| | bit-exact | spectrum diff |
|---|---|---|---|---|
| 64×64×128, seed 7, 4 faults requested | 4–30 Hz o4, lateral 3 | 0 (peak 3.64) | 100 % of 524,288 samples | 0 |
| 96×80×200, seed 11, 6 faults requested | 5.3–33.7 Hz o4, lateral 5 | 0 | 100 % of 1,536,000 samples | 0 |

Both runs also check that 16×16 and 5×7 tiles give bit-identical output.
(These rows were measured in #25 with the Ricker kept; they are reproduced by
the `keep_ricker` stack.)

**Ricker skip parity.** `examples/plot_ricker_skip.py` feeds the Rust *raw
reflectivity* (`angle_rfc.f32`, the angle stack before any wavelet) to the
legacy `apply_bandlimits` + `apply_lateral_filter` and compares it with the
Rust filtered stack (Ricker skipped):

| Cube | Filters | Comparison | max \|Rust − legacy\| | bit-exact | spectrum diff |
|---|---|---|---|---|---|
| 64×64×128, seed 7, 4 faults | 4–30 Hz o4, lateral 3 | skip vs legacy on Rust reflectivity | 0 (peak 3.18) | 100 % of 524,288 | 0 |
| same | same | `keep_ricker` vs legacy on Rust Ricker stack | 0 (peak 3.64) | 100 % | — |

Mean-spectrum peak moves from 25.4 Hz (Ricker + bandpass) to 7.8 Hz
(reflectivity + bandpass, as in legacy; the toy reflectivity is red). Figure:
`ricker_skip_spectra.png` (inputs with |W(f)| and |H(f)|², before/after
filtered spectra with the legacy curve, and the relative difference).

**Known possible gap.** scipy's `uniform_filter1d` uses a *running* mean,
`tmp += (x[i+a] − x[i−b−1]) / n`. The Rust version sums each window directly,
which is what makes it tiling-invariant. On seismic-like data the two agree
bit for bit: 0 mismatches in every test above and in a 300×257×8 stress test
per size n = 3, 4, 5, 7. With a synthetic dynamic range of 12 decades between
neighbouring traces, scipy's running sum accumulates cancellation error. There,
0.001 % to 0.1 % of samples differ, by at most about 1e-8 of the local peak.
The direct sum is the more accurate of the two.

filtfilt padding follows scipy's `method="pad"`, `padtype="odd"`,
`padlen = 3·max(len(a), len(b))` exactly, including the float32 odd
extension, so there is no edge gap. Traces need `nk > padlen`, the same
constraint scipy raises.

Figures, produced by `plot_filters_demo.py`:
`filters_before_after.png` (inline, crossline and time slices: unfiltered,
Rust, legacy, difference) and `filters_spectra.png` (mean amplitude spectra
with the ideal `|H(f)|²`).

## Deferred items and follow-ups

- **Noise (`add_weighted_noise`).** It draws exponential noise with a random
  sign for 0° and 45°, Hilterman-weights it per angle (`cos²` / `sin²`), and
  scales it by the *global* std of the middle-angle RFC below seabed + 15 and by
  `sqrt(10^(sn_db/10))`. It also depends on `_calculate_snr_after_lateral_filter`
  (pre-smear SNR). The existing kernels `snr_std_ratio` and
  `hilterman_noise_weights` cover the scalar parts. A tiled port needs two things:
  - a global statistics pass, computable from labels in a first sweep;
  - a counter-based per-voxel RNG, because bit parity with the numpy stream is
    impossible in tiles.

  Noise must be added *before* the bandpass, so it slots in at the start of
  `fuse_tile_filtered`.
- `_scale_seismic` (global std → 100) and the rpm near/mid/far factors both
  need a global std pass.
- The relative acoustic impedance deliverable (`apply_cumsum`): the kernel is
  ported and bit-exact, but the extra MDIO variable is not wired. Its 2–100 Hz
  bandpass needs `nk > 27`.
- The random draws of `lowfreq`, `highfreq` and `lateral_filter_size` from the
  config.
- The wavelet path (`bandlimit_volumes_wavelets`) and augmentations / RMO.
- CLI: `--bandpass` / `--lateral-filter` currently require single-worker
  `--e2e --chunked`, like `--faults`. Multi-process children do not receive
  the flags yet. The library API supports every path.

## How to run

```bash
cd rust
# CLI (single worker, chunked); traces need > 27 samples for an order-4 bandpass
cargo run -p synthoseis -- run --e2e --chunked --shape 48,48,64 --faults 3 \
    --bandpass 4,30 --lateral-filter 3 --store /tmp/filtered.mdio
# kernel parity vs the legacy fixtures, then the pipeline invariance tests
cargo test -p synthoseis-seismic --test filters_parity -- --nocapture
cargo test -p synthoseis-core --test filters_pipeline
# realistic-cube parity + figures (needs numpy, scipy, matplotlib)
cargo run --release -p synthoseis-core --example filters_demo -- /tmp/fd 7 4 64 64 128 4 30 3
python synthoseis-core/examples/plot_filters_demo.py /tmp/fd /tmp/filters
python synthoseis-core/examples/plot_ricker_skip.py /tmp/fd /tmp/ricker_skip_spectra.png
# old combined behaviour (Ricker, then bandpass)
cargo run -p synthoseis -- run --e2e --chunked --bandpass 4,30 --keep-ricker --store /tmp/keep.mdio
# regenerate the fixture
python ../tests/fixtures/generate_seismic_filters.py
```
