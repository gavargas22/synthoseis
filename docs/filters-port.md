# Post-convolution seismic filter port (Python `datagenerator/Seismic.py` → Rust)

This is the second slice of the *geology realism* track. The Rust pipeline can now
apply the legacy post-convolution **Butterworth bandpass** (zero-phase
`filtfilt`) and the **lateral box filter** (`uniform_filter` over inline and
crossline) to every angle stack it produces.

It also ports the legacy additive noise (`add_weighted_noise`) as a
deterministic, tiling-invariant, statistically equivalent stage (see
[Noise](#noise-add_weighted_noise)).

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
| 2 | Noise | `add_weighted_noise(depth_maps)` | **yes**, opt-in: deterministic and statistically equivalent, not bit-exact (see [Noise](#noise-add_weighted_noise)) |
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

### Trailing sample (legacy parity bug fix)

Legacy `create_rfc_volumes` produces `nk − 1` reflectivity samples per trace
(one per interface; `rfc_raw` is one sample shorter than the elastic cube), and
`apply_bandlimits` filters exactly those `nk − 1` samples. Legacy never has an
`nk`-th reflectivity sample: augmentation crops the seismic and the labels to
`cube_shape[2] + pad − 1`. The Rust fuse keeps the output grid at `nk` samples
and writes the trailing sample as 0 ("no interface").

Up to #35 the Rust bandpass filtered the whole `nk`-sample trace. The extra 0
moved filtfilt's odd-extension edge and changed the deepest part of every trace
(up to about 1.8e-2 against stack peaks of 0.06–0.09 in the last 10 samples;
see [`angle-stack-e2e-parity.md`](angle-stack-e2e-parity.md)). Now:

| `FilterConfig` | Bandpassed samples | Trailing sample |
|---|---|---|
| bandpass on, Ricker skipped (default) | first `nk − 1` (the legacy trace) | written as 0, the fuse value; legacy has no sample there |
| same with `bandpass_trailing_sample: true` / CLI `--bandpass-trailing-sample` | all `nk` | filtered (bit-identical to master before the fix) |
| `keep_ricker: true` | all `nk` (unchanged, bit-identical to master `9d5d2051`) | filtered: a Ricker-convolved trace has real signal there; legacy has no Ricker + bandpass path to match (see below) |
| lateral only / filters off | no bandpass | unchanged |

- `FilterConfig::bandpass_excludes_trailing_sample()` (`skips_ricker() &&
  !bandpass_trailing_sample`) is the switch, applied in the shared
  `bandpass_traces` used by `fuse_tile_filtered` (every tiled, streaming,
  strip, multi-process and geometry-once path) and `apply_filters_to_volume`
  (classic path). Tiles always hold whole traces, so tiling and worker
  invariance are unaffected.
- With noise on, the fuse adds noise to all `nk` samples; the trailing sample
  is still written as 0, because legacy has neither signal nor noise there.
  The lateral filter then sees an all-zero depth slice and keeps it 0.
- The padlen check counts the filtered samples: default mode needs
  `nk − 1 > padlen` (order 4: `nk ≥ 29`); `--bandpass-trailing-sample` and
  `keep_ricker` still need `nk > padlen` (`nk ≥ 28`). A 28-sample
  order-4 run that used to pass is now rejected unless the flag is given.
- Edge handling itself is unchanged (scipy odd extension, `padlen = 3·max(len(a),
  len(b))`); mirrored padding or a taper is a later realism item.
- **The zeroed trailing sample is a dead sample.** It exists only for legacy
  parity: every bandpassed trace now ends in an exact 0 whatever the geology
  above it, so the bottom depth slice of every angle stack is identically 0
  (and, with noise on, noise-free). A network trained on these cubes could
  learn to spot that slice, or use it as a depth / position cue. Crop the
  last sample (legacy's own grid is `nk − 1`) or mask it in training until
  the edge-handling realism item (mirrored padding or a taper at the trace
  ends) lands; that item should remove the dead sample by giving the last
  sample real, edge-treated signal instead of a hard 0.
- Tests: `synthoseis-core/tests/bandpass_trailing_sample.rs` pins both modes'
  hashes for five configurations (planar and folded geometry, orders 2 and 4,
  lateral 1/3/5, with and without noise), pins the Ricker-only paths (filters
  off, lateral only, `keep_ricker`, noise without a bandpass) under both flag
  values, checks that fixed mode equals the whole-trace filter on the first
  `nk − 1` samples with a 0 trailing sample, and checks the padlen rule. The
  `filters_pipeline.rs` chunk-shape and worker/path invariance tests include
  the flag mode. `angle_stack_legacy_e2e.rs` checks the default against the
  real legacy generator all the way to the base.
- **Ricker path (`--keep-ricker`): no legacy counterpart, left unchanged.**
  Traced in legacy `datagenerator/Seismic.py` and checked by running the real
  legacy `build_seismic_volumes` on the #35 fixture model (seed 3, nk = 510)
  with every filter call logged:
  - The default legacy chain never convolves with a wavelet:
    `apply_bandlimits` (on `(5, 16, 16, 509)`, i.e. `nk − 1`) →
    `apply_lateral_filter` → `apply_cumsum` (2–100 Hz on 509 samples);
    `apply_wavelet` is never called.
  - Legacy's only wavelet path, `bandlimit_volumes_wavelets`, runs only if
    `cfg.wavelets` exists. `Parameters` never sets it and no config or
    filter-spec `.npy` ships, so it is dormant. Forcing it (with legacy's own
    `wavelets.ricker(40, 4 ms, 1)`, the exact taps of the Rust Ricker, max
    difference 1.1e-16) shows `apply_wavelet` (`oaconvolve`, `mode="same"`)
    on the `nk − 1` = 509-sample trace, then `apply_lateral_filter`, and
    **no bandpass** on the stack; only its cumsum is bandpassed (2–100 Hz,
    509 samples). The regular bandpassed cubes are still written from the
    unconvolved reflectivity. (`wavelets.ricker` itself only seeds the
    spectral template inside `generate_wavelet`.)
  - So legacy never bandpasses a wavelet-convolved trace, and there is no
    legacy Ricker → bandpass output for `--keep-ricker` to match. The Rust
    analogue of the legacy wavelet path, Ricker + lateral filter without a
    bandpass, already matches it to the base: on the fixture models
    (5 angles × 2 seeds, first `nk − 1` samples) max |Δ| ≤ 1.5e-8 in every
    depth band except ≤ 3e-8 beyond 200 samples, rel RMS ≤ 1.1e-7 (Ricker-convolved
    reflectivity peaks 0.36–0.58; FFT `oaconvolve` vs direct convolution
    rounding). A trailing 0 is a no-op for the first `nk − 1` outputs of a
    "same" convolution, so that path has no trailing-sample gap.
  - For scale only: if legacy's primitives were composed into a wavelet →
    bandpass chain on `nk − 1` samples, `--keep-ricker` would differ from it
    by 0.85–1.6e-2 in the last 10 samples (peaks 0.06–0.08), decaying like
    the no-Ricker gap. Nothing in legacy produces that cube, so
    `keep_ricker` keeps its contract (bit-identical to #25 / `9d5d2051`).
- Legacy-reproduction goldens recorded before the fix with a bandpass
  (`rock_physics_cli.rs` RICH set, `legacy_toy_depth_switch_reproduces_master_10f4dcd`,
  `legacy_zoeppritz_reproduces_master_33a3a93`) now pass
  `--bandpass-trailing-sample` / `bandpass_trailing_sample: true`.

## Edges (physical filter edges, time mode)

Every filter needs samples beyond the end of a trace or the side of the cube.
Up to master `1c22b653` each filter made them up its own way. In time mode
(the default axis) the rule is now one rule: **continue the earth the way the
physics says, and mirror only where nothing is known** (spec "filter edge
handling", Strata).

| Filter | Edge rule before (`1c22b653`, `--legacy-filter-edges`) | Edge rule now (time mode) |
|---|---|---|
| Ricker wavelet (17 taps, reaches 8 samples) | zero outside the window | zero above time 0 (water); the model's own reflectivity for `h = 8` samples below the window |
| Butterworth bandpass, forward-backward | odd mirror (`2·x₀ − x`) over `padlen` = 27 samples, SciPy start states | zero above time 0 (water, zero start state); the model's reflectivity for `Pb = h + edge_pad()` samples below (369 for 4–30 Hz order 4 at 4 ms), then the half-space; backward start state `zi·y_end` |
| #36 dead last sample | sample `nt − 1` left out of the bandpass and written as 0 | filtered like every other sample (it has real data under it now) |
| Noise (Laplace, Philox) | window samples only | window samples keep their keys; the pads get noise too, from a separate counter domain (Philox word 3 = 1), so the strength is the same up to the trace ends |
| Lateral box filter (3 or 5) | reflect | **reflect** (unchanged: nothing is known beyond the cube) |
| Salt drag smoothing (σ = 3) | reflect | **reflect** (unchanged; labels do not move) |
| Sinc spike insertion | interfaces deeper than the buffer skipped | unchanged; the buffer is now `nt + Pb` long, and the first `nt` samples are bit-identical to before |

How it runs, per column (`time_mode::finish_trace_padded`):

1. The time reflectivity is computed into `nt + Pb` samples with the existing
   column routines (whole voxels, partial `subcell`, partial `cell`). Below
   the model base there are no interfaces, so the pad is zero there.
2. Top pad: `Pt = Pb` with noise; `Pt = h` with a kept Ricker + bandpass and
   no noise (the Ricker precursor above time 0); else 0. Window samples get
   `sample(col·nt + k)` (today's field, bit for bit); pad samples get
   `sample_pad(col, pad_index)`, top pad first. `data_std` is unchanged.
3. The Ricker (unless skipped) is a "same" convolution over the padded buffer.
4. The bandpass (`IirFilter::filtfilt_padded_f32`) runs forward-backward over
   the padded buffer: forward start state zero, or `zi·x₀` when `Pt > 0`;
   backward start state `zi·y_end`. No odd extension, no `padlen`, so any
   `nt ≥ 1` works (the "needs more than 27 filtered samples" check stays with
   the legacy edges).
5. The `nt` window samples are kept. The lateral filter then runs as before.

`IirFilter::edge_pad()` is the smallest `n` such that the forward-backward
impulse response stays below 1e-6 of its peak from `n` on (computed once per
design in f64; capped at 8,192 samples, with an error for an unstable design).
Measured: 369 (4–30 Hz), 421 (3–35 Hz), 366 (6–20 Hz) at 4 ms, order 4.

Every step is per column and keyed by global indices, so halo recompute,
tiling, strips and processes stay bit-identical. The padded buffer lives in
the per-column scratch; `WorkingSetStats` counts the filtered halo at its
padded length.

What it changes (seed 7, 32 × 32 × 128, 15°):

- **Default run** (Ricker only): only samples 121–127 of columns whose model
  continues below the window (4.65 % of cells, rel. RMS 0.017). The max change
  is seed-dependent (median 3.9 % of peak across seeds 1–30; up to 35 % where
  a strong reflector such as a salt top sits just below the window, seed 17).
  Columns whose padded reflectivity is zero below the window are unchanged
  bit for bit.
- **`--bandpass 4,30`**: every sample. The old edges were off from the
  physically correct answer (a 2,048-sample pad down to the model base) by
  rel. RMS 0.17 and up to 48 % of peak; the new default matches it to 3e-8 of
  peak (the f32 rounding of the output).
- **`--keep-ricker` + bandpass (no noise):** the top pad is `h` (8 at 4 ms, 16 at 2 ms)
  so the Ricker precursor above time 0 reaches the forward bandpass (with
  `Pt = 0` the error was up to 7.6e-5 of peak).
- **Noise**: strength at the first and last sample 0.99 / 0.97 of mid-trace
  (was 0.16 at the first sample).
- **Labels**: unchanged everywhere.

**Order 6 at 2 ms:** the transfer-function (`ba`) recursion carries up to
~7e-4 of peak of f64 round-off (6–20 Hz: 3.5e-4) in **both** edge
modes (physical and `--legacy-filter-edges`). Use order ≤ 5 or dt 4 ms. The
nightly edge_pad sweep gates against an SOS-form reference (≤ 1e-5 for
orders 2–5, ≤ 5e-3 for order 6 as the documented `ba` floor). Converting
the physical bandpass to SOS is a deferred follow-up.

**Opt-out:** `--legacy-filter-edges` (CLI, forwarded to multi-process
workers) / `TimeConfig::legacy_filter_edges = true` (library) restores
master `1c22b653` bit for bit (apart from the `created` stamp) on every path.
It is exit 2 with `--legacy-depth-as-time` or `--legacy-toy-depth`: the
legacy depth axis is a parity mode against the Python generator and keeps
SciPy's edges. Time-mode stores with the physical edges carry the root
attribute `filter_edges = "physical"`; legacy-edge stores omit it, so they
stay byte-identical to `1c22b653`. The run summary prints
`filter edges: physical (water above, model below; reflect sideways)` or
`filter edges: legacy 1c22b653 (--legacy-filter-edges)`.

Tests: `synthoseis-seismic/tests/filter_edges.rs` (edge pad, padded
filtfilt, noise keys), `synthoseis-core/tests/filter_edges.rs` (prefix
identity, Ricker churn, truth gate, shift invariance, noise stationarity,
invariance, labels; nightly sweeps) and `synthoseis/tests/filter_edges_cli.rs`
(`1c22b653` byte identity on every path).

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
- `invalid_filter_config_is_an_error`: too few filtered samples (`nk − 1 ≤
  padlen` by default, `nk ≤ padlen` with `--bandpass-trailing-sample` or
  `keep_ricker`; padlen is 27 for order 4) or a corner at or above Nyquist
  returns `Err` from every `run_*` entry point.

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

(Measured before the trailing-sample fix, filtering the whole `nk`-sample
trace on both sides; the skip rows are reproduced by
`--bandpass-trailing-sample`. The default now filters `nk − 1` samples, see
[Trailing sample](#trailing-sample-legacy-parity-bug-fix).)

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
extension, so filtering the *same trace* has no edge gap. The filtered trace
needs more than `padlen` samples, the same constraint scipy raises.

Two notes, both measured on full legacy models in
[`angle-stack-e2e-parity.md`](angle-stack-e2e-parity.md):

- **Same trace as legacy (fixed).** Up to #35 the production pipeline
  bandpassed the `nk`-sample Rust reflectivity trace, trailing 0 included,
  while legacy bandpasses its `nk − 1` samples. That moved the
  odd-extension edge (up to about 1.8e-2 against stack peaks of 0.06–0.09 in
  the last 10 samples). The default now bandpasses the first `nk − 1`
  samples and matches legacy to rounding all the way to the base (max
  |Δ| ≤ 9.3e-10 on the sampled columns); `--bandpass-trailing-sample` keeps
  the old behaviour. See [Trailing sample](#trailing-sample-legacy-parity-bug-fix).
- **scipy version.** Bit-exactness holds against scipy 1.18.1 (closed-form
  `lfilter_zi`). The repository's locked scipy 1.17.1 computes `zi` with
  `linalg.solve`, and the outputs differ by a few f32 ulps.

Figures, produced by `plot_filters_demo.py`:
`filters_before_after.png` (inline, crossline and time slices: unfiltered,
Rust, legacy, difference) and `filters_spectra.png` (mean amplitude spectra
with the ideal `|H(f)|²`).

## Noise (`add_weighted_noise`)

Noise is **off by default** (`FilterConfig::noise == NoiseConfig::default()`,
`snr_db: None`). With it off, every output is bit-identical to before (the
filters-off golden hashes and the `keep_ricker` hashes recorded on master
`9d5d2051` are pinned in `noise_pipeline.rs`).

### What the legacy code does

`SeismicVolume.add_weighted_noise(faulted_depth_maps)` (`Seismic.py`) runs
before `postprocess_rfc_cubes`:

1. `noise_3d` (called twice, for `noise_0deg` and `noise_45deg`): draws
   `rng.exponential(1/100, N)` and a sign from `rng.binomial(1, 0.5, N)`
   (0 → −1). That is white **Laplace** noise with scale 0.01, no frequency
   shaping. (It also draws two unused seed integers.) Both cubes are shared by
   every angle.
2. Per angle: `weighted = noise_0deg·cos(ang)² + noise_45deg·sin(ang)²`,
   with `from math import sin, cos` fed the angle **in degrees** (the bug that
   `tests/test_seismic_noise.py` documents; the fix was never applied to
   `Seismic.py`).
3. `noise = weighted · (data_std / weighted.std()) / std_ratio`, with
   `std_ratio = sqrt(10^(sn_db/10))`, stored as float32 and added to the raw
   reflectivity in float32 (`rfc_noise_added = noise + rfc_raw`). It is added
   everywhere, including the water column.
4. `data_std` is the std of the **middle** angle's raw reflectivity
   (`rfc_raw[1]`, or `[2]` with `model_qc_volumes`) over the samples
   `k >= wb / (digi + 15) · digi`, where `wb = faulted_depth_maps[..., 0]`
   (the seabed in `digi` units; NaN/0 holes are infilled). The variable is
   called `wb_plus_15samples`, so `wb / digi + 15` was probably intended. As
   written, for digi = 4 the threshold is `0.84 ×` the seabed sample, i.e.
   slightly *above* the seabed.

   **Rust default: the cutoff is the actual seabed** (`k >= seabed`, only
   sub-seabed reflectivity feeds `data_std`). `NoiseConfig::legacy_seabed` /
   `--noise-legacy-seabed` restores the exact legacy expression. On the
   32×32×96 seed-7 test cube (seabed 0.5–3.7 samples), `data_std` goes from
   7.849921e-3 (legacy mask) to 7.829226e-3 (seabed), −0.26 %; the noise
   amplitude scales by the same factor.
5. `sn_db` is drawn per model from a triangular distribution over
   `signal_to_noise_ratio_db` (example config 7.5 / 12.5 / 17.5 dB).
   `_calculate_snr_after_lateral_filter` is never called.

### Rust design

| Legacy | Rust |
|---|---|
| numpy `default_rng` stream, `exponential` × `binomial` sign | **Philox4x32-10** counter-based RNG (Random123 known-answer tests pass), key = SplitMix64(seed ^ salt), counter = global voxel index `(i·nj + j)·nk + k`. One 128-bit block per voxel gives two independent unit Laplace draws `(n0, n45)`: `−ln(u)` with `u = (m + 1)/2^53` from the top 53 bits and the sign from bit 0 |
| `n0·cos² + n45·sin²` | same, with `(w0, w45)` = `hilterman_noise_weights` (radians, the default) or, with `legacy_angle_weights`, `legacy_degree_noise_weights` (`math.cos(deg)`, exact legacy) |
| `/ weighted.std()` (sample std of the whole cube) | `/ sqrt(2 (w0² + w45²))`, the analytic population std of the mix. This needs no second pass, and every tile knows its scale up front. It is identical in expectation; the relative gap is `O(1/sqrt(N))` (0.04–0.14 % on 97 k voxels) |
| `data_std` = `rfc_raw[mid][mask].std()` (float32 numpy) | `noise_signal_std`: one streaming pass that fuses the raw reflectivity at `NOISE_NORM_ANGLE_DEG` (15°) one inline row at a time (memory = one `nj × nk` row). Samples `k >= seabed` (default) or `k >= legacy_noise_mask_threshold(seabed, digi)` (`legacy_seabed`), via `noise_mask_threshold`, of the `nk − 1` Zoeppritz samples are reduced with Welford in fixed global `(i, j, k)` order, in f64. The result does not depend on chunking or workers, and every worker or process recomputes the same bits |
| `noise.astype(f32) + rfc_raw` | `WeightedNoise::sample(g) = f32(mix · scale)`, added in f32 to the fused raw-reflectivity tile |
| seabed `faulted_depth_maps[..., 0]` | toy seabed `fault_seabed(cfg)` (top horizon, unfaulted) in samples × digi |

**Where it sits.** When noise is on, `fuse_tile_filtered` works through the
halo tile in legacy order:

1. fuse the halo tile as raw reflectivity (`NO_WAVELET`);
2. add the noise, keyed by global voxel index, so halo columns get exactly
   the noise of the tile that owns them;
3. convolve the Ricker wavelet when it is not skipped (noise only, lateral
   only, or `keep_ricker`), using the same f32 → f64 `convolve_same_1d` → f32
   path as the fused kernel;
4. apply the bandpass, then the lateral filter.

The classic whole-cube path (`generate_tiny_cube`) does the same steps on the
full cube. `SeismicFilters::resolve(cfg, labels, shape)` builds the filters
and runs the `data_std` pass. Every generation path (chunked, streaming,
overlapped, strip-stitch, multi-process worker, geometry-once, classic) calls
it after `generate_labels`, so geometry still runs once. All angles share the
same Philox counters, which matches legacy sharing `noise_0deg` and
`noise_45deg` across angles.

**GPU.** With `--gpu`, the reflectivity tile comes from the WGSL kernel with
`NO_WAVELET` (the Ricker-skip PR #26). The noise and the Ricker wavelet are then applied
on the CPU. `data_std` then uses the GPU reflectivity (0.14 % difference on
24×20×64). On llvmpipe, the GPU−CPU gap with noise on (max 1.1e-2, at the
k = 0 sample) is the same as with noise off. That gap is a pre-existing f32
Zoeppritz difference, not caused by noise.

### Legacy statistical equivalence

Bit parity with the numpy stream is impossible (and not wanted) in tiles, so
equivalence is statistical. `tests/fixtures/generate_seismic_noise.py` runs
the **real** legacy `add_weighted_noise` for 64 seeds. Its input is the Rust
raw reflectivity from the `noise_demo` example (32×32×96, seed 7, no faults,
angles 5/15/25°, S/N 12.5 dB). It records the noise mean, std, excess
kurtosis, a 6-band amplitude spectrum and the inter-angle correlations. It
does this twice: legacy as written (`legacy_degrees`), and legacy with the
radian fix (`radians`).

`noise_pipeline.rs::noise_statistics_match_legacy` compares 16 Rust seeds
(spectra from 4) against those references, with `legacy_seabed = true` (the
legacy mask). `SE` below is the legacy seed-to-seed std × `sqrt(1/64 +
1/S_rust)`.

`noise_statistics_default_seabed` runs the same comparison in the default
mode. Its `data_std` is checked against the fixture's `seabed_data_std`:
numpy, with the legacy masking code (`mute_above_seafloor`) applied at the
seabed. Rust gives 7.829225678e-3 against 7.829225622e-3 (7e-9 relative). The
std and mean references are rescaled by `seabed_data_std / legacy_data_std`.
Kurtosis, spectrum and correlation do not depend on scale and are compared
unchanged. The results match the table below, with std −0.14 % … +0.04 %
against the rescaled 1.856602e-3.

| Statistic | Tolerance | Result (worst case over 2 modes × 3 angles) |
|---|---|---|
| `data_std` | relative 1e-6 | Rust 7.849921151e-3, legacy 7.849921472e-3 (4e-8) |
| mean | ≤ 5 SE (≈ 8e-6, 0.4 % of std) | ≤ 3.7e-6 |
| std | seed-averaged ≤ 0.5 %, every seed ≤ 2 % | −0.14 % … +0.04 % (legacy std = `data_std/std_ratio` = 1.861510e-3 exactly) |
| excess kurtosis | ≤ 5 SE (0.08–0.16) | radians 2.99 / 2.96 / 2.74 vs 2.99 / 2.96 / 2.73; legacy weights 2.90 / 1.65 / 2.99 vs 2.97 / 1.63 / 2.99 |
| amplitude spectrum, 6 bands (white, ≈ 0.885 = √π/2) | ≤ 5 SE (0.010–0.016) | max \|Δ\| 0.0060 |
| corr(5°, 25°), corr(5°, 15°) | ≤ 5 SE | radians 0.9788 / 0.9980 vs 0.9787 / 0.9979; legacy weights 0.1054 / 0.6588 vs 0.1046 / 0.6592 |

The test can tell the two weightings apart. At 15°, the legacy-degree mix has
excess kurtosis 1.63, against 2.96 for the radian mix.

**Chain parity.** `plot_noise_demo.py` feeds the Rust reflectivity plus the
Rust noise to the real legacy `apply_bandlimits` + `apply_lateral_filter`
(4–30 Hz, order 4, lateral 3). The result matches the Rust pipeline output
**bit for bit**: 98,304 of 98,304 samples, max error 0. The 16×16 and 5×7
tilings are also bit-identical.

**Invariance** (`noise_pipeline.rs`, exact `to_bits`, six noise configs.
Four use the default seabed cutoff: noise only with the Ricker kept, legacy
chain with bandpass + lateral 3, `keep_ricker` + lateral 5, and legacy
weights + bandpass. Two use `legacy_seabed`: the legacy chain, and fully
legacy noise with degree weights and the legacy mask). Checked across:

- chunk shapes 24×20, 8×5, 5×7, 1×20, 24×1 (ck 16), 7×3 (ck 32), plus the
  classic path;
- streaming and overlapped streaming (2 chunkings);
- strip-stitch with 2, 3 and 4 workers;
- multi-process with 1, 2 and 3 workers;
- geometry-once (0° and 15°);
- `data_std` bits across chunk shapes.

Other checks: the same seed gives identical output and different seeds
differ (< 0.1 % equal samples). The default seed is `E2eConfig::seed`.

Figures: `noise_slices.png` (Rust and legacy noise, the noisy reflectivity,
the noisy bandpassed stack, and the Rust − legacy chain difference) and
`noise_spectra.png` (amplitude spectra, pdfs with kurtosis, std, and
inter-angle correlation, for both weightings).

## Deferred items and follow-ups

- **Noise follow-ups.**
  - The per-model draw of `sn_db` (triangular over
    `signal_to_noise_ratio_db`) is not ported; `NoiseConfig::snr_db` is
    explicit.
  - `data_std` is recomputed by every worker or process (one extra fused
    angle pass). Caching it in the multi-process plan sidecar would avoid
    that.
  - The seabed is the unfaulted toy top horizon. Legacy uses the faulted
    depth map and infills NaN/0 holes.
  - The legacy degree weights and the legacy `wb / (digi + 15) · digi`
    `data_std` threshold can still be selected (`legacy_angle_weights`,
    `legacy_seabed`). The defaults are radians and the true seabed.
- `_scale_seismic` (global std → 100) and the rpm near/mid/far factors both
  need a global std pass.
- The relative acoustic impedance deliverable (`apply_cumsum`): the kernel is
  ported and bit-exact, but the extra MDIO variable is not wired. Its 2–100 Hz
  bandpass needs `nk > 27`.
- The random draws of `lowfreq`, `highfreq` and `lateral_filter_size` from the
  config.
- The wavelet path (`bandlimit_volumes_wavelets`, dormant in legacy: see the
  Ricker-path note under Trailing sample) and augmentations / RMO.
- Edge-handling realism: done in time mode (see [Edges](#edges-physical-filter-edges-time-mode)),
  including the dead trailing sample. The legacy depth axis keeps SciPy's
  edges and the trailing-sample rule (parity mode).
- CLI: `--bandpass` / `--lateral-filter` / `--noise-snr-db` currently require single-worker
  `--e2e --chunked`, like `--faults`. Multi-process children do not receive
  the flags yet. The library API supports every path.

## How to run

```bash
cd rust
# CLI (single worker, chunked); with the legacy edges (--legacy-depth-as-time or
# --legacy-filter-edges) an order-4 bandpass needs >= 29 samples (>= 28 with
# --bandpass-trailing-sample or --keep-ricker); time mode has no minimum
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
# old whole-trace bandpass (trailing sample included; master before the fix)
cargo run -p synthoseis -- run --e2e --chunked --bandpass 4,30 --bandpass-trailing-sample --store /tmp/old.mdio
cargo test -p synthoseis-core --test bandpass_trailing_sample
# regenerate the fixture
python ../tests/fixtures/generate_seismic_filters.py
# noise (legacy add_weighted_noise, opt-in): CLI, tests, legacy stats + figures
cargo run -p synthoseis -- run --e2e --chunked --shape 48,48,64 --bandpass 4,30 \
    --lateral-filter 3 --noise-snr-db 12.5 --noise-seed 7 --store /tmp/noisy.mdio
cargo test -p synthoseis-core --test noise_pipeline -- --nocapture
cargo run --release -p synthoseis-core --example noise_demo -- /tmp/noise_demo
python ../tests/fixtures/generate_seismic_noise.py /tmp/noise_demo   # seismic_noise.json
python synthoseis-core/examples/plot_noise_demo.py /tmp/noise_demo /tmp/noise
```
