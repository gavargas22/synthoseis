# Depth-to-time conversion: PR A, the conversion core (no output change)

Spec: [Spec: depth-to-time conversion (Strata)](https://app.notion.com/p/3eb0272f1f7b81e89801dc568eef3ee1),
§8 approved plan. The work is split into two PRs that share one review slot:

- **PR A (this change)** adds the conversion kernels, a library-only
  `TimeConfig`, and the analytic tests. **No pipeline path reads any of it**,
  so every output and every pinned hash is unchanged by construction.
- **PR B** moves the seismic chain into two-way time and flips the default (see
  [What PR B will do](#what-pr-b-will-do)).

## Why

Today every sample means both 4 m of depth (`RockPhysicsConfig::depth_step_m`)
and 4 ms of time (`TINY_DIGI`). That is a conversion at a constant 2000 m/s,
while the rock-physics Vp spans 1500 m/s (water) to 4500 m/s (salt). Salt
pull-up, gas push-down and correct thin-bed tuning all need the time axis to
follow the actual Vp.

## What lands in PR A

`rust/synthoseis-seismic/src/kernels/twt.rs` (re-exported from
`synthoseis_seismic`):

| Item | Spec | What it does |
|---|---|---|
| `twt_column(vp, dz, out)` | §1 | `T_0 = 0`, `T_{k+1} = T_k + 2·dz/Vp_k`, in ms, summed in f64 in order. The increment is `2000·dz/Vp`, so uniform 2000 m/s at dz = 4 m gives exactly `4k` ms. |
| `default_twt_samples(nz, dz, dt)` | §2 | `nt₀ = round(nz · (2·dz / 2000 m/s) / dt)`. This equals `nz` at the defaults. |
| `KaiserSinc` (`standard()`: M = 8, β = 6.5, P = 512) | §3.3 | Kaiser-windowed sinc cut off at the output Nyquist, tabulated as (P+1) × 2M f64 = 66 KB, with linear interpolation between phases. `h(0) = 1` and `h(m ≠ 0) = 0` are stored exactly, so the kernel interpolates. The table is a pure function of (M, β, P), so every worker builds identical bits. |
| `insert_spikes(r, t_iface, dt, kernel, x)` | §3.1 step 4, §2 | `x[n] += r_k · h(n − T_{k+1}/dt)` in interface order. An exact integer position is a delta. Taps outside `[0, nt)` are dropped. Interfaces later than `t_{nt−1} + M·dt` are skipped, while those in the last M samples still add their tails. |
| `linear_split` / `TwtKernel::Linear` | §3.3 | 2-tap spike splitting: a fast option that aliases above about 0.4 f_N. |
| `point_sample_labels(labels_z, T, dt, out)` | §3.4 | `L_t[n] = L_z[k(n)]` with `k(n) = max{k : T_k ≤ t_n}`, clamped to `nz − 1` (short columns forward-fill). One two-pointer merge. It never invents a class, and it is the identity at 2000 m/s. It is generic over the label type, so `labels`, `fault_labels` and `salt_labels` all go through the same `k(n)`. |
| `reflectivity_time_column(vp, vs, rho, dz, angle, form, dt, kernel, scratch, x)` | §3.1 steps 2–4 | For one column and angle: T, then Zoeppritz on the depth interfaces (same kernel and form as the depth fuse), then insertion at `T_{k+1}`. Below `T_nz` there is no reflectivity (half-space). |
| `output_nyquist_ok`, `depth_staircase_ok` | §3.3 | `f_hi ≤ 0.4/dt` (will be enforced), and `dz ≤ Vp_min·dt/1.2` (a warning). |

In `synthoseis_core` (`pipeline.rs`), `TimeConfig { enabled: false, dt_ms: 4.0,
samples: None, kernel: Sinc }` provides:

- `output_samples(nz, dz)`;
- `validate(nz, dz, filters)`: `dt` must be in 0.5–8.0, `16 ≤ nt ≤ 8·nz`, and
  the output Nyquist rule applies. `f_hi` is the bandpass high corner when
  the bandpass replaces the Ricker (`skips_ricker()`), otherwise 2.5 × the
  40 Hz Ricker peak = 100 Hz. That includes `--bandpass --keep-ricker`, so
  dt = 8 ms fails there;
- `staircase_warning`.

It is **not** a field of `E2eConfig`, and there is no CLI yet (§8: "library
only, `enabled = false`, no CLI"). PR B adds both, so time mode can't be half
switched on in A.

## Tests owned by PR A (spec §8)

In `rust/synthoseis-seismic/tests/depth_to_time.rs`:

| Spec | Test | Result |
|---|---|---|
| 5.1 | `constant_2000_is_the_legacy_axis_shifted_one_sample` | `T_k = 4k` exactly. The label resample is the identity. The inserted reflectivity equals the depth reflectivity shifted down one sample, **bit for bit**, for `sinc` and `linear` at 0/15/30°. |
| 5.1 | `constant_3000_times` | `T_k = 8k/3` to 2.3e-11 ms over 1000 cells (gate 1e-9). |
| 5.1 | `constant_4500_short_column_half_space` | nz = nt = 256: `T_nz` = 455.11 ms, so the column is short from sample 114. Every label below forward-fills the last cell, and there is no reflectivity below the last interface's band-limited tail. |
| 5.2 | `linear_gradient_closed_form` | V0 = 1600 m/s, k = 0.6 1/s, dz = 4 m, nz = 250, f32 midpoint Vp: `T(1000 m)` = 1061.5123 ms (closed form 1061.5124). Max \|T_k − t(k·dz)\| = **1.47e-4 ms** (gate 1e-3 ms). |
| 5.3 (synthetic part) | `salt_pull_up_synthetic_columns` | Base reflection at **640.000 ms** (column A) and **568.884 ms** (column B, salt), against 640.000 / 568.889 predicted. Pull-up is **71.116 ms** against 71.111 ms (gates ±0.02 ms, on a 16× FFT pick). The `linear` kernel fails the gate by design: column B 568.712 ms (−0.177 ms), pull-up 71.288 ms. |
| 5.6 | `thin_bed_wedge_tuning` | 400 cases, 0–20 ms, random sub-sample offsets. `sinc` max error **0.136 %** of the single-wavelet peak (mean 0.068 %, worst tuning-peak error 0.088 %; gate 0.5 %). `linear` max 21.5 %, mean 12.8 % (reported only). |
| §3.3 | `kaiser_sinc_table_properties`, `long_column_truncation`, `nyquist_and_staircase_constraints` | The kernel interpolates exactly and is deterministic, with DC gain within 2.0e-4 over all phases (gate 5e-4). Long-column skipping keeps the tails. The constraint helpers behave as specified. |
| §3.3 | `kaiser_sinc_frequency_response` | Continuous Fourier transform of the tabulated kernel (table rows plus phase interpolation): **−0.14 dB at 0.8 f_N** (gate ≥ −0.2 dB), −6.02 dB at f_N, **−35.61 dB at 1.2 f_N** (gate ≤ −30 dB), −78.1 dB at 1.4 f_N, ≤ −77.3 dB beyond. The closed form gives the same values to 0.01 dB. (The spec table lists −0.1 dB at 0.8 f_N; the measurement is −0.14 dB.) |

`rust/synthoseis-core/tests/time_config.rs` checks the defaults (disabled,
`nt₀ = nz`; dt = 2 ms gives 2·nz), the validation matrix and the staircase
warning.

**Salt pick.** The spec asks for a "parabolic sub-sample pick". A three-point
parabola on the 4 ms samples of a 40 Hz Ricker is biased by up to about
0.17 ms, depending on the sub-sample phase: for column B it gives 568.736 ms
(−0.153 ms). The gate is instead applied to a pick on a **16× FFT-upsampled**
trace (spectrum zero padding, a direct DFT in test code), followed by a
parabola through the three best fine samples. That gives 568.884 ms
(−0.005 ms) and a pull-up of 71.116 ms. The reconstruction deliberately does
not use the conversion's windowed-sinc kernel, so the test doesn't grade the
kernel with itself (Strata's review condition). Strata's independent picks
agree: FFT 568.884 ms, pull-up 71.116 ms, and an analytic-Ricker
least-squares fit of 568.889 ms. The gate is ±0.02 ms, which the 2-tap
`linear` kernel fails (−0.177 ms); the test asserts that failure. The test
prints the FFT, three-point and predicted picks.

**Linear wedge number.** The metric divides the max error by the peak of a
single wavelet (the analytic Ricker peak). The spec's 54 % divided by the
peak of the two-spike trace, which shrinks as the spikes merge, so that
figure tends toward about 63 %. Under our metric, 21.5 % is correct. It is
reported, not gated.

**Tests owned by PR B** (spec §8): tiling invariance (5.4, including dt = 2 ms
and nt = nz + 37), the label round trip (5.5), the demo end-to-end checks
(5.3), the dead last time sample (§3.7 / 5.8), the #35 time-mode test (§4),
the switch and exit-2 matrix (5.7), and the golden-set additions. They need
the pipeline paths that PR B adds.

## Per-tile cost (spec §6 target ≤ 1.3×)

`rust/synthoseis-core/examples/d2t_tile_cost.rs` times a prototype time-mode
fuse tile built from the PR A kernels against the production legacy fuse tile
(`fuse_tile_local`, default rock physics, 15°):

- prototype: `tile_properties`, then `reflectivity_time_column`, then the same
  17-tap Ricker on the time trace;
- legacy: exactly `fuse_tile_local` (properties, Zoeppritz in depth, Ricker).

| Cube | Tiles | Legacy | Time mode, `sinc` | Time mode, `linear` |
|---|---|---|---|---|
| 64×64×256, seed 7 (demo) | 16 of 16×16 | 210.7 ms | 223.7 ms, **1.06×** | 218.1 ms, 1.04× |
| 128×128×384, seed 11 | 16 of 32×32 | 1301.7 ms | 1395.5 ms, **1.07×** | 1340.4 ms, 1.03× |

These are the best of 5 and 3 runs on the shared dev box. Exact Zoeppritz
dominates both sides (about 85 % of the tile). The prefix sum plus the
16-tap insertion add about 6–7 %. No new halo: every step is per column.

## Numerical choices (spec §3)

- Reflectivity is computed on the depth interfaces and inserted, band-limited,
  at the exact `T_{k+1}` (scheme B). Every interface is used once, and
  insertion is the anti-alias filter and resampler in one.
- The velocity is the voxel's final f32 Vp as `tile_properties` writes it.
- Vertical-incidence traveltime: every angle uses the same `T_{k+1}`, with no
  moveout.
- Placement: under uniform 2000 m/s the new mode is legacy shifted down by
  exactly one sample, because legacy stores `r_k` one sample above the label
  change.

## What PR B will do

- Wire the conversion into every path: classic, chunked, streaming, overlap,
  strip, multi-process and geometry-once. That covers the label, fault and
  salt writers, the output-domain read-backs, noise and filters on the time
  grid, and the dead last sample at `nt − 1` (§3.7).
- Add `--legacy-depth-as-time`, `--dt-ms`, `--twt-samples` and `--twt-kernel`,
  with multi-process forwarding and the exit-2 matrix.
- Flip the default to `enabled = true`.
- Add `--legacy-depth-as-time` to every golden set, pin `MASTERD2T_*` and the
  new default hashes, and add the MDIO attributes (`time_conversion`,
  `depth_step_m`, `twt_kernel`).
- Route the GPU fuse to the CPU in time mode.
- Run the invariance matrix, the label round trip, the demo checks, the #35
  time-mode test and the dead-last-sample test.
- Provide before/after evidence on seeds 7, 1, 2, 3 and 30.
