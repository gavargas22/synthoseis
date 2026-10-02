# Depth-to-time conversion

Spec: [Spec: depth-to-time conversion (Strata)](https://app.notion.com/p/3eb0272f1f7b81e89801dc568eef3ee1),
§8 approved plan. The work landed in two PRs that share one review slot:

- **PR A (#37)** added the conversion kernels, a library-only `TimeConfig`, and
  the analytic tests, with no output change.
- **PR B** wires the conversion into every pipeline path and **makes two-way
  time the default output**. `--legacy-depth-as-time` (library:
  `TimeConfig::legacy()`) reproduces master 0eb937b5 byte for byte. See
  [PR B: time output by default](#pr-b-time-output-by-default).

## Why

Today every sample means both 4 m of depth (`RockPhysicsConfig::depth_step_m`)
and 4 ms of time (`TINY_DIGI`). That is a conversion at a constant 2000 m/s,
while the rock-physics Vp spans 1500 m/s (water) to 4500 m/s (salt). Salt
pull-up, gas push-down and correct thin-bed tuning all need the time axis to
follow the actual Vp.

## PR A: the conversion core

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

In PR A it was not a field of `E2eConfig` and had no CLI (§8: "library only,
`enabled = false`, no CLI"). PR B adds both and flips the default.

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

`rust/synthoseis-core/tests/time_config.rs` checks the defaults (enabled since
PR B, `nt₀ = nz`; dt = 2 ms gives 2·nz), the validation matrix and the
staircase warning.

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

The pipeline tests (tiling invariance, label round trip, dead last sample,
the #35 time-mode test, the switch and exit-2 matrix) are PR B's; see below.

## Per-tile cost (spec §6 target ≤ 1.3×), PR A prototype

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

## PR B: time output by default

### What changes

- **Default output is two-way time.** `TimeConfig::default()` is
  `enabled = true, dt_ms = 4, samples = None (nt₀), kernel = Sinc`, and it is a
  field of `E2eConfig` (`E2eConfig::time`). Labels, rock physics and Zoeppritz
  stay in depth. Each fuse tile builds `T` per column from the voxel's final
  f32 Vp (`time_mode::tile_twt` / `column_twt`), computes Zoeppritz on the depth
  interfaces and inserts it at `T_{k+1}` (`time_mode::fuse_props_tile_time`),
  then applies the Ricker, bandpass, lateral filter and noise on the time grid.
  The dead last sample is `nt − 1` (§3.7). `labels`, `fault_labels` and
  `salt_labels` are point-sampled through the same `k(n)` (§3.4), so the
  per-voxel invariants carry over.
- **Every path**: classic (`generate_tiny_cube`), chunked / fused, streaming,
  overlap, strip-stitch, geometry-once, multi-worker and multi-process. All of
  them write the output shape `(ni, nj, nt)`, the output-domain label cubes and
  the time attributes. Their parity read-backs compare against the
  output-domain references.
- **`RICKER_PEAK_HZ`** (40 Hz) feeds every Ricker call site through
  `E2eConfig::ricker()`, sampled at `E2eConfig::digi_ms()`: `dt` in time mode,
  4 ms (`TINY_DIGI`) on the legacy axis.
- **GPU**: the GPU fuse is depth-only. In time mode `--gpu` logs the CPU
  fallback and runs the CPU path (spec §6).
- **MDIO root attributes** in time mode: `time_conversion = "vp-twt"`,
  `depth_step_m`, `twt_kernel`, with `digi = dt` and `units = "ms"`. The legacy
  axis writes none of them, so its stores are master's.

### CLI

| Flag | Meaning |
|---|---|
| `--legacy-depth-as-time` | Master 0eb937b5's axis (one 4 m cell = one 4 ms sample, i.e. a constant 2000 m/s), byte for byte. `--legacy-toy-depth` implies it. |
| `--dt-ms DT` | Output sample interval, 0.5–8.0 ms (default 4). |
| `--twt-samples N` | Output trace length. The default is `nt₀ = round(nz·2dz/2000/dt)`, which equals NK at the defaults. Range `min(16, nt₀) ≤ N ≤ 8·NK`. |
| `--twt-kernel sinc\|linear` | Spike insertion: Kaiser-windowed sinc (default) or the 2-tap linear split (fast; aliases above about 0.4 f_N). |

**Exit 2** (stderr names the problem):

- any time option together with `--legacy-depth-as-time` or
  `--legacy-toy-depth`;
- `--bandpass-trailing-sample` without `--legacy-depth-as-time` (time mode
  always zeroes the dead last sample);
- `--dt-ms` outside 0.5–8.0 or not finite;
- `--twt-samples` outside its range;
- an unknown kernel;
- `f_hi > 0.4/dt`, for example `--dt-ms 8` with the Ricker on.

Multi-process forwards the flags to every worker. The staircase constraint
`dz ≤ Vp_min·dt/1.2` is a stderr **warning** only, for example at
`--dt-ms 2` on the 4 m grid. In time mode the run summary prints the axis and
the short and long columns:

```
time axis: two-way time from voxel Vp (vp-twt), dt=4 ms, nt=256, kernel=sinc, depth step 4 m (--legacy-depth-as-time for master's depth-as-time axis)
time columns: base TWT 733.7-1032.6 ms vs trace end 1020.0 ms; short 89.2% (worst shortfall 286.3 ms, zero-filled below the model base), long 10.8% (worst excess 12.6 ms, truncated)
```

(64×64×256, seed 3.) Short columns are zero below the model base and their
labels forward-fill the deepest cell (half-space). Long columns are truncated
at `nt − 1`.

### Goldens

- Every pinned library test that encodes master's output uses
  `TimeConfig::legacy()`: faults, filters, bandpass trailing sample, layered
  geometry, lithology, rock physics, salt, Zoeppritz, and #35's `filter_cfg`.
- `rock_physics_cli.rs` `run()` appends `--legacy-depth-as-time`, so every CLI
  golden is unchanged. A `run_time()` helper covers the default.
- `synthoseis/tests/depth_to_time_cli.rs` pins:
  - `MASTERD2T_{PLAIN,RICH,RICH_TRAILING,MP}`, hashes of stores written by the
    master 0eb937b5 binary;
  - the new defaults `TIME_{PLAIN,RICH,MP}`.

### Tests owned by PR B

`rust/synthoseis-core/tests/depth_to_time_pipeline.rs`:

| Spec | Test | What it checks |
|---|---|---|
| 5.4 | `time_mode_tiling_invariance_rich` | RICH flags (3 faults, 4–30 Hz bandpass, 12.5 dB noise, salt), seeds 30 and 11, nt = nz and nz + 37. Chunk shapes `[1,1,nt]`, `[5,7,nt]`, `[3,20,16]`, `[8,5,16]`. The classic, chunked, streaming, strip 2/3/4, multiprocess 1/2/3 and geometry-once paths produce bit-identical angle stacks, labels, fault labels and salt labels. At nt = nz the columns are both short and long. |
| 5.4, §3.3 | `dt_2ms_invariance_and_staircase_warning` | dt = 2 ms at dz = 2 m is tiling invariant with no warning. dt = 2 ms at dz = 4 m fires the staircase warning. |
| 5.5 | `label_round_trip` | Every output sample carries the class of the depth cell whose time interval contains it, for labels, fault labels and salt labels alike. No class is invented. Every cell at least `dt` thick is sampled. fault ∧ salt appears only where the depth cell has it. 255 stays above the seabed. |
| §3.7, 5.8 | `dead_last_sample` | `nt − 1` is exactly 0 and is the only sample the zeroing touches (sinc and linear, nt = nz and nz + 37, plus the chunked path). |
| §4, 5.2 | `uniform_2000_time_reflectivity_is_the_depth_fuse_one_sample_down` | At a uniform 2000 m/s, the production time fuse equals the production depth fuse one sample down, bit for bit. |
| §6 | `time_mode_fuse_tile_cost_within_1_3x` | Time fuse tile ≤ 1.3× the depth fuse tile. Measured **1.06–1.07×** (release, 32×32×256). |
| 5.7 | `legacy_depth_as_time_reproduces_master_0eb937b5` | Library angle stack, labels and fault labels on the demo cube (plain, bandpass + noise, 3 faults, both) equal master 0eb937b5's hashes. |

Other PR B tests:

- `angle_stack_legacy_e2e.rs` `legacy_fixture_time_mode_uniform_2000` (§4) runs
  on #35's legacy Python fixture at uniform 2000 m/s:
  - The time-mode raw reflectivity equals the depth fuse one sample down, bit
    for bit.
  - The time-mode stack, read one sample down, meets #35's tolerance
    (≤ 8 ulp, relative RMS ≤ 1e-7) on every column whose lateral footprint has
    no reflection in the first `padlen` = 27 samples.
  - On seed 25's two top-edge columns (seabed at sample 21), the legacy
    `filtfilt` odd extension about t = 0 differs from the time trace's correct
    zero top. That leaves a decaying transient of 2.2–2.8e-3 (≤ 4 % of the
    peak) in the first 50 samples, which is within tolerance below sample 400
    and is pinned by band.
- `synthoseis/tests/depth_to_time_cli.rs`:
  - the legacy store hashes;
  - time-mode attributes and summary;
  - the flags reach multi-process workers bit for bit;
  - the exit-2 matrix;
  - the staircase warning and the GPU fallback log.

### Byte identity of the legacy switch

The CLI stores of master 0eb937b5 and this branch with
`--legacy-depth-as-time` are byte-identical in all 16 cases checked, after
normalising only the `created` timestamp attribute, and stdout is identical:

- demo 64×64×256 seed 7;
- demo with bandpass + noise;
- 3 faults, and 3 faults with bandpass + noise;
- bandpass trailing sample;
- multiprocess 3 workers, and multiprocess 2 workers seed 30;
- strip 3, classic, classic planar, overlap, angles;
- plain, rich, deep salt, toy.

### Evidence

Images are in [`img/depth-to-time/`](../img/depth-to-time/). Before is
`--legacy-depth-as-time`, which is master 0eb937b5 and so #35's stack-parity
baseline. After is the time default. All cubes are 64×64×256 at the default
config.

- Salt seeds 1, 2, 3 and 30: stack and label sections, depth vs time, on the
  inline through the thickest salt.
- `pullup_measured_vs_predicted.png`, salt pull-up per column for seeds 1, 2,
  3, 7 and 30 at sub-sample resolution (see below).
- `seed7_faults_labels_depth_vs_time.png`: seed 7, 3 faults, `--no-salt`. The
  time `fault_labels` equal an independent numpy resample of the depth fault
  labels through T(Vp), with 0 mismatching voxels (33 812 depth fault voxels
  become 33 290 time fault voxels).
- `uniform2000_time_vs_pr35_baseline.png`: #35's legacy Python stack against
  the time output at uniform 2000 m/s.

**Salt pull-up, sub-sample.** Script:
`scripts/depth_to_time_evidence/d2t_b_pullup_subsample.py`. The pick is the
16× FFT one from #37's salt test. Labels can only be picked on whole samples,
so the measurement uses the seismic event instead.

- **Horizon.** One sub-salt layer top L per seed. Its top lies below the salt
  base in both runs, and it is the horizon with the most salt-footprint columns
  whose event *dominates* the raw time reflectivity in both runs. "Dominates"
  means: at n = the first output sample of layer L in the time label cube, the
  larger |x| of samples n−1 and n is at least 2× every other |x| within
  ±6 samples. This is an amplitude QC only and uses no timing. It excludes weak
  contrasts next to strong ones, which a peak pick would lock onto: with
  coverage-only horizon choice, seeds 2 and 3 had picks off by 10–17 ms.
- **Trace.** The production time-mode fuse without a wavelet (raw two-way-time
  reflectivity at the default incidence), from both the salt run and the
  `--no-salt` run.
- **Pick.**
  1. Polarity from samples n−1 and n.
  2. Coarse extremum in n−2 … n+1.
  3. Extremum of the 16× FFT-upsampled trace within ±1 sample.
  4. Parabola through the three best fine samples.

  Columns whose polarity differs between the runs are dropped: 3, 0, 0, 4 and
  2 columns for seeds 1, 2, 3, 7 and 30.
- **Measured** = `t_nosalt − t_salt`. **Predicted** = `T_nosalt(z_L) −
  T_salt(z_L)`, the cell-boundary TWT from each run's Vp (it includes the salt
  drag of the horizon).

| Seed | Horizon | Columns (QC / salt) | Predicted median / max | **Measured** median / max | Residual mean / std / max \|·\| | Legacy axis median / max |
|---|---|---|---|---|---|---|
| 1 | L36 | 329 / 761 | 112.41 / 145.31 ms | **112.85 / 144.87 ms** | +0.04 / 0.54 / 2.05 ms | 0 / 20 ms |
| 2 | L44 | 278 / 516 | 97.36 / 239.61 ms | **97.61 / 240.16 ms** | +0.03 / 0.25 / 0.67 ms | −4 / 0 ms |
| 3 | L28 | 401 / 787 | 99.18 / 227.04 ms | **99.27 / 226.85 ms** | +0.11 / 0.54 / 3.33 ms | 4 / 52 ms |
| 7 | L26 | 444 / 597 | 67.69 / 130.35 ms | **67.61 / 129.79 ms** | −0.01 / 0.35 / 1.04 ms | 0 / 24 ms |
| 30 | L37 | 358 / 438 | 83.65 / 214.86 ms | **83.78 / 214.45 ms** | −0.07 / 0.33 / 1.34 ms | 0 / 40 ms |

- **Bias.** Residual means are within ±0.11 ms: there is no bias at the
  sub-sample level.
- **Spread.** The 0.25–0.54 ms spread is interference: the tails of
  neighbouring interfaces shift the peak. The absolute pick against `T(z_L)` in
  a single run has std 0.18–0.40 ms.
- **Stack cross-check.** The same picks on the deliverable stack (40 Hz Ricker)
  give residual std 0.15–0.41 ms, max 0.47–3.42 ms.
- **Pooled.** Every sub-salt horizon × column that passes the QC
  (479 / 573 / 1442 / 1070 / 878 picks) gives residual means of −0.01 to
  +0.09 ms, std 0.35–0.63 ms, and 90–99 % of picks within 1 ms.
- **Before.** The legacy axis (4 ms label picks) shows essentially no pull-up.
  Its non-zero values are horizon drag by the salt (a depth change), not
  velocity.
- **Columns.** The QC keeps 43–82 % of the salt columns, so the maximum
  predicted pull-up among the measured columns (130–240 ms) is below the
  thickest-salt values (135–282 ms over all salt columns). The thickest
  columns' sub-salt horizons are often weak or missing; in seeds 2 and 3 the
  salt reaches the model base.
- **CI gate.** `rust/synthoseis-core/tests/depth_to_time_pullup.rs` runs the
  same measurement on seed 1 at 32×32×128 (125 columns, max predicted pull-up
  66 ms). It asserts ≥ 100 columns, |mean| ≤ 0.25 ms, std ≤ 0.6 ms and
  max |residual| ≤ 1.5 ms. Measured: +0.045 / 0.345 / 0.923 ms.

### Spec deviations

- `--twt-samples` floor is `min(16, nt₀)` rather than 16, so tiny test cubes
  (nz < 16) keep their default axis.
- The `TimeAxis` reaches the fuse through `RpmModel` / `E2eConfig::time_axis()`
  rather than a separate argument on every writer.
- The #35 time-mode test meets #35's tolerance on clean columns only. Seed 25's
  two top-edge columns carry the filtfilt top-padding transient described
  above.
- #35's `filter_cfg` gains `time: TimeConfig::legacy()`, its one-line legacy
  pin.
- The time summary lines print in time mode only, so legacy stdout stays
  master's.
- Fault ∧ salt overlap is inherited from depth: the depth cubes overlap on
  master. The time resample preserves the overlap per voxel, so the separate
  fault-label salt-mask fix carries over unchanged.
- `TimeConfig::constant_twt_vp` is a doc-hidden test hook (T from one constant
  velocity) for the uniform-2000 tests.
