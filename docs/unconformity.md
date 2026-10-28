# Angular unconformity: bevel and ponded fill (roadmap item 6)

Spec: Notion "Spec: erosion and onlap". Code:
`rust/synthoseis-core/src/unconformity.rs`, called once at the end of
`toy_geometry::layered_stack`, after `apply_folds` and before rounding, so
faults, salt, closures, partial voxels and burial depth all see the
truncated and onlapping horizons. It is a closed-form pass per column with
no halo.

## Model

Stack order is deepest first. `u` is the stack index of the unconformity
horizon, `s = nh - 1 - u` its index from the seabed, `P = z_u`.

* **Erosion surface** `Ebar = L + a r(c)`: `L` is the `q`-quantile of `P`,
  `r` a smooth map (3-octave fBm plus a regional tilt, `|r| <= 1`). The cut
  is `E = min(max(P, Ebar), P + 0.8 (z_0 - P))`.
* **Truncation**: `z'_k = max(z_k, E)` for `k <= u`.
* **Ponded fill**: `F = max(E - Ebar)`, `D = Ebar + F - P`, `S = rho max D`,
  `w_k = clamp(1 - mean(P - z_k) / S, 0, 1)`, `z'_k = min(z_k + w_k D, E)`
  for the horizons above `u`. The seabed never moves, layers only get
  thicker upward, and they onlap `E` where `z_k + w_k D` reaches it.
* **Dip limit**: `a <- min(a, s_max / max|grad r|)`. The default `s_max` is
  1 sample per column, so the erosion surface is never steeper than that.
  The steepest horizon of the stack does not get steeper (global maximum
  only, see "Slopes" below).
* **Exhumation**: the burial of every interval at or below `s` grows by the
  eroded thickness `e` in metres (`--no-exhumation-burial` to turn it off).
  The term is added to the depth of every voxel kind, the layer voxels of
  both rock-physics kernels and the hidden-interval parts of the partial
  voxels. It is a proxy for the missing overburden (a pure depth shift of
  the compaction trend), not a model of the removed section: the velocity
  step across the surface is small, `dVp` p10 / p50 / p90 = 2.8 / 15.4 /
  35.9 m/s. The step is the exhumation term only. The net velocity step
  across the surface is not sign-consistent: with the default layer shifts
  it is negative in 52% of columns.
  What holds in every eroded column is that the exhumation term itself
  raises Vp on the label voxel and nowhere above it
  (`t9_velocity_step_sits_on_the_label_interface`, both kernels).
* **Shale seal**: `--unconformity-shale below|above|both|none` (default
  `below`) forces the layer at the surface to shale.

The draws use `STREAM_UNCONF = 0x0C0F` and are a pure function of
`(seed, shape, config)`. The result does not depend on tiling, threads or
processes.

## Output

`data/unconformity_labels` (u8, one voxel per column): the number of the
unconformity (1 = shallowest) at the layer boundary of the surface in each
column, carried through the fault lookup and cleared in salt. It is written
only when an unconformity was applied (time mode too). The multi-process
verifier compares it with the single-pass reference.

With one unconformity (N = 1) every column outside salt has exactly one label
voxel, and none inside salt. That is the only case where one label voxel per
column holds. With N = 2 and in time mode some columns carry fewer voxels
than there are unconformities (see "Known limits" below). By mode:

* **Depth, no faults**: the first sample at or below the surface (the layer
  boundary, as before).
* **Depth, faults**: the first output sample whose source position, read
  through the column's fault lookup, is at or below the surface. A throw
  can neither duplicate nor skip it. The voxel is in layer `s`, or in a
  deeper layer where a throw skips layer `s` altogether (89% in layer `s`
  in `t11_faults_displace_the_mask_and_the_layers_together`).
* **Time mode (Vp-TWT axis)**: the surface depth is converted through the
  column's own time map (`time_mode::resample_marks`) and the label is the
  first time sample at or below it. The legacy axis
  (`--legacy-depth-as-time`) equals the depth labels exactly.

### Known limits

* **N = 1** (depth): 4 faults, 40 x 36 x 128, seeds 4, 7, 12, 3 (5760
  columns): 0 columns with two voxels and 0 with none (the earlier
  nearest-sample rule gave 97 duplicated and 754 vanished over the four
  seeds, 91 and 294 of 1440 columns on seed 4 alone).
* **N = 2** (depth): near the cube bottom the second surface can fall
  below the last sample, so mark 2 is missing in those columns: 30 columns
  in depth mode with faults.
* **Time mode** (Vp-TWT axis): the label is the first time sample at or
  below the converted surface, so a mark below the last time sample is
  dropped: 75 columns (3.3%) without faults, 359 columns with faults.

The verifier and the tests check the labels against the crossing rule, not
against a count of one per column
(`t8_labels_are_one_voxel_per_column_at_the_layer_boundary`,
`t8_time_mode_labels`). The array is identical across every writer, tiling,
worker and process count
(`t6_every_writer_produces_the_same_store_content` and
`t3_tiling_worker_and_process_invariance_with_the_unconformity_active_64_columns`).

## Slopes

The dip limit applies to the erosion surface. The folded stack can still
have steep horizons, and the pass does not make the steepest one steeper:
the largest slope over all horizons after the pass is at most
`max(largest before, s_max)` in every seed tested
(`t5_no_horizon_steeper_than_before_on_folded_stacks`). That is a statement
about the global maximum only. An individual younger horizon can be steeper
than it was, because it is clamped to the erosion surface. Over 200 seeds
(default folds, no faults):

| Cube | N | Seeds with a steeper horizon | Largest increase | Largest slope before = after |
|---|---|---|---|---|
| 64 x 64 x 256 | 1 | 106 of 200 | +0.393 | 3.4993 |
| 64 x 64 x 256 | 2 | 143 of 200 | +0.357 | 3.4993 |
| 64 x 48 x 128 | 1 | 82 of 200 | +0.189 | 2.5256 |
| 64 x 48 x 128 | 2 | 103 of 200 | +0.252 | 2.5256 |

With dip-limited folds (`max_dip` 0.5) and N = 1: 105 of 200, up to +0.413,
largest slope 2.3742 (64 x 64 x 256); 75 of 200, up to +0.258, largest
slope 1.7883 (64 x 48 x 128). The slopes are in samples per column.

## Hidden intervals

Truncation and onlap taper layers to zero thickness, so an interval with a
positive thickness but no whole voxel (a sub-voxel wedge tip) is expected
and legitimate. With partial voxels on (default), no salt, no faults,
64 x 64 x 256, 200 seeds, columns with at least one hidden interval,
mean / max: N = 1 10.0% / 24.7%, N = 2 18.9% / 37.5% (#56 and the legacy
switches: none). The exhumation term reaches these parts too
(`partial_model::tests`). Measured by
`t10_hidden_interval_column_rates` (ignored, nightly).

## Flags

| Flag | Default | Range |
|---|---|---|
| `--unconformities N` | 1 | 0, 1 or 2 |
| `--unconformity-position F` | drawn | 0.1-0.9 |
| `--unconformity-level Q` | drawn in [0.35, 0.65] | 0.1-0.9 |
| `--unconformity-relief-m R` | drawn | 0-200 |
| `--unconformity-max-dip S` | 1 | 0.05-2 |
| `--unconformity-fill-ratio P` | 2 | 1-6 |
| `--unconformity-shale M` | below | below, above, both, none |
| `--no-exhumation-burial` | burial on | |
| `--legacy-unconformity` | off | |

`--legacy-unconformity`, `--unconformities 0`, `--legacy-fold-geometry` and
`--legacy-toy-depth` give the geometry of #56 (analytic folds) or earlier
byte for byte. An unconformity flag together with a legacy switch, with
`--toy-geometry planar`, or `--unconformity-position` with `--unconformities
0` exits with code 2.

An unconformity is skipped when the stack is too short to keep a 4-index
margin: `nh < 10` for one, `nh < 16` for two (the default 8x8x32 and 32^3
cubes have none).

## Measured

See the PR body for the closure fractions, timing and the slope table
(`cargo run --release -p synthoseis-core --example unconformity_demo --
sweep|slopes|phases|OUT`, figures from `examples/plot_unconformity_demo.py`).

## Timing

Default runs take 7.9-8.0 s and `--legacy-unconformity` runs 7.8 s on the
same cube: the difference is within run-to-run noise, so the unconformity
pass has no measurable cost there and the default is not faster or slower
than the legacy switch. Both are faster than the #56 head, for a reason
that has not been traced, so the speed-up is not credited to the
unconformity.
