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
  1 sample per column, so the bevel is never steeper than that. Folds can
  still make a steeper horizon; the bevel does not add to it
  (`tests/unconformity.rs::t5_no_horizon_steeper_than_before_on_folded_stacks`).
* **Exhumation**: the burial of every interval at or below `s` grows by the
  eroded thickness (`--no-exhumation-burial` to turn it off).
* **Shale seal**: `--unconformity-shale below|above|both|none` (default
  `below`) forces the layer at the surface to shale.

The draws use `STREAM_UNCONF = 0x0C0F` and are a pure function of
`(seed, shape, config)`. The result does not depend on tiling, threads or
processes.

## Output

`data/unconformity_labels` (u8, one voxel per column): the number of the
unconformity (1 = shallowest) at the layer boundary of the surface in each
column, carried through the fault lookup and cleared in salt. It is written
only when an unconformity was applied (time mode too, point sampled through
the column's own time map). The multi-process verifier compares it with the
single-pass reference.

### Known limit: the one voxel per column is not always one

The label is a single voxel per column only before faults and before the
time conversion. After either, it is the faulted or resampled copy of that
voxel and can be duplicated or lost:

* **Faults.** The label is carried through the same lookup as the layer
  labels, so every labelled voxel stays within one voxel of a layer boundary
  of the faulted labels (5103 of 5103 in the test below), but a fault
  throw can map two source samples of a column onto the label voxel
  (duplicated: two voxels) or skip it (vanished: no voxel). Measured by
  `t8_labels_follow_faults_and_are_cleared_in_salt`: 40 x 36 x 128, 4 faults,
  no salt, columns with a duplicated / vanished label voxel against the
  fault-free run:

  | Seed | Duplicated | Vanished | Columns |
  |---|---|---|---|
  | 4 | 91 | 294 | 1440 |
  | 7 | 6 | 328 | 1440 |
  | 12 | 0 | 0 | 1440 |
  | 3 | 0 | 132 | 1440 |
  | all four | 97 | 754 | 5760 |

  So 97 duplicated and 754 vanished is the total over the four seeds
  (15% of columns); on seed 4 alone it is 91 and 294 (27%).
* **Time mode (Vp-TWT axis).** The depth label is point sampled through the
  column's own time map, so a column can end up with two voxels. Seed 4,
  32 x 28 x 128, no salt: 802 columns carry one voxel, 94 carry two, none
  carry zero (of 896). Never three. The legacy axis
  (`--legacy-depth-as-time`) equals the depth labels exactly.

Consumers should treat `data/unconformity_labels` as a thin marker of the
surface (a column may hold 0, 1 or 2 voxels after faults, 1 or 2 in time
mode), not as a count of columns. The array is identical across every
writer, tiling, worker and process count
(`t6_every_writer_produces_the_same_store_content` and
`t3_tiling_worker_and_process_invariance_with_the_unconformity_active_64_columns`).

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
