# Changelog (Rust CLI and crates)

Output-changing defaults are listed here with their opt-out flag.

## Unreleased

### Changed

- **Fault heave** (spec "fault kinematics", 7a). Faults slip down the dip
  of their surface, so the hanging wall also moves sideways (heave = throw /
  tan dip, with a dip floor `--fault-min-dip`, default 30 degrees, range
  15-60) and horizons show a gap along each fault. Crossing faults compose
  youngest first; rock physics reads each voxel's source position; fault
  labels are a one-voxel band around the surface. Every faulted cube
  changes (stack, labels, fault labels); unfaulted cubes do not. Opt out
  with `--legacy-fault-kinematics` (byte-identical to ce905d9e). The summary
  prints `fault kinematics: heave, min dip D deg` or the legacy line.

- **Angular unconformity: bevel and ponded fill** (spec "erosion and
  onlap"). Every layered cube with at least 10 horizons gets one erosion
  surface, cut at the 0.35-0.65 quantile of a folded horizon, with the
  older layers truncated against it and the younger layers ponding and
  onlapping in the hollows. Exhumed intervals get the eroded thickness as
  extra burial and the layer at the surface is forced to shale. Horizons,
  labels, closures and stacks change. New output `data/unconformity_labels`
  (u8, at the first sample at or below the surface; exactly one voxel per
  column outside salt for one unconformity, in depth, faulted or not. With
  two unconformities mark 2 is missing near the cube bottom, and in time
  mode a mark below the last time sample is dropped. Seed 2, 48x48x256
  (2304 columns), no salt, N = 2, dt 4 ms: depth 0 missing without faults
  and 30 (1.3%) with 4 faults; time 75 (3.3%) without faults and 389 with
  4 faults (359 from the time resampling plus the 30 of depth mode). Seeds
  1, 3, 4, 5, 6: 0 missing everywhere; over the six seeds: no faults 0
  (depth) and 0.27% (time), 4 faults 0.11% and 1.41%. N = 1: 0 missing and
  0 duplicated), written only when
  an unconformity was applied. The exhumation term is a depth
  shift of the compaction trend, a proxy rather than a model of the removed
  section. See `docs/unconformity.md`.
  The default 8x8x32 and 32^3 cubes are below the size limit and do not
  change. See `docs/unconformity.md`.
  - Opt-outs: **`--legacy-unconformity`** or **`--unconformities 0`**
    reproduce #56 byte for byte; `--legacy-fold-geometry` and
    `--legacy-toy-depth` imply no unconformity.
  - New flags: `--unconformities`, `--unconformity-position`,
    `--unconformity-level`, `--unconformity-relief-m`,
    `--unconformity-max-dip`, `--unconformity-fill-ratio`,
    `--unconformity-shale`, `--no-exhumation-burial`.
  - The whole-run MDIO writers now build the output-domain `fault_labels`,
    `salt_labels` and `unconformity_labels` in one pass
    (`time_mode::generate_output_masks`) instead of one model rebuild per
    mask. Store content is unchanged.

- **Salt partial-voxel maps keep sub-sample positions** (spec "salt
  partial-voxel far-field maps"). With salt on, the partial-voxel horizon
  maps were built from the *rounded* maps plus the drag, so far from salt
  every interface snapped to a whole sample (partial voxels were
  effectively off in every salt cube, including the 8³ default whose salt
  body has 0 voxels in the cube). They are now the continuous maps plus the
  production drag offset (`salt::drag_offsets`; hit test and lift counter
  still on the rounded maps), min-cascaded, then clamped into the rounding
  cell of the production map, so `round(z) == toy_horizon_maps` still
  holds. Far from salt the maps equal `--no-salt` bit for bit.
  **Depth labels, legacy-axis `labels` / `salt_labels` / `fault_labels`
  are unchanged.** **Salt-on partial-voxel stacks change** (time mode and
  the legacy axis; 64×64×256 demo: 94.5 % of cells, rel. RMS 0.40 at 15°)
  and **time-domain labels change**, because the traveltime follows the
  partial-voxel Vp (2.1 % of samples on the demo; time `salt_labels` and
  `fault_labels` move with them). Far-field 15° stack vs `--no-salt`: rel.
  RMS 0.43 → 0.0002, far time labels 13 326 → 0. Nothing changes with
  `--no-salt`, `--legacy-whole-voxels`, the planar geometry or
  `--legacy-toy-depth`. Applies under `--salt-smooth-all-horizons` too.
  - Opt-out: **`--legacy-salt-pv-maps`** /
    `RockPhysicsConfig::salt_legacy_pv_maps = true` restores master
    a60ec34a bit for bit on every path (forwarded to multi-process workers;
    exit 2 with `--no-salt`, planar, `--legacy-toy-depth` or
    `--legacy-whole-voxels`).
  - Summary line (salt and partial voxels on): `salt partial-voxel maps:
    continuous (clamped to the label cell)` or `salt partial-voxel maps:
    rounded (--legacy-salt-pv-maps)`. No store attribute.
  - The partial-voxel counters return to the `--no-salt` level (32² demo:
    mixed 8.1 % → 11.9 %, label guard 0.015 % → 0.46 %, under the 1 %
    limit).
  - `cli_jobs::worker_args()` now builds the whole worker argv; the spawn
    site and the worker round-trip test share it.

- **Physical filter edges in time mode** (spec "filter edge handling").
  The filters now see water above time 0 and the model's own reflectivity
  below the window (then the half-space below the model base), instead of
  zero padding (Ricker), a 27-sample odd mirror (bandpass) and window-only
  noise. The bandpass filters all `nt` samples (the #36 dead last sample is
  gone in time mode), noise is drawn in the pads too, and the cube sides
  keep reflect. **Stacks change, time mode only**: with the Ricker alone
  only the last 7 samples of columns that continue below the window move
  (seed 7: 4.65 % of cells, rel. RMS 0.017); with `--bandpass` every sample
  moves (rel. RMS 0.17 against the old edges, which were off from the
  physically correct answer by that much). **Labels, salt_labels and
  fault_labels are unchanged**; the legacy depth axis is unchanged.
  - Opt-out: **`--legacy-filter-edges`** /
    `TimeConfig::legacy_filter_edges = true` restores master 1c22b653 bit
    for bit on every path (exit 2 with `--legacy-depth-as-time` /
    `--legacy-toy-depth`).
  - Summary: `filter edges: physical (water above, model below; reflect
    sideways)` or `filter edges: legacy 1c22b653 (--legacy-filter-edges)`.
    Store attribute `filter_edges = "physical"` (omitted with the opt-out).
  - The bandpass no longer needs more than 27 samples per trace in time
    mode. With `--keep-ricker` the top pad is `h` (8 at 4 ms, 16 at 2 ms) so the Ricker
    precursor above time 0 reaches the bandpass.
  - Order 6 at 2 ms carries up to ~7e-4 of peak (6–20 Hz: 3.5e-4) of
    transfer-function (`ba`) round-off in both edge modes; use order ≤ 5
    or dt 4 ms (SOS conversion is a deferred follow-up).
  - API: `IirFilter::edge_pad()` (1e-6 tail, cap 8192),
    `IirFilter::filtfilt_padded_f32`, `WeightedNoise::sample_pad`,
    `time_mode::{EdgePads, TraceChain, finish_trace_padded,
    fuse_tile_time_chain}`.

- **Salt Vs defaults to 2600 m/s** (spec "salt Vs ≈ 2600"). Density and Vp
  stay 2.17 / 4500. Legacy and master f15b87ac used Vs 2250 (Vp/Vs = 2.0,
  Poisson 0.33); real halite is closer to Vp/Vs ≈ 1.7–1.8. **Angle stacks
  change at salt contacts (AVO)**, far angles more (0° bit-identical; 30°
  top-salt reflection about 30–40 % weaker); labels / salt_labels /
  fault_labels and depth→time pull-up are unchanged.
  - Opt-out: **`--salt-legacy-vs`** /
    `RockPhysicsConfig::salt_legacy_vs = true` restores Vs 2250 and master
    f15b87ac salt-on output bit for bit.
  - Summary: `salt properties: rho 2.17, Vp 4500, Vs 2600` or
    `… Vs 2250 (--salt-legacy-vs)`.
  - API: [`synthoseis_rpm::salt_elastic`]`(legacy_vs)` is the single source
    of truth (core property fill uses it; no divergent `SALT_VS` in
    `salt.rs`).

- **Salt horizon drag is lift-only by default** (spec "lift-only salt
  smoothing"). Today's (and legacy's) rule was `G_σ3(m + L)` with
  `L = −2r · salt_mask`, which also blurred the undragged map far from
  salt. The default is now `m + G_σ3(L)`: same flank drag, far-field maps
  bit-identical to the undragged maps before push-down. The salt body /
  hull is unchanged in depth; the time-domain `salt_labels` (default time
  output) can still move by a sample where the overburden traveltime
  changes (docs/salt-bodies.md, "Time output and velocity pull-up").
  **Stacks and labels change** wherever salt is on.
  - Opt-out: **`--salt-smooth-all-horizons`** /
    `RockPhysicsConfig::salt_smooth_all_horizons = true` restores master
    ccce5cc9 salt-on output bit for bit.
  - Summary: `drag: lift-only` or `drag: smooth-all (--salt-smooth-all-horizons)`.

- **Closure minimum scaled with the cube** (spec "cube-size-scaled closure
  minimum"). A closure compartment needs `clamp(round(ni·nj / 180), 20,
  500)` whole cells to get a fluid draw instead of a fixed 500 (legacy
  `min_closure_voxels_simple`, set for its 300 × 300 × 1250 cube). **Stacks
  change** wherever a trap between the new minimum and 500 voxels exists
  (most cubes below 300 × 300); time labels and the time-domain fault and
  salt cubes move with them; depth, fault and salt labels are unchanged.
  Cubes with ni·nj ≥ 89,910 keep 500. Counting stays in whole cells under
  partial voxels.
  - Opt-out: **`--legacy-closure-minimum`** (CLI) /
    **`ClosureMinimum::LEGACY`** (the `Legacy` variant, 500; library).
    `--min-closure-voxels 500` / `Fixed(500)` applies the same threshold
    under its own label.
  - New: **`--min-closure-voxels N`** (N ≥ 1) / `ClosureMinimum::Fixed(N)`.
  - Scaled stores carry the root attributes `closure_min_voxels` and
    `closure_minimum = "scaled-area"` (none with a fixed minimum), and the
    run summary prints `closures: minimum T voxels (...), kept K of C
    compartments`.
  - Not ported: legacy's 2,500-voxel minimum for faulted closures (Rust has
    no closure types).
- **Partial-voxel closure contact reaches the true unit base.** With
  partial voxels, a segmented closure that fills its sand unit now stores a
  fluid contact capped at base + ½ cell instead of the base cell, so the
  sub-cell sand sliver below the base cell is hydrocarbon instead of brine
  (+0.6 to +1.1 % hydrocarbon volume per cube, no trap loses volume).
  Whole voxels are bit-identical.
  - Opt-out: **`--legacy-closure-contact-cap`** /
    `RockPhysicsConfig::legacy_closure_contact_cap = true`.
- With both `--legacy-closure-minimum` and `--legacy-closure-contact-cap`
  the output is master bad1daa8 byte for byte.
- API: `RockPhysicsConfig::min_closure_voxels: usize` is replaced by
  `closure_minimum: ClosureMinimum` (use `ClosureMinimum::voxels(ni, nj)`
  for the applied threshold; variants `Scaled`, `Fixed(N)`, `Legacy`, plus
  `ClosureMinimum::LEGACY` and `LEGACY_VOXELS`); new
  `legacy_closure_contact_cap`, `ClosureRun::fluid_contact`,
  `closure_segments::segmented_sand_unit_fluids_with`, `closure_census` /
  `ClosureCensus` and `write_closure_attrs`.
- **Partial voxels are on by default** (PR B2; spec "partial voxels"). The
  CLI now Backus-mixes cells that straddle a horizon, the seabed, a salt top
  or a fluid contact, and on the time axis places every sub-cell interface at
  its exact ray time (`--partial-voxel-reflectivity subcell`; `cell` on
  `--legacy-depth-as-time`). **Stacks change for every layered-geometry run
  without a flag**; time labels (and the time-domain fault and salt cubes)
  move where the traveltime crosses a sample boundary, mostly by one sample;
  depth, fault and salt labels are unchanged. Partial-voxel stores carry
  `voxel_model = "partial-z"`, `partial_voxel_mixing = "backus"` and
  `partial_voxel_reflectivity`.
  - **Library default too:** `PartialVoxelConfig::default()` (in
    `RockPhysicsConfig::default()` / `E2eConfig::default()`) is partial
    voxels (`Subcell` in time mode, `Cell` on the legacy axis).
  - Opt-out: **`--legacy-whole-voxels`** (CLI) /
    **`PartialVoxelConfig::whole_voxels()`** (`legacy_whole_voxels: true`,
    library) reproduces the previous default (master d8b96e69) byte for
    byte.
  - API: B1's `PartialVoxelConfig::enabled` field is replaced by the
    `legacy_whole_voxels` field and an `enabled()` method.
  - Unchanged: the planar geometry and `--legacy-toy-depth` (always whole
    voxels). The Python extension `synthoseis_mdio` is MDIO I/O only (no
    generation API), so it has no partial-voxel option.
  - The default equals d8b96e69 run with `--partial-voxel-reflectivity
    subcell` (time axis) or `cell` (legacy axis), and in the library
    d8b96e69 with `PartialVoxelConfig::on()`.
  - See `docs/partial-voxels.md`.

### Earlier

- This file starts with PR B2 (#43). Earlier output-changing defaults (each with
  its `--legacy-*` / restore flag) are in the PR history: #31–#34, #38–#40;
  `--partial-voxel-reflectivity` and `--legacy-whole-voxels` landed in #42
  (PR B1) and the kernels in #41 (PR A).
