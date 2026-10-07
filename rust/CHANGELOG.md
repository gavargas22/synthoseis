# Changelog (Rust CLI and crates)

Output-changing defaults are listed here with their opt-out flag.

## Unreleased

### Changed

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
