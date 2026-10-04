# Changelog (Rust CLI and crates)

Output-changing defaults are listed here with their opt-out flag.

## Unreleased

### Changed

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
  - Opt-out: **`--legacy-whole-voxels`** reproduces the previous default
    (master d8b96e69) byte for byte.
  - Unchanged: the planar geometry and `--legacy-toy-depth` (always whole
    voxels); the library default `PartialVoxelConfig::default()` (off) and
    the Python bindings.
  - The default equals d8b96e69 run with `--partial-voxel-reflectivity
    subcell` (time axis) or `cell` (legacy axis).
  - See `docs/partial-voxels.md`.

### Earlier

- This file starts with PR B2. Earlier output-changing defaults (each with
  its `--legacy-*` / restore flag) are in the PR history: #31–#34, #38–#40;
  `--partial-voxel-reflectivity` and `--legacy-whole-voxels` landed in #42
  (PR B1) and the kernels in #41 (PR A).
