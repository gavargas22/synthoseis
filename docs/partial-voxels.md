# Partial voxels — kernels (PR A)

Spec: `spec-partial-voxels.md` (Notion copy:
https://app.notion.com/p/3ee0272f1f7b8153b164f797a23bb987).

PR A adds the numerical kernels for partial voxels. **None of them is wired
into the pipeline yet**, so every output is byte-identical to master
`d51ab237`. PR B1/B2 will wire them in behind `--legacy-whole-voxels` and
`--partial-voxel-reflectivity subcell|cell`.

## Coordinates

- The depth index ζ is in samples. Cell k is `[k, k+1)` and its centre is
  k + ½. This is the depth-to-time convention.
- Boundary positions (spec §1.1):

| Boundary | Position |
|---|---|
| horizon / seabed h | unrounded `z_h` |
| salt top / base | `lo + ½`, `hi + ½` (continuous hull bounds) |
| fluid contact of interval h | `contact` (HC where ζ < contact) |

- Rounding these positions reproduces today's labels exactly. That is the
  centre rule, and ties go to the shallower unit. So labels are never
  recomputed.

## Kernels

### Fractions: `synthoseis_core::partial_voxels`

- **`column_parts` / `column_parts_range`.** These give the exact 1D
  vertical overlap fractions `f_u(k) = |[k,k+1) ∩ [top_u, base_u)|` for
  every cell of one column.
  - Every boundary is sorted once, then merged with the cell edges by a
    moving pointer.
  - Each sub-interval is classified at its midpoint, in this priority order:
    salt > water (ζ < z_0) > interval h (HC if ζ < contact) > below the
    deepest horizon.
  - Neighbours of the same kind are merged.
  - Fractions are computed per column and are transient: nothing new is
    stored per voxel.
- **`EPS_FRAC = 1e-6`.**
  - A boundary within `EPS_FRAC` of a cell edge snaps to that edge, so
    integer boundaries give pure cells.
  - A boundary within `EPS_FRAC` of the previous breakpoint is dropped, so
    there are no sliver parts and the fractions of a cell sum to 1.
- **`cell_model`.**
  - Pure cells short-circuit to the whole-voxel path, so they stay
    bit-identical.
  - If a mixed cell has no part of its labelled unit (f = 0), it falls back
    to the whole voxel. This is the label-support guard, and the fallback is
    counted in `PartialVoxelStats::label_guard`.
- **`hc_fraction`.** Gives f_HC per cell: the hook for later closure work
  (spec §3.7).

### Continuous geometry (byte-preserving refactors)

| Function | What it returns |
|---|---|
| `toy_geometry::layered_horizon_maps_continuous` | the f64 stack before `.round()`, with the thickness push applied as an exact `min` cascade |
| `pipeline_stream::toy_horizon_maps_continuous` | with salt, the drag of the rounded maps without the final re-round (spec §3.1: the sub-sample position below the drag comes from the drag) |
| `salt::hull_bounds` / `SaltBody::hull_bounds` | the continuous per-column `(lo, hi)` that `hull_runs` rasterises; `hull_runs` now calls the shared per-column helper |

- `round(continuous)` equals the production maps bit for bit. The unit test
  checks this on 6 seeds × 2 shapes, with and without salt.

### Backus mixer: `synthoseis_rpm::backus_mix`

- Density is the arithmetic average. The moduli are harmonic averages:
  - P-wave modulus M = ρVp²;
  - shear modulus μ = ρVs²; if any part has μ = 0 (a fluid), the mix has
    μ̄ = 0.
- The output is Vp = √(M̄/ρ̄) and Vs = √(μ̄/ρ̄).
- The arithmetic runs in f64. A single part, or parts with identical bits,
  return the end-member bits.
- `slowness_sum` gives the ray time through a mixed cell (spec §1.4).
- `voigt_mix` is the upper bound, used in tests.
- The legacy NTG rule `BackusModuli` (`legacy.rs`) is untouched. It averages
  λ, not M, and is ill-conditioned at the seabed.

### Sub-cell splitter: `synthoseis_seismic::subcell_column`

- Splits each cell into pure sub-layers.
- Gives every internal boundary its exact two-way time:
  `t_j = T_k + 2dz Σ_{j'≤j} f_j'/Vp_j'`.
- Gives the cell boundaries their times `T_k`.
- `subcell_reflectivity` computes each boundary's Zoeppritz coefficient and
  inserts it with the existing `insert_spikes` (windowed sinc).
- With pure cells, both functions reproduce `twt_column` and
  `reflectivity_time_column` bit for bit.

## Validation (`synthoseis-core/tests/partial_voxels.rs`)

Setup: 40 Hz Ricker, dt = 1 ms, dz = 4 m, end-members at 1 km. Errors are
measured against the analytic convolutional trace.

**Dipping interface, `z_b = 30 + 0.137 i` (spec §5.1/§5.2).** Values are
rms / max time error in ms.

| Pair | whole | S (sub-cell) | C (Backus, slowness T) | C, T from Backus Vp (reported) |
|---|---|---|---|---|
| shale/gas | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.057 / 0.084 | 0.062 / 0.093 |
| shale/brine | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.050 / 0.075 | 0.050 / 0.075 |
| shale/salt | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.020 / 0.037 | 0.051 / 0.083 |
| water/shale | 1.552 / 2.635 | 0.0001 / 0.0001 | 0.057 / 0.108 | 0.558 / 0.886 |

- **S:**
  - amplitude error 0.009 %;
  - residual 0.017 % of the peak;
  - E = 2e-8;
  - staircase ratio ≥ 8900×.
- **C:** staircase ratio 15.7–44×.

**Wedge, h = 0.08–12 m (spec §5.3).** Values are the error as a share of
`|R_top|`.

| Thin layer | whole (max) | S (max) | C (max / mean) |
|---|---|---|---|
| gas | 64 % | 0.021 % | 17.0 / 4.8 % |
| salt | 47 % | 0.019 % | 7.5 / 2.0 % |

**Also tested:**
- Σf = 1 per cell (1e-12), and Σ_k f_u equals the unit thickness (1e-9).
- Salt mass differs from the salt label count by less than 1 per column.
- Centre rule = labels: 0 mismatches, and the label guard fires 0 times
  unfaulted (5 seeds × salt on/off).
- The fractions are bit-identical across chunk shapes [1,1,nz], [5,7,nz],
  [3,20,16] and [8,5,16], across k-splits, and in a deeper cube.
