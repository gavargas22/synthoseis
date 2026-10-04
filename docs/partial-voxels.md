# Partial voxels — kernels (PR A) and wiring (PR B1)

Spec: `spec-partial-voxels.md` (Notion copy:
https://app.notion.com/p/3ee0272f1f7b8153b164f797a23bb987).

PR A adds the numerical kernels for partial voxels. PR B1 wires them into
every path (classic, chunked/streaming, overlap, strip-stitch,
multi-process, geometry-once) behind two switches. **In B1 the default is
still whole voxels**, so every store written without the new flag is
byte-identical to master. PR B2 flips the default to partial voxels.

## Switches (PR B1)

| Flag | Effect |
|---|---|
| (none) | whole voxels (B1 default) |
| `--legacy-whole-voxels` | whole voxels, explicitly (the B2 opt-out); same bytes as the default |
| `--partial-voxel-reflectivity subcell` | partial voxels, sub-cell interfaces at exact ray times (time axis only) |
| `--partial-voxel-reflectivity cell` | partial voxels, Backus voxels with the cell-to-cell reflectivity (time axis or `--legacy-depth-as-time`) |

- Exit 2: `--legacy-whole-voxels` with `--toy-geometry planar` /
  `--legacy-toy-depth`; `--partial-voxel-reflectivity` with
  `--legacy-whole-voxels`, with the planar geometry, or `subcell` with
  `--legacy-depth-as-time`.
- Multi-process workers receive the flag (`cli_jobs::rock_physics_args`).
- Library: `RockPhysicsConfig::partial_voxels` (`PartialVoxelConfig`);
  `E2eConfig::effective_partial_voxels()` gives the mode (layered geometry
  only; enabled without a mode = subcell in time mode, cell on the legacy
  axis).

### What changes with the switch on

- **Properties (C, and the end-members of S).** `RpmModel::column_partial`
  runs the whole-voxel column first, then rewrites mixed cells only: each
  part's end-member is `voxel_properties` with that part's kind at the TVDML
  the label-run logic gives the adjacent (±1) cell labelled with that
  interval, else the map TVDML (spec §3.4); the cell gets their Backus mix.
  Pure cells keep the whole-voxel bits.
- **Time axis.** T is the slowness sum through each cell's parts. S places
  every sub-cell interface at its exact time (`subcell_reflectivity`); C uses
  the Backus voxels at the same T. The same T feeds the time labels, the
  noise seabed time and the time summaries.
- **Labels.** Depth labels, fault labels and salt labels are unchanged. Time
  labels are point-sampled through the partial T, so they move by at most
  one sample where T crosses a sample boundary.
- **Root attributes** (switch on only): `voxel_model: "partial-z"`,
  `partial_voxel_mixing: "backus"`, `partial_voxel_reflectivity:
  "subcell"|"cell"`. No new arrays.
- **Summary line** (stdout, switch on only):
  `partial voxels: mixed=…% of cells, >=3-unit=…, hidden-intervals=…,
  label-guard=… (…% of sediment cells), below-base=…, water-below-seabed=…,
  mode=…`.

### Faulted columns: pull-back (spec §3.3)

- σ_k = L(k) + ½ from the fault tile lookup. Breaks where any fault's
  `inside()` flips, where the lookup is clamped, at a jump (Δσ > 1.5) or at a
  fold (Δσ ≤ 0). Edges are midpoints, or σ ∓ g/2 at a break (g from the
  unbroken side, 1 if both sides break). Horizon crossings map to
  `ζ = k + (z_h − σ⁻)/g`; salt and contacts are output coordinates.
- **Deviation from the spec's safety rule.** The spec breaks on
  `|Δσ − 1| > ½`, which also breaks smooth compression (fault drag,
  Δσ ≈ 0.05). Those cells then get a unit-width source window around a
  nearly constant σ and mix, over many cells, a unit the labels do not have
  (a 6 ms T change and 2-sample time-label moves at 10×12×48). B1 keeps
  midpoint edges for 0 < Δσ ≤ 1.5. At 64×64×256 the guard counts equal the
  spec rule's for 4 of 5 seeds (seed 1: 0.174 % vs 0.162 %).

### Fallbacks (counted, logged in the summary)

- **label-guard**: a cell (pure or mixed) with no fraction of its labelled
  unit stays whole. 0 unfaulted; with 4 faults 0.013–0.36 % at 64×64×256.
- **below-base**: a mixed cell with a part below the deepest horizon
  (whole-voxel `Unfilled`) stays whole.
- **water-below-seabed**: a fault lookup that folds back near the seabed
  maps cells below the output seabed above the source seabed; those water
  parts take the cell's sediment unit (`absorb_water_below`).
- **hidden-intervals**: column-intervals with a fraction, no labelled cell,
  and both horizons rounding to the same sample (the probe's definition;
  intervals that the cube bottom truncates inside the last cell are not
  counted). They are modelled with their map-TVDML end-member.

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

Setup: 40 Hz Ricker, dt = 1 ms, dz = 4 m, end-members at 1 km, production
water (1500/1000/1.028) and salt (4500/2250/2.17) constants. Errors are
measured against the analytic convolutional trace.

**Dipping interface, `z_b = 30 + 0.137 i` (spec §5.1/§5.2).** Values are
rms / max time error in ms.

| Pair | whole | S (sub-cell) | C (Backus, slowness T) | C, T from Backus Vp (reported) |
|---|---|---|---|---|
| shale/gas | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.057 / 0.084 | 0.062 / 0.093 |
| shale/brine | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.050 / 0.075 | 0.050 / 0.075 |
| shale/salt | 0.902 / 1.532 | 0.0001 / 0.0001 | 0.020 / 0.037 | 0.052 / 0.083 |
| water/shale (1 km) | 1.552 / 2.635 | 0.0001 / 0.0001 | 0.060 / 0.114 | 0.539 / 0.857 |
| water/mudline shale (1580/279/1.957) | 1.552 / 2.635 | 0.0001 / 0.0001 | 0.260 / 0.381 | 0.375 / 0.643 |

- **S:**
  - amplitude error 0.009 %;
  - residual 0.017 % of the peak;
  - E = 2e-8;
  - staircase ratio ≥ 8900×.
- **C:** staircase ratio 15.7–45× in sediments, 6× over mudline shale.

**Wedge, h = 0.08–12 m, 7 fractional top positions (spec §5.3).** Values
are the error as a share of `|R_top|`.

| Thin layer | whole (max) | S (max) | C (max / mean) |
|---|---|---|---|
| gas | 75 % | 0.025 % | 17.5 / 5.5 % |
| salt | 64 % | 0.020 % | 7.9 / 2.5 % |

**Also tested:**
- Σf = 1 per cell (1e-12), and Σ_k f_u equals the unit thickness (1e-9).
- Salt mass differs from the salt label count by less than 1 per column.
- Centre rule = labels: 0 mismatches, and the label guard fires 0 times
  unfaulted (5 seeds × salt on/off).
- The fractions are bit-identical across chunk shapes [1,1,nz], [5,7,nz],
  [3,20,16] and [8,5,16], across k-splits, and in a deeper cube.

## Pipeline validation (PR B1, `tests/partial_voxels_pipeline.rs`,
`synthoseis/tests/partial_voxels_cli.rs`)

- Switch on: classic, chunked, streaming and overlap at chunk shapes
  [1,1,nt], [5,7,nt], [3,20,16], [8,5,16], strip-stitch 2/3/4,
  multi-process 1/2/3 and geometry-once write identical stacks, labels and
  attributes: S in time mode (seed 30 with salt + 3 faults, seed 7 with
  salt + 4 faults; nt = nz + 37, dt = 2 ms, dz = 2 m, RICH filters), C in
  time mode, C on the legacy axis. The CLI multi-process store equals the
  single-process store with the flag.
- Switch off is the default; `--legacy-whole-voxels` gives the same store;
  S ≠ C ≠ off; the attributes appear only with the switch on.
- Labels (5 seeds × salt on/off × 0/4 faults): legacy-axis labels, fault
  labels and salt labels identical on vs off; time labels move by at most
  one sample (0.50 % of samples at 10×12×48), no new classes, fault ∧ salt
  = 0.
- Label guard with 4 faults ≤ 1 % of sediment cells (5 seeds × salt on/off
  at 32×32×128: max 0.70 %); 0 unfaulted; 0 hidden intervals without salt.
