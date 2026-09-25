# Fault modelling port (Python `datagenerator/Faults.py` → Rust `synthoseis-geo::faults`)

First slice of the *geology realism* track: the Rust pipeline can now insert
legacy-style ellipsoidal faults into the geology, displace the layer labels
(and any continuous volume such as geologic age), and write a binary
`fault_labels` volume into MDIO. Faulting is **off by default**
(`FaultConfig::count == 0`), and with it off every existing output stays
bit-identical (a test pins hashes recorded on master `9c0614c`).

## What the legacy code does (per fault, `Faults.build_faults`)

1. **Ellipsoid** (`rotate_3d_ellipsoid`, `create_ellipsoid`). Voxel coordinates
   are rotated about the array origin, around the strike axis
   `(sin(π−dip), cos(π−dip), 0)`, by `θ = atan2(tilt·r, nz)`. Here `r` is the
   distance from the ellipsoid origin `(x0, y0)` to the cube centre. The level
   set is `Σ (p′−o)² / axes²`, with the semi-axes and `z0` scaled by `infill = 10`.
   Voxels with a level below 1 are the hanging wall.
2. **Fault surface** (`get_fault_plane_sobel`). This is the sum of `|sobel|` of
   the hanging-wall indicator along the three axes, divided by a 5³ maximum
   filter and thresholded at 0.5. An exact 0.5 stays 0.5.
3. **Centre** (`get_fault_centre` → `get_middle_z`). Pick a random surface voxel
   below the seabed that is near the middle inline.
4. **Displacement** (`xyz_dis`).
   - Vertical profile: a `general_gaussian` of length `nz + 2·int(10σ)` with the
     peak rolled to the centre sample. It is then rolled down in steps of 5 until
     it is ≤ 1 at the seabed, or until it gives up after `nz − wb` samples.
   - Lateral profile: a 2-D gaussian with variance `16000·(0.0013·int(throw))^1.3258 / L(34)`
     and `vy = coef·vx`, rotated by `atan2` of the centre position and re-rolled
     so that its argmax lands on the centre.
   - The displacement is `d = inside ? lateral·vertical : 0`.
   - Hockey-stick drag applies when throw ≥ 29.75.
5. **Composition** (`apply_xyz_displacement`, `reassign_*`). A trace-wise
   lookup `F ← interp(clip(k − d), F)` is accumulated across faults. The label
   level `L` is carried by the same interpolation, re-binarised (below 0.25 → 0,
   0.25 to 1 → 1), and incremented by `seg > 0.25 && d > 1`. The final mask is
   `L > 0.05`, and the faulted age is `interp(F, age)`.

## Python → Rust mapping

| Python (`datagenerator/Faults.py`) | Rust (`rust/synthoseis-geo/src/faults/`) |
|---|---|
| `_fault_params_random`, `rng.uniform` draws for σ, p, coef in `xyz_dis` | `params.rs`: `RandomFaultConfig`, `sample_random_faults`, `FaultRng` (SplitMix64, one forked stream per fault) |
| per-fault parameter tuple (`x0,y0,z0,a,b,c,dip,strike,throw,tilt`) | `params.rs`: `FaultParams::from_legacy` (+ `with_profile`, `with_center`, `with_lateral_offset` for replay) |
| `rotate_3d_ellipsoid`, `create_ellipsoid` | `model.rs`: `FaultGeometry::{new, value, inside}` (Rodrigues rotation, evaluated per voxel) |
| `get_fault_plane_sobel` (sobel + `maximum_filter`) | `segments.rs`: `segment_block` (tile-local, analytic 3-voxel halo recomputed from `FaultGeometry`) |
| `get_fault_centre`, `get_middle_z` | `model.rs`: `survey_surface` (streamed tile pass), `middle_candidates`, `resolve_fault` |
| `xyz_dis` vertical gaussian, seabed taper loop | `model.rs`: `vertical_profile`, `ResolvedFault::seabed_roll` |
| `xyz_dis` lateral gaussian + `ndimage.rotate` + re-roll | `model.rs`: `throw_variance`, analytic `LateralGaussian` (closed form, no rotated raster) |
| `apply_xyz_displacement` (`np.interp`) | `apply.rs`: `interp_uniform`, `interp_trace_into` (numpy semantics, f32 rounding) |
| label-level update and `> 0.05` threshold | `apply.rs`: `FaultModel::compute_tile` → `FaultTile { lookup, mask, segment_id }` |
| faulted age / facies (`interp(F, age)`) | `FaultModel::apply_to_volume_f32`; categorical labels use `apply_to_labels` / `remap_nearest` (nearest sample) |
| `faulted_depth_maps` | `horizon_depth_from_age` (inverse of the faulted age per trace; used by the parity test) |
| `fault_planes` → `fault_segments` output | MDIO variable `data/fault_labels` (u8 0/1), `synthoseis-io` `write_fault_labels_chunk` / `read_fault_labels_u8` |

Pipeline wiring (`rust/synthoseis-core`):

- `E2eConfig.faults: FaultConfig { count, throw_min, throw_max }`.
- `fault_model(cfg)` resolves the faults once. Its seed is
  `cfg.seed ^ FAULT_SEED_SALT`, and the seabed is the top toy horizon.
- `generate_labels` applies the fault model to the labels.
- `generate_fault_labels` produces the fault mask.
- Tile-wise `fault_labels` writes are wired into chunked, streaming,
  strip-stitch and multiprocess runs, each verified against the single-worker
  reference.
- CLI: `synthoseis run --e2e --chunked --faults N [--shape NI,NJ,NK]`.

## Memory and tiling design

- **Global state is tiny.** It holds only the resolved per-fault parameters:
  geometry, centre, the vertical profile (`nz` floats) and the lateral
  gaussian constants.
- **Resolving a fault** (choosing its centre) needs the set of surface voxels
  below the seabed. That set comes from `survey_surface`, a streamed pass over
  16×16 column tiles that computes the sobel/max-filter segments of the
  analytic ellipsoid with a recomputed halo. Its memory is O(tile + surface),
  never O(cube).
- **Faults are resolved independently and composed in order.** Like the
  legacy loop, each fault's centre and taper come from its own ellipsoid and
  the original seabed. Composition stacks the per-trace lookups in fault
  order: fault *n* displaces the output of faults 0…n−1.
- **Voxel values are local.** Every voxel's displaced value, mask bit and
  segment id is a pure function of the global fault parameters and the
  (i, j) column. `compute_tile(i0..i1, j0..j1)` needs only a 3-voxel analytic
  halo. The result is independent of tile size, worker count and order. Tests
  cover chunk shapes {32², 8², 5×7, 1×32, 16×3×24}, 4-thread strips, strip
  stitch with 4 workers, and multiprocess with 3 workers.

## Parity vs the legacy Python

`tests/fixtures/generate_fault_cubes.py` runs the real `Faults.build_faults`
and `apply_xyz_displacement`, using stub cfg/vols and a recording RNG. It
records the explicit fault parameters plus the draws (centre, σ, p, coef), so
both sides get identical parameters and the random draws are bypassed.
`rust/synthoseis-geo/tests/faults_parity.rs` replays them.

| Set | Cases | Mask IoU / agreement | Horizon depth max abs diff |
|---|---|---|---|
| committed fixture (32×32×48, 4 cases, 1–3 faults, one all-skipped) | 4 | **1.0 / 1.0** (2109, 3594, 2864, 0 voxels) | 5.0e-5 samples (the fixture's 4-decimal rounding); faulted age 4.8e-7 (≈1 f32 ulp) |
| 60-seed sweep (48×48×64, not committed), replay mode | 30 with faults, 118,116 fault voxels | **1.0 / 1.0 in every case** | worst 1.4e-3 samples, 100 % within 0.01 |
| same sweep, analytic lateral argmax (no replay) | 31 active faults | masks identical | 9/31 faults choose the other half-pixel argmax: worst 0.135 samples |

**Known, explained mismatch (analytic mode only).** The legacy lateral
gaussian is centred between pixels and then rotated with a cubic spline.
Opposite pixel pairs `(±½, ±½)` are exactly symmetric, so the argmax always
ties analytically. Python breaks the tie through spline round-off, which is
not reproducible from first principles. Rust picks the first candidate in
row-major order (numpy `argmax`). As a result the lateral displacement surface
can sit one pixel over, which changes horizon depths by ≤ 0.14 samples and
leaves the masks unchanged. The parity harness replays Python's actual choice
through `FaultParams::with_lateral_offset`, and then all quantities match.

## Legacy behaviours kept on purpose (worth a decision)

- **σ is sized for large cubes.** σ is drawn from `U(10·throw − 50, 300)`
  samples, which assumes a cube with more than 1000 samples. On small cubes
  (≤ 128 samples) the vertical profile is almost flat, so the seabed taper
  loop gives up ("seafloor will not have 0 throw"). The hanging wall then
  displaces the whole column, including the seabed, and `interp(clip(k−d))`
  smears the top sample downward. That smear can include a slab of fault-mask
  voxels in the water column. The Rust port reproduces this faithfully (the
  sweep matched Python on those cases).
- **Random draws often miss or go flat.** Many random draws miss a small cube
  entirely and are skipped (`FaultSkip`). Some produce near-horizontal
  "faults" (detachments).

## Deferred (follow-ups)

- Hockey-stick drag zone (throw ≥ 29.75). Faults are flagged
  `hockey_stick_deferred`, and the default throw range 5–29 avoids it.
- Fault intersections volume and dilation; `fault_intersections` output.
- Throw and azimuth volumes; `max_fault_throw` per horizon.
- `improve_depth_maps` onlap and fan fixes; channel and salt interaction.
- Clustered modes: `self_branching`, `stairs`, `relay_ramps`, `horst_graben`.
- Writing `fault_labels` on the `pipeline_overlap` path (its labels are
  already faulted). Forwarding `--faults` / `--shape` to CLI multi-worker,
  overlap and multiprocess children; the library APIs already support these.
- PyO3 exposure. The Python extension (`synthoseis_mdio`) is MDIO I/O only
  today and has no geology-parameter binding pattern yet.
- Faulting real geology once the toy three-layer horizons are replaced.

## How to run

```bash
cd rust
cargo test -p synthoseis-geo            # unit tests + parity vs committed Python fixture
cargo test -p synthoseis-core --test faults_pipeline
cargo run -p synthoseis -- run --e2e --chunked --faults 4 --seed 4 --shape 48,48,64 --store /tmp/faults.mdio

# Regenerate / extend the Python fixture (needs numpy, scipy, tqdm, ... in a venv)
python tests/fixtures/generate_fault_cubes.py --out tests/fixtures/fault_cubes.json
python tests/fixtures/generate_fault_cubes.py --sweep 60 --out /tmp/sweep.json
SYNTHOSEIS_FAULT_FIXTURE=/tmp/sweep.json cargo test -p synthoseis-geo --test faults_parity -- --nocapture

# Figures
cargo run --release -p synthoseis-core --example faults_demo -- /tmp/fd 4 4 96 96 128
python synthoseis-core/examples/plot_faults_demo.py /tmp/fd /tmp/faults_seed4
```
