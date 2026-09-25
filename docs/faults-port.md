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
| `xyz_dis` vertical gaussian, seabed taper loop | `model.rs`: `taper` / `vertical_profile` (legacy, exact), `vertical_profile_mode` + `ReachMode` (see [Vertical reach](#vertical-reach-reachmode)), `ResolvedFault::{seabed_roll, sigma_used, seabed_ok, reach_rescued}` |
| `xyz_dis` lateral gaussian + `ndimage.rotate` + re-roll | `model.rs`: `throw_variance`, analytic `LateralGaussian` (closed form, no rotated raster) |
| `apply_xyz_displacement` (`np.interp`) | `apply.rs`: `interp_uniform`, `interp_trace_into` (numpy semantics, f32 rounding) |
| label-level update and `> 0.05` threshold | `apply.rs`: `FaultModel::compute_tile` → `FaultTile { lookup, mask, segment_id }` |
| faulted age / facies (`interp(F, age)`) | `FaultModel::apply_to_volume_f32`; categorical labels use `apply_to_labels` / `remap_nearest` (nearest sample) |
| `faulted_depth_maps` | `horizon_depth_from_age` (inverse of the faulted age per trace; used by the parity test) |
| `fault_planes` → `fault_segments` output | MDIO variable `data/fault_labels` (u8 0/1), `synthoseis-io` `write_fault_labels_chunk` / `read_fault_labels_u8` |

Pipeline wiring (`rust/synthoseis-core`):

- `E2eConfig.faults: FaultConfig { count, throw_min, throw_max, legacy_reach }`
  (`legacy_reach: false` by default = `ReachMode::FitColumn`).
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
| 60-seed sweep (48×48×64, not committed), replay mode | 30 with faults, 118,116 fault voxels | **1.0 / 1.0 in every case** | worst 5.2e-5 samples, 100 % within 0.01 |
| committed tall fixture `fault_cubes_tall.json` (16×16×640, 2 cases, 3 active faults), **default `FitColumn` mode**, replay | 2 | **1.0 / 1.0** (7213, 1466 voxels) | 4.9e-5 samples |
| 40-seed tall sweep (16×16×640, not committed), default mode, replay | 31 with faults, 88,522 fault voxels | **1.0 / 1.0 in every case** | 100 % within 0.01 in 30/31; `tall_8` 0.50 (numpy SIMD `exp`, below) |
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

**Vertical-profile argmax (fixed in the reach PR).** Python rolls the
vertical gaussian by `centre + g.argmax() + roll_int`. For large σ and p the
gaussian has a plateau of samples that are exactly `throw` in f64, and
`argmax` returns the *first* one. The port used the analytic centre, which
put the profile peak a few samples too deep on tall cubes (masks off by
~0.4 %, horizons by up to 0.9 samples on `tall_5`). `taper` now walks left
across the plateau like `argmax`. On short cubes the profile is flat across
the column, so the committed fixture and sweep were unaffected.

**Platform note: numpy's SIMD `exp`.** Where the plateau ends depends on
whether `exp(-y)` rounds to exactly 1.0 for tiny `y`. numpy 2.x on AVX-512
returns `0.9999999999999999` for `y ≥ 4.51e-17` (0.8125·2⁻⁵⁴), while libm (and
Rust) round correctly to 1.0 up to 2⁻⁵⁴. So on rare draws the legacy output
itself depends on the CPU. In the 40-seed tall sweep one case (`tall_8`, σ=280,
p=4.7) has its plateau one sample longer in Rust. Its masks are still
identical, and its horizons differ by ≤ 0.5 samples. The port keeps the
correctly rounded result.

## Vertical reach (`ReachMode`)

σ for the vertical profile is drawn from `U(10·throw − 50, 300)` samples. The
reach of the profile, meaning the distance from the peak at which the throw
falls to 1 sample, is

```text
R(σ) = σ · (2 ln throw)^(1/(2p))          worst legacy draw: σ=300, p=1.5, throw→35  ⇒  R ≈ 577
```

The legacy taper loop pushes the peak down in steps of 5 until the throw at
the seabed is ≤ 1. It gives up after `nz − wb` samples, so it can only
succeed when the column below the seabed can hold the reach. On cubes with
less than ~582 samples below the seabed it often gives up. The hanging wall
then displaces the whole column, including the seabed and the water, and
`interp(clip(k−d))` smears the top sample down. The result is a slab of
fault-mask voxels in and just below the water column (see the figures).

`ReachMode` (in `FaultModel::resolve_with_mode`, or
`FaultConfig::legacy_reach`):

- **`Legacy`**: exact legacy behaviour. The parity tests use it on the short
  committed fixture, where Python's own taper gives up.
- **`FitColumn`** (default): per fault, the smallest possible intervention.
  1. Run the legacy taper unchanged. If the throw at the seabed ends ≤ 1,
     keep the result as is. **The fault is then bit-identical to `Legacy`.**
  2. Only if the legacy taper gives up, scale σ so that the reach fits the
     column the loop can use:
     ```text
     d_max = (centre_k − 1.5 − wb_max) + 5·floor((nz − wb_max)/5)   # furthest the loop can put the peak below the seabed probe
     σ'    = min(σ, 0.98 · d_max / (2 ln throw)^(1/(2p)))          # R(σ') = 0.98·d_max
     ```
     This is the same as scaling the σ draw by `0.98·d_max / R(σ)`. The
     unchanged legacy loop then runs again with σ'. If it still gives up, σ'
     shrinks by 10 % per try, down to 0.5. If nothing succeeds (no such case
     has been seen), the legacy profile is kept.
  3. Fault mask and segment ids are zeroed where `k < seabed(i, j)`.

  The random draws, geometry, centres and lateral profiles do not change;
  only the vertical profiles of rescued faults change.

**Why the default is legacy-exact wherever legacy works.** Step 1 is the
legacy code path. The clamp in step 3 removes nothing when every taper
succeeds against a flat seabed: once the loop succeeds, the peak lies below
the seabed probe, and above it the profile is monotone and ≤ 1, so no mask
increment happens above the seabed. Later faults only move content down.
The following were measured, not only argued:

- On the committed tall fixture and a 40-seed tall sweep (16×16×640),
  `FitColumn` equals `Legacy` bit for bit (mask and faulted age) with 0 rescues.
  It therefore matches Python equally well (table above).
- On pipeline cubes 24×24×704 (seeds 4 and 10, a mapped seabed; this is a
  test) and 48×48×704 (seeds 1–10), there were 0 rescues, identical masks, and
  0 fault voxels above the seabed in `Legacy` itself.
- On the short fixture and the 60-seed 48×48×64 sweep, all 31 active faults
  are rescued. Every taper succeeds, no mask voxel sits above the seabed, and
  the results are tiling-invariant.

**Before and after** (`faults_demo`, 96×96×128, 4 faults, mapped seabed from
0.5 to about 14 samples):

| seed | legacy fault voxels (above seabed) | `FitColumn` fault voxels (above seabed) | rescued faults | voxels removed by the clamp itself |
|---|---|---|---|---|
| 4 | 74,098 (3,879) | 64,985 (0) | 4/4 | 0 |
| 10 | 63,903 (2,708) | 61,248 (0) | 4/4 | 0 |

The σ fit alone removes the whole slab. The clamp is a safety net for mapped
seabeds, where a column with fault segments but no surface voxel can sit
deeper than `wb_max`.

## Legacy behaviours kept on purpose (worth a decision)

- **Random draws often miss or go flat.** Many random draws miss a small cube
  entirely and are skipped (`FaultSkip`). Some produce near-horizontal
  "faults" (detachments).
- **Rescued faults keep the legacy shift rule.** σ' is the *largest* σ that
  still lets the unchanged loop clear the seabed. The loop therefore still
  rolls the peak down, often by about 100 samples on a 128-sample cube, so the
  maximum throw can sit near or below the cube bottom. Fitting σ so that the
  peak stays at the chosen centre (zero shift) would be the alternative. It is
  not the default because it departs further from the legacy σ distribution.

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
cargo test -p synthoseis-geo            # unit tests + parity vs committed Python fixtures (short + tall)
cargo test -p synthoseis-core --test faults_pipeline
cargo run -p synthoseis -- run --e2e --chunked --faults 4 --seed 4 --shape 48,48,64 --store /tmp/faults.mdio

# Regenerate / extend the Python fixture (needs numpy, scipy, tqdm, ... in a venv)
python tests/fixtures/generate_fault_cubes.py --out tests/fixtures/fault_cubes.json
python tests/fixtures/generate_fault_cubes.py --sweep 60 --out /tmp/sweep.json
SYNTHOSEIS_FAULT_FIXTURE=/tmp/sweep.json cargo test -p synthoseis-geo --test faults_parity -- --nocapture
# Tall cubes (default FitColumn must equal legacy): committed fixture + optional sweep
python tests/fixtures/generate_fault_cubes.py --tall            # -> tests/fixtures/fault_cubes_tall.json
python tests/fixtures/generate_fault_cubes.py --tall --tall-seeds $(seq -s, 1 40) --out /tmp/tall40.json
SYNTHOSEIS_FAULT_TALL_FIXTURE=/tmp/tall40.json cargo test -p synthoseis-geo --test faults_parity tall -- --nocapture

# Figures
cargo run --release -p synthoseis-core --example faults_demo -- /tmp/fd 4 4 96 96 128
python synthoseis-core/examples/plot_faults_demo.py /tmp/fd /tmp/faults_seed4
# Before/after vertical reach (legacy vs default FitColumn)
cargo run --release -p synthoseis-core --example faults_demo -- /tmp/r4l 4 4 96 96 128 legacy
cargo run --release -p synthoseis-core --example faults_demo -- /tmp/r4f 4 4 96 96 128 fit
python synthoseis-core/examples/plot_faults_reach.py /tmp/r4l /tmp/r4f /tmp/faults_reach_seed4.png
```
