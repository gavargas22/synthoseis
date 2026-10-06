# Rock physics port

Status (PR #28): the **corrected legacy depth model is the default**, and
`--legacy-toy-depth` (`RockPhysicsConfig::legacy_toy()`) reproduces master
10f4dcd bit for bit. Sections 1–2 give the root cause and the legacy survey
that led to the port. Sections 3–7 describe what was built and how it was
checked.

## 1. Root cause: DC offset and reflectivity down to −16.1 (master)

On the 64×64×128, seed 7, 4-fault demo cube at 15°, master's raw
reflectivity ranged from −16.13 to 0.119 (mean −0.049; 0.40 % of samples
with |r| > 1; ~100 % non-zero). Real PP coefficients lie in [−1, 1] and are
zero inside homogeneous layers.

`synthoseis-seismic::zoeppritz_pp` matches `datagenerator/zoeppritz_kernel.py`
term for term. `tests/fixtures/rpm_reflectivity_rootcause.py` rebuilds
master's reflectivity bit for bit from the labels, the `rpm_example` trends
and that formula. The bug was in the elastic *inputs*:

1. **The depth scale was 25× legacy.** Master used 100 m per sample; legacy
   uses `digi` (4 m). A 128-sample cube reached 12.7 km. The `rpm_example`
   quadratics turn over at about 4–10 km (shale Vp is 0 at 9.92 km). The
   −16.1 sample was a post-critical shale (Vp 447) over brine sand
   (Vp 5809) interface at 9.6 km.
2. **Depth was per sample from the cube top.** Legacy uses one depth per
   layer: the layer-base TVD below the mudline. With per-sample depth,
   every sample inside a layer is an impedance step, which gives +0.01 to
   +0.03 at every sample (the DC offset and the red spectrum).
3. **The water column used oil sand.** Label 255 above the seabed fell
   through to the oil-sand trend. Legacy `water_properties` uses
   1.028 g/cc, 1500 m/s, 1000 m/s.

## 2. Legacy survey

| Legacy | What it does | Rust (this PR) |
|---|---|---|
| `Faults.build_faulted_property_geomodels` | re-picks the horizons from the *faulted* age cube; depth = (layer base − seabed) × `digi`, constant per layer and trace | per-column from the labels (§3.1); faulted approximation (§5) |
| `RPMExample` | shale / brine / oil / gas sand depth trends | `synthoseis_rpm::example_f32` (float32 numpy semantics) |
| `build_property_models_randomised_depth` | per layer above `first_random_lyr`: random layer shift ±`layershiftsamples` and per-property shifts ±`RPshiftsamples`; forward-fill zeros at the base | keyed shifts (§3.3); forward-fill ported |
| `calculate_sands` + `EndMemberMixing` | sand (brine/oil/gas trend by closure fluid) mixed with shale by net-to-gross: inverse velocity (default) or Backus moduli | `mix_f32`, both methods, bit-exact (§4) |
| `create_random_net_over_gross_map` | opensimplex fBm per sand layer, rescaled to U(0.45, 0.9) mean and U(0.01, 0.05) sd | keyed value-noise fBm with the same rescaling (§3.2) |
| closures → oil/gas masks | fluid per closure selects the trend set (no Gassmann in legacy) | 2-D closures per sand layer (§3.4) |
| `water_properties` | 1.028 / 1500 / 1000 | same |

## 3. Design

`RockPhysicsConfig` is an `E2eConfig` field. `elastic_model(cfg, labels, shape)`
returns an `ElasticModel`:

- **`LegacyToy`**: master's 1-D `[Vec<f64>; 9]` trends and the old fuse
  kernels, untouched. It is selected by `legacy_toy_depth: true`, i.e. CLI
  `--legacy-toy-depth`.
- **`Rpm`** (default): per-label layer models, per-sand-layer N/G maps and
  closures, and keyed shifts. The properties of a trace are computed per
  column, from that column's labels and the small per-(i, j) maps, and
  written into `(ti, tj, nk)` tile buffers. `synthoseis-gpu` then fuses them:
  `fuse_props_tile_cpu`, or WGSL "mode 1" under `--gpu`.

Everything a tile needs is a pure function of (config, labels, column). So
chunk shape, halos, strip workers, OS processes and geometry-once all give
identical bits. The only extra memory is O(ni·nj) maps per sand layer plus
one tile of properties (counted in the overlap peak-bytes accounting).

CLI flags:

| flag | effect |
|---|---|
| `--legacy-toy-depth` | master 10f4dcd model |
| `--mixing inverse-velocity\|backus` | N/G mixing method |
| `--net-to-gross X` | constant N/G instead of legacy maps |
| `--first-random-layer N` | random depth shifts for legacy layers > N (default 20, as in the example config) |
| `--no-fluids` | brine everywhere |

The custom flags are rejected (exit 2) together with `--legacy-toy-depth`.
Multi-process children receive the same flags.

### 3.1 Depth (4 m per sample, per-layer TVDML)

Label `n` spans horizon interval `[z_n, z_{n+1})` of the toy maps. Its depth
is legacy's `(f32(z_{n+1}) − f32(z_0)) × digi`, computed in f32 exactly like
the float32 legacy cubes. The water column (label 255 above the seabed) gets
depth 0 and the water properties. Unfilled samples below the deepest horizon
are forward-filled like legacy `fix_zero_values_at_base`.

On faulted cubes the run is shifted by the observed integer throw
`(run end − floor(z_{n+1})) − seabed shift` (§5).

Lithology comes from the legacy sand-fraction Markov chain on the layered
geometry; see [toy-lithology.md](toy-lithology.md). With the planar
geometry, `--legacy-toy-depth` or `--toy-lithology alternating`, even
intervals are shale and odd are sand. Sand is mixed with shale by N/G.

### 3.2 Net-to-gross maps

Per sand layer: keyed value-noise fBm (lacunarity 1.9, persistence 0.5,
9 octaves). It is rescaled like legacy to mean ~ U(0.45, 0.9) and
sd ~ U(0.01, 0.05), then clipped to [0, 1].

Legacy uses opensimplex, which has no stable cross-language stream. Parity
is therefore statistical, on the real `create_random_net_over_gross_map`
over 200 seeds:

| | Rust | legacy |
|---|---|---|
| mean | 0.678 | 0.687 |
| sd | 0.0274 | 0.0283 |
| lag-1 autocorrelation | 0.970 | 0.972 |
| range | [0.45, 0.9] | [0.45, 0.9] |

### 3.3 Random depth shifts

For each (seed, layer, property), a splitmix counter draw replaces
`np.random.uniform` / `triangular`:

- the default half ranges are `triangular(35, 75, 125)` (layer) and
  `triangular(5, 11, 20)` (property);
- shifts apply to legacy layer `L + 1 > first_random_layer`;
- the shifted index uses numpy's `clip(0, nk − 10)` with negative wrap.

The shift PMF and the half-range distributions match legacy draws; see
`random_parts_match_legacy_statistics`.

### 3.4 Fluids (closures → trend selection)

For each sand layer, closures are found on the post-fault top-of-layer
surface:

1. A priority-flood fill from the map boundary gives the spill depth.
2. Connected components give the closures.
3. The contact is `min(spill, crest + max_column_m / digi)`.
4. The fluid (brine/oil/gas, uniform) is keyed by seed, layer and closure
   rank.
5. Closures smaller than the closure minimum stay brine (see "Closure
   minimum" below).

### Closure minimum

Legacy keeps a closure only if it holds at least `min_closure_voxels_simple`
voxels: 500 in `config/example.json`, whose cube is 300 × 300 × 1250
(`Closures.py` `remove_small_objects` on the output grid). Legacy's own
16 × 16 e2e fixture lowers it to 20 (`docs/angle-stack-e2e-parity.md`).
The Rust default scales it with the map area (`ClosureMinimum::Scaled`):

    minimum = clamp(round(ni · nj / 180), 20, 500)   (integer: (ni·nj + 90) / 180)

- **Map area.** Trap width scales with the cube (dome radius 0.22–0.35 ×
  min(ni, nj)); trap height is capped by the 37.5-cell maximum column and the
  sand unit, so nk does not enter. 180 cells² per voxel gives exactly 500 at
  300 × 300.
- **Floor 20.** Compartment sizes are bimodal: specks (1–3 column pits and
  fault slivers) are mostly 1–10 voxels, real traps on small cubes 20–500.
  Floors 10–50 give the same kept counts within ±0.2 per seed.
- **Cap 500.** Never stricter than legacy: every closure kept at 500 stays
  kept with the same contact and fluid (the fluid draw is keyed by rank over
  every closed region, kept or not). Cubes with ni·nj ≥ 89,910 are
  unchanged. (`ClosureMinimum::SCALED_CAP`; `None` would let the rule grow
  past 500.)
- **Values.** 8 × 8 to 60 × 60: 20; 64 × 64: 23; 96 × 96: 51; 128 × 128:
  91; 192 × 192: 205; 300 × 300 and larger: 500.
- **Counting.** Whole cells (`Σ (k1 − k0)` per compartment), also under
  partial voxels: no circular dependency on the contact, the same traps in
  both voxel modes, and consistent with whole-cell detection.
- **Scope.** All closure modes: 3D-segmented per sand unit (default),
  `--closures-unsegmented`, `--closures-per-layer` and the planar geometry.
  Salt walls only shape the fill; the minimum applies the same way.
- **Measured** (default configuration, 12 seeds per cube): the kept share of
  the closure volume rises from 59–99.8 % to ≥ 98.8 % on 22 cube/fault
  combinations; on 32 × 32 × 128 without faults the seeds without any kept
  trap drop from 9 to 4 of 12.
- **Not ported.** Legacy also has `min_closure_voxels_faulted` (2,500, 5×
  the simple minimum) and `_onlap` (500) for its closure types. Rust has no
  closure types, so one minimum applies to every compartment.
- **Switches.** `--legacy-closure-minimum` / `ClosureMinimum::LEGACY`
  (`Fixed(500)`, master bad1daa8), `--min-closure-voxels N` /
  `Fixed(N)`. Scaled stores carry `closure_min_voxels` and
  `closure_minimum = "scaled-area"`.

Voxels above the contact use the oil or gas sand trend. There is no
Gassmann step, as in legacy. `dome_closure_selects_fluid_above_contact`
checks crest, spill point, contact, fluid and the properties above and
below the contact.

## 4. Parity with legacy

Fixture: `tests/fixtures/generate_rock_physics.py` calls the **real** legacy
`Faults.build_faulted_property_geomodels` and
`SeismicVolume.build_property_models_randomised_depth` (recording the
shifts) on an 8×6×48 cube with 3 layers, oil and gas masks, N/G maps and
pure-shale columns. It writes `tests/fixtures/rock_physics.json`
(run-length encoded).

| check | result |
|---|---|
| depth model vs legacy `faulted_depth` | **1903 / 1903 voxels bit-identical** where both assign the same layer. Layer agreement is 89.8 % of legacy sediment voxels: Rust labels `[ceil z_n, floor z_{n+1})` with a one-sample 255 gap at non-integer horizons, while legacy rounds. The labels are master's and unchanged. |
| properties, inverse velocity | **2304 voxels × (ρ, Vp, Vs) bit-identical** (trends, shifts, water, brine/oil/gas sand mixing, base forward-fill) |
| properties, Backus | **2304 voxels × 3 bit-identical** |
| N/G maps, shifts, fluid choice | statistical (§3.2, §3.3; fluid frequencies uniform over 3) |
| `--legacy-toy-depth` vs master 10f4dcd | bit-identical: 5 library hashes (labels, angle stacks 15° / 30°, raw reflectivity, filters + noise, `tiny(42)`) and 3 CLI store hashes (plain, faults + bandpass + noise, multi-process) |

**Float32 and AVX-512.** Legacy builds float32 cubes with numpy. The Rust
kernels reproduce numpy's float32 semantics:

- coefficients are rounded to f32;
- `z**2` is `z*z`;
- `z**3` is `powf`;
- operations follow the same order.

The golden is generated with numpy's portable dispatch
(`NPY_DISABLE_CPU_FEATURES` = AVX-512 groups). A probe with AVX-512 SVML
enabled shows the raw `z**3` differs by up to 1 ULP in about 21 % of values,
but after ρ rounding there are **0 mismatches** on ρ, Vp and Vs.

## 5. Faulted cubes: the approximation

Legacy faults the age cube, **re-picks the horizon depths from the faulted
age column**, and then computes per-layer TVDML from those faulted horizons,
which have sub-sample throws.

The Rust default avoids a second global pass. It takes the unfaulted
interval depth from the maps and adds the integer throw observed in the
faulted label run of that column.

The reference re-implements legacy exactly: `horizon_depth_from_age` on the
Rust-faulted age cube. It matches the legacy fault fixture's re-picked
horizons to 5.1e-5 samples. On the 4 faulted cases of
`tests/fixtures/fault_cubes.json`, over 85,143 voxels where both assign the
same layer (73.7 % of legacy sediment):

| metric | label-run approximation | naive "fault the depth cube" |
|---|---|---|
| mean \|Δz\| | **0.90 m** | 9.72 m |
| p50 / p95 / p99 | 0.07 / 4.80 / 7.61 m | — |
| max | 15.95 m | 101.05 m |
| exact | 36.4 % | — |
| within one sample (4 m) | 93.5 % | — |

The residual comes from rounding the throw to whole samples, and from runs
truncated at a fault, which reach up to ~4 samples. Replicating legacy
exactly would need the faulted age cube and a horizon re-pick per column.
That is possible per column but doubles the fault work. Deferred.

## 6. Invariance, GPU and features on top

- `default_model_invariant_to_tiling_workers_and_paths` compares exact bits
  for 3 configurations:
  - "rich": Backus, random shifts on every layer, closures enabled,
    bandpass + lateral filter + noise, 3 faults;
  - plain default with 2 faults;
  - the legacy switch.

  Each is run across chunks 1×1, 5×7, full and 3×20×16; the classic path;
  streaming and overlapped streaming at 2 chunk shapes; strip-stitch 2, 3
  and 4; multi-process 1, 2 and 3; and geometry-once 0/15/30 against the
  per-angle runs.
- The CLI tests check that multi-process with model flags equals single
  process, and that `--legacy-toy-depth` equals the master stores.
- Filters (bandpass, lateral, `--keep-ricker`) and noise run on top of
  both models.
- **GPU.** `--gpu` fuses the default model's property tiles in WGSL (mode
  1). The mode-1 output is bit-identical to mode 0 (trends) for the same
  properties. Two WGSL fixes came with it:
  1. **The WGSL Zoeppritz now mirrors the CPU/legacy kernel.** Legacy
     `zoeppritz_kernel.py` / `tests/_zoeppritz_reference.py` use
     `aa + det * ct / vp1 * cp2 / vs2` where the textbook (bruges) has `d`.
     The CPU port kept legacy; the WGSL used `d`. At 30° over a water /
     sediment contrast the backends differed by 2.9e-2 (bruges agrees with
     the old WGSL).
  2. **Exact pre-critical sin/cos(asin x) identities and a host-side angle
     sine/cosine.** WGSL only bounds `sin`, `cos`, `log` and `atan2` to
     ~2^-11.

  GPU vs CPU max |Δ| on the demo cube (llvmpipe) is now **1.9e-7**
  (15° reflectivity and stack), and at most 4.9e-7 in the new
  `parity_props_tile` test. It was 6.6e-3 to 2.9e-2 before.

## 7. Reflectivity before / after

64×64×128, seed 7, 4 faults, raw 15° reflectivity:

| model | min | max | mean | \|r\| > 1 | non-zero |
|---|---|---|---|---|---|
| before: master toy (`--legacy-toy-depth`) | −16.13 | 0.119 | −0.0485 | 0.40 % | 99.2 % |
| after: default (inverse velocity) | 0 | 0.466 | 0.00427 | 0 | 1.56 % |
| after: Backus | 0 | 0.466 | 0.00417 | 0 | 1.56 % |

The planar toy geometry (the default when these numbers were taken, now
`--toy-geometry planar`) has 3 labels, so there are 2 interfaces per trace:
the seabed and one shale/sand boundary. Both are positive. The layered
default has ~45 layers on 256 samples (31 % non-zero reflectivity at 15°);
see [layered-toy-geometry.md](layered-toy-geometry.md).

Figures come from `examples/rock_physics_demo.rs` and
`examples/plot_rock_physics_demo.py`:

- `rock_physics_reflectivity.png`
- `rock_physics_angle_stacks.png`
- `rock_physics_trends.png`
- `reflectivity_rootcause_before_after.png`

## 8. Deferred / not replicated

- ~~Salt.~~ Done: salt bodies with legacy properties, on by default in
  the layered geometry (`--no-salt` = master b4f4259); see
  [salt-bodies.md](salt-bodies.md).
- The `rpm_scaling_factors` multipliers, `partial_voxels`
  (fraction-weighted depth), `variable_shale_ng`, and the Poisson-ratio
  clips.
- 3-D closures with fault seals. Closures here are 2-D spill analysis on the
  post-fault surface per sand layer. The planar toy geometry forms no
  closures; the layered default (a dome) does, see
  [layered-toy-geometry.md](layered-toy-geometry.md).
- Legacy's deepest-layer skip and its below-base water fill. Rust
  forward-fills instead.
- The exact faulted re-pick (§5).
- ~~The textbook Zoeppritz term.~~ Done: the textbook form is now the
  default on CPU and GPU, `--legacy-zoeppritz` keeps the legacy `det` form;
  see [zoeppritz-fix.md](zoeppritz-fix.md).
- The noise statistics fixture (`noise_pipeline.rs`) was generated on
  master reflectivity, so it stays pinned to `--legacy-toy-depth`.
- Python bindings (`synthoseis-py`) do not expose the rock-physics options
  yet. They use the default model.

## 9. Decisions for Guillermo

1. ~~**Zoeppritz `det` vs `d`.**~~ Decided (2026-09-28): fixed, default
   on, `--legacy-zoeppritz` restores the typo; `--legacy-toy-depth`
   implies it. See [zoeppritz-fix.md](zoeppritz-fix.md).
2. ~~**Richer toy geometry.**~~ Done: the default toy geometry is now a
   domed, many-layer stack (`--toy-geometry layered`) where the default
   shifts and closure fluids trigger; `--toy-geometry planar` keeps the
   3-label master geometry. See [layered-toy-geometry.md](layered-toy-geometry.md).
3. **Exact faulted depth (§5).** Is a mean error of 0.9 m acceptable, or
   should the per-column re-pick be implemented?
