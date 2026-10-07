# Salt bodies

The Rust core now inserts a salt body into every layered model, as legacy
`datagenerator/Salt.py` does. Legacy's shipped `config/example.json` has
`include_salt: true`, so salt is **on by default** in the layered geometry.
The port covers:

- the salt shape: a convex hull around a cloud of 218 jittered points,
- the horizon drag against the salt flanks,
- the salt rock properties (rho 2.17 g/cc, Vp 4500 m/s, **Vs 2600 m/s** default; 2250 with `--salt-legacy-vs`),
- the closure walls that seal traps against salt,
- a `salt_labels` volume (0/1) in the MDIO store.

Switches:

- `--no-salt` (`RockPhysicsConfig::salt = false`) reproduces master
  b4f4259 bit for bit.
- `--salt-legacy-top-offset` (`RockPhysicsConfig::salt_legacy_top_offset`)
  uses legacy's absolute top offset (see Deviations).
- `--salt-legacy-vs` (`RockPhysicsConfig::salt_legacy_vs`) restores salt
  Vs = 2250 m/s (master f15b87ac); default is 2600. Density and Vp stay
  2.17 / 4500. See [`synthoseis_rpm::salt_elastic`].
- `--fault-labels-through-salt` keeps fault labels inside the salt (master
  2b3850ba); by default they are masked (see Fault labels and salt).
- The planar geometry and `--legacy-toy-depth` never have salt. Passing
  either switch with them exits with code 2.

Code: `synthoseis-core/src/salt.rs`. Tests: `synthoseis-core/tests/salt.rs`
and `synthoseis/tests/rock_physics_cli.rs`.

## What legacy does

After faulting, `Faults.py` calls, when `include_salt` is set:

1. **`SaltModel.compute_salt_body_segmentation`.** This builds the shape
   on the padded grid `(ni, nj, nk + pad_samples)`:
   - The radius is `R = triangular(ni/6, ni/5, ni/4)` columns.
   - The top is the 99th percentile of horizon 1 plus `uniform(150, 300)`
     samples.
   - **`insertSalt3D`** builds a "bag of points":
     - a wide **cap**: three circles of 36 jittered points (radii about
       R, 0.4 R and 0.2 R), stacked just above depth `1.1 (top + R)`,
       plus a tip point;
     - a narrow **stem**: three smaller circles plus a tip near the base
       of the padded cube, centred at a random lateral offset from the cap.
   - Salt is every voxel inside the convex hull of those points
     (`util.is_it_in_hull`, i.e. scipy `Delaunay.find_simplex >= 0`).
2. **`update_depth_maps_with_salt_segments_drag`**, horizon by horizon:
   - If the horizon lies inside salt in some column, a counter `r` goes up
     by one. `r` is cumulative over horizons.
   - The horizon is lifted by `2 r` samples in its salt columns
     (`L = −2r · salt_mask`).
   - Default (lift-only): `m_out = m + gaussian_filter(L, sigma = 3)`.
     Opt-out `--salt-smooth-all-horizons` restores legacy
     `m_out = gaussian_filter(m + L, sigma = 3)`, which also flattens the
     undragged map far from salt.
   - Then `push_down_remove_negative_thickness` runs. Despite the name,
     this is **geometry only**: it moves horizon depths down so that no
     layer has negative thickness after the drag. It is not the seismic
     velocity "push-down" or "pull-up" under salt.
   - Near salt the result is horizons that bend up against the flanks
     (drag).
   - The horizon gaps are the columns where a dragged horizon lies inside
     salt.
3. **Lithology and properties.**
   - Lithology is set to 2 inside salt. The age volume is rebuilt from the
     dragged maps.
   - In `Seismic.py`, `lith == 2` voxels get `rho = 2.17`, `vp = 4500`,
     `vs = 2250` after the layer loop. The Rust default raises Vs to
     **2600** (Falcon-Suarez et al. 2024 grain ≈ 2.6 km/s; Vp/Vs ≈ 1.73,
     Poisson ≈ 0.25); `--salt-legacy-vs` keeps 2250. Vp 4500 and ρ 2.17
     match Jones & Davison (2014) / Yan et al. (2016). The base
     forward-fill runs after that, and the final scaling factors skip salt.
4. **Closures.**
   - The salt gaps become `-1` in the closure maps (NaN → 0 → `/digi`).
   - `Closures._flood_fill` walls them off. Gap cells get `-1 + max_column`,
     their 8-neighbours get `depth + max_column`, and none of these cells
     can be closed.
   - Spill paths cannot cross the salt, so traps seal against the salt
     flank.

## What the port does

| step | Rust | vs legacy |
|---|---|---|
| draws | one keyed unit draw per legacy draw (`salt::keyed_draws(seed)`), in legacy order. numpy `uniform` and `triangular` are replayed with numpy's exact operation order | bit-exact given the same unit draws. The distribution is validated by KS |
| top | numpy linear `percentile(horizon 1, 99)` + offset | bit-exact |
| shape | incremental 3D convex hull of the 218 points. Per column, the salt is the sample run `[k0, k1)` inside every hull facet | bit-exact mask (a hull meets each vertical line in one segment) |
| drag | lift `L`, scipy `gaussian_filter(sigma=3)` on `L` (default) or on `m+L` (`--salt-smooth-all-horizons`); kernel weights as IEEE bits, `reflect`, axis 0 then 1; push-down | bit-exact vs scipy either way |
| geometry | the dragged maps are rounded to whole samples, as the layered geometry already does, and fed to label fill, faults and the depth model | Rust has whole-sample horizons |
| properties | `VoxelKind::Salt` → `salt_elastic(legacy_vs)` (2.17, 4500, **2600** as f32; 2250 with `--salt-legacy-vs`) before the forward-fill. It overrides water, layers and closures | default Vs differs from legacy 2250; opt-out restores it |
| closures | salt-aware fills in all three closure modes (segmented default, `--closures-unsegmented`, `--closures-per-layer`). A column is a gap when the unit's top sample is salt. Gap cells get `-1 + max_column`, ring cells `top + max_column`, and both stay open | bit-exact vs `_flood_fill` |
| MDIO | `data/salt_labels` (uint8 0/1, fill 0), written chunk by chunk on every path (classic, chunked, streaming, overlap, strip-stitch, multi-process). Each path checks it against the salt body on read-back | new output |

Memory: the salt body is one `(k0, k1)` pair per column (8 bytes per
column) plus 218 points. The drag works on the horizon maps, which are
already `O(ni * nj * nh)`. The salt labels are produced per chunk, so
nothing depends on the chunk shape. The body is a pure function of the
config, so every worker and process rebuilds the same salt.

The CLI prints e.g. `salt: top 65.9 samples, radius 12.2 columns, 18535
voxels in 597 columns`. With `--no-salt` it prints
`salt: off (--no-salt, master b4f4259)`.

**Time output and velocity pull-up.** When salt bodies were first added,
the stacks were written on the depth-sample axis, so there was no velocity
pull-up. Since the depth-to-time work ([depth-to-time.md](depth-to-time.md))
the default output is two-way time, built per column from the voxel Vp. The
fast salt (4500 m/s) now pulls up the reflections beneath it. The
"push-down" in the drag step above is different: it only adjusts horizon
depths.

The salt body is built in depth, and the depth body is what the drag mode
and the salt switches leave alone. `data/salt_labels` in a time-mode store
(the default) is that depth mask point-sampled through each column's
traveltime (depth-to-time §3.4). So:

- **In time, the salt is thinner than in depth**, because it is fast. On
  the seed-7 32×32×128 cube (3 faults), 2 336 depth salt voxels become
  1 060 time samples.
- **Time-domain `salt_labels` can move with two-way time while the depth
  salt body does not.** Anything that changes the overburden above the salt
  changes each column's traveltime to the salt, and so which time samples
  the salt fills. The drag mode is one example. Comparing lift-only with
  `--salt-smooth-all-horizons` on the same seed-7 cube: the depth
  `salt_labels` are identical (0 of 2 336 voxels differ), but 63 time
  samples in 53 columns move (1 060 vs 1 055 salt samples). In the same
  way, when lift-only became the default, the time-domain `masked_in_salt`
  count of the seed-11 fault-mask smoke moved from 63 to 72 (whole voxels)
  and from 68 to 70 (partial voxels), while the depth count stayed at 155.
  So "the salt body is unchanged" is a statement about depth. To check it,
  compare depth cubes (`--legacy-depth-as-time`).

## Deviations (documented, not bugs)

- **Top offset scaled by `min(nk / 1250, 1)`.** Legacy adds 150–300
  samples below horizon 1, sized for its 1250-sample example cube. In a
  256-sample toy cube that would put the salt entirely below the data. The
  default scales the offset by `min(samples / 1250, 1)`. That is exactly
  legacy at 1250 samples or more, so the salt is never deeper than legacy on
  large cubes (`salt_top_offset_scale_capped_at_one`, on a 1600-sample
  cube). `--salt-legacy-top-offset` keeps the absolute legacy offset. The radius
  already scales with `ni`, as in legacy.
- **The drag acts on the unfaulted toy horizons, before faulting.** In Rust
  the faults displace labels, and there are no faulted depth maps. So:
  - the p99 top uses the unfaulted horizon 1;
  - the smoothing does not blur fault offsets. Legacy smooths the faulted
    maps, so its horizons lose their sharp fault steps (not ported);
  - the salt body itself is not faulted, as in legacy.
- **Gaps come from the labels.** A column is a gap when the closure unit's
  top sample lies in salt. Legacy's `faulted_depth_maps_gaps` test on the
  dragged horizon depth is equivalent for whole-sample horizons. Legacy's
  gap maps are not output.
- **Walls at the array border** are skipped. Legacy zeroes a 3-cell border
  first. Cells with an absent unit (NaN) stay outlets, unchanged from
  #32/#33, where legacy would wall pinch-outs too.
- **No water facies at the base.** Legacy resets lith −1 → 0 in the last 50
  samples, because the drag can lift the base horizon into the cube. The
  Rust base horizon starts below the cube, and unfilled samples are
  forward-filled, so this is not needed.

## Legacy bugs, quirks and dead code found (reported, not fixed)

None was fixed, so there is no bug-fix legacy flag. The port reproduces
each one below or leaves it out as noted.

1. **`center_y` range uses `cube_shape[0]`.** `insertSalt3D` draws
   `center_y = cube_shape[1]/2 + uniform(-0.4 cube_shape[0], 0.4
   cube_shape[0])`. On non-square cubes the salt can land off-centre or
   outside in `j`. Kept; the geometry fixture has non-square cubes. The
   default toy cube is square.
2. **Every horizon is smoothed**, including horizons that never touch salt
   (legacy / `--salt-smooth-all-horizons`). **Default is now lift-only**
   (`m + G(L)`): far-field maps stay bit-identical to the undragged maps
   before push-down (never-touch horizons and cells with Chebyshev distance
   ≥ 13 from any lifted cell). On the demo cube under smooth-all, about
   2.5 % of labels far from salt still changed. Under lift-only that drops
   to 0 (gate ≤ 0.1 %; see "Effect on the demo cube"). Near-flank uplift
   keeps the same shape (RMS |legacy − fix| within 4 columns ≈ 0.35–0.52
   samples on 64²). The depth salt body is the same in both modes. The
   time-domain `salt_labels` can still differ by a sample (see "Time output
   and velocity pull-up").
3. **`push_down_remove_negative_thickness` never fixes horizons 0/1.** The
   loop stops at `i = 2`. Kept (the Rust `enforce_nonnegative_thicknesses`
   is the same loop).
4. **`SaltModel.update_depth_maps_with_salt_segments`** (the no-drag
   variant) is **dead code**: nothing calls it.
5. **The `facies_label.npy` QC dump** in the drag function saves the last
   horizon's salt hit map, not a facies volume. QC only; not ported.
6. The comment "divide by 2 since lith=2" in `Seismic.py` is misleading
   but harmless: salt properties are set directly.
7. **Closure voxels inside salt.** Legacy closure volumes restricted to
   `lith > 0` include salt voxels (lith 2). Their fluid is overwritten by
   the salt properties, so only legacy's closure *outputs* count them.
   Rust properties are the same.
8. Found while studying the chain, unrelated to salt:
   - `Parameters.noise_stretch_factor` is drawn and logged but never used
     (dead);
   - `base_structure_depth_map[isnan(top)] = 0` in
     `create_closure_labels_from_depth_maps` is a no-op, because the top
     NaNs were already replaced.

## Validation

Fixture `tests/fixtures/salt_reference.json` comes from
`tests/fixtures/generate_salt_reference.py`, which runs the real legacy
code (numpy 2.5.3, scipy 1.18.1). The generator uses a stub `cfg`, whose
`horizon_ss.spawn` returns a fixed `SeedSequence` child, so the model's own
unit draws can be replayed.

### Deterministic: bit-exact

| check | legacy | inputs | result |
|---|---|---|---|
| point cloud | `SaltModel.compute_salt_body_segmentation` (radius, p99 top, `insertSalt3D`) | 8 cubes, 28–48 × 30–48 × 360–440 (+10 pad), 4 non-square (the `center_y` bug), random dome-shaped horizon 1 | 8 × 218 points identical to the last bit |
| salt mask | `util.is_it_in_hull` (scipy `Delaunay`) | same | 59 111 salt voxels in 1 459 columns, identical on every column (1 per column run, asserted contiguous by the generator) |
| horizon drag | `update_depth_maps_with_salt_segments_drag` + push-down | the 4 first salt masks with 10–15 horizons spanning the cube (float and whole-sample maps), plus 4 maps smaller than the kernel radius (5×7, 1×9, 13×3, 2×2) | 76 660 map values bit-exact (72 875 moved by more than a sample) |
| closure depth with salt gaps | `Closures._flood_fill` + `min(max(cd, top), base)` with gap cells `-1` | 24 domes with a salt gap disc, the flanks dragged up, 0–2 extra domes, 0–1 fault cliff, max column 12 / 20 / 37.5 | 12 361 closed columns with bit-exact contacts. All 2 204 gap columns stay open. The walls change the closure on 10 789 columns versus an unwalled fill |

As in #32/#33, the fill fixture's generator asserts the conditions under
which the Rust fill and legacy agree by construction: ≥ 5-cell plateau
margins, closed regions ≥ 50 cells, the same 4- and 8-connectivity, and gap
rings away from the border.

### Random: at the 5 % level

4 000 legacy bodies (numpy `default_rng` on fixed `SeedSequence`s; the hull
test is stubbed out, so only the draws are compared) vs 16 000 Rust bodies
(keyed draws), on the legacy example grid (64 × 64 × 1250 + pad 10, flat
horizon 1 at 20 samples). The two-sample KS 5 % critical is
`1.358 · sqrt((n + m)/(n m)) = 0.0240`. p is from scipy.

| statistic | KS D | 5 % critical | p |
|---|---|---|---|
| radius R | 0.0163 | 0.0240 | 0.358 |
| top | 0.0160 | 0.0240 | 0.382 |
| crest tip depth | 0.0159 | 0.0240 | 0.387 |
| cap centre i | 0.0184 | 0.0240 | 0.224 |
| cap centre j | 0.0236 | 0.0240 | 0.056 |
| cap radius | 0.0091 | 0.0240 | 0.953 |
| stem base depth | 0.0202 | 0.0240 | 0.145 |
| stem centre i | 0.0138 | 0.0240 | 0.570 |
| stem centre j | 0.0148 | 0.0240 | 0.479 |
| stem radius | 0.0144 | 0.0240 | 0.518 |

None rejects at 5 %. Cap centre j is the closest (p = 0.056). A
one-sample KS against its theoretical distribution (uniform on
`[6.4, 57.6]`) gives legacy D = 0.0162 (p = 0.241) and Rust D = 0.0089
(p = 0.155): both samples fit the theory, and the gap between them is
sampling noise in the 4 000-body legacy sample.

The same population with 4 000 Rust bodies gives the largest D = 0.0302
against a critical value of 0.0304 (p = 0.051, cap centre i/j and stem
centre i). That is why the test uses 16 000 Rust bodies. The seed range
(`0x5A170000 + s`) is the same one used from the start; it was not
re-chosen.

The statistics come from
`cargo run --release -p synthoseis-core --example salt_demo -- stats OUT 16000`,
and `examples/plot_salt_demo.py` prints p.

## Effect on the demo cube (64×64×256, seed 7, 4 faults, defaults)

Regenerated on master `1c22b653` with the current defaults: two-way time
output, partial voxels, lift-only drag, salt Vs 2600 and the scaled closure
minimum. The earlier tables here were measured at #34, on the depth axis
with whole voxels, smooth-all drag and Vs 2250. They are replaced.

| | salt | closures with a fluid draw (brine / oil / gas) | HC voxels (oil / gas) | CLI store hash |
|---|---|---|---|---|
| `--no-salt` | — | 2 / 2 / 3 | 8 255 (5 161 / 3 094) | `0xe33574e175a89307` |
| salt (default) | top 65.9 samples, R 12.2 columns, 18 535 voxels (1.8 %) in 597 columns | 2 / 2 / 4 | 7 610 (4 974 / 2 636) | `0x2b632665f11b7a16` |

At these default flags, `--no-salt` is no longer the b4f4259 store, because
time output, partial voxels and other defaults have changed since then.
With the legacy switches the tests add, it still reproduces b4f4259 (see
Locks).

Angle-stack change, salt vs `--no-salt` (relative RMS of the difference,
relative to `--no-salt`):

| angle | rel RMS | changed samples |
|---|---|---|
| 0° | 0.616 | 999 094 of 1 048 576 (95.3 %) |
| 15° | 0.587 | 999 094 (95.3 %) |
| 30° | 0.494 | 999 248 (95.3 %) |

- **Labels.** Depth labels change in 2.87 % of the voxels. Changes outside
  the salt are 1.67 % of the cube; the salt itself is 1.77 %.
- **Far from salt.** More than 8 columns (Chebyshev) from any salt column
  (2 466 of 4 096 columns), **no label changes** (0.0000 %). At #34, under
  smooth-all, 2.5 % changed.
- **Far-field stacks still differ, but not because of the labels.** The 15°
  relative RMS there is 0.434 (0.429 at 0°, 0.442 at 30°). With salt on,
  the partial-voxel horizon maps are the drag of the *rounded* maps
  (`toy_horizon_maps_continuous`, partial-voxels spec §3.1). Far from salt,
  every interface therefore sits on a whole sample. `--no-salt` keeps the
  continuous sub-sample positions instead. The partial-voxel interface cells
  differ in every far column (about one cell per layer per column, 105 804
  of 631 296 far-field voxels). With `--legacy-whole-voxels`, the far-field
  properties and the 15° stack are bit-identical, salt vs `--no-salt`
  (0 of 631 296 samples differ). Near the salt (≤ 8 columns), the 15°
  relative RMS is 0.778.
- **HC voxels.** Far from the salt they are unchanged (127 → 127). Near the
  salt, the salt, the drag and the walls reorganise the closures
  (8 128 → 7 483).

### Salt Vs 2600 vs `--salt-legacy-vs` (2250)

Labels, `salt_labels` and `fault_labels` are identical in depth and in
time (`salt_vs_2600_label_identity`). Vs does not enter the traveltime.
The stacks change only at salt contacts, and **far angles change more**,
as the AVO predicts. Demo cube (same flags as above, `--angles 0,15,30`,
relative to `--salt-legacy-vs`):

| angle | changed samples | rel RMS | max abs Δ |
|---|---|---|---|
| 0° | 0 (bit-identical) | 0 | 0 |
| 15° | 25 782 (2.46 %) | 0.0204 | 0.034 |
| 30° | 26 120 (2.49 %) | 0.0792 | 0.318 |

- At 30°, the same few percent of cells change about 4× more than at 15°.
- On a smaller case, the 30° stack changes **7.72 % of the cells, rel RMS
  0.110** (15°: 7.61 %, 0.026). That case is seed 1,
  48×40×96, 0 faults, `--mixing backus --toy-lithology alternating`.
  These are the numbers measured in the #48 review, reproduced here
  exactly.
- The seed-7 32×32×128 3-fault cube changes 4.22 % of the cells at 15°,
  rel RMS 0.0229 (`salt_vs_2600_stack_change`).

Top-salt reflection, shale (ρ 2.10, Vp 2500, Vs 1000) over salt, textbook
Zoeppritz (`ZoeppritzForm::Exact`, the pipeline default):

| angle | R, Vs 2250 | R, Vs 2600 | change |
|---|---|---|---|
| 0° | 0.30070 | 0.30070 | 0 |
| 15° | 0.28001 | 0.26323 | −6.0 % |
| 30° | 0.33589 | 0.23563 | −29.8 % |

Over the overburdens in the salt-Vs spec, the 30° top-salt reflection drops
by **about 30–40 %**: −29.8 % in the case above, up to −40.2 % for the
spec's deeper shale case. The spec first quoted 65–82 %.
That figure came from the legacy `det`-typo Zoeppritz kernel (−79.6 % for
the shale case above), not from the textbook form the pipeline uses.

Reproduce:

```text
synthoseis run --e2e --chunked --seed 7 --shape 64,64,256 --faults 4 --store OUT.mdio                 # hash: salt default
synthoseis run --e2e --chunked --seed 7 --shape 64,64,256 --faults 4 --no-salt --store OUT.mdio       # hash: --no-salt
synthoseis run --e2e --chunked --seed 7 --shape 64,64,256 --faults 4 [--no-salt | --salt-legacy-vs] --angles 0,15,30 --store OUT.mdio
cargo run --release -p synthoseis-core --example salt_demo -- /tmp/saltd 7 4 64 64 256              # closures, HC voxels, labels
```

The CLI store hash is the FNV-1a hash of the angle-stack f32 bits (as
`store_hash` in `rock_physics_cli.rs`).

Figures (on the bench box, not committed), from `examples/salt_demo.rs` and
`examples/plot_salt_demo.py`:

- `salt_bodies.png` (`/workspace/synthoseis-bench/out/salt_bodies.png`):
  - Row 1: the inline through the salt, as facies without salt
    (b4f4259), facies with salt, and layer labels with salt, where the
    horizons drag up against the flank.
  - Row 2: 0 / 15 / 30° angle stacks with the salt outlined.
  - Row 3: the salt thickness map, the crossline facies, and the 15°
    stack without salt.
- `salt_validation.png` (`/workspace/synthoseis-bench/out/salt_validation.png`):
  legacy vs Rust histograms of the 10 shape statistics with KS D, the 5 %
  critical and p.

```text
cargo run --release -p synthoseis-core --example salt_demo -- /tmp/saltd 7 4 64 64 256
cargo run --release -p synthoseis-core --example salt_demo -- stats /tmp/salts 16000
python rust/synthoseis-core/examples/plot_salt_demo.py /tmp/saltd /tmp/salts OUT_DIR
```

## Locks

- **Master b4f4259 (`--no-salt`).** The plain, rich, multi-process, deep,
  sandy and faulted-sandy stores equal the angle-stack hashes written by a
  b4f4259 build, and have no `salt_labels` array
  (`no_salt_flag_reproduces_master_b4f4259`). Whole store directories of
  the faulted-sandy and demo runs are byte-identical to b4f4259, except the
  creation timestamp. The same test pins the six salt-default hashes and
  checks `salt_labels` against the salt voxel count.
- **Older master switches still hold.** They are asserted with
  `--no-salt` / `salt: false`:
  - `--toy-lithology alternating` (after #30), `--closures-per-layer`
    (8b5988f), `--closures-unsegmented` (ef2dc42);
  - the core goldens and pinned scenarios in `layered_geometry.rs`,
    `lithology.rs`, `closure_units.rs` and `closure_segments.rs`.
  - Without the switch those hashes now differ, as expected.
- **Invariance matrix** (`tests/rock_physics.rs`): a new salt case (seed
  102, 24×20×128, 4 faults, sand fraction 0.4).
  - The salt sits mid-cube (inlines 7–20, crosslines 5–15, samples
    59–116), so it crosses the inline, crossline and sample chunk
    boundaries of the tilings. The test asserts this.
  - The case keeps closures of 100 voxels or more (default 500), so the
    closures beside the salt reach the model.
  - The test asserts that the salt walls trigger: 70 closure columns
    outside the salt gaps differ from an unwalled fill of the same labels
    (at least 20 required; 508 walled closure columns). The 4 closures kept
    in the model also differ from those of an unwalled model.
  - It is bit-identical across 4 chunk shapes, the classic path,
    streaming ×2, overlap ×2, strip-stitch 2/3/4, multi-process 1/2/3 and
    geometry-once. It differs from `salt: false`. Every MDIO path also
    checks `salt_labels` against the salt body. The streaming, overlap,
    strip-stitch and multi-process paths do this chunk by chunk, with one
    chunk in memory (`salt::verify_salt_labels`); a test corrupts one voxel
    and checks that it is caught.
- **Multi-process:** `--no-salt`, `--salt-legacy-top-offset`,
  `--salt-smooth-all-horizons` and `--salt-legacy-vs` are forwarded to
  workers.
  - Real worker processes accept each switch and match one process
    (`lithology_flags_reach_multiprocess_workers`). That run is on the 8³
    multi-process cube, which has no salt voxels, so it cannot show a
    salt effect.
  - `main.rs::tests::salt_flags_reach_multiprocess_workers` covers salt.
    It rebuilds each worker's config from the forwarded flags through the
    CLI parse chain and checks that it round-trips. It then runs the
    workers' library path on a salt-bearing 12×10×64 cube: multi-process
    equals one process for the default and for `--salt-legacy-vs`, and the
    two differ.
  - The library invariance tests cover multi-process 1/2/3 with salt.
- **GPU:** matches CPU with salt (`lithology_gpu.rs`, two salt cases).
- **Invalid combinations exit 2:**
  - `--no-salt`, `--salt-legacy-top-offset`, `--salt-smooth-all-horizons` or
    `--salt-legacy-vs` with `--toy-geometry planar` or `--legacy-toy-depth`;
  - `--salt-legacy-top-offset`, `--salt-smooth-all-horizons` or
    `--salt-legacy-vs` with `--no-salt`.
- **Planar and `--legacy-toy-depth`:** unchanged (no salt); their goldens
  pass untouched.
- **CI smoke** (`rust-ci.yml`): default (checks the summary line and
  `data/salt_labels`), `--no-salt` (checks there is no `salt_labels`),
  `--salt-legacy-top-offset`, GPU, multi-process and overlap runs.

## Fault labels and salt

Since this change, `data/fault_labels` is **fault AND NOT salt**: no voxel is
both a fault label and salt, and `fault_labels` agrees with `salt_labels`.

- **Why.** Halite is weak and creeps, so brittle faults in the overburden
  die out against salt or sole into it instead of offsetting it (Jackson &
  Hudec 2017, doi:10.1017/9781139003988; Vendeville & Jackson 1992,
  doi:10.1016/0264-8172(92)90047-I). The salt has constant properties, so
  it has zero reflectivity inside: a fault label there has no seismic
  expression. It is label noise.
- **What changes.** Only `data/fault_labels` and the CLI `fault_voxels`
  count. The fault displacement of the sediment labels, the angle stacks,
  layer labels, `salt_labels`, closures and properties are unchanged (the
  tests check the angle-stack hashes). A fault cut by the salt keeps one id
  on both sides.
- **How.** `salt::mask_fault_tile_salt` zeroes the salt run of each column
  in a fault tile (mask and internal `segment_id`), right after the tile is
  computed (streaming and strip / multi-process paths), and
  `generate_fault_labels` does the same per column (classic and chunked
  writers, CLI summary, read-back references). It is per column with no
  halo, so every tiling gives the same labels. The salt body is the one
  already held by the elastic model; no new memory on the streaming paths.
- **Switch.** `--fault-labels-through-salt`
  (`RockPhysicsConfig::fault_labels_through_salt`) keeps the labels inside
  the salt and reproduces master 2b3850ba bit for bit (whole store
  directories identical except the creation timestamp). It exits 2 with
  `--no-salt`, the planar geometry, `--legacy-toy-depth`, or without
  `--faults`. With `--no-salt` the fault labels are those of master
  b4f4259.
- **CLI summary.** `faults: ... fault_voxels=32193, masked_in_salt=1619`
  (or `(--fault-labels-through-salt)` with the switch).

Before / after (`fault ∧ salt` = fault-label voxels inside the salt):

| config | 2b3850ba / `--fault-labels-through-salt` fault voxels | default fault voxels | removed (fault ∧ salt before) | salt voxels |
|---|---|---|---|---|
| demo 64×64×256, seed 7, `--faults 3` or `4` (2 inserted either way) | 33 812 | 32 193 | **1 619** (4.79 %) | 18 535 |
| seed 11, 24×24×128, 4 faults, sand 0.4 (invariance case) | 4 074 | 3 919 | 155 | 2 851 |
| `FAULTED --seed 7` (CLI test) | 9 421 | 8 998 | 423 | 2 336 |
| `FAULTED --seed 4` (CLI test) | 11 003 | 10 293 | 710 | 3 007 |
| `RICH` (CLI test) | 240 | 240 | 0 | 212 |

After the mask, fault ∧ salt is 0 in every case.

Figure (bench box, not committed): `fault_labels_salt_mask.png`
(`/workspace/synthoseis-bench/out/fault_labels_salt_mask.png`). It shows
inline 10 of the demo cube: fault labels before and after the mask over the
salt, and the unchanged 15° stack with the removed labels in blue. To
reproduce it, see `examples/fault_salt_dump.rs` and
`examples/plot_fault_salt_mask.py`.

Tests:
- `tests/salt.rs::fault_labels_exclude_salt` pins the counts above, and
  checks masked == unmasked AND NOT salt voxel for voxel and the tile
  invariant `segment_id != 0 <=> mask == 1`.
- The invariance matrix (`tests/rock_physics.rs`) has the seed-11 case
  masked and with the switch. Its fault labels are identical across 4 chunk
  shapes, streaming ×2, overlap ×2, strip-stitch 2/3/4 and multi-process
  1/2/3. The overlap path writes fault labels only since the overlap
  fault-label fix (see faults-port.md, "Overlapped writer"). The mask does
  not change the angle stack.
- `rock_physics_cli.rs::fault_labels_through_salt_flag_reproduces_master_2b3850ba`
  pins the fault-label hashes of b4f4259 (`--no-salt`), 2b3850ba (switch)
  and the new default, plus the exit-2 cases.

## Deferred

- **Salt-bounded closure typing.** `find_salt_bounded_closures` (the
  `wide_salt` 9-step lateral growth), `grow_to_salt` and the salt-closure
  counts and outputs. Like `grow_to_fault2`, these grow and classify
  closures; they do not create them.
- **Legacy gap maps** (`depth_maps_gaps` output) and the salt QC volumes.
- **Faulted-map smoothing:** legacy's drag blurs fault offsets in the
  horizons. This is not ported, because the Rust faults displace labels.
- **Walls at pinch-outs** (absent-unit NaN cells), which legacy also walls.
- **Python bindings:** salt settings are not exposed yet. `salt_labels` is
  readable from the MDIO store.
- **`--shape` for multi-process / overlap runs.** This limit predates salt,
  so the CLI smoke runs multi-process and overlap on the 8×8×8 default,
  where the salt lies below the cube. The library invariance tests cover
  larger salt cubes on every path.
