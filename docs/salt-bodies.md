# Salt bodies

The Rust core now inserts a salt body into every layered model, as legacy
`datagenerator/Salt.py` does. Legacy's shipped `config/example.json` has
`include_salt: true`, so salt is **on by default** in the layered geometry.
The port covers:

- the salt shape: a convex hull around a cloud of 218 jittered points,
- the horizon drag against the salt flanks,
- the salt rock properties (rho 2.17 g/cc, Vp 4500 m/s, Vs 2250 m/s),
- the closure walls that seal traps against salt,
- a `salt_labels` volume (0/1) in the MDIO store.

Switches:

- `--no-salt` (`RockPhysicsConfig::salt = false`) reproduces master
  b4f4259 bit for bit.
- `--salt-legacy-top-offset` (`RockPhysicsConfig::salt_legacy_top_offset`)
  uses legacy's absolute top offset (see Deviations).
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
   - The horizon is lifted by `2 r` samples in its salt columns.
   - Every horizon is smoothed with `gaussian_filter(sigma = 3)`.
   - Then `push_down_remove_negative_thickness` runs.
   - Near salt the result is horizons that bend up against the flanks
     (drag).
   - The horizon gaps are the columns where a dragged horizon lies inside
     salt.
3. **Lithology and properties.**
   - Lithology is set to 2 inside salt. The age volume is rebuilt from the
     dragged maps.
   - In `Seismic.py`, `lith == 2` voxels get `rho = 2.17`, `vp = 4500`,
     `vs = 2250` after the layer loop. The base forward-fill runs after
     that, and the final scaling factors skip salt.
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
| drag | shift, scipy `gaussian_filter(sigma=3)` replayed exactly (scipy's kernel weights stored as IEEE bits, `reflect` edges, axis 0 then 1, scipy's symmetric summation order), push-down | bit-exact |
| geometry | the dragged maps are rounded to whole samples, as the layered geometry already does, and fed to label fill, faults and the depth model | Rust has whole-sample horizons |
| properties | `VoxelKind::Salt` → `SALT` (2.17, 4500, 2250 as f32) before the forward-fill. It overrides water, layers and closures | as legacy |
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

## Deviations (documented, not bugs)

- **Top offset scaled by `nk / 1250`.** Legacy adds 150–300 samples below
  horizon 1, sized for its 1250-sample example cube. In a 256-sample toy
  cube that would put the salt entirely below the data. The default scales
  the offset by `samples / 1250`, which is exactly legacy at 1250 samples.
  `--salt-legacy-top-offset` keeps the absolute legacy offset. The radius
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
2. **Every horizon is smoothed**, including horizons that never touch salt.
   Kept. On the demo cube this moves horizons by about a sample in 2.5 % of
   the voxels far from the salt (see Effect).
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

| | salt | closures (brine / oil / gas) | HC voxels (oil / gas) | CLI store hash |
|---|---|---|---|---|
| `--no-salt` (= master b4f4259) | — | 2 / 2 / 2 | 8 022 (5 161 / 2 861) | `0xe69121b8b492872b` (same as the b4f4259 binary) |
| salt (default) | top 65.9 samples, R 12.2 columns, 18 535 voxels (1.8 %) in 597 columns | 3 / 1 / 2 | 5 645 (3 850 / 1 795) | `0xd63947b9302852f7` |

Angle-stack change (relative RMS of the difference):

| angle | rel RMS | changed samples |
|---|---|---|
| 0° | 0.602 | 582 457 of 1 048 576 |
| 15° | 0.587 | 592 312 |
| 30° | 0.506 | 595 721 |

- Labels change in 5.9 % of the voxels, 4.7 % of them outside the salt
  (the salt itself is 1.8 % of the cube).
- More than 8 columns from any salt column, 2.5 % of labels still change
  (legacy smooths every horizon, then the maps are rounded). That alone
  gives a 15° relative RMS of 0.41 there, because one-sample shifts of
  sharp reflections dominate an RMS difference. Near the salt the relative
  RMS is 0.87.
- HC voxels far from the salt are almost unchanged (308 → 302). The
  closures near the salt are reorganised by the salt, the drag and the
  walls (7 714 → 5 343).

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
  30, 24×20×128, 4 faults, sand fraction 0.4, with a closure beside the
  salt). It is bit-identical across 4 chunk shapes, the classic path,
  streaming ×2, overlap ×2, strip-stitch 2/3/4, multi-process 1/2/3 and
  geometry-once. It differs from `salt: false`. Every MDIO path also
  checks `salt_labels` against the salt body.
- **Multi-process:** `--no-salt` and `--salt-legacy-top-offset` are
  forwarded to workers; multi-process equals a single process for both and
  for the default.
- **GPU:** matches CPU with salt (`lithology_gpu.rs`, two salt cases).
- **Invalid combinations exit 2:**
  - `--no-salt` or `--salt-legacy-top-offset` with `--toy-geometry planar`
    or `--legacy-toy-depth`;
  - `--salt-legacy-top-offset` with `--no-salt`.
- **Planar and `--legacy-toy-depth`:** unchanged (no salt); their goldens
  pass untouched.
- **CI smoke** (`rust-ci.yml`): default (checks the summary line and
  `data/salt_labels`), `--no-salt` (checks there is no `salt_labels`),
  `--salt-legacy-top-offset`, GPU, multi-process and overlap runs.

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
