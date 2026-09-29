# 3D closure segmentation across faults

Closures per sand unit (#32) are now segmented in 3D, the way legacy
`datagenerator/Closures.py` does it. Each 3D connected piece of closure
sand becomes its own compartment with its own fluid draw:

- A trap that a fault offsets by more than the closed sand is thick splits
  into separate compartments (fault-bounded compartments).
- Closed sand juxtaposed across a fault joins the closures on both sides
  into one compartment, even when they belong to different sand units.
- Each compartment keeps the spill point of its own 2D closure (fault
  block), and the min-voxel filter applies per compartment.

`--closures-unsegmented` (`RockPhysicsConfig::closures_unsegmented`)
restores master ef2dc42 bit for bit. `--closures-per-layer` (8b5988f), the
planar geometry and `--legacy-toy-depth` are unchanged. Code:
`synthoseis-core/src/closure_segments.rs`.

## What legacy does

`create_closures` → `create_closure_labels_from_depth_maps` →
`segment_closures` → `assign_fluid_types`:

1. **Closure map per unit.** `_flood_fill` on the unit's post-fault top map
   (`faults.faulted_depth_maps_gaps`), then `cd = min(max(cd, top), base)`.
   `_flood_fill` keeps the per-column fill level and caps each
   8-connected closed region of ≥ 50 cells at `crest + max_column`. The
   closure cube marks samples `top + 1 .. int(cd)`.
2. **Segmentation.** `segment_closures` clips the closure cube, applies a
   `3x3x1` opening (minimum then maximum filter), restricts it to sand,
   labels it with `measure.label(connectivity=2)` (3D 18-connectivity), and
   applies `remove_small_objects(closure_min_voxels)`.
3. **Fluids.** `assign_fluid_types` draws one `rng.integers(3)` (brine /
   oil / gas) per 3D component.
4. **Faulted closures.** `find_faulted_closures` / `grow_to_fault2` grow
   closures inside fault blocks toward the fault plane, with per-type size
   filters.

`_flood_fill` also has fault-gap walls: cells with depth < 1 (fault gaps
in `faulted_depth_maps_gaps`) are zeroed, and their 8-neighbours are
deepened by the max column so closures stop at the gap. In current legacy
the gaps are only NaN (then 0) for zero-thickness layers, because
`Faults.partial_faulting`, which inserts the fault gaps, is never called.
So the walls never engage in legacy runs, and a faulted top map is a
continuous surface with cliffs.

## What the port does

1. **Closure map.** Unchanged from ef2dc42 in voxels. Per column, the
   closure is `[top, ceil(min(fill, crest + max_column, base)))`.
   - The fill level is constant over a connected closed region. A fault
     block that forms its own pit is its own region with its own crest and
     spill point, as in legacy.
   - Unchanged deviations from #32: 4-connected regions, every region
     capped, no border zeroing (see Deferred).
2. **Segmentation.** The closure runs of all units form 3D components with
   18-connectivity. Same-column and in-plane face neighbours connect when
   their sample ranges touch (|dk| ≤ 1); in-plane diagonals need an
   overlap (dk = 0). `closure_segments::segment_runs` uses union-find on
   runs, so memory is O(closure columns). Components below
   `min_closure_voxels` stay brine.
3. **Fluids.** One draw per compartment:
   - A compartment that contains the first column of a 2D closure is keyed
     like ef2dc42, `closure_fluid(seed, unit top, rank)`, using the smallest
     `(unit top, rank)` present. An unsplit closure keeps its ef2dc42
     fluid, and fault-free cubes are bit-identical to ef2dc42.
   - A split-off compartment is keyed
     `(seed, STREAM_FLUID, unit top, rank, 1 + first column)`.
4. **Determinism.** The segmentation runs on the full label volume, which
   every path already rebuilds, inside `RpmModel::build`. It is identical
   across tiling, workers, processes, streaming and GPU.

## Validation

Fixture `tests/fixtures/closure_segments_reference.json` comes from
`tests/fixtures/generate_closure_segments_reference.py`, which runs the
real legacy code (numpy 2.5.3, scipy 1.18.1, scikit-image 0.26.0). Tests
are in `synthoseis-core/tests/closure_segments.rs`.

### Deterministic: bit-exact

| check | legacy | inputs | result |
|---|---|---|---|
| per-column closure depth | `Closures._flood_fill` + `min(max(cd, top), base)` | 24 faulted domes (1–4 domes, 1–2 fault cliffs with 3–21 samples throw, max column 12 / 20 / 37.5 / 1e9, varying unit thickness) | 8 054 closed columns: contact bit-exact (`2·cd` as integer), same closed set; 6 111 clamped to the unit base, where a fault offsets the unit |
| closure voxel count | `int(cd) − top` | same | equal on every column with an integer contact. 295 columns with a fractional cap (37.5) get one more voxel (`ceil` vs `int`), a pre-existing rounding difference kept from ef2dc42 |
| 3D components + min-voxel filter | `Closures.segment_closures` (stub `self`) | 40 run sets on up to 24×24×40 grids, min voxels 1 / 20 / 60 / 150 | 11 772 runs, 291 kept components: identical partition and kept/removed set (4 905 removed runs) |
| split by a fault | `segment_closures` on the same cube | dome with a fault at i = 20, unit 4 samples thick | throw 8 → 2 compartments, throw 3 → 1, in both (`fault_offset_splits_a_closure`) |

The fixture's cases are drawn so that the legacy details not ported cannot
matter. The generator asserts it: every depth is ≥ 1, a ≥ 5-cell deep
plateau lines every edge, closed regions have ≥ 50 cells and the same
4- and 8-connectivity, and footprints of the run sets are aligned 3×3
blocks (the `3x3x1` opening is a no-op).

### Random: at the 5 % level

Legacy draws `rng.integers(3)` per component, which is uniform and
independent. χ² with p from scipy:

| draws | n | statistic | 5 % critical | p |
|---|---|---|---|---|
| split-off compartment keys, uniform | 2 000 000 | χ² = 0.600 (df 2) | 5.991 | 0.741 |
| split-off vs the primary draw of the same closure, 3×3 independence | 2 000 000 | χ² = 2.398 (df 4) | 9.488 | 0.663 |
| kept compartments of 2 000 faulted sandy models (24×20×128, 4 faults, f = 0.5, T = 1), uniform | 3 073 (1 040 / 1 038 / 995) | χ² = 1.262 (df 2) | 5.991 | 0.532 |
| multi-unit (fault-joined) compartments of those models, uniform | 502 (172 / 184 / 146) | χ² = 4.510 (df 2) | 5.991 | 0.105 |
| test-size split-off keys, uniform (`split_compartment_fluid_uniform_and_independent`) | 30 000 | χ² = 1.253 (df 2) | 5.991 | — |
| test-size split-off vs primary | 30 000 | χ² = 6.353 (df 4) | 9.488 | — |

None rejects at 5 %. The 2 000 small models have no split-off compartment
(splits need throw greater than the closed thickness inside one closed
region), so split-off draws are tested directly.

Statistics come from
`cargo run --release -p synthoseis-core --example closure_segments_demo -- stats OUT 2000`,
with p printed by `examples/plot_closure_segments_demo.py`.

## Effect on the demo cube (64×64×256, seed 7, 4 faults, Markov default)

| | closures (brine / oil / gas) | compartments | HC voxels | closure voxels | angle stacks 0 / 15 / 30° |
|---|---|---|---|---|---|
| `--closures-unsegmented` (ef2dc42) | 2 / 2 / 2 | — | 8 022 | 19 788 | — |
| segmented (default) | 2 / 2 / 2 | 6 kept (11 below min voxels), 0 split-off, 0 joined | 8 022 | 19 788 | identical (0 changed samples) |

On the demo cube, no fault splits or joins a closure, so the output equals
ef2dc42. The CLI store hash `0xe69121b8b492872b` is the same for both
flags and the ef2dc42 binary.

Across seeds on the same setup (seeds 0–15), 3 of 16 cubes have a
compartment that joins pieces, and 2 change the stack (15° relative RMS
0.0044 and 0.0324). None has a split-off compartment.

**Faulted sandy figure cube** (64×64×256, seed 4, 8 faults, f = 0.3,
T = 1):

| | closures (brine / oil / gas) | compartments | HC voxels (oil / gas) | closure voxels | stack rel RMS 0 / 15 / 30° |
|---|---|---|---|---|---|
| ef2dc42 | 4 / 1 / 2 | — | 5 392 (766 / 4 626) | 11 104 | — |
| segmented | 3 / 13 / 7 closure pieces | 7 kept (3 brine / 3 oil / 1 gas): 3 join 2+ units, 1 split-off | 6 148 (3 200 / 2 948) | 11 121 | 0.107 / 0.111 / 0.121 (18 588 samples changed) |

Across seeds 0–15 on that setup, 14 of 16 cubes change (15° relative RMS
0.004–0.21) and 2 have a split-off compartment.

Figure: `closures_faults.png` (at
`/workspace/synthoseis-bench/out/closures_faults.png` on the bench box),
from `examples/closure_segments_demo.rs` and
`examples/plot_closure_segments_demo.py`:

```text
cargo run --release -p synthoseis-core --example closure_segments_demo -- /tmp/csf 4 8 64 64 256 0.3 1
cargo run --release -p synthoseis-core --example closure_segments_demo -- /tmp/csd 7 4 64 64 256
cargo run --release -p synthoseis-core --example closure_segments_demo -- stats /tmp/css 2000
python rust/synthoseis-core/examples/plot_closure_segments_demo.py /tmp/csf /tmp/csd /tmp/css closures_faults.png
```

## Locks

- **Master ef2dc42:** `--closures-unsegmented` reproduces the ef2dc42 CLI
  stores bit for bit. The plain, rich, multi-process, deep and sandy
  (32×32×128 f = 0.5 T = 3) stores, plus two faulted sandy stores
  (32×32×128, 4 faults, f = 0.5, T = 1, seeds 7 and 4: `0x7d718e21c6a7aaab`,
  `0xac94e1795801bd3d`) were written by an ef2dc42 build
  (`closures_unsegmented_flag_reproduces_master_ef2dc42`).
  - The segmented default equals ef2dc42 on the first five and differs on
    the two faulted ones.
  - The core goldens (`tests/lithology.rs` Markov unit demo
    `0x82708d880146cf10`, `tests/layered_geometry.rs`) are asserted under
    the flag. The segmented demo stack is the same hash.
  - Without faults, segmented equals unsegmented bit for bit
    (`faults_merge_juxtaposed_closures_and_switch_restores_ef2dc42`).
- **Invariance matrix** (`tests/rock_physics.rs`): a new case, seed 7,
  24×20×128, 4 faults, f = 0.5, T = 1, where faults join closures of
  different units. It is bit-identical across chunk shapes, the classic
  path, streaming, overlap, strip-stitch 2/3/4, multi-process 1/2/3 and
  geometry-once, and differs from the unsegmented model.
- **CLI:** the segmented faulted stores are the same for `--chunk-i 16` and
  3×7×16 tiles. With `--closures-unsegmented`, multi-process gives the same
  output as a single process (the flag is forwarded to workers).
- **GPU:** matches CPU on the faulted segmented case (1.5e-7).
- **Invalid combinations** exit 2: `--closures-unsegmented` with
  `--legacy-toy-depth`, `--no-fluids`, `--closures-per-layer`, or the
  planar geometry.
- **Planar and `--legacy-toy-depth`:** unchanged; they never segment.
- **Memory:** one set of top/base/fill maps per unit at a time, plus the
  closure runs (one small record per closed column per unit) and
  union-find arrays over runs. It does not depend on chunking. The label
  volume was already rebuilt in full by every path.
- **CI smoke:** default, `--closures-unsegmented`, GPU and multi-process
  runs in `rust-ci.yml`.

## Deferred

- **Fault-gap walls** in `_flood_fill`. They are dead in current legacy
  (`Faults.partial_faulting` is never called). Port them together with
  fault gaps if legacy's partial faulting is revived.
- **The `3x3x1` opening** in `segment_closures`. It trims closure edges
  narrower than 3 columns. Porting it would change fault-free cubes versus
  ef2dc42 and legacy placement parity from #32, so it stays out.
- **`find_faulted_closures` / `grow_to_fault2`** and legacy's per-type
  (oil / gas / brine) size filters.
- **Legacy `_flood_fill` specifics**, as in #32: 8-connected regions with
  the ≥ 50-cell cap threshold, 3-cell border zeroing, and `int(cd)` vs
  `ceil` on fractional max columns.
- **Python bindings** do not expose `--closures-unsegmented`.
