# Closures per sand unit

By default, the layered toy model now finds hydrocarbon closures the way
legacy `datagenerator/Closures.py` does. Consecutive sand layers merge into
one **sand unit**, and each unit gets closures on its top only. Before this
change (master 8b5988f), every sand layer got its own closures, so a sand
unit two layers thick had closures on both layers.

`--closures-per-layer` (`RockPhysicsConfig::closures_per_layer`) switches
back to the previous behaviour. With the same other flags, it reproduces
master 8b5988f bit for bit.

## Legacy rule (ported)

- **Unit grouping** (`Closures.find_top_lith_horizons`):
  - `top_lith_indices` holds every legacy layer `i` in `1 .. len(facies) - 2`
    whose facies differs from layer `i - 1`. Legacy `facies[0]` is water, so
    layer 1 always qualifies. The toy has no onlaps, so `onlap_list - 1`
    adds nothing.
  - Each run between two kept tops is one lithology unit.
- **Which units get closures** (`create_closure_labels_from_depth_maps`):
  - The loop is `for ihorizon in range(len(top_lith) - 1)` and keeps
    `top_lith_facies > 0`, so only sand units get closures.
  - The deepest unit, which runs into the model base, is never processed.
  - Rust: `lithology::closure_units(sand)`. `sand` has one entry per
    interval, and the last entry is legacy `facies[max_layers]`, below the
    deepest horizon.
- **Placement:**
  - The top map is the unit's top horizon. The base map is the next
    lithology-change horizon, i.e. the unit base, which lies below all of
    the unit's layers.
  - The closure depth is `min(max(flood_fill(top), top), base)`, and every
    voxel from the unit top down to that depth is labelled. The
    hydrocarbon column therefore crosses the internal horizons of the unit.
  - Rust: `rock_physics::unit_fluids(labels, shape, members, key, ...)`.
    - Per column, the top is the first sample of any member label, and the
      base is the end of that contiguous run of member labels.
    - It then applies the same priority-flood fill, 4-connected closures,
      spill point and max column as `layer_fluids`.
    - With one member it is exactly `layer_fluids`.
- **Fluids:** legacy `segment_closures` takes 3D connected components of
  the closure voxels, and `assign_fluid_types` draws one `rng.integers(3)`
  per component. So a closure has one fluid across all layers of its unit.
  - Rust draws one fluid per closure, keyed by
    `(seed, unit top interval, closure rank)`.
  - All member layers share the unit's contact and fluid maps. A sand voxel
    of any member is hydrocarbon if `k < contact`, otherwise brine.
  - The closure list (`LayerFluids::closures`) is reported once, on the
    shallowest member.
  - Sand layers of the skipped deepest unit are brine.
- **Where the old rule stays:** the planar geometry (and so
  `--legacy-toy-depth`) always uses per-layer closures. Its only sand layer
  is the deepest unit, which the legacy rule would skip. Planar and legacy
  toy goldens are unchanged.

## Validation

Fixture `tests/fixtures/closure_units_reference.json` comes from
`tests/fixtures/generate_closure_units_reference.py`, which calls the real
legacy `Closures.find_top_lith_horizons` (stub `self`), `Facies` /
`MarkovChainFacies` and `flood_fill_heap`. The tests are in
`rust/synthoseis-core/tests/closure_units.rs`.

All statistical tests use the **5 % level** with fixed, pre-set designs.
KS criticals are the asymptotic two-sample values `1.358 sqrt((n + m) / (n m))`,
which are conservative for these discrete samples. p-values come from scipy
on the same data.

| check | Rust | legacy / expected | statistic | 5 % critical | p |
|---|---|---|---|---|---|
| unit grouping and selection, 222 facies sequences (200 legacy Markov chains, 22 edge cases) | — | — | bit-exact, all 222 | — | — |
| closure placement, 2-layer unit with pinch-outs (40×36 columns) | 5016 voxels | 5016 voxels | per-column counts bit-exact | — | — |
| closure units per model (3000 vs 3000 models, 40 layers, f ~ U(0.05, 0.25), T = 2) | mean 2.988 | mean 2.945 | KS D 0.0153 | 0.0351 | 0.87 |
| multi-layer closure units per model | mean 1.467 | mean 1.448 | KS D 0.0140 | 0.0351 | 0.93 |
| sand layers inside closure units per model | mean 5.817 | mean 5.742 | KS D 0.0130 | 0.0351 | 0.96 |
| closure unit thickness (layers; 8965 vs 8836 units) | mean 1.947 | mean 1.949 | KS D 0.0027 | 0.0204 | 1.00 |
| fluid draw `closure_fluid`, 2M draws (seeds 0..10000 × unit-top layers 0..50 × closure ranks 0..4), brine / oil / gas | 666 344 / 667 585 / 666 071 | 1/3 each (`rng.integers(3)`) | χ² 1.95, df 2 | 5.991 | 0.38 |
| fluid independent of the unit-top layer (50 × 3 table) | — | — | χ² 104.7, df 98 | 122.1 | 0.30 |
| fluid independent of the closure rank (4 × 3 table) | — | — | χ² 4.85, df 6 | 12.59 | 0.56 |
| fluid through the closure path (`unit_fluids`), 30 000 seeds | 10 008 / 9 946 / 10 046 | 1/3 each | χ² 0.51, df 2 | 5.991 | 0.78 |

The first version of this PR checked the split on only 3000 seeds (one
unit-top key). It got 949 / 985 / 1066, χ² 7.18, p = 0.028, which fails at
5 %, and it wrongly quoted the 0.1 % critical (13.8). At large n there is no
bias: the draw is `floor(3 u)` of a 53-bit keyed splitmix64 unit, so the
rounding bias is below 1e-15. The 3000-seed result was an ordinary
fluctuation (about 1 in 36). No code change to the draw was needed, so
every golden, including `--closures-per-layer` = 8b5988f, is unchanged.

On the placement case, the same unit under per-layer closures gives 5443
voxels in 2 + 3 closures (upper and lower layer). Legacy closes the upper
layer alone at 1471 voxels. The unit closure is a single closure.

## Effect on the demo cube (64×64×256, seed 7, 4 faults, f = 0.175, T = 2)

| | closures (brine / oil / gas) | hydrocarbon voxels |
|---|---|---|
| per layer (`--closures-per-layer`, 8b5988f) | 6 / 7 / 6 | 27 212 |
| per unit (default) | 2 / 2 / 2 | 8 022 |

- 15 of 46 intervals are sand, in 6 units: intervals 0–1, 5–6, 18–21,
  27–28, 37–39 and 43–44. All six are eligible, because the deepest unit is
  shale here.
- Five of the units close, with 6 closures in total. The unit at 43–44
  does not close.
- The per-layer run found 19 closures, 13 of them on layers below a unit
  top.
- The 15° stack changes by 23.9 % relative RMS.
- Labels are unchanged.

Figure: `closures_per_unit.png`, from `examples/closures_unit_demo.rs` and
`examples/plot_closures_unit_demo.py`:

```text
cargo run --release -p synthoseis-core --example closures_unit_demo -- /tmp/cu
python rust/synthoseis-core/examples/plot_closures_unit_demo.py /tmp/cu OUT_DIR
```

## Locks

- **Invariance matrix** (`tests/rock_physics.rs`): two new cases, seed 6,
  24×20×128, f = 0.4, T = 3, rich filters and 4 faults, one per unit and
  one per layer. The per-unit case has a multi-layer unit that closes. Both
  are bit-identical across chunk shapes, the classic path, streaming,
  overlap, strip-stitch 2/3/4, multi-process 1/2/3 and geometry-once. The
  existing layered Markov cases now run per unit.
- **Master 8b5988f:**
  - `--closures-per-layer` reproduces the 8b5988f CLI stores (plain, rich,
    multi-process, deep, and a 32×32×128 f = 0.5 T = 3 store) and the core
    Markov demo stack (`0xd952d8da8616c9e3`).
  - The alternating (#30) goldens, the planar goldens and the
    `--legacy-toy-depth` goldens are unchanged without the flag.
- **Multi-process:** gives the same output as a single process with and
  without the flag. The flag is forwarded to workers.
- **Invalid combinations** exit 2: `--closures-per-layer` with
  `--legacy-toy-depth`, with `--no-fluids`, or with the planar geometry.
- **GPU:** matches CPU on a multi-layer closing unit (1.7e-7).
- **Memory:** the same as per layer. There is one contact and fluid map per
  sand label, cloned from the unit's, and one temporary top/base map pair
  while a unit is computed. It does not depend on chunking.

## Deferred

- **Legacy `_flood_fill` specifics.** Legacy zeroes a 3-cell border, adds
  the max column near empty picks and drops the 2 % percentile. Rust ports
  only the core `flood_fill_heap`, as before.
- **3D closure segmentation across faults.** Legacy merges closure voxels
  into 3D connected components and removes shale-sealed parts. Rust keeps
  2D 4-connected closures of the post-fault unit top, so a faulted closure
  may split differently.
- **Onlap tops.** `onlap_list - 1` is not ported; the toy geometry has no
  onlaps. Fan overrides are not ported either.
- **Variable max column height.** Legacy uses `variable_max_column_height`,
  which is constant 150 m in `config/example.json`.
- **Python bindings** do not expose `--closures-per-layer`.
