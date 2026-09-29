# Toy lithology: legacy sand-fraction Markov chain (default)

## What changed

Until now, every layer of the toy model alternated: even intervals were
shale and odd intervals sand. On the layered geometry the default is now the
legacy lithology from `datagenerator/Horizons.py`, `create_facies_array` →
`Facies.sand_shale_facies_markov` → `MarkovChainFacies`:

1. **Model sand fraction `f`.** Legacy `Parameters` draws `sand_layer_pct =
   rng.uniform(sand_layer_fraction.min, .max)`, which is U(0.05, 0.25) in
   `config/example.json`. Rust draws it once per model, keyed by the seed.
   `--sand-layer-fraction F` fixes it instead.
2. **Sand unit thickness `T` layers.** This is legacy
   `sand_layer_thickness`, 2 by default. `--sand-layer-thickness T` changes
   it.
3. **Two-state Markov chain.** The rows are the current state (shale,
   sand). With `a = f / (T (1 − f))` and `b = 1 / T`, the transition matrix
   is:

   | from \ to | shale | sand |
   |---|---|---|
   | shale | 1 − a | a |
   | sand | b | 1 − b |

   Its stationary sand fraction is `f` and the mean sand run is `T` layers.
   A valid chain needs `f ≤ T / (T + 1)`; other values exit 2.
4. **Initial state.** Legacy `rng.choice(2)`.
5. **One transition per layer**, shallowest first. Legacy
   `rng.choice(p=row)`: `cdf = cumsum(row) / sum(row)`, and the state is the
   number of cdf entries `≤ u`.
6. **Mapping.** Legacy facies index `i` (0 = water) is legacy layer `i`.
   Rust interval `h` (0 = the layer just below the seabed) is legacy layer
   `h + 1`, the same mapping as the depth shifts. So interval `h` takes
   chain state `h`.

Sand layers keep the existing N/G maps, closure fluids and mixing. Shale
layers have no N/G and no closures.

## Switches and goldens

| flags | lithology |
|---|---|
| (none), layered geometry | Markov, f ~ U(0.05, 0.25), T = 2 |
| `--sand-layer-fraction F --sand-layer-thickness T` | Markov with fixed f / T |
| `--toy-lithology alternating` | previous rule; reproduces master after #30 bit for bit |
| `--toy-geometry planar` or `--legacy-toy-depth` | always alternating (33a3a93 / 10f4dcd goldens unchanged) |

- **Rejected combinations.** Markov options with the planar geometry, and
  sand options with `alternating`, exit 2.
- **Workers.** All options are forwarded to multi-process workers.
- **Config.** In the API: `RockPhysicsConfig { lithology, sand_layer_fraction,
  sand_layer_thickness }`, `E2eConfig::effective_lithology()` and
  `synthoseis_core::lithology`.
- **Goldens.**
  - The layered goldens from #30 are pinned under `alternating`, both in
    core (labels and stack15 on the demo cube) and as four CLI stores from
    the master binary after #30.
  - The Markov default is pinned on the demo cube. Its labels are unchanged;
    the stack differs.

## Validation (`rust/synthoseis-core/tests/lithology.rs`)

Fixture: `tests/fixtures/lithology_reference.json`, written by
`tests/fixtures/generate_lithology_reference.py`, which runs the real legacy
`Facies` / `MarkovChainFacies`.

**Bit-exact where legacy is deterministic.** Legacy draws come from numpy
PCG64. Rust uses keyed hashes instead, which keeps the draws tiling and
process invariant. So the check replays legacy's own draws:

- For 10 chains (f from 0.05 to 0.6666, T in {1, 2, 3, 5}, 200 layers each),
  the Rust chain reproduces all 2000 legacy facies exactly.
- The transition matrix matches to the bit (floats are compared as IEEE
  bits).
- The generator asserts that the replay reproduces legacy before writing.

**Statistical checks** use 3000 legacy models against 3000 Rust seeds, each
with 60 layers, f ~ U(0.05, 0.25) and T = 2:

| check | Rust | legacy | test |
|---|---|---|---|
| drawn fraction | — | — | KS 0.021 vs legacy, 0.015 vs U(0.05, 0.25) (5 % critical 0.035 / 0.025) |
| per-model sand proportion | mean 0.1526 | mean 0.1518 | KS 0.014 (5 % critical 0.035) |
| sand run lengths | — | — | χ² 10.4, df 11 (5 % critical 19.7) |
| shale run lengths | — | — | χ² 5.4, df 11 (5 % critical 19.7) |
| stationary f / mean run, fixed (f, T) | 0.1016 / 1.96, 0.2498 / 1.98, 0.4004 / 2.96 | 0.1, 0.25, 0.4 / T | within 0.02 and 10 % |

**Demo cube (64×64×256, seed 7, f = 0.175).**

| | sand layers | sand, % of sediment | closures (brine / oil / gas) | hydrocarbon voxels |
|---|---|---|---|---|
| Markov | 15 of 46 | 30 % | 6 / 7 / 6 | 27k |
| alternating | 23 of 46 | 52 % | 12 / 12 / 8 | 45k |

The 15° stack changes by 79 % relative RMS.

## Invariance and memory

- **Invariance matrix.** It adds an explicit Markov case (f = 0.4, T = 3,
  rich settings); the two layered cases from #30 now run the Markov default.
  All are bit-identical across chunk shapes, the classic path, streaming,
  overlap, strip-stitch 2/3/4, multi-process 1/2/3 and geometry-once.
- **CLI.** Single process and multi-process give identical output for
  Markov, fixed f / T, `alternating` and planar.
- **Determinism.** The chain is a prefix-stable function of `(seed, f, T)`:
  interval `h` does not depend on the layer count, and every worker rebuilds
  it identically.
- **Memory.** One bool per interval. Shale layers skip their N/G and contact
  maps, so the model gets smaller.

## Figure

`toy_lithology.png`, from `examples/lithology_demo.rs` and
`examples/plot_lithology_demo.py`:

- facies, fluids, rfc15 and 15/30° stacks for both rules;
- the per-layer lithology logs and the stack difference;
- the Rust vs legacy population of per-model sand proportions.

## Deferred

- ~~**Closures per lithology unit.**~~ Done: see
  [closures-per-sand-unit.md](closures-per-sand-unit.md). The demo-cube
  numbers above are per layer (`--closures-per-layer`).
- **Onlaps and fans.** The legacy overrides are not ported: the layer below
  an onlap surface is shale, and fans are sand wrapped in shale. The toy
  geometry has neither.
- **Variable shale N/G.** `variable_shale_ng` is not ported (off in
  `config/example.json`).
- **Python bindings** do not expose the lithology options.
