# Layered toy geometry (default) and `--toy-geometry planar`

## Why

The master toy geometry is two seed-dependent dipping planes over a flat
base: 3 labels and 2 interfaces per trace. With it, two parts of the ported
rock-physics model never ran. Random depth shifts start at legacy layer 20
(`first_random_layer`), and the closure analysis needs a structural high to
find traps, so oil / gas / brine selection never triggered. The layered
geometry keeps the same pipeline (labels → faults → elastic model → Zoeppritz
→ stacks) and replaces only the horizon maps.

## Recommendation and switches

**Layered is the default toy geometry.** The old geometry stays available,
bit for bit:

| flags | geometry | rock physics | Zoeppritz | reproduces |
|---|---|---|---|---|
| (none) | layered | corrected | textbook | new layered goldens |
| `--toy-geometry planar` | planar | corrected | textbook | PR #29 (Zoeppritz fix) outputs |
| `--toy-geometry planar --legacy-zoeppritz` | planar | corrected | legacy `det` | master 33a3a93 |
| `--legacy-toy-depth` | planar (implied) | master toy | legacy (implied) | master 10f4dcd |

- `--legacy-toy-depth` implies planar, because the 10f4dcd guarantee needs
  it. An explicit `--toy-geometry layered` with it exits 2, while
  `--toy-geometry planar` with it is accepted.
- `--legacy-zoeppritz` does **not** imply planar. It only switches the
  Zoeppritz expression, so it also works on the layered geometry.
- The Rust API has `E2eConfig.geometry: ToyGeometry` (default `Layered`) and
  `E2eConfig::effective_geometry()`. Multi-process workers receive
  `--toy-geometry`. The runners also gained `run_e2e_with_geometry` and
  `run_e2e_strip_stitched_with_geometry`.
- The CLI summary prints `toy geometry: layered (N horizons)`.

Why default rather than behind a flag: the rock-physics defaults (shifts
from layer 20, closure fluids) only mean something on a many-layer
structure. The planar default gave 0.8 % non-zero reflectivity, which made
the demo cube a poor stand-in for real synthoseis output. All existing
goldens are pinned under `planar` and still pass unchanged.

## Construction (`synthoseis-core/src/toy_geometry.rs`)

All draws are keyed hashes of `(seed, stream 0x6E0, purpose, index)`, the
same scheme as the rock-physics module. The maps are therefore a pure
function of `(seed, ni, nj, nk)` and are independent of the chunking, the
worker count and the code path.

1. **Seabed minimum:** a legacy-style integer draw in `[20, 50)` m, divided
   by 4 m/sample and capped at 15 % of the column.
2. **Base horizon:** starts at `nk + 10` samples. A Gaussian dome is
   subtracted from it: centre U(0.35, 0.65) of the grid, sigma
   U(0.22, 0.35)·min(ni, nj), relief U(0.10, 0.18)·nk. A regional tilt is
   added, at most a quarter of the relief over the half extent. The base is
   then lowered so that it lies below the cube everywhere, which leaves no
   unfilled samples at depth.
3. **Layers, built bottom-up:**
   - Each thickness is `2 + Gamma(4, 1)` samples, the legacy
     `stats.gamma.rvs(4, 2)`: mean 6, variance 4, minimum 2.
   - Each thickness is modulated laterally by a normalised fbm map with
     strength U(0.1, 0.35) per layer.
   - Each thickness is thinned over the crest by `1 − growth · dome`, with
     `growth = 0.8 · relief / column`. This is syn-depositional growth, so
     the relief decays upward and the shallow section is flatter.
   - Stacking stops when a horizon would reach the seabed minimum. The
     shallowest horizon is the seabed.
   - The result is capped at 255 horizons (u8 labels).
4. **Rounding:** horizons are rounded to whole samples. The label fill uses
   `[ceil(z_h), floor(z_h+1))`, so fractional horizons would leave a
   one-sample 255 gap under every horizon. After rounding, the only 255
   samples are the water column.
5. **Downstream:** faults, the depth trace, N/G maps, shifts, closures and
   Zoeppritz are unchanged. The layers alternate shale and sand (even and
   odd labels) as before.

## Results (64×64×256, seed 7, 4 faults requested)

- 47 horizons and 46 layers. Shifts are active on 23 of the 26 layers at or
  beyond interval 20. The legacy draw can give a zero shift, which accounts
  for the rest.
- 32 closures: 12 brine, 12 oil and 8 gas. 45,118 hydrocarbon voxels.
- 15° raw reflectivity is non-zero in 31.4 % of samples, against 0.78 % for
  planar. The maximum |r| is below 1 everywhere.
- The shifts change 23 % of the rfc15 samples (rel RMS 27 %). The closure
  fluids change 18 % of the stack15 samples (rel RMS 37 %).
- Standard 64×64×128 demo cube: 22 layers, shifts on 2, 6 hydrocarbon
  closures.

Figures (from `examples/layered_geometry_demo.rs` and
`examples/plot_layered_geometry_demo.py`):

- `layered_geometry_sections.png`:
  - labels, facies and fluid, Vp and rfc15 on an inline and a crossline
    through the crest;
  - the closure map and a horizon depth map;
  - planar labels and rfc15 for comparison.
- `layered_geometry_stacks.png`:
  - 0/15/30° stacks on the inline and the crossline;
  - the planar stack and traces through a hydrocarbon sand;
  - per-layer shifts, the rfc15 change from the shifts and the stack15
    change from the fluids.

## Legacy statistics (tests/layered_geometry.rs)

| check | result |
|---|---|
| thickness mean / var, 20k draws | 6.004 / 3.960 (legacy 6 / 4), KS D 0.0033 vs 5 % critical 0.0096 |
| seabed minimum | integer metres, all 30 values in [20, 50) drawn |
| layer count ≈ (nk − seabed) / 6 | nk 256: 46 (~41); 512: 92 (~83); 1250: 216 (~206); within 20 % + 3 |
| closure fluids over 24 seeds | brine 73 / oil 76 / gas 61, χ² 1.80 (5 % crit 5.99) |

## Invariance and memory

- The default-model invariance matrix gained two layered cases:
  - rich (Backus, shifts, closures, filters, noise, faults);
  - plain default at 24×20×160, where the default shifts are asserted
    active.
- Both cases are bit-identical across chunk shapes, the classic path,
  streaming, overlap, strip-stitch 2/3/4, multi-process 1/2/3 and
  geometry-once.
- Golden hashes are pinned for the layered 64×64×128 demo cube: labels and
  the 15° stack.
- The model keeps the unfaulted horizon maps as `(ni, nj, nh)` f64, which is
  about 8 bytes per horizon per column, or ~1.3 B per voxel at 6-sample
  layers. It also keeps N/G and contact maps per sand layer. On 64×64×256
  this totals 2.39 MB, 2.3 B per voxel, comparable to the u8 label volume.
  The streaming scratch beyond the model stays within one tile budget
  (asserted).
- For very large cubes the maps could be stored as f32 or generated lazily
  per strip. That is deferred.

## Deferred

- Onlaps, channels and salt bodies.
- Legacy Perlin thickness maps: fbm is used instead.
- Legacy sand-fraction lithology, which replaces the even/odd shale/sand
  alternation.
- Legacy `partial_voxels`.
- Python bindings do not expose `--toy-geometry` (or the other rock-physics
  options).
