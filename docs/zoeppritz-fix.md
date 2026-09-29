# Zoeppritz PP: textbook form by default (`det` → `d`)

## The bug

Legacy `datagenerator/zoeppritz_kernel.py` (and `tests/_zoeppritz_reference.py`,
which the Rust port was checked against) evaluates the explicit Aki &
Richards (1980) PP coefficient:

```text
Rpp = [ F (b cosθ1/vp1 − c cosθ2/vp2) − H p² (a + d cosθ1/vp1 · cosφ2/vs2) ] / D
```

but it writes `det` (the denominator `D = E F + G H p²`) where the textbook
(Aki & Richards eq. 5.40, bruges `reflection.zoeppritz_rpp`, CREWES) has
`d = 2 (ρ2 vs2² − ρ1 vs1²)`. The two coincide at normal incidence (`p = 0`) and
differ at every other angle. In SI-like units `D` is ~1e-6 and `d` is ~1e7, so
the typo in effect **drops** the term `H p² d cosθ1 cosφ2 / (vp1 vs2)`. That
term is second order in the elastic contrast, so the legacy result still
agrees with Aki-Richards to first order, but it is wrong for strong contrasts:
the seabed, gas sands and post-critical angles.

## What changed

| | default | `--legacy-zoeppritz` | `--legacy-toy-depth` |
|---|---|---|---|
| depth / rock physics | corrected model | corrected model | master 10f4dcd toy |
| Zoeppritz | textbook (`d`) | legacy (`det`) | legacy (`det`, implied) |
| reproduces | — | master 33a3a93 bit for bit | master 10f4dcd bit for bit |

- `synthoseis_seismic::ZoeppritzForm { Exact (default), Legacy }`,
  `zoeppritz_pp_form`, `zoeppritz_pp_complex`, `zoeppritz_pp_exact`,
  `compute_rfc_volumes_form`. The historical `zoeppritz_pp` and
  `compute_rfc_volumes` keep the legacy form and stay bit-identical; they
  are pinned by the Python goldens in `seismic_kernels.json`.
- `RockPhysicsConfig::legacy_zoeppritz` (CLI `--legacy-zoeppritz`). The
  multi-process workers get the flag too.
- `RockPhysicsConfig::zoeppritz_form()` returns `Legacy` when
  `legacy_zoeppritz` **or** `legacy_toy_depth` is set.
  - **Choice:** `--legacy-toy-depth` implies the legacy Zoeppritz. This is
    required: master 10f4dcd used the typo, so the switch's bit-for-bit
    guarantee needs it.
  - Passing both flags is accepted and is redundant.
- Every fuse path (classic, chunked, streaming, overlap, strip, multi-process,
  geometry-once) takes the form from the elastic model:
  - The CPU adapter `fuse_props_tile_cpu` has a `form` argument.
  - WGSL has a `legacy_det` uniform: 0 selects `d`, 1 selects `det`. The
    label-trend path (toy model) always sends 1.

## Validation against independent references

The tests are in `rust/synthoseis-seismic/tests/zoeppritz_reference.rs`:

| reference | cases | max \|textbook − ref\| | max \|legacy − ref\| |
|---|---|---|---|
| native 4×4 complex solve of the full Zoeppritz system (Rust, no explicit formula), 2000 random interfaces × 0–60° | 122,000 (11,544 post-critical) | 1.4e-15 (complex) | 0.384 |
| numpy 4×4 solve (`tests/fixtures/generate_zoeppritz_reference.py`) | 585 (45 interfaces × 0–60°) | 2.9e-8 (f32 output) | 0.183 |
| bruges 0.5.4 `zoeppritz_rpp` | 585 | 2.9e-8 | — |

- **Legacy expression:** `ZoeppritzForm::Legacy` matches the legacy Python
  expression to 2.9e-8 (f32 output). It is bit-identical to the previous
  Rust kernel on random interfaces.
- **Aki-Richards at small contrasts** (4 contrast directions, 5–30°):
  - `|textbook − AR|` = 0.198 ε² … 0.220 ε² for ε = 8 % … 0.5 %. Halving ε
    divides the error by 3.8–4.0, i.e. second order.
  - The typo itself, `|textbook − legacy|`, is 0.177 ε², also second order.
    The linearisation therefore cannot distinguish the two forms; the full
    matrix solution does.

## Size of the change on the demo cube

The demo cube is 64×64×128, seed 7, 4 faults, default rock physics. The
change is textbook − legacy:

| angle | raw rfc max \|Δ\| | ref max \|r\| | raw rfc rel. RMS | Ricker stack max \|Δ\| | stack rel. RMS |
|---|---|---|---|---|---|
| 0° | 0 (bit-identical) | 0.458 | 0 | 0 | 0 |
| 15° | 6.64e-3 | 0.466 | 1.29 % | 6.64e-3 | 1.29 % |
| 30° | 2.46e-2 | 0.492 | 4.45 % | 2.46e-2 | 4.44 % |
| 45° | 4.85e-2 | 0.582 | 7.33 % | 4.85e-2 | 7.30 % |

Only the interface voxels change, which is 1.56 % of the cube on the toy
geometry. The largest change is at the seabed (water / shale). At 30° the
seabed reflection goes from 0.492 (legacy) to 0.502 (textbook), and the
shale / sand reflection from 0.0566 to 0.0659 (+17 %).

## GPU

The WGSL kernel applies the same fix, selected by `legacy_det`. Measured on
llvmpipe (Vulkan/CPU), GPU vs CPU max |Δ| on the demo cube:

| angle | textbook, raw rfc | textbook, stack | legacy, raw rfc |
|---|---|---|---|
| 0° | 1.2e-7 | 1.2e-7 | 1.2e-7 |
| 15° | 1.9e-7 | 1.9e-7 | 1.9e-7 |
| 30° | 1.8e-7 | 1.8e-7 | 1.6e-7 |
| 45° | 2.5e-7 | 2.5e-7 | 2.5e-7 |

The `parity_props_tile` test (both forms, 0° / 30°, with and without the
wavelet) stays at or below 6.0e-7. The GPU remains outside the bit-exact
matrix.

## Invariance and goldens

- `default_model_invariant_to_tiling_workers_and_paths` now covers four
  configurations:
  - rich default, which uses the textbook form;
  - plain default;
  - `--legacy-toy-depth`;
  - rich + `--legacy-zoeppritz`.

  Each is checked across:
  - chunks 1×1, 5×7, full and 3×20×16;
  - the classic path;
  - streaming and overlap at 2 chunk shapes;
  - strip 2/3/4;
  - multi-process 1/2/3;
  - geometry-once at 0/15/30°.
- The master 33a3a93 goldens under `--legacy-zoeppritz`
  (`synthoseis-core/tests/zoeppritz_fix.rs`):
  - demo labels, 15° stack, raw 15° reflectivity and 30° stack;
  - the rich Backus + shifts + closures + filters + noise case;
  - `tiny(42)`.
- The CLI store hashes (`synthoseis/tests/rock_physics_cli.rs`) cover plain,
  rich, multi-process and multi-process with model flags.
- The master 10f4dcd goldens stay pinned under `--legacy-toy-depth`.

## Figure

`examples/zoeppritz_demo.rs` and `examples/plot_zoeppritz_demo.py` produce
`zoeppritz_fix.png`, which shows:

- AVO curves for textbook, legacy and the matrix solve;
- the error vs the matrix solution over the reference fixture;
- the size of the change per angle and the GPU gap;
- the 30° inline sections and their difference.
