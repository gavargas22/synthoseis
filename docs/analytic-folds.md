# Analytic folds (roadmap item 5)

Spec: Notion "Spec: analytic folds (roadmap item 5)". Code:
`rust/synthoseis-core/src/folds.rs`, called once at the end of
`toy_geometry::layered_stack` (before rounding), so salt, drag, faults,
closures, partial voxels and burial depth all see the folded horizons.
Planar geometry, `--legacy-toy-depth` and the GPU path are untouched.

## Model

* Shift field `F(x, y)` (metres, positive down) = Σ cylindrical folds
  `A sin(2π s/λ + φ) E(a) + a tan p` (periclinal envelope `E = exp(−a²/2L²)`
  or `L = ∞`) + elliptical Gaussian domes / basins `σ A exp(−u²/2Ra² − v²/2Rb²)`.
  The existing dome and tilt of the layered geometry stay as fold 0.
* **Similar** (default, vertical shear): `z_h = z_h⁰ + τ_h F_h/Δz`, growth
  taper `τ_h = (burial_h/burial_base)^q` (q = 1), `τ_seabed = 0`; leaning
  axial surface shifts the profile by `d_h cot δ`. Exactly reversible.
* **Parallel** (`--fold-style parallel`, opt-in, flexural slip): each
  horizon sits at its fixed true distance above the folded reference; foot
  point by an 8-step Newton solve whose Jacobian is a central difference of
  the analytic normal (step 1e-3 bin; the spec asks for the analytic
  Hessian). It is applied as an offset of the fold field only, added to the
  unfolded structure: no axial-plane lean, horizons clamped at the seabed,
  and true thickness is preserved for the fold part only (not for fold 0,
  the existing dome and tilt). Exact where `t κ ≤ 0.8`.
* Dip limit (spec §4.3, review fix): **all** amplitudes of a cube are
  scaled by one factor so that the measured largest |slope| of the summed
  added folds on the grid, from the analytic gradient (similar: every folded
  horizon at its burial lean and taper; parallel: the reference surface), is
  at most `--fold-max-dip` samples per column (default **2**) with Δx =
  `--bin-m` (12.5 m). Plunge does not scale with amplitude, so the factor is
  a fixed 48-step bisection. The limit covers the added folds only, not fold
  0 (the existing dome and tilt). The old summed per-fold peak bound
  (`FoldSet::dip_bound`) over-counted by 12–25 % on the demo seeds and is kept only as a
  diagnostic. Also λ ≥ 8 columns, dome radii ≥ 4 columns, and parallel
  `t κ_max ≤ 0.8`, where the curvature bound uses the full column `tp[0]`
  (base to seabed): correct, conservative for shallower horizons. The run
  summary prints `all folds scaled by X` when the limit bites.
* Δz is the fixed geometry step of 4 m (`GEOMETRY_DIGI_M`, the step of the
  layered stack), not `depth_step_m`; tying the fold geometry to
  `depth_step_m` is a follow-up.
* Draws: own stream `STREAM_FOLD = 0xF01D`; every existing draw unchanged.

## Flags

| Flag | Default | Range |
|---|---|---|
| `--legacy-fold-geometry` | off | master 7e80f2af byte for byte (19 `*_7E80` test constants) |
| `--fold-style` | `similar` | `similar`, `parallel` (opt-in) |
| `--folds-cylindrical` | 1 | 0–4 |
| `--folds-domes` | 1 | 0–6 |
| `--bin-m` | 12.5 | 5–50 m |
| `--fold-growth` | 1 | 0–3 |
| `--fold-max-dip` | 2 | 0.1–2 samples/column (measured, added folds only) |

Fold flags with `--legacy-fold-geometry`, `--toy-geometry planar` or
`--legacy-toy-depth`, and out-of-range values, exit 2. All flags reach
multi-process workers (`cli_jobs::worker_args`).

## Build once

The horizon maps and the salt body are a pure function of the
configuration. `E2eConfig::with_geometry_cache` builds them once from one
layered stack, and every pipeline entry point (and the CLI) calls it, so the
value travels with the configuration (no global state; each use re-checks a
key of the geometry inputs, so a changed configuration recomputes). The
seed-4 demo run builds the layered stack 2 times instead of 54 (the second
for the `--fault-labels-through-salt` summary count, a different
configuration).

## Evidence

`cargo run --release -p synthoseis-core --example folds_demo -- OUT` prints
closure counts and brine / oil / gas closure voxels and writes inline
sections; `examples/plot_folds_demo.py OUT figures` draws
`figures/folds_*_horizons_inline.png`, `figures/folds_*_stack_inline.png`
and the map view `figures/folds_*_map_mid_horizon.png`. `folds_demo -- sweep`
runs seeds 1–20 (64×64×256, 4 faults): 83 kept closures folded vs 76 at
7e80f2af; closure voxels brine / oil / gas 36.6 / 29.8 / 33.6 % folded vs
33.4 / 39.7 / 27.0 %. `folds_demo -- slopes` prints the measured slope,
factor and amplitudes per seed.
