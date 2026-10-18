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
* **Parallel** (`--fold-style parallel`, flexural slip): each horizon sits
  at its fixed true distance above the folded reference; foot point by an
  8-step Newton solve. True thickness preserved where `t κ ≤ 0.8`.
* Limits (spec §4.3), enforced when drawing by scaling **all** amplitudes
  of a cube by one factor (so the summed field meets the limit): dip
  `≤ --fold-max-dip` samples per column (default 1) with Δx = `--bin-m`
  (12.5 m), λ ≥ 8 columns, parallel `t κ_max ≤ 0.8`. The run summary
  reports how many folds were scaled.
* Draws: own stream `STREAM_FOLD = 0xF01D`; every existing draw unchanged.

## Flags

| Flag | Default | Range |
|---|---|---|
| `--legacy-fold-geometry` | off | master 7e80f2af byte for byte |
| `--fold-style` | `similar` | `similar`, `parallel` |
| `--folds-cylindrical` | 1 | 0–4 |
| `--folds-domes` | 1 | 0–6 |
| `--bin-m` | 12.5 | 5–50 m |
| `--fold-growth` | 1 | 0–3 |
| `--fold-max-dip` | 1 | 0.1–2 samples/column |

Fold flags with `--legacy-fold-geometry`, `--toy-geometry planar` or
`--legacy-toy-depth`, and out-of-range values, exit 2. All flags reach
multi-process workers (`cli_jobs::worker_args`).

## Evidence

`cargo run --release -p synthoseis-core --example folds_demo -- OUT` prints
closure counts and brine / oil / gas closure voxels and writes inline
sections; `examples/plot_folds_demo.py OUT figures` draws
`figures/folds_*_horizons_inline.png` and `figures/folds_*_stack_inline.png`.
