//! Per-layer lithology (sand / shale) of the rock-physics model.
//!
//! * [`ToyLithology::Markov`] (default): port of legacy
//!   `Horizons.create_facies_array` → `Facies.sand_shale_facies_markov`:
//!   - The model sand fraction `f` is legacy `sand_layer_pct =
//!     rng.uniform(sand_layer_fraction.min, .max)`, which is U(0.05, 0.25) in
//!     `config/example.json`. It is drawn once per model, keyed by the seed.
//!   - The sand unit thickness is `T` layers (legacy `sand_layer_thickness`,
//!     2).
//!   - A two-state Markov chain (legacy `MarkovChainFacies`) uses the
//!     transition matrix `[[1 - a, a], [b, 1 - b]]` over (shale, sand), with
//!     `a = f / (T (1 - f))` and `b = 1 / T`. Its stationary sand fraction is
//!     `f` and the mean sand run is `T` layers.
//!   - The initial state is legacy `rng.choice(2)`. Each layer, from the
//!     shallowest down, takes one transition, which is legacy `rng.choice(p=row)`:
//!     `cdf = cumsum(row) / sum(row)` and the state is the number of cdf
//!     entries `<= u`.
//!   - Legacy facies index `i` (0 = water) is legacy layer `i`, and Rust
//!     interval `h` is legacy layer `h + 1` (as for the depth shifts). So
//!     interval `h` takes chain state `h` (0-based).
//!   - Given the same initial state and uniforms, the chain is bit-identical
//!     to legacy (`tests/fixtures/lithology_reference.json`). The uniforms
//!     themselves are keyed hashes of `(seed, layer)` rather than numpy
//!     PCG64 draws.
//! * [`ToyLithology::Alternating`]: the previous rule, where even intervals
//!   are shale and odd ones sand. CLI `--toy-lithology alternating`. The
//!   planar geometry and `--legacy-toy-depth` always use it, which keeps
//!   their goldens.
//!
//! Legacy onlap and fan overrides (the layer below an onlap surface is
//! shale; fans are sand wrapped in shale) are not ported, because the toy
//! geometry has no onlaps or fans.

use crate::rock_physics::keyed_unit;

/// Lithology rule of the toy model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToyLithology {
    /// Even intervals shale, odd intervals sand (previous default).
    Alternating,
    /// Legacy sand-fraction Markov chain (default).
    #[default]
    Markov,
}

impl ToyLithology {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToyLithology::Alternating => "alternating",
            ToyLithology::Markov => "markov",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "markov" => Ok(ToyLithology::Markov),
            "alternating" => Ok(ToyLithology::Alternating),
            other => Err(format!(
                "--toy-lithology expects markov or alternating, got {other:?}"
            )),
        }
    }
}

/// Legacy `sand_layer_fraction` range (`config/example.json`).
pub const SAND_LAYER_FRACTION: [f64; 2] = [0.05, 0.25];
/// Legacy `sand_layer_thickness` (layers).
pub const SAND_LAYER_THICKNESS: f64 = 2.0;

/// Keyed draw stream of the lithology (independent of the rock-physics
/// streams 1-4 and of the geometry stream).
const STREAM_LITH: u64 = 0x117;

/// Legacy `MarkovChainFacies._transition_matrix`: rows (from shale, from
/// sand), columns (to shale, to sand).
pub fn transition_matrix(sand_fraction: f64, sand_thickness: f64) -> [[f64; 2]; 2] {
    let beta = 1.0 / sand_thickness;
    let alpha = sand_fraction / (sand_thickness * (1.0 - sand_fraction));
    [[1.0 - alpha, alpha], [beta, 1.0 - beta]]
}

/// Checks `sand_fraction` and `sand_thickness` give a valid transition
/// matrix (`0 < f < 1`, `T >= 1`, `a = f / (T (1 - f)) <= 1`).
pub fn validate(sand_fraction: f64, sand_thickness: f64) -> Result<(), String> {
    if !(sand_thickness >= 1.0 && sand_thickness.is_finite()) {
        return Err(format!(
            "--sand-layer-thickness must be >= 1 (layers), got {sand_thickness}"
        ));
    }
    if !(sand_fraction > 0.0 && sand_fraction < 1.0) {
        return Err(format!(
            "--sand-layer-fraction must be in (0, 1), got {sand_fraction}"
        ));
    }
    let a = transition_matrix(sand_fraction, sand_thickness)[0][1];
    if a > 1.0 {
        return Err(format!(
            "--sand-layer-fraction {sand_fraction} is unreachable with --sand-layer-thickness {sand_thickness} \
             (needs f <= T / (T + 1) = {:.3})",
            sand_thickness / (sand_thickness + 1.0)
        ));
    }
    Ok(())
}

/// numpy `Generator.choice([0, 1], p=row)` for a given unit draw `u`:
/// `cdf = cumsum(p); cdf /= cdf[-1]; searchsorted(cdf, u, side="right")`.
#[inline]
// `total / total` is numpy's `cdf /= cdf[-1]` on the last entry (1.0, or NaN
// for a zero / non-finite row), kept as written for parity.
#[allow(clippy::eq_op)]
pub fn legacy_choice(row: [f64; 2], u: f64) -> u8 {
    let total = row[0] + row[1];
    let c0 = row[0] / total;
    let c1 = total / total;
    (c0 <= u) as u8 + (c1 <= u) as u8
}

/// Legacy `MarkovChainFacies.generate_states(initial, num)` given the unit
/// draws `u` (one per state).
pub fn markov_states(initial: u8, u: &[f64], sand_fraction: f64, sand_thickness: f64) -> Vec<u8> {
    let t = transition_matrix(sand_fraction, sand_thickness);
    let mut s = initial.min(1);
    u.iter()
        .map(|&x| {
            s = legacy_choice(t[s as usize], x).min(1);
            s
        })
        .collect()
}

/// Model sand fraction: fixed, or the legacy per-model draw
/// U(0.05, 0.25) keyed by the seed.
pub fn sand_fraction(seed: u64, fixed: Option<f64>) -> f64 {
    fixed.unwrap_or_else(|| {
        let [lo, hi] = SAND_LAYER_FRACTION;
        lo + (hi - lo) * keyed_unit(&[seed, STREAM_LITH, 1])
    })
}

/// Initial chain state (legacy `rng.choice(2, 1)[0]`).
pub fn initial_state(seed: u64) -> u8 {
    (keyed_unit(&[seed, STREAM_LITH, 0]) >= 0.5) as u8
}

/// Unit draw of the transition into interval `h`.
pub fn transition_unit(seed: u64, h: usize) -> f64 {
    keyed_unit(&[seed, STREAM_LITH, 2, h as u64])
}

/// Sand flags of intervals `0..n`. The Markov chain is a prefix-stable
/// function of `(seed, fraction, thickness)`: interval `h` does not depend
/// on `n`.
pub fn interval_sand(
    lithology: ToyLithology,
    seed: u64,
    n: usize,
    fixed_fraction: Option<f64>,
    sand_thickness: f64,
) -> Vec<bool> {
    match lithology {
        ToyLithology::Alternating => (0..n).map(|h| h % 2 == 1).collect(),
        ToyLithology::Markov => {
            let f = sand_fraction(seed, fixed_fraction);
            let u: Vec<f64> = (0..n).map(|h| transition_unit(seed, h)).collect();
            markov_states(initial_state(seed), &u, f, sand_thickness)
                .into_iter()
                .map(|s| s == 1)
                .collect()
        }
    }
}

/// Sand units that carry closures, as half-open interval ranges `[top, end)`.
/// This ports legacy `Closures.find_top_lith_horizons` and the unit loop of
/// `create_closure_labels_from_depth_maps`:
/// - Legacy keeps the horizons where the facies changes from the layer
///   above (`top_lith_indices`). Legacy layer 1 always qualifies, because
///   `facies[0]` is water. The scan is `enumerate(facies[:-1])`, so the
///   last facies entry never starts a unit.
/// - Each run between two kept horizons is one unit. Closures are computed
///   on the top of each sand unit, down to the unit base. The loop is
///   `range(len(top_lith) - 1)`, so the deepest unit, which runs into the
///   model base, is skipped.
///
/// `sand` holds the facies of intervals `0..n`, where interval `h` is legacy
/// layer `h + 1`. The last entry is legacy `facies[max_layers]`, the layer
/// below the deepest horizon. Onlap tops (`onlap_list - 1`) do not occur in
/// the toy geometry. Bit-exact against the legacy functions in
/// `tests/fixtures/closure_units_reference.json`.
pub fn closure_units(sand: &[bool]) -> Vec<(usize, usize)> {
    // Intervals 0..m can start a unit; the last entry only extends the
    // deepest unit.
    let m = sand.len().saturating_sub(1);
    let mut units = Vec::new();
    let mut h = 0;
    while h < m {
        let start = h;
        while h < m && sand[h] == sand[start] {
            h += 1;
        }
        // `h == m`: the deepest unit (legacy skips it).
        if sand[start] && h < m {
            units.push((start, h));
        }
    }
    units
}
