//! Toy earth-model horizon geometries (the input of label generation and of
//! the rock-physics depth model).
//!
//! * [`ToyGeometry::Layered`] (default): a legacy-style layer cake draped over
//!   a dome. Following legacy `Horizons.create_depth_maps`, layers are built
//!   from the base upwards:
//!   - Each layer has a thickness `2 + Gamma(4, 1)` samples, which is legacy
//!     `stats.gamma.rvs(4.0, 2)` (mean 6, sd 2).
//!   - That thickness is modulated by a smooth lateral factor map and thinned
//!     over the dome crest (structural growth, so deeper horizons are more
//!     domed).
//!   - Layers are added until a horizon would rise above the seabed minimum
//!     depth. That depth is legacy `seabed_min_depth` in `[20, 50)` m at 4 m
//!     per sample, capped at 15 % of the cube for tiny cubes. The shallowest
//!     horizon is the seabed.
//!   - The base horizon starts `pad` (10, legacy `pad_samples`) samples below
//!     the cube, raised by the dome (a Gaussian bump with a seeded centre,
//!     radius and amplitude) and tilted by a small regional dip.
//!
//!   The result has many layers (about `nk / 6`), so the default random depth
//!   shifts (legacy layers > 20) apply in cubes of more than ~130 samples, and
//!   sand tops form spill-point closures over the dome, so oil / gas / brine
//!   selection triggers.
//! * [`ToyGeometry::Planar`]: the master geometry (two seed-dependent dipping
//!   planes and a flat base, i.e. 2 layers). This is kept for existing
//!   goldens. `--legacy-toy-depth` implies it.
//!
//! Every draw is keyed by `(seed, stream, layer, ...)`, and every map is a
//! function of the full `(ni, nj)` grid. Each worker or process that
//! rebuilds the maps therefore gets identical horizons, whatever the tiling.
//! Memory is `O(ni * nj * nh)` f64 for the maps (about `1.3 * nk` bytes per
//! column for `nh ~ nk / 6`, comparable to the u8 label volume). It does not
//! depend on the chunk shape.

use synthoseis_geo::{enforce_nonnegative_thicknesses, eval_plane, fit_plane_lsq};

use crate::rock_physics::{fbm_map, keyed_unit};

/// Horizon geometry of the toy earth model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToyGeometry {
    /// Master geometry (2 dipping planes + flat base).
    Planar,
    /// Legacy-style layer cake over a dome (default).
    #[default]
    Layered,
}

impl ToyGeometry {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToyGeometry::Planar => "planar",
            ToyGeometry::Layered => "layered",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "layered" => Ok(ToyGeometry::Layered),
            "planar" => Ok(ToyGeometry::Planar),
            other => Err(format!(
                "--toy-geometry expects layered or planar, got {other:?}"
            )),
        }
    }
}

/// Legacy thickness distribution: `loc + Gamma(shape, 1)` samples.
pub const THICKNESS_LOC: f64 = 2.0;
pub const THICKNESS_SHAPE: u32 = 4;
/// Legacy `seabed_min_depth` range in metres (`rng.integers(20, 50)`).
pub const SEABED_MIN_DEPTH_M: [u32; 2] = [20, 50];
/// Metres per sample of the geometry (legacy `digi`).
pub const GEOMETRY_DIGI_M: f64 = 4.0;
/// Legacy `pad_samples`: the base horizon starts this far below the cube.
pub const PAD_SAMPLES: f64 = 10.0;
/// Label ids are u8 and 255 means unfilled.
pub const MAX_HORIZONS: usize = 255;

/// Keyed draw stream for the geometry (independent of the rock-physics
/// streams 1-4).
const STREAM_GEOM: u64 = 0x6E0;

fn unit(seed: u64, parts: &[u64]) -> f64 {
    let mut p = Vec::with_capacity(parts.len() + 2);
    p.push(seed);
    p.push(STREAM_GEOM);
    p.extend_from_slice(parts);
    keyed_unit(&p)
}

fn uniform(lo: f64, hi: f64, seed: u64, parts: &[u64]) -> f64 {
    lo + (hi - lo) * unit(seed, parts)
}

/// Structural parameters of the layered geometry for one cube.
#[derive(Debug, Clone, PartialEq)]
pub struct LayeredParams {
    /// Seabed minimum depth in samples (build stops above it).
    pub seabed_min: f64,
    /// Dome centre (inline, crossline) in columns.
    pub dome_center: [f64; 2],
    /// Gaussian dome radius (sigma) in columns.
    pub dome_radius: f64,
    /// Dome relief of the base horizon in samples.
    pub dome_amp: f64,
    /// Regional dip in samples per column (inline, crossline).
    pub tilt: [f64; 2],
    /// Fractional crest thinning per layer; relief decays to ~20 % at the top.
    pub growth: f64,
}

impl LayeredParams {
    pub fn new(seed: u64, shape: [usize; 3]) -> Self {
        let [ni, nj, nk] = shape;
        let nkf = nk as f64;
        let [lo, hi] = SEABED_MIN_DEPTH_M;
        let metres = lo + ((hi - lo) as f64 * unit(seed, &[0])) as u32;
        let seabed_min = (metres as f64 / GEOMETRY_DIGI_M).min(0.15 * nkf);
        let dome_center = [
            uniform(0.35, 0.65, seed, &[1]) * ni as f64,
            uniform(0.35, 0.65, seed, &[2]) * nj as f64,
        ];
        let dome_radius = uniform(0.22, 0.35, seed, &[3]) * ni.min(nj).max(1) as f64;
        let dome_amp = uniform(0.10, 0.18, seed, &[4]) * nkf;
        let az = uniform(0.0, std::f64::consts::TAU, seed, &[5]);
        let tilt_mag = uniform(0.0, 0.25, seed, &[6]) * dome_amp / (0.5 * ni.max(nj).max(1) as f64);
        let column = (nkf + PAD_SAMPLES - seabed_min).max(1.0);
        Self {
            seabed_min,
            dome_center,
            dome_radius,
            dome_amp,
            tilt: [tilt_mag * az.cos(), tilt_mag * az.sin()],
            growth: (0.8 * dome_amp / column).min(0.5),
        }
    }
}

/// Base thickness (samples) of the `layer`-th layer built from the base:
/// `2 + Gamma(4, 1)` (legacy `stats.gamma.rvs(4.0, 2)`), by summing 4 unit
/// exponentials.
pub fn layer_thickness(seed: u64, layer: usize) -> f64 {
    let g: f64 = (0..THICKNESS_SHAPE as u64)
        .map(|e| -(1.0 - unit(seed, &[16, layer as u64, e])).ln())
        .sum();
    THICKNESS_LOC + g
}

/// Unrounded horizon stack of [`layered_horizon_maps`], deepest first.
fn layered_stack(seed: u64, shape: [usize; 3]) -> Vec<Vec<f64>> {
    let [ni, nj, _nk] = shape;
    let n = ni * nj;
    let p = LayeredParams::new(seed, shape);
    let mut dome = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    for i in 0..ni {
        for j in 0..nj {
            let (di, dj) = (i as f64 - p.dome_center[0], j as f64 - p.dome_center[1]);
            let d = (-(di * di + dj * dj) / (2.0 * p.dome_radius * p.dome_radius)).exp();
            dome[i * nj + j] = d;
            z[i * nj + j] =
                shape[2] as f64 + PAD_SAMPLES - p.dome_amp * d + p.tilt[0] * di + p.tilt[1] * dj;
        }
    }
    // Push the base below the cube everywhere so no column ends in unfilled
    // samples under the deepest horizon.
    let base_min = z.iter().copied().fold(f64::INFINITY, f64::min);
    let lift = (shape[2] as f64 + 1.0 - base_min).max(0.0);
    for v in &mut z {
        *v += lift;
    }
    let mut stack = vec![z.clone()];
    for layer in 0..MAX_HORIZONS - 1 {
        let t = layer_thickness(seed, layer);
        let mut f = fbm_map(ni, nj, 3, &[seed, STREAM_GEOM, 17, layer as u64]);
        let mean = f.iter().sum::<f64>() / n.max(1) as f64;
        let amp = f.iter().fold(0.0f64, |m, v| m.max((v - mean).abs()));
        let s = uniform(0.1, 0.35, seed, &[18, layer as u64]);
        for v in &mut f {
            *v = if amp > 0.0 { (*v - mean) / amp } else { 0.0 };
        }
        let next: Vec<f64> = (0..n)
            .map(|c| z[c] - t * (1.0 + s * f[c]).max(0.05) * (1.0 - p.growth * dome[c]))
            .collect();
        let top = next.iter().copied().fold(f64::INFINITY, f64::min);
        if top <= p.seabed_min {
            if stack.len() < 2 {
                // Tiny cubes: keep one layer, clamped at the seabed minimum.
                stack.push(next.iter().map(|&v| v.max(p.seabed_min)).collect());
            }
            break;
        }
        stack.push(next.clone());
        z = next;
    }
    stack
}

/// Horizon maps `(ni, nj, nh)` in samples, shallow first (horizon 0 = seabed).
pub fn layered_horizon_maps(seed: u64, shape: [usize; 3]) -> (Vec<f64>, usize) {
    let [ni, nj, _nk] = shape;
    let n = ni * nj;
    let mut stack = layered_stack(seed, shape);
    stack.reverse();
    let nh = stack.len();
    let mut maps = vec![0.0f64; n * nh];
    for (h, m) in stack.iter().enumerate() {
        for c in 0..n {
            // Whole samples: the label fill takes `[ceil(z_h), floor(z_h+1))`,
            // so fractional horizons would leave a one-sample unfilled (255)
            // gap under every horizon.
            maps[c * nh + h] = m[c].round();
        }
    }
    enforce_nonnegative_thicknesses(&mut maps, [ni, nj, nh]);
    (maps, nh)
}

/// Continuous (unrounded) counterpart of [`layered_horizon_maps`]
/// (partial-voxels spec §3.1): the same f64 stack before `.round()`, with the
/// non-negative-thickness push applied as `z_{i-1} = min(z_{i-1}, z_i)` for
/// `i = nh-1 … 2` (the same cascade as `enforce_nonnegative_thicknesses`
/// without its `deep − (deep − shallow)` rounding). Rounding is monotone, so
/// `round(continuous) == layered_horizon_maps` exactly.
pub fn layered_horizon_maps_continuous(seed: u64, shape: [usize; 3]) -> (Vec<f64>, usize) {
    let [ni, nj, _nk] = shape;
    let n = ni * nj;
    let mut stack = layered_stack(seed, shape);
    stack.reverse();
    let nh = stack.len();
    let mut maps = vec![0.0f64; n * nh];
    for (h, m) in stack.iter().enumerate() {
        for c in 0..n {
            maps[c * nh + h] = m[c];
        }
    }
    min_cascade(&mut maps, nh);
    (maps, nh)
}

/// `z_{i-1} = min(z_{i-1}, z_i)` for `i = nh-1 … 2` per column: the order
/// `enforce_nonnegative_thicknesses` imposes, exact in floating point.
pub(crate) fn min_cascade(maps: &mut [f64], nh: usize) {
    for col in maps.chunks_exact_mut(nh.max(1)) {
        for i in (2..nh).rev() {
            col[i - 1] = col[i - 1].min(col[i]);
        }
    }
}

/// Master planar geometry: two seed-dependent dipping planes + flat base.
pub fn planar_horizon_maps(seed: u64, shape: [usize; 3]) -> (Vec<f64>, usize) {
    let [ni, nj, nk] = shape;
    let seed_f = seed as f64;
    let a0 = 0.05 + (seed_f % 7.0) * 0.01;
    let b0 = 0.03 + ((seed_f / 3.0) % 5.0) * 0.01;
    let c0 = 0.5;
    let a1 = a0 * 0.5;
    let b1 = b0 * 0.5;
    let c1 = (nk as f64) * 0.55;

    let pts0 = [[0.0, 0.0, c0], [1.0, 0.0, a0 + c0], [0.0, 1.0, b0 + c0]];
    let pts1 = [[0.0, 0.0, c1], [1.0, 0.0, a1 + c1], [0.0, 1.0, b1 + c1]];
    let [fa, fb, fc] = fit_plane_lsq(&pts0);
    let [ga, gb, gc] = fit_plane_lsq(&pts1);
    let z0 = eval_plane(ni, nj, fa, fb, fc);
    let z1 = eval_plane(ni, nj, ga, gb, gc);
    let z2 = vec![(nk as f64) - 0.5; ni * nj];

    let nh = 3usize;
    let mut maps = vec![0.0f64; ni * nj * nh];
    for n in 0..(ni * nj) {
        maps[n * nh] = z0[n];
        maps[n * nh + 1] = z1[n];
        maps[n * nh + 2] = z2[n];
    }
    enforce_nonnegative_thicknesses(&mut maps, [ni, nj, nh]);
    (maps, nh)
}
