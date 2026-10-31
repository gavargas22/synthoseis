//! Angular unconformity of the layered toy geometry ("bevel and ponded
//! fill", spec `erosion-onlap`, roadmap item 6).
//!
//! One closed-form pass per unconformity, applied to the f64 horizon stack
//! at the end of `toy_geometry::layered_stack`, after `apply_folds` and
//! before rounding. Stack order is deepest first (index 0 = base,
//! `nh - 1` = seabed); `u` is the stack index of the unconformity horizon and
//! `s = nh - 1 - u` its shallow-first index (the production-map column of
//! its surface).
//!
//! * **Erosion surface.** `Ebar = L + a r(c)`, with `L` the `q`-quantile of
//!   the folded horizon `P = z_u` and `r` a smooth map (3-octave fBm plus a
//!   regional tilt, `|r| <= 1`). The surface actually cut is
//!   `E = min(max(P, Ebar), P + KAPPA (z_0 - P))`: horizon `u` is bevelled
//!   down to `Ebar` where it stood above it and left alone where it sits in
//!   a hollow. `e = E - P >= 0` is the eroded thickness.
//! * **Truncation.** `z'_u = E`, `z'_k = max(z_k, E)` for `k < u`.
//! * **Ponded fill.** Hollows are filled up to the datum `Ebar + F`,
//!   `F = max(E - Ebar)`: with `D = Ebar + F - P >= 0`, `S = rho max D` and
//!   `w_k = clamp(1 - mean(P - z_k) / S, 0, 1)`, the younger horizons become
//!   `z'_k = min(z_k + w_k D, E)` (the seabed is never changed). Layer
//!   thickness only grows (`w` falls with height) and the layers above the
//!   surface onlap it where `z_k + w_k D` reaches `E`.
//! * **Dip limit.** `a <- min(a, s_max / max|grad r|)` (forward differences,
//!   no iteration), so the bevel never is steeper than `s_max` samples per
//!   column.
//!
//! Everything is f64, fixed-order and a pure function of
//! `(seed, shape, config)`: the result does not depend on tiling, thread or
//! process count. Draws use [`STREAM_UNCONF`] and are not made with
//! `--legacy-unconformity` or `--unconformities 0`.

use crate::rock_physics::{fbm_map, keyed_unit};

/// Keyed draw stream of the unconformities (independent of `STREAM_GEOM`
/// 0x6E0, `STREAM_FOLD` 0xF01D, the rock-physics streams 1-4 and the fault
/// seed salt).
pub const STREAM_UNCONF: u64 = 0x0C0F;
/// Cap of the cut: `E <= P + KAPPA (z_0 - P)` keeps the base intact.
pub const KAPPA: f64 = 0.8;
/// Weight of the regional tilt in `r`.
pub const TILT_WEIGHT: f64 = 0.8;
/// Octaves of the fBm part of `r`.
pub const OCTAVES: usize = 3;
/// Minimum index distance between two unconformities.
pub const MIN_SEPARATION: usize = 6;
/// Distance of an unconformity from the seabed and from the base (indices).
pub const MARGIN: usize = 4;
/// Metres per sample of the geometry (exhumation is `e` samples of it).
pub const DZ_M: f64 = crate::toy_geometry::GEOMETRY_DIGI_M;

/// Parameters of the unconformities of one run (spec §5). `None` fields are
/// drawn per seed.
#[derive(Debug, Clone, PartialEq)]
pub struct UnconformityConfig {
    /// Number of unconformities (0-2).
    pub count: usize,
    /// Position of the (shallower) unconformity, fraction of the stack from
    /// the seabed (0.1-0.9).
    pub position: Option<f64>,
    /// Quantile `q` of the folded horizon that sets the bevel level
    /// (0.1-0.9).
    pub level: Option<f64>,
    /// Relief `a` of the erosion surface in metres (0-200).
    pub relief_m: Option<f64>,
    /// Steepest dip of the erosion surface, samples per column (0.05-2).
    pub max_dip: f64,
    /// `rho`: fill ratio, `S = rho max D` (1-6).
    pub fill_ratio: f64,
}

impl Default for UnconformityConfig {
    fn default() -> Self {
        Self {
            count: 1,
            position: None,
            level: None,
            relief_m: None,
            max_dip: 1.0,
            fill_ratio: 2.0,
        }
    }
}

impl UnconformityConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.count > 2 {
            return Err(format!(
                "--unconformities must be 0, 1 or 2, got {}",
                self.count
            ));
        }
        let chk = |name: &str, v: f64, lo: f64, hi: f64| -> Result<(), String> {
            if v.is_finite() && (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("--{name} must be in [{lo}, {hi}], got {v}"))
            }
        };
        if let Some(v) = self.position {
            chk("unconformity-position", v, 0.1, 0.9)?;
        }
        if let Some(v) = self.level {
            chk("unconformity-level", v, 0.1, 0.9)?;
        }
        if let Some(v) = self.relief_m {
            chk("unconformity-relief-m", v, 0.0, 200.0)?;
        }
        chk("unconformity-max-dip", self.max_dip, 0.05, 2.0)?;
        chk("unconformity-fill-ratio", self.fill_ratio, 1.0, 6.0)?;
        Ok(())
    }
}

/// Outcome of one unconformity (kept for the labels, exhumation, summary).
#[derive(Debug, Clone, PartialEq)]
pub struct UnconformityRecord {
    /// 1 = shallowest. This is the value written to `unconformity_labels`.
    pub number: u8,
    /// Shallow-first horizon index of the surface (production-map column).
    pub s: usize,
    /// `false` when the draw gave nothing to erode ("skipped").
    pub applied: bool,
    /// Eroded thickness `e = E - P` in samples, `(ni, nj)` raster
    /// (empty when skipped).
    pub eroded: Vec<f32>,
    /// Bevel level `L` (samples), relief `a` after the dip limit (samples),
    /// quantile `q`, deepest hollow `F`, largest `D`, fill height `S`.
    pub level: f64,
    pub quantile: f64,
    pub relief: f64,
    pub relief_drawn: f64,
    pub hollow: f64,
    pub max_fill: f64,
    pub fill_height: f64,
    pub max_eroded: f64,
}

impl UnconformityRecord {
    /// One summary line (horizon, level, relief, eroded and filled extent).
    pub fn summary(&self) -> String {
        if !self.applied {
            return format!(
                "unconformity {} skipped (nothing to erode at horizon {})",
                self.number, self.s
            );
        }
        format!(
            "unconformity {}: horizon {} from the seabed, level {:.1} samples (q={:.2}), relief {:.1} samples (drawn {:.1}), \
             max eroded {:.1} samples, hollow {:.1}, max fill {:.1}, ponded fill height {:.1}",
            self.number,
            self.s,
            self.level,
            self.quantile,
            self.relief,
            self.relief_drawn,
            self.max_eroded,
            self.hollow,
            self.max_fill,
            self.fill_height,
        )
    }
}

/// Does the cube have room for `count` unconformities (spec D10)?
pub fn fits(nh: usize, count: usize) -> bool {
    match count {
        0 => false,
        1 => nh >= 10,
        _ => nh >= 16,
    }
}

fn draw(seed: u64, m: u64, kind: u64) -> f64 {
    keyed_unit(&[seed, STREAM_UNCONF, m, kind])
}

/// Shallow-first surface indices `s`, ascending (index 0 = shallowest =
/// label 1). Empty when the cube is too shallow.
pub fn surface_indices(seed: u64, nh: usize, cfg: &UnconformityConfig) -> Vec<usize> {
    if !fits(nh, cfg.count) {
        return Vec::new();
    }
    let span = (nh - 1) as f64;
    let (lo, hi) = (MARGIN, nh - 1 - MARGIN);
    if cfg.count == 1 {
        let f = cfg.position.unwrap_or_else(|| 0.3 + 0.4 * draw(seed, 0, 0));
        return vec![((f * span).round() as usize).clamp(lo, hi)];
    }
    let f1 = cfg
        .position
        .unwrap_or_else(|| 0.2 + 0.25 * draw(seed, 0, 0));
    let f2 = 0.55 + 0.25 * draw(seed, 1, 0);
    let s1 = ((f1 * span).round() as usize).clamp(lo, hi - MIN_SEPARATION);
    let s2 = ((f2 * span).round() as usize).clamp(s1 + MIN_SEPARATION, hi);
    vec![s1, s2]
}

/// Steepest forward-difference slope (samples per column, hypot of the two
/// directions) of an `(ni, nj)` raster.
pub fn slope_max(map: &[f64], ni: usize, nj: usize) -> f64 {
    let mut g = 0.0f64;
    for i in 0..ni {
        for j in 0..nj {
            let v = map[i * nj + j];
            let dx = if i + 1 < ni {
                map[(i + 1) * nj + j] - v
            } else {
                0.0
            };
            let dy = if j + 1 < nj {
                map[i * nj + j + 1] - v
            } else {
                0.0
            };
            g = g.max(dx.hypot(dy));
        }
    }
    g
}

/// The smooth map `r` of unconformity `m`: centred, `|r| <= 1`.
pub fn relief_map(seed: u64, m: u64, ni: usize, nj: usize) -> Vec<f64> {
    let n = ni * nj;
    let center_scale = |v: &mut [f64]| {
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        let mut mx = 0.0f64;
        for x in v.iter_mut() {
            *x -= mean;
            mx = mx.max(x.abs());
        }
        if mx > 0.0 {
            for x in v.iter_mut() {
                *x /= mx;
            }
        }
    };
    let mut noise = fbm_map(ni, nj, OCTAVES, &[seed, STREAM_UNCONF, m, 4]);
    center_scale(&mut noise);
    let az = std::f64::consts::TAU * draw(seed, m, 3);
    let (ca, sa) = (az.cos(), az.sin());
    let norm = ca.abs() + sa.abs();
    let (hi, hj) = (
        0.5 * (ni.max(1) as f64 - 1.0),
        0.5 * (nj.max(1) as f64 - 1.0),
    );
    let mut r = vec![0.0f64; n];
    for i in 0..ni {
        let x = if hi > 0.0 { (i as f64 - hi) / hi } else { 0.0 };
        for j in 0..nj {
            let y = if hj > 0.0 { (j as f64 - hj) / hj } else { 0.0 };
            let plane = (ca * x + sa * y) / norm;
            r[i * nj + j] = noise[i * nj + j] + TILT_WEIGHT * plane;
        }
    }
    center_scale(&mut r);
    r
}

/// `q`-quantile (lower, index `floor(q (n - 1))`) by selection.
fn quantile(p: &[f64], q: f64) -> f64 {
    let mut v = p.to_vec();
    let k = (((v.len() - 1) as f64) * q).floor() as usize;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

/// Apply unconformity `number` (1 = shallowest) at shallow-first index `s`
/// to `stack` (deepest first) in place.
pub fn apply_one(
    stack: &mut [Vec<f64>],
    seed: u64,
    shape: [usize; 3],
    cfg: &UnconformityConfig,
    number: u8,
    s: usize,
) -> UnconformityRecord {
    let [ni, nj, nk] = shape;
    let nh = stack.len();
    let m = (number - 1) as u64;
    let u = nh - 1 - s;
    let n = ni * nj;
    let q = cfg.level.unwrap_or_else(|| 0.35 + 0.30 * draw(seed, m, 1));
    let drawn = cfg.relief_m.map_or_else(
        || (0.02 + 0.03 * draw(seed, m, 2)) * nk as f64,
        |r| r / DZ_M,
    );
    let r = relief_map(seed, m, ni, nj);
    let gmax = slope_max(&r, ni, nj);
    let a = if gmax > 0.0 {
        drawn.min(cfg.max_dip / gmax)
    } else {
        drawn
    };
    let p = stack[u].clone();
    let level = quantile(&p, q);
    let mut e_bar = vec![0.0f64; n];
    let mut cut = vec![0.0f64; n];
    for c in 0..n {
        e_bar[c] = level + a * r[c];
        let capped = p[c] + KAPPA * (stack[0][c] - p[c]);
        cut[c] = e_bar[c].max(p[c]).min(capped);
    }
    let mut hollow = f64::NEG_INFINITY;
    let mut max_eroded = 0.0f64;
    for c in 0..n {
        hollow = hollow.max(cut[c] - e_bar[c]);
        max_eroded = max_eroded.max(cut[c] - p[c]);
    }
    let mut max_fill = f64::NEG_INFINITY;
    for c in 0..n {
        max_fill = max_fill.max(e_bar[c] + hollow - p[c]);
    }
    let skipped = UnconformityRecord {
        number,
        s,
        applied: false,
        eroded: Vec::new(),
        level,
        quantile: q,
        relief: a,
        relief_drawn: drawn,
        hollow,
        max_fill,
        fill_height: 0.0,
        max_eroded,
    };
    if !(max_fill > 0.0) || !(max_eroded > 0.0) {
        return skipped;
    }
    let fill_height = cfg.fill_ratio * max_fill;
    // Truncation: horizons below (older) are cut to E.
    for k in 0..u {
        for c in 0..n {
            stack[k][c] = stack[k][c].max(cut[c]);
        }
    }
    // Ponded fill of the younger horizons (the seabed stays).
    for k in u + 1..nh - 1 {
        let mut sum = 0.0f64;
        for c in 0..n {
            sum += p[c] - stack[k][c];
        }
        let mean = sum / n as f64;
        let w = (1.0 - mean / fill_height).clamp(0.0, 1.0);
        for c in 0..n {
            let d = e_bar[c] + hollow - p[c];
            stack[k][c] = (stack[k][c] + w * d).min(cut[c]);
        }
    }
    stack[u].copy_from_slice(&cut);
    UnconformityRecord {
        applied: true,
        eroded: (0..n).map(|c| (cut[c] - p[c]) as f32).collect(),
        fill_height,
        ..skipped
    }
}

/// Apply every unconformity of `cfg`, deepest first, each with its own
/// draws. Returns one record per unconformity (shallowest first); empty when
/// there is none or the cube is too shallow.
pub fn apply_unconformities(
    stack: &mut [Vec<f64>],
    seed: u64,
    shape: [usize; 3],
    cfg: &UnconformityConfig,
) -> Vec<UnconformityRecord> {
    let idx = surface_indices(seed, stack.len(), cfg);
    let mut recs: Vec<UnconformityRecord> = Vec::with_capacity(idx.len());
    for (k, &s) in idx.iter().enumerate().rev() {
        recs.push(apply_one(stack, seed, shape, cfg, (k + 1) as u8, s));
    }
    recs.reverse();
    recs
}

/// Summed eroded thickness (samples) over the unconformities at or above
/// interval `h` (burial added to interval `h`, spec §3.6): those with
/// `s <= h`.
#[inline]
pub fn exhumation_at(records: &[UnconformityRecord], c: usize, h: usize) -> f32 {
    let mut t = 0.0f32;
    for r in records {
        if r.applied && h >= r.s {
            t += r.eroded[c];
        }
    }
    t
}

/// `Some(records)` that were applied (non-skipped), shallowest first.
pub fn applied(records: &[UnconformityRecord]) -> impl Iterator<Item = &UnconformityRecord> {
    records.iter().filter(|r| r.applied)
}

/// Depth-domain `unconformity_labels` of the tile `[i0, i1) x [j0, j1)`
/// (`(ti, tj, nk)`, samples fastest): the first voxel of the layer under
/// each surface, i.e. the production map `maps[c, s]`. With faults it is the
/// first output voxel at or below the surface in the faulted labels (the
/// crossing: one per column for N = 1, and mark 2 is missing where the
/// second surface is below the last sample), and it is cleared in `salt`.
#[allow(clippy::too_many_arguments)]
pub fn label_tile(
    records: &[UnconformityRecord],
    maps: &[f64],
    nh: usize,
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    fault_tile: Option<&synthoseis_geo::faults::FaultTile>,
    salt: Option<&crate::salt::SaltBody>,
) -> Vec<u8> {
    let [ni, nj, nk] = shape;
    let tj = j1 - j0;
    let mut out = vec![0u8; (i1 - i0) * tj * nk];
    for i in i0..i1 {
        for j in j0..j1 {
            let c = i * nj + j;
            let col = &mut out[((i - i0) * tj + (j - j0)) * nk..][..nk];
            if let Some(t) = fault_tile.filter(|t| t.src_ij.is_some()) {
                // Heave: the crossing rule at each voxel's source column
                // (the surface mark of the column the voxel came from).
                for r in records.iter().filter(|r| r.applied) {
                    let hit = (0..nk).position(|k| {
                        let [si, sj, sk] = t.nearest_source(i, j, k, ni, nj);
                        let z = maps[(si * nj + sj) * nh + r.s];
                        z >= 0.0 && z < nk as f64 && sk >= z as usize
                    });
                    if let Some(k) = hit {
                        col[k] = r.number;
                    }
                }
            } else {
                for r in records.iter().filter(|r| r.applied) {
                    let z = maps[c * nh + r.s];
                    if z < 0.0 || z >= nk as f64 {
                        continue;
                    }
                    let kmark = z as usize;
                    match fault_tile {
                        None => col[kmark] = r.number,
                        Some(t) => {
                            // Crossing rule: the first output sample whose
                            // nearest source sample (the one the layer labels
                            // take) is at or below the surface voxel. Exactly
                            // one per column, wherever a fault gap or overlap
                            // moves the surface voxel (a nearest-voxel match
                            // can miss it or hit it twice).
                            let look = &t.lookup[t.col_offset(i, j)..][..nk];
                            let at_or_below = |&x: &f32| {
                                let src =
                                    ((x as f64) + 0.5).floor().clamp(0.0, (nk - 1) as f64) as usize;
                                src >= kmark
                            };
                            if let Some(k) = look.iter().position(at_or_below) {
                                col[k] = r.number;
                            }
                        }
                    }
                }
            }
            if let Some(sb) = salt {
                let (a, b) = sb.runs[c];
                let (a, b) = ((a as usize).min(nk), (b as usize).min(nk));
                if a < b {
                    col[a..b].fill(0);
                }
            }
        }
    }
    out
}
