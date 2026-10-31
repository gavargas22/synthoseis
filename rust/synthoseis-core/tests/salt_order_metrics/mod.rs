//! Salt / fault order metrics (spec-salt-fault-order §4), shared by
//! `tests/salt_fault_order.rs` and `examples/salt_order_evidence.rs`.
#![allow(dead_code)]

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::pipeline_stream::generate_labels_warp;
use synthoseis_core::salt::SaltBody;

/// Band half-width (columns) of the near-salt metrics.
pub const BAND: usize = 12;

/// Which order a run uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Post-fault drag (the default).
    PostFault,
    /// `--legacy-salt-fault-order` (master 481805cc).
    Master,
    /// `--faults 0` (drag, no faults).
    FaultsOff,
}

pub fn cfg(seed: u64, shape: [usize; 3], faults: usize, mode: Mode) -> E2eConfig {
    let mut c = E2eConfig::tiny(seed);
    c.inline_count = shape[0];
    c.crossline_count = shape[1];
    c.samples = shape[2];
    c.faults = FaultConfig::with_count(if mode == Mode::FaultsOff { 0 } else { faults });
    c.rock_physics.legacy_salt_fault_order = mode == Mode::Master;
    c.with_geometry_cache()
}

/// One run's labels, warp index, fault sources and fault ids.
pub struct Run {
    pub shape: [usize; 3],
    pub labels: Vec<u8>,
    /// Source cell of the post-fault drag warp (empty: identity).
    pub warp: Vec<u32>,
    /// Gather source `(i, j, k)` per voxel (`None` without faults).
    pub src: Option<(Vec<(f32, f32)>, Vec<f32>)>,
    /// 1-based fault id per voxel (0 = none; empty without faults).
    pub fault_id: Vec<u8>,
}

pub fn run(c: &E2eConfig) -> Run {
    let shape = c.shape();
    let mut warp = Vec::new();
    let (labels, _) = generate_labels_warp(c, Some(&mut warp));
    let (src, fault_id) = match synthoseis_core::fault_model(c).filter(|m| !m.is_empty()) {
        None => (None, Vec::new()),
        Some(m) => {
            let t = m.compute_tile(0, shape[0], 0, shape[1]);
            let ij = t.src_ij.clone().unwrap_or_else(|| {
                (0..shape[0] * shape[1])
                    .flat_map(|q| {
                        let (i, j) = ((q / shape[1]) as f32, (q % shape[1]) as f32);
                        std::iter::repeat((i, j)).take(shape[2])
                    })
                    .collect()
            });
            (Some((ij, t.lookup.clone())), t.segment_id.clone())
        }
    };
    Run {
        shape,
        labels,
        warp,
        src,
        fault_id,
    }
}

/// Undragged, unfaulted labels of `seed` (no salt: the base maps).
pub fn undragged(seed: u64, shape: [usize; 3]) -> Vec<u8> {
    let mut c = cfg(seed, shape, 0, Mode::FaultsOff);
    c.rock_physics.salt = false;
    let c = c.with_geometry_cache();
    synthoseis_core::generate_labels(&c).0
}

pub fn salt(c: &E2eConfig) -> Option<SaltBody> {
    synthoseis_core::salt::salt_body(c)
}

/// Footprint columns (salt in the first `nk` samples).
pub fn footprint(s: &SaltBody, shape: [usize; 3]) -> Vec<bool> {
    s.runs
        .iter()
        .take(shape[0] * shape[1])
        .map(|&(a, b)| b > a && (a as usize) < shape[2])
        .collect()
}

/// Chebyshev column distance to the footprint (capped at `cap + 1`).
pub fn distance(fp: &[bool], ni: usize, nj: usize, cap: usize) -> Vec<usize> {
    let mut d = vec![cap + 1; ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            if !fp[i * nj + j] {
                continue;
            }
            for a in i.saturating_sub(cap)..(i + cap + 1).min(ni) {
                for b in j.saturating_sub(cap)..(j + cap + 1).min(nj) {
                    let r = a.abs_diff(i).max(b.abs_diff(j));
                    let q = a * nj + b;
                    d[q] = d[q].min(r);
                }
            }
        }
    }
    d
}

fn near(x: f32, n: usize) -> usize {
    (x.round().max(0.0) as usize).min(n - 1)
}

/// Exposed hidden-lifted strata (§4 row 1): non-salt output voxels whose
/// source (the warp, then the fault gather; nearest voxel) lies inside the
/// salt body and whose label differs from the undragged label there.
pub fn exposed(r: &Run, s: &SaltBody, und: &[u8]) -> usize {
    let [ni, nj, nk] = r.shape;
    let Some((ij, lk)) = &r.src else { return 0 };
    let mut n = 0;
    for q in 0..ni * nj {
        for k in 0..nk {
            let v = q * nk + k;
            if s.contains(q, k) {
                continue;
            }
            let kw = if r.warp.is_empty() {
                k
            } else {
                r.warp[v] as usize
            };
            let w = q * nk + kw;
            let sq = near(ij[w].0, ni) * nj + near(ij[w].1, nj);
            let sk = near(lk[w], nk);
            if !s.contains(sq, sk) {
                continue;
            }
            n += (r.labels[v] != und[sq * nk + sk]) as usize;
        }
    }
    n
}

/// Flank stair-step index (§4 row 2): fraction of the non-salt voxels
/// within `BAND` columns of the footprint at which a layer boundary has
/// no matching boundary within 2 samples in some 4-neighbour column, with
/// no fault label within 1 voxel.
pub fn stair_index(r: &Run, s: &SaltBody) -> f64 {
    let [ni, nj, nk] = r.shape;
    let fp = footprint(s, r.shape);
    let dist = distance(&fp, ni, nj, BAND);
    let lab = |q: usize, k: usize| r.labels[q * nk + k];
    let fault_near = |i: usize, j: usize, k: usize| {
        if r.fault_id.is_empty() {
            return false;
        }
        for a in i.saturating_sub(1)..(i + 2).min(ni) {
            for b in j.saturating_sub(1)..(j + 2).min(nj) {
                for kk in k.saturating_sub(1)..(k + 2).min(nk) {
                    if r.fault_id[(a * nj + b) * nk + kk] != 0 {
                        return true;
                    }
                }
            }
        }
        false
    };
    let (mut stairs, mut total) = (0usize, 0usize);
    for i in 0..ni {
        for j in 0..nj {
            let q = i * nj + j;
            if dist[q] == 0 || dist[q] > BAND {
                continue;
            }
            for k in 0..nk {
                if s.contains(q, k) {
                    continue;
                }
                total += 1;
                if k == 0 || lab(q, k) == lab(q, k - 1) || s.contains(q, k - 1) {
                    continue;
                }
                if fault_near(i, j, k) {
                    continue;
                }
                let pair = (lab(q, k - 1), lab(q, k));
                let nb = [
                    (i.wrapping_sub(1), j),
                    (i + 1, j),
                    (i, j.wrapping_sub(1)),
                    (i, j + 1),
                ];
                let jump = nb.iter().any(|&(a, b)| {
                    if a >= ni || b >= nj {
                        return false;
                    }
                    let p = a * nj + b;
                    if s.contains(p, k) || fp[p] {
                        return false;
                    }
                    let lo = k.saturating_sub(2).max(1);
                    let hi = (k + 2).min(nk - 1);
                    !(lo..=hi).any(|kk| (lab(p, kk - 1), lab(p, kk)) == pair)
                });
                stairs += jump as usize;
            }
        }
    }
    stairs as f64 / total.max(1) as f64
}

/// Near-salt layer-label Jaccard (§4 row 6): `Σ_l |A_l ∩ B_l| / Σ_l |A_l ∪
/// B_l|` over the non-salt voxels within `BAND` columns of the footprint.
pub fn jaccard(a: &Run, b: &Run, s: &SaltBody) -> f64 {
    let [ni, nj, nk] = a.shape;
    let fp = footprint(s, a.shape);
    let dist = distance(&fp, ni, nj, BAND);
    let (mut inter, mut uni) = (0usize, 0usize);
    for q in 0..ni * nj {
        if dist[q] > BAND {
            continue;
        }
        for k in 0..nk {
            if s.contains(q, k) {
                continue;
            }
            let v = q * nk + k;
            // Each voxel adds 1 to the intersection when equal, else 2 to
            // the union (one per label).
            if a.labels[v] == b.labels[v] {
                inter += 1;
                uni += 1;
            } else {
                uni += 2;
            }
        }
    }
    inter as f64 / uni.max(1) as f64
}

/// Spec test 4 (drag sense / lift) and 5 (monotone, unset) on a run:
/// `(max output-minus-source cell, mean lift where moved in the first 3
/// columns off the flank, non-monotone columns, unset voxels)`; unset =
/// 255 between two labelled cells of a column.
pub fn warp_checks(r: &Run, s: &SaltBody) -> (i64, f64, usize, usize) {
    let [ni, nj, nk] = r.shape;
    let fp = footprint(s, r.shape);
    let dist = distance(&fp, ni, nj, BAND);
    let (mut worst, mut lift, mut moved, mut nonmono, mut unset) = (i64::MIN, 0u64, 0u64, 0, 0);
    for q in 0..ni * nj {
        let mut prev = 0u32;
        let mut mono = true;
        for k in 0..nk {
            let v = q * nk + k;
            let s_k = if r.warp.is_empty() {
                k as u32
            } else {
                r.warp[v]
            };
            worst = worst.max(k as i64 - s_k as i64);
            if k > 0 && s_k < prev {
                mono = false;
            }
            prev = s_k;
            if (1..=3).contains(&dist[q]) && !s.contains(q, k) && s_k as usize > k {
                lift += (s_k as usize - k) as u64;
                moved += 1;
            }
        }
        nonmono += (!mono) as usize;
        let col = &r.labels[q * nk..(q + 1) * nk];
        if let (Some(a), Some(b)) = (
            col.iter().position(|&l| l != 255),
            col.iter().rposition(|&l| l != 255),
        ) {
            unset += col[a..=b].iter().filter(|&&l| l == 255).count();
        }
    }
    (worst, lift as f64 / moved.max(1) as f64, nonmono, unset)
}
