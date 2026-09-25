//! Fault-surface segmentation: port of `Faults.get_fault_plane_sobel`.
//!
//! Python builds `inside = ellipsoid <= 1`, takes
//! `|sobel_x| + |sobel_y| + |sobel_z|` (scipy `mode="reflect"`), divides by a
//! `5×5×5` `maximum_filter` of itself and thresholds at `0.5` (values exactly
//! `0.5` are kept as `0.5`). That is a radius-3 stencil over an analytic
//! indicator, so a tile only needs a 3-voxel lateral halo that is *recomputed
//! from the fault parameters* (not read from neighbours) — tiles are therefore
//! exactly independent of each other.

use super::model::FaultGeometry;

/// `fault_segments == 0`.
pub const SEG_ZERO: u8 = 0;
/// `fault_segments == 0.5` (exact tie of `edge / edge_max`).
pub const SEG_HALF: u8 = 1;
/// `fault_segments == 1`.
pub const SEG_ONE: u8 = 2;

#[inline]
fn clampi(v: isize, n: usize) -> usize {
    v.clamp(0, n as isize - 1) as usize
}

/// Compute segment codes for tile `[i0,i1) × [j0,j1) × [0,nk)` into `out`
/// (row-major `(i1-i0, j1-j0, nk)`).
pub fn segment_block(
    geom: &FaultGeometry,
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    out: &mut Vec<u8>,
) {
    let [ni, nj, nk] = shape;
    let (ti, tj) = (i1 - i0, j1 - j0);
    out.clear();
    out.resize(ti * tj * nk, SEG_ZERO);
    if ti == 0 || tj == 0 || nk == 0 {
        return;
    }

    // Inside indicator on the halo block (clamped to the cube).
    let bi0 = i0.saturating_sub(3);
    let bi1 = (i1 + 3).min(ni);
    let bj0 = j0.saturating_sub(3);
    let bj1 = (j1 + 3).min(nj);
    let (bni, bnj) = (bi1 - bi0, bj1 - bj0);
    let mut inside = vec![0i32; bni * bnj * nk];
    let mut n_in = 0usize;
    for i in bi0..bi1 {
        for j in bj0..bj1 {
            let base = ((i - bi0) * bnj + (j - bj0)) * nk;
            for k in 0..nk {
                if geom.inside(i, j, k) {
                    inside[base + k] = 1;
                    n_in += 1;
                }
            }
        }
    }
    if n_in == 0 || n_in == inside.len() {
        return; // uniform block → sobel is identically zero
    }
    let at = |i: usize, j: usize, k: usize| inside[((i - bi0) * bnj + (j - bj0)) * nk + k];

    // Edge magnitude on the 2-voxel halo range.
    let ei0 = i0.saturating_sub(2);
    let ei1 = (i1 + 2).min(ni);
    let ej0 = j0.saturating_sub(2);
    let ej1 = (j1 + 2).min(nj);
    let (eni, enj) = (ei1 - ei0, ej1 - ej0);
    let mut edge = vec![0i32; eni * enj * nk];
    const W: [i32; 3] = [1, 2, 1];
    for i in ei0..ei1 {
        let im = clampi(i as isize - 1, ni);
        let ip = clampi(i as isize + 1, ni);
        for j in ej0..ej1 {
            let jm = clampi(j as isize - 1, nj);
            let jp = clampi(j as isize + 1, nj);
            let is = [im, i, ip];
            let js = [jm, j, jp];
            for k in 0..nk {
                let km = clampi(k as isize - 1, nk);
                let kp = clampi(k as isize + 1, nk);
                let ks = [km, k, kp];
                let (mut sx, mut sy, mut sz) = (0i32, 0i32, 0i32);
                for a in 0..3 {
                    for b in 0..3 {
                        let w = W[a] * W[b];
                        sx += w * (at(ip, js[a], ks[b]) - at(im, js[a], ks[b]));
                        sy += w * (at(is[a], jp, ks[b]) - at(is[a], jm, ks[b]));
                        sz += w * (at(is[a], js[b], kp) - at(is[a], js[b], km));
                    }
                }
                edge[((i - ei0) * enj + (j - ej0)) * nk + k] = sx.abs() + sy.abs() + sz.abs();
            }
        }
    }

    // Separable 5×5×5 maximum filter (reflect == clamp for max), k → j → i.
    let mut m1 = vec![0i32; edge.len()];
    for ij in 0..eni * enj {
        let base = ij * nk;
        for k in 0..nk {
            let lo = k.saturating_sub(2);
            let hi = (k + 2).min(nk - 1);
            m1[base + k] = edge[base + lo..=base + hi].iter().copied().max().unwrap();
        }
    }
    let mut m2 = vec![0i32; edge.len()];
    for i in 0..eni {
        for j in 0..enj {
            let gj = ej0 + j;
            let lo = gj.saturating_sub(2).max(ej0) - ej0;
            let hi = (gj + 2).min(nj - 1).min(ej1 - 1) - ej0;
            for k in 0..nk {
                let mut m = i32::MIN;
                for jj in lo..=hi {
                    m = m.max(m1[(i * enj + jj) * nk + k]);
                }
                m2[(i * enj + j) * nk + k] = m;
            }
        }
    }
    for i in i0..i1 {
        let lo = i.saturating_sub(2) - ei0;
        let hi = (i + 2).min(ni - 1) - ei0;
        for j in j0..j1 {
            let jl = j - ej0;
            for k in 0..nk {
                let e = edge[((i - ei0) * enj + jl) * nk + k];
                if e == 0 {
                    continue;
                }
                let mut emax = i32::MIN;
                for ii in lo..=hi {
                    emax = emax.max(m2[(ii * enj + jl) * nk + k]);
                }
                let code = match (2 * e).cmp(&emax) {
                    std::cmp::Ordering::Less => SEG_ZERO,
                    std::cmp::Ordering::Equal => SEG_HALF,
                    std::cmp::Ordering::Greater => SEG_ONE,
                };
                out[((i - i0) * tj + (j - j0)) * nk + k] = code;
            }
        }
    }
}
