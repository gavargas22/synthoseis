//! CPU software adapter for per-tile Zoeppritz RFC + wavelet convolution.
//!
//! Bit-identical to the former `synthoseis_core::fuse_tile_local` hot path:
//! label→elastic props → Zoeppritz PP → `convolve_same_1d` per trace.

use synthoseis_seismic::{convolve_same_1d, zoeppritz_pp};

/// Resolve (vp, vs, rho) for one label at depth sample `k` from the 9 RPM trends.
#[inline]
pub fn props_f32(lab: u8, k: usize, trends: &[Vec<f64>; 9]) -> (f32, f32, f32) {
    let (vp, vs, rho) = match lab {
        0 => (trends[0][k], trends[1][k], trends[2][k]),
        1 => (trends[3][k], trends[4][k], trends[5][k]),
        _ => (trends[6][k], trends[7][k], trends[8][k]),
    };
    (vp as f32, vs as f32, rho as f32)
}

/// Fuse one spatial tile on the host CPU.
///
/// `labels` is the full `(ni, nj, nk)` volume (row-major). The tile covers
/// inlines `[i0, i1)` and crosslines `[j0, j1)`; `tile_out` is
/// `((i1-i0)*(j1-j0)*nk)` samples in the same layout.
///
/// Temps stay at trace scale (RAM-bounded), matching the chunked streaming invariant.
pub fn fuse_tile_cpu(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
) {
    let [_ni, nj, nk] = shape;
    let ti = i1 - i0;
    let tj = j1 - j0;
    assert_eq!(tile_out.len(), ti * tj * nk);
    assert!(i1 <= shape[0] && j1 <= shape[1]);
    assert!(nk >= 1);

    let mut vp_tr = vec![0.0f32; nk];
    let mut vs_tr = vec![0.0f32; nk];
    let mut rho_tr = vec![0.0f32; nk];
    let mut rfc_tr = vec![0.0f32; nk];
    let mut trace_f64 = vec![0.0f64; nk];

    for (di, i) in (i0..i1).enumerate() {
        for (dj, j) in (j0..j1).enumerate() {
            for k in 0..nk {
                let idx = (i * nj + j) * nk + k;
                let (vp, vs, rho) = props_f32(labels[idx], k, trends);
                vp_tr[k] = vp;
                vs_tr[k] = vs;
                rho_tr[k] = rho;
            }
            for k in 0..(nk - 1) {
                rfc_tr[k] = zoeppritz_pp(
                    vp_tr[k] as f64,
                    vs_tr[k] as f64,
                    rho_tr[k] as f64,
                    vp_tr[k + 1] as f64,
                    vs_tr[k + 1] as f64,
                    rho_tr[k + 1] as f64,
                    angle_deg,
                );
            }
            rfc_tr[nk - 1] = 0.0;
            for k in 0..nk {
                trace_f64[k] = rfc_tr[k] as f64;
            }
            let conv = convolve_same_1d(&trace_f64, wavelet);
            for k in 0..nk {
                tile_out[(di * tj + dj) * nk + k] = conv[k] as f32;
            }
        }
    }
}

/// Scratch bytes used by [`fuse_tile_cpu`] for one tile (trace temps only).
pub fn fuse_tile_scratch_bytes(nk: usize) -> usize {
    // vp + vs + rho + rfc (f32) + trace_f64
    (4 * 4 + 8) * nk
}
