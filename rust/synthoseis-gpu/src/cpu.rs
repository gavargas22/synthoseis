//! CPU software adapter for per-tile Zoeppritz RFC + wavelet convolution.
//!
//! Bit-identical to the former `synthoseis_core::fuse_tile_local` hot path:
//! label→elastic props → Zoeppritz PP → `convolve_same_1d` per trace.

use synthoseis_seismic::{convolve_same_1d, zoeppritz_pp, zoeppritz_pp_form, ZoeppritzForm};

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

/// Zoeppritz + convolution of one trace from its f32 properties.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fuse_trace(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    wavelet: &[f64],
    angle_deg: f64,
    form: ZoeppritzForm,
    rfc_tr: &mut [f32],
    trace_f64: &mut [f64],
    out: &mut [f32],
) {
    let nk = vp.len();
    for k in 0..(nk - 1) {
        rfc_tr[k] = zoeppritz_pp_form(
            vp[k] as f64,
            vs[k] as f64,
            rho[k] as f64,
            vp[k + 1] as f64,
            vs[k + 1] as f64,
            rho[k + 1] as f64,
            angle_deg,
            form,
        );
    }
    rfc_tr[nk - 1] = 0.0;
    for k in 0..nk {
        trace_f64[k] = rfc_tr[k] as f64;
    }
    let conv = convolve_same_1d(trace_f64, wavelet);
    for k in 0..nk {
        out[k] = conv[k] as f32;
    }
}

/// Fuse one tile from precomputed per-voxel properties.
///
/// `vp`, `vs`, `rho` and `tile_out` are `(ti, tj, nk)` row-major tile
/// buffers (`n_traces = ti * tj`). Same per-trace arithmetic as
/// [`fuse_tile_cpu`]: f32 properties, f64 Zoeppritz, `rfc[nk-1] = 0`,
/// `convolve_same_1d`. `form` selects the Zoeppritz expression
/// ([`ZoeppritzForm::Legacy`] reproduces [`fuse_tile_cpu`]).
#[allow(clippy::too_many_arguments)]
pub fn fuse_props_tile_cpu(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    nk: usize,
    wavelet: &[f64],
    angle_deg: f64,
    form: ZoeppritzForm,
    tile_out: &mut [f32],
) {
    assert!(nk >= 1);
    let n = tile_out.len();
    assert!(vp.len() == n && vs.len() == n && rho.len() == n && n % nk == 0);
    let mut rfc_tr = vec![0.0f32; nk];
    let mut trace_f64 = vec![0.0f64; nk];
    for t in 0..n / nk {
        let r = t * nk..(t + 1) * nk;
        fuse_trace(
            &vp[r.clone()],
            &vs[r.clone()],
            &rho[r.clone()],
            wavelet,
            angle_deg,
            form,
            &mut rfc_tr,
            &mut trace_f64,
            &mut tile_out[r],
        );
    }
}

/// Scratch bytes used by [`fuse_tile_cpu`] for one tile (trace temps only).
pub fn fuse_tile_scratch_bytes(nk: usize) -> usize {
    // vp + vs + rho + rfc (f32) + trace_f64
    (4 * 4 + 8) * nk
}
