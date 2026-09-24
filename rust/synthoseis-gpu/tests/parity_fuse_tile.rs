//! Parity: `fuse_tile_cpu` / `fuse_tile_auto` / `fuse_tile_gpu` vs an independent
//! reference that mirrors the historical `fuse_tile_local` body.

use synthoseis_gpu::{fuse_tile_auto, fuse_tile_cpu, fuse_tile_gpu, FuseBackend};
use synthoseis_seismic::{convolve_same_1d, ricker, zoeppritz_pp};

fn props_f32(lab: u8, k: usize, trends: &[Vec<f64>; 9]) -> (f32, f32, f32) {
    let (vp, vs, rho) = match lab {
        0 => (trends[0][k], trends[1][k], trends[2][k]),
        1 => (trends[3][k], trends[4][k], trends[5][k]),
        _ => (trends[6][k], trends[7][k], trends[8][k]),
    };
    (vp as f32, vs as f32, rho as f32)
}

/// Independent copy of the historical fuse_tile_local loop (for bit-identical check).
fn fuse_tile_reference(
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
    let tj = j1 - j0;
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

fn fixture() -> (Vec<u8>, [usize; 3], [Vec<f64>; 9], Vec<f64>) {
    let shape = [4usize, 3, 16];
    let n = shape[0] * shape[1] * shape[2];
    let mut labels = vec![0u8; n];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let idx = (i * shape[1] + j) * shape[2] + k;
                labels[idx] = ((i + j + k) % 3) as u8;
            }
        }
    }
    let nk = shape[2];
    let trends = [
        (0..nk).map(|k| 2900.0 + k as f64).collect(),
        (0..nk).map(|k| 1450.0 + 0.5 * k as f64).collect(),
        vec![2.35; nk],
        (0..nk).map(|k| 3100.0 + k as f64).collect(),
        (0..nk).map(|k| 1550.0 + 0.4 * k as f64).collect(),
        vec![2.25; nk],
        (0..nk).map(|k| 2700.0 + 0.8 * k as f64).collect(),
        (0..nk).map(|k| 1350.0 + 0.3 * k as f64).collect(),
        vec![2.05; nk],
    ];
    let wav = ricker(30.0, 4.0, 2);
    (labels, shape, trends, wav)
}

#[test]
fn fuse_tile_cpu_bit_identical_to_reference() {
    let (labels, shape, trends, wav) = fixture();
    let (i0, i1, j0, j1) = (1, 3, 0, 2);
    let tile_n = (i1 - i0) * (j1 - j0) * shape[2];
    let mut ref_out = vec![0.0f32; tile_n];
    let mut cpu_out = vec![0.0f32; tile_n];
    fuse_tile_reference(
        &labels, shape, i0, i1, j0, j1, &trends, &wav, 15.0, &mut ref_out,
    );
    fuse_tile_cpu(
        &labels, shape, i0, i1, j0, j1, &trends, &wav, 15.0, &mut cpu_out,
    );
    assert_eq!(ref_out, cpu_out);
}

#[test]
fn fuse_tile_gpu_and_auto_match_cpu() {
    let (labels, shape, trends, wav) = fixture();
    let (i0, i1, j0, j1) = (0, 4, 1, 3);
    let tile_n = (i1 - i0) * (j1 - j0) * shape[2];
    let mut cpu = vec![0.0f32; tile_n];
    let mut gpu = vec![0.0f32; tile_n];
    let mut auto = vec![0.0f32; tile_n];
    fuse_tile_cpu(&labels, shape, i0, i1, j0, j1, &trends, &wav, 0.0, &mut cpu);
    let b_gpu = fuse_tile_gpu(&labels, shape, i0, i1, j0, j1, &trends, &wav, 0.0, &mut gpu);
    let b_auto = fuse_tile_auto(&labels, shape, i0, i1, j0, j1, &trends, &wav, 0.0, &mut auto);
    assert_eq!(b_gpu, FuseBackend::Cpu);
    assert_eq!(b_auto, FuseBackend::Cpu);
    assert_eq!(cpu, gpu);
    assert_eq!(cpu, auto);
}

#[test]
fn fuse_tile_cpu_matches_at_multiple_angles() {
    let (labels, shape, trends, wav) = fixture();
    let (i0, i1, j0, j1) = (0, 2, 0, 3);
    let tile_n = (i1 - i0) * (j1 - j0) * shape[2];
    for &ang in &[0.0_f64, 10.0, 25.0, 40.0] {
        let mut a = vec![0.0f32; tile_n];
        let mut b = vec![0.0f32; tile_n];
        fuse_tile_reference(&labels, shape, i0, i1, j0, j1, &trends, &wav, ang, &mut a);
        fuse_tile_cpu(&labels, shape, i0, i1, j0, j1, &trends, &wav, ang, &mut b);
        assert_eq!(a, b, "mismatch at angle {ang}");
    }
}
