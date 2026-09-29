//! Property-tile fuse path (default rock-physics model): the CPU adapter is
//! bit-identical to the trend path when fed the same properties, honours the
//! no-wavelet skip, and WGSL mode 1 is near-parity when an adapter exists.

use synthoseis_gpu::{
    fuse_props_tile_cpu, fuse_props_tile_dispatch, fuse_props_tile_gpu, fuse_tile_cpu,
    gpu_device_available, props_f32, set_prefer_gpu, FuseBackend, GPU_CPU_MAX_ABS_TOL,
    NO_WAVELET,
};
use synthoseis_seismic::ricker;

fn fixture() -> (Vec<u8>, [usize; 3], [Vec<f64>; 9], Vec<f64>) {
    let shape = [4usize, 3, 40];
    let [ni, nj, nk] = shape;
    let mut labels = vec![0u8; ni * nj * nk];
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let b1 = 8 + i + j;
                let b2 = 22 + 2 * i;
                labels[(i * nj + j) * nk + k] = if k < b1 {
                    0
                } else if k < b2 {
                    1
                } else {
                    2
                };
            }
        }
    }
    let trends = [
        (0..nk).map(|k| 1500.0 + 0.1 * k as f64).collect(),
        (0..nk).map(|_| 1000.0).collect(),
        vec![1.028; nk],
        (0..nk).map(|k| 2400.0 + 3.0 * k as f64).collect(),
        (0..nk).map(|k| 1000.0 + 1.5 * k as f64).collect(),
        vec![2.2; nk],
        (0..nk).map(|k| 2700.0 + 2.0 * k as f64).collect(),
        (0..nk).map(|k| 1350.0 + 0.9 * k as f64).collect(),
        vec![2.05; nk],
    ];
    (labels, shape, trends, ricker(30.0, 4.0, 2))
}

/// Tile property buffers derived from the trends (the legacy-toy lookup).
fn tile_props(
    labels: &[u8],
    shape: [usize; 3],
    trends: &[Vec<f64>; 9],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = labels.len();
    let nk = shape[2];
    let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for (idx, &lab) in labels.iter().enumerate() {
        let (p, s, r) = props_f32(lab, idx % nk, trends);
        vp[idx] = p;
        vs[idx] = s;
        rho[idx] = r;
    }
    (vp, vs, rho)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn props_cpu_bit_identical_to_trend_path() {
    let (labels, shape, trends, wav) = fixture();
    let [ni, nj, nk] = shape;
    let (vp, vs, rho) = tile_props(&labels, shape, &trends);
    for wavelet in [&wav[..], NO_WAVELET] {
        for angle in [0.0, 15.0, 30.0] {
            let mut a = vec![0f32; labels.len()];
            let mut b = vec![0f32; labels.len()];
            fuse_tile_cpu(&labels, shape, 0, ni, 0, nj, &trends, wavelet, angle, &mut a);
            fuse_props_tile_cpu(&vp, &vs, &rho, nk, wavelet, angle, &mut b);
            assert_eq!(a, b, "angle {angle} wavelet len {}", wavelet.len());
        }
    }
}

#[test]
fn props_gpu_near_parity_when_adapter() {
    let (labels, shape, trends, wav) = fixture();
    let nk = shape[2];
    let (vp, vs, rho) = tile_props(&labels, shape, &trends);
    for wavelet in [&wav[..], NO_WAVELET] {
        for angle in [0.0, 30.0] {
            let mut cpu = vec![0f32; labels.len()];
            let mut gpu = vec![0f32; labels.len()];
            fuse_props_tile_cpu(&vp, &vs, &rho, nk, wavelet, angle, &mut cpu);
            let backend = fuse_props_tile_gpu(&vp, &vs, &rho, nk, wavelet, angle, &mut gpu);
            if gpu_device_available() {
                assert_eq!(backend, FuseBackend::Gpu);
                let d = max_abs_diff(&cpu, &gpu);
                assert!(d <= GPU_CPU_MAX_ABS_TOL, "props gpu vs cpu {d}");
                // Pre-critical interfaces use exact sin/cos(asin) identities
                // and the host angle, so the gap is f32 rounding (~5e-7 on
                // llvmpipe) even for the strong water/sediment contrast.
                assert!(d <= 1e-4, "props gpu vs cpu {d} (angle {angle})");
            } else {
                assert_eq!(backend, FuseBackend::Cpu);
                assert_eq!(cpu, gpu);
            }
        }
    }
}

#[test]
fn props_dispatch_honours_prefer_gpu() {
    let (labels, shape, trends, wav) = fixture();
    let nk = shape[2];
    let (vp, vs, rho) = tile_props(&labels, shape, &trends);
    let mut cpu = vec![0f32; labels.len()];
    let mut out = vec![0f32; labels.len()];
    fuse_props_tile_cpu(&vp, &vs, &rho, nk, &wav, 15.0, &mut cpu);
    set_prefer_gpu(false);
    let b = fuse_props_tile_dispatch(&vp, &vs, &rho, nk, &wav, 15.0, &mut out);
    assert_eq!(b, FuseBackend::Cpu);
    assert_eq!(cpu, out);
}
