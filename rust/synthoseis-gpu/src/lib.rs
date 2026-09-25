//! Per-tile Zoeppritz + wavelet fuse kernels for scale stage 6 / 6b.
//!
//! # Backend choice
//!
//! - **CPU software** ([`fuse_tile_cpu`]): bit-identical to the prior core
//!   `fuse_tile_local` path. Always available; CI-safe on `ubuntu-latest`.
//! - **wgpu / WGSL** ([`fuse_tile_gpu`] when feature `wgpu` + adapter): f32
//!   Zoeppritz + same-mode wavelet convolution on the GPU. Falls back to CPU
//!   when no adapter is present. GPU vs CPU is **near-parity** (documented
//!   tolerances), not bit-identical — CPU uses f64 complex Zoeppritz.
//!
//! Host still owns strip partition + MDIO writes; this crate accelerates the
//! per-tile hot path.
//!
//! # Feature `wgpu`
//!
//! Optional. Default **on** so the CLI `--gpu` path can dispatch when a device
//! exists. Disable with `--no-default-features` for lean builds. Requires a
//! recent stable Rust (workspace `rust-version`; wgpu 24 needs ≥1.76 in
//! practice — CI uses `dtolnay/rust-toolchain@stable`).
//!
//! # API
//!
//! - [`fuse_tile_cpu`] — explicit CPU path
//! - [`fuse_tile_gpu`] — request GPU; CPU fallback when no device / no feature
//! - [`fuse_tile_auto`] — prefer GPU when available, else CPU
//! - [`fuse_tile_dispatch`] — honors [`set_prefer_gpu`] (CLI `--gpu`)

mod cpu;
mod device;

#[cfg(feature = "wgpu")]
mod wgpu_fuse;

use std::sync::atomic::{AtomicBool, Ordering};

pub use cpu::{fuse_tile_cpu, fuse_tile_scratch_bytes, props_f32};
pub use device::{backend_status, gpu_device_available, FuseBackend};

static PREFER_GPU: AtomicBool = AtomicBool::new(false);

/// Prefer the GPU auto path for subsequent [`fuse_tile_dispatch`] calls.
///
/// Used by CLI `--gpu`. Safe to call from the main thread before workers start;
/// workers inherit the process-global flag.
pub fn set_prefer_gpu(prefer: bool) {
    PREFER_GPU.store(prefer, Ordering::SeqCst);
}

/// Whether [`set_prefer_gpu`] was enabled.
pub fn prefer_gpu() -> bool {
    PREFER_GPU.load(Ordering::SeqCst)
}

/// Fuse a tile on CPU and report [`FuseBackend::Cpu`].
pub fn fuse_tile_cpu_report(
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
) -> FuseBackend {
    fuse_tile_cpu(
        labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
    );
    FuseBackend::Cpu
}

/// Request GPU acceleration for one tile.
///
/// Falls back to the CPU software adapter when the `wgpu` feature is off or no
/// adapter is available. Returns the backend that ran.
pub fn fuse_tile_gpu(
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
) -> FuseBackend {
    #[cfg(feature = "wgpu")]
    {
        if let Some(backend) = crate::wgpu_fuse::fuse_tile_wgpu(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
        ) {
            return backend;
        }
    }
    fuse_tile_cpu_report(
        labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
    )
}

/// Prefer GPU when available; otherwise CPU.
pub fn fuse_tile_auto(
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
) -> FuseBackend {
    if gpu_device_available() {
        fuse_tile_gpu(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
        )
    } else {
        fuse_tile_cpu_report(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
        )
    }
}

/// Dispatch based on [`prefer_gpu`]: auto path when set, else explicit CPU.
pub fn fuse_tile_dispatch(
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
) -> FuseBackend {
    if prefer_gpu() {
        fuse_tile_auto(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
        )
    } else {
        fuse_tile_cpu_report(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
        )
    }
}

/// Documented max-abs tolerance for GPU (f32 WGSL) vs CPU (f64) on tiny tiles.
///
/// Complex Zoeppritz in f32 WGSL vs f64 `num_complex` drifts most at larger
/// incidence angles (post-critical edge). Parity tests allow ≤1e-2 max-abs on
/// fixture tiles; operators should treat GPU stacks as near-parity deliverables
/// (same philosophy as angle-stack MAE), not bit-identical.
pub const GPU_CPU_MAX_ABS_TOL: f32 = 1e-2;

#[cfg(test)]
mod tests {
    use super::*;
    use synthoseis_seismic::ricker;

    fn tiny_trends(nk: usize) -> [Vec<f64>; 9] {
        [
            vec![3000.0; nk],
            vec![1500.0; nk],
            vec![2.3; nk],
            vec![3200.0; nk],
            vec![1600.0; nk],
            vec![2.2; nk],
            vec![2800.0; nk],
            vec![1400.0; nk],
            vec![2.0; nk],
        ]
    }

    #[test]
    fn fuse_tile_cpu_runs_tiny_tile() {
        let shape = [2usize, 2, 8];
        let n = 2 * 2 * 8;
        let mut labels = vec![0u8; n];
        labels[8] = 1;
        labels[16] = 2;
        let trends = tiny_trends(8);
        let wav = ricker(25.0, 4.0, 1);
        let mut out = vec![0.0f32; 1 * 2 * 8];
        fuse_tile_cpu(&labels, shape, 0, 1, 0, 2, &trends, &wav, 15.0, &mut out);
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn fuse_tile_gpu_returns_cpu_or_gpu() {
        let shape = [2usize, 2, 8];
        let labels = vec![0u8; 2 * 2 * 8];
        let trends = tiny_trends(8);
        let wav = ricker(25.0, 4.0, 1);
        let mut out = vec![0.0f32; 2 * 2 * 8];
        let backend = fuse_tile_gpu(&labels, shape, 0, 2, 0, 2, &trends, &wav, 0.0, &mut out);
        assert!(matches!(backend, FuseBackend::Cpu | FuseBackend::Gpu));
        if !gpu_device_available() {
            assert_eq!(backend, FuseBackend::Cpu);
        }
    }

    #[test]
    fn prefer_gpu_dispatch_toggle() {
        set_prefer_gpu(false);
        assert!(!prefer_gpu());
        set_prefer_gpu(true);
        assert!(prefer_gpu());
        set_prefer_gpu(false);
    }

    #[test]
    fn backend_status_is_nonempty() {
        assert!(!backend_status().is_empty());
    }
}
