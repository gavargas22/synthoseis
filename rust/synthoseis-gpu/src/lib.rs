//! Per-tile Zoeppritz + wavelet fuse kernels for scale stage 6.
//!
//! # Backend choice (this PR)
//!
//! **CPU software adapter** is the default and the only fuse executor in this
//! cut. Host still owns strip partition + MDIO writes; this crate accelerates
//! (today: hosts) the **per-tile** hot path that used to live as
//! `fuse_tile_local` inside `synthoseis-core`.
//!
//! A follow-up within stage 6 will add wgpu adapter probe + WGSL Zoeppritz /
//! convolution dispatch. CI on `ubuntu-latest` stays green without hardware
//! because the default backend is CPU software.
//!
//! # API
//!
//! - [`fuse_tile_cpu`] — explicit CPU path (bit-identical to prior core fuse)
//! - [`fuse_tile_gpu`] — request GPU; falls back to CPU when no device / no feature
//! - [`fuse_tile_auto`] — prefer GPU when available, else CPU
//!
//! CLI `--gpu` sets [`set_prefer_gpu`]; pipeline wrappers call [`fuse_tile_auto`]
//! when prefer-gpu is on, otherwise [`fuse_tile_cpu`].

mod cpu;
mod device;

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
/// Falls back to the CPU software adapter when no wgpu device is available
/// (always in the default feature set). Returns the backend that ran.
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
    // Compute shaders deferred: even when an adapter exists, execute the CPU
    // software path so results stay bit-identical. Availability is exposed via
    // `gpu_device_available` / `backend_status` for operators and follow-up PRs.
    fuse_tile_cpu_report(
        labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out,
    )
}

/// Prefer GPU when available; otherwise CPU. Same kernels as [`fuse_tile_gpu`]
/// in this cut (CPU software).
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

#[cfg(test)]
mod tests {
    use super::*;
    use synthoseis_seismic::ricker;

    fn tiny_trends(nk: usize) -> [Vec<f64>; 9] {
        // Constant-ish elastic props per facies (shale / brine / gas).
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
    fn fuse_tile_gpu_falls_back_to_cpu() {
        let shape = [2usize, 2, 8];
        let labels = vec![0u8; 2 * 2 * 8];
        let trends = tiny_trends(8);
        let wav = ricker(25.0, 4.0, 1);
        let mut out = vec![0.0f32; 2 * 2 * 8];
        let backend = fuse_tile_gpu(&labels, shape, 0, 2, 0, 2, &trends, &wav, 0.0, &mut out);
        assert_eq!(backend, FuseBackend::Cpu);
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
    fn gpu_device_available_false_in_this_cut() {
        assert!(!gpu_device_available());
    }
}
