//! Device availability helpers and [`FuseBackend`] reporting.
//!
//! With the `wgpu` feature, [`gpu_device_available`] probes for an adapter once
//! (hardware or software such as llvmpipe). Without the feature, or when no
//! adapter exists, the CPU software backend is the only path.

/// Which backend actually executed a fuse call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseBackend {
    /// Host CPU software adapter (default; CI-safe; bit-identical to prior core path).
    Cpu,
    /// WGSL compute dispatch via wgpu (f32; near-parity with CPU, not bit-identical).
    Gpu,
}

impl FuseBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            FuseBackend::Cpu => "cpu",
            FuseBackend::Gpu => "gpu",
        }
    }
}

/// True when a GPU compute device is available for tile fuse.
///
/// Without the `wgpu` feature this is always `false`. With `wgpu`, returns the
/// result of a one-shot adapter probe (may be software Vulkan / Metal / DX12).
pub fn gpu_device_available() -> bool {
    #[cfg(feature = "wgpu")]
    {
        crate::wgpu_fuse::gpu_device_available()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        false
    }
}

/// Human-readable status for CLI / logs.
pub fn backend_status() -> String {
    #[cfg(feature = "wgpu")]
    {
        crate::wgpu_fuse::backend_status()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        "cpu-software (wgpu feature disabled)".to_string()
    }
}
