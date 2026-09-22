//! Device availability helpers.
//!
//! This cut ships the **CPU software** fuse backend only.
//! [`gpu_device_available`] is always false until a follow-up wires wgpu/WGSL.

/// Which backend actually executed a fuse call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseBackend {
    /// Host CPU software adapter (default; CI-safe; bit-identical to prior core path).
    Cpu,
}

impl FuseBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            FuseBackend::Cpu => "cpu",
        }
    }
}

/// True when a GPU compute device is available for tile fuse.
///
/// Always `false` in this cut (CPU software only). Follow-up: wgpu + WGSL.
pub fn gpu_device_available() -> bool {
    false
}

/// Human-readable status for CLI / logs.
pub fn backend_status() -> String {
    "cpu-software (default; WGSL/wgpu dispatch deferred)".to_string()
}
