//! Rock-physics model (RPM) depth-trend ports from `rockphysics/`.
//!
//! # First landed kernels
//! - [`polyval`] — numpy-compatible polynomial evaluation
//! - [`RpmExampleTrends`] — example shale / brine / oil / gas sand trends
//! - Tagilsk shale / brine / gas sand helpers (`tagilsk_*`)
//!
//! - Legacy property builder in float32 ([`legacy_column_properties`],
//!   [`mix_f32`], [`example_f32`]) used by the default rock-physics model
//!
//! Golden fixtures: `tests/fixtures/rpm_trends.json`
//! (regenerate via `tests/fixtures/generate_seismic_kernels.py`).

mod kernels;
#[cfg(test)]
#[path = "tests_kernels.rs"]
mod tests_kernels;

pub use kernels::{
    arithmetic_mean, backus_mix, example_f32, forward_fill_zeros, harmonic_mean, legacy_column_properties,
    slowness_sum, voigt_mix,
    mix_f32, sand_f32, shale_f32, shifted_index, voxel_properties, Elastic32, Fluid, LayerShifts,
    MixingMethod, VoxelKind, SALT, WATER,
    polyval, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp, tagilsk_brine_sand_vs,
    tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho, tagilsk_shale_vp,
    tagilsk_shale_vs, RpmExampleTrends, TagilskTrends,
};
