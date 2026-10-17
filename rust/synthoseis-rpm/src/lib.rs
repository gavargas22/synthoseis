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
#[cfg(test)]
#[path = "tests_physical.rs"]
mod tests_physical;

pub use kernels::{
    arithmetic_mean, athy_porosity, backus_mix, bw_brine, bw_gas, bw_oil, bw_water,
    clay_quartz_mineral, effective_pressure, elastic_from_moduli, end_member, end_member_f64,
    example_f32, forward_fill_zeros, gassmann, gassmann_inverse, harmonic_mean, hertz_mindlin,
    legacy_column_properties, mix_f32, physical_column_properties, physical_sand, physical_shale,
    physical_voxel_properties, polyval, salt_elastic, sand_f32, shale_f32, shifted_index,
    slowness_sum, soft_sand_dry, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp,
    tagilsk_brine_sand_vs, tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho,
    tagilsk_shale_vp, tagilsk_shale_vs, voigt_mix, voxel_properties, wood, CompactionForm,
    DryFrame, Elastic32, Fluid, LayerShifts, Lithology, MixingMethod, PhysParams, PoreFluid,
    RockModel, RpmExampleTrends, TagilskTrends, VoxelKind, CLAY, GUARDED_VOXELS, VP_MIN_OBSERVED, observed_vp_min, QUARTZ, SALT,
    SALT_RHO, SALT_VP, SALT_VS, SALT_VS_LEGACY, SHALE_CLAY_FRACTION, SHALE_COORDINATION, WATER,
};
