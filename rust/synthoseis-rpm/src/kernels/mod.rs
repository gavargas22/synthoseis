//! Bounded RPM depth-trend kernels.

mod backus;
mod legacy;
mod physical;
mod trends;

pub use legacy::{
    arithmetic_mean, example_f32, forward_fill_zeros, harmonic_mean, legacy_column_properties,
    mix_f32, sand_f32, shale_f32, shifted_index, voxel_properties, Elastic32, Fluid, LayerShifts,
    MixingMethod, VoxelKind, SALT, SALT_RHO, SALT_VP, SALT_VS, SALT_VS_LEGACY, WATER, salt_elastic,
};

pub use backus::{backus_mix, slowness_sum, voigt_mix};

pub use trends::{
    polyval, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp, tagilsk_brine_sand_vs,
    tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho, tagilsk_shale_vp,
    tagilsk_shale_vs, RpmExampleTrends, TagilskTrends,
};

pub use physical::{
    athy_porosity, bw_brine, bw_gas, bw_oil, bw_water, clay_quartz_mineral, effective_pressure,
    elastic_from_moduli, end_member, end_member_f64, gassmann, gassmann_inverse, hertz_mindlin,
    physical_column_properties, physical_sand, physical_shale, physical_voxel_properties,
    soft_sand_dry, wood, CompactionForm, DryFrame, Lithology, PhysParams, PoreFluid, RockModel,
    CLAY, GUARDED_VOXELS, QUARTZ, SHALE_CLAY_FRACTION, SHALE_COORDINATION,
};
