//! Bounded RPM depth-trend kernels.

mod backus;
mod legacy;
mod trends;

pub use legacy::{
    arithmetic_mean, example_f32, forward_fill_zeros, harmonic_mean, legacy_column_properties,
    mix_f32, sand_f32, shale_f32, shifted_index, voxel_properties, Elastic32, Fluid, LayerShifts,
    MixingMethod, VoxelKind, WATER,
};

pub use backus::{backus_mix, slowness_sum, voigt_mix};

pub use trends::{
    polyval, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp, tagilsk_brine_sand_vs,
    tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho, tagilsk_shale_vp,
    tagilsk_shale_vs, RpmExampleTrends, TagilskTrends,
};
