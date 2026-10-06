//! Helpers shared by the `synthoseis-core` integration tests
//! (`mod common;` in each test crate).

use synthoseis_core::partial_voxels::PartialVoxelConfig;
use synthoseis_core::pipeline::E2eConfig;

/// `c` with partial voxels opted out (`PartialVoxelConfig::whole_voxels`,
/// = the d8b96e69 library default). Partial voxels are the library default
/// since PR B2 (#43); goldens that assert master (whole-voxel) output run
/// through this.
pub fn whole(c: &E2eConfig) -> E2eConfig {
    let mut c = c.clone();
    c.rock_physics.partial_voxels = PartialVoxelConfig::whole_voxels();
    c
}
