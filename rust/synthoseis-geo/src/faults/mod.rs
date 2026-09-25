//! Fault modelling — port of `datagenerator/Faults.py`.
//!
//! See `docs/faults-port.md` for the Python → Rust mapping, what is deferred,
//! and the parity numbers against the legacy generator.
//!
//! Pipeline:
//! 1. [`sample_random_faults`] (or explicit [`FaultParams`]) — deterministic,
//!    seeded parameter draw (`_fault_params_random` + `xyz_dis` draws).
//! 2. [`FaultModel::resolve`] — the only global step: per fault, a streamed
//!    survey of the fault surface to pick the max-displacement centre
//!    (`get_middle_z`) and taper the vertical profile under the seabed.
//!    Memory is O(fault surface), never O(cube). [`ReachMode`] picks the
//!    exact legacy taper or the default `FitColumn` (legacy whenever the
//!    legacy taper succeeds; sigma fitted to the column when it gives up).
//! 3. [`FaultModel::compute_tile`] — per spatial tile: displacement lookup,
//!    binary fault mask, fault segment ids. Tiles are exactly independent.

mod apply;
mod model;
mod params;
mod segments;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_reach;

pub use apply::{
    horizon_depth_from_age, interp_trace_into, interp_uniform, remap_nearest, FaultModel,
    FaultTile, SURVEY_TILE,
};
pub use model::{
    fit_sigma_to_column, middle_candidates, resolve_fault, resolve_fault_mode, survey_surface,
    taper, throw_variance, vertical_profile, vertical_profile_mode, vertical_reach, FaultGeometry,
    FaultSkip, ReachMode, ResolvedFault, Seabed, Taper, VerticalProfile, MIN_RESCUE_SIGMA,
};
pub use params::{
    sample_random_faults, FaultParams, FaultRng, RandomFaultConfig, HOCKEY_STICK_MIN_THROW,
    LEGACY_INFILL_FACTOR, THROW_LUT_MAX, THROW_LUT_MIN,
};
pub use segments::{segment_block, SEG_HALF, SEG_ONE, SEG_ZERO};
