//! Geology / horizons ports from `datagenerator/Horizons.py`.
//!
//! # First landed kernels
//! - [`fit_plane_lsq`] / [`eval_plane`] — dipping-plane helpers used for layer thickness
//! - [`rotate_point`] — 2-D rotation about origin
//! - [`enforce_nonnegative_thicknesses`] — negative-thickness clip from
//!   `Horizons.insert_feature_into_horizon_stack`
//! - [`fill_layer_labels`] — discrete labels between successive horizon depths
//!
//! The Python generator remains the reference baseline; these kernels are covered by
//! golden fixtures in `tests/fixtures/parity_cubes_8.json`.

mod kernels;
#[cfg(test)]
#[path = "tests_kernels.rs"]
mod tests_kernels;

pub use kernels::{
    enforce_nonnegative_thicknesses, eval_plane, fill_layer_labels, fit_plane_lsq, rotate_point,
};
