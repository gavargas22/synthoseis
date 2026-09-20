//! Closure / trap geometry ports from `datagenerator/Closures.py` and
//! `datagenerator/_closures_vectorised.py`.
//!
//! # First landed kernels
//! - [`bincount_label_sizes`] / [`relabel_consecutive`] — label bookkeeping
//! - [`filter_labels_by_min_voxels`] / [`closure_size_filter_sizes`] — voxel threshold
//! - [`parse_closure_codes`] / [`assign_fluid_types`] — HC code + fluid masks
//! - [`get_top_of_closure`] — trap crest mask along depth
//! - [`bbox_for_label_and_fault`] — padded AABB helper
//! - [`flood_fill_heap_2d`] — priority-flood depression fill (border-edge core)
//!
//! Golden fixtures: `tests/fixtures/closure_cubes_8.json`
//! (regenerate via `tests/fixtures/generate_closure_cubes.py`).

mod kernels;
#[cfg(test)]
#[path = "tests_kernels.rs"]
mod tests_kernels;

pub use kernels::{
    assign_fluid_types, bincount_label_sizes, bbox_for_label_and_fault,
    closure_size_filter_sizes, filter_labels_by_min_voxels, flood_fill_heap_2d,
    get_top_of_closure, parse_closure_codes, relabel_consecutive,
};
