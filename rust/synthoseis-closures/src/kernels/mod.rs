//! Closure / trap kernel modules.
mod flood;
mod geometry;
mod labels;

pub use flood::flood_fill_heap_2d;
pub use geometry::{bbox_for_label_and_fault, get_top_of_closure};
pub use labels::{
    assign_fluid_types, bincount_label_sizes, closure_size_filter_sizes,
    filter_labels_by_min_voxels, parse_closure_codes, relabel_consecutive,
};
