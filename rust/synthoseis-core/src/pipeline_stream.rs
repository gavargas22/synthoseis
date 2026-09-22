//! Memory-bounded chunked e2e: fuse elastic + RFC + wavelet per spatial tile.
//!
//! # Working-set invariant
//! Horizon maps are O(ni×nj). Labels stay as the u8 deliverable (1 B/voxel).
//! Elastic (vp/vs/rho), RFC, and angle **temps** are allocated only at
//! tile/trace scale — never three full elastic volumes + full RFC + full stack
//! at once.
//!
//! Chunk keys are `(i_chunk, j_chunk, k_chunk)` over `[ci, cj, ck]` so a later
//! multi-worker strip partition can own contiguous inline ranges without reshape.

include!("pipeline_stream_a.rs");
include!("pipeline_stream_b.rs");
