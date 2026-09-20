//! Rock-physics model (RPM) depth-trend ports from `rockphysics/`.
//!
//! # First landed kernels
//! - [`polyval`] — numpy-compatible polynomial evaluation
//! - [`RpmExampleTrends`] — example shale / brine / oil / gas sand trends
//! - Tagilsk shale / brine / gas sand helpers (`tagilsk_*`)
//!
//! Golden fixtures: `tests/fixtures/rpm_trends.json`
//! (regenerate via `tests/fixtures/generate_seismic_kernels.py`).

mod kernels;
#[cfg(test)]
#[path = "tests_kernels.rs"]
mod tests_kernels;

pub use kernels::{
    polyval, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp, tagilsk_brine_sand_vs,
    tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho, tagilsk_shale_vp,
    tagilsk_shale_vs, RpmExampleTrends, TagilskTrends,
};
