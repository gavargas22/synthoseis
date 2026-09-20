//! Bounded RPM depth-trend kernels.

mod trends;

pub use trends::{
    polyval, tagilsk_brine_sand_rho, tagilsk_brine_sand_vp, tagilsk_brine_sand_vs,
    tagilsk_gas_sand_vp, tagilsk_gas_sand_vs, tagilsk_shale_rho, tagilsk_shale_vp,
    tagilsk_shale_vs, RpmExampleTrends, TagilskTrends,
};
