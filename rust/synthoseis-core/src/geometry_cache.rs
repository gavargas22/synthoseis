//! Per-run cache of the toy horizon maps and the salt body (folds review,
//! timing should-fix). The maps are a pure function of the configuration;
//! before this cache they were rebuilt by every caller (54 times per run on
//! the seed-4 demo). A pipeline entry point builds them once with
//! [`crate::pipeline::E2eConfig::with_geometry_cache`] and the value travels
//! with the configuration it was built from. There is no global state: each
//! use re-checks a key of the geometry inputs, so a configuration changed
//! after caching (or a default one) recomputes, and results are bit-identical
//! with and without the cache.

use std::sync::Arc;

use crate::pipeline::E2eConfig;
use crate::salt::SaltBody;

pub(crate) struct Cached {
    key: String,
    pub maps: (Vec<f64>, usize),
    pub continuous: Option<(Vec<f64>, usize)>,
    pub salt: Option<SaltBody>,
    pub unconf: Arc<Vec<crate::unconformity::UnconformityRecord>>,
    /// Fault model of the run with its shared per-tile gather cache
    /// (heave only), keyed by [`fault_key`]; built on first use.
    pub faults: std::sync::OnceLock<(String, Option<synthoseis_geo::faults::FaultModel>)>,
    /// Salt drag offsets of the undragged maps (post-fault drag only),
    /// built on first use.
    pub drag: std::sync::OnceLock<Option<Vec<f64>>>,
}

/// Every input of the fault model besides the geometry key.
pub(crate) fn fault_key(cfg: &E2eConfig) -> String {
    format!(
        "{:?}|{:?}|{:?}",
        cfg.faults,
        cfg.rock_physics.fold,
        crate::pipeline_stream::fault_tile(cfg)
    )
}

/// Optional cached geometry of an [`E2eConfig`]; empty by default. It does
/// not take part in `Debug` output.
#[derive(Clone, Default)]
pub struct GeometryCache(pub(crate) Option<Arc<Cached>>);

impl std::fmt::Debug for GeometryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "GeometryCache(built)"
        } else {
            "GeometryCache(empty)"
        })
    }
}

/// Every input of the maps and the salt body.
pub(crate) fn key(cfg: &E2eConfig) -> String {
    format!(
        "{}|{:?}|{:?}|{:?}|{}",
        cfg.seed,
        cfg.shape(),
        cfg.effective_geometry(),
        cfg.rock_physics,
        cfg.salt_post_fault_drag()
    )
}

impl GeometryCache {
    pub(crate) fn get(cfg: &E2eConfig) -> Option<&Cached> {
        let c = cfg.maps_cache.0.as_deref()?;
        (c.key == key(cfg)).then_some(c)
    }

    pub(crate) fn build(cfg: &E2eConfig) -> Self {
        let mut plain = cfg.clone();
        plain.maps_cache = GeometryCache::default();
        let (maps, continuous, salt, unconf) = match plain.effective_geometry() {
            crate::toy_geometry::ToyGeometry::Planar => (
                crate::pipeline_stream::toy_horizon_maps(&plain),
                None,
                None,
                Vec::new(),
            ),
            crate::toy_geometry::ToyGeometry::Layered => {
                // One layered stack for all three.
                let base = crate::pipeline_stream::layered_base_maps(&plain);
                let maps = crate::pipeline_stream::toy_horizon_maps_from(&plain, &base);
                let cont = crate::pipeline_stream::toy_horizon_maps_continuous_from(
                    &plain, &base, &maps.0,
                );
                let salt = if plain.effective_salt() {
                    crate::salt::salt_body_from_maps(&plain, &base.0 .0, base.0 .1)
                } else {
                    None
                };
                (maps, Some(cont), salt, base.2)
            }
        };
        GeometryCache(Some(Arc::new(Cached {
            key: key(cfg),
            maps,
            continuous,
            salt,
            unconf: Arc::new(unconf),
            faults: std::sync::OnceLock::new(),
            drag: std::sync::OnceLock::new(),
        })))
    }
}
