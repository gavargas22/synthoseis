use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::faults::{
    sample_random_faults, FaultModel, FaultTile, RandomFaultConfig, ReachMode, Seabed,
};
use synthoseis_geo::fill_layer_labels;
use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_rpm::RpmExampleTrends;

use crate::parity;
use crate::rock_physics::{elastic_model, ElasticModel};
use crate::pipeline::{
    E2eConfig, E2eReport, E2eVolumes, DEPTH_PER_SAMPLE,
};

/// Peak temporary buffer bytes during fused generation / streaming write.
#[derive(Debug, Clone, Default)]
pub struct WorkingSetStats {
    pub peak_temp_bytes: usize,
    pub chunk_shape: [usize; 3],
    pub volume_shape: [usize; 3],
    pub tiles_processed: usize,
}

impl WorkingSetStats {
    pub(crate) fn observe(&mut self, bytes: usize) {
        self.peak_temp_bytes = self.peak_temp_bytes.max(bytes);
    }

    /// Peak temps stay within a slack multiple of one chunk of scratch bytes.
    pub fn is_bounded_by_chunk(&self, slack: usize) -> bool {
        let [ci, cj, ck] = self.chunk_shape;
        let tile = ci.saturating_mul(cj).saturating_mul(ck).max(1);
        let budget = tile.saturating_mul(32).saturating_mul(slack.max(1));
        let nk = self.volume_shape[2];
        let floor = 9 * nk * 8 + 4096;
        self.peak_temp_bytes <= budget.max(floor)
    }
}

/// Call counter for [`generate_labels`] — used by geometry-once tests.
pub static GENERATE_LABELS_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Resolve MDIO / generation chunk shape.
///
/// Prefer explicit `cfg.chunk_shape`. Otherwise pick a proper **sub-volume**
/// tile so e2e never writes `chunks == [ni,nj,nk]` when the grid allows.
///
/// The k chunk is along the output axis (`nt` in time mode).
pub fn resolve_chunk_shape(cfg: &E2eConfig) -> [usize; 3] {
    let [ni, nj, nk] = cfg.output_shape();
    if let Some(c) = cfg.chunk_shape {
        return [
            c[0].clamp(1, ni.max(1)),
            c[1].clamp(1, nj.max(1)),
            c[2].clamp(1, nk.max(1)),
        ];
    }
    default_subvolume_chunks([ni, nj, nk])
}

/// Default sub-volume chunks: `min(8, max(1, n/2))` spatially × full samples.
pub fn default_subvolume_chunks(shape: [usize; 3]) -> [usize; 3] {
    let [ni, nj, nk] = shape;
    let ci = if ni <= 1 {
        1
    } else {
        (ni / 2).clamp(1, 8).min(ni)
    };
    let cj = if nj <= 1 {
        1
    } else {
        (nj / 2).clamp(1, 8).min(nj)
    };
    [ci, cj, nk.max(1)]
}

/// Label generation — bit-identical to the geo+closures section of
/// [`crate::pipeline::generate_tiny_cube`].
pub fn generate_labels(cfg: &E2eConfig) -> (Vec<u8>, [usize; 3]) {
    GENERATE_LABELS_CALLS.fetch_add(1, Ordering::SeqCst);
    let [ni, nj, nk] = cfg.shape();
    assert!(nk >= 2, "need at least 2 samples for reflectivity");

    let (maps, nh) = toy_horizon_maps(cfg);
    let mut labels = fill_layer_labels(&maps, [ni, nj, nh], nk);

    let as_i32: Vec<i32> = labels
        .iter()
        .map(|&v| if v == 255 { 0 } else { (v as i32) + 1 })
        .collect();
    let (_vals, relabeled) = relabel_consecutive(&as_i32);
    let filtered = filter_labels_by_min_voxels(&relabeled, 1);
    for (dst, &src) in labels.iter_mut().zip(filtered.iter()) {
        *dst = if src == 0 {
            255
        } else {
            (src - 1).clamp(0, 254) as u8
        };
    }
    apply_faults_to_labels(cfg, &maps, nh, &mut labels);
    (labels, [ni, nj, nk])
}

/// Toy horizon stack `(ni, nj, nh)` shared by label generation, the fault
/// model's seabed (top horizon) and the rock-physics depth model; see
/// [`crate::toy_geometry`]. `--legacy-toy-depth` always uses the planar
/// master geometry.
pub(crate) fn toy_horizon_maps(cfg: &E2eConfig) -> (Vec<f64>, usize) {
    match cfg.effective_geometry() {
        crate::toy_geometry::ToyGeometry::Planar => {
            crate::toy_geometry::planar_horizon_maps(cfg.seed, cfg.shape())
        }
        crate::toy_geometry::ToyGeometry::Layered => {
            let (maps, nh) = crate::toy_geometry::layered_horizon_maps(cfg.seed, cfg.shape());
            match crate::salt::salt_body_from_maps(cfg, &maps, nh) {
                None => (maps, nh),
                Some(body) => {
                    let [ni, nj, _] = cfg.shape();
                    let mut dragged = crate::salt::drag_horizon_maps(&maps, [ni, nj, nh], &body, cfg.rock_physics.salt_smooth_all_horizons);
                    // Whole samples, as the layered geometry (see
                    // `layered_horizon_maps`); rounding keeps the order.
                    for v in &mut dragged {
                        *v = v.round();
                    }
                    synthoseis_geo::enforce_nonnegative_thicknesses(&mut dragged, [ni, nj, nh]);
                    (dragged, nh)
                }
            }
        }
    }
}

/// Continuous (unrounded) counterpart of [`toy_horizon_maps`] for partial
/// voxels (spec §3.1); `None` for the planar geometry, which is whole-voxel
/// by construction. Layered: [`crate::toy_geometry::layered_horizon_maps_continuous`];
/// with salt, the drag of the *rounded* maps without the final re-round (the
/// drag is not re-derived from continuous maps, which would move labels).
/// `round(continuous) == toy_horizon_maps(cfg)` exactly. Used by the
/// partial-voxel model (`partial_model`) only.
pub fn toy_horizon_maps_continuous(cfg: &E2eConfig) -> Option<(Vec<f64>, usize)> {
    match cfg.effective_geometry() {
        crate::toy_geometry::ToyGeometry::Planar => None,
        crate::toy_geometry::ToyGeometry::Layered => {
            let (maps, nh) = crate::toy_geometry::layered_horizon_maps(cfg.seed, cfg.shape());
            match crate::salt::salt_body_from_maps(cfg, &maps, nh) {
                None => Some(crate::toy_geometry::layered_horizon_maps_continuous(cfg.seed, cfg.shape())),
                Some(body) => {
                    let [ni, nj, _] = cfg.shape();
                    let mut dragged = crate::salt::drag_horizon_maps(&maps, [ni, nj, nh], &body, cfg.rock_physics.salt_smooth_all_horizons);
                    crate::toy_geometry::min_cascade(&mut dragged, nh);
                    Some((dragged, nh))
                }
            }
        }
    }
}

/// Seed stream for fault parameter draws (independent of geology draws).
const FAULT_SEED_SALT: u64 = 0xFA17_0000_0000_0001;

/// Build the fault model for `cfg` from the horizon stack (`None` when
/// faulting is disabled). The top horizon acts as the seabed.
fn build_fault_model(cfg: &E2eConfig, maps: &[f64], nh: usize) -> Option<FaultModel> {
    if !cfg.faults.enabled() {
        return None;
    }
    let shape = cfg.shape();
    let seed = cfg.seed ^ FAULT_SEED_SALT;
    let params = sample_random_faults(
        shape,
        &RandomFaultConfig {
            count: cfg.faults.count,
            throw_min: cfg.faults.throw_min,
            throw_max: cfg.faults.throw_max,
        },
        seed,
    );
    let seabed = Seabed::Map(maps.iter().step_by(nh).copied().collect());
    let mode = if cfg.faults.legacy_reach {
        ReachMode::Legacy
    } else {
        ReachMode::FitColumn
    };
    Some(FaultModel::resolve_with_mode(shape, &params, &seabed, seed, mode))
}

/// Resolved fault model for `cfg` (`None` when `cfg.faults.count == 0`).
///
/// Deterministic in `cfg` alone, so every worker/process can rebuild it and
/// evaluate its own tiles independently.
pub fn fault_model(cfg: &E2eConfig) -> Option<FaultModel> {
    if !cfg.faults.enabled() {
        return None;
    }
    let (maps, nh) = toy_horizon_maps(cfg);
    build_fault_model(cfg, &maps, nh)
}

/// Seabed map `(ni, nj)` (top toy horizon, in samples) that the fault model
/// tapers against and, in the default reach mode, clamps fault labels below.
pub fn fault_seabed(cfg: &E2eConfig) -> Vec<f64> {
    let (maps, nh) = toy_horizon_maps(cfg);
    maps.iter().step_by(nh).copied().collect()
}

/// Spatial tile used to evaluate faults (MDIO chunk footprint). Results are
/// tiling-invariant; this only bounds scratch memory.
pub fn fault_tile(cfg: &E2eConfig) -> [usize; 2] {
    let c = resolve_chunk_shape(cfg);
    [c[0], c[1]]
}

/// Apply the optional fault model to generated labels in place (no-op when
/// disabled, so fault-free output stays bit-identical).
pub(crate) fn apply_faults_to_labels(cfg: &E2eConfig, maps: &[f64], nh: usize, labels: &mut [u8]) {
    if let Some(model) = build_fault_model(cfg, maps, nh) {
        let _ = model.apply_to_labels(labels, fault_tile(cfg));
    }
}

/// Binary fault-label volume for `cfg` (`None` when faulting is disabled).
/// Evaluated tile by tile.
pub fn generate_fault_labels(cfg: &E2eConfig) -> Option<Vec<u8>> {
    let model = fault_model(cfg)?;
    let [ni, nj, nk] = cfg.shape();
    // `fault AND NOT salt` (see [`crate::salt::mask_fault_tile_salt`]).
    let salt = if cfg.effective_fault_salt_mask() {
        crate::salt::salt_body(cfg)
    } else {
        None
    };
    let mut mask = vec![0u8; ni * nj * nk];
    model.for_each_tile(fault_tile(cfg), |t| {
        for i in t.i0..t.i1 {
            for j in t.j0..t.j1 {
                let g = (i * nj + j) * nk;
                let l = t.col_offset(i, j);
                mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                if let Some(s) = &salt {
                    let (a, b) = s.runs[i * nj + j];
                    let (a, b) = ((a as usize).min(nk), (b as usize).min(nk));
                    if a < b {
                        mask[g + a..g + b].fill(0);
                    }
                }
            }
        }
    });
    Some(mask)
}

/// Copy one MDIO chunk `[k0, k1)` of a fault tile's mask into `out`.
#[allow(dead_code)]
pub(crate) fn fault_tile_chunk(t: &FaultTile, k0: usize, k1: usize, out: &mut Vec<u8>) {
    out.clear();
    for i in t.i0..t.i1 {
        for j in t.j0..t.j1 {
            let l = t.col_offset(i, j);
            out.extend_from_slice(&t.mask[l + k0..l + k1]);
        }
    }
}

pub(crate) fn depth_trends(nk: usize) -> [Vec<f64>; 9] {
    let depths: Vec<f64> = (0..nk).map(|k| k as f64 * DEPTH_PER_SAMPLE).collect();
    [
        RpmExampleTrends::shale_vp(&depths),
        RpmExampleTrends::shale_vs(&depths),
        RpmExampleTrends::shale_rho(&depths),
        RpmExampleTrends::brine_sand_vp(&depths),
        RpmExampleTrends::brine_sand_vs(&depths),
        RpmExampleTrends::brine_sand_rho(&depths),
        RpmExampleTrends::oil_sand_vp(&depths),
        RpmExampleTrends::oil_sand_vs(&depths),
        RpmExampleTrends::oil_sand_rho(&depths),
    ]
}
