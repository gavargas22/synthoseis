use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::faults::{
    sample_random_faults, FaultKinematics, FaultModel, FaultTile, HeaveParams, RandomFaultConfig,
    ReachMode, Seabed, TipParams,
};
use synthoseis_geo::fill_layer_labels;
use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_rpm::RpmExampleTrends;

use crate::parity;
use crate::pipeline::{E2eConfig, E2eReport, E2eVolumes, DEPTH_PER_SAMPLE};
use crate::rock_physics::{elastic_model, ElasticModel};

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
    if let Some(c) = crate::geometry_cache::GeometryCache::get(cfg) {
        return c.maps.clone();
    }
    match cfg.effective_geometry() {
        crate::toy_geometry::ToyGeometry::Planar => {
            crate::toy_geometry::planar_horizon_maps(cfg.seed, cfg.shape())
        }
        crate::toy_geometry::ToyGeometry::Layered => {
            toy_horizon_maps_from(cfg, &layered_base_maps(cfg))
        }
    }
}

/// Production and continuous layered maps before salt drag, from one
/// layered stack.
pub(crate) type BaseMaps = (
    (Vec<f64>, usize),
    (Vec<f64>, usize),
    Vec<crate::unconformity::UnconformityRecord>,
);

pub(crate) fn layered_base_maps(cfg: &E2eConfig) -> BaseMaps {
    crate::toy_geometry::layered_horizon_maps_both(
        cfg.seed,
        cfg.shape(),
        cfg.rock_physics.fold_config(),
        cfg.rock_physics.unconformity_config(),
    )
}

/// Unconformity records of `cfg` (shallowest first; empty without any, with
/// the planar geometry, or when every unconformity was skipped for lack of
/// room). From the geometry cache when it is built.
pub fn unconformity_records(
    cfg: &E2eConfig,
) -> std::sync::Arc<Vec<crate::unconformity::UnconformityRecord>> {
    if let Some(c) = crate::geometry_cache::GeometryCache::get(cfg) {
        return c.unconf.clone();
    }
    match cfg.effective_geometry() {
        crate::toy_geometry::ToyGeometry::Planar => Default::default(),
        crate::toy_geometry::ToyGeometry::Layered => {
            if cfg.rock_physics.unconformity_config().is_none() {
                return Default::default();
            }
            std::sync::Arc::new(layered_base_maps(cfg).2)
        }
    }
}

/// Whether any unconformity of `cfg` was applied (the run then writes
/// `data/unconformity_labels`).
pub fn unconformity_applied(cfg: &E2eConfig) -> bool {
    cfg.rock_physics.unconformity_config().is_some()
        && cfg.effective_geometry() == crate::toy_geometry::ToyGeometry::Layered
        && crate::unconformity::applied(&unconformity_records(cfg))
            .next()
            .is_some()
}

/// `interval_sand` of `cfg` with the shale seal of its unconformities
/// (spec §3.7). Identical to `interval_sand` without an applied one.
pub fn sealed_interval_sand(cfg: &E2eConfig, n: usize) -> Vec<bool> {
    let rp = &cfg.rock_physics;
    let mut sand = crate::lithology::interval_sand(
        cfg.effective_lithology(),
        cfg.seed,
        n,
        rp.sand_layer_fraction,
        rp.sand_layer_thickness,
    );
    if rp.unconformity_shale != crate::lithology::UnconformityShale::None
        && rp.unconformity_config().is_some()
    {
        let recs = unconformity_records(cfg);
        let surfaces: Vec<usize> = crate::unconformity::applied(&recs).map(|r| r.s).collect();
        crate::lithology::apply_unconformity_seal(&mut sand, &surfaces, rp.unconformity_shale);
    }
    sand
}

/// [`toy_horizon_maps`] of the layered geometry from its base maps.
pub(crate) fn toy_horizon_maps_from(cfg: &E2eConfig, base: &BaseMaps) -> (Vec<f64>, usize) {
    {
        {
            let (maps, nh) = base.0.clone();
            match crate::salt::salt_body_from_maps(cfg, &maps, nh) {
                None => (maps, nh),
                Some(body) => {
                    let [ni, nj, _] = cfg.shape();
                    let mut dragged = crate::salt::drag_horizon_maps(
                        &maps,
                        [ni, nj, nh],
                        &body,
                        cfg.rock_physics.salt_smooth_all_horizons,
                    );
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
/// by construction. Layered: [`crate::toy_geometry::layered_horizon_maps_continuous`].
/// With salt (spec salt-pv-far-field-maps §4.1): the continuous maps `z⁰`
/// plus the production drag offset `D` ([`crate::salt::drag_offsets`]; hit
/// test and lift counter on the rounded maps, as production), min-cascaded,
/// then each value clamped into the rounding cell of the production map.
/// Far from salt (`D = 0`) this equals the no-salt continuous maps bit for
/// bit. `RockPhysicsConfig::salt_legacy_pv_maps` restores the old rule (the
/// drag of the rounded maps without the final re-round). Either way
/// `round(continuous) == toy_horizon_maps(cfg)` exactly, so labels never
/// move. Used by the partial-voxel model (`partial_model`) only.
pub fn toy_horizon_maps_continuous(cfg: &E2eConfig) -> Option<(Vec<f64>, usize)> {
    if let Some(c) = crate::geometry_cache::GeometryCache::get(cfg) {
        return c.continuous.clone();
    }
    match cfg.effective_geometry() {
        crate::toy_geometry::ToyGeometry::Planar => None,
        crate::toy_geometry::ToyGeometry::Layered => {
            let base = layered_base_maps(cfg);
            let prod = toy_horizon_maps_from(cfg, &base);
            Some(toy_horizon_maps_continuous_from(cfg, &base, &prod.0))
        }
    }
}

/// [`toy_horizon_maps_continuous`] of the layered geometry from its base
/// maps and production maps.
pub(crate) fn toy_horizon_maps_continuous_from(
    cfg: &E2eConfig,
    base: &BaseMaps,
    prod: &[f64],
) -> (Vec<f64>, usize) {
    {
        {
            let (maps, nh) = (&base.0 .0, base.0 .1);
            match crate::salt::salt_body_from_maps(cfg, maps, nh) {
                None => base.1.clone(),
                Some(body) => {
                    let [ni, nj, _] = cfg.shape();
                    let smooth_all = cfg.rock_physics.salt_smooth_all_horizons;
                    if cfg.rock_physics.salt_legacy_pv_maps {
                        let mut dragged =
                            crate::salt::drag_horizon_maps(maps, [ni, nj, nh], &body, smooth_all);
                        crate::toy_geometry::min_cascade(&mut dragged, nh);
                        return (dragged, nh);
                    }
                    let d = crate::salt::drag_offsets(maps, [ni, nj, nh], &body, smooth_all);
                    let (mut z, nh0) = base.1.clone();
                    debug_assert_eq!(nh0, nh);
                    for (v, dv) in z.iter_mut().zip(&d) {
                        *v += *dv;
                    }
                    crate::toy_geometry::min_cascade(&mut z, nh);
                    for (v, p) in z.iter_mut().zip(prod) {
                        *v = crate::toy_geometry::clamp_to_round_cell(*v, *p);
                    }
                    // Order holds without a second cascade: each value sits
                    // in its own rounding cell and production is ordered.
                    debug_assert!(z
                        .chunks_exact(nh)
                        .all(|c| (2..nh).all(|h| c[h - 1] <= c[h])));
                    (z, nh)
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
    let model = FaultModel::resolve_with_mode(shape, &params, &seabed, seed, mode);
    let model = model.with_kinematics(fault_kinematics(cfg));
    if cfg.faults.legacy_kinematics || cfg.faults.legacy_profile {
        return Some(model);
    }
    // 7b: tip lines from their own sub-stream of the run seed.
    Some(model.with_tips(cfg.seed, &TipParams::default()))
}

/// Fault kinematics of `cfg`: legacy vertical lookup under
/// `--legacy-fault-kinematics`, else heave with the `--fault-min-dip` floor
/// and the geometry's bin / depth step (spec-fault-kinematics 7a).
pub fn fault_kinematics(cfg: &E2eConfig) -> FaultKinematics {
    if cfg.faults.legacy_kinematics {
        return FaultKinematics::Legacy;
    }
    let fold = &cfg.rock_physics.fold;
    FaultKinematics::Heave(HeaveParams {
        min_dip_deg: cfg.faults.min_dip_deg,
        dz_m: fold.dz_m,
        bin_m: fold.bin_m,
        ..HeaveParams::default()
    })
}

/// Resolved fault model for `cfg` (`None` when `cfg.faults.count == 0`).
///
/// Deterministic in `cfg` alone, so every worker/process can rebuild it and
/// evaluate its own tiles independently.
pub fn fault_model(cfg: &E2eConfig) -> Option<FaultModel> {
    if !cfg.faults.enabled() {
        return None;
    }
    if let Some(c) = crate::geometry_cache::GeometryCache::get(cfg) {
        // One model per run, with a bounded gather cache (the last few
        // tiles) shared by the labels, rock physics, time-mode labels and
        // partial voxels.
        let key = crate::geometry_cache::fault_key(cfg);
        let (k, m) = c.faults.get_or_init(|| {
            let m = build_fault_model(cfg, &c.maps.0, c.maps.1).map(|m| {
                if m.has_heave() {
                    m.with_gather_cache(fault_tile(cfg))
                } else {
                    m
                }
            });
            (key.clone(), m)
        });
        if *k == key {
            return m.clone();
        }
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
    let cached = crate::geometry_cache::GeometryCache::get(cfg)
        .is_some_and(|c| c.maps.0 == maps && c.maps.1 == nh);
    let model = if cached {
        fault_model(cfg)
    } else {
        build_fault_model(cfg, maps, nh)
    };
    if let Some(model) = model {
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

/// `unconformity_labels` volume for `cfg` in depth (`None` unless an
/// unconformity was applied): the voxel of each applied unconformity's
/// production-map boundary in every column, valued by its number, carried
/// through the fault displacement like the layer labels and cleared in salt
/// (spec §6.2). Evaluated tile by tile; a pure function of the column.
pub fn generate_unconformity_labels(cfg: &E2eConfig) -> Option<Vec<u8>> {
    let recs = unconformity_records(cfg);
    crate::unconformity::applied(&recs).next()?;
    let [ni, nj, nk] = cfg.shape();
    let (maps, nh) = toy_horizon_maps(cfg);
    let salt = crate::salt::salt_body(cfg);
    let mut out = vec![0u8; ni * nj * nk];
    let mut put = |t: &[u8], i0: usize, i1: usize, j0: usize, j1: usize| {
        let tj = j1 - j0;
        for i in i0..i1 {
            let src = ((i - i0) * tj) * nk;
            let dst = (i * nj + j0) * nk;
            out[dst..dst + tj * nk].copy_from_slice(&t[src..src + tj * nk]);
        }
    };
    match fault_model(cfg) {
        Some(model) => model.for_each_tile(fault_tile(cfg), |t| {
            let lab = crate::unconformity::label_tile(
                &recs,
                &maps,
                nh,
                [ni, nj, nk],
                t.i0,
                t.i1,
                t.j0,
                t.j1,
                Some(t),
                salt.as_ref(),
            );
            put(&lab, t.i0, t.i1, t.j0, t.j1);
        }),
        None => {
            let [ci, cj] = fault_tile(cfg);
            for i0 in (0..ni).step_by(ci.max(1)) {
                let i1 = (i0 + ci.max(1)).min(ni);
                for j0 in (0..nj).step_by(cj.max(1)) {
                    let j1 = (j0 + cj.max(1)).min(nj);
                    let lab = crate::unconformity::label_tile(
                        &recs,
                        &maps,
                        nh,
                        [ni, nj, nk],
                        i0,
                        i1,
                        j0,
                        j1,
                        None,
                        salt.as_ref(),
                    );
                    put(&lab, i0, i1, j0, j1);
                }
            }
        }
    }
    Some(out)
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
