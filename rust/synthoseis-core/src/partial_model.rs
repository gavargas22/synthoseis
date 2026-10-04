//! Partial voxels in the rock-physics model and the fuse (spec
//! partial-voxels §3.3–§3.6, PR B1). On by default since PR B2 (library and
//! CLI); with [`crate::partial_voxels::PartialVoxelConfig::legacy_whole_voxels`]
//! nothing here runs and every output is byte-identical to whole voxels
//! (the d8b96e69 default).
//!
//! * [`PartialModel`] rides on [`RpmModel::partial`]: the continuous horizon
//!   offsets `δ = z − round(z)` (f32 per horizon per column, spec §3.1), the
//!   continuous salt hull bounds, the fault model (for the pull-back), and
//!   the end-member layers of intervals that no cell is labelled with.
//! * [`RpmModel::column_partial`] runs the whole-voxel column, then
//!   recomputes only the mixed cells: Backus voxel properties (cell mode,
//!   and the property tiles every axis uses) and the per-cell sub-layers the
//!   time fuse needs (sub-cell mode, and the slowness-sum T of spec §1.4).
//!   Pure cells, and mixed cells whose labelled unit has no fraction (label
//!   guard, counted), keep the whole-voxel bits.
//! * Fractions are per column: chunk-, worker- and process-invariant. The
//!   fault pull-back uses the fault tile's per-column source lookup, which is
//!   tiling-invariant too.

use std::sync::Arc;

use synthoseis_geo::faults::{FaultModel, FaultTile};
use synthoseis_rpm::{
    backus_mix, voxel_properties, Elastic32, Fluid, LayerShifts, VoxelKind, SALT, WATER,
};
use synthoseis_seismic::{subcell_column, SubLayer, SubcellColumn};

use crate::partial_voxels::{
    absorb_water_below, cell_model, column_parts, column_parts_faulted, source_intervals,
    CellModel, ColumnGeometry, ColumnParts, PartKind, PartialVoxelStats, PvReflectivity, Unit,
};
use crate::pipeline::{E2eConfig, TimeAxis};
use crate::rock_physics::{
    layer_shifts, net_to_gross_map, ColumnScratch, ElasticModel, LayerModel, RockPhysicsConfig,
    RpmModel,
};

/// Partial-voxel state of an [`RpmModel`] (see the module docs).
#[derive(Debug, Clone)]
pub struct PartialModel {
    pub reflectivity: PvReflectivity,
    /// `z_h − round(z_h)` per `(column, horizon)`, f32 in `[−½, ½]`.
    pub delta: Vec<f32>,
    /// Continuous hull bounds per column (empty without salt).
    pub salt_bounds: Vec<Option<(f64, f64)>>,
    /// Fault model for the pull-back (`None` without faults).
    pub faults: Option<Arc<FaultModel>>,
    /// Interval h → layer index: a label id, or `layers.len() + n` for the
    /// n-th entry of `hidden`.
    pub interval_layer: Vec<usize>,
    /// End-member layers of intervals with no labelled cell anywhere.
    pub hidden: Vec<LayerModel>,
    /// Shifts of the labelled layers followed by those of `hidden`.
    pub shifts: Vec<LayerShifts>,
}

impl PartialEq for PartialModel {
    fn eq(&self, o: &Self) -> bool {
        self.reflectivity == o.reflectivity
            && self.delta == o.delta
            && self.salt_bounds == o.salt_bounds
            && self.faults.is_some() == o.faults.is_some()
            && self.interval_layer == o.interval_layer
            && self.hidden == o.hidden
            && self.shifts == o.shifts
    }
}

impl PartialModel {
    /// Partial state for `model` (built by [`crate::rock_physics::elastic_model`]
    /// from `cfg`, whose effective rock physics is `rp`); `sand` are the
    /// per-interval sand flags.
    pub fn build(
        cfg: &E2eConfig,
        model: &RpmModel,
        rp: &RockPhysicsConfig,
        sand: &[bool],
        reflectivity: PvReflectivity,
    ) -> Self {
        let (cont, nh) = crate::pipeline_stream::toy_horizon_maps_continuous(cfg)
            .expect("partial voxels need the layered geometry");
        assert_eq!(nh, model.nh);
        let delta = cont
            .iter()
            .zip(&model.maps)
            .map(|(c, r)| (c - r) as f32)
            .collect();
        let salt_bounds = model.salt.as_ref().map_or(Vec::new(), |s| s.hull_bounds());
        let faults = crate::pipeline_stream::fault_model(cfg)
            .filter(|m| !m.is_empty())
            .map(Arc::new);
        let [ni, nj, _] = model.shape;
        let n_labels = model.layers.len();
        let mut hidden = Vec::new();
        let interval_layer = (0..nh.saturating_sub(1))
            .map(|h| match model.intervals.iter().position(|&x| x == h) {
                Some(l) => l,
                None => {
                    let sand = sand.get(h).copied().unwrap_or(false);
                    hidden.push(LayerModel {
                        interval: h,
                        sand,
                        shifts: layer_shifts(cfg.seed, rp, h),
                        ng: if sand {
                            net_to_gross_map(cfg.seed, &rp.net_to_gross, ni, nj, h)
                        } else {
                            Vec::new()
                        },
                        fluids: None,
                    });
                    n_labels + hidden.len() - 1
                }
            })
            .collect();
        let shifts = model
            .shifts
            .iter()
            .copied()
            .chain(hidden.iter().map(|l: &LayerModel| l.shifts))
            .collect();
        Self {
            reflectivity,
            delta,
            salt_bounds,
            faults,
            interval_layer,
            hidden,
            shifts,
        }
    }
}

/// Per-tile partial context: the fault tile of `[i0, i1) x [j0, j1)` for
/// the pull-back (`None` without faults).
#[derive(Debug)]
pub struct PartialTile {
    fault: Option<FaultTile>,
}

impl PartialTile {
    /// Source lookup of column `(i, j)`, `None` when unfaulted (identity).
    fn lookup(&self, i: usize, j: usize) -> Option<&[f32]> {
        let t = self.fault.as_ref()?;
        let o = t.col_offset(i, j);
        let col = &t.lookup[o..o + t.nk];
        col.iter()
            .enumerate()
            .any(|(k, &v)| v != k as f32)
            .then_some(col)
    }
}

/// Reusable per-column partial scratch (in [`ColumnScratch`]).
#[derive(Debug, Default, Clone)]
pub struct PartialScratch {
    parts: ColumnParts,
    horizons: Vec<f64>,
    contacts: Vec<f32>,
    src: Vec<(f64, f64)>,
    brk: Vec<bool>,
    mix: Vec<(f64, Elastic32)>,
    seen: Vec<bool>,
    labelled: Vec<bool>,
    /// Sub-layers of the column, cell k at `sub[sub_off[k]..sub_off[k + 1]]`
    /// (one full-cell layer with the voxel properties for whole cells).
    pub sub: Vec<SubLayer>,
    pub sub_off: Vec<usize>,
    /// Counters accumulated over every column this scratch has seen.
    pub stats: PartialVoxelStats,
}

impl PartialScratch {
    /// Cell-wise sub-layers of the last column.
    pub fn cells(&self) -> impl Iterator<Item = &[SubLayer]> {
        self.sub_off.windows(2).map(|w| &self.sub[w[0]..w[1]])
    }
}

impl RpmModel {
    /// Partial context of the tile `[i0, i1) x [j0, j1)`.
    pub fn partial_tile(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> PartialTile {
        let fault = self
            .partial
            .as_ref()
            .and_then(|p| p.faults.as_ref())
            .map(|m| m.compute_tile(i0, i1, j0, j1));
        PartialTile { fault }
    }

    /// Elastic properties of column `(i, j)` with partial voxels (see the
    /// module docs); falls back to [`RpmModel::column`] when the model has no
    /// partial state. Leaves the column's sub-layers in `scratch.partial`.
    #[allow(clippy::too_many_arguments)]
    pub fn column_partial(
        &self,
        i: usize,
        j: usize,
        col: &[u8],
        tile: &PartialTile,
        scratch: &mut ColumnScratch,
        rho: &mut [f32],
        vp: &mut [f32],
        vs: &mut [f32],
    ) {
        self.column(i, j, col, scratch, rho, vp, vs);
        let Some(pm) = self.partial.as_deref() else {
            return;
        };
        let nk = col.len();
        let nh = self.nh;
        let c = i * self.shape[1] + j;
        let ColumnScratch {
            depth,
            kinds,
            partial: ps,
        } = scratch;
        ps.horizons.clear();
        ps.horizons
            .extend((0..nh).map(|h| self.maps[c * nh + h] + pm.delta[c * nh + h] as f64));
        ps.contacts.clear();
        ps.contacts.extend(pm.interval_layer.iter().map(|&l| {
            self.layers
                .get(l)
                .and_then(|lm| lm.fluids.as_ref())
                .map_or(f32::NEG_INFINITY, |f| f.contact[c])
        }));
        let geom = ColumnGeometry {
            horizons: &ps.horizons,
            salt: pm.salt_bounds.get(c).copied().flatten(),
            contacts: &ps.contacts,
        };
        match tile.lookup(i, j) {
            None => column_parts(&geom, nk, &mut ps.parts),
            Some(lookup) => {
                ps.brk.clear();
                ps.brk.resize(nk + 1, false);
                for f in pm.faults.as_ref().map_or(&[][..], |m| m.faults()) {
                    let mut prev = f.geometry.inside(i, j, 0);
                    for k in 1..nk {
                        let cur = f.geometry.inside(i, j, k);
                        ps.brk[k] |= cur != prev;
                        prev = cur;
                    }
                }
                source_intervals(lookup, &mut ps.brk, &mut ps.src);
                column_parts_faulted(&geom, &ps.src, &mut ps.parts);
            }
        }
        if tile.lookup(i, j).is_some() {
            let first = kinds
                .iter()
                .position(|k| !matches!(k, VoxelKind::Water))
                .unwrap_or(nk);
            ps.stats.water_below += absorb_water_below(&mut ps.parts, first + 1);
        }
        ps.sub.clear();
        ps.sub_off.clear();
        ps.sub_off.push(0);
        ps.seen.clear();
        ps.seen.resize(nh, false);
        ps.labelled.clear();
        ps.labelled.resize(nh, false);
        for k in 0..nk {
            let parts = ps.parts.cell(k);
            // "Labelled" from the label cube (salt does not overwrite it).
            if let Some(&h) = self
                .intervals
                .get(col[k] as usize)
                .filter(|_| col[k] != 255)
            {
                ps.labelled[h] = true;
            }
            for p in parts {
                if let PartKind::Interval { h, .. } = p.kind {
                    ps.seen[h] = true;
                }
            }
            let unit = match kinds[k] {
                VoxelKind::Salt => Some(Unit::Salt),
                VoxelKind::Water => Some(Unit::Water),
                VoxelKind::Layer { layer, .. } => {
                    ps.stats.sediment += 1;
                    Some(Unit::Interval(self.intervals[layer]))
                }
                VoxelKind::Unfilled => None,
            };
            let mixed = match unit {
                None => {
                    ps.stats.cells += 1;
                    false
                }
                Some(_) if parts.len() > 1 && parts.iter().any(|p| p.kind == PartKind::Below) => {
                    ps.stats.cells += 1;
                    ps.stats.below += 1;
                    false
                }
                Some(u) => matches!(cell_model(parts, u, &mut ps.stats), CellModel::Mixed(_)),
            };
            if mixed {
                ps.mix.clear();
                for p in parts {
                    let e = self.part_properties(pm, p.kind, k, c, depth, kinds);
                    ps.mix.push((p.frac, e));
                    ps.sub.push(SubLayer {
                        frac: p.frac,
                        vp: e.vp,
                        vs: e.vs,
                        rho: e.rho,
                    });
                }
                let b = backus_mix(&ps.mix);
                rho[k] = b.rho;
                vp[k] = b.vp;
                vs[k] = b.vs;
            } else {
                ps.sub.push(SubLayer {
                    frac: 1.0,
                    vp: vp[k],
                    vs: vs[k],
                    rho: rho[k],
                });
            }
            ps.sub_off.push(ps.sub.len());
        }
        // Hidden intervals (spec §3.6, the probe's definition): a fraction in
        // this column, no labelled cell, and both horizons round to the same
        // sample. The last condition leaves out intervals that the cube
        // bottom truncates inside the last cell.
        let z = &ps.horizons;
        ps.stats.hidden_intervals += (0..nh.saturating_sub(1))
            .filter(|&h| ps.seen[h] && !ps.labelled[h] && z[h].round() == z[h + 1].round())
            .count() as u64;
    }

    /// End-member properties of one part of cell k (spec §3.4): water and
    /// salt constants; interval h with its NTG, fluid and shifts, at the
    /// depth the label-run logic gives cell k or an adjacent cell labelled
    /// h, else at the map TVDML `(round(z_{h+1}) − round(z_0))·dz`.
    fn part_properties(
        &self,
        pm: &PartialModel,
        kind: PartKind,
        k: usize,
        c: usize,
        depth: &[f32],
        kinds: &[VoxelKind],
    ) -> Elastic32 {
        let (h, hc) = match kind {
            PartKind::Water => return WATER,
            PartKind::Salt => return SALT,
            PartKind::Below => return Elastic32::default(),
            PartKind::Interval { h, hc } => (h, hc),
        };
        let l = pm.interval_layer[h];
        let lm = self
            .layers
            .get(l)
            .unwrap_or_else(|| &pm.hidden[l - self.layers.len()]);
        let (ng, fluid) = if lm.sand {
            let fluid = if hc {
                lm.fluids.as_ref().map_or(Fluid::Brine, |f| f.fluid[c])
            } else {
                Fluid::Brine
            };
            (lm.ng[c], fluid)
        } else {
            (0.0, Fluid::Brine)
        };
        let vk = VoxelKind::Layer {
            layer: l,
            ng,
            fluid,
        };
        let nk = kinds.len();
        let is_l = |kk: usize| matches!(kinds[kk], VoxelKind::Layer { layer, .. } if layer == l);
        let at = if is_l(k) {
            Some(k)
        } else if k > 0 && is_l(k - 1) {
            Some(k - 1)
        } else if k + 1 < nk && is_l(k + 1) {
            Some(k + 1)
        } else {
            None
        };
        match at {
            Some(kk) => voxel_properties(depth, kk, vk, &pm.shifts, self.mixing),
            None => {
                let z = &self.maps[c * self.nh..(c + 1) * self.nh];
                let tv = (z[h + 1] as f32 - z[0] as f32) * self.step;
                voxel_properties(&[tv], 0, vk, &pm.shifts, self.mixing)
            }
        }
    }

    /// Two-way times of the last [`RpmModel::column_partial`] column through
    /// the slowness sum (spec §1.4) and its interfaces, into `out`.
    pub fn partial_column_twt(
        &self,
        scratch: &ColumnScratch,
        axis: &TimeAxis,
        out: &mut SubcellColumn,
    ) {
        subcell_column(scratch.partial.cells(), axis.dz, out);
    }
}

/// One-line run summary of partial voxels (spec §3.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartialVoxelSummary {
    pub stats: PartialVoxelStats,
    pub reflectivity: PvReflectivity,
}

impl std::fmt::Display for PartialVoxelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = &self.stats;
        let pct = |a: u64, b: u64| {
            if b == 0 {
                0.0
            } else {
                100.0 * a as f64 / b as f64
            }
        };
        write!(
            f,
            "partial voxels: mixed={:.1}% of cells, >=3-unit={}, hidden-intervals={}, label-guard={} ({:.3}% of sediment cells), below-base={}, water-below-seabed={}, mode={}",
            pct(s.mixed, s.cells),
            s.multi,
            s.hidden_intervals,
            s.label_guard,
            pct(s.label_guard, s.sediment),
            s.below,
            s.water_below,
            self.reflectivity.as_str()
        )
    }
}

/// Partial-voxel counters of the whole run of `cfg` (`None` when partial
/// voxels are off): one pass over every column, deterministic (sums).
pub fn partial_voxel_summary(cfg: &E2eConfig) -> Option<PartialVoxelSummary> {
    crate::time_mode::run_summary(cfg).partial
}

/// One pass over every column of `model` (tiled by the fault tile) adding
/// the partial-voxel counters to `stats`; a no-op without partial state.
pub fn partial_stats_pass(
    cfg: &E2eConfig,
    model: &ElasticModel,
    labels: &[u8],
    shape: [usize; 3],
    stats: &mut PartialVoxelStats,
) {
    let ElasticModel::Rpm(m) = model else {
        return;
    };
    if m.partial.is_none() {
        return;
    }
    let [ni, nj, nk] = shape;
    let [ci, cj] = crate::pipeline_stream::fault_tile(cfg);
    let mut scratch = ColumnScratch::default();
    let (mut rho, mut vp, mut vs) = (vec![0.0f32; nk], vec![0.0f32; nk], vec![0.0f32; nk]);
    for i0 in (0..ni).step_by(ci) {
        let i1 = (i0 + ci).min(ni);
        for j0 in (0..nj).step_by(cj) {
            let j1 = (j0 + cj).min(nj);
            let tile = m.partial_tile(i0, i1, j0, j1);
            for i in i0..i1 {
                for j in j0..j1 {
                    let g = (i * nj + j) * nk;
                    m.column_partial(
                        i,
                        j,
                        &labels[g..g + nk],
                        &tile,
                        &mut scratch,
                        &mut rho,
                        &mut vp,
                        &mut vs,
                    );
                }
            }
        }
    }
    stats.add(&scratch.partial.stats);
}

/// Root MDIO attributes of a partial-voxel store (spec §4): `voxel_model`
/// `"partial-z"`, `partial_voxel_mixing` `"backus"` and
/// `partial_voxel_reflectivity`. Whole-voxel runs write none (bytes kept).
pub fn write_partial_voxel_attrs(
    store: &synthoseis_io::MdioStore,
    cfg: &E2eConfig,
) -> Result<(), String> {
    let Some(r) = cfg.effective_partial_voxels() else {
        return Ok(());
    };
    store
        .set_root_attrs(&[
            ("voxel_model", serde_json::json!("partial-z")),
            ("partial_voxel_mixing", serde_json::json!("backus")),
            ("partial_voxel_reflectivity", serde_json::json!(r.as_str())),
        ])
        .map_err(|e| e.to_string())
}
