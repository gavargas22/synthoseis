//! Depth-to-time conversion of the seismic chain (spec "depth-to-time
//! conversion" §3, PR B): the time-mode fuse, per-column two-way time,
//! label point sampling onto the output axis, output-domain label
//! generators for the MDIO read-backs, MDIO attributes and the column
//! summary. Every step is per column (execution class A): no lateral halo,
//! no new vertical halo, and T is rebuilt from each column's Vp wherever it
//! is needed (no volume-sized T cache).
//!
//! On the legacy axis (`--legacy-depth-as-time`) none of this runs and every
//! function here returns the depth-domain result unchanged.

use std::sync::Once;

use synthoseis_geo::faults::FaultTile;
use synthoseis_seismic::{
    reflectivity_time_column_with_twt, subcell_reflectivity, twt_column, SubcellColumn, TwtScratch, ZoeppritzForm,
};

use crate::partial_voxels::PvReflectivity;
use crate::pipeline::{E2eConfig, TimeAxis};
use crate::rock_physics::{elastic_model, ColumnScratch, ElasticModel, RpmModel};
use crate::salt::SaltBody;

static GPU_FALLBACK_LOG: Once = Once::new();

/// Fuse one tile in time mode from `(ti, tj, nz)` property buffers into
/// `(ti, tj, nt)` output (spec §3.1 steps 2–4 and 6): per trace, the
/// cumulative two-way time from Vp, Zoeppritz on the depth interfaces (same
/// kernel and form as the depth fuse), band-limited insertion at the
/// interface times, a cast to f32 (the raw reflectivity the noise path also
/// sees), then the wavelet in time (`f32 -> f64 convolve_same_1d -> f32`,
/// exactly like the depth fuse). An empty wavelet leaves the raw time
/// reflectivity.
///
/// The GPU fuse is not ported to time mode: with `--gpu` this runs on the
/// CPU and logs it once (spec §6).
#[allow(clippy::too_many_arguments)]
pub fn fuse_props_tile_time(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    nz: usize,
    axis: &TimeAxis,
    wavelet: &[f64],
    angle_deg: f64,
    form: ZoeppritzForm,
    tile_out: &mut [f32],
) {
    if synthoseis_gpu::prefer_gpu() {
        GPU_FALLBACK_LOG.call_once(|| {
            eprintln!("gpu: depth-to-time mode fuses on the CPU (time-mode WGSL kernel not ported; --legacy-depth-as-time keeps the GPU path)");
        });
    }
    let nt = axis.nt;
    let n_traces = vp.len() / nz;
    assert!(vp.len() == n_traces * nz && vs.len() == vp.len() && rho.len() == vp.len());
    assert!(tile_out.len() >= n_traces * nt);
    let mut scratch = TwtScratch::default();
    let mut x = vec![0.0f64; nt];
    let mut twt = vec![0.0f64; nz + 1];
    for t in 0..n_traces {
        let r = t * nz..(t + 1) * nz;
        column_twt(axis, &vp[r.clone()], &mut twt);
        reflectivity_time_column_with_twt(
            &vp[r.clone()],
            &vs[r.clone()],
            &rho[r],
            &twt,
            angle_deg,
            form,
            axis.dt_ms,
            axis.kernel,
            &mut scratch,
            &mut x,
        );
        finish_trace(&mut x, wavelet, &mut tile_out[t * nt..(t + 1) * nt]);
    }
}

/// Cast the raw time reflectivity `x` to f32 (as the depth fuse's rfc) and
/// apply the wavelet in time into `out` (`f32 -> f64 convolve_same_1d ->
/// f32`); an empty wavelet leaves the raw reflectivity.
fn finish_trace(x: &mut [f64], wavelet: &[f64], out: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v as f32 as f64;
    }
    if wavelet.is_empty() {
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            *o = v as f32;
        }
    } else {
        let conv = synthoseis_seismic::convolve_same_1d(x, wavelet);
        for (o, &c) in out.iter_mut().zip(conv.iter()) {
            *o = c as f32;
        }
    }
}

/// `true` when the time fuse of `model` takes the partial-voxel path: a
/// rock-physics model with partial state and the physical (not the
/// constant test-hook) velocity.
pub fn partial_time_path(model: &ElasticModel) -> Option<&RpmModel> {
    match model {
        ElasticModel::Rpm(m) if m.partial.is_some() && m.time.is_some_and(|a| a.constant_twt_vp.is_none()) => {
            Some(m)
        }
        _ => None,
    }
}

/// Partial-voxel time fuse of the tile `[i0, i1) x [j0, j1)` (spec §1.4,
/// §1.5) into `(ti, tj, nt)` `tile_out`: per column the parts and their
/// end-members ([`RpmModel::column_partial`]), T through the slowness sum,
/// then either every sub-cell interface at its exact time (`subcell`) or
/// the Backus voxels' cell-to-cell reflectivity at the same T (`cell`),
/// inserted with the windowed sinc; then the same f32 cast and wavelet as
/// [`fuse_props_tile_time`].
#[allow(clippy::too_many_arguments)]
pub fn fuse_tile_time_partial(
    m: &RpmModel,
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
) {
    if synthoseis_gpu::prefer_gpu() {
        GPU_FALLBACK_LOG.call_once(|| {
            eprintln!("gpu: depth-to-time mode fuses on the CPU (time-mode WGSL kernel not ported; --legacy-depth-as-time keeps the GPU path)");
        });
    }
    let axis = m.time.expect("time axis");
    let pm = m.partial.as_ref().expect("partial state");
    let [_, nj, nz] = shape;
    let nt = axis.nt;
    let tj = j1 - j0;
    let tile = m.partial_tile(i0, i1, j0, j1);
    let mut scratch = ColumnScratch::default();
    let (mut rho, mut vp, mut vs) = (vec![0.0f32; nz], vec![0.0f32; nz], vec![0.0f32; nz]);
    let mut col = SubcellColumn::default();
    let mut r = Vec::new();
    let mut ts = TwtScratch::default();
    let mut x = vec![0.0f64; nt];
    for i in i0..i1 {
        for j in j0..j1 {
            let g = (i * nj + j) * nz;
            m.column_partial(i, j, &labels[g..g + nz], &tile, &mut scratch, &mut rho, &mut vp, &mut vs);
            m.partial_column_twt(&scratch, &axis, &mut col);
            match pm.reflectivity {
                PvReflectivity::Subcell => {
                    subcell_reflectivity(&col, angle_deg, m.zoeppritz, axis.dt_ms, axis.kernel, &mut r, &mut x)
                }
                PvReflectivity::Cell => reflectivity_time_column_with_twt(
                    &vp,
                    &vs,
                    &rho,
                    &col.t_cells,
                    angle_deg,
                    m.zoeppritz,
                    axis.dt_ms,
                    axis.kernel,
                    &mut ts,
                    &mut x,
                ),
            }
            let t = (i - i0) * tj + (j - j0);
            finish_trace(&mut x, wavelet, &mut tile_out[t * nt..(t + 1) * nt]);
        }
    }
}

/// Two-way times `T_0 … T_nz` (ms) of one column from its Vp (`nz + 1`
/// entries in `out`), or from the constant test-hook velocity
/// ([`crate::TimeConfig::constant_twt_vp`]).
pub fn column_twt(axis: &TimeAxis, vp: &[f32], out: &mut [f64]) {
    match axis.constant_twt_vp {
        None => twt_column(vp, axis.dz, out),
        Some(v) => {
            assert_eq!(out.len(), vp.len() + 1);
            let step = 2000.0 * axis.dz / v;
            let mut t = 0.0f64;
            out[0] = 0.0;
            for o in out[1..].iter_mut() {
                t += step;
                *o = t;
            }
        }
    }
}

/// Two-way times `T_0 … T_nz` (ms) of every column of the tile
/// `[i0, i1) x [j0, j1)`, `nz + 1` values per column in tile order, from the
/// model's Vp (spec §1). Rebuilt per tile; never cached for the volume.
#[allow(clippy::too_many_arguments)]
pub fn tile_twt(
    model: &ElasticModel,
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    axis: &TimeAxis,
) -> Vec<f64> {
    let nz = shape[2];
    if let Some(m) = partial_time_path(model) {
        // Partial voxels: the slowness-sum T of the fuse (spec §1.4), so
        // labels, the noise seabed time and the summaries see the same T.
        let nj = shape[1];
        let tile = m.partial_tile(i0, i1, j0, j1);
        let mut scratch = ColumnScratch::default();
        let (mut rho, mut vp, mut vs) = (vec![0.0f32; nz], vec![0.0f32; nz], vec![0.0f32; nz]);
        let mut col = SubcellColumn::default();
        let mut t = Vec::with_capacity((i1 - i0) * (j1 - j0) * (nz + 1));
        for i in i0..i1 {
            for j in j0..j1 {
                let g = (i * nj + j) * nz;
                m.column_partial(i, j, &labels[g..g + nz], &tile, &mut scratch, &mut rho, &mut vp, &mut vs);
                m.partial_column_twt(&scratch, axis, &mut col);
                t.extend_from_slice(&col.t_cells);
            }
        }
        return t;
    }
    let n = (i1 - i0) * (j1 - j0) * nz;
    let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    model.tile_properties(labels, shape, i0, i1, j0, j1, &mut vp, &mut vs, &mut rho);
    let mut t = vec![0.0f64; (n / nz) * (nz + 1)];
    for (c, col) in vp.chunks_exact(nz).enumerate() {
        column_twt(axis, col, &mut t[c * (nz + 1)..(c + 1) * (nz + 1)]);
    }
    t
}

/// Point-sample `(cols, nz)` depth columns onto `(cols, nt)` time columns
/// with the per-column times `t` (`nz + 1` per column), spec §3.4:
/// `L_t[n] = L_z[k(n)]`, `k(n) = max{k : T_k ≤ t_n}`, short columns
/// forward-fill the last cell. The same `k(n)` serves every label cube, so
/// per-voxel invariants (fault ∧ salt = 0, 255 above the seabed) carry over.
pub fn resample_columns<L: Copy + Default>(depth: &[L], nz: usize, t: &[f64], nt: usize, dt_ms: f64) -> Vec<L> {
    let cols = depth.len() / nz;
    assert_eq!(t.len(), cols * (nz + 1));
    let mut out = vec![L::default(); cols * nt];
    for c in 0..cols {
        synthoseis_seismic::point_sample_labels(
            &depth[c * nz..(c + 1) * nz],
            &t[c * (nz + 1)..(c + 1) * (nz + 1)],
            dt_ms,
            &mut out[c * nt..(c + 1) * nt],
        );
    }
    out
}

/// Output-domain label cubes of one tile, `(ti, tj, nk_out)` each.
#[derive(Debug, Clone, Default)]
pub struct OutputLabelTile {
    pub labels: Vec<u8>,
    pub faults: Option<Vec<u8>>,
    pub salt: Option<Vec<u8>>,
    /// Samples per column (`nt` in time mode, else `nz`).
    pub nk: usize,
}

impl OutputLabelTile {
    /// Append column-major `[k0, k1)` slices of `cube` (one of this tile's
    /// cubes) to `out` in MDIO chunk order.
    pub fn chunk(&self, cube: &[u8], k0: usize, k1: usize, out: &mut Vec<u8>) {
        out.clear();
        for col in cube.chunks_exact(self.nk) {
            out.extend_from_slice(&col[k0..k1]);
        }
    }
}

/// The deliverable label cubes of the tile `[i0, i1) x [j0, j1)` in the
/// output domain: `labels` from the full depth label volume, `fault_labels`
/// from the tile's fault mask and `salt_labels` from the salt body, each
/// point-sampled onto the time axis through the same per-column `k(n)` in
/// time mode (spec §3.4, #34 review rule (a)); copied unchanged on the
/// legacy axis.
#[allow(clippy::too_many_arguments)]
pub fn output_label_tile(
    model: &ElasticModel,
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    fault_tile: Option<&FaultTile>,
    salt: Option<&SaltBody>,
) -> OutputLabelTile {
    let [_, nj, nz] = shape;
    let mut depth_labels = Vec::with_capacity((i1 - i0) * (j1 - j0) * nz);
    for i in i0..i1 {
        let g = (i * nj + j0) * nz;
        depth_labels.extend_from_slice(&labels[g..g + (j1 - j0) * nz]);
    }
    let depth_faults = fault_tile.map(|t| {
        debug_assert_eq!((t.i0, t.i1, t.j0, t.j1), (i0, i1, j0, j1));
        t.mask.clone()
    });
    let depth_salt = salt.map(|s| {
        let mut out = Vec::new();
        crate::salt::salt_chunk(s, i0, i1, j0, j1, 0, nz, &mut out);
        out
    });
    match model.time() {
        None => OutputLabelTile {
            labels: depth_labels,
            faults: depth_faults,
            salt: depth_salt,
            nk: nz,
        },
        Some(axis) => {
            let t = tile_twt(model, labels, shape, i0, i1, j0, j1, &axis);
            let rs = |d: &[u8]| resample_columns(d, nz, &t, axis.nt, axis.dt_ms);
            OutputLabelTile {
                labels: rs(&depth_labels),
                faults: depth_faults.as_deref().map(rs),
                salt: depth_salt.as_deref().map(rs),
                nk: axis.nt,
            }
        }
    }
}

/// Full-volume output-domain label cubes ([`generate_output_labels`]).
#[derive(Debug, Clone)]
pub struct OutputLabels {
    pub labels: Vec<u8>,
    pub faults: Option<Vec<u8>>,
    pub salt: Option<Vec<u8>>,
    /// Output shape `(ni, nj, nk_out)`.
    pub shape: [usize; 3],
}

/// Every deliverable label cube in the output domain, from the depth
/// labels `labels` (= [`crate::generate_labels`]`(cfg)`) and their model.
/// On the legacy axis these are exactly the depth cubes
/// ([`crate::generate_fault_labels`], [`crate::salt::generate_salt_labels`]).
/// Evaluated tile by tile; tiling-invariant (per column).
pub fn generate_output_labels(cfg: &E2eConfig, labels: &[u8], model: &ElasticModel) -> OutputLabels {
    let shape = cfg.shape();
    let oshape = cfg.output_shape();
    if model.time().is_none() {
        return OutputLabels {
            labels: labels.to_vec(),
            faults: crate::pipeline_stream::generate_fault_labels(cfg),
            salt: crate::salt::generate_salt_labels(cfg),
            shape: oshape,
        };
    }
    let [ni, nj, _] = shape;
    let nt = oshape[2];
    let faults = crate::pipeline_stream::fault_model(cfg);
    let [ci, cj] = crate::pipeline_stream::fault_tile(cfg);
    let fault_salt = crate::salt::fault_label_salt(cfg, model);
    let mut out = OutputLabels {
        labels: vec![0u8; ni * nj * nt],
        faults: faults.as_ref().map(|_| vec![0u8; ni * nj * nt]),
        salt: model.salt().map(|_| vec![0u8; ni * nj * nt]),
        shape: oshape,
    };
    for i0 in (0..ni).step_by(ci) {
        let i1 = (i0 + ci).min(ni);
        for j0 in (0..nj).step_by(cj) {
            let j1 = (j0 + cj).min(nj);
            // `fault AND NOT salt` in depth (#38), before the point
            // sampling: every cube then uses the same k(n), so the time
            // labels keep fault ∧ salt = 0 exactly (spec §3.4, §8).
            let mut ft = faults.as_ref().map(|m| m.compute_tile(i0, i1, j0, j1));
            if let (Some(t), Some(s)) = (ft.as_mut(), fault_salt) {
                crate::salt::mask_fault_tile_salt(t, s);
            }
            let tile = output_label_tile(model, labels, shape, i0, i1, j0, j1, ft.as_ref(), model.salt());
            let tj = j1 - j0;
            for di in 0..i1 - i0 {
                for dj in 0..tj {
                    let src = (di * tj + dj) * nt;
                    let dst = ((i0 + di) * nj + j0 + dj) * nt;
                    out.labels[dst..dst + nt].copy_from_slice(&tile.labels[src..src + nt]);
                    if let (Some(o), Some(t)) = (out.faults.as_mut(), tile.faults.as_ref()) {
                        o[dst..dst + nt].copy_from_slice(&t[src..src + nt]);
                    }
                    if let (Some(o), Some(t)) = (out.salt.as_mut(), tile.salt.as_ref()) {
                        o[dst..dst + nt].copy_from_slice(&t[src..src + nt]);
                    }
                }
            }
        }
    }
    out
}

/// `labels` in the output domain (`generate_labels` on the legacy axis).
pub fn generate_labels_output(cfg: &E2eConfig) -> Vec<u8> {
    let (labels, shape) = crate::pipeline_stream::generate_labels(cfg);
    if !cfg.time_enabled() {
        return labels;
    }
    let model = elastic_model(cfg, &labels, shape);
    generate_output_labels(cfg, &labels, &model).labels
}

/// `fault_labels` in the output domain (`None` without faults; equals
/// [`crate::generate_fault_labels`] on the legacy axis).
pub fn generate_fault_labels_output(cfg: &E2eConfig) -> Option<Vec<u8>> {
    if !cfg.time_enabled() {
        return crate::pipeline_stream::generate_fault_labels(cfg);
    }
    cfg.faults.enabled().then_some(())?;
    let (labels, shape) = crate::pipeline_stream::generate_labels(cfg);
    let model = elastic_model(cfg, &labels, shape);
    generate_output_labels(cfg, &labels, &model).faults
}

/// `salt_labels` in the output domain (`None` without salt; equals
/// [`crate::salt::generate_salt_labels`] on the legacy axis).
pub fn generate_salt_labels_output(cfg: &E2eConfig) -> Option<Vec<u8>> {
    if !cfg.time_enabled() {
        return crate::salt::generate_salt_labels(cfg);
    }
    cfg.effective_salt().then_some(())?;
    let (labels, shape) = crate::pipeline_stream::generate_labels(cfg);
    let model = elastic_model(cfg, &labels, shape);
    generate_output_labels(cfg, &labels, &model).salt
}

/// Root MDIO attributes of a time-mode store (spec §2): `time_conversion`
/// `"vp-twt"`, `depth_step_m` and `twt_kernel`. The legacy axis writes none.
pub fn write_time_attrs(store: &synthoseis_io::MdioStore, cfg: &E2eConfig) -> Result<(), String> {
    let Some(axis) = cfg.time_axis() else {
        return Ok(());
    };
    store
        .set_root_attrs(&[
            ("time_conversion", serde_json::json!("vp-twt")),
            ("depth_step_m", serde_json::json!(axis.dz)),
            ("twt_kernel", serde_json::json!(axis.kernel.as_str())),
        ])
        .map_err(|e| e.to_string())
}

/// Two-way time (ms) at a fractional depth-sample position `s` of a column
/// with cell times `t` (`nz + 1` values): linear interpolation of T (spec
/// §1, horizon times).
pub fn twt_at(t: &[f64], s: f64) -> f64 {
    let nz = t.len() - 1;
    let s = s.clamp(0.0, nz as f64);
    let k = (s.floor() as usize).min(nz.saturating_sub(1));
    t[k] + (s - k as f64) * (t[k + 1] - t[k])
}

/// Short / long column statistics of a time-mode run (spec §2 reporting).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimeColumnSummary {
    pub columns: usize,
    /// Columns with `T_nz < t_{nt-1}` (half-space below the model base).
    pub short: usize,
    /// Columns with `T_nz > t_{nt-1}` (truncated).
    pub long: usize,
    /// Worst shortfall `t_{nt-1} - T_nz` (ms, 0 when none).
    pub max_shortfall_ms: f64,
    /// Worst excess `T_nz - t_{nt-1}` (ms, 0 when none).
    pub max_excess_ms: f64,
    /// Range of `T_nz` over the columns (ms).
    pub base_twt_ms: [f64; 2],
}

/// Column summary for `cfg` (`None` on the legacy axis).
pub fn time_column_summary(cfg: &E2eConfig) -> Option<TimeColumnSummary> {
    let axis = cfg.time_axis()?;
    let (labels, shape) = crate::pipeline_stream::generate_labels(cfg);
    let model = elastic_model(cfg, &labels, shape);
    let [ni, nj, nz] = shape;
    let t_last = (axis.nt - 1) as f64 * axis.dt_ms;
    let mut s = TimeColumnSummary {
        base_twt_ms: [f64::INFINITY, f64::NEG_INFINITY],
        ..Default::default()
    };
    for i in 0..ni {
        let t = tile_twt(&model, &labels, shape, i, i + 1, 0, nj, &axis);
        for col in t.chunks_exact(nz + 1) {
            let base = col[nz];
            s.columns += 1;
            s.base_twt_ms = [s.base_twt_ms[0].min(base), s.base_twt_ms[1].max(base)];
            if base < t_last {
                s.short += 1;
                s.max_shortfall_ms = s.max_shortfall_ms.max(t_last - base);
            } else if base > t_last {
                s.long += 1;
                s.max_excess_ms = s.max_excess_ms.max(base - t_last);
            }
        }
    }
    Some(s)
}
