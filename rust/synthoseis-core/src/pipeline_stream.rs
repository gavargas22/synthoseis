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

use std::path::Path;

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::{
    enforce_nonnegative_thicknesses, eval_plane, fill_layer_labels, fit_plane_lsq,
};
use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_rpm::RpmExampleTrends;
use synthoseis_seismic::{convolve_same_1d, ricker, zoeppritz_pp};

use crate::parity;
use crate::pipeline::{
    E2eConfig, E2eReport, E2eVolumes, DEPTH_PER_SAMPLE, TINY_DIGI,
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
    fn observe(&mut self, bytes: usize) {
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

/// Resolve MDIO / generation chunk shape.
///
/// Prefer explicit `cfg.chunk_shape`. Otherwise pick a proper **sub-volume**
/// tile so e2e never writes `chunks == [ni,nj,nk]` when the grid allows.
pub fn resolve_chunk_shape(cfg: &E2eConfig) -> [usize; 3] {
    let [ni, nj, nk] = cfg.shape();
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
    let [ni, nj, nk] = cfg.shape();
    assert!(nk >= 2, "need at least 2 samples for reflectivity");

    let seed_f = cfg.seed as f64;
    let a0 = 0.05 + (seed_f % 7.0) * 0.01;
    let b0 = 0.03 + ((seed_f / 3.0) % 5.0) * 0.01;
    let c0 = 0.5;
    let a1 = a0 * 0.5;
    let b1 = b0 * 0.5;
    let c1 = (nk as f64) * 0.55;

    let pts0 = [
        [0.0, 0.0, c0],
        [1.0, 0.0, a0 + c0],
        [0.0, 1.0, b0 + c0],
    ];
    let pts1 = [
        [0.0, 0.0, c1],
        [1.0, 0.0, a1 + c1],
        [0.0, 1.0, b1 + c1],
    ];
    let [fa, fb, fc] = fit_plane_lsq(&pts0);
    let [ga, gb, gc] = fit_plane_lsq(&pts1);
    let z0 = eval_plane(ni, nj, fa, fb, fc);
    let z1 = eval_plane(ni, nj, ga, gb, gc);
    let z2 = vec![(nk as f64) - 0.5; ni * nj];

    let nh = 3usize;
    let mut maps = vec![0.0f64; ni * nj * nh];
    for n in 0..(ni * nj) {
        maps[n * nh] = z0[n];
        maps[n * nh + 1] = z1[n];
        maps[n * nh + 2] = z2[n];
    }
    enforce_nonnegative_thicknesses(&mut maps, [ni, nj, nh]);
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
    (labels, [ni, nj, nk])
}

fn depth_trends(nk: usize) -> [Vec<f64>; 9] {
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

#[inline]
fn props_f32(lab: u8, k: usize, trends: &[Vec<f64>; 9]) -> (f32, f32, f32) {
    let (vp, vs, rho) = match lab {
        0 => (trends[0][k], trends[1][k], trends[2][k]),
        1 => (trends[3][k], trends[4][k], trends[5][k]),
        _ => (trends[6][k], trends[7][k], trends[8][k]),
    };
    (vp as f32, vs as f32, rho as f32)
}

fn fuse_tile_into_volume(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let [_ni, nj, nk] = shape;
    let mut vp_tr = vec![0.0f32; nk];
    let mut vs_tr = vec![0.0f32; nk];
    let mut rho_tr = vec![0.0f32; nk];
    let mut rfc_tr = vec![0.0f32; nk];
    let mut trace_f64 = vec![0.0f64; nk];
    stats.observe(
        (vp_tr.capacity() + vs_tr.capacity() + rho_tr.capacity() + rfc_tr.capacity()) * 4
            + trace_f64.capacity() * 8,
    );

    for i in i0..i1 {
        for j in j0..j1 {
            for k in 0..nk {
                let idx = (i * nj + j) * nk + k;
                let (vp, vs, rho) = props_f32(labels[idx], k, trends);
                vp_tr[k] = vp;
                vs_tr[k] = vs;
                rho_tr[k] = rho;
            }
            for k in 0..(nk - 1) {
                rfc_tr[k] = zoeppritz_pp(
                    vp_tr[k] as f64,
                    vs_tr[k] as f64,
                    rho_tr[k] as f64,
                    vp_tr[k + 1] as f64,
                    vs_tr[k + 1] as f64,
                    rho_tr[k + 1] as f64,
                    15.0,
                );
            }
            rfc_tr[nk - 1] = 0.0;
            for k in 0..nk {
                trace_f64[k] = rfc_tr[k] as f64;
            }
            let conv = convolve_same_1d(&trace_f64, wavelet);
            let base = (i * nj + j) * nk;
            for k in 0..nk {
                angle_out[base + k] = conv[k] as f32;
            }
        }
    }
}

/// Generate labels + angle stack with fused per-tile elastic/RFC/wavelet.
///
/// Bit-identical to [`crate::pipeline::generate_tiny_cube`] for the same seed.
pub fn generate_chunked(cfg: &E2eConfig) -> (E2eVolumes, WorkingSetStats) {
    let (labels, shape) = generate_labels(cfg);
    let [ni, nj, nk] = shape;
    let chunk = resolve_chunk_shape(cfg);
    let mut stats = WorkingSetStats {
        chunk_shape: chunk,
        volume_shape: shape,
        ..WorkingSetStats::default()
    };

    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);
    stats.observe(wavelet.len() * 8 + 9 * nk * 8);

    let mut angle_stack = vec![0.0f32; ni * nj * nk];
    let ci = chunk[0];
    let cj = chunk[1];
    let mut i0 = 0;
    while i0 < ni {
        let i1 = (i0 + ci).min(ni);
        let mut j0 = 0;
        while j0 < nj {
            let j1 = (j0 + cj).min(nj);
            fuse_tile_into_volume(
                &labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                &trends,
                &wavelet,
                &mut angle_stack,
                &mut stats,
            );
            stats.tiles_processed += 1;
            j0 = j1;
        }
        i0 = i1;
    }

    (
        E2eVolumes {
            labels,
            angle_stack,
            shape,
        },
        stats,
    )
}

/// Write labels + angles using a **sub-volume** MDIO chunk shape.
pub fn write_e2e_mdio_chunked(
    path: &Path,
    cfg: &E2eConfig,
    volumes: &E2eVolumes,
) -> Result<[usize; 3], String> {
    let [ni, nj, nk] = volumes.shape;
    let chunks = resolve_chunk_shape(cfg);
    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(chunks),
        digi: TINY_DIGI,
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(path, &create).map_err(|e| e.to_string())?;
    store
        .write_volume(&volumes.angle_stack)
        .map_err(|e| e.to_string())?;
    store
        .write_labels_u8(&volumes.labels)
        .map_err(|e| e.to_string())?;
    Ok(chunks)
}

fn fuse_tile_local(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    tile_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let [_ni, nj, nk] = shape;
    let tj = j1 - j0;
    let mut vp_tr = vec![0.0f32; nk];
    let mut vs_tr = vec![0.0f32; nk];
    let mut rho_tr = vec![0.0f32; nk];
    let mut rfc_tr = vec![0.0f32; nk];
    let mut trace_f64 = vec![0.0f64; nk];
    stats.observe(
        (vp_tr.capacity() + vs_tr.capacity() + rho_tr.capacity() + rfc_tr.capacity()) * 4
            + trace_f64.capacity() * 8,
    );

    for (di, i) in (i0..i1).enumerate() {
        for (dj, j) in (j0..j1).enumerate() {
            for k in 0..nk {
                let idx = (i * nj + j) * nk + k;
                let (vp, vs, rho) = props_f32(labels[idx], k, trends);
                vp_tr[k] = vp;
                vs_tr[k] = vs;
                rho_tr[k] = rho;
            }
            for k in 0..(nk - 1) {
                rfc_tr[k] = zoeppritz_pp(
                    vp_tr[k] as f64,
                    vs_tr[k] as f64,
                    rho_tr[k] as f64,
                    vp_tr[k + 1] as f64,
                    vs_tr[k + 1] as f64,
                    rho_tr[k + 1] as f64,
                    15.0,
                );
            }
            rfc_tr[nk - 1] = 0.0;
            for k in 0..nk {
                trace_f64[k] = rfc_tr[k] as f64;
            }
            let conv = convolve_same_1d(&trace_f64, wavelet);
            for k in 0..nk {
                tile_out[(di * tj + dj) * nk + k] = conv[k] as f32;
            }
        }
    }
}

/// Stream fuse+write without a full angle deliverable buffer.
///
/// Labels remain a full u8 volume. Angle samples are fused per spatial tile
/// (full nk for the wavelet) and written via [`MdioStore::write_chunk`].
pub fn run_e2e_streaming(cfg: &E2eConfig) -> Result<(E2eReport, WorkingSetStats), String> {
    let path = cfg
        .store_path
        .as_ref()
        .ok_or_else(|| "run_e2e_streaming requires store_path".to_string())?
        .clone();

    let (labels, shape) = generate_labels(cfg);
    let [ni, nj, nk] = shape;
    let chunks = resolve_chunk_shape(cfg);
    let mut stats = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: shape,
        ..WorkingSetStats::default()
    };

    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(chunks),
        digi: TINY_DIGI,
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
    store.ensure_labels_array().map_err(|e| e.to_string())?;

    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);
    stats.observe(wavelet.len() * 8 + 9 * nk * 8);

    let [ci, cj, ck] = chunks;
    let mut tile_angles = vec![0.0f32; ci * cj * nk];
    let mut chunk_angles = Vec::new();
    let mut chunk_labels = Vec::new();
    let mut stats_samples: Vec<f32> = Vec::new();
    stats.observe(tile_angles.capacity() * 4);

    let mut i0 = 0usize;
    let mut i_chunk = 0usize;
    while i0 < ni {
        let i1 = (i0 + ci).min(ni);
        let ti = i1 - i0;
        let mut j0 = 0usize;
        let mut j_chunk = 0usize;
        while j0 < nj {
            let j1 = (j0 + cj).min(nj);
            let tj = j1 - j0;
            fuse_tile_local(
                &labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                &trends,
                &wavelet,
                &mut tile_angles,
                &mut stats,
            );
            stats.tiles_processed += 1;

            let mut k0 = 0usize;
            let mut k_chunk = 0usize;
            while k0 < nk {
                let k1 = (k0 + ck).min(nk);
                let tk = k1 - k0;
                let n = ti * tj * tk;
                chunk_angles.resize(n, 0.0);
                chunk_labels.resize(n, 0);
                let mut bi = 0;
                for di in 0..ti {
                    for dj in 0..tj {
                        for dk in 0..tk {
                            let local = (di * tj + dj) * nk + (k0 + dk);
                            chunk_angles[bi] = tile_angles[local];
                            let gi = i0 + di;
                            let gj = j0 + dj;
                            let gk = k0 + dk;
                            chunk_labels[bi] = labels[(gi * nj + gj) * nk + gk];
                            bi += 1;
                        }
                    }
                }
                store
                    .write_chunk([i_chunk, j_chunk, k_chunk], &chunk_angles)
                    .map_err(|e| e.to_string())?;
                store
                    .write_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk_labels)
                    .map_err(|e| e.to_string())?;
                stats_samples.extend_from_slice(&chunk_angles);
                k0 = k1;
                k_chunk += 1;
            }
            j0 = j1;
            j_chunk += 1;
        }
        i0 = i1;
        i_chunk += 1;
    }

    store
        .finalize_after_chunked_write(&stats_samples)
        .map_err(|e| e.to_string())?;

    let (second, _) = generate_chunked(cfg);
    let opened = MdioStore::open(&path).map_err(|e| e.to_string())?;
    let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
    let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
    let parity = parity::compare_volumes(
        &second.labels,
        &back_labels,
        &second.angle_stack,
        &back_angles,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "streaming MDIO parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    Ok((
        E2eReport {
            volumes: second,
            parity,
            store_path: Some(path),
            status: "ok-e2e-chunked",
        },
        stats,
    ))
}

/// Run chunked e2e: fused generate → optional MDIO (sub-volume chunks) → parity.
pub fn run_e2e_chunked(cfg: &E2eConfig) -> Result<(E2eReport, WorkingSetStats), String> {
    let (volumes, stats) = generate_chunked(cfg);
    let (second, _) = generate_chunked(cfg);
    let parity = parity::compare_volumes(
        &volumes.labels,
        &second.labels,
        &volumes.angle_stack,
        &second.angle_stack,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "chunked e2e self-parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    let mut store_path = None;
    if let Some(ref path) = cfg.store_path {
        write_e2e_mdio_chunked(path, cfg, &volumes)?;
        let opened = MdioStore::open(path).map_err(|e| e.to_string())?;
        let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
        let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
        let mdio_parity = parity::compare_volumes(
            &volumes.labels,
            &back_labels,
            &volumes.angle_stack,
            &back_angles,
        );
        if !mdio_parity.passes_defaults() {
            return Err(format!(
                "chunked e2e MDIO round-trip parity failed: {mdio_parity:?}"
            ));
        }
        store_path = Some(path.clone());
    }

    Ok((
        E2eReport {
            volumes,
            parity,
            store_path,
            status: "ok-e2e-chunked",
        },
        stats,
    ))
}


/// Strip-stitch multi-worker e2e on the chunked fused path.
///
/// - `workers <= 1`: delegates to [`run_e2e_streaming`] when `store_path` is set,
///   else [`run_e2e_chunked`].
/// - `workers > 1`: creates **one** shared MDIO store with sub-volume chunks;
///   each local worker owns a contiguous inline strip snapped to `chunk_i`,
///   fuse-generates its tiles, and `write_chunk` / `write_labels_chunk` into the
///   shared store (no overlapping chunk keys). After join: finalize + full-volume
///   parity vs a single-worker [`generate_chunked`] reference.
pub fn run_e2e_strip_stitched(
    cfg: &E2eConfig,
    workers: usize,
) -> Result<(E2eReport, WorkingSetStats), String> {
    let workers = workers.max(1);
    if workers <= 1 {
        return if cfg.store_path.is_some() {
            run_e2e_streaming(cfg)
        } else {
            run_e2e_chunked(cfg)
        };
    }

    let path = cfg
        .store_path
        .clone()
        .ok_or_else(|| "run_e2e_strip_stitched requires store_path when workers > 1".to_string())?;

    let (labels, shape) = generate_labels(cfg);
    let [ni, nj, nk] = shape;
    let chunks = resolve_chunk_shape(cfg);
    let [ci, _cj, _ck] = chunks;

    let run_cfg = crate::RunConfig {
        seed: cfg.seed,
        workers,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
    };
    let plan = crate::partition::JobPartitionPlan::from_config_chunk_aligned(&run_cfg, ci);

    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(chunks),
        digi: TINY_DIGI,
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
    store.ensure_labels_array().map_err(|e| e.to_string())?;

    let labels = std::sync::Arc::new(labels);
    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);

    // Track written chunk keys to prove no overlap across workers.
    let written: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<[usize; 3]>>> =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));

    let worker_results: Vec<Result<(WorkingSetStats, Vec<f32>), String>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(plan.partitions.len());
            for part in &plan.partitions {
                let part = part.clone();
                let labels = labels.clone();
                let wavelet = wavelet.clone();
                let trends = trends.clone();
                let store_path = path.clone();
                let written = written.clone();
                handles.push(scope.spawn(move || {
                    worker_strip_fuse_write(
                        &store_path,
                        &part,
                        &labels,
                        shape,
                        chunks,
                        &trends,
                        &wavelet,
                        &written,
                    )
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("strip worker thread"))
                .collect()
        });

    let mut stats = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: shape,
        ..WorkingSetStats::default()
    };
    let mut all_samples: Vec<f32> = Vec::new();
    for r in worker_results {
        let (ws, samples) = r?;
        stats.peak_temp_bytes = stats.peak_temp_bytes.max(ws.peak_temp_bytes);
        stats.tiles_processed += ws.tiles_processed;
        all_samples.extend_from_slice(&samples);
    }

    // Re-open for finalize (same root); workers only wrote chunk files.
    let store = MdioStore::open(&path).map_err(|e| e.to_string())?;
    store
        .finalize_after_chunked_write(&all_samples)
        .map_err(|e| e.to_string())?;

    let (reference, _) = generate_chunked(cfg);
    let opened = MdioStore::open(&path).map_err(|e| e.to_string())?;
    let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
    let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
    let parity = parity::compare_volumes(
        &reference.labels,
        &back_labels,
        &reference.angle_stack,
        &back_angles,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "strip-stitch MDIO parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    // Bit-identical labels vs reference (same generate_labels).
    if *labels != reference.labels {
        return Err("strip-stitch labels diverged from chunked reference".into());
    }

    Ok((
        E2eReport {
            volumes: reference,
            parity,
            store_path: Some(path),
            status: "ok-e2e-strip-stitch",
        },
        stats,
    ))
}

fn worker_strip_fuse_write(
    store_path: &Path,
    part: &crate::partition::JobPartition,
    labels: &[u8],
    shape: [usize; 3],
    chunks: [usize; 3],
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    written: &std::sync::Mutex<std::collections::BTreeSet<[usize; 3]>>,
) -> Result<(WorkingSetStats, Vec<f32>), String> {
    let [ni, nj, nk] = shape;
    let [ci, cj, ck] = chunks;
    let mut stats = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: shape,
        ..WorkingSetStats::default()
    };
    stats.observe(wavelet.len() * 8 + 9 * nk * 8);

    let Some(strip) = part.to_spatial_strip(nj) else {
        return Ok((stats, Vec::new()));
    };
    if strip.is_empty() {
        return Ok((stats, Vec::new()));
    }

    let store = MdioStore::open(store_path).map_err(|e| e.to_string())?;
    let mut tile_angles = vec![0.0f32; ci * cj * nk];
    let mut chunk_angles = Vec::new();
    let mut chunk_labels = Vec::new();
    let mut samples: Vec<f32> = Vec::new();
    stats.observe(tile_angles.capacity() * 4);

    let i_chunk_start = strip.i0 / ci.max(1);
    let mut i0 = strip.i0;
    let mut i_chunk = i_chunk_start;
    while i0 < strip.i1 {
        let i1 = (i0 + ci).min(strip.i1).min(ni);
        let ti = i1 - i0;
        let mut j0 = 0usize;
        let mut j_chunk = 0usize;
        while j0 < nj {
            let j1 = (j0 + cj).min(nj);
            let tj = j1 - j0;
            fuse_tile_local(
                labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                trends,
                wavelet,
                &mut tile_angles,
                &mut stats,
            );
            stats.tiles_processed += 1;

            let mut k0 = 0usize;
            let mut k_chunk = 0usize;
            while k0 < nk {
                let k1 = (k0 + ck).min(nk);
                let tk = k1 - k0;
                let n = ti * tj * tk;
                chunk_angles.resize(n, 0.0);
                chunk_labels.resize(n, 0);
                let mut bi = 0;
                for di in 0..ti {
                    for dj in 0..tj {
                        for dk in 0..tk {
                            let local = (di * tj + dj) * nk + (k0 + dk);
                            chunk_angles[bi] = tile_angles[local];
                            let gi = i0 + di;
                            let gj = j0 + dj;
                            let gk = k0 + dk;
                            chunk_labels[bi] = labels[(gi * nj + gj) * nk + gk];
                            bi += 1;
                        }
                    }
                }
                let key = [i_chunk, j_chunk, k_chunk];
                {
                    let mut guard = written.lock().map_err(|e| e.to_string())?;
                    if !guard.insert(key) {
                        return Err(format!("overlapping chunk write at {key:?}"));
                    }
                }
                store
                    .write_chunk(key, &chunk_angles)
                    .map_err(|e| e.to_string())?;
                store
                    .write_labels_chunk(key, &chunk_labels)
                    .map_err(|e| e.to_string())?;
                samples.extend_from_slice(&chunk_angles);
                k0 = k1;
                k_chunk += 1;
            }
            j0 = j1;
            j_chunk += 1;
        }
        i0 = i1;
        i_chunk += 1;
    }

    Ok((stats, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::generate_tiny_cube;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn chunked_matches_tiny_bit_identical() {
        let cfg = E2eConfig::tiny(42);
        let classic = generate_tiny_cube(&cfg);
        let (chunked, stats) = generate_chunked(&cfg);
        assert_eq!(classic.labels, chunked.labels);
        assert_eq!(classic.angle_stack, chunked.angle_stack);
        assert!(stats.tiles_processed >= 1);
    }

    #[test]
    fn chunked_mdio_uses_subvolume_chunks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunked.mdio");
        let cfg = E2eConfig {
            store_path: Some(path.clone()),
            chunk_shape: Some([4, 4, 8]),
            ..E2eConfig::tiny(7)
        };
        let (report, _stats) = run_e2e_chunked(&cfg).expect("chunked e2e");
        assert_eq!(report.status, "ok-e2e-chunked");
        let zarray = fs::read_to_string(path.join("data").join("chunked_012").join(".zarray"))
            .expect(".zarray");
        let compact: String = zarray.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            compact.contains("\"chunks\":[4,4,8]"),
            "chunks not sub-volume: {zarray}"
        );
        assert!(path.join("data").join("chunked_012").join("0.0.0").is_file());
        assert!(
            path.join("data").join("chunked_012").join("1.0.0").is_file()
                || path.join("data").join("chunked_012").join("0.1.0").is_file()
        );
    }

    #[test]
    fn larger_cube_chunked_parity_and_bound() {
        let cfg = E2eConfig {
            seed: 3,
            inline_count: 32,
            crossline_count: 32,
            samples: 64,
            store_path: None,
            chunk_shape: Some([8, 8, 64]),
        };
        let (a, stats) = generate_chunked(&cfg);
        let (b, _) = generate_chunked(&cfg);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.angle_stack, b.angle_stack);
        assert_eq!(a.shape, [32, 32, 64]);
        let n = 32 * 32 * 64;
        let one_elastic = n * 4;
        assert!(
            stats.peak_temp_bytes < one_elastic,
            "peak_temp_bytes {} should be << one elastic volume {}",
            stats.peak_temp_bytes,
            one_elastic
        );
        assert!(
            stats.is_bounded_by_chunk(64),
            "stats not bounded: {:?}",
            stats
        );
    }

    #[test]
    fn streaming_write_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.mdio");
        let cfg = E2eConfig {
            seed: 11,
            inline_count: 16,
            crossline_count: 16,
            samples: 32,
            store_path: Some(path.clone()),
            chunk_shape: Some([8, 8, 32]),
        };
        let (report, stats) = run_e2e_streaming(&cfg).expect("stream");
        assert_eq!(report.status, "ok-e2e-chunked");
        assert!(path.join("data").join("chunked_012").join("0.0.0").is_file());
        assert!(path.join("data").join("chunked_012").join("1.1.0").is_file());
        let n = 16 * 16 * 32;
        assert!(stats.peak_temp_bytes < n * 4);
    }

    #[test]
    fn default_chunks_are_subvolume_for_8() {
        assert_eq!(default_subvolume_chunks([8, 8, 8]), [4, 4, 8]);
        assert_eq!(default_subvolume_chunks([32, 32, 64]), [8, 8, 64]);
    }
    #[test]
    fn strip_stitch_workers4_parity_vs_single() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("strip.mdio");
        let cfg = E2eConfig {
            seed: 42,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
            store_path: Some(path.clone()),
            // chunk_i=2 → 4 i-chunks → one per worker
            chunk_shape: Some([2, 4, 8]),
        };
        let (report, stats) = run_e2e_strip_stitched(&cfg, 4).expect("strip-stitch");
        assert_eq!(report.status, "ok-e2e-strip-stitch");
        assert!(report.parity.passes_defaults());
        assert!((report.parity.label_iou - 1.0).abs() < 1e-12);
        assert_eq!(report.parity.angle_mae, 0.0);
        assert!(stats.tiles_processed >= 4);

        let (single, _) = generate_chunked(&cfg);
        assert_eq!(report.volumes.labels, single.labels);
        assert_eq!(report.volumes.angle_stack, single.angle_stack);
    }

    #[test]
    fn strip_stitch_16x16x32_workers4_near_parity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("strip16.mdio");
        let cfg = E2eConfig {
            seed: 7,
            inline_count: 16,
            crossline_count: 16,
            samples: 32,
            store_path: Some(path.clone()),
            chunk_shape: Some([4, 4, 32]),
        };
        let (report, _) = run_e2e_strip_stitched(&cfg, 4).expect("strip-stitch 16");
        assert_eq!(report.status, "ok-e2e-strip-stitch");
        assert!(report.parity.passes_defaults());
        let (single, _) = generate_chunked(&cfg);
        assert_eq!(report.volumes.labels, single.labels);
        assert_eq!(report.volumes.angle_stack, single.angle_stack);
    }

    #[test]
    fn strip_stitch_workers_one_matches_chunked() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("one.mdio");
        let cfg = E2eConfig {
            seed: 3,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
            store_path: Some(path),
            chunk_shape: Some([4, 4, 8]),
        };
        let (a, _) = run_e2e_strip_stitched(&cfg, 1).expect("w1");
        assert_eq!(a.status, "ok-e2e-chunked");
        assert!(a.parity.passes_defaults());
    }
}
