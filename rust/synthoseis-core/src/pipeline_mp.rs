//! Multi-process JobPartitionPlan e2e on a shared FS store.
//!
//! Callers must use non-overlapping partitions from
//! [`crate::JobPartitionPlan::from_config_chunk_aligned`]. File-system ownership
//! of distinct MDIO chunk keys is the lock (no shared Mutex).

use std::path::Path;

use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_seismic::ricker;

use crate::parity;
use crate::pipeline::{E2eConfig, E2eReport, TINY_DIGI};
use crate::pipeline_stream::{
    depth_trends, fault_model, generate_chunked, generate_fault_labels, generate_labels,
    resolve_chunk_shape, write_strip_partition, SeismicFilters, WorkingSetStats,
};

/// Default path for the JobPartitionPlan JSON written next to a multiprocess store.
pub fn multiprocess_plan_path(store_path: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.partition-plan.json", store_path.display()))
}

fn multiprocess_sidecar_dir(store_path: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.mp", store_path.display()))
}

fn worker_stats_path(store_path: &Path, worker_id: usize) -> std::path::PathBuf {
    multiprocess_sidecar_dir(store_path).join(format!("worker_{worker_id}.stats.json"))
}

fn worker_samples_path(store_path: &Path, worker_id: usize) -> std::path::PathBuf {
    multiprocess_sidecar_dir(store_path).join(format!("worker_{worker_id}.samples.f32"))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkerStatsSidecar {
    worker_id: usize,
    peak_temp_bytes: usize,
    tiles_processed: usize,
    chunk_shape: [usize; 3],
    volume_shape: [usize; 3],
    sample_count: usize,
}

/// Prepare a shared MDIO store + chunk-aligned [`JobPartitionPlan`] for multi-process e2e.
///
/// Generates labels once (determinism check), creates an empty MDIO with sub-volume
/// chunks, ensures the labels array exists, and writes the plan JSON next to the store.
/// Angle/label chunk bytes are written later by [`run_worker_partition`].
pub fn prepare_multiprocess_store(
    cfg: &E2eConfig,
    workers: usize,
    chunk_shape: Option<[usize; 3]>,
) -> Result<
    (
        std::path::PathBuf,
        crate::partition::JobPartitionPlan,
        WorkingSetStats,
    ),
    String,
> {
    let workers = workers.max(1);
    let path = cfg
        .store_path
        .clone()
        .ok_or_else(|| "prepare_multiprocess_store requires store_path".to_string())?;
    SeismicFilters::from_config(cfg)?;

    let mut cfg = cfg.clone();
    if let Some(c) = chunk_shape {
        cfg.chunk_shape = Some(c);
    }

    let (_labels, shape) = generate_labels(&cfg);
    let [ni, nj, nk] = shape;
    let chunks = resolve_chunk_shape(&cfg);
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
    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|e| format!("remove existing store: {e}"))?;
    }
    let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
    store.ensure_labels_array().map_err(|e| e.to_string())?;
    if cfg.faults.enabled() {
        store
            .ensure_fault_labels_array()
            .map_err(|e| e.to_string())?;
    }

    let plan_path = multiprocess_plan_path(&path);
    if let Some(parent) = plan_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    std::fs::write(&plan_path, plan.to_json().map_err(|e| e.to_string())?)
        .map_err(|e| format!("write plan: {e}"))?;

    let side = multiprocess_sidecar_dir(&path);
    if side.exists() {
        std::fs::remove_dir_all(&side).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&side).map_err(|e| e.to_string())?;

    let prep = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: shape,
        peak_temp_bytes: 0,
        tiles_processed: 0,
    };
    Ok((path, plan, prep))
}

/// Open an existing store and fuse-write only `plan.for_worker(worker_id)`.
///
/// Regenerates labels from `cfg.seed` (bit-identical to prepare). Writes a small
/// sidecar under `{store}.mp/` with stats + raw f32 samples for finalize.
pub fn run_worker_partition(
    cfg: &E2eConfig,
    plan: &crate::partition::JobPartitionPlan,
    worker_id: usize,
    store_path: &Path,
) -> Result<WorkingSetStats, String> {
    let part = plan
        .for_worker(worker_id)
        .ok_or_else(|| {
            format!(
                "worker_id {worker_id} out of range (workers={})",
                plan.worker_count
            )
        })?
        .clone();

    let (labels, shape) = generate_labels(cfg);
    let chunks = resolve_chunk_shape(cfg);
    let [_, _, nk] = shape;
    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);

    // Each worker rebuilds the (deterministic) fault model and evaluates only
    // its own tiles.
    let faults = fault_model(cfg);
    let filters = SeismicFilters::from_config(cfg)?;
    let (stats, samples) = write_strip_partition(
        store_path,
        &part,
        &labels,
        shape,
        chunks,
        &trends,
        &wavelet,
        faults.as_ref(),
        filters.as_ref(),
    )?;

    let side = multiprocess_sidecar_dir(store_path);
    std::fs::create_dir_all(&side).map_err(|e| e.to_string())?;
    let sidecar = WorkerStatsSidecar {
        worker_id,
        peak_temp_bytes: stats.peak_temp_bytes,
        tiles_processed: stats.tiles_processed,
        chunk_shape: stats.chunk_shape,
        volume_shape: stats.volume_shape,
        sample_count: samples.len(),
    };
    std::fs::write(
        worker_stats_path(store_path, worker_id),
        serde_json::to_string_pretty(&sidecar).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write worker stats: {e}"))?;

    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(worker_samples_path(store_path, worker_id), &bytes)
        .map_err(|e| format!("write worker samples: {e}"))?;

    Ok(stats)
}

/// Finalize a multi-process store: collect worker sidecars, finalize MDIO, parity vs chunked.
pub fn finalize_multiprocess_e2e(
    cfg: &E2eConfig,
    store_path: &Path,
) -> Result<(E2eReport, WorkingSetStats), String> {
    let chunks = resolve_chunk_shape(cfg);
    let shape = cfg.shape();
    let side = multiprocess_sidecar_dir(store_path);

    let mut stats = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: shape,
        ..WorkingSetStats::default()
    };
    let mut all_samples: Vec<f32> = Vec::new();

    if side.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&side)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".stats.json"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for ent in entries {
            let raw = std::fs::read_to_string(ent.path()).map_err(|e| e.to_string())?;
            let sc: WorkerStatsSidecar =
                serde_json::from_str(&raw).map_err(|e| format!("parse worker stats: {e}"))?;
            stats.peak_temp_bytes = stats.peak_temp_bytes.max(sc.peak_temp_bytes);
            stats.tiles_processed += sc.tiles_processed;
            let samples_path = worker_samples_path(store_path, sc.worker_id);
            if samples_path.is_file() {
                let bytes = std::fs::read(&samples_path).map_err(|e| e.to_string())?;
                if bytes.len() % 4 != 0 {
                    return Err(format!(
                        "worker {} samples length {} not multiple of 4",
                        sc.worker_id,
                        bytes.len()
                    ));
                }
                for chunk in bytes.chunks_exact(4) {
                    all_samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
        }
    }

    let store = MdioStore::open(store_path).map_err(|e| e.to_string())?;
    store
        .finalize_after_chunked_write(&all_samples)
        .map_err(|e| e.to_string())?;

    let (reference, _) = generate_chunked(cfg);
    let opened = MdioStore::open(store_path).map_err(|e| e.to_string())?;
    let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
    let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
    let parity = parity::compare_volumes(
        &reference.labels,
        &back_labels,
        &reference.angle_stack,
        &back_angles,
    );
    if let Some(reference) = generate_fault_labels(cfg) {
        let back = opened.read_fault_labels_u8().map_err(|e| e.to_string())?;
        if back != reference {
            return Err("multiprocess fault_labels diverged from single-pass reference".into());
        }
    }
    if !parity.passes_defaults() {
        return Err(format!(
            "multiprocess MDIO parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    Ok((
        E2eReport {
            volumes: reference,
            parity,
            store_path: Some(store_path.to_path_buf()),
            status: "ok-e2e-multiprocess",
        },
        stats,
    ))
}

/// In-process multi-process e2e (prepare → N× [`run_worker_partition`] → finalize).
///
/// For API/parity tests. Real OS-process fan-out lives in the CLI
/// (`--multiprocess` spawns child `--worker-id` processes).
pub fn run_e2e_multiprocess(
    cfg: &E2eConfig,
    workers: usize,
) -> Result<(E2eReport, WorkingSetStats), String> {
    let (store_path, plan, _prep) = prepare_multiprocess_store(cfg, workers, cfg.chunk_shape)?;
    for wid in 0..plan.worker_count {
        run_worker_partition(cfg, &plan, wid, &store_path)?;
    }
    finalize_multiprocess_e2e(cfg, &store_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_stream::generate_chunked;
    use tempfile::tempdir;

    #[test]
    fn multiprocess_prepare_workers_finalize_parity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mp.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            filters: Default::default(),
            seed: 42,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
            store_path: Some(path.clone()),
            chunk_shape: Some([2, 4, 8]),
        };
        let (report, stats) = run_e2e_multiprocess(&cfg, 4).expect("multiprocess");
        assert_eq!(report.status, "ok-e2e-multiprocess");
        assert!(report.parity.passes_defaults());
        assert!((report.parity.label_iou - 1.0).abs() < 1e-12);
        assert_eq!(report.parity.angle_mae, 0.0);
        assert!(stats.tiles_processed >= 4);
        assert!(multiprocess_plan_path(&path).is_file());

        let (single, _) = generate_chunked(&cfg);
        assert_eq!(report.volumes.labels, single.labels);
        assert_eq!(report.volumes.angle_stack, single.angle_stack);
    }
}
