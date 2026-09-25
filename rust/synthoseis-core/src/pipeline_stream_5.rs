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

    let filters = SeismicFilters::from_config(cfg)?;
    let filters_ref = filters.as_ref();
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
    let faults = fault_model(cfg);
    if faults.is_some() {
        store.ensure_fault_labels_array().map_err(|e| e.to_string())?;
    }
    let faults_ref = faults.as_ref();

    let labels = std::sync::Arc::new(labels);
    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);

    // FS ownership is the lock: partitions from from_config_chunk_aligned never
    // share an MDIO chunk key. write_strip_partition takes no Mutex.
    let worker_results: Vec<Result<(WorkingSetStats, Vec<f32>), String>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(plan.partitions.len());
            for part in &plan.partitions {
                let part = part.clone();
                let labels = labels.clone();
                let wavelet = wavelet.clone();
                let trends = trends.clone();
                let store_path = path.clone();
                handles.push(scope.spawn(move || {
                    write_strip_partition(
                        &store_path,
                        &part,
                        &labels,
                        shape,
                        chunks,
                        &trends,
                        &wavelet,
                        faults_ref,
                        filters_ref,
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

    // Fault labels written by N workers must equal the tile-wise reference.
    if let Some(reference) = generate_fault_labels(cfg) {
        let back = opened.read_fault_labels_u8().map_err(|e| e.to_string())?;
        if back != reference {
            return Err("strip-stitch fault_labels diverged from single-pass reference".into());
        }
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
