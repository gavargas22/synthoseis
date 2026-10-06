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

    SeismicFilters::from_config(cfg)?;
    let (labels, shape) = generate_labels(cfg);
    let filters = SeismicFilters::resolve(cfg, &labels, shape)?;
    // Output shape: `(ni, nj, nt)` in time mode (labels are point-sampled
    // per tile onto the time axis), the depth shape on the legacy axis.
    let oshape = cfg.output_shape();
    let [ni, nj, nk] = oshape;
    let chunks = resolve_chunk_shape(cfg);
    let mut stats = WorkingSetStats {
        chunk_shape: chunks,
        volume_shape: oshape,
        ..WorkingSetStats::default()
    };

    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(chunks),
        digi: cfg.digi_ms(),
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
    crate::time_mode::write_time_attrs(&store, cfg)?;
    crate::partial_model::write_partial_voxel_attrs(&store, cfg)?;
    crate::rock_physics::write_closure_attrs(&store, cfg)?;
    store.ensure_labels_array().map_err(|e| e.to_string())?;
    let faults = fault_model(cfg);
    if faults.is_some() {
        store.ensure_fault_labels_array().map_err(|e| e.to_string())?;
    }
    let mut chunk_faults = Vec::new();

    let trends = elastic_model(cfg, &labels, shape);
    if trends.salt().is_some() {
        store.ensure_salt_labels_array().map_err(|e| e.to_string())?;
    }
    let mut chunk_salt = Vec::new();
    let wavelet = cfg.ricker();
    stats.observe(wavelet.len() * 8 + trends.model_bytes());

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
            fuse_tile_filtered(
                &labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                &trends,
                &wavelet,
                DEFAULT_INCIDENCE_DEG,
                filters.as_ref(),
                &mut tile_angles,
                &mut stats,
            );
            stats.tiles_processed += 1;
            let mut fault_tile = faults.as_ref().map(|m| m.compute_tile(i0, i1, j0, j1));
            let fault_salt = crate::salt::fault_label_salt(cfg, &trends);
            if let (Some(t), Some(s)) = (fault_tile.as_mut(), fault_salt) {
                crate::salt::mask_fault_tile_salt(t, s);
            }
            if let Some(t) = &fault_tile {
                stats.observe(t.lookup.capacity() * 4 + t.mask.capacity() * 2);
            }
            // Output-domain label cubes of this tile (time mode: point
            // sampled through the tile's own T; legacy: the depth values).
            let out_labels = crate::time_mode::output_label_tile(
                &trends,
                &labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                fault_tile.as_ref(),
                trends.salt(),
            );

            let mut k0 = 0usize;
            let mut k_chunk = 0usize;
            while k0 < nk {
                let k1 = (k0 + ck).min(nk);
                let tk = k1 - k0;
                let n = ti * tj * tk;
                chunk_angles.resize(n, 0.0);
                let mut bi = 0;
                for di in 0..ti {
                    for dj in 0..tj {
                        for dk in 0..tk {
                            let local = (di * tj + dj) * nk + (k0 + dk);
                            chunk_angles[bi] = tile_angles[local];
                            bi += 1;
                        }
                    }
                }
                out_labels.chunk(&out_labels.labels, k0, k1, &mut chunk_labels);
                store
                    .write_chunk([i_chunk, j_chunk, k_chunk], &chunk_angles)
                    .map_err(|e| e.to_string())?;
                store
                    .write_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk_labels)
                    .map_err(|e| e.to_string())?;
                if let Some(t) = &out_labels.faults {
                    out_labels.chunk(t, k0, k1, &mut chunk_faults);
                    store
                        .write_fault_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk_faults)
                        .map_err(|e| e.to_string())?;
                }
                if let Some(t) = &out_labels.salt {
                    out_labels.chunk(t, k0, k1, &mut chunk_salt);
                    store
                        .write_salt_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk_salt)
                        .map_err(|e| e.to_string())?;
                }
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
    if let Some(reference) = crate::time_mode::generate_fault_labels_output(cfg) {
        let back = opened.read_fault_labels_u8().map_err(|e| e.to_string())?;
        if back != reference {
            return Err("streaming fault_labels diverged from tile-wise reference".into());
        }
    }
    crate::salt::verify_salt_labels(&opened, cfg)
        .map_err(|e| format!("streaming salt_labels diverged from the salt body: {e}"))?;
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
