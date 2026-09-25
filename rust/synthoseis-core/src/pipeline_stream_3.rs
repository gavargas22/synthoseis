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

    let filters = SeismicFilters::from_config(cfg)?;
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
    let faults = fault_model(cfg);
    if faults.is_some() {
        store.ensure_fault_labels_array().map_err(|e| e.to_string())?;
    }
    let mut chunk_faults = Vec::new();

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
            let fault_tile = faults.as_ref().map(|m| m.compute_tile(i0, i1, j0, j1));
            if let Some(t) = &fault_tile {
                stats.observe(t.lookup.capacity() * 4 + t.mask.capacity() * 2);
            }

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
                if let Some(t) = &fault_tile {
                    fault_tile_chunk(t, k0, k1, &mut chunk_faults);
                    store
                        .write_fault_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk_faults)
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
    if let Some(reference) = generate_fault_labels(cfg) {
        let back = opened.read_fault_labels_u8().map_err(|e| e.to_string())?;
        if back != reference {
            return Err("streaming fault_labels diverged from tile-wise reference".into());
        }
    }
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
