//! Single-worker compute/write overlap with one writer thread and one in-flight chunk.

use std::sync::mpsc::sync_channel;
use std::thread;

use synthoseis_io::{CreateConfig, Dimension, MdioStore};

use crate::parity;
use crate::pipeline::{E2eConfig, E2eReport};
use crate::pipeline_stream::{
    fault_model, fuse_tile_filtered, generate_chunked, generate_labels, resolve_chunk_shape,
    SeismicFilters, WorkingSetStats, DEFAULT_INCIDENCE_DEG,
};
use crate::rock_physics::elastic_model;

/// `(key, angles, labels, salt, faults)`. `salt` is `Some` in time mode (the
/// producer point-samples the output-domain salt labels with the tile's T);
/// on the legacy axis the writer rebuilds salt chunks from the body.
/// `faults` is `Some` in time mode with faulting on (output-domain
/// `fault_labels`); the legacy axis writes no `fault_labels` here, exactly
/// as master f3720fb2.
type WriteChunk = ([usize; 3], Vec<f32>, Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>);

/// Fuse tile N+1 on the caller while a dedicated std thread flushes tile N.
///
/// A rendezvous acknowledgement keeps at most one owned write chunk in flight;
/// the requested one-slot channel provides backpressure without an async runtime.
pub fn run_e2e_streaming_overlapped(
    cfg: &E2eConfig,
) -> Result<(E2eReport, WorkingSetStats), String> {
    let path = cfg
        .store_path
        .as_ref()
        .ok_or_else(|| "run_e2e_streaming_overlapped requires store_path".to_string())?
        .clone();
    SeismicFilters::from_config(cfg)?;
    let (labels, shape) = generate_labels(cfg);
    let filters = SeismicFilters::resolve(cfg, &labels, shape)?;
    let oshape = cfg.output_shape();
    let [ni, nj, nk] = oshape;
    let chunks = resolve_chunk_shape(cfg);
    let [ci, cj, ck] = chunks;
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
    {
        let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
        crate::time_mode::write_time_attrs(&store, cfg)?;
        crate::partial_model::write_partial_voxel_attrs(&store, cfg)?;
        store.ensure_labels_array().map_err(|e| e.to_string())?;
        if cfg.effective_salt() {
            store.ensure_salt_labels_array().map_err(|e| e.to_string())?;
        }
        if cfg.time_enabled() && cfg.faults.enabled() {
            store.ensure_fault_labels_array().map_err(|e| e.to_string())?;
        }
    }
    // Salt labels are rebuilt per chunk by the writer from the salt body.
    let salt = crate::salt::salt_body(cfg);
    let has_salt = salt.is_some();
    let time_mode = cfg.time_enabled();
    // Time mode: fault model for output-domain fault_labels (see WriteChunk).
    let faults = if time_mode { fault_model(cfg) } else { None };

    let (tx, rx) = sync_channel::<WriteChunk>(1);
    let (ack_tx, ack_rx) = sync_channel::<()>(0);
    let writer_path = path.clone();
    let writer = thread::spawn(move || -> Result<(), String> {
        let store = MdioStore::open(&writer_path).map_err(|e| e.to_string())?;
        let mut chunk_salt = Vec::new();
        for (key, angles, chunk_labels, out_salt, out_faults) in rx {
            store.write_chunk(key, &angles).map_err(|e| e.to_string())?;
            store
                .write_labels_chunk(key, &chunk_labels)
                .map_err(|e| e.to_string())?;
            if let Some(f) = out_faults {
                store
                    .write_fault_labels_chunk(key, &f)
                    .map_err(|e| e.to_string())?;
            }
            if let Some(t) = out_salt {
                store
                    .write_salt_labels_chunk(key, &t)
                    .map_err(|e| e.to_string())?;
            } else if let Some(s) = &salt {
                let [i0, j0, k0] = [key[0] * ci, key[1] * cj, key[2] * ck];
                let (i1, j1, k1) = ((i0 + ci).min(ni), (j0 + cj).min(nj), (k0 + ck).min(nk));
                crate::salt::salt_chunk(s, i0, i1, j0, j1, k0, k1, &mut chunk_salt);
                store
                    .write_salt_labels_chunk(key, &chunk_salt)
                    .map_err(|e| e.to_string())?;
            }
            ack_tx
                .send(())
                .map_err(|_| "overlap producer dropped acknowledgement channel".to_string())?;
        }
        Ok(())
    });

    let trends = elastic_model(cfg, &labels, shape);
    let wavelet = cfg.ricker();
    let fault_salt = crate::salt::fault_label_salt(cfg, &trends);
    let mut tile_angles = vec![0.0f32; ci * cj * nk];
    // Elastic model + trace scratch + wavelet (+ tile Vp/Vs/rho for the
    // rock-physics model) + tile output + exactly one writer-owned chunk.
    let fixed_bytes = trends.model_bytes()
        + 4 * nk * 4
        + nk * 8
        + wavelet.len() * 8
        + props_tile_bytes(&trends, ci * cj * nk);
    let tile_bytes = tile_angles.capacity() * 4;
    let in_flight_bytes = ci * cj * ck * (4 + 1 + has_salt as usize + faults.is_some() as usize);
    stats.peak_temp_bytes = fixed_bytes + tile_bytes + in_flight_bytes;

    let mut pending_write = false;
    let mut producer_error = None;
    'tiles: for (i_chunk, i0) in (0..ni).step_by(ci).enumerate() {
        let i1 = (i0 + ci).min(ni);
        let ti = i1 - i0;
        for (j_chunk, j0) in (0..nj).step_by(cj).enumerate() {
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
            // Time mode: output-domain labels/faults/salt for this tile.
            // Masked by the salt body (`fault AND NOT salt`, #38) before the
            // point sampling onto the time axis.
            let mut fault_tile = faults.as_ref().map(|m| m.compute_tile(i0, i1, j0, j1));
            if let (Some(t), Some(s)) = (fault_tile.as_mut(), fault_salt) {
                crate::salt::mask_fault_tile_salt(t, s);
            }
            if let Some(t) = &fault_tile {
                stats.observe(t.lookup.capacity() * 4 + t.mask.capacity() * 2);
            }
            let out_labels = time_mode.then(|| {
                crate::time_mode::output_label_tile(
                    &trends,
                    &labels,
                    shape,
                    i0,
                    i1,
                    j0,
                    j1,
                    fault_tile.as_ref(),
                    trends.salt(),
                )
            });

            for (k_chunk, k0) in (0..nk).step_by(ck).enumerate() {
                // Tile fusion above overlaps the previous flush. Wait only at handoff,
                // before allocating the next owned write message.
                if pending_write {
                    if ack_rx.recv().is_err() {
                        producer_error =
                            Some("overlap writer stopped before acknowledgement".to_string());
                        break 'tiles;
                    }
                    pending_write = false;
                }

                let k1 = (k0 + ck).min(nk);
                let tk = k1 - k0;
                let n = ti * tj * tk;
                let mut chunk_angles = Vec::with_capacity(n);
                let mut chunk_labels = Vec::with_capacity(n);
                let mut chunk_salt = None;
                let mut chunk_faults = None;
                if let Some(t) = &out_labels {
                    for di in 0..ti {
                        for dj in 0..tj {
                            for k in k0..k1 {
                                chunk_angles.push(tile_angles[(di * tj + dj) * nk + k]);
                            }
                        }
                    }
                    t.chunk(&t.labels, k0, k1, &mut chunk_labels);
                    if let Some(sc) = &t.salt {
                        let mut v = Vec::with_capacity(n);
                        t.chunk(sc, k0, k1, &mut v);
                        chunk_salt = Some(v);
                    }
                    if let Some(fc) = &t.faults {
                        let mut v = Vec::with_capacity(n);
                        t.chunk(fc, k0, k1, &mut v);
                        chunk_faults = Some(v);
                    }
                } else {
                    for di in 0..ti {
                        for dj in 0..tj {
                            for k in k0..k1 {
                                chunk_angles.push(tile_angles[(di * tj + dj) * nk + k]);
                                chunk_labels.push(labels[((i0 + di) * nj + (j0 + dj)) * nk + k]);
                            }
                        }
                    }
                }
                if tx
                    .send(([i_chunk, j_chunk, k_chunk], chunk_angles, chunk_labels, chunk_salt, chunk_faults))
                    .is_err()
                {
                    producer_error = Some("overlap writer stopped while sending chunk".to_string());
                    break 'tiles;
                }
                pending_write = true;
            }
        }
    }

    if pending_write && producer_error.is_none() && ack_rx.recv().is_err() {
        producer_error = Some("overlap writer stopped before final acknowledgement".to_string());
    }
    drop(tx);
    let writer_result = writer
        .join()
        .map_err(|_| "overlap writer thread panicked".to_string())?;
    if let Some(error) = producer_error {
        return Err(match writer_result {
            Ok(()) => error,
            Err(writer_error) => format!("{error}: {writer_error}"),
        });
    }
    writer_result?;

    let store = MdioStore::open(&path).map_err(|e| e.to_string())?;
    // Depth-axis fault labels (none on master f3720fb2 / d51ab237 depth mode;
    // see `pipeline_overlap_faults`). Guarded: time mode already wrote its own
    // masked output-domain fault_labels in the producer loop above, and depth
    // tiles here would overwrite them.
    if !time_mode {
        crate::pipeline_overlap_faults::write_overlap_fault_labels(
            &store, cfg, &trends, shape, chunks, &mut stats,
        )?;
    }
    store
        .finalize_after_chunked_write(&[])
        .map_err(|e| e.to_string())?;
    let back_angles = store.read_volume().map_err(|e| e.to_string())?;
    let back_labels = store.read_labels_u8().map_err(|e| e.to_string())?;
    let (reference, _) = generate_chunked(cfg);
    let parity = parity::compare_volumes(
        &reference.labels,
        &back_labels,
        &reference.angle_stack,
        &back_angles,
    );
    crate::salt::verify_salt_labels(&store, cfg)
        .map_err(|e| format!("overlapped salt_labels diverged from the salt body: {e}"))?;
    if time_mode {
        if let Some(reference) = crate::time_mode::generate_fault_labels_output(cfg) {
            let back = store.read_fault_labels_u8().map_err(|e| e.to_string())?;
            if back != reference {
                return Err("overlapped fault_labels diverged from the output-domain reference".into());
            }
        }
    }
    if reference.labels != back_labels || reference.angle_stack != back_angles {
        return Err(format!(
            "overlapped MDIO exact parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    Ok((
        E2eReport {
            volumes: reference,
            parity,
            store_path: Some(path),
            status: "ok-e2e-chunked-overlap",
        },
        stats,
    ))
}

/// Tile-scale Vp / Vs / rho bytes the rock-physics fuse path holds.
fn props_tile_bytes(model: &crate::rock_physics::ElasticModel, tile_voxels: usize) -> usize {
    if model.is_legacy_toy() {
        0
    } else {
        3 * tile_voxels * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn assert_overlap(shape: [usize; 3], chunks: [usize; 3], seed: u64) {
        let dir = tempdir().unwrap();
        let cfg = E2eConfig {
            time: Default::default(),
            geometry: crate::ToyGeometry::Planar,
            faults: Default::default(),
            filters: Default::default(),
            seed,
            inline_count: shape[0],
            crossline_count: shape[1],
            samples: shape[2],
            store_path: Some(dir.path().join("overlap.mdio")),
            chunk_shape: Some(chunks),
            rock_physics: Default::default(),
        };
        let (report, stats) = run_e2e_streaming_overlapped(&cfg).expect("overlap e2e");
        let (reference, _) = generate_chunked(&cfg);
        assert_eq!(report.status, "ok-e2e-chunked-overlap");
        assert_eq!(report.volumes.labels, reference.labels);
        assert_eq!(report.volumes.angle_stack, reference.angle_stack);
        assert_eq!(report.parity.label_iou, 1.0);
        assert_eq!(report.parity.angle_mae, 0.0);
        assert!(stats.is_bounded_by_chunk(64), "unbounded stats: {stats:?}");
        let (labels, _) = generate_labels(&cfg);
        let model = elastic_model(&cfg, &labels, shape);
        let expected_peak = model.model_bytes()
            + props_tile_bytes(&model, chunks[0] * chunks[1] * shape[2])
            + 4 * shape[2] * 4
            + shape[2] * 8
            + cfg.ricker().len() * 8
            + chunks[0] * chunks[1] * shape[2] * 4
            + chunks.iter().product::<usize>() * 5;
        assert_eq!(stats.peak_temp_bytes, expected_peak);
    }

    #[test]
    fn overlap_8_cube_exact_parity_and_bound() {
        assert_overlap([8, 8, 8], [4, 4, 8], 42);
    }

    #[test]
    fn overlap_16x16x32_exact_parity_and_bound() {
        assert_overlap([16, 16, 32], [8, 8, 32], 7);
    }
}
