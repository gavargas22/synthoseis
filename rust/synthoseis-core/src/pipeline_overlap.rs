//! Single-worker compute/write overlap with one writer thread and one in-flight chunk.

use std::sync::mpsc::sync_channel;
use std::thread;

use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_seismic::ricker;

use crate::parity;
use crate::pipeline::{E2eConfig, E2eReport, TINY_DIGI};
use crate::pipeline_stream::{
    depth_trends, fuse_tile_local, generate_chunked, generate_labels, resolve_chunk_shape,
    WorkingSetStats, DEFAULT_INCIDENCE_DEG,
};

type WriteChunk = ([usize; 3], Vec<f32>, Vec<u8>);

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
    let (labels, shape) = generate_labels(cfg);
    let [ni, nj, nk] = shape;
    let chunks = resolve_chunk_shape(cfg);
    let [ci, cj, ck] = chunks;
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
    {
        let store = MdioStore::create_empty(&path, &create).map_err(|e| e.to_string())?;
        store.ensure_labels_array().map_err(|e| e.to_string())?;
    }

    let (tx, rx) = sync_channel::<WriteChunk>(1);
    let (ack_tx, ack_rx) = sync_channel::<()>(0);
    let writer_path = path.clone();
    let writer = thread::spawn(move || -> Result<(), String> {
        let store = MdioStore::open(&writer_path).map_err(|e| e.to_string())?;
        for (key, angles, chunk_labels) in rx {
            store.write_chunk(key, &angles).map_err(|e| e.to_string())?;
            store
                .write_labels_chunk(key, &chunk_labels)
                .map_err(|e| e.to_string())?;
            ack_tx
                .send(())
                .map_err(|_| "overlap producer dropped acknowledgement channel".to_string())?;
        }
        Ok(())
    });

    let trends = depth_trends(nk);
    let wavelet = ricker(40.0, TINY_DIGI, 1);
    let mut tile_angles = vec![0.0f32; ci * cj * nk];
    // Trends + trace scratch + wavelet + tile output + exactly one writer-owned chunk.
    let fixed_bytes = 9 * nk * 8 + 4 * nk * 4 + nk * 8 + wavelet.len() * 8;
    let tile_bytes = tile_angles.capacity() * 4;
    let in_flight_bytes = ci * cj * ck * (4 + 1);
    stats.peak_temp_bytes = fixed_bytes + tile_bytes + in_flight_bytes;

    let mut pending_write = false;
    let mut producer_error = None;
    'tiles: for (i_chunk, i0) in (0..ni).step_by(ci).enumerate() {
        let i1 = (i0 + ci).min(ni);
        let ti = i1 - i0;
        for (j_chunk, j0) in (0..nj).step_by(cj).enumerate() {
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
                DEFAULT_INCIDENCE_DEG,
                &mut tile_angles,
                &mut stats,
            );
            stats.tiles_processed += 1;

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
                for di in 0..ti {
                    for dj in 0..tj {
                        for k in k0..k1 {
                            chunk_angles.push(tile_angles[(di * tj + dj) * nk + k]);
                            chunk_labels.push(labels[((i0 + di) * nj + (j0 + dj)) * nk + k]);
                        }
                    }
                }
                if tx
                    .send(([i_chunk, j_chunk, k_chunk], chunk_angles, chunk_labels))
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn assert_overlap(shape: [usize; 3], chunks: [usize; 3], seed: u64) {
        let dir = tempdir().unwrap();
        let cfg = E2eConfig {
            seed,
            inline_count: shape[0],
            crossline_count: shape[1],
            samples: shape[2],
            store_path: Some(dir.path().join("overlap.mdio")),
            chunk_shape: Some(chunks),
        };
        let (report, stats) = run_e2e_streaming_overlapped(&cfg).expect("overlap e2e");
        let (reference, _) = generate_chunked(&cfg);
        assert_eq!(report.status, "ok-e2e-chunked-overlap");
        assert_eq!(report.volumes.labels, reference.labels);
        assert_eq!(report.volumes.angle_stack, reference.angle_stack);
        assert_eq!(report.parity.label_iou, 1.0);
        assert_eq!(report.parity.angle_mae, 0.0);
        assert!(stats.is_bounded_by_chunk(64), "unbounded stats: {stats:?}");
        let expected_peak = 9 * shape[2] * 8
            + 4 * shape[2] * 4
            + shape[2] * 8
            + ricker(40.0, TINY_DIGI, 1).len() * 8
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
