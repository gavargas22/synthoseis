/// Fuse-write one strip partition into an existing shared MDIO store.
///
/// Callers **must** pass non-overlapping partitions from
/// [`crate::JobPartitionPlan::from_config_chunk_aligned`] (or equivalent).
/// File-system ownership of distinct `(i_chunk, j_chunk, k_chunk)` keys is the
/// lock — there is no shared Mutex. Safe for multi-process writers on a shared FS.
pub fn write_strip_partition(
    store_path: &Path,
    part: &crate::partition::JobPartition,
    labels: &[u8],
    shape: [usize; 3],
    chunks: [usize; 3],
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
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
