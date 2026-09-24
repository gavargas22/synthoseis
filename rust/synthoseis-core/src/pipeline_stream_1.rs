fn fuse_tile_into_volume(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_deg: f64,
    angle_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let [_ni, nj, nk] = shape;
    let ti = i1 - i0;
    let tj = j1 - j0;
    let mut tile = vec![0.0f32; ti * tj * nk];
    stats.observe(tile.capacity() * 4);
    fuse_tile_local(
        labels,
        shape,
        i0,
        i1,
        j0,
        j1,
        trends,
        wavelet,
        angle_deg,
        &mut tile,
        stats,
    );
    for (di, i) in (i0..i1).enumerate() {
        for (dj, j) in (j0..j1).enumerate() {
            let src = (di * tj + dj) * nk;
            let dst = (i * nj + j) * nk;
            angle_out[dst..dst + nk].copy_from_slice(&tile[src..src + nk]);
        }
    }
}

/// Default mid-angle incidence used by the single-stack fused path.
pub const DEFAULT_INCIDENCE_DEG: f64 = 15.0;

/// Generate labels + angle stack with fused per-tile elastic/RFC/wavelet.
///
/// Bit-identical to [`crate::pipeline::generate_tiny_cube`] for the same seed
/// at [`DEFAULT_INCIDENCE_DEG`].
pub fn generate_chunked(cfg: &E2eConfig) -> (E2eVolumes, WorkingSetStats) {
    generate_chunked_at_angle(cfg, DEFAULT_INCIDENCE_DEG)
}

/// Like [`generate_chunked`], but with an explicit Zoeppritz incidence angle.
pub fn generate_chunked_at_angle(cfg: &E2eConfig, angle_deg: f64) -> (E2eVolumes, WorkingSetStats) {
    let (labels, shape) = generate_labels(cfg);
    generate_angle_stack_from_labels(cfg, &labels, shape, angle_deg)
}

/// Fuse an angle stack from already-generated labels (geometry amortization).
pub fn generate_angle_stack_from_labels(
    cfg: &E2eConfig,
    labels: &[u8],
    shape: [usize; 3],
    angle_deg: f64,
) -> (E2eVolumes, WorkingSetStats) {
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
                labels,
                shape,
                i0,
                i1,
                j0,
                j1,
                &trends,
                &wavelet,
                angle_deg,
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
            labels: labels.to_vec(),
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
