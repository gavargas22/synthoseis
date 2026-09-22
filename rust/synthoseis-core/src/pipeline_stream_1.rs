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
                    angle_deg,
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
