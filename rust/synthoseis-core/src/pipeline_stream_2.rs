/// Fuse one spatial tile: elastic props → Zoeppritz RFC → wavelet convolution.
///
/// `shape` is the depth shape; `tile_out` holds `(ti, tj, nk_out)` samples,
/// `nk_out` = [`ElasticModel::output_nk`] (`nt` in time mode, where the
/// reflectivity is inserted on the two-way-time axis before the wavelet).
///
/// `trends` is the [`ElasticModel`]: the master toy trends go through the
/// label kernel unchanged (bit-identical to master); the default rock-physics
/// model fills tile-scale Vp / Vs / rho buffers first and fuses those.
///
/// Delegates to [`synthoseis_gpu::fuse_tile_dispatch`] so `--gpu` / prefer-gpu
/// can select the auto path. Default remains the CPU software adapter
/// (bit-identical to the historical inline implementation).
pub fn fuse_tile_local(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &ElasticModel,
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let nk = shape[2];
    stats.observe(synthoseis_gpu::fuse_tile_scratch_bytes(nk));
    match trends {
        ElasticModel::LegacyToy(t) => {
            let _backend = synthoseis_gpu::fuse_tile_dispatch(
                labels, shape, i0, i1, j0, j1, t, wavelet, angle_deg, tile_out,
            );
        }
        ElasticModel::Rpm(m) if crate::time_mode::partial_time_path(trends).is_some() => {
            // Partial voxels in time mode: per-column fused path (spec §1.5).
            let n = (i1 - i0) * (j1 - j0) * nk;
            // Time trace scratch: the window plus the wavelet's bottom pad
            // `h` (physical filter edges).
            let pad = edge_pad_bytes(m.time.as_ref(), wavelet);
            stats.observe(
                synthoseis_gpu::fuse_tile_scratch_bytes(nk) + 3 * nk * 4 + (nk + 1) * 8 + m.time.map_or(0, |a| a.nt * 16) + pad + n,
            );
            crate::time_mode::fuse_tile_time_partial(m, labels, shape, i0, i1, j0, j1, wavelet, angle_deg, tile_out);
        }
        ElasticModel::Rpm(m) => {
            let n = (i1 - i0) * (j1 - j0) * nk;
            let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            stats.observe(synthoseis_gpu::fuse_tile_scratch_bytes(nk) + 3 * n * 4);
            trends.tile_properties(labels, shape, i0, i1, j0, j1, &mut vp, &mut vs, &mut rho);
            match &m.time {
                // Time mode: `(ti, tj, nt)` output (CPU only, see
                // `time_mode::fuse_props_tile_time`). T + time trace scratch.
                Some(axis) => {
                    stats.observe(
                        synthoseis_gpu::fuse_tile_scratch_bytes(nk)
                            + 3 * n * 4
                            + (nk + 1) * 8
                            + axis.nt * 16
                            + edge_pad_bytes(Some(axis), wavelet),
                    );
                    crate::time_mode::fuse_props_tile_time(
                        &vp, &vs, &rho, nk, axis, wavelet, angle_deg, trends.zoeppritz_form(), tile_out,
                    );
                }
                None => {
                    let _backend = synthoseis_gpu::fuse_props_tile_dispatch(
                        &vp, &vs, &rho, nk, wavelet, angle_deg, trends.zoeppritz_form(), tile_out,
                    );
                }
            }
        }
    }
}

/// Extra per-column scratch bytes of the wavelet-only physical edge chain:
/// `h` more f64 reflectivity samples, plus the padded f32 buffer and its
/// f64 convolution copy; 0 on the legacy axis and under
/// `--legacy-filter-edges`.
fn edge_pad_bytes(axis: Option<&crate::pipeline::TimeAxis>, wavelet: &[f64]) -> usize {
    axis.and_then(|a| crate::time_mode::TraceChain::wavelet_only(a, wavelet).map(|c| (a.nt, c.pads)))
        .map_or(0, |(nt, pads)| pads.bottom * 8 + pads.len(nt) * (4 + 8))
}
