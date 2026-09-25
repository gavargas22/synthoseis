/// Resolved post-convolution filters (designed once per run from
/// [`crate::pipeline::FilterConfig`]). See `docs/filters-port.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeismicFilters {
    /// Legacy Butterworth bandpass (`filtfilt`, zero phase) along each trace.
    pub bandpass: Option<synthoseis_seismic::IirFilter>,
    /// Legacy lateral box filter size (`<= 1` = off).
    pub lateral_size: usize,
}

impl SeismicFilters {
    /// Validate `cfg.filters` and design the bandpass (`Ok(None)` when the
    /// filters are disabled).
    pub fn from_config(cfg: &E2eConfig) -> Result<Option<Self>, String> {
        let fc = &cfg.filters;
        if !fc.enabled() {
            return Ok(None);
        }
        let bandpass = match fc.bandpass_hz {
            Some([low, high]) => {
                let f = synthoseis_seismic::butterworth_bandpass(
                    low,
                    high,
                    synthoseis_seismic::legacy_digitisation_ms(TINY_DIGI),
                    fc.bandpass_order,
                )
                .map_err(|e| format!("bandpass {low}-{high} Hz: {e}"))?;
                if cfg.samples <= f.padlen() {
                    return Err(format!(
                        "bandpass order {} needs more than {} samples per trace (got {})",
                        fc.bandpass_order,
                        f.padlen(),
                        cfg.samples
                    ));
                }
                Some(f)
            }
            None => None,
        };
        Ok(Some(Self {
            bandpass,
            lateral_size: fc.lateral_size.max(1),
        }))
    }

    /// Filter one fused source tile in place (bandpass on every trace).
    fn bandpass_traces(&self, src: &mut [f32], nk: usize) {
        if let Some(f) = &self.bandpass {
            f.filtfilt_traces_f32(src, nk)
                .expect("trace length validated in SeismicFilters::from_config");
        }
    }
}

/// Resolved filters for `cfg` (`None` when disabled).
///
/// # Panics
/// On an invalid [`crate::pipeline::FilterConfig`]; the `run_*` entry points
/// validate first and return an error instead.
pub fn seismic_filters(cfg: &E2eConfig) -> Option<SeismicFilters> {
    SeismicFilters::from_config(cfg).unwrap_or_else(|e| panic!("invalid FilterConfig: {e}"))
}

/// Apply the optional filters to a whole `(ni, nj, nk)` angle stack in place
/// (classic full-cube path). No-op when disabled.
pub fn apply_filters_to_volume(cfg: &E2eConfig, volume: &mut [f32]) {
    let Some(f) = seismic_filters(cfg) else {
        return;
    };
    let shape = cfg.shape();
    f.bandpass_traces(volume, shape[2]);
    if f.lateral_size > 1 {
        let out = synthoseis_seismic::lateral_uniform_volume(volume, shape, f.lateral_size);
        volume.copy_from_slice(&out);
    }
}

/// Fuse output tile `[i0, i1) x [j0, j1)` and apply the optional
/// post-convolution filters.
///
/// With `filters == None` this is exactly [`fuse_tile_local`]. Otherwise the
/// tile is fused with a lateral halo (the columns the box filter reads,
/// including reflected boundary columns; see
/// [`synthoseis_seismic::lateral_source_range`]), every halo trace is
/// bandpassed, and the lateral filter is evaluated for the tile's own
/// columns. Fusing and bandpassing are per-trace functions of the labels, so
/// recomputing the halo gives exactly the values a neighbouring tile, strip
/// worker or process computes: the output is bit-identical for any chunk
/// shape and worker count, with memory bounded by the halo tile.
#[allow(clippy::too_many_arguments)]
pub fn fuse_tile_filtered(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_deg: f64,
    filters: Option<&SeismicFilters>,
    tile_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let [ni, nj, nk] = shape;
    let n_out = (i1 - i0) * (j1 - j0) * nk;
    let tile_out = &mut tile_out[..n_out];
    let Some(f) = filters else {
        fuse_tile_local(
            labels, shape, i0, i1, j0, j1, trends, wavelet, angle_deg, tile_out, stats,
        );
        return;
    };
    let (si0, si1) = synthoseis_seismic::lateral_source_range(i0, i1, ni, f.lateral_size);
    let (sj0, sj1) = synthoseis_seismic::lateral_source_range(j0, j1, nj, f.lateral_size);
    let mut src = vec![0.0f32; (si1 - si0) * (sj1 - sj0) * nk];
    // Halo tile + lateral intermediate + filtfilt scratch.
    stats.observe(src.capacity() * 4 * 2 + (nk + 2 * 64) * 8 * 2);
    fuse_tile_local(
        labels, shape, si0, si1, sj0, sj1, trends, wavelet, angle_deg, &mut src, stats,
    );
    f.bandpass_traces(&mut src, nk);
    synthoseis_seismic::lateral_uniform_tile(
        &src,
        (si0, si1),
        (sj0, sj1),
        shape,
        (i0, i1),
        (j0, j1),
        f.lateral_size,
        tile_out,
    );
}
