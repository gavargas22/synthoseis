/// Resolved post-convolution filters (designed once per run from
/// [`crate::pipeline::FilterConfig`]). See `docs/filters-port.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeismicFilters {
    /// Legacy Butterworth bandpass (`filtfilt`, zero phase) along each trace.
    pub bandpass: Option<synthoseis_seismic::IirFilter>,
    /// Legacy lateral box filter size (`<= 1` = off).
    pub lateral_size: usize,
    /// Skip the Ricker convolution (bandpass on, `keep_ricker == false`):
    /// the tile is fused as raw reflectivity, then bandpassed (legacy chain).
    pub skip_wavelet: bool,
    /// Additive noise before the wavelet / bandpass (`None` = off).
    pub noise: Option<SeismicNoise>,
}

/// Resolved noise stage (see [`crate::pipeline::NoiseConfig`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SeismicNoise {
    pub config: crate::pipeline::NoiseConfig,
    /// Effective noise seed ([`crate::pipeline::NoiseConfig::seed`] or
    /// [`E2eConfig::seed`]).
    pub seed: u64,
    /// Signal-to-noise ratio (dB).
    pub sn_db: f64,
    /// Legacy `data_std` (std of the raw reflectivity at
    /// [`crate::pipeline::NOISE_NORM_ANGLE_DEG`] below the legacy seabed
    /// threshold). `None` until [`SeismicFilters::resolve`] computes it.
    pub data_std: Option<f64>,
}

impl SeismicNoise {
    /// The noise field at `angle_deg` (same seed and voxel counters for
    /// every angle, like legacy's shared `noise_0deg` / `noise_45deg`).
    ///
    /// # Panics
    /// If `data_std` has not been resolved.
    pub fn at_angle(&self, angle_deg: f64) -> synthoseis_seismic::WeightedNoise {
        let data_std = self
            .data_std
            .expect("noise data_std unresolved: build filters with SeismicFilters::resolve");
        let (w0, w45) = self.config.weights(angle_deg);
        synthoseis_seismic::WeightedNoise::new(self.seed, w0, w45, data_std, self.sn_db)
    }
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
        let noise = match fc.noise.snr_db {
            Some(db) if !db.is_finite() => {
                return Err(format!("noise S/N must be finite (got {db} dB)"));
            }
            Some(sn_db) => Some(SeismicNoise {
                config: fc.noise.clone(),
                seed: fc.noise.seed.unwrap_or(cfg.seed),
                sn_db,
                data_std: None,
            }),
            None => None,
        };
        Ok(Some(Self {
            bandpass,
            lateral_size: fc.lateral_size.max(1),
            skip_wavelet: fc.skips_ricker(),
            noise,
        }))
    }

    /// [`SeismicFilters::from_config`] plus the global noise statistics
    /// ([`noise_signal_std`]) when noise is on. Use this wherever tiles are
    /// fused; `labels` must be [`generate_labels`]`(cfg)`.
    pub fn resolve(
        cfg: &E2eConfig,
        labels: &[u8],
        shape: [usize; 3],
    ) -> Result<Option<Self>, String> {
        let mut f = Self::from_config(cfg)?;
        if let Some(n) = f.as_mut().and_then(|f| f.noise.as_mut()) {
            n.data_std = Some(noise_signal_std(cfg, labels, shape));
        }
        Ok(f)
    }

    /// The wavelet to fuse with: `wavelet`, or [`synthoseis_gpu::NO_WAVELET`]
    /// (identity: raw reflectivity) when the Ricker convolution is skipped.
    pub fn wavelet<'a>(&self, wavelet: &'a [f64]) -> &'a [f64] {
        if self.skip_wavelet {
            synthoseis_gpu::NO_WAVELET
        } else {
            wavelet
        }
    }

    /// Filter one fused source tile in place (bandpass on every trace).
    fn bandpass_traces(&self, src: &mut [f32], nk: usize) {
        if let Some(f) = &self.bandpass {
            f.filtfilt_traces_f32(src, nk)
                .expect("trace length validated in SeismicFilters::from_config");
        }
    }
}

/// Resolved filters for `cfg` (`None` when disabled). Noise statistics are
/// not resolved here; use [`SeismicFilters::resolve`] when noise may be on.
///
/// # Panics
/// On an invalid [`crate::pipeline::FilterConfig`]; the `run_*` entry points
/// validate first and return an error instead.
pub fn seismic_filters(cfg: &E2eConfig) -> Option<SeismicFilters> {
    SeismicFilters::from_config(cfg).unwrap_or_else(|e| panic!("invalid FilterConfig: {e}"))
}

/// Legacy noise normalisation `data_std`: population std of the raw
/// reflectivity at [`crate::pipeline::NOISE_NORM_ANGLE_DEG`] over samples
/// `k >= wb / (digi + 15) * digi` (legacy mask, see
/// [`synthoseis_seismic::legacy_noise_mask_threshold`]) of the `nk - 1`
/// Zoeppritz samples per trace (legacy `rfc_raw` has no pad sample).
///
/// One streaming pass: inline rows are fused one at a time (memory = one
/// `nj x nk` row) and reduced with Welford in fixed global `(i, j, k)` order,
/// so the value is independent of chunk shape and worker count; every
/// worker / process recomputes the same bits.
pub fn noise_signal_std(cfg: &E2eConfig, labels: &[u8], shape: [usize; 3]) -> f64 {
    let [ni, nj, nk] = shape;
    let trends = depth_trends(nk);
    let seabed = fault_seabed(cfg);
    let mut row = vec![0.0f32; nj * nk];
    let mut ws = WorkingSetStats::default();
    let mut acc = synthoseis_seismic::RunningStats::default();
    for i in 0..ni {
        fuse_tile_local(
            labels,
            shape,
            i,
            i + 1,
            0,
            nj,
            &trends,
            synthoseis_gpu::NO_WAVELET,
            crate::pipeline::NOISE_NORM_ANGLE_DEG,
            &mut row,
            &mut ws,
        );
        for j in 0..nj {
            let thr =
                synthoseis_seismic::legacy_noise_mask_threshold(seabed[i * nj + j], TINY_DIGI);
            for (k, &v) in row[j * nk..(j + 1) * nk - 1].iter().enumerate() {
                if k as f64 >= thr {
                    acc.push(v as f64);
                }
            }
        }
    }
    acc.std()
}

/// The noise field the pipeline adds to the raw reflectivity at `angle_deg`
/// (`None` when noise is off). Full `(ni, nj, nk)` volume; for tests, QC and
/// statistics.
pub fn generate_noise(cfg: &E2eConfig, angle_deg: f64) -> Option<Vec<f32>> {
    cfg.filters.noise.enabled().then_some(())?;
    let (labels, shape) = generate_labels(cfg);
    let f = SeismicFilters::resolve(cfg, &labels, shape)
        .unwrap_or_else(|e| panic!("invalid FilterConfig: {e}"))?;
    let noise = f.noise?.at_angle(angle_deg);
    let [ni, nj, nk] = shape;
    let mut out = vec![0.0f32; ni * nj * nk];
    noise.add_to_tile(&mut out, (0, ni), (0, nj), shape);
    Some(out)
}

/// Convolve every `nk`-sample trace of `src` with `wavelet` in place,
/// exactly like the fused CPU kernel (f32 -> f64 `convolve_same_1d` -> f32).
fn convolve_traces_in_place(src: &mut [f32], nk: usize, wavelet: &[f64]) {
    let mut trace = vec![0.0f64; nk];
    for t in src.chunks_exact_mut(nk) {
        for (d, &s) in trace.iter_mut().zip(t.iter()) {
            *d = s as f64;
        }
        let conv = synthoseis_seismic::convolve_same_1d(&trace, wavelet);
        for (d, &c) in t.iter_mut().zip(conv.iter()) {
            *d = c as f32;
        }
    }
}

/// Wavelet the pipeline convolves with for `cfg`: `wavelet`, or the empty
/// [`synthoseis_gpu::NO_WAVELET`] (no convolution) when the bandpass is on and
/// [`crate::pipeline::FilterConfig::keep_ricker`] is off.
pub fn effective_wavelet<'a>(cfg: &E2eConfig, wavelet: &'a [f64]) -> &'a [f64] {
    if cfg.filters.skips_ricker() {
        synthoseis_gpu::NO_WAVELET
    } else {
        wavelet
    }
}

/// Raw reflectivity stack for `cfg` at `angle_deg`: fused elastic →
/// Zoeppritz with no wavelet and no filters (the legacy `rfc_raw` angle
/// cube). This is the input the legacy `postprocess_rfc_cubes` bandpasses;
/// used for parity checks and QC. Memory is the full `(ni, nj, nk)` output.
pub fn generate_reflectivity(cfg: &E2eConfig, angle_deg: f64) -> Vec<f32> {
    let (labels, shape) = generate_labels(cfg);
    let [ni, nj, nk] = shape;
    let trends = depth_trends(nk);
    let mut out = vec![0.0f32; ni * nj * nk];
    let mut stats = WorkingSetStats::default();
    fuse_tile_local(
        &labels,
        shape,
        0,
        ni,
        0,
        nj,
        &trends,
        synthoseis_gpu::NO_WAVELET,
        angle_deg,
        &mut out,
        &mut stats,
    );
    out
}

/// Apply the optional bandpass + lateral filter to a whole `(ni, nj, nk)`
/// angle stack in place (classic full-cube path). No-op when disabled. Noise
/// is added before the wavelet by the caller (see
/// [`crate::pipeline::generate_tiny_cube`]).
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
/// tile is fused (as raw reflectivity when [`SeismicFilters::skip_wavelet`],
/// i.e. the bandpass replaces the Ricker wavelet) with a lateral halo (the
/// columns the box filter reads, including reflected boundary columns; see
/// [`synthoseis_seismic::lateral_source_range`]), every halo trace is
/// bandpassed, and the lateral filter is evaluated for the tile's own
/// columns. With [`SeismicFilters::noise`] the halo is fused as raw
/// reflectivity, the counter-based noise is added, and the wavelet (unless
/// skipped) is convolved before the bandpass (legacy order). Fusing, noise
/// and bandpassing are per-trace functions of the labels and global voxel
/// indices, so recomputing the halo gives exactly the values a neighbouring
/// tile, strip worker or process computes: the output is bit-identical for
/// any chunk shape and worker count, with memory bounded by the halo tile.
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
    let wavelet = f.wavelet(wavelet);
    match &f.noise {
        None => fuse_tile_local(
            labels, shape, si0, si1, sj0, sj1, trends, wavelet, angle_deg, &mut src, stats,
        ),
        Some(noise) => {
            // Legacy order: raw reflectivity + noise, then (optionally) the
            // wavelet, then the bandpass. Noise is keyed by the global voxel
            // index, so halo columns get the same noise as their owner tile.
            fuse_tile_local(
                labels,
                shape,
                si0,
                si1,
                sj0,
                sj1,
                trends,
                synthoseis_gpu::NO_WAVELET,
                angle_deg,
                &mut src,
                stats,
            );
            noise
                .at_angle(angle_deg)
                .add_to_tile(&mut src, (si0, si1), (sj0, sj1), shape);
            if !wavelet.is_empty() {
                convolve_traces_in_place(&mut src, nk, wavelet);
            }
        }
    }
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
