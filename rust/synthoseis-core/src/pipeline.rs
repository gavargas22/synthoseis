//! End-to-end single-worker tiny-cube pipeline.
//!
//! # Shape
//! MDIO create → geo (horizons/labels) → closures (relabel/filter) → RPM
//! (elastic props) → seismic (Zoeppritz + wavelet) → MDIO write (labels +
//! angle stack) → parity check (second deterministic pass).
//!
//! # Product locks
//! - Single local worker, CPU path only
//! - Parity = labels + angle stacks (not bit-identical full seismic)
//! - Tiny fixed dims (default 8³); Python `main.py` generator stays untouched

use std::path::{Path, PathBuf};

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::fill_layer_labels;
use synthoseis_io::{CreateConfig, DeliverableWriter, Dimension, MdioStore};
use synthoseis_seismic::{apply_wavelet_traces, compute_rfc_volumes_form, ricker};

use crate::parity::{self, ParityReport};
pub use crate::rock_physics::RockPhysicsConfig;
pub use crate::toy_geometry::ToyGeometry;

/// Default tiny-cube edge length used by the e2e smoke path.
pub const TINY_DIM: usize = 8;

/// Default digi (ms) for the tiny-cube store.
pub const TINY_DIGI: f64 = 4.0;

/// Synthetic depth scale (m per sample index) of the master toy trends
/// ([`RockPhysicsConfig::legacy_toy_depth`] only; the default model uses
/// [`RockPhysicsConfig::depth_step_m`]).
pub(crate) const DEPTH_PER_SAMPLE: f64 = 100.0;

/// Configuration for the single-worker e2e pipeline.
#[derive(Debug, Clone)]
pub struct E2eConfig {
    pub seed: u64,
    pub inline_count: usize,
    pub crossline_count: usize,
    pub samples: usize,
    /// Optional MDIO root; when set, labels + angle stack are written.
    pub store_path: Option<PathBuf>,
    /// Optional MDIO / fused-generation chunk shape `[ci, cj, ck]`.
    ///
    /// When `None`, [`crate::pipeline_stream::resolve_chunk_shape`] picks a
    /// sub-volume default (never full-array when the grid allows). Chunk keys
    /// are strip-friendly for a later multi-worker partition.
    pub chunk_shape: Option<[usize; 3]>,
    /// Optional fault modelling (port of `datagenerator/Faults.py`).
    ///
    /// Default is disabled (`count == 0`): geology, labels and angle stacks
    /// are bit-identical to the pre-fault pipeline.
    pub faults: FaultConfig,
    /// Optional post-convolution seismic filters (port of the legacy
    /// Butterworth bandpass + lateral filter, `datagenerator/Seismic.py`).
    ///
    /// Default is disabled: angle stacks are bit-identical to the unfiltered
    /// pipeline. See `docs/filters-port.md`.
    pub filters: FilterConfig,
    /// Rock physics (elastic properties from the labels). Default: the
    /// corrected legacy model (4 m per sample, per-layer depth below the
    /// seabed, water column, net-to-gross mixing, closures).
    /// [`RockPhysicsConfig::legacy_toy`] reproduces master 10f4dcd bit for
    /// bit. See `docs/rock-physics-port.md`.
    pub rock_physics: RockPhysicsConfig,
    /// Toy horizon geometry: layered dome (default) or the master planar
    /// stack. `rock_physics.legacy_toy_depth` forces planar. See
    /// [`crate::toy_geometry`] and `docs/layered-toy-geometry.md`.
    pub geometry: ToyGeometry,
    /// Depth-to-time conversion of the seismic chain (default: on, two-way
    /// time from the voxel Vp). `TimeConfig::legacy()` (CLI
    /// `--legacy-depth-as-time`) keeps the legacy axis, where each depth
    /// sample is also one 4 ms time sample, bit for bit. Ignored with
    /// `rock_physics.legacy_toy_depth` (no physical Vp). See
    /// `docs/depth-to-time.md`.
    pub time: TimeConfig,
}

/// Post-convolution filters applied to each fused angle-stack tile.
///
/// Legacy `postprocess_rfc_cubes` order: bandpass (`apply_bandlimits`), then
/// the lateral filter (`apply_lateral_filter`, only when `size > 1`).
///
/// **Wavelet.** Legacy bandpasses the raw reflectivity (plus noise) and never
/// convolves a wavelet in its default path: the Butterworth filter *is* the
/// wavelet. So when the bandpass is on, the Rust pipeline skips the Ricker
/// convolution by default (reflectivity then bandpass, exactly the legacy
/// chain). Set [`FilterConfig::keep_ricker`] to restore the combined
/// Ricker + bandpass behaviour of the first filter port (#25). With the
/// bandpass off (filters off, or lateral filter only) the Ricker wavelet is
/// always applied, so filters-off output is unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterConfig {
    /// Butterworth bandpass corners `[low, high]` in Hz (`None` = off).
    /// Legacy draws `low ~ U(bandwidth_low)`, `high ~ U(bandwidth_high)`
    /// (example config: 3-6 Hz and 20-35 Hz).
    pub bandpass_hz: Option<[f64; 2]>,
    /// Butterworth order (legacy `bandwidth_ord`, default 4).
    pub bandpass_order: usize,
    /// Lateral box-filter size `n` (legacy `lateral_filter_size`: 1, 3 or 5).
    /// `<= 1` = off.
    pub lateral_size: usize,
    /// Keep the Ricker wavelet convolution when the bandpass is on (the #25
    /// behaviour: Ricker, then bandpass). Default `false` = skip the Ricker
    /// wavelet when the bandpass is on (legacy chain). Ignored when the
    /// bandpass is off.
    pub keep_ricker: bool,
    /// Bandpass the whole `nk`-sample trace, including the trailing
    /// reflectivity sample (always 0 before noise), as master did before the
    /// trailing-sample fix (CLI `--bandpass-trailing-sample`).
    ///
    /// Default `false` (legacy parity): when the Ricker is skipped, the
    /// bandpass runs on the first `nk - 1` samples only, which is exactly the
    /// trace legacy filters (`rfc_raw` has `nk - 1` samples), and the
    /// trailing sample, which legacy never produces, is written as 0 (the
    /// fuse's "no interface below" value). Ignored when the bandpass is off
    /// or [`FilterConfig::keep_ricker`] is set (a Ricker-convolved trace has
    /// real signal in its last sample). See `docs/filters-port.md`.
    pub bandpass_trailing_sample: bool,
    /// Additive random noise before the wavelet / bandpass (legacy
    /// `add_weighted_noise`). Off by default.
    pub noise: NoiseConfig,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            bandpass_hz: None,
            bandpass_order: 4,
            lateral_size: 1,
            keep_ricker: false,
            bandpass_trailing_sample: false,
            noise: NoiseConfig::default(),
        }
    }
}

/// Deterministic replacement for legacy `SeismicVolume.add_weighted_noise`.
///
/// Legacy draws two white Laplace cubes (`noise_0deg`, `noise_45deg`), mixes
/// them per angle with Hilterman weights `cos²θ n0 + sin²θ n45`, rescales the
/// mix to `data_std / std_ratio` (`data_std` = std of the middle-angle raw
/// reflectivity below the seabed cutoff, see [`NoiseConfig::legacy_seabed`];
/// `std_ratio = sqrt(10^(sn_db/10))`) and adds it to the raw reflectivity
/// before the bandpass. The Rust port draws the same distribution from a
/// counter-based RNG (Philox4x32-10 keyed by the seed, counter = global voxel
/// index), so the noise is identical for any tiling, worker split or process
/// count. `data_std` comes from one deterministic streaming pass over the
/// reflectivity at [`NOISE_NORM_ANGLE_DEG`]. See `docs/filters-port.md`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NoiseConfig {
    /// Signal-to-noise ratio in dB (legacy `sn_db`; example config draws it
    /// from a triangular distribution over `signal_to_noise_ratio_db`).
    /// `None` = no noise.
    pub snr_db: Option<f64>,
    /// Noise seed. `None` = [`E2eConfig::seed`].
    pub seed: Option<u64>,
    /// Replicate the legacy angle weights exactly: `math.cos(ang)` with the
    /// angle in degrees fed to a radian function (see
    /// `tests/test_seismic_noise.py`). Default `false` = correct radian
    /// weights ([`synthoseis_seismic::hilterman_noise_weights`]).
    pub legacy_angle_weights: bool,
    /// Replicate the legacy `data_std` mask exactly: samples `k >= wb /
    /// (digi + 15) * digi` (~0.84 x the seabed sample for digi = 4, so part
    /// of the water column is included). Default `false` = the cutoff is the
    /// actual seabed (`k >= seabed`, sub-seabed reflectivity only); see
    /// [`synthoseis_seismic::noise_mask_threshold`].
    pub legacy_seabed: bool,
}

/// Incidence angle whose raw reflectivity normalises the noise (legacy uses
/// the middle angle of `incident_angles`; the example config's is 15 deg).
pub const NOISE_NORM_ANGLE_DEG: f64 = 15.0;

impl NoiseConfig {
    /// Noise at `snr_db` dB with the default seed.
    pub fn snr(snr_db: f64) -> Self {
        Self {
            snr_db: Some(snr_db),
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.snr_db.is_some()
    }

    /// `data_std` mask threshold (samples) for a column whose seabed sits at
    /// `seabed_samples` ([`NoiseConfig::legacy_seabed`]).
    pub fn mask_threshold(&self, seabed_samples: f64, digi: f64) -> f64 {
        synthoseis_seismic::noise_mask_threshold(seabed_samples, digi, self.legacy_seabed)
    }

    /// `(w0, w45)` noise weights at `angle_deg`.
    pub fn weights(&self, angle_deg: f64) -> (f64, f64) {
        if self.legacy_angle_weights {
            synthoseis_seismic::legacy_degree_noise_weights(angle_deg)
        } else {
            synthoseis_seismic::hilterman_noise_weights(angle_deg)
        }
    }
}

impl FilterConfig {
    /// Legacy defaults: order-4 bandpass `[low, high]` Hz and an `n x n`
    /// lateral filter.
    pub fn legacy(low_hz: f64, high_hz: f64, lateral_size: usize) -> Self {
        Self {
            bandpass_hz: Some([low_hz, high_hz]),
            bandpass_order: 4,
            lateral_size,
            keep_ricker: false,
            bandpass_trailing_sample: false,
            noise: NoiseConfig::default(),
        }
    }

    /// `true` when any post-convolution stage (bandpass, lateral filter or
    /// noise) is on.
    pub fn enabled(&self) -> bool {
        self.bandpass_hz.is_some() || self.lateral_size > 1 || self.noise.enabled()
    }

    /// `true` when the Ricker convolution is skipped: the bandpass is on and
    /// [`FilterConfig::keep_ricker`] is `false`.
    pub fn skips_ricker(&self) -> bool {
        self.bandpass_hz.is_some() && !self.keep_ricker
    }

    /// `true` when the bandpass runs on the first `nk - 1` samples only and
    /// the trailing sample is written as 0 (legacy parity): the Ricker is
    /// skipped and [`FilterConfig::bandpass_trailing_sample`] is off.
    pub fn bandpass_excludes_trailing_sample(&self) -> bool {
        self.skips_ricker() && !self.bandpass_trailing_sample
    }
}

/// Fault settings for the e2e pipeline (random-mode draw, seeded from
/// [`E2eConfig::seed`]). See `docs/faults-port.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultConfig {
    /// Number of faults to draw (`0` = faulting disabled).
    pub count: usize,
    /// Minimum throw in samples (Python `low_fault_throw / infill_factor`).
    pub throw_min: f64,
    /// Maximum throw in samples. Default `29.0` keeps below the hockey-stick
    /// threshold (`0.85 * 35`), whose drag zone is deferred in the port.
    pub throw_max: f64,
    /// `true` = exact legacy vertical reach (`ReachMode::Legacy`). Default
    /// `false` = `ReachMode::FitColumn`: identical to legacy whenever the
    /// legacy seabed taper succeeds, otherwise sigma is fitted to the
    /// sub-seabed column and fault labels are clamped below the seabed.
    pub legacy_reach: bool,
    /// Legacy switch (library only): the overlapped writer
    /// ([`crate::run_e2e_streaming_overlapped`]) writes no `data/fault_labels`,
    /// reproducing master f3720fb2 bit for bit. Default `false`: it writes
    /// them like every other MDIO path (see `crate::pipeline_overlap_faults`).
    pub overlap_legacy_no_fault_labels: bool,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            count: 0,
            throw_min: 5.0,
            throw_max: 29.0,
            legacy_reach: false,
            overlap_legacy_no_fault_labels: false,
        }
    }
}

impl FaultConfig {
    /// `count` faults with default throw range.
    pub fn with_count(count: usize) -> Self {
        Self {
            count,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.count > 0
    }
}

/// Ricker peak frequency of the e2e pipeline (Hz), `ricker(40.0, dt, 1)`.
pub const RICKER_PEAK_HZ: f64 = 40.0;

/// Depth-to-time conversion settings (spec "depth-to-time conversion", §2),
/// [`E2eConfig::time`].
///
/// Time mode (the default) computes each column's two-way time from the
/// voxel Vp, inserts the depth reflectivity band-limited at the interface
/// times, and runs the wavelet, noise and filters on a uniform `dt` grid of
/// `nt` samples. Labels, fault labels and salt labels are point-sampled
/// onto the same grid. `enabled = false` ([`TimeConfig::legacy`], CLI
/// `--legacy-depth-as-time`) is the legacy axis: each depth sample is also
/// one 4 ms time sample (a constant 2000 m/s), bit for bit.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeConfig {
    /// Convert the seismic chain to two-way time (default `true`; `false` =
    /// the legacy depth-as-time axis).
    pub enabled: bool,
    /// Output sample interval (ms), 0.5–8.0. Drives the Ricker, the
    /// Butterworth design and the MDIO `digi` attribute in time mode.
    pub dt_ms: f64,
    /// Output length `nt` (time samples). `None` = `nt₀ =
    /// round(nz · (2·dz / 2000 m/s) / dt)` ([`synthoseis_seismic::default_twt_samples`]),
    /// which is `nz` at the defaults. Valid range 16 ≤ nt ≤ 8·nz.
    pub samples: Option<usize>,
    /// Resampling kernel (`sinc` default, `linear` fast option).
    pub kernel: synthoseis_seismic::TwtKernel,
    /// Test hook: build every column's two-way time from this constant
    /// velocity (m/s) instead of the voxel Vp (Zoeppritz still uses the
    /// voxel properties). `Some(2000.0)` reproduces the legacy axis' implied
    /// velocity, which pins time mode against the depth fuse (spec §5.2).
    /// Not a CLI option.
    #[doc(hidden)]
    pub constant_twt_vp: Option<f64>,
}

impl Default for TimeConfig {
    fn default() -> Self {
        TimeConfig {
            enabled: true,
            dt_ms: TINY_DIGI,
            samples: None,
            kernel: synthoseis_seismic::TwtKernel::Sinc,
            constant_twt_vp: None,
        }
    }
}

/// Lowest P velocity of a sediment cell (m/s): the shale trend at z = 0
/// (`RPMExample.shale_vp(0)`). Drives the depth-staircase warning (spec
/// §3.3, constraint 2); water has no internal interfaces and does not count.
pub const VP_MIN_SEDIMENT: f64 = 1580.0;

impl TimeConfig {
    /// The legacy depth-as-time axis (`--legacy-depth-as-time`): every
    /// output, label and MDIO attribute as on master before the conversion.
    pub fn legacy() -> Self {
        TimeConfig {
            enabled: false,
            ..TimeConfig::default()
        }
    }

    /// Output length for a depth model of `nz` cells of `dz` m: the explicit
    /// [`TimeConfig::samples`] or `nt₀`. A pure function of the config.
    pub fn output_samples(&self, nz: usize, dz: f64) -> usize {
        self.samples
            .unwrap_or_else(|| synthoseis_seismic::default_twt_samples(nz, dz, self.dt_ms))
    }

    /// Highest signal frequency the output must carry (Hz): the bandpass
    /// high corner when the bandpass replaces the Ricker
    /// ([`FilterConfig::skips_ricker`]), else 2.5 × the Ricker peak (100 Hz
    /// for 40 Hz). With `keep_ricker` the 40 Hz Ricker is convolved before the
    /// bandpass, so it counts even though a bandpass is on.
    pub fn signal_max_hz(filters: &FilterConfig) -> f64 {
        match filters.bandpass_hz {
            Some([_, hi]) if filters.skips_ricker() => hi,
            _ => 2.5 * RICKER_PEAK_HZ,
        }
    }

    /// Validate against the depth model and filters; returns `nt`.
    ///
    /// Errors (the CLI will exit 2 in PR B): `dt_ms` outside 0.5–8.0, `nt`
    /// outside 16 ≤ nt ≤ 8·nz, or the output Nyquist constraint
    /// `f_hi > 0.4 / dt` (spec §3.3). The depth-staircase constraint is a
    /// warning ([`TimeConfig::staircase_warning`]).
    pub fn validate(&self, nz: usize, dz: f64, filters: &FilterConfig) -> Result<usize, String> {
        if !(0.5..=8.0).contains(&self.dt_ms) {
            return Err(format!("dt-ms {} outside 0.5-8.0", self.dt_ms));
        }
        let nt = self.output_samples(nz, dz);
        // The floor is 16 samples, or nt₀ when the default axis itself is
        // shorter (tiny cubes, e.g. the 8³ CLI default), so the default
        // never fails; an explicit `--twt-samples` below both is rejected.
        let floor = 16.min(synthoseis_seismic::default_twt_samples(nz, dz, self.dt_ms)).max(2);
        if nt < floor || nt > 8 * nz {
            return Err(format!(
                "twt-samples {nt} outside {floor}..={} (8 x {nz} depth samples)",
                8 * nz
            ));
        }
        let f_hi = Self::signal_max_hz(filters);
        if !synthoseis_seismic::output_nyquist_ok(f_hi, self.dt_ms) {
            return Err(format!(
                "dt-ms {} too coarse: highest signal frequency {f_hi} Hz exceeds 0.4/dt = {} Hz",
                self.dt_ms,
                0.4 / (self.dt_ms / 1000.0)
            ));
        }
        Ok(nt)
    }

    /// Warning text when the per-cell depth staircase would not fall in the
    /// kernel stopband (`dz > Vp_min · dt / 1.2`, spec §3.3 constraint 2).
    pub fn staircase_warning(&self, dz: f64, vp_min_sediment: f64) -> Option<String> {
        (!synthoseis_seismic::depth_staircase_ok(dz, vp_min_sediment, self.dt_ms)).then(|| {
            format!(
                "depth step {dz} m > Vp_min {vp_min_sediment} m/s x dt / 1.2 = {:.2} m: the depth staircase comb may alias into the band",
                vp_min_sediment * self.dt_ms / 1000.0 / 1.2
            )
        })
    }
}

impl Default for E2eConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            inline_count: TINY_DIM,
            crossline_count: TINY_DIM,
            samples: TINY_DIM,
            store_path: None,
            chunk_shape: None,
            faults: FaultConfig::default(),
            filters: FilterConfig::default(),
            rock_physics: RockPhysicsConfig::default(),
            geometry: ToyGeometry::default(),
            time: TimeConfig::default(),
        }
    }
}

/// Resolved output time axis of a time-mode run ([`E2eConfig::time_axis`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeAxis {
    /// Output sample interval (ms).
    pub dt_ms: f64,
    /// Output samples per trace.
    pub nt: usize,
    /// Depth cell size (m), [`RockPhysicsConfig::depth_step_m`].
    pub dz: f64,
    /// Spike-insertion kernel.
    pub kernel: synthoseis_seismic::TwtKernel,
    /// [`TimeConfig::constant_twt_vp`] (test hook).
    #[doc(hidden)]
    pub constant_twt_vp: Option<f64>,
}

impl E2eConfig {
    /// Geometry actually used: planar when `legacy_toy_depth` is set (the
    /// master 10f4dcd guarantee), else [`E2eConfig::geometry`].
    pub fn effective_geometry(&self) -> ToyGeometry {
        if self.rock_physics.legacy_toy_depth {
            ToyGeometry::Planar
        } else {
            self.geometry
        }
    }

    /// Lithology actually used: alternating for the planar geometry and
    /// `legacy_toy_depth` (their goldens), else
    /// [`RockPhysicsConfig::lithology`].
    pub fn effective_lithology(&self) -> crate::lithology::ToyLithology {
        if self.effective_geometry() == ToyGeometry::Planar {
            crate::lithology::ToyLithology::Alternating
        } else {
            self.rock_physics.lithology
        }
    }

    /// Closures per sand layer (`RockPhysicsConfig::closures_per_layer`):
    /// forced for the planar geometry (and so `--legacy-toy-depth`), which
    /// keeps the planar goldens. Its only sand layer is the deepest unit,
    /// which per-unit closures would skip.
    pub fn effective_closures_per_layer(&self) -> bool {
        self.rock_physics.closures_per_layer || self.effective_geometry() == ToyGeometry::Planar
    }

    /// Salt body present ([`RockPhysicsConfig::salt`]): layered geometry
    /// only, so planar and `--legacy-toy-depth` keep their goldens.
    pub fn effective_salt(&self) -> bool {
        self.rock_physics.salt && self.effective_geometry() == ToyGeometry::Layered
    }

    /// Fault labels are masked by the salt body (`fault AND NOT salt`):
    /// salt present, faults enabled and no `--fault-labels-through-salt`.
    pub fn effective_fault_salt_mask(&self) -> bool {
        self.effective_salt() && self.faults.enabled() && !self.rock_physics.fault_labels_through_salt
    }

    pub fn tiny(seed: u64) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    /// Depth-model shape `(ni, nj, nz)`: geometry, faults, closures, salt
    /// and the elastic model always use it (never `nt`).
    pub fn shape(&self) -> [usize; 3] {
        [self.inline_count, self.crossline_count, self.samples]
    }

    /// `true` when the seismic chain runs in two-way time: [`TimeConfig::enabled`]
    /// and a physical Vp (`--legacy-toy-depth` implies
    /// `--legacy-depth-as-time`).
    pub fn time_enabled(&self) -> bool {
        self.time.enabled && !self.rock_physics.legacy_toy_depth
    }

    /// Output time axis in time mode (`None` on the legacy axis).
    pub fn time_axis(&self) -> Option<TimeAxis> {
        self.time_enabled().then(|| TimeAxis {
            dt_ms: self.time.dt_ms,
            nt: self.time.output_samples(self.samples, self.rock_physics.depth_step_m),
            dz: self.rock_physics.depth_step_m,
            kernel: self.time.kernel,
            constant_twt_vp: self.time.constant_twt_vp,
        })
    }

    /// Output samples per trace: `nt` in time mode, else the depth samples.
    pub fn output_samples(&self) -> usize {
        self.time_axis().map_or(self.samples, |t| t.nt)
    }

    /// Output (deliverable) shape `(ni, nj, nt)`: angle stacks and every
    /// label cube written to MDIO. Equals [`E2eConfig::shape`] on the legacy
    /// axis.
    pub fn output_shape(&self) -> [usize; 3] {
        [self.inline_count, self.crossline_count, self.output_samples()]
    }

    /// Output sample interval (ms): `dt` in time mode, else 4 ms. Drives the
    /// Ricker, the Butterworth design, the noise mask and MDIO `digi`.
    pub fn digi_ms(&self) -> f64 {
        self.time_axis().map_or(TINY_DIGI, |t| t.dt_ms)
    }

    /// Validate the time settings (time mode only): see
    /// [`TimeConfig::validate`].
    pub fn validate_time(&self) -> Result<(), String> {
        if self.time_enabled() {
            self.time
                .validate(self.samples, self.rock_physics.depth_step_m, &self.filters)?;
        }
        Ok(())
    }

    /// The depth-staircase warning for this run (time mode only), see
    /// [`TimeConfig::staircase_warning`].
    pub fn time_warning(&self) -> Option<String> {
        self.time_enabled()
            .then(|| self.time.staircase_warning(self.rock_physics.depth_step_m, VP_MIN_SEDIMENT))
            .flatten()
    }

    /// The 40 Hz Ricker wavelet sampled at [`E2eConfig::digi_ms`].
    pub fn ricker(&self) -> Vec<f64> {
        ricker(RICKER_PEAK_HZ, self.digi_ms(), 1)
    }
}

/// Volumes produced by one deterministic pipeline pass.
#[derive(Debug, Clone)]
pub struct E2eVolumes {
    pub labels: Vec<u8>,
    pub angle_stack: Vec<f32>,
    pub shape: [usize; 3],
}

/// Full e2e report: volumes, parity vs second pass, optional store path.
#[derive(Debug, Clone)]
pub struct E2eReport {
    pub volumes: E2eVolumes,
    pub parity: ParityReport,
    pub store_path: Option<PathBuf>,
    pub status: &'static str,
}

/// Generate labels + angle stack for a tiny cube (deterministic for fixed seed).
pub fn generate_tiny_cube(cfg: &E2eConfig) -> E2eVolumes {
    let [ni, nj, nk] = cfg.shape();
    assert!(nk >= 2, "need at least 2 samples for reflectivity");

    // --- geo: toy horizon stack (layered dome by default, planar master) ---
    let (maps, nh) = crate::pipeline_stream::toy_horizon_maps(cfg);
    let mut labels = fill_layer_labels(&maps, [ni, nj, nh], nk);

    // --- closures: treat unset as background, relabel, drop tiny bodies ---
    let as_i32: Vec<i32> = labels
        .iter()
        .map(|&v| if v == 255 { 0 } else { (v as i32) + 1 })
        .collect();
    let (_vals, relabeled) = relabel_consecutive(&as_i32);
    let filtered = filter_labels_by_min_voxels(&relabeled, 1);
    for (dst, &src) in labels.iter_mut().zip(filtered.iter()) {
        *dst = if src == 0 {
            255
        } else {
            (src - 1).clamp(0, 254) as u8
        };
    }

    // --- faults (optional; no-op when cfg.faults.count == 0) ---
    crate::pipeline_stream::apply_faults_to_labels(cfg, &maps, nh, &mut labels);

    if cfg.time_enabled() {
        return time_tiny_cube(cfg, &labels, [ni, nj, nk]);
    }

    // --- RPM: elastic props (master toy trends or the rock-physics model) ---
    let mut vp = vec![0.0f32; ni * nj * nk];
    let mut vs = vec![0.0f32; ni * nj * nk];
    let mut rho = vec![0.0f32; ni * nj * nk];
    let model = crate::rock_physics::elastic_model(cfg, &labels, [ni, nj, nk]);
    model.tile_properties(&labels, [ni, nj, nk], 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);

    // --- seismic: single mid-angle RFC + Ricker wavelet → angle stack ---
    let angles = [15.0_f64];
    let rfc = compute_rfc_volumes_form(&vp, &vs, &rho, [ni, nj, nk], &angles, model.zoeppritz_form());
    // rfc shape: (1, ni, nj, nk-1) — pad last sample with 0 to match nk.
    let mut angle_cube = vec![0.0f32; ni * nj * nk];
    let zm1 = nk - 1;
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..zm1 {
                // Angle index 0 of the (1, ni, nj, nk - 1) RFC volume.
                let src = (i * nj + j) * zm1 + k;
                let dst = (i * nj + j) * nk + k;
                angle_cube[dst] = rfc[src];
            }
        }
    }
    // --- optional additive noise on the raw reflectivity (legacy order) ---
    let filters = crate::pipeline_stream::SeismicFilters::resolve(cfg, &labels, [ni, nj, nk])
        .unwrap_or_else(|e| panic!("invalid FilterConfig: {e}"));
    if let Some(noise) = filters.as_ref().and_then(|f| f.noise.as_ref()) {
        noise
            .at_angle(angles[0])
            .add_to_tile(&mut angle_cube, (0, ni), (0, nj), [ni, nj, nk]);
    }

    // Short wavelet: higher frequency keeps support reasonable for tiny nk.
    // Skipped (empty wavelet = identity) when the bandpass replaces it.
    let wavelet = cfg.ricker();
    let wavelet = crate::pipeline_stream::effective_wavelet(cfg, &wavelet);
    let mut angle_stack = apply_wavelet_traces(&angle_cube, [ni, nj, nk], wavelet);

    // --- optional post-convolution filters (no-op when disabled) ---
    crate::pipeline_stream::apply_filters_to_volume(cfg, &mut angle_stack);

    E2eVolumes {
        labels,
        angle_stack,
        shape: [ni, nj, nk],
    }
}

/// Time-mode classic path: one full-volume tile through the same fused
/// per-column chain as the chunked writers (props → T → depth Zoeppritz →
/// insertion → noise → Ricker at `dt` → bandpass → lateral), so classic,
/// chunked, strip and multiprocess outputs agree bit for bit (spec §5.4).
fn time_tiny_cube(cfg: &E2eConfig, labels: &[u8], shape: [usize; 3]) -> E2eVolumes {
    let [ni, nj, _] = shape;
    let model = crate::rock_physics::elastic_model(cfg, labels, shape);
    let filters = crate::pipeline_stream::SeismicFilters::resolve(cfg, labels, shape)
        .unwrap_or_else(|e| panic!("invalid FilterConfig: {e}"));
    let oshape = cfg.output_shape();
    let mut angle_stack = vec![0.0f32; oshape.iter().product()];
    let mut stats = crate::pipeline_stream::WorkingSetStats::default();
    crate::pipeline_stream::fuse_tile_filtered(
        labels,
        shape,
        0,
        ni,
        0,
        nj,
        &model,
        &cfg.ricker(),
        crate::pipeline_stream::DEFAULT_INCIDENCE_DEG,
        filters.as_ref(),
        &mut angle_stack,
        &mut stats,
    );
    let labels = crate::time_mode::generate_output_labels(cfg, labels, &model).labels;
    E2eVolumes {
        labels,
        angle_stack,
        shape: oshape,
    }
}

/// Write labels + angle stack into an MDIO store (create or overwrite path).
pub fn write_e2e_mdio(path: &Path, cfg: &E2eConfig, volumes: &E2eVolumes) -> Result<(), String> {
    let [ni, nj, nk] = volumes.shape;
    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(crate::pipeline_stream::resolve_chunk_shape(cfg)),
        digi: cfg.digi_ms(),
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(path, &create).map_err(|e| e.to_string())?;
    crate::time_mode::write_time_attrs(&store, cfg)?;
    DeliverableWriter::write_volume(&store, &volumes.angle_stack).map_err(|e| e.to_string())?;
    DeliverableWriter::write_labels(&store, &volumes.labels).map_err(|e| e.to_string())?;
    if let Some(mask) = crate::time_mode::generate_fault_labels_output(cfg) {
        store
            .write_fault_labels_u8(&mask)
            .map_err(|e| e.to_string())?;
    }
    if let Some(mask) = crate::time_mode::generate_salt_labels_output(cfg) {
        store.write_salt_labels_u8(&mask).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Run the full e2e path: generate → (optional MDIO write) → second-pass parity.
pub fn run_e2e(cfg: &E2eConfig) -> Result<E2eReport, String> {
    cfg.validate_time()?;
    crate::pipeline_stream::SeismicFilters::from_config(cfg)?;
    let volumes = generate_tiny_cube(cfg);
    let second = generate_tiny_cube(cfg);
    let parity = parity::compare_volumes(
        &volumes.labels,
        &second.labels,
        &volumes.angle_stack,
        &second.angle_stack,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "e2e self-parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    let mut store_path = None;
    if let Some(ref path) = cfg.store_path {
        write_e2e_mdio(path, cfg, &volumes)?;
        // Round-trip check from MDIO.
        let opened = MdioStore::open(path).map_err(|e| e.to_string())?;
        let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
        let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
        let mdio_parity = parity::compare_volumes(
            &volumes.labels,
            &back_labels,
            &volumes.angle_stack,
            &back_angles,
        );
        if !mdio_parity.passes_defaults() {
            return Err(format!(
                "e2e MDIO round-trip parity failed: {mdio_parity:?}"
            ));
        }
        store_path = Some(path.clone());
    }

    Ok(E2eReport {
        volumes,
        parity,
        store_path,
        status: "ok-e2e",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn e2e_self_parity_exact() {
        let cfg = E2eConfig::tiny(42);
        let a = generate_tiny_cube(&cfg);
        let b = generate_tiny_cube(&cfg);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.angle_stack, b.angle_stack);
        let report = parity::compare_volumes(&a.labels, &b.labels, &a.angle_stack, &b.angle_stack);
        assert!(report.passes_defaults());
        assert!((report.label_iou - 1.0).abs() < 1e-12);
        assert!(report.angle_mae == 0.0);
    }

    #[test]
    fn e2e_mdio_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("e2e.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            store_path: Some(path.clone()),
            chunk_shape: None,
            ..E2eConfig::tiny(7)
        };
        let report = run_e2e(&cfg).expect("e2e");
        assert_eq!(report.status, "ok-e2e");
        assert!(path
            .join("data")
            .join("chunked_012")
            .join(".zarray")
            .is_file());
        assert!(path.join("data").join("labels").join(".zarray").is_file());
        assert_eq!(report.volumes.shape, [8, 8, 8]);
        assert_eq!(report.volumes.labels.len(), 512);
        assert_eq!(report.volumes.angle_stack.len(), 512);
    }

    #[test]
    fn e2e_seed_changes_output() {
        let a = generate_tiny_cube(&E2eConfig::tiny(1));
        let b = generate_tiny_cube(&E2eConfig::tiny(99));
        // Different seeds should change plane coefficients → labels or angles.
        assert!(a.labels != b.labels || a.angle_stack != b.angle_stack);
    }
}
