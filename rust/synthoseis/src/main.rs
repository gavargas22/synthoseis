//! Synthoseis CLI binary.
//!
//! Local multi-worker partition, strip-stitch, and multi-process JobPartitionPlan
//! e2e on a shared FS. Cloud K8s/AWS execution is out of scope.

mod cli_e2e;
mod cli_jobs;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use synthoseis_core::RunConfig;

#[derive(Debug, Parser)]
#[command(
    name = "synthoseis",
    version,
    about = "Synthetic seismic generation (Rust rewrite)"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run a local generation job (single- or multi-worker partition).
    ///
    /// Without `--e2e`: placeholder summary (+ optional MDIO smoke write).
    /// With `--e2e`: tiny-cube pipeline geo→closures→RPM→seismic→MDIO + parity.
    /// With `--e2e --chunked --workers N` (N>1): strip-stitch multi-worker fused path.
    /// With `--e2e --chunked --multiprocess --workers N`: OS multi-process via plan artifact.
    /// With `--worker-id K --partition-plan P --store S --e2e --chunked`: worker-only mode.
    /// With `--e2e --chunked --angles 0,15,30`: geometry once / seismic many.
    /// With `--gpu`: prefer per-tile GPU fuse (CPU software fallback when no device).
    Run {
        /// RNG seed for reproducible stubs / e2e.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Number of local workers for job partition (default 1).
        ///
        /// Contiguous chunks over `inline × crossline` job ids. When jobs <
        /// workers, some workers get empty lists (slots are kept).
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// Optional path for an MDIO store (smoke write, or e2e labels+angles).
        #[arg(long)]
        store: Option<PathBuf>,
        /// Write (or, with existing file semantics, emit) a JSON partition plan
        /// for cloud handoff smoke (`JobPartitionPlan`).
        #[arg(long)]
        partition_plan: Option<PathBuf>,
        /// Run the end-to-end tiny-cube (8³) pipeline with parity.
        #[arg(long, default_value_t = false)]
        e2e: bool,
        /// Use memory-bounded fused chunked e2e (elastic+RFC+wavelet per tile).
        #[arg(long, default_value_t = false)]
        chunked: bool,
        /// Overlap single-worker tile compute with a one-deep std writer thread.
        #[arg(long, default_value_t = false)]
        overlap: bool,
        /// Comma-separated incidence angles in degrees (geometry once / seismic many).
        /// Example: `--angles 0,15,30`. Writes sibling stores `{stem}_a{deg}.mdio`.
        #[arg(long)]
        angles: Option<String>,
        /// Shortcut: first N angles from the default list [0, 15, 30, ...].
        #[arg(long)]
        seismic_many: Option<usize>,
        /// Chunk size along inline (with --chunked / --e2e MDIO sub-volume chunks).
        #[arg(long)]
        chunk_i: Option<usize>,
        /// Chunk size along crossline.
        #[arg(long)]
        chunk_j: Option<usize>,
        /// Chunk size along samples (default: full nk).
        #[arg(long)]
        chunk_k: Option<usize>,
        /// Orchestrator: prepare store + chunk-aligned plan, spawn N child processes
        /// each with `--worker-id`, wait, finalize, print parity.
        #[arg(long, default_value_t = false)]
        multiprocess: bool,
        /// Worker mode: load `--partition-plan`, write only that partition into
        /// `--store` (must already exist). Requires `--e2e --chunked --store --partition-plan`.
        #[arg(long)]
        worker_id: Option<usize>,
        /// Prefer per-tile GPU fuse (`synthoseis-gpu` auto path). Falls back to the
        /// CPU software adapter when no device is available (CI-safe no-op path).
        #[arg(long, default_value_t = false)]
        gpu: bool,
        /// Insert N faults (port of datagenerator/Faults.py random mode; seeded
        /// from --seed). Default 0 = no faulting (bit-identical outputs).
        /// Currently requires single-worker `--e2e --chunked`; also writes
        /// `data/fault_labels` into the MDIO store.
        #[arg(long, default_value_t = 0)]
        faults: usize,
        /// Cube shape `NI,NJ,NK` for single-worker `--e2e --chunked` (default 8,8,8).
        #[arg(long)]
        shape: Option<String>,
        /// Legacy post-convolution Butterworth bandpass `LOW,HIGH[,ORDER]` in Hz
        /// (port of Seismic.apply_bandlimits; zero-phase filtfilt, order 4 by
        /// default). Off by default. Requires single-worker `--e2e --chunked`
        /// and more than 6*ORDER+3 samples per trace.
        #[arg(long)]
        bandpass: Option<String>,
        /// Legacy lateral box filter size N (port of
        /// Seismic.apply_lateral_filter; legacy draws 1, 3 or 5). Default 1 = off.
        /// Requires single-worker `--e2e --chunked`.
        #[arg(long, default_value_t = 1)]
        lateral_filter: usize,
        /// With `--bandpass`: keep the Ricker wavelet convolution (the first
        /// filter port's Ricker + bandpass). Default: the bandpass replaces
        /// the Ricker wavelet (legacy chain: reflectivity, then bandpass).
        #[arg(long, default_value_t = false)]
        keep_ricker: bool,
        /// With `--bandpass` (Ricker skipped): also bandpass the trailing
        /// reflectivity sample, reproducing master before the trailing-sample
        /// fix bit for bit. Default: bandpass the first NK-1 samples, exactly
        /// the trace legacy filters, and write the trailing sample as 0.
        /// Legacy axis only (`--legacy-depth-as-time`): in time mode, when the
        /// bandpass replaces the Ricker, the dead last sample NT-1 is always
        /// 0; with the Ricker on there is no dead sample.
        #[arg(long, default_value_t = false)]
        bandpass_trailing_sample: bool,
        /// Add deterministic random noise at this signal-to-noise ratio (dB)
        /// to the raw reflectivity before the wavelet / bandpass (port of
        /// Seismic.add_weighted_noise; legacy example config 7.5-17.5 dB).
        /// Off by default. Requires single-worker `--e2e --chunked`.
        #[arg(long)]
        noise_snr_db: Option<f64>,
        /// Noise seed (default: `--seed`). Requires `--noise-snr-db`.
        #[arg(long)]
        noise_seed: Option<u64>,
        /// Use the exact legacy noise angle weights (`math.cos` of the angle
        /// in degrees). Default: correct radian weights. Requires
        /// `--noise-snr-db`.
        #[arg(long, default_value_t = false)]
        noise_legacy_weights: bool,
        /// Normalise the noise with the exact legacy `data_std` mask
        /// (`k >= wb / (digi + 15) * digi`, ~0.84 x the seabed sample, so it
        /// includes part of the water column). Default: the cutoff is the
        /// actual seabed. Requires `--noise-snr-db`.
        #[arg(long, default_value_t = false)]
        noise_legacy_seabed: bool,
        /// Reproduce master 10f4dcd elastic properties bit for bit: toy depth
        /// `k * 100 m` from the cube top and label >= 2 (including 255) as oil
        /// sand, legacy Zoeppritz (implies `--legacy-zoeppritz`). Default: the
        /// corrected legacy rock physics (4 m per sample, one depth per layer
        /// below the seabed, water column, net-to-gross mixing, closure
        /// fluids). See docs/rock-physics-port.md.
        #[arg(long, default_value_t = false)]
        legacy_toy_depth: bool,
        /// Evaluate Zoeppritz with the legacy kernel's `det` typo (bit-exact
        /// to master 33a3a93) instead of the textbook Aki & Richards / bruges
        /// expression (default). See docs/zoeppritz-fix.md.
        #[arg(long, default_value_t = false)]
        legacy_zoeppritz: bool,
        /// Sand/shale mixing: `inverse-velocity` (legacy default) or `backus`.
        #[arg(long, default_value = "inverse-velocity")]
        mixing: String,
        /// Constant net-to-gross for every sand voxel (0-1). Default: legacy
        /// random net-to-gross maps per sand layer.
        #[arg(long)]
        net_to_gross: Option<f32>,
        /// Random per-layer depth shifts apply to legacy layers deeper than
        /// this (legacy `first_random_lyr`, default 20).
        #[arg(long, default_value_t = 20)]
        first_random_layer: usize,
        /// All sands brine (no oil / gas from closures).
        #[arg(long, default_value_t = false)]
        no_fluids: bool,
        /// Toy horizon geometry: `layered` (default: legacy-style layer cake
        /// over a dome, ~nk/6 layers, closures) or `planar` (master geometry:
        /// 2 dipping planes, 2 layers). `--legacy-toy-depth` implies planar.
        /// See docs/layered-toy-geometry.md.
        #[arg(long)]
        toy_geometry: Option<String>,
        /// Per-layer lithology of the layered geometry: `markov` (default:
        /// legacy sand-fraction Markov chain, `create_facies_array`) or
        /// `alternating` (even layers shale, odd sand; the previous default).
        /// The planar geometry always alternates. See docs/toy-lithology.md.
        #[arg(long)]
        toy_lithology: Option<String>,
        /// Fixed model sand fraction (0-1) for `--toy-lithology markov`.
        /// Default: legacy per-model draw U(0.05, 0.25).
        #[arg(long)]
        sand_layer_fraction: Option<f64>,
        /// Mean sand unit thickness in layers for `--toy-lithology markov`
        /// (legacy `sand_layer_thickness`, default 2).
        #[arg(long)]
        sand_layer_thickness: Option<f64>,
        /// Legacy switch: closures on every sand layer's own top (master
        /// 8b5988f) instead of on the top of each sand unit (consecutive sand
        /// layers merged, deepest unit skipped, as legacy `Closures`). See
        /// docs/closures-per-sand-unit.md.
        #[arg(long, default_value_t = false)]
        closures_per_layer: bool,
        /// Legacy switch: closures per sand unit without 3D segmentation
        /// across faults (master ef2dc42). See
        /// docs/closure-segmentation-faults.md.
        #[arg(long, default_value_t = false)]
        closures_unsegmented: bool,
        /// Legacy switch: fixed 500-voxel closure minimum (master bad1daa8,
        /// legacy min_closure_voxels_simple for its 300 x 300 x 1250 cube).
        /// Default: scaled with the map area, clamp(round(NI*NJ/180), 20,
        /// 500) whole cells (20 up to 60 x 60, 23 at 64 x 64, 91 at
        /// 128 x 128, 500 from 300 x 300). See docs/rock-physics-port.md.
        #[arg(long, default_value_t = false)]
        legacy_closure_minimum: bool,
        /// Fixed closure minimum of N >= 1 whole cells (smaller closure
        /// compartments stay brine), instead of the scaled default.
        #[arg(long)]
        min_closure_voxels: Option<usize>,
        /// Legacy switch: cap the fluid contact of segmented closures at the
        /// integer unit base cell (master bad1daa8). Default: base + 1/2
        /// cell, so partial voxels fill a full-unit trap down to the true
        /// sub-cell sand base. Rejected where it has no effect
        /// (--legacy-whole-voxels, planar, --closures-unsegmented,
        /// --closures-per-layer). See docs/partial-voxels.md.
        #[arg(long, default_value_t = false)]
        legacy_closure_contact_cap: bool,
        /// Legacy switch: no salt body (master b4f4259). Salt is on by
        /// default for the layered geometry (legacy `include_salt: true`).
        /// See docs/salt-bodies.md.
        #[arg(long, default_value_t = false)]
        no_salt: bool,
        /// Legacy switch: absolute U(150, 300)-sample salt top offset below
        /// horizon 1 (legacy 1250-sample cubes) instead of the default
        /// scaled by min(samples / 1250, 1).
        #[arg(long, default_value_t = false)]
        salt_legacy_top_offset: bool,
        /// Legacy switch: smooth every horizon map after the salt-flank lift
        /// (`G(m+L)`), restoring master ccce5cc9 salt-on output. Default is
        /// lift-only: `m + G(L)`. Rejected with --no-salt, planar, or
        /// --legacy-toy-depth. See docs/salt-bodies.md.
        #[arg(long, default_value_t = false)]
        salt_smooth_all_horizons: bool,
        /// Legacy switch: salt Vs = 2250 m/s (master f15b87ac), instead of
        /// the default 2600. Density and Vp stay 2.17 / 4500. Rejected with
        /// --no-salt, planar, or --legacy-toy-depth. See docs/salt-bodies.md.
        #[arg(long, default_value_t = false)]
        salt_legacy_vs: bool,
        /// Legacy switch: write the depth-sampled seismic as time (each depth
        /// sample = one 4 ms sample, an implied constant 2000 m/s), master
        /// f3720fb2 bit for bit. Default: convert to two-way time from the
        /// voxel Vp. `--legacy-toy-depth` implies it. See
        /// docs/depth-to-time.md.
        #[arg(long, default_value_t = false)]
        legacy_depth_as_time: bool,
        /// Output sample interval in ms (0.5-8.0, default 4). Time mode only.
        #[arg(long)]
        dt_ms: Option<f64>,
        /// Output trace length in time samples (default: nz * 2 dz / 2000 m/s
        /// / dt, = NK at the defaults; 16 <= N <= 8 x NK). Time mode only.
        #[arg(long)]
        twt_samples: Option<usize>,
        /// Reflectivity insertion kernel: `sinc` (default, Kaiser-windowed)
        /// or `linear` (fast 2-tap split; aliases above ~0.4 f_N, so best
        /// with the bandpass-only chain). Time mode only.
        #[arg(long)]
        twt_kernel: Option<String>,
        /// Legacy switch: keep fault labels inside the salt body (master
        /// 2b3850ba). By default data/fault_labels is fault AND NOT salt.
        /// See docs/salt-bodies.md.
        #[arg(long, default_value_t = false)]
        fault_labels_through_salt: bool,
        /// Opt out of partial voxels (on by default since PR B2): whole-voxel
        /// rasterisation, byte for byte as master d8b96e69's default. Stacks
        /// change by default without it. Rejected with --toy-geometry planar
        /// / --legacy-toy-depth (whole-voxel by construction). See
        /// docs/partial-voxels.md.
        #[arg(long, default_value_t = false)]
        legacy_whole_voxels: bool,
        /// Partial voxels (on by default for the layered geometry): exact
        /// vertical volume fractions of every depth cell. `subcell` (the
        /// time-mode default): every sub-cell interface at its exact two-way
        /// time (time mode only); `cell` (the --legacy-depth-as-time
        /// default): Backus voxels with the cell-to-cell reflectivity. Opt
        /// out with --legacy-whole-voxels. See docs/partial-voxels.md.
        #[arg(long)]
        partial_voxel_reflectivity: Option<String>,
    },
}

/// `--legacy-whole-voxels` / `--partial-voxel-reflectivity` (partial-voxels
/// spec §2, §4). Partial voxels are on by default for the layered geometry
/// (subcell in time mode, cell on the legacy axis); `--legacy-whole-voxels`
/// opts out. The planar geometry and `--legacy-toy-depth` are whole-voxel
/// by construction. The resolved config always carries an explicit
/// reflectivity, so multi-process workers receive it verbatim.
fn parse_partial_voxels(
    legacy_whole_voxels: bool,
    reflectivity: Option<&str>,
    geometry: synthoseis_core::ToyGeometry,
    time_enabled: bool,
) -> Result<synthoseis_core::partial_voxels::PartialVoxelConfig, String> {
    use synthoseis_core::partial_voxels::{PartialVoxelConfig, PvReflectivity};
    let planar = geometry == synthoseis_core::ToyGeometry::Planar;
    if legacy_whole_voxels && planar {
        return Err("--legacy-whole-voxels has no effect with --toy-geometry planar or --legacy-toy-depth (whole-voxel by construction)".into());
    }
    let Some(r) = reflectivity else {
        if legacy_whole_voxels || planar {
            return Ok(PartialVoxelConfig::whole_voxels());
        }
        let r = if time_enabled {
            PvReflectivity::Subcell
        } else {
            PvReflectivity::Cell
        };
        return Ok(PartialVoxelConfig::with(r));
    };
    if legacy_whole_voxels {
        return Err("--partial-voxel-reflectivity has no effect with --legacy-whole-voxels".into());
    }
    let r = PvReflectivity::parse(r)?;
    if planar {
        return Err("--partial-voxel-reflectivity requires the layered geometry (not --toy-geometry planar / --legacy-toy-depth)".into());
    }
    if r == PvReflectivity::Subcell && !time_enabled {
        return Err("--partial-voxel-reflectivity subcell requires the time axis (not --legacy-depth-as-time); use cell".into());
    }
    Ok(PartialVoxelConfig::with(r))
}

/// `true` would make `--legacy-closure-minimum` also restore the integer
/// contact cap (one combined opt-out for both closure changes, master
/// bad1daa8), wherever the cap has an effect.
const LEGACY_CLOSURE_MINIMUM_IMPLIES_CONTACT_CAP: bool = false;

/// `--legacy-closure-minimum` / `--min-closure-voxels` /
/// `--legacy-closure-contact-cap` (closure-minimum spec §5). Runs after
/// [`parse_partial_voxels`]: the contact cap only has an effect on the
/// segmented partial-voxel path.
fn apply_closures(
    mut rock: synthoseis_core::RockPhysicsConfig,
    geometry: synthoseis_core::ToyGeometry,
    legacy_minimum: bool,
    min_voxels: Option<usize>,
    legacy_contact_cap: bool,
) -> Result<synthoseis_core::RockPhysicsConfig, String> {
    use synthoseis_core::ClosureMinimum;
    if legacy_minimum && min_voxels.is_some() {
        return Err("--legacy-closure-minimum and --min-closure-voxels are mutually exclusive".into());
    }
    if min_voxels == Some(0) {
        return Err("--min-closure-voxels expects N >= 1".into());
    }
    let any = legacy_minimum || min_voxels.is_some() || legacy_contact_cap;
    if any && (!rock.fluids || rock.legacy_toy_depth) {
        return Err("--legacy-closure-minimum / --min-closure-voxels / --legacy-closure-contact-cap have no effect with --no-fluids or --legacy-toy-depth".into());
    }
    let planar = geometry == synthoseis_core::ToyGeometry::Planar;
    let cap_applies =
        !(planar || rock.closures_unsegmented || rock.closures_per_layer || !rock.partial_voxels.enabled());
    if legacy_contact_cap && !cap_applies {
        return Err("--legacy-closure-contact-cap has no effect with --legacy-whole-voxels, --toy-geometry planar, --closures-unsegmented or --closures-per-layer".into());
    }
    rock.closure_minimum = match (legacy_minimum, min_voxels) {
        (true, _) => ClosureMinimum::LEGACY,
        (false, Some(n)) => ClosureMinimum::Fixed(n),
        (false, None) => ClosureMinimum::Scaled,
    };
    rock.legacy_closure_contact_cap =
        legacy_contact_cap || (LEGACY_CLOSURE_MINIMUM_IMPLIES_CONTACT_CAP && legacy_minimum && cap_applies);
    Ok(rock)
}

/// `--legacy-depth-as-time` / `--dt-ms` / `--twt-samples` / `--twt-kernel`
/// (spec §2). The time options are rejected with either legacy switch
/// (`--legacy-toy-depth` implies the legacy axis), and
/// `--bandpass-trailing-sample` is legacy-axis only: in time mode the dead
/// last sample `nt - 1` is zeroed whenever the bandpass replaces the Ricker
/// (spec §3.7), and there is no dead sample when the Ricker is on.
fn parse_time(
    legacy_depth_as_time: bool,
    legacy_toy_depth: bool,
    dt_ms: Option<f64>,
    twt_samples: Option<usize>,
    twt_kernel: Option<&str>,
    bandpass_trailing_sample: bool,
) -> Result<synthoseis_core::TimeConfig, String> {
    let time_opts = dt_ms.is_some() || twt_samples.is_some() || twt_kernel.is_some();
    if (legacy_depth_as_time || legacy_toy_depth) && time_opts {
        return Err(
            "--dt-ms / --twt-samples / --twt-kernel have no effect with --legacy-depth-as-time or --legacy-toy-depth"
                .into(),
        );
    }
    if legacy_depth_as_time || legacy_toy_depth {
        return Ok(synthoseis_core::TimeConfig::legacy());
    }
    if bandpass_trailing_sample {
        return Err(
            "--bandpass-trailing-sample requires --legacy-depth-as-time (in time mode the dead last sample nt-1 is zeroed whenever the bandpass replaces the Ricker)"
                .into(),
        );
    }
    let mut t = synthoseis_core::TimeConfig::default();
    if let Some(dt) = dt_ms {
        if !dt.is_finite() {
            return Err(format!("--dt-ms must be finite, got {dt}"));
        }
        t.dt_ms = dt;
    }
    t.samples = twt_samples;
    if let Some(k) = twt_kernel {
        t.kernel = synthoseis_core::TwtKernel::parse(k)
            .ok_or_else(|| format!("--twt-kernel expects sinc or linear, got {k:?}"))?;
    }
    Ok(t)
}

fn resolve_chunk_shape_cli(
    _seed: u64,
    inline_count: usize,
    crossline_count: usize,
    samples: usize,
    chunk_i: Option<usize>,
    chunk_j: Option<usize>,
    chunk_k: Option<usize>,
    chunked: bool,
) -> Option<[usize; 3]> {
    match (chunk_i, chunk_j, chunk_k) {
        (None, None, None) if !chunked => None,
        _ => {
            let (ni, nj, nk) = (inline_count, crossline_count, samples);
            Some([
                chunk_i.unwrap_or(ni / 2).max(1).min(ni),
                chunk_j.unwrap_or(nj / 2).max(1).min(nj),
                chunk_k.unwrap_or(nk).max(1).min(nk),
            ])
        }
    }
}

fn parse_shape(s: &str) -> Result<(usize, usize, usize), String> {
    let v: Vec<usize> = s
        .split(',')
        .map(|t| t.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("--shape: {e}"))?;
    match v.as_slice() {
        [a, b, c] if *a >= 1 && *b >= 1 && *c >= 2 => Ok((*a, *b, *c)),
        _ => Err(format!("--shape expects NI,NJ,NK (NK >= 2), got {s:?}")),
    }
}

fn parse_filters(
    bandpass: Option<&str>,
    lateral_filter: usize,
    keep_ricker: bool,
    bandpass_trailing_sample: bool,
    noise: synthoseis_core::NoiseConfig,
) -> Result<synthoseis_core::FilterConfig, String> {
    if keep_ricker && bandpass.is_none() {
        return Err("--keep-ricker requires --bandpass".into());
    }
    if bandpass_trailing_sample && bandpass.is_none() {
        return Err("--bandpass-trailing-sample requires --bandpass".into());
    }
    if !noise.enabled()
        && (noise.seed.is_some() || noise.legacy_angle_weights || noise.legacy_seabed)
    {
        return Err(
            "--noise-seed / --noise-legacy-weights / --noise-legacy-seabed require --noise-snr-db"
                .into(),
        );
    }
    if let Some(db) = noise.snr_db {
        if !db.is_finite() {
            return Err(format!("--noise-snr-db must be finite, got {db}"));
        }
    }
    let mut fc = synthoseis_core::FilterConfig {
        lateral_size: lateral_filter.max(1),
        keep_ricker,
        bandpass_trailing_sample,
        noise,
        ..Default::default()
    };
    if let Some(s) = bandpass {
        let v: Vec<f64> = s
            .split(',')
            .map(|t| t.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("--bandpass: {e}"))?;
        match v.as_slice() {
            [lo, hi] => fc.bandpass_hz = Some([*lo, *hi]),
            [lo, hi, ord] if *ord >= 1.0 && ord.fract() == 0.0 => {
                fc.bandpass_hz = Some([*lo, *hi]);
                fc.bandpass_order = *ord as usize;
            }
            _ => return Err(format!("--bandpass expects LOW,HIGH[,ORDER], got {s:?}")),
        }
    }
    Ok(fc)
}

fn parse_rock_physics(
    legacy_toy_depth: bool,
    legacy_zoeppritz: bool,
    mixing: &str,
    net_to_gross: Option<f32>,
    first_random_layer: usize,
    no_fluids: bool,
    closures_per_layer: bool,
    closures_unsegmented: bool,
) -> Result<synthoseis_core::RockPhysicsConfig, String> {
    let mixing = match mixing {
        "inverse-velocity" | "inv-vel" => synthoseis_core::MixingMethod::InverseVelocity,
        "backus" => synthoseis_core::MixingMethod::BackusModuli,
        other => {
            return Err(format!(
                "--mixing expects inverse-velocity or backus, got {other:?}"
            ))
        }
    };
    let defaults = synthoseis_core::RockPhysicsConfig::default();
    let customised = mixing != defaults.mixing
        || net_to_gross.is_some()
        || first_random_layer != defaults.first_random_layer
        || no_fluids;
    if legacy_toy_depth && customised {
        return Err(
            "--mixing / --net-to-gross / --first-random-layer / --no-fluids have no effect with --legacy-toy-depth"
                .into(),
        );
    }
    if closures_unsegmented && (legacy_toy_depth || no_fluids || closures_per_layer) {
        return Err(
            "--closures-unsegmented has no effect with --legacy-toy-depth, --no-fluids or --closures-per-layer"
                .into(),
        );
    }
    if closures_per_layer && (legacy_toy_depth || no_fluids) {
        return Err(
            "--closures-per-layer has no effect with --legacy-toy-depth or --no-fluids".into(),
        );
    }
    let rp = synthoseis_core::RockPhysicsConfig {
        legacy_toy_depth,
        // `--legacy-toy-depth` implies the legacy Zoeppritz (master guarantee).
        legacy_zoeppritz: legacy_zoeppritz || legacy_toy_depth,
        mixing,
        net_to_gross: match net_to_gross {
            Some(v) => synthoseis_core::NetToGross::Constant(v),
            None => defaults.net_to_gross.clone(),
        },
        first_random_layer,
        fluids: !no_fluids,
        closures_per_layer,
        closures_unsegmented,
        ..defaults
    };
    rp.validate()?;
    Ok(rp)
}

/// `--toy-geometry` (default layered). `--legacy-toy-depth` implies planar,
/// so an explicit `layered` with it is rejected.
fn parse_geometry(
    arg: Option<&str>,
    legacy_toy_depth: bool,
) -> Result<synthoseis_core::ToyGeometry, String> {
    let g = match arg {
        Some(s) => synthoseis_core::ToyGeometry::parse(s)?,
        None => synthoseis_core::ToyGeometry::default(),
    };
    if legacy_toy_depth && arg == Some("layered") {
        return Err(
            "--toy-geometry layered has no effect with --legacy-toy-depth (planar master geometry)"
                .into(),
        );
    }
    Ok(if legacy_toy_depth {
        synthoseis_core::ToyGeometry::Planar
    } else {
        g
    })
}

/// `--toy-lithology` / `--sand-layer-fraction` / `--sand-layer-thickness`.
/// The planar geometry (and `--legacy-toy-depth`) always alternates, so the
/// Markov options are rejected there; so are the sand options with
/// `alternating`.
fn apply_lithology(
    mut rock: synthoseis_core::RockPhysicsConfig,
    geometry: synthoseis_core::ToyGeometry,
    arg: Option<&str>,
    sand_layer_fraction: Option<f64>,
    sand_layer_thickness: Option<f64>,
) -> Result<synthoseis_core::RockPhysicsConfig, String> {
    let lith = match arg {
        Some(s) => synthoseis_core::ToyLithology::parse(s)?,
        None => synthoseis_core::ToyLithology::default(),
    };
    let sand_opts = sand_layer_fraction.is_some() || sand_layer_thickness.is_some();
    if geometry == synthoseis_core::ToyGeometry::Planar {
        if arg == Some("markov") || sand_opts {
            return Err(
                "--toy-lithology markov / --sand-layer-fraction / --sand-layer-thickness have no effect with the planar geometry (always alternating)"
                    .into(),
            );
        }
        if rock.closures_unsegmented {
            return Err(
                "--closures-unsegmented has no effect with the planar geometry (always per layer)"
                    .into(),
            );
        }
        if rock.closures_per_layer && !rock.legacy_toy_depth {
            return Err(
                "--closures-per-layer has no effect with the planar geometry (always per layer)"
                    .into(),
            );
        }
        rock.lithology = synthoseis_core::ToyLithology::Alternating;
        return Ok(rock);
    }
    if lith == synthoseis_core::ToyLithology::Alternating && sand_opts {
        return Err(
            "--sand-layer-fraction / --sand-layer-thickness have no effect with --toy-lithology alternating"
                .into(),
        );
    }
    rock.lithology = lith;
    rock.sand_layer_fraction = sand_layer_fraction;
    if let Some(t) = sand_layer_thickness {
        rock.sand_layer_thickness = t;
    }
    rock.validate()?;
    Ok(rock)
}

/// `--no-salt` / `--salt-legacy-top-offset` / `--salt-smooth-all-horizons` /
/// `--salt-legacy-vs` / `--fault-labels-through-salt`: salt exists only in
/// the layered geometry, so all are rejected with the planar geometry (and
/// `--legacy-toy-depth`); the offset, smooth-all, legacy-vs and fault-label
/// switches are rejected with `--no-salt`, and the fault-label switch
/// without faults.
fn apply_salt(
    mut rock: synthoseis_core::RockPhysicsConfig,
    geometry: synthoseis_core::ToyGeometry,
    no_salt: bool,
    legacy_top_offset: bool,
    smooth_all_horizons: bool,
    salt_legacy_vs: bool,
    fault_labels_through_salt: bool,
    faults: usize,
) -> Result<synthoseis_core::RockPhysicsConfig, String> {
    if fault_labels_through_salt {
        if geometry == synthoseis_core::ToyGeometry::Planar {
            return Err(
                "--fault-labels-through-salt has no effect with the planar geometry or --legacy-toy-depth (no salt)"
                    .into(),
            );
        }
        if no_salt {
            return Err("--fault-labels-through-salt has no effect with --no-salt".into());
        }
        if faults == 0 {
            return Err("--fault-labels-through-salt has no effect without --faults (no fault labels)".into());
        }
    }
    rock.fault_labels_through_salt = fault_labels_through_salt;
    if geometry == synthoseis_core::ToyGeometry::Planar
        && (no_salt || legacy_top_offset || smooth_all_horizons || salt_legacy_vs)
    {
        return Err(
            "--no-salt / --salt-legacy-top-offset / --salt-smooth-all-horizons / --salt-legacy-vs have no effect with the planar geometry or --legacy-toy-depth (no salt)"
                .into(),
        );
    }
    if no_salt && legacy_top_offset {
        return Err("--salt-legacy-top-offset has no effect with --no-salt".into());
    }
    if no_salt && smooth_all_horizons {
        return Err("--salt-smooth-all-horizons has no effect with --no-salt".into());
    }
    if no_salt && salt_legacy_vs {
        return Err("--salt-legacy-vs has no effect with --no-salt".into());
    }
    rock.salt = !no_salt;
    rock.salt_legacy_top_offset = legacy_top_offset;
    rock.salt_smooth_all_horizons = smooth_all_horizons;
    rock.salt_legacy_vs = salt_legacy_vs;
    Ok(rock)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            println!(
                "synthoseis {} — try `synthoseis --help` or `synthoseis run --e2e --store /tmp/e2e.mdio`.",
                env!("CARGO_PKG_VERSION")
            );
        }
        Some(Commands::Run {
            seed,
            workers,
            store,
            partition_plan,
            e2e,
            chunked,
            overlap,
            angles,
            seismic_many,
            chunk_i,
            chunk_j,
            chunk_k,
            multiprocess,
            worker_id,
            gpu,
            faults,
            shape,
            bandpass,
            lateral_filter,
            keep_ricker,
            bandpass_trailing_sample,
            noise_snr_db,
            noise_seed,
            noise_legacy_weights,
            noise_legacy_seabed,
            legacy_toy_depth,
            legacy_zoeppritz,
            mixing,
            net_to_gross,
            first_random_layer,
            no_fluids,
            toy_geometry,
            toy_lithology,
            sand_layer_fraction,
            sand_layer_thickness,
            closures_per_layer,
            closures_unsegmented,
            legacy_closure_minimum,
            min_closure_voxels,
            legacy_closure_contact_cap,
            no_salt,
            salt_legacy_top_offset,
            salt_smooth_all_horizons,
            salt_legacy_vs,
            legacy_depth_as_time,
            dt_ms,
            twt_samples,
            twt_kernel,
            fault_labels_through_salt,
            legacy_whole_voxels,
            partial_voxel_reflectivity,
        }) => {
            let workers = workers.max(1);
            synthoseis_gpu::set_prefer_gpu(gpu);
            if gpu {
                eprintln!(
                    "gpu: requested; backend={}",
                    synthoseis_gpu::backend_status()
                );
            }
            if overlap && !(e2e && chunked) {
                eprintln!("--overlap requires --e2e --chunked");
                std::process::exit(2);
            }
            if overlap && (workers > 1 || multiprocess || worker_id.is_some()) {
                eprintln!("--overlap currently supports only single-worker, non-multiprocess runs");
                std::process::exit(2);
            }
            if overlap && store.is_none() {
                eprintln!("--overlap requires --store");
                std::process::exit(2);
            }
            let geo_many = angles.is_some() || seismic_many.is_some();
            if geo_many && !(e2e && chunked) {
                eprintln!("--angles / --seismic-many require --e2e --chunked");
                std::process::exit(2);
            }
            if geo_many && (workers > 1 || multiprocess || worker_id.is_some() || overlap) {
                eprintln!(
                    "--angles / --seismic-many currently support only single-worker non-overlap runs"
                );
                std::process::exit(2);
            }
            if (faults > 0 || shape.is_some())
                && !(e2e
                    && chunked
                    && workers == 1
                    && !multiprocess
                    && worker_id.is_none()
                    && !overlap)
            {
                eprintln!(
                    "--faults / --shape currently require single-worker `--e2e --chunked` (no --overlap / --multiprocess / --worker-id)"
                );
                std::process::exit(2);
            }
            let noise = synthoseis_core::NoiseConfig {
                snr_db: noise_snr_db,
                seed: noise_seed,
                legacy_angle_weights: noise_legacy_weights,
                legacy_seabed: noise_legacy_seabed,
            };
            let filters = parse_filters(
                bandpass.as_deref(),
                lateral_filter,
                keep_ricker,
                bandpass_trailing_sample,
                noise,
            )
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
            if filters.enabled()
                && !(e2e && chunked && workers == 1 && !multiprocess && worker_id.is_none())
            {
                eprintln!(
                    "--bandpass / --lateral-filter / --noise-snr-db currently require single-worker `--e2e --chunked` (no --multiprocess / --worker-id)"
                );
                std::process::exit(2);
            }
            let rock = parse_rock_physics(
                legacy_toy_depth,
                legacy_zoeppritz,
                &mixing,
                net_to_gross,
                first_random_layer,
                no_fluids,
                closures_per_layer,
                closures_unsegmented,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            let geometry = parse_geometry(toy_geometry.as_deref(), legacy_toy_depth)
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                });
            let rock = apply_lithology(
                rock,
                geometry,
                toy_lithology.as_deref(),
                sand_layer_fraction,
                sand_layer_thickness,
            )
            .and_then(|r| {
                apply_salt(
                    r,
                    geometry,
                    no_salt,
                    salt_legacy_top_offset,
                    salt_smooth_all_horizons,
                    salt_legacy_vs,
                    fault_labels_through_salt,
                    faults,
                )
            })
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            let (inline_count, crossline_count, samples) = if let Some(ref s) = shape {
                parse_shape(s).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                })
            } else if e2e || workers > 1 || multiprocess {
                (8, 8, 8)
            } else {
                (2, 2, 4)
            };
            let time = parse_time(
                legacy_depth_as_time,
                legacy_toy_depth,
                dt_ms,
                twt_samples,
                twt_kernel.as_deref(),
                bandpass_trailing_sample,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            let mut rock = rock;
            rock.partial_voxels = parse_partial_voxels(
                legacy_whole_voxels,
                partial_voxel_reflectivity.as_deref(),
                geometry,
                time.enabled,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            let rock = apply_closures(
                rock,
                geometry,
                legacy_closure_minimum,
                min_closure_voxels,
                legacy_closure_contact_cap,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            // Validate the output axis against the depth model and filters
            // (dt range, nt range, output Nyquist) before any work: exit 2.
            let probe = synthoseis_core::pipeline::E2eConfig {
                seed,
                inline_count,
                crossline_count,
                samples,
                filters: filters.clone(),
                rock_physics: rock.clone(),
                geometry,
                time: time.clone(),
                ..Default::default()
            };
            if e2e {
                if let Err(e) = probe.validate_time() {
                    eprintln!("{e}");
                    std::process::exit(2);
                }
            }
            // Output samples per trace (`nt` in time mode): the default
            // `--chunk-k` and every chunk clamp are along the output axis.
            let out_samples = probe.output_samples();
            let config = RunConfig {
                seed,
                workers,
                inline_count,
                crossline_count,
                samples,
            };
            let chunk_shape = resolve_chunk_shape_cli(
                seed,
                inline_count,
                crossline_count,
                out_samples,
                chunk_i,
                chunk_j,
                chunk_k,
                chunked,
            );

            if cli_jobs::maybe_run_worker(
                worker_id,
                e2e,
                chunked,
                &store,
                &partition_plan,
                seed,
                inline_count,
                crossline_count,
                samples,
                chunk_shape,
                &rock,
                geometry,
                &time,
            ) {
                return;
            }

            cli_jobs::maybe_write_partition_plan(
                &partition_plan,
                multiprocess,
                chunked,
                chunk_i,
                &config,
                workers,
                seed,
                inline_count,
                crossline_count,
                samples,
                chunk_shape,
                &time,
            );

            if cli_jobs::maybe_run_multiprocess(
                multiprocess,
                e2e,
                chunked,
                workers,
                &store,
                &partition_plan,
                seed,
                inline_count,
                crossline_count,
                samples,
                chunk_shape,
                &rock,
                geometry,
                &time,
            ) {
                return;
            }

            if e2e {
                cli_e2e::run_e2e(
                    geo_many,
                    &angles,
                    seismic_many,
                    chunked,
                    workers,
                    overlap,
                    &store,
                    seed,
                    inline_count,
                    crossline_count,
                    samples,
                    chunk_shape,
                    &config,
                    faults,
                    filters,
                    rock,
                    geometry,
                    time,
                );
            } else if workers == 1 {
                cli_e2e::run_single_worker_placeholder(&config, store, seed);
            } else {
                cli_e2e::run_multi_worker_placeholder(config);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use synthoseis_core::{ClosureMinimum, RockPhysicsConfig};

    /// Multi-process workers rebuild the closure options from these flags,
    /// and clap accepts them.
    #[test]
    fn closure_options_are_forwarded_to_workers() {
        for (minimum, cap, want) in [
            (ClosureMinimum::Scaled, false, vec![]),
            (ClosureMinimum::LEGACY, false, vec!["--legacy-closure-minimum"]),
            (ClosureMinimum::Fixed(7), true, vec!["--min-closure-voxels", "7", "--legacy-closure-contact-cap"]),
            // `--min-closure-voxels 500` keeps its own label (same threshold
            // as the legacy switch).
            (ClosureMinimum::Fixed(500), false, vec!["--min-closure-voxels", "500"]),
        ] {
            let rock = RockPhysicsConfig {
                closure_minimum: minimum,
                legacy_closure_contact_cap: cap,
                partial_voxels: synthoseis_core::partial_voxels::PartialVoxelConfig::with(
                    synthoseis_core::partial_voxels::PvReflectivity::Subcell,
                ),
                ..RockPhysicsConfig::default()
            };
            let args = crate::cli_jobs::rock_physics_args(&rock);
            let closure: Vec<&str> = args
                .iter()
                .map(String::as_str)
                .skip_while(|a| !a.contains("closure"))
                .take_while(|a| a.contains("closure") || a.parse::<usize>().is_ok())
                .collect();
            assert_eq!(closure, want, "{minimum:?} cap {cap}");
            let mut argv = vec!["synthoseis".to_string(), "run".into(), "--e2e".into(), "--chunked".into()];
            argv.extend(args);
            let cli = <super::Cli as clap::Parser>::try_parse_from(argv).expect("workers accept the flags");
            let Some(super::Commands::Run {
                legacy_closure_minimum,
                min_closure_voxels,
                legacy_closure_contact_cap,
                ..
            }) = cli.command
            else {
                panic!()
            };
            let back = match (legacy_closure_minimum, min_closure_voxels) {
                (true, _) => ClosureMinimum::LEGACY,
                (false, Some(n)) => ClosureMinimum::Fixed(n),
                (false, None) => ClosureMinimum::Scaled,
            };
            assert_eq!((back, legacy_closure_contact_cap), (minimum, cap));
        }
    }

    /// Rebuild `(rock, geometry, time)` from a `run` argv through the same
    /// parse chain as `main` (rock physics, geometry, lithology, salt, time,
    /// partial voxels, closures). Used for the orchestrator's own flags and
    /// for the argv a multi-process worker receives.
    fn rebuild_from_argv(
        argv: &[String],
    ) -> (RockPhysicsConfig, synthoseis_core::ToyGeometry, synthoseis_core::TimeConfig) {
        let cli = <super::Cli as clap::Parser>::try_parse_from(argv).expect("clap accepts the argv");
        let Some(super::Commands::Run {
            faults,
            bandpass_trailing_sample,
            legacy_toy_depth,
            legacy_zoeppritz,
            mixing,
            net_to_gross,
            first_random_layer,
            no_fluids,
            toy_geometry,
            toy_lithology,
            sand_layer_fraction,
            sand_layer_thickness,
            closures_per_layer,
            closures_unsegmented,
            legacy_closure_minimum,
            min_closure_voxels,
            legacy_closure_contact_cap,
            no_salt,
            salt_legacy_top_offset,
            salt_smooth_all_horizons,
            salt_legacy_vs,
            legacy_depth_as_time,
            dt_ms,
            twt_samples,
            twt_kernel,
            fault_labels_through_salt,
            legacy_whole_voxels,
            partial_voxel_reflectivity,
            ..
        }) = cli.command
        else {
            panic!("expected `run`")
        };
        let rock = super::parse_rock_physics(
            legacy_toy_depth,
            legacy_zoeppritz,
            &mixing,
            net_to_gross,
            first_random_layer,
            no_fluids,
            closures_per_layer,
            closures_unsegmented,
        )
        .unwrap();
        let geometry = super::parse_geometry(toy_geometry.as_deref(), legacy_toy_depth).unwrap();
        let rock = super::apply_lithology(
            rock,
            geometry,
            toy_lithology.as_deref(),
            sand_layer_fraction,
            sand_layer_thickness,
        )
        .unwrap();
        let mut rock = super::apply_salt(
            rock,
            geometry,
            no_salt,
            salt_legacy_top_offset,
            salt_smooth_all_horizons,
            salt_legacy_vs,
            fault_labels_through_salt,
            faults,
        )
        .unwrap();
        let time = super::parse_time(
            legacy_depth_as_time,
            legacy_toy_depth,
            dt_ms,
            twt_samples,
            twt_kernel.as_deref(),
            bandpass_trailing_sample,
        )
        .unwrap();
        rock.partial_voxels = super::parse_partial_voxels(
            legacy_whole_voxels,
            partial_voxel_reflectivity.as_deref(),
            geometry,
            time.enabled,
        )
        .unwrap();
        let rock = super::apply_closures(
            rock,
            geometry,
            legacy_closure_minimum,
            min_closure_voxels,
            legacy_closure_contact_cap,
        )
        .unwrap();
        (rock, geometry, time)
    }

    /// `--salt-legacy-vs` (and the other salt switches) reach multi-process
    /// workers, and a salt-bearing cube run through the multi-process path
    /// equals one process for each Vs setting; the two settings differ.
    ///
    /// The CLI cannot run this end to end with real worker processes:
    /// `--multiprocess` and `--worker-id` reject `--shape`, and the 8³ default
    /// never contains salt (the salt top is ~10–20 samples down). So each
    /// worker's config is rebuilt here from the exact flags the orchestrator
    /// passes (`rock_physics_args` + `time_args` + `--toy-geometry`, see
    /// `cli_jobs::maybe_run_multiprocess`) through `main`'s parse chain, and
    /// the workers' library calls (`prepare_multiprocess_store`,
    /// `run_worker_partition` per worker, `finalize_multiprocess_e2e`) run
    /// in this process on a 12×10×64 cube (seed 42, 212 salt voxels in depth).
    /// `rock_physics_cli.rs::lithology_flags_reach_multiprocess_workers`
    /// runs the real worker processes with the same flags on the 8³ cube.
    #[test]
    fn salt_flags_reach_multiprocess_workers() {
        use synthoseis_core::pipeline::E2eConfig;
        use synthoseis_core::{
            finalize_multiprocess_e2e, prepare_multiprocess_store, run_worker_partition,
        };
        let argv = |extra: &[&str]| -> Vec<String> {
            ["synthoseis", "run", "--e2e", "--chunked"]
                .iter()
                .chain(extra)
                .map(|s| s.to_string())
                .collect()
        };
        // Every salt switch round-trips through the worker flags.
        for extra in [
            &[][..],
            &["--salt-legacy-vs"][..],
            &["--salt-smooth-all-horizons"][..],
            &["--salt-legacy-top-offset"][..],
            &["--salt-legacy-vs", "--salt-smooth-all-horizons", "--salt-legacy-top-offset"][..],
            &["--no-salt"][..],
            &["--salt-legacy-vs", "--legacy-depth-as-time"][..],
            &["--salt-legacy-vs", "--legacy-whole-voxels"][..],
        ] {
            let (rock, geometry, time) = rebuild_from_argv(&argv(extra));
            let mut worker = argv(&[]);
            worker.extend(crate::cli_jobs::rock_physics_args(&rock));
            worker.extend(crate::cli_jobs::time_args(&time));
            if !rock.partial_voxels.enabled() {
                worker.push("--legacy-whole-voxels".into());
            }
            worker.extend(["--toy-geometry".to_string(), geometry.as_str().to_string()]);
            let back = rebuild_from_argv(&worker);
            assert_eq!(back, (rock.clone(), geometry, time.clone()), "{extra:?}: worker config");
            assert_eq!(
                worker.iter().any(|a| a == "--salt-legacy-vs"),
                extra.contains(&"--salt-legacy-vs"),
                "{extra:?}: --salt-legacy-vs in the worker argv"
            );
        }

        // Multi-process (workers rebuilt from their argv) vs one process,
        // on a salt-bearing cube, with and without --salt-legacy-vs.
        let dir = tempfile::tempdir().unwrap();
        let read = |p: &std::path::Path| -> (Vec<u32>, Vec<u8>, Vec<u8>) {
            let s = synthoseis_io::MdioStore::open(p).unwrap();
            (
                s.read_volume().unwrap().iter().map(|x| x.to_bits()).collect(),
                s.read_labels_u8().unwrap(),
                s.read_salt_labels_u8().unwrap(),
            )
        };
        let mut stacks = Vec::new();
        for extra in [&[][..], &["--salt-legacy-vs"][..]] {
            let (rock, geometry, time) = rebuild_from_argv(&argv(extra));
            let cfg = |rock: RockPhysicsConfig, store: &std::path::Path| E2eConfig {
                seed: 42,
                inline_count: 12,
                crossline_count: 10,
                samples: 64,
                chunk_shape: Some([2, 4, 8]),
                rock_physics: rock,
                geometry,
                time: time.clone(),
                store_path: Some(store.to_path_buf()),
                ..E2eConfig::default()
            };
            let tag = if extra.is_empty() { "default" } else { "legacy-vs" };
            let single = dir.path().join(format!("{tag}-single.mdio"));
            synthoseis_core::pipeline_stream::run_e2e_chunked(&cfg(rock.clone(), &single)).unwrap();

            let mut worker = argv(&[]);
            worker.extend(crate::cli_jobs::rock_physics_args(&rock));
            worker.extend(crate::cli_jobs::time_args(&time));
            worker.extend(["--toy-geometry".to_string(), geometry.as_str().to_string()]);
            let (worker_rock, worker_geometry, worker_time) = rebuild_from_argv(&worker);
            assert_eq!((worker_geometry, &worker_time), (geometry, &time));
            let mp = dir.path().join(format!("{tag}-mp.mdio"));
            let orchestrator = cfg(rock.clone(), &mp);
            let (path, plan, _) = prepare_multiprocess_store(&orchestrator, 3, orchestrator.chunk_shape).unwrap();
            assert_eq!(plan.worker_count, 3);
            let worker_cfg = cfg(worker_rock, &mp);
            for wid in 0..plan.worker_count {
                run_worker_partition(&worker_cfg, &plan, wid, &path).unwrap();
            }
            finalize_multiprocess_e2e(&orchestrator, &path).unwrap();

            let (s1, l1, salt1) = read(&single);
            let (s2, l2, salt2) = read(&mp);
            assert!(salt1.contains(&1), "{tag}: salt in the (time-domain) cube");
            assert_eq!(s1, s2, "{tag}: multi-process angle stack vs one process");
            assert_eq!(l1, l2, "{tag}: labels");
            assert_eq!(salt1, salt2, "{tag}: salt_labels");
            stacks.push(s1);
        }
        assert_ne!(stacks[0], stacks[1], "Vs 2600 and --salt-legacy-vs (Vs 2250) must differ");
    }
}
