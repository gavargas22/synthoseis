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
    },
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
) -> Result<synthoseis_core::FilterConfig, String> {
    let mut fc = synthoseis_core::FilterConfig {
        lateral_size: lateral_filter.max(1),
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
            let filters = parse_filters(bandpass.as_deref(), lateral_filter).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            });
            if filters.enabled()
                && !(e2e && chunked && workers == 1 && !multiprocess && worker_id.is_none())
            {
                eprintln!(
                    "--bandpass / --lateral-filter currently require single-worker `--e2e --chunked` (no --multiprocess / --worker-id)"
                );
                std::process::exit(2);
            }
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
                samples,
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
                );
            } else if workers == 1 {
                cli_e2e::run_single_worker_placeholder(&config, store, seed);
            } else {
                cli_e2e::run_multi_worker_placeholder(config);
            }
        }
    }
}
