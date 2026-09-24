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
            let (inline_count, crossline_count, samples) = if e2e || workers > 1 || multiprocess {
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
                );
            } else if workers == 1 {
                cli_e2e::run_single_worker_placeholder(&config, store, seed);
            } else {
                cli_e2e::run_multi_worker_placeholder(config);
            }
        }
    }
}
