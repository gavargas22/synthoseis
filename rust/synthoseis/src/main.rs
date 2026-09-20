//! Synthoseis CLI binary.
//!
//! Single-worker local path only; cloud orchestration is out of scope.

use clap::{Parser, Subcommand};
use synthoseis_core::{JobPartition, RunConfig, SingleWorkerRunner};
use synthoseis_io::{DeliverableWriter, MdioStore, StoreMeta};

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
    /// Run a single local worker job.
    ///
    /// Without `--e2e`: placeholder summary (+ optional MDIO smoke write).
    /// With `--e2e`: tiny-cube pipeline geo→closures→RPM→seismic→MDIO + parity.
    Run {
        /// RNG seed for reproducible stubs / e2e.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional path for an MDIO store (smoke write, or e2e labels+angles).
        #[arg(long)]
        store: Option<std::path::PathBuf>,
        /// Run the end-to-end tiny-cube (8³) single-worker pipeline with parity.
        #[arg(long, default_value_t = false)]
        e2e: bool,
    },
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
        Some(Commands::Run { seed, store, e2e }) => {
            if e2e {
                let config = RunConfig {
                    seed,
                    workers: 1,
                    inline_count: 8,
                    crossline_count: 8,
                    samples: 8,
                };
                let partition = JobPartition::single_worker(&config);
                let runner = SingleWorkerRunner::new(config, partition);
                let report = runner
                    .run_e2e(store.clone())
                    .unwrap_or_else(|e| {
                        eprintln!("e2e failed: {e}");
                        std::process::exit(1);
                    });
                println!(
                    "e2e single-worker complete: seed={}, shape={:?}, status={}, parity(iou={:.4}, agr={:.4}, mae={:.3e}, maxabs={:.3e})",
                    seed,
                    report.volumes.shape,
                    report.status,
                    report.parity.label_iou,
                    report.parity.label_agreement,
                    report.parity.angle_mae,
                    report.parity.angle_max_abs
                );
                if let Some(path) = report.store_path {
                    println!(
                        "wrote MDIO labels+angle-stack at {} (parity round-trip ok)",
                        path.display()
                    );
                }
            } else {
                let config = RunConfig {
                    seed,
                    workers: 1,
                    ..RunConfig::default()
                };
                let partition = JobPartition::single_worker(&config);
                let runner = SingleWorkerRunner::new(config.clone(), partition);
                let summary = runner.run_placeholder();
                println!(
                    "single-worker run complete: seed={}, workers={}, jobs={}, status={}",
                    summary.seed, summary.workers, summary.job_count, summary.status
                );

                if let Some(path) = store {
                    let meta = StoreMeta {
                        dims: ["inline".into(), "crossline".into(), "time".into()],
                        shape: [2, 2, 4],
                        digi: 4.0,
                        seed,
                        units: "ms".into(),
                    };
                    let mdio = MdioStore::create(&path, &meta).expect("create MDIO store");
                    let samples: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
                    DeliverableWriter::write_smoke_volume(&mdio, &samples).expect("write volume");
                    let back = mdio.read_volume().expect("read volume");
                    assert_eq!(back.len(), 16);
                    assert_eq!(mdio.shape(), [2, 2, 4]);
                    assert_eq!(mdio.read_live_mask().expect("live_mask"), vec![1, 1, 1, 1]);
                    println!(
                        "wrote MDIO store at {} (shape {:?}, live traces marked)",
                        path.display(),
                        mdio.shape()
                    );
                }
            }
        }
    }
}
