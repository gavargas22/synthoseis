//! Synthoseis CLI binary stub.
//!
//! Single-worker local path only; cloud orchestration is out of scope for the skeleton.

use clap::{Parser, Subcommand};
use synthoseis_core::{JobPartition, RunConfig, SingleWorkerRunner};
use synthoseis_io::{DeliverableWriter, MdioStore, StoreMeta};

#[derive(Debug, Parser)]
#[command(name = "synthoseis", version, about = "Synthetic seismic generation (Rust skeleton)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run a single local worker job (placeholder; no algorithms yet).
    Run {
        /// RNG seed for reproducible stubs.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional path for a tiny MDIO-intent smoke store.
        #[arg(long)]
        store: Option<std::path::PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            // clap prints help when invoked with --help; bare invocation shows a short banner.
            println!("synthoseis {} — Rust skeleton. Try `synthoseis --help` or `synthoseis run`.", env!("CARGO_PKG_VERSION"));
        }
        Some(Commands::Run { seed, store }) => {
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
                let mdio = MdioStore::create(&path, &meta).expect("create MDIO-intent store");
                DeliverableWriter::write_smoke_volume(&mdio, &[0.0_f32; 16])
                    .expect("write smoke volume");
                println!("wrote MDIO-intent smoke store at {}", path.display());
            }
        }
    }
}
