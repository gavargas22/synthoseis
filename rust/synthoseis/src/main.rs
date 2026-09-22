//! Synthoseis CLI binary.
//!
//! Local multi-worker partition, strip-stitch, and multi-process JobPartitionPlan
//! e2e on a shared FS. Cloud K8s/AWS execution is out of scope.

use std::path::PathBuf;
use std::process::Command;

use clap::{Parser, Subcommand};
use synthoseis_core::{
    finalize_multiprocess_e2e, multiprocess_plan_path, prepare_multiprocess_store,
    run_worker_partition, JobPartition, JobPartitionPlan, MultiWorkerRunner, RunConfig,
    SingleWorkerRunner,
};
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
    /// Run a local generation job (single- or multi-worker partition).
    ///
    /// Without `--e2e`: placeholder summary (+ optional MDIO smoke write).
    /// With `--e2e`: tiny-cube pipeline geo→closures→RPM→seismic→MDIO + parity.
    /// With `--e2e --chunked --workers N` (N>1): strip-stitch multi-worker fused path.
    /// With `--e2e --chunked --multiprocess --workers N`: OS multi-process via plan artifact.
    /// With `--worker-id K --partition-plan P --store S --e2e --chunked`: worker-only mode.
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
            chunk_i,
            chunk_j,
            chunk_k,
            multiprocess,
            worker_id,
        }) => {
            let workers = workers.max(1);
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
            // Grid: e2e / multi-worker placeholder use 8³; single-worker smoke stays 2×2×4.
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

            // --- Worker mode (child process) ---
            if let Some(wid) = worker_id {
                if !(e2e && chunked) {
                    eprintln!("--worker-id requires --e2e --chunked");
                    std::process::exit(2);
                }
                let store_path = store.clone().unwrap_or_else(|| {
                    eprintln!("--worker-id requires --store");
                    std::process::exit(2);
                });
                let plan_path = partition_plan.clone().unwrap_or_else(|| {
                    eprintln!("--worker-id requires --partition-plan");
                    std::process::exit(2);
                });
                let json = std::fs::read_to_string(&plan_path).unwrap_or_else(|e| {
                    eprintln!("read partition plan {}: {e}", plan_path.display());
                    std::process::exit(1);
                });
                let plan = JobPartitionPlan::from_json(&json).unwrap_or_else(|e| {
                    eprintln!("parse partition plan: {e}");
                    std::process::exit(1);
                });
                let cfg = synthoseis_core::pipeline::E2eConfig {
                    seed,
                    inline_count,
                    crossline_count,
                    samples,
                    store_path: Some(store_path.clone()),
                    chunk_shape: chunk_shape.or_else(|| {
                        Some(synthoseis_core::pipeline_stream::resolve_chunk_shape(
                            &synthoseis_core::pipeline::E2eConfig {
                                seed,
                                inline_count,
                                crossline_count,
                                samples,
                                store_path: None,
                                chunk_shape: None,
                            },
                        ))
                    }),
                };
                let stats =
                    run_worker_partition(&cfg, &plan, wid, &store_path).unwrap_or_else(|e| {
                        eprintln!("worker {wid} failed: {e}");
                        std::process::exit(1);
                    });
                println!(
                    "worker {wid} complete: tiles={}, peak_temp_bytes={}, store={}",
                    stats.tiles_processed,
                    stats.peak_temp_bytes,
                    store_path.display()
                );
                return;
            }

            // Partition-plan write (orchestrator / smoke). Prefer chunk-aligned when chunked.
            if let Some(ref plan_path) = partition_plan {
                if !multiprocess {
                    let plan = if chunked || chunk_i.is_some() {
                        let ci = chunk_shape.map(|c| c[0]).unwrap_or_else(|| {
                            synthoseis_core::pipeline_stream::resolve_chunk_shape(
                                &synthoseis_core::pipeline::E2eConfig {
                                    seed,
                                    inline_count,
                                    crossline_count,
                                    samples,
                                    store_path: None,
                                    chunk_shape: None,
                                },
                            )[0]
                        });
                        JobPartitionPlan::from_config_chunk_aligned(&config, ci)
                    } else {
                        JobPartitionPlan::from_config(&config)
                    };
                    let json = plan.to_json().expect("serialize partition plan");
                    if let Some(parent) = plan_path.parent() {
                        if !parent.as_os_str().is_empty() {
                            std::fs::create_dir_all(parent).expect("create plan dir");
                        }
                    }
                    std::fs::write(plan_path, &json).expect("write partition plan");
                    let back = JobPartitionPlan::from_json(&json).expect("parse plan");
                    assert_eq!(back.worker_count, workers);
                    println!(
                        "wrote partition plan {} (workers={}, jobs={}, chunk_aligned={})",
                        plan_path.display(),
                        back.worker_count,
                        back.partitions
                            .iter()
                            .map(|p| p.job_ids.len())
                            .sum::<usize>(),
                        chunked || chunk_i.is_some()
                    );
                }
            }

            // --- Multiprocess orchestrator ---
            if multiprocess {
                if !(e2e && chunked) {
                    eprintln!("--multiprocess requires --e2e --chunked");
                    std::process::exit(2);
                }
                if workers <= 1 {
                    eprintln!("--multiprocess requires --workers N with N > 1");
                    std::process::exit(2);
                }
                let store_path = store.clone().unwrap_or_else(|| {
                    std::env::temp_dir().join(format!("synthoseis-mp-{}-{}.mdio", seed, workers))
                });
                let resolved = chunk_shape.or_else(|| {
                    Some(synthoseis_core::pipeline_stream::resolve_chunk_shape(
                        &synthoseis_core::pipeline::E2eConfig {
                            seed,
                            inline_count,
                            crossline_count,
                            samples,
                            store_path: None,
                            chunk_shape: None,
                        },
                    ))
                });
                let cfg = synthoseis_core::pipeline::E2eConfig {
                    seed,
                    inline_count,
                    crossline_count,
                    samples,
                    store_path: Some(store_path.clone()),
                    chunk_shape: resolved,
                };
                let (store_path, plan, _prep) = prepare_multiprocess_store(&cfg, workers, resolved)
                    .unwrap_or_else(|e| {
                        eprintln!("prepare multiprocess failed: {e}");
                        std::process::exit(1);
                    });
                let plan_path = partition_plan
                    .clone()
                    .unwrap_or_else(|| multiprocess_plan_path(&store_path));
                // Ensure plan is at the path children will load (prepare already wrote default).
                if plan_path != multiprocess_plan_path(&store_path) {
                    std::fs::write(&plan_path, plan.to_json().expect("plan json"))
                        .expect("write partition plan");
                }
                println!(
                    "multiprocess prepare: store={} plan={} workers={}",
                    store_path.display(),
                    plan_path.display(),
                    plan.worker_count
                );

                let exe = std::env::current_exe().expect("current_exe");
                let mut children = Vec::with_capacity(plan.worker_count);
                for wid in 0..plan.worker_count {
                    // Deterministic child argv (documented in SCALE.md):
                    // synthoseis run --e2e --chunked --worker-id K --partition-plan PLAN
                    //   --store STORE --seed S --workers N [--chunk-i/j/k ...]
                    let mut cmd = Command::new(&exe);
                    cmd.arg("run")
                        .arg("--e2e")
                        .arg("--chunked")
                        .arg("--worker-id")
                        .arg(wid.to_string())
                        .arg("--partition-plan")
                        .arg(&plan_path)
                        .arg("--store")
                        .arg(&store_path)
                        .arg("--seed")
                        .arg(seed.to_string())
                        .arg("--workers")
                        .arg(workers.to_string());
                    if let Some([ci, cj, ck]) = resolved {
                        cmd.arg("--chunk-i")
                            .arg(ci.to_string())
                            .arg("--chunk-j")
                            .arg(cj.to_string())
                            .arg("--chunk-k")
                            .arg(ck.to_string());
                    }
                    let child = cmd.spawn().unwrap_or_else(|e| {
                        eprintln!("spawn worker {wid}: {e}");
                        std::process::exit(1);
                    });
                    children.push((wid, child));
                }
                for (wid, mut child) in children {
                    let status = child.wait().unwrap_or_else(|e| {
                        eprintln!("wait worker {wid}: {e}");
                        std::process::exit(1);
                    });
                    if !status.success() {
                        eprintln!("worker {wid} exited with {status}");
                        std::process::exit(1);
                    }
                }

                let (report, stats) =
                    finalize_multiprocess_e2e(&cfg, &store_path).unwrap_or_else(|e| {
                        eprintln!("finalize multiprocess failed: {e}");
                        std::process::exit(1);
                    });
                println!(
                    "multiprocess e2e complete: seed={}, workers={}, shape={:?}, chunks={:?}, status={}, peak_temp_bytes={}, parity(iou={:.4}, agr={:.4}, mae={:.3e}, maxabs={:.3e})",
                    seed,
                    workers,
                    report.volumes.shape,
                    resolved,
                    report.status,
                    stats.peak_temp_bytes,
                    report.parity.label_iou,
                    report.parity.label_agreement,
                    report.parity.angle_mae,
                    report.parity.angle_max_abs
                );
                if let Some(path) = report.store_path {
                    println!(
                        "wrote shared MDIO labels+angle-stack at {} (multi-process JobPartitionPlan, parity vs single-worker ok)",
                        path.display()
                    );
                }
                return;
            }

            if e2e {
                // Strip-stitch: --e2e --chunked --workers N>1
                if chunked && workers > 1 {
                    let store_path = store.clone().unwrap_or_else(|| {
                        std::env::temp_dir()
                            .join(format!("synthoseis-strip-{}-{}.mdio", seed, workers))
                    });
                    let resolved = chunk_shape.or_else(|| {
                        Some(synthoseis_core::pipeline_stream::resolve_chunk_shape(
                            &synthoseis_core::pipeline::E2eConfig {
                                seed,
                                inline_count,
                                crossline_count,
                                samples,
                                store_path: None,
                                chunk_shape: None,
                            },
                        ))
                    });
                    let runner = MultiWorkerRunner::new(config.clone());
                    let (report, stats) = runner
                        .run_e2e_strip_stitched(Some(store_path), resolved)
                        .unwrap_or_else(|e| {
                            eprintln!("strip-stitch e2e failed: {e}");
                            std::process::exit(1);
                        });
                    println!(
                        "strip-stitch e2e complete: seed={}, workers={}, shape={:?}, chunks={:?}, status={}, peak_temp_bytes={}, parity(iou={:.4}, agr={:.4}, mae={:.3e}, maxabs={:.3e})",
                        seed,
                        workers,
                        report.volumes.shape,
                        resolved,
                        report.status,
                        stats.peak_temp_bytes,
                        report.parity.label_iou,
                        report.parity.label_agreement,
                        report.parity.angle_mae,
                        report.parity.angle_max_abs
                    );
                    if let Some(path) = report.store_path {
                        println!(
                            "wrote shared MDIO labels+angle-stack at {} (strip-stitch, parity vs single-worker ok)",
                            path.display()
                        );
                    }
                } else if chunked || chunk_shape.is_some() {
                    let cfg = synthoseis_core::pipeline::E2eConfig {
                        seed,
                        inline_count,
                        crossline_count,
                        samples,
                        store_path: store.clone(),
                        chunk_shape: chunk_shape.or_else(|| {
                            Some(synthoseis_core::pipeline_stream::resolve_chunk_shape(
                                &synthoseis_core::pipeline::E2eConfig {
                                    seed,
                                    inline_count,
                                    crossline_count,
                                    samples,
                                    store_path: None,
                                    chunk_shape: None,
                                },
                            ))
                        }),
                    };
                    let result = if overlap {
                        synthoseis_core::run_e2e_streaming_overlapped(&cfg)
                    } else {
                        synthoseis_core::pipeline_stream::run_e2e_chunked(&cfg)
                    };
                    let (report, stats) = result.unwrap_or_else(|e| {
                        eprintln!("chunked e2e failed: {e}");
                        std::process::exit(1);
                    });
                    let mode = if overlap {
                        "overlapped chunked"
                    } else {
                        "chunked"
                    };
                    println!(
                        "{mode} e2e complete: seed={}, workers={}, shape={:?}, chunks={:?}, status={}, peak_temp_bytes={}, parity(iou={:.4}, agr={:.4}, mae={:.3e}, maxabs={:.3e})",
                        seed,
                        workers,
                        report.volumes.shape,
                        cfg.chunk_shape,
                        report.status,
                        stats.peak_temp_bytes,
                        report.parity.label_iou,
                        report.parity.label_agreement,
                        report.parity.angle_mae,
                        report.parity.angle_max_abs
                    );
                    if let Some(path) = report.store_path {
                        println!(
                            "wrote MDIO labels+angle-stack at {} (sub-volume chunks, parity round-trip ok)",
                            path.display()
                        );
                    }
                } else {
                    let config = config.clone();
                    let runner = MultiWorkerRunner::new(config);
                    let report = runner.run_e2e(store.clone()).unwrap_or_else(|e| {
                        eprintln!("e2e failed: {e}");
                        std::process::exit(1);
                    });
                    println!(
                        "e2e complete: seed={}, workers={}, shape={:?}, status={}, parity(iou={:.4}, agr={:.4}, mae={:.3e}, maxabs={:.3e})",
                        seed,
                        workers,
                        report.volumes.shape,
                        report.status,
                        report.parity.label_iou,
                        report.parity.label_agreement,
                        report.parity.angle_mae,
                        report.parity.angle_max_abs
                    );
                    if workers > 1 {
                        println!(
                            "note: non-chunked e2e runs the full cube once; use --chunked --workers N for strip-stitch or --multiprocess for OS processes"
                        );
                    }
                    if let Some(path) = report.store_path {
                        println!(
                            "wrote MDIO labels+angle-stack at {} (parity round-trip ok)",
                            path.display()
                        );
                    }
                }
            } else if workers == 1 {
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
            } else {
                let summary = MultiWorkerRunner::new(config).run_placeholder();
                println!(
                    "multi-worker run complete: seed={}, workers={}, jobs={}, status={}",
                    summary.seed, summary.workers, summary.job_count, summary.status
                );
                for (i, w) in summary.per_worker.iter().enumerate() {
                    println!("  worker[{i}]: jobs={}, status={}", w.job_count, w.status);
                }
            }
        }
    }
}
