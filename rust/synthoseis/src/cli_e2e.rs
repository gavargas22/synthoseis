//! E2e / placeholder CLI paths including geometry-once / seismic-many.
use std::path::PathBuf;

use synthoseis_core::{
    angles_from_seismic_many, parse_angles_csv, run_e2e_geometry_once_seismic_many, JobPartition,
    MultiWorkerRunner, RunConfig, SingleWorkerRunner,
};
use synthoseis_io::{DeliverableWriter, MdioStore, StoreMeta};

pub fn run_e2e(
    geo_many: bool,
    angles: &Option<String>,
    seismic_many: Option<usize>,
    chunked: bool,
    workers: usize,
    overlap: bool,
    store: &Option<PathBuf>,
    seed: u64,
    inline_count: usize,
    crossline_count: usize,
    samples: usize,
    chunk_shape: Option<[usize; 3]>,
    config: &RunConfig,
) {
    if geo_many {
        let angle_list = if let Some(ref csv) = angles {
            parse_angles_csv(csv).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            })
        } else {
            angles_from_seismic_many(seismic_many.unwrap()).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2);
            })
        };
        let cfg = synthoseis_core::pipeline::E2eConfig {
            seed,
            inline_count,
            crossline_count,
            samples,
            store_path: store.clone(),
            chunk_shape: chunk_shape.or_else(|| {
                Some(synthoseis_core::resolve_chunk_shape(
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
        let (report, stats) = run_e2e_geometry_once_seismic_many(&cfg, &angle_list).unwrap_or_else(
            |e| {
                eprintln!("geometry-once seismic-many failed: {e}");
                std::process::exit(1);
            },
        );
        println!(
            "geometry-once seismic-many complete: seed={}, angles={:?}, stacks={}, labels_generated={}, status={}, peak_temp_bytes={}",
            seed,
            angle_list,
            report.stacks.len(),
            report.labels_generated,
            report.status,
            stats.peak_temp_bytes
        );
        for st in &report.stacks {
            println!(
                "  angle={:.0}° parity(iou={:.4}, mae={:.3e}) store={:?}",
                st.angle_deg,
                st.parity.label_iou,
                st.parity.angle_mae,
                st.store_path.as_ref().map(|p| p.display().to_string())
            );
        }
        return;
    }

    if chunked && workers > 1 {
        let store_path = store.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("synthoseis-strip-{}-{}.mdio", seed, workers))
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
        return;
    }

    if chunked || chunk_shape.is_some() {
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
        return;
    }

    let runner = MultiWorkerRunner::new(config.clone());
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

pub fn run_single_worker_placeholder(config: &RunConfig, store: Option<PathBuf>, seed: u64) {
    let partition = JobPartition::single_worker(config);
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

pub fn run_multi_worker_placeholder(config: RunConfig) {
    let summary = MultiWorkerRunner::new(config).run_placeholder();
    println!(
        "multi-worker run complete: seed={}, workers={}, jobs={}, status={}",
        summary.seed, summary.workers, summary.job_count, summary.status
    );
    for (i, w) in summary.per_worker.iter().enumerate() {
        println!("  worker[{i}]: jobs={}, status={}", w.job_count, w.status);
    }
}
