//! Worker-mode, partition-plan write, and multiprocess orchestrator CLI paths.
use std::path::PathBuf;
use std::process::Command;

use synthoseis_core::{
    finalize_multiprocess_e2e, multiprocess_plan_path, prepare_multiprocess_store,
    run_worker_partition, JobPartitionPlan, RunConfig,
};

/// Returns true if worker mode ran (caller should return).
pub fn maybe_run_worker(
    worker_id: Option<usize>,
    e2e: bool,
    chunked: bool,
    store: &Option<PathBuf>,
    partition_plan: &Option<PathBuf>,
    seed: u64,
    inline_count: usize,
    crossline_count: usize,
    samples: usize,
    chunk_shape: Option<[usize; 3]>,
) -> bool {
    let Some(wid) = worker_id else {
        return false;
    };
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
    let stats = run_worker_partition(&cfg, &plan, wid, &store_path).unwrap_or_else(|e| {
        eprintln!("worker {wid} failed: {e}");
        std::process::exit(1);
    });
    println!(
        "worker {wid} complete: tiles={}, peak_temp_bytes={}, store={}",
        stats.tiles_processed,
        stats.peak_temp_bytes,
        store_path.display()
    );
    true
}

pub fn maybe_write_partition_plan(
    partition_plan: &Option<PathBuf>,
    multiprocess: bool,
    chunked: bool,
    chunk_i: Option<usize>,
    config: &RunConfig,
    workers: usize,
    seed: u64,
    inline_count: usize,
    crossline_count: usize,
    samples: usize,
    chunk_shape: Option<[usize; 3]>,
) {
    let Some(ref plan_path) = partition_plan else {
        return;
    };
    if multiprocess {
        return;
    }
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
        JobPartitionPlan::from_config_chunk_aligned(config, ci)
    } else {
        JobPartitionPlan::from_config(config)
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

/// Returns true if multiprocess ran (caller should return).
pub fn maybe_run_multiprocess(
    multiprocess: bool,
    e2e: bool,
    chunked: bool,
    workers: usize,
    store: &Option<PathBuf>,
    partition_plan: &Option<PathBuf>,
    seed: u64,
    inline_count: usize,
    crossline_count: usize,
    samples: usize,
    chunk_shape: Option<[usize; 3]>,
) -> bool {
    if !multiprocess {
        return false;
    }
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

    let (report, stats) = finalize_multiprocess_e2e(&cfg, &store_path).unwrap_or_else(|e| {
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
    true
}
