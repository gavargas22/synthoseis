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
    faults: usize,
    filters: synthoseis_core::FilterConfig,
    rock: synthoseis_core::RockPhysicsConfig,
    geometry: synthoseis_core::ToyGeometry,
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
            faults: synthoseis_core::FaultConfig::with_count(faults),
            filters: filters.clone(),
            seed,
            inline_count,
            crossline_count,
            samples,
            store_path: store.clone(),
            chunk_shape: chunk_shape.or_else(|| {
                Some(synthoseis_core::resolve_chunk_shape(
                    &synthoseis_core::pipeline::E2eConfig {
                        faults: Default::default(),
                        filters: Default::default(),
                        seed,
                        inline_count,
                        crossline_count,
                        samples,
                        store_path: None,
                        chunk_shape: None,
                        rock_physics: Default::default(),
                        geometry: Default::default(),
                    },
                ))
            }),
            rock_physics: rock.clone(),
            geometry,
        };
        let (report, stats) =
            run_e2e_geometry_once_seismic_many(&cfg, &angle_list).unwrap_or_else(|e| {
                eprintln!("geometry-once seismic-many failed: {e}");
                std::process::exit(1);
            });
        println!(
            "geometry-once seismic-many complete: seed={}, angles={:?}, stacks={}, labels_generated={}, status={}, peak_temp_bytes={}",
            seed,
            angle_list,
            report.stacks.len(),
            report.labels_generated,
            report.status,
            stats.peak_temp_bytes
        );
        print_fault_summary(&cfg);
        print_filter_summary(&cfg);
        print_rock_summary(&cfg.rock_physics);
        print_geometry_summary(&cfg);
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
                    faults: Default::default(),
                    filters: Default::default(),
                    seed,
                    inline_count,
                    crossline_count,
                    samples,
                    store_path: None,
                    chunk_shape: None,
                    rock_physics: Default::default(),
                    geometry: Default::default(),
                },
            ))
        });
        let runner = MultiWorkerRunner::new(config.clone());
        let (report, stats) = runner
            .run_e2e_strip_stitched_with_geometry(Some(store_path), resolved, &rock, geometry)
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
            faults: synthoseis_core::FaultConfig::with_count(faults),
            filters: filters.clone(),
            seed,
            inline_count,
            crossline_count,
            samples,
            store_path: store.clone(),
            chunk_shape: chunk_shape.or_else(|| {
                Some(synthoseis_core::pipeline_stream::resolve_chunk_shape(
                    &synthoseis_core::pipeline::E2eConfig {
                        faults: Default::default(),
                        filters: Default::default(),
                        seed,
                        inline_count,
                        crossline_count,
                        samples,
                        store_path: None,
                        chunk_shape: None,
                        rock_physics: Default::default(),
                        geometry: Default::default(),
                    },
                ))
            }),
            rock_physics: rock.clone(),
            geometry,
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
        print_fault_summary(&cfg);
        print_filter_summary(&cfg);
        print_rock_summary(&cfg.rock_physics);
        print_geometry_summary(&cfg);
        if let Some(path) = report.store_path {
            println!(
                "wrote MDIO labels+angle-stack at {} (sub-volume chunks, parity round-trip ok)",
                path.display()
            );
        }
        return;
    }

    let runner = MultiWorkerRunner::new(config.clone());
    let report = runner
        .run_e2e_with_geometry(store.clone(), &rock, geometry)
        .unwrap_or_else(|e| {
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

/// Toy geometry line (`layered` / `planar`, horizon count).
pub fn print_geometry_summary(cfg: &synthoseis_core::pipeline::E2eConfig) {
    let g = cfg.effective_geometry();
    let nh = match g {
        synthoseis_core::ToyGeometry::Planar => 3,
        synthoseis_core::ToyGeometry::Layered => {
            synthoseis_core::toy_geometry::layered_horizon_maps(cfg.seed, cfg.shape()).1
        }
    };
    println!("toy geometry: {} ({nh} horizons)", g.as_str());
    let rp = &cfg.rock_physics;
    let lith = cfg.effective_lithology();
    let sand = synthoseis_core::lithology::interval_sand(
        lith,
        cfg.seed,
        nh.saturating_sub(1),
        rp.sand_layer_fraction,
        rp.sand_layer_thickness,
    );
    let n_sand = sand.iter().filter(|&&s| s).count();
    match lith {
        synthoseis_core::ToyLithology::Alternating => {
            println!("toy lithology: alternating ({n_sand}/{} sand layers)", sand.len())
        }
        synthoseis_core::ToyLithology::Markov => println!(
            "toy lithology: markov (sand fraction {:.3}, sand unit {} layers, {n_sand}/{} sand layers)",
            synthoseis_core::lithology::sand_fraction(cfg.seed, rp.sand_layer_fraction),
            rp.sand_layer_thickness,
            sand.len()
        ),
    }
    if rp.fluids && !rp.legacy_toy_depth {
        if g == synthoseis_core::ToyGeometry::Planar {
            println!("closures: per sand layer (planar geometry)");
        } else if rp.closures_per_layer {
            println!("closures: per sand layer (--closures-per-layer, master 8b5988f)");
        } else {
            // `closure_units` takes all `nh` entries (the last is below the
            // deepest horizon).
            let full = synthoseis_core::lithology::interval_sand(
                lith,
                cfg.seed,
                nh,
                rp.sand_layer_fraction,
                rp.sand_layer_thickness,
            );
            let units = synthoseis_core::lithology::closure_units(&full);
            let multi = units.iter().filter(|(a, b)| b - a > 1).count();
            println!(
                "closures: per sand unit ({} units with closures, {multi} multi-layer), {}",
                units.len(),
                if rp.closures_unsegmented {
                    "unsegmented (--closures-unsegmented, master ef2dc42)"
                } else {
                    "3D-segmented across faults"
                }
            );
        }
    }
}

pub fn print_rock_summary(rp: &synthoseis_core::RockPhysicsConfig) {
    if rp.legacy_toy_depth {
        println!("rock physics: legacy toy depth (master 10f4dcd: k * 100 m, label >= 2 oil)");
        return;
    }
    let ng = match &rp.net_to_gross {
        synthoseis_core::NetToGross::Constant(v) => format!("constant {v}"),
        synthoseis_core::NetToGross::Legacy { avg, .. } => {
            format!("legacy maps {}-{}", avg[0], avg[1])
        }
    };
    println!(
        "rock physics: {} m/sample below seabed, mixing={}, net_to_gross={}, first_random_layer={}, fluids={}, zoeppritz={}",
        rp.depth_step_m,
        rp.mixing.as_str(),
        ng,
        rp.first_random_layer,
        if rp.fluids { "closures" } else { "brine" },
        if rp.legacy_zoeppritz { "legacy (det typo)" } else { "textbook" }
    );
}

fn print_filter_summary(cfg: &synthoseis_core::pipeline::E2eConfig) {
    let fc = &cfg.filters;
    if !fc.enabled() {
        return;
    }
    let bandpass = match fc.bandpass_hz {
        Some([lo, hi]) => format!("{lo}-{hi} Hz order {}", fc.bandpass_order),
        None => "off".into(),
    };
    let wavelet = if fc.skips_ricker() {
        "none (bandpass replaces Ricker, legacy chain)"
    } else {
        "ricker 40 Hz"
    };
    println!(
        "filters: bandpass={bandpass}, lateral_size={}, wavelet={wavelet} (applied to data/angle_stack)",
        fc.lateral_size
    );
    if let Some(db) = fc.noise.snr_db {
        println!(
            "noise: snr={db} dB, seed={}, weights={}, data_std_cutoff={} (added to raw reflectivity before wavelet/bandpass)",
            fc.noise.seed.unwrap_or(cfg.seed),
            if fc.noise.legacy_angle_weights {
                "legacy-degrees"
            } else {
                "radians"
            },
            if fc.noise.legacy_seabed {
                "legacy-0.84-seabed"
            } else {
                "seabed"
            }
        );
    }
}

fn print_fault_summary(cfg: &synthoseis_core::pipeline::E2eConfig) {
    if !cfg.faults.enabled() {
        return;
    }
    let Some(model) = synthoseis_core::fault_model(cfg) else {
        return;
    };
    let voxels: usize = synthoseis_core::generate_fault_labels(cfg)
        .map(|m| m.iter().map(|&v| v as usize).sum())
        .unwrap_or(0);
    println!(
        "faults: requested={}, inserted={}, skipped={}, fault_voxels={} (data/fault_labels)",
        cfg.faults.count,
        model.faults().len(),
        model.skipped().len(),
        voxels
    );
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
