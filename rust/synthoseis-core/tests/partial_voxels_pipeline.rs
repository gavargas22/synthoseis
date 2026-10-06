//! Partial voxels wired through the pipeline (spec "partial voxels" PR B1,
//! §3.3, §3.6, §4, §5.5, §5.6). Partial voxels are the library default
//! since PR B2; the `rich` fixture opts out (`PartialVoxelConfig::whole_voxels`,
//! = the d8b96e69 default) and the tests turn the switch on explicitly.
//! * Switch on: every path (classic, chunked, streaming, overlap,
//!   strip-stitch 2/3/4, multiprocess 1/2/3, geometry-once) writes the same
//!   bytes for every chunk shape, in time mode (S and C) and on the legacy
//!   axis (C), with the RICH flags, salt and faults.
//! * The default is on (= the explicit S / C); on differs from off, S from C.
//! * The three root attributes are written only when on.
//! * Depth, fault and salt labels are identical on vs off; time labels move
//!   by at most one sample where the traveltime changes.
//! * Faulted label guard ≤ 1 % of sediment cells (4 faults), 0 unfaulted.
//! * Pull-back (§3.3) unit checks.
use synthoseis_core::partial_model::partial_voxel_summary;
use synthoseis_core::partial_voxels::{
    absorb_water_below, column_parts, column_parts_faulted, source_intervals, ColumnGeometry,
    ColumnParts, PartKind, PartialVoxelConfig, PvReflectivity,
};
use synthoseis_core::pipeline::{generate_tiny_cube, E2eConfig};
use synthoseis_core::{
    elastic_model, generate_chunked, generate_labels, generate_output_labels,
    run_e2e_geometry_once_seismic_many, run_e2e_multiprocess, run_e2e_streaming,
    run_e2e_streaming_overlapped, run_e2e_strip_stitched, FaultConfig, FilterConfig, NoiseConfig,
    RockPhysicsConfig, TimeConfig, DEFAULT_INCIDENCE_DEG,
};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// RICH flags (bandpass 4-30 Hz, 12.5 dB noise, salt, layered geometry)
/// with `faults` faults; time mode with `nt` samples, or the legacy axis.
/// Whole voxels (the opt-out): the tests switch partial voxels on.
fn rich(seed: u64, shape: [usize; 3], faults: usize, time: TimeConfig) -> E2eConfig {
    let mut filters = FilterConfig::legacy(4.0, 30.0, 3);
    filters.noise = NoiseConfig {
        snr_db: Some(12.5),
        ..NoiseConfig::default()
    };
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(faults),
        filters,
        time,
        rock_physics: RockPhysicsConfig {
            partial_voxels: PartialVoxelConfig::whole_voxels(),
            ..RockPhysicsConfig::default()
        },
        ..E2eConfig::default()
    }
}

fn with_pv(cfg: &E2eConfig, r: Option<PvReflectivity>) -> E2eConfig {
    let mut c = cfg.clone();
    c.rock_physics.partial_voxels = match r {
        Some(r) => PartialVoxelConfig::with(r),
        None => PartialVoxelConfig::whole_voxels(),
    };
    c
}

/// Spec §5.6 time axis: nt = nz + 37, dt = 2 ms, dz = 2 m.
fn time_long(nz: usize) -> TimeConfig {
    TimeConfig {
        samples: Some(nz + 37),
        dt_ms: 2.0,
        ..TimeConfig::default()
    }
}

fn dz2(mut cfg: E2eConfig) -> E2eConfig {
    cfg.rock_physics.depth_step_m = 2.0;
    cfg
}

struct Store {
    angles: Vec<u32>,
    labels: Vec<u8>,
    faults: Option<Vec<u8>>,
    salt: Option<Vec<u8>>,
    attrs: serde_json::Map<String, serde_json::Value>,
}

const PV_ATTRS: [&str; 3] = [
    "voxel_model",
    "partial_voxel_mixing",
    "partial_voxel_reflectivity",
];

fn read_store(p: &std::path::Path) -> Store {
    let s = synthoseis_io::MdioStore::open(p).unwrap();
    let has = |a: &str| p.join("data").join(a).join(".zarray").is_file();
    let all = s.root_attrs().unwrap();
    let attrs = PV_ATTRS
        .iter()
        .filter_map(|k| all.get(*k).map(|v| (k.to_string(), v.clone())))
        .collect();
    Store {
        angles: bits(&s.read_volume().unwrap()),
        labels: s.read_labels_u8().unwrap(),
        faults: has("fault_labels").then(|| s.read_fault_labels_u8().unwrap()),
        salt: has("salt_labels").then(|| s.read_salt_labels_u8().unwrap()),
        attrs,
    }
}

fn expected_attrs(cfg: &E2eConfig) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    if let Some(r) = cfg.effective_partial_voxels() {
        m.insert("voxel_model".into(), "partial-z".into());
        m.insert("partial_voxel_mixing".into(), "backus".into());
        m.insert("partial_voxel_reflectivity".into(), r.as_str().into());
    }
    m
}

fn assert_store(name: &str, s: &Store, r: &Store) {
    assert!(s.angles == r.angles, "{name}: angle_stack");
    assert!(s.labels == r.labels, "{name}: labels");
    assert!(s.faults == r.faults, "{name}: fault_labels");
    assert!(s.salt == r.salt, "{name}: salt_labels");
    assert_eq!(s.attrs, r.attrs, "{name}: partial-voxel attrs");
}

/// Reference: the classic full-volume pass plus the output label cubes.
fn reference(cfg: &E2eConfig) -> Store {
    let classic = generate_tiny_cube(cfg);
    let (labels, shape) = generate_labels(cfg);
    let model = elastic_model(cfg, &labels, shape);
    let out = generate_output_labels(cfg, &labels, &model);
    assert_eq!(
        classic.labels, out.labels,
        "classic labels vs output labels"
    );
    Store {
        angles: bits(&classic.angle_stack),
        labels: out.labels,
        faults: out.faults,
        salt: out.salt,
        attrs: expected_attrs(cfg),
    }
}

/// Spec §5.6: every path, every chunk shape, bit-identical with the switch
/// on (stack, labels, attrs).
fn assert_tiling_invariant(cfg: &E2eConfig, chunk_shapes: &[[usize; 3]], mp: bool) {
    let r = reference(cfg);
    let nt = cfg.output_samples();
    let dir = tempfile::tempdir().unwrap();
    for (n, &c) in chunk_shapes.iter().enumerate() {
        let c = [c[0], c[1], if c[2] == 0 { nt } else { c[2] }];
        let cc = E2eConfig {
            chunk_shape: Some(c),
            ..cfg.clone()
        };
        let (v, _) = generate_chunked(&cc);
        assert!(
            bits(&v.angle_stack) == r.angles,
            "chunked {c:?}: angle_stack"
        );
        assert!(v.labels == r.labels, "chunked {c:?}: labels");
        let p = dir.path().join(format!("stream{n}.mdio"));
        run_e2e_streaming(&E2eConfig {
            store_path: Some(p.clone()),
            ..cc.clone()
        })
        .unwrap();
        assert_store(&format!("streaming {c:?}"), &read_store(&p), &r);
        let p = dir.path().join(format!("overlap{n}.mdio"));
        run_e2e_streaming_overlapped(&E2eConfig {
            store_path: Some(p.clone()),
            ..cc.clone()
        })
        .unwrap();
        assert_store(&format!("overlap {c:?}"), &read_store(&p), &r);
    }
    for workers in [2, 3, 4] {
        let p = dir.path().join(format!("strip{workers}.mdio"));
        let sc = E2eConfig {
            store_path: Some(p.clone()),
            chunk_shape: Some([3, 4, 16]),
            ..cfg.clone()
        };
        run_e2e_strip_stitched(&sc, workers).unwrap();
        assert_store(&format!("strip {workers}"), &read_store(&p), &r);
    }
    if mp {
        for workers in [1, 2, 3] {
            let p = dir.path().join(format!("mp{workers}.mdio"));
            let mc = E2eConfig {
                store_path: Some(p.clone()),
                chunk_shape: Some([2, 5, 8]),
                ..cfg.clone()
            };
            run_e2e_multiprocess(&mc, workers).unwrap();
            assert_store(&format!("multiprocess {workers}"), &read_store(&p), &r);
        }
    }
    let p = dir.path().join("geo.mdio");
    let gc = E2eConfig {
        chunk_shape: Some([4, 4, nt]),
        store_path: Some(p),
        ..cfg.clone()
    };
    let (g, _) = run_e2e_geometry_once_seismic_many(&gc, &[DEFAULT_INCIDENCE_DEG]).unwrap();
    assert!(
        bits(&g.stacks[0].volumes.angle_stack) == r.angles,
        "geometry-once: angle_stack"
    );
    assert!(g.labels == r.labels, "geometry-once: labels");
    let gp = g.stacks[0]
        .store_path
        .as_ref()
        .expect("geometry-once store");
    assert_store("geometry-once store", &read_store(gp), &r);
}

const CHUNKS: [[usize; 3]; 4] = [[1, 1, 0], [5, 7, 0], [3, 20, 16], [8, 5, 16]];
const SHAPE: [usize; 3] = [12, 20, 48];

/// Switch on, time mode S: seed 30 (salt, 3 faults) and seed 7 (4 faults +
/// salt), nt = nz + 37, dt = 2 ms, dz = 2 m (spec §5.6).
#[test]
fn switch_on_tiling_invariance_time_subcell() {
    for (seed, faults) in [(30u64, 3usize), (7, 4)] {
        let cfg = with_pv(
            &dz2(rich(seed, SHAPE, faults, time_long(SHAPE[2]))),
            Some(PvReflectivity::Subcell),
        );
        let s = partial_voxel_summary(&cfg).unwrap();
        println!("PV B1 seed {seed}: {s}");
        assert!(s.stats.mixed > 0, "seed {seed}: mixed cells");
        assert_tiling_invariant(&cfg, &CHUNKS, true);
    }
}

/// Switch on, time mode C, default axis (seed 30).
#[test]
fn switch_on_tiling_invariance_time_cell() {
    let cfg = with_pv(
        &rich(30, SHAPE, 3, TimeConfig::default()),
        Some(PvReflectivity::Cell),
    );
    assert_tiling_invariant(&cfg, &CHUNKS[1..3], true);
}

/// Switch on, the legacy depth-as-time axis (C; seed 7, 4 faults + salt).
#[test]
fn switch_on_tiling_invariance_legacy_axis() {
    let cfg = with_pv(
        &rich(7, SHAPE, 4, TimeConfig::legacy()),
        Some(PvReflectivity::Cell),
    );
    assert_eq!(cfg.effective_partial_voxels(), Some(PvReflectivity::Cell));
    assert_tiling_invariant(&cfg, &CHUNKS, true);
}

/// The library default is on (PR B2) and equals the explicit S in time mode
/// and C on the legacy axis; `whole_voxels()` is off; on differs from off
/// and S from C; the attrs are written only when on; the planar geometry
/// and an explicit S on the legacy axis are rejected by validation.
#[test]
fn switch_default_on_and_modes_differ() {
    assert!(PartialVoxelConfig::default().enabled());
    assert_eq!(PartialVoxelConfig::default(), PartialVoxelConfig::on());
    assert!(!PartialVoxelConfig::whole_voxels().enabled());
    assert_eq!(
        E2eConfig::default().rock_physics.partial_voxels,
        PartialVoxelConfig::default()
    );
    let base = rich(30, [10, 12, 48], 0, TimeConfig::default());
    assert_eq!(base.effective_partial_voxels(), None);
    for time in [TimeConfig::default(), TimeConfig::legacy()] {
        let mut dflt = rich(30, [10, 12, 48], 0, time.clone());
        dflt.rock_physics.partial_voxels = PartialVoxelConfig::default();
        let r = if time.enabled {
            PvReflectivity::Subcell
        } else {
            PvReflectivity::Cell
        };
        assert_eq!(dflt.effective_partial_voxels(), Some(r));
        let explicit = generate_tiny_cube(&with_pv(&dflt, Some(r)));
        let d = generate_tiny_cube(&dflt);
        assert_eq!(
            bits(&d.angle_stack),
            bits(&explicit.angle_stack),
            "default = explicit {r:?}"
        );
        let whole = generate_tiny_cube(&with_pv(&dflt, None));
        assert_ne!(
            bits(&d.angle_stack),
            bits(&whole.angle_stack),
            "default vs whole voxels"
        );
    }
    let off = generate_tiny_cube(&base);
    let s = generate_tiny_cube(&with_pv(&base, Some(PvReflectivity::Subcell)));
    let c = generate_tiny_cube(&with_pv(&base, Some(PvReflectivity::Cell)));
    assert_ne!(bits(&off.angle_stack), bits(&s.angle_stack), "S vs off");
    assert_ne!(bits(&off.angle_stack), bits(&c.angle_stack), "C vs off");
    assert_ne!(bits(&s.angle_stack), bits(&c.angle_stack), "S vs C");
    // Enabled without a reflectivity: S in time mode, C on the legacy axis.
    let mut on = base.clone();
    on.rock_physics.partial_voxels = PartialVoxelConfig::on();
    assert_eq!(on.effective_partial_voxels(), Some(PvReflectivity::Subcell));
    on.time = TimeConfig::legacy();
    assert_eq!(on.effective_partial_voxels(), Some(PvReflectivity::Cell));
    let dir = tempfile::tempdir().unwrap();
    for (name, cfg) in [
        ("off", base.clone()),
        ("subcell", with_pv(&base, Some(PvReflectivity::Subcell))),
        ("cell", with_pv(&base, Some(PvReflectivity::Cell))),
    ] {
        let p = dir.path().join(format!("{name}.mdio"));
        run_e2e_streaming(&E2eConfig {
            store_path: Some(p.clone()),
            chunk_shape: Some([4, 5, 48]),
            ..cfg.clone()
        })
        .unwrap();
        let st = read_store(&p);
        assert_eq!(st.attrs, expected_attrs(&cfg), "{name}");
        assert_eq!(st.attrs.is_empty(), name == "off", "{name}");
    }
    let mut planar = with_pv(&base, Some(PvReflectivity::Cell));
    planar.geometry = synthoseis_core::ToyGeometry::Planar;
    assert_eq!(
        planar.effective_partial_voxels(),
        None,
        "planar has no partial state"
    );
    let mut bad = with_pv(&base, Some(PvReflectivity::Subcell));
    bad.time = TimeConfig::legacy();
    assert!(bad.validate_time().is_err(), "S on the legacy axis");
}

/// Spec §5.5: depth-domain (legacy axis) labels, fault labels and salt
/// labels are identical on vs off; in time mode the labels move by at most
/// one sample, only where the traveltime moves, with no new classes, 255
/// above the seabed and fault ∧ salt = 0. 5 seeds × {salt, no salt} ×
/// {0, 4 faults}.
#[test]
fn label_parity_on_vs_off() {
    let shape = [10, 12, 48];
    let mut changed_total = (0usize, 0usize);
    for seed in [7u64, 1, 2, 3, 30] {
        for salt in [true, false] {
            for faults in [0usize, 4] {
                let mk = |time: TimeConfig| {
                    let mut c = rich(seed, shape, faults, time);
                    c.rock_physics.salt = salt;
                    c
                };
                // Legacy axis: the label cubes are the depth labels.
                let off = mk(TimeConfig::legacy());
                let on = with_pv(&off, Some(PvReflectivity::Cell));
                let (lo, sh) = generate_labels(&off);
                let (ln, _) = generate_labels(&on);
                assert_eq!(
                    lo, ln,
                    "seed {seed} salt {salt} faults {faults}: depth labels"
                );
                let a = generate_output_labels(&off, &lo, &elastic_model(&off, &lo, sh));
                let b = generate_output_labels(&on, &ln, &elastic_model(&on, &ln, sh));
                assert_eq!(a.labels, b.labels, "seed {seed}: legacy-axis labels");
                assert_eq!(a.faults, b.faults, "seed {seed}: legacy-axis fault labels");
                assert_eq!(a.salt, b.salt, "seed {seed}: legacy-axis salt labels");
                // Time mode.
                let off = mk(TimeConfig::default());
                let on = with_pv(&off, Some(PvReflectivity::Subcell));
                let a = generate_output_labels(&off, &lo, &elastic_model(&off, &lo, sh));
                let b = generate_output_labels(&on, &lo, &elastic_model(&on, &lo, sh));
                let nt = on.output_samples();
                let classes = |v: &[u8]| {
                    let mut c: Vec<u8> = v.to_vec();
                    c.sort_unstable();
                    c.dedup();
                    c
                };
                let ca = classes(&a.labels);
                assert!(
                    classes(&b.labels).iter().all(|x| ca.contains(x)),
                    "seed {seed}: new classes"
                );
                let mut changed = 0;
                for t in 0..a.labels.len() / nt {
                    let (x, y) = (
                        &a.labels[t * nt..(t + 1) * nt],
                        &b.labels[t * nt..(t + 1) * nt],
                    );
                    // 255 above the seabed stays a prefix.
                    let top = |v: &[u8]| v.iter().take_while(|&&l| l == 255).count();
                    assert!(
                        !y[top(y)..].contains(&255) || x[top(x)..].contains(&255),
                        "seed {seed}: 255 below the seabed"
                    );
                    for k in 0..nt {
                        if x[k] != y[k] {
                            changed += 1;
                            // At the last sample the neighbour below lies past nt.
                            let near = k == nt - 1
                                || (k.saturating_sub(1)..=(k + 1).min(nt - 1))
                                    .any(|m| x[m] == y[k]);
                            assert!(near, "seed {seed} salt {salt} faults {faults} trace {t} sample {k}: moved more than one sample\noff {:?}\non  {:?}", &x[k.saturating_sub(6)..], &y[k.saturating_sub(6)..]);
                        }
                    }
                }
                if let (Some(f), Some(s)) = (&b.faults, &b.salt) {
                    assert_eq!(
                        f.iter().zip(s).filter(|(&f, &s)| f == 1 && s == 1).count(),
                        0,
                        "fault ∧ salt"
                    );
                }
                changed_total.0 += changed;
                changed_total.1 += a.labels.len();
            }
        }
    }
    println!(
        "PV B1 time labels: {} of {} samples changed ({:.3}%)",
        changed_total.0,
        changed_total.1,
        100.0 * changed_total.0 as f64 / changed_total.1 as f64
    );
    assert!(changed_total.0 > 0, "partial T must move some time labels");
}

/// Spec §5.5 label guard: with 4 faults, cells with no fraction of their
/// labelled unit stay ≤ 1 % of sediment cells (Strata's probe: 0.013–0.357 %
/// at 64×64×256); 0 unfaulted. Hidden intervals: 0 without salt.
#[test]
fn faulted_label_guard() {
    let shape = [32, 32, 128];
    for seed in [7u64, 1, 2, 3, 30] {
        for salt in [false, true] {
            for faults in [0usize, 4] {
                let mut cfg = with_pv(
                    &E2eConfig {
                        seed,
                        inline_count: shape[0],
                        crossline_count: shape[1],
                        samples: shape[2],
                        faults: FaultConfig::with_count(faults),
                        ..E2eConfig::default()
                    },
                    Some(PvReflectivity::Subcell),
                );
                cfg.rock_physics = RockPhysicsConfig {
                    salt,
                    ..cfg.rock_physics
                };
                let s = partial_voxel_summary(&cfg).unwrap();
                let pct = 100.0 * s.stats.label_guard as f64 / s.stats.sediment as f64;
                println!("PV B1 guard seed {seed} salt {salt} faults {faults}: {pct:.4}% | {s}");
                assert!(s.stats.sediment > 0);
                if faults == 0 {
                    assert_eq!(s.stats.label_guard, 0, "seed {seed}: unfaulted guard");
                } else {
                    assert!(pct <= 1.0, "seed {seed}: guard {pct}%");
                }
                if !salt && faults == 0 {
                    assert_eq!(
                        s.stats.hidden_intervals, 0,
                        "seed {seed}: hidden intervals without salt"
                    );
                }
            }
        }
    }
}

/// Pull-back (spec §3.3): identity lookup gives unit cells; a throw breaks
/// the column and the edges use the unbroken neighbour; clamps break;
/// stretched cells keep midpoint edges; the faulted fractions of an
/// identity column equal the unfaulted ones; a horizon inside a source
/// interval maps to `ζ = k + (z − σ⁻)/g`.
#[test]
fn pull_back_source_intervals() {
    let nz = 12;
    let ident: Vec<f32> = (0..nz).map(|k| k as f32).collect();
    let mut brk = vec![false; nz + 1];
    let mut src = Vec::new();
    source_intervals(&ident, &mut brk, &mut src);
    for (k, &(lo, hi)) in src.iter().enumerate().skip(1).take(nz - 2) {
        assert_eq!((lo, hi), (k as f64, k as f64 + 1.0), "identity cell {k}");
    }
    assert!(
        brk[0] && brk[nz] && brk[1] && brk[nz - 1],
        "clamped ends break"
    );
    // Throw of +3.25 below cell 5 (lookup jumps): a break at 6.
    let thrown: Vec<f32> = (0..nz)
        .map(|k| k as f32 + if k >= 6 { 3.25 } else { 0.0 } + 0.1)
        .collect();
    let thrown: Vec<f32> = thrown
        .into_iter()
        .map(|v| v.min((nz - 1) as f32 - 0.01))
        .collect();
    let mut brk = vec![false; nz + 1];
    source_intervals(&thrown, &mut brk, &mut src);
    assert!(brk[6], "throw breaks");
    assert!(
        (src[5].1 - (5.6 + 0.5)).abs() < 1e-5,
        "upper side edge from the unbroken neighbour: {:?}",
        src[5]
    );
    assert!(
        (src[6].0 - (9.85 - 0.5)).abs() < 1e-5,
        "lower side edge: {:?}",
        src[6]
    );
    // Stretch 1.2: midpoint edges, g = 1.2.
    let st: Vec<f32> = (0..nz).map(|k| 1.0 + 0.8 * k as f32).collect();
    let mut brk = vec![false; nz + 1];
    source_intervals(&st, &mut brk, &mut src);
    assert!(!brk[5]);
    assert!(((src[5].1 - src[5].0) - 0.8).abs() < 1e-6);
    // Fault drag: a nearly flat lookup (Δσ = 0.05) keeps midpoint edges.
    let flat: Vec<f32> = (0..nz).map(|k| 4.0 + 0.05 * k as f32).collect();
    let mut brk = vec![false; nz + 1];
    source_intervals(&flat, &mut brk, &mut src);
    assert!(
        !brk[2..nz - 1].iter().any(|&b| b),
        "compression does not break"
    );
    assert!(((src[5].1 - src[5].0) - 0.05).abs() < 1e-5, "{:?}", src[5]);
    // A fold (σ decreasing) breaks.
    let fold: Vec<f32> = (0..nz)
        .map(|k| 6.0 - (k as f32 - 6.0).abs() * 0.3)
        .collect();
    let mut brk = vec![false; nz + 1];
    source_intervals(&fold, &mut brk, &mut src);
    assert!(brk[7] && !brk[3], "fold breaks, rising part does not");
    // An inside()-flip break given by the caller is kept.
    let mut brk = vec![false; nz + 1];
    brk[4] = true;
    source_intervals(&st, &mut brk, &mut src);
    assert!(brk[4] && (src[4].0 - (st[4] as f64 + 0.5 - 0.4)).abs() < 1e-6);
    // Identity column: faulted fractions = unfaulted fractions.
    let z = [0.0, 2.3, 5.71, 9.05, 1e9];
    let geom = ColumnGeometry {
        horizons: &z,
        salt: Some((6.2, 7.9)),
        contacts: &[],
    };
    let (mut a, mut b) = (ColumnParts::default(), ColumnParts::default());
    column_parts(&geom, nz, &mut a);
    let unit: Vec<(f64, f64)> = (0..nz).map(|k| (k as f64, k as f64 + 1.0)).collect();
    column_parts_faulted(&geom, &unit, &mut b);
    for k in 0..nz {
        assert_eq!(a.cell(k), b.cell(k), "identity cell {k}");
    }
    // Stretched cell 3 covering source (2.9, 4.1): the horizon at 3.5 maps
    // to ζ = 3 + (3.5 − 2.9)/1.2 = 3.5.
    let z2 = [0.0, 3.5, 1e9];
    let geom = ColumnGeometry {
        horizons: &z2,
        salt: None,
        contacts: &[],
    };
    let mut src2 = unit.clone();
    src2[3] = (2.9, 4.1);
    column_parts_faulted(&geom, &src2, &mut b);
    let c3 = b.cell(3);
    assert_eq!(c3.len(), 2);
    assert!((c3[0].frac - 0.6 / 1.2).abs() < 1e-12, "{c3:?}");
}

/// A fold of the fault lookup near the seabed reaches above the source
/// seabed from cells below the output seabed: those water parts take the
/// cell's sediment unit (counted as `water_below`); the seabed cell keeps its
/// water.
#[test]
fn water_below_seabed_is_absorbed() {
    let z = [2.4, 6.0, 1e9];
    let geom = ColumnGeometry {
        horizons: &z,
        salt: None,
        contacts: &[],
    };
    let nz = 8;
    // Cells 0-2 identity; cell 3 folds back to source (2.2, 3.2); 4.. identity.
    let mut src: Vec<(f64, f64)> = (0..nz).map(|k| (k as f64, k as f64 + 1.0)).collect();
    src[3] = (2.2, 3.2);
    let mut parts = ColumnParts::default();
    column_parts_faulted(&geom, &src, &mut parts);
    assert!(parts.cell(3).iter().any(|p| p.kind == PartKind::Water));
    let seabed = parts.cell(2).to_vec();
    assert_eq!(absorb_water_below(&mut parts, 3), 1);
    assert_eq!(
        parts.cell(2),
        &seabed[..],
        "the seabed cell keeps its water"
    );
    let c3 = parts.cell(3);
    assert_eq!(c3.len(), 1, "{c3:?}");
    assert_eq!(c3[0].kind, PartKind::Interval { h: 0, hc: false });
    assert!((c3[0].frac - 1.0).abs() < 1e-12);
    assert_eq!(absorb_water_below(&mut parts, 3), 0, "idempotent");
}
