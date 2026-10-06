//! Cube-size-scaled closure minimum and the partial-voxel closure contact
//! cap (closure-minimum spec, Strata, against master bad1daa8).
//!
//! * `ClosureMinimum::Scaled` = `clamp(round(ni·nj / 180), 20, 500)` whole
//!   cells (rule table, §6.1).
//! * Never stricter than master's fixed 500: the kept set grows, kept
//!   closures keep their contact and fluid (§6.2).
//! * Survival on small cubes (§6.3) and the salt case (§6.4).
//! * Contact cap: whole voxels bit-identical, no column loses sub-cell
//!   hydrocarbon volume, per-cube total +0-1.5 % (§6.5).
//! * Tiling, strip, overlap and multi-process invariance with the
//!   `closure_min_voxels` attribute (§6.6).
use synthoseis_core::partial_voxels::{PartKind, PartialVoxelConfig};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, RockPhysicsConfig};
use synthoseis_core::rock_physics::{elastic_model, ColumnScratch, ElasticModel, RpmModel};
use synthoseis_core::{closure_census, generate_labels, ClosureMinimum, ToyGeometry};

/// The default configuration (layered, salt, Markov lithology, segmented
/// closures, partial voxels, time mode) on an `ni × nj × nk` cube.
fn cfg(shape: [usize; 3], faults: usize, seed: u64) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(faults),
        ..E2eConfig::default()
    }
}

fn with_minimum(c: &E2eConfig, m: ClosureMinimum) -> E2eConfig {
    let mut c = c.clone();
    c.rock_physics.closure_minimum = m;
    c
}

fn rpm(c: &E2eConfig, labels: &[u8], shape: [usize; 3]) -> RpmModel {
    match elastic_model(c, labels, shape) {
        ElasticModel::Rpm(m) => *m,
        ElasticModel::LegacyToy(_) => panic!("rock physics model expected"),
    }
}

/// §6.1: integer rule table.
#[test]
fn scaled_rule_table() {
    let s = ClosureMinimum::Scaled;
    for (ni, nj, want) in [
        (8, 8, 20),
        (24, 20, 20),
        (32, 32, 20),
        (48, 48, 20),
        (64, 64, 23),
        (96, 96, 51),
        (128, 128, 91),
        (192, 192, 205),
        (300, 300, 500),
        (512, 512, 500),
    ] {
        assert_eq!(s.voxels(ni, nj), want, "{ni}x{nj}");
    }
    // Round half up at the boundary, depth-independent, cap from 89,910.
    assert_eq!(s.voxels(90, 41), 21); // 3690 / 180 = 20.5
    assert_eq!(s.voxels(299, 300), 498);
    assert_eq!(s.voxels(330, 273), 500); // 90,090 cells
    assert_eq!(ClosureMinimum::LEGACY, ClosureMinimum::Fixed(500));
    assert_eq!(ClosureMinimum::LEGACY.voxels(8, 8), 500);
    assert_eq!(ClosureMinimum::Fixed(0).voxels(8, 8), 1);
    assert_eq!(ClosureMinimum::default(), ClosureMinimum::Scaled);
    assert_eq!(
        RockPhysicsConfig::default().closure_minimum,
        ClosureMinimum::Scaled
    );
    assert!(!RockPhysicsConfig::default().legacy_closure_contact_cap);
}

/// §6.2 gate: on 32×32×128, 64×64×256 and 128×128×256 (faults 0 and 3)
/// every closure master's fixed 500 keeps stays kept with the same contact
/// and fluid in every column; the scaled minimum only adds closures.
#[test]
fn scaled_minimum_is_never_stricter() {
    let mut added = 0;
    for (shape, seeds) in [
        ([32, 32, 128], 1..=6u64),
        ([64, 64, 256], 1..=6),
        ([128, 128, 256], 1..=2),
    ] {
        for faults in [0, 3] {
            for seed in seeds.clone() {
                let c = cfg(shape, faults, seed);
                let (labels, sh) = generate_labels(&c);
                let old = rpm(&with_minimum(&c, ClosureMinimum::LEGACY), &labels, sh);
                let new = rpm(&c, &labels, sh);
                for (lo, ln) in old.layers.iter().zip(&new.layers) {
                    let (Some(fo), Some(fnew)) = (&lo.fluids, &ln.fluids) else {
                        assert_eq!(lo.fluids.is_some(), ln.fluids.is_some());
                        continue;
                    };
                    for col in 0..fo.contact.len() {
                        if fo.contact[col] > f32::NEG_INFINITY {
                            assert_eq!(
                                fnew.contact[col].to_bits(),
                                fo.contact[col].to_bits(),
                                "{shape:?} f{faults} s{seed}"
                            );
                            assert_eq!(
                                fnew.fluid[col], fo.fluid[col],
                                "{shape:?} f{faults} s{seed}"
                            );
                        } else if fnew.contact[col] > f32::NEG_INFINITY {
                            added += 1;
                        }
                    }
                }
                let census = closure_census(&c, &labels, sh).unwrap();
                let legacy =
                    closure_census(&with_minimum(&c, ClosureMinimum::LEGACY), &labels, sh).unwrap();
                assert_eq!(
                    census.sizes, legacy.sizes,
                    "the census does not depend on the minimum"
                );
                assert!(census.kept() >= legacy.kept());
                assert_eq!(legacy.minimum, 500);
                assert_eq!(
                    census.minimum,
                    ClosureMinimum::Scaled.voxels(shape[0], shape[1])
                );
            }
        }
    }
    assert!(
        added > 0,
        "the scaled minimum keeps extra closures somewhere"
    );
}

/// §6.3 gate: 32×32×128 without faults keeps a trap in at least 8 of 12
/// seeds (master's 500: 3), and the kept closure volume is >= 98 % of all
/// closure voxels on small and medium cubes (master: >= 59.1 %).
#[test]
fn scaled_minimum_survival() {
    let with_trap = (1..=12u64)
        .filter(|&s| {
            let c = cfg([32, 32, 128], 0, s);
            let (labels, sh) = generate_labels(&c);
            closure_census(&c, &labels, sh).unwrap().kept() > 0
        })
        .count();
    let with_trap_500 = (1..=12u64)
        .filter(|&s| {
            let c = with_minimum(&cfg([32, 32, 128], 0, s), ClosureMinimum::LEGACY);
            let (labels, sh) = generate_labels(&c);
            closure_census(&c, &labels, sh).unwrap().kept() > 0
        })
        .count();
    eprintln!(
        "32x32x128 f0: seeds with a kept trap: {with_trap_500}/12 at 500, {with_trap}/12 scaled"
    );
    assert!(with_trap >= 8, "{with_trap}/12");
    assert_eq!(with_trap_500, 3);
    for (shape, seeds) in [
        ([32, 32, 128], 1..=12u64),
        ([32, 32, 256], 1..=12),
        ([64, 64, 256], 1..=12),
        ([128, 128, 256], 1..=3),
    ] {
        for faults in [0, 3] {
            let (mut kept, mut total) = (0usize, 0usize);
            for seed in seeds.clone() {
                let c = cfg(shape, faults, seed);
                let (labels, sh) = generate_labels(&c);
                let k = closure_census(&c, &labels, sh).unwrap();
                kept += k.sizes.iter().filter(|&&v| v >= k.minimum).sum::<usize>();
                total += k.sizes.iter().sum::<usize>();
            }
            let frac = kept as f64 / total.max(1) as f64;
            eprintln!(
                "{shape:?} f{faults}: kept closure volume {:.1} %",
                100.0 * frac
            );
            assert!(frac >= 0.98, "{shape:?} f{faults}: {frac}");
        }
    }
}

/// §6.4: the salt case of `tests/rock_physics.rs` (24×20×128, seed 102, 4
/// faults, sand 0.4) keeps 5 compartments on the default (T = 20), the
/// closures walled against the salt flank among them; master's 500 kept 0.
#[test]
fn salt_case_keeps_walled_closures_on_the_default() {
    let mut c = cfg([24, 20, 128], 4, 102);
    c.geometry = ToyGeometry::Layered;
    c.rock_physics.sand_layer_fraction = Some(0.4);
    let (labels, sh) = generate_labels(&c);
    let k = closure_census(&c, &labels, sh).unwrap();
    let mut sizes = k.sizes.clone();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    eprintln!("salt case compartments {sizes:?}");
    assert_eq!(sizes, [419, 391, 187, 134, 46, 9, 2, 2, 1]);
    assert_eq!((k.minimum, k.kept()), (20, 5));
    let legacy = closure_census(&with_minimum(&c, ClosureMinimum::LEGACY), &labels, sh).unwrap();
    assert_eq!(legacy.kept(), 0);
    // The kept closures reach the model as hydrocarbon or brine contacts.
    let m = rpm(&c, &labels, sh);
    let contacts = m
        .layers
        .iter()
        .filter_map(|l| l.fluids.as_ref())
        .flat_map(|f| f.contact.iter())
        .filter(|&&v| v > f32::NEG_INFINITY)
        .count();
    assert!(contacts > 0);
}

/// Elastic properties of every column of `m` (whole or partial voxels).
fn columns(m: &RpmModel, labels: &[u8], partial: bool) -> Vec<f32> {
    let [ni, nj, nk] = m.shape;
    let mut out = Vec::with_capacity(3 * ni * nj * nk);
    let mut sc = ColumnScratch::default();
    let (mut rho, mut vp, mut vs) = (vec![0f32; nk], vec![0f32; nk], vec![0f32; nk]);
    for i in 0..ni {
        let tile = m.partial_tile(i, i + 1, 0, nj);
        for j in 0..nj {
            let col = &labels[(i * nj + j) * nk..(i * nj + j + 1) * nk];
            if partial {
                m.column_partial(i, j, col, &tile, &mut sc, &mut rho, &mut vp, &mut vs);
            } else {
                m.column(i, j, col, &mut sc, &mut rho, &mut vp, &mut vs);
            }
            out.extend_from_slice(&rho);
            out.extend_from_slice(&vp);
            out.extend_from_slice(&vs);
        }
    }
    out
}

/// §6.5b: with whole voxels the base + ½ contact cap is bit-identical
/// (64×64×256, seeds 1/2/3/7/30 × faults 0/3): every column's elastic
/// properties, hence the stack and the labels.
#[test]
fn contact_cap_is_bit_identical_with_whole_voxels() {
    for faults in [0, 3] {
        for seed in [1u64, 2, 3, 7, 30] {
            let mut c = cfg([64, 64, 256], faults, seed);
            c.rock_physics.partial_voxels = PartialVoxelConfig::whole_voxels();
            let (labels, sh) = generate_labels(&c);
            let mut legacy = c.clone();
            legacy.rock_physics.legacy_closure_contact_cap = true;
            let (a, b) = (rpm(&c, &labels, sh), rpm(&legacy, &labels, sh));
            let (x, y) = (columns(&a, &labels, false), columns(&b, &labels, false));
            assert!(
                x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "f{faults} s{seed}: whole voxels must not see the contact cap"
            );
        }
    }
}

/// Sub-cell hydrocarbon volume (cells) of each column under partial voxels.
fn hc_volume(m: &RpmModel, labels: &[u8]) -> Vec<f64> {
    let [ni, nj, nk] = m.shape;
    let mut out = vec![0f64; ni * nj];
    let mut sc = ColumnScratch::default();
    let (mut rho, mut vp, mut vs) = (vec![0f32; nk], vec![0f32; nk], vec![0f32; nk]);
    for i in 0..ni {
        let tile = m.partial_tile(i, i + 1, 0, nj);
        for j in 0..nj {
            let c = i * nj + j;
            m.column_partial(
                i,
                j,
                &labels[c * nk..(c + 1) * nk],
                &tile,
                &mut sc,
                &mut rho,
                &mut vp,
                &mut vs,
            );
            out[c] = (0..nk)
                .flat_map(|k| sc.partial.parts().cell(k))
                .filter(|p| matches!(p.kind, PartKind::Interval { hc: true, .. }))
                .map(|p| p.frac)
                .sum();
        }
    }
    out
}

/// §6.5c: with partial voxels no column loses sub-cell hydrocarbon volume
/// to the base + ½ cap, and the per-cube total rises by 0-1.5 % (64×64×256,
/// seeds 1-5, no faults).
#[test]
fn contact_cap_only_adds_hydrocarbon_volume() {
    for seed in 1..=5u64 {
        let c = cfg([64, 64, 256], 0, seed);
        let (labels, sh) = generate_labels(&c);
        let mut legacy = c.clone();
        legacy.rock_physics.legacy_closure_contact_cap = true;
        let (new, old) = (
            hc_volume(&rpm(&c, &labels, sh), &labels),
            hc_volume(&rpm(&legacy, &labels, sh), &labels),
        );
        for (col, (a, b)) in new.iter().zip(&old).enumerate() {
            assert!(a >= b, "seed {seed} column {col}: {a} < {b}");
        }
        let (a, b): (f64, f64) = (new.iter().sum(), old.iter().sum());
        let gain = if b > 0.0 { a / b - 1.0 } else { 0.0 };
        eprintln!(
            "seed {seed}: sub-cell hydrocarbon volume {b:.1} -> {a:.1} cells ({:+.2} %)",
            100.0 * gain
        );
        assert!((0.0..=0.015).contains(&gain), "seed {seed}: {gain}");
    }
}

fn root_attrs(p: &std::path::Path) -> serde_json::Value {
    synthoseis_io::MdioStore::open(p)
        .unwrap()
        .root_attrs()
        .unwrap()
}

/// §6.6: the scaled default is identical across the classic, streaming
/// (three tilings), strip-stitched, overlapped and multi-process writers
/// on the salt case (T = 20 keeps 5 compartments, 500 keeps none), and
/// every store carries `closure_min_voxels = 20` /
/// `closure_minimum = "scaled-area"` (the multi-process coordinator and
/// workers agree). `ClosureMinimum::LEGACY` writes neither attribute.
#[test]
fn scaled_minimum_invariant_across_paths_and_processes() {
    use synthoseis_core::{
        run_e2e_multiprocess, run_e2e_streaming, run_e2e_streaming_overlapped,
        run_e2e_strip_stitched,
    };
    let dir = tempfile::tempdir().unwrap();
    let mut base = cfg([24, 20, 128], 4, 102);
    base.rock_physics.sand_layer_fraction = Some(0.4);
    let read = |p: &std::path::Path| {
        let v = synthoseis_io::MdioStore::open(p)
            .unwrap()
            .read_volume()
            .unwrap();
        v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>()
    };
    let mut want: Option<Vec<u32>> = None;
    type Run = fn(&E2eConfig);
    let paths: [(&str, Option<[usize; 3]>, Run); 7] = [
        ("classic", None, |c| {
            synthoseis_core::pipeline::run_e2e(c).unwrap();
        }),
        ("streaming 8x5x16", Some([8, 5, 16]), |c| {
            run_e2e_streaming(c).unwrap();
        }),
        ("streaming 5x7xnt", Some([5, 7, 128]), |c| {
            run_e2e_streaming(c).unwrap();
        }),
        ("streaming 24x20xnt", Some([24, 20, 128]), |c| {
            run_e2e_streaming(c).unwrap();
        }),
        ("strip 3", Some([8, 5, 16]), |c| {
            run_e2e_strip_stitched(c, 3).unwrap();
        }),
        ("overlap", Some([8, 5, 16]), |c| {
            run_e2e_streaming_overlapped(c).unwrap();
        }),
        ("multiprocess 2", Some([8, 5, 16]), |c| {
            run_e2e_multiprocess(c, 2).unwrap();
        }),
    ];
    for (name, chunks, run) in paths {
        let p = dir.path().join(format!("{}.mdio", name.replace(' ', "")));
        run(&E2eConfig {
            store_path: Some(p.clone()),
            chunk_shape: chunks,
            ..base.clone()
        });
        let a = root_attrs(&p);
        assert_eq!(a["closure_min_voxels"], serde_json::json!(20), "{name}");
        assert_eq!(
            a["closure_minimum"],
            serde_json::json!("scaled-area"),
            "{name}"
        );
        let v = read(&p);
        match &want {
            None => want = Some(v),
            Some(w) => assert!(*w == v, "{name}: angle stack differs from classic"),
        }
    }
    for (name, chunks, run) in [paths[0], paths[6]] {
        let p = dir
            .path()
            .join(format!("{}-legacy.mdio", name.replace(' ', "")));
        run(&E2eConfig {
            store_path: Some(p.clone()),
            chunk_shape: chunks,
            ..with_minimum(&base, ClosureMinimum::LEGACY)
        });
        let a = root_attrs(&p);
        assert!(
            a.get("closure_min_voxels").is_none() && a.get("closure_minimum").is_none(),
            "{name}"
        );
        assert!(
            want.as_ref().unwrap() != &read(&p),
            "{name}: the legacy minimum must change the salt case"
        );
    }
}
