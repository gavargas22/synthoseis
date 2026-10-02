//! Layered toy geometry (default): many layers so the default random depth
//! shifts (legacy layers > 20) apply, a dome so sand tops form closures and
//! oil / gas / brine selection triggers end to end, legacy statistics of the
//! random parts, determinism and the switches that keep the old goldens.
//! The tiling / worker / process / path invariance of the layered geometry
//! is covered in `rock_physics.rs`
//! (`default_model_invariant_to_tiling_workers_and_paths`).

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, RockPhysicsConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid};
use synthoseis_core::toy_geometry::{
    layer_thickness, layered_horizon_maps, LayeredParams, SEABED_MIN_DEPTH_M,
};
use synthoseis_core::{generate_chunked, generate_labels, generate_reflectivity, ToyGeometry, ToyLithology};

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn ah(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn demo(seed: u64, shape: [usize; 3], faults: usize) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some([16, 16, shape[2]]),
        faults: FaultConfig::with_count(faults),
        filters: FilterConfig::default(),
        // Goldens of the layered geometry before the Markov lithology (and
        // before 3D closure segmentation, which changes the faulted cube).
        rock_physics: RockPhysicsConfig {
            lithology: ToyLithology::Alternating,
            closures_unsegmented: true,
            // ... and before salt bodies (`salt: false` = master b4f4259).
            salt: false,
            ..RockPhysicsConfig::default()
        },
        geometry: ToyGeometry::Layered,
    }
}

fn rpm(c: &E2eConfig) -> Box<synthoseis_core::RpmModel> {
    let (labels, shape) = generate_labels(c);
    match elastic_model(c, &labels, shape) {
        ElasticModel::Rpm(m) => m,
        ElasticModel::LegacyToy(_) => panic!("default must be the rpm model"),
    }
}

#[test]
fn layered_is_the_default_and_legacy_toy_depth_forces_planar() {
    assert_eq!(E2eConfig::default().geometry, ToyGeometry::Layered);
    let mut c = demo(7, [32, 32, 64], 0);
    assert_eq!(c.effective_geometry(), ToyGeometry::Layered);
    c.rock_physics = RockPhysicsConfig::legacy_toy();
    assert_eq!(c.effective_geometry(), ToyGeometry::Planar);
    // master 10f4dcd raw 15 deg reflectivity, whatever `geometry` says.
    let t = E2eConfig {
        rock_physics: RockPhysicsConfig::legacy_toy(),
        ..demo(7, [64, 64, 128], 4)
    };
    assert_eq!(ah(&generate_reflectivity(&t, 15.0)), 0xee47_4d76_0e21_eb09);
}

/// End to end on the demo cube: > 20 layers so the default shifts apply,
/// closures over the dome with oil / gas / brine, and both change the output.
#[test]
fn shifts_and_closure_fluids_trigger_end_to_end() {
    let c = demo(7, [64, 64, 256], 4);
    let m = rpm(&c);
    let shifted: Vec<usize> = m
        .layers
        .iter()
        .filter(|l| l.shifts.layer != 0)
        .map(|l| l.interval)
        .collect();
    let mut by_fluid = [0usize; 3];
    let mut n_closures = 0;
    for l in &m.layers {
        for cl in l.fluids.iter().flat_map(|f| f.closures.iter()) {
            n_closures += 1;
            by_fluid[match cl.0 {
                Fluid::Brine => 0,
                Fluid::Oil => 1,
                Fluid::Gas => 2,
            }] += 1;
        }
    }
    eprintln!(
        "64x64x256 seed 7: {} horizons, {} layers, shifted intervals {:?}, closures {} (brine {}, oil {}, gas {})",
        m.nh,
        m.layers.len(),
        shifted,
        n_closures,
        by_fluid[0],
        by_fluid[1],
        by_fluid[2]
    );
    assert!(m.layers.len() > 21, "need layers beyond legacy layer 20");
    assert!(!shifted.is_empty() && shifted.iter().all(|&h| h >= 20));
    assert!(m
        .layers
        .iter()
        .filter(|l| l.interval < 20)
        .all(|l| l.shifts.layer == 0));
    assert!(
        n_closures >= 3 && by_fluid[1] + by_fluid[2] > 0,
        "{by_fluid:?}"
    );

    // Hydrocarbon voxels exist in the elastic model.
    let (labels, shape) = generate_labels(&c);
    let hc: usize = m
        .layers
        .iter()
        .filter_map(|l| l.fluids.as_ref())
        .map(|f| {
            (0..shape[0] * shape[1])
                .filter(|&col| f.fluid[col] != Fluid::Brine && f.contact[col].is_finite())
                .count()
        })
        .sum();
    assert!(hc > 0);
    let _ = labels;

    // Both mechanisms change the angle stack.
    let h = |c: &E2eConfig| ah(&generate_chunked(c).0.angle_stack);
    let reference = h(&c);
    let no_fluids = E2eConfig {
        rock_physics: RockPhysicsConfig {
            fluids: false,
            ..c.rock_physics.clone()
        },
        ..c.clone()
    };
    let no_shifts = E2eConfig {
        rock_physics: RockPhysicsConfig {
            first_random_layer: 10_000,
            ..c.rock_physics.clone()
        },
        ..c.clone()
    };
    assert_ne!(h(&no_fluids), reference, "fluids must change the output");
    assert_ne!(h(&no_shifts), reference, "shifts must change the output");

    // Physical reflectivity with many interfaces.
    let r = generate_reflectivity(&c, 15.0);
    let nz = r.iter().filter(|&&x| x != 0.0).count() as f64 / r.len() as f64;
    let big = r.iter().filter(|x| x.abs() > 1.0).count();
    eprintln!("rfc15: non-zero {:.2}%, |r| > 1: {big}", 100.0 * nz);
    assert_eq!(big, 0);
    assert!(nz > 0.05, "many interfaces expected, got {nz}");
}

/// Default settings on the standard 64x64x128 demo cube: shifts reach at
/// least one layer and closures carry hydrocarbons. Golden hashes pin the
/// layered default (labels, 15 deg stack).
#[test]
fn default_demo_cube_goldens() {
    let c = demo(7, [64, 64, 128], 4);
    let m = rpm(&c);
    let shifted = m.layers.iter().filter(|l| l.shifts.layer != 0).count();
    let hc = m
        .layers
        .iter()
        .flat_map(|l| l.fluids.iter().flat_map(|f| f.closures.iter()))
        .filter(|cl| cl.0 != Fluid::Brine)
        .count();
    eprintln!(
        "64x64x128: {} layers, {shifted} shifted, {hc} hydrocarbon closures",
        m.layers.len()
    );
    assert!(shifted >= 1 && hc >= 1);
    let (v, _) = generate_chunked(&c);
    // Horizons sit on whole samples and the base lies below the cube, so the
    // only unfilled (255) samples are the water column above the seabed.
    let nk = 128;
    for col in v.labels.chunks(nk) {
        let first = col
            .iter()
            .position(|&l| l != 255)
            .expect("sediment in every column");
        assert!(
            col[first..].iter().all(|&l| l != 255),
            "unfilled sample below the seabed"
        );
    }
    eprintln!(
        "labels {:#018x} stack {:#018x}",
        fnv(v.labels.iter().copied()),
        ah(&v.angle_stack)
    );
    assert_eq!(fnv(v.labels.iter().copied()), LAYERED_LABELS);
    assert_eq!(ah(&v.angle_stack), LAYERED_STACK15);
}

const LAYERED_LABELS: u64 = 0x021e_4d94_9085_f056;
const LAYERED_STACK15: u64 = 0x5d4c_ba89_7f5a_ef44;

/// Legacy statistics of the random parts.
#[test]
fn layered_statistics_match_legacy() {
    // Thickness: legacy `stats.gamma.rvs(4.0, 2)` = 2 + Gamma(4, 1):
    // mean 6, variance 4, support [2, inf).
    let t: Vec<f64> = (0..400u64)
        .flat_map(|s| (0..50).map(move |l| layer_thickness(s, l)))
        .collect();
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let mut sorted = t.clone();
    sorted.sort_by(f64::total_cmp);
    let cdf = |x: f64| {
        let y = (x - 2.0).max(0.0);
        1.0 - (-y).exp() * (1.0 + y + y * y / 2.0 + y * y * y / 6.0)
    };
    let ks = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = cdf(x);
            (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
        })
        .fold(0.0f64, f64::max);
    eprintln!(
        "thickness: n {n}, mean {mean:.3} (legacy 6), var {var:.3} (legacy 4), min {:.3}, KS D {ks:.4} (5% crit {:.4})",
        sorted[0],
        1.36 / n.sqrt()
    );
    assert!((mean - 6.0).abs() < 0.05 && (var - 4.0).abs() < 0.2 && sorted[0] >= 2.0);
    assert!(ks < 1.36 / n.sqrt());

    // Seabed minimum depth: legacy `rng.integers(20, 50)` metres at 4 m.
    let mut seen = [false; 30];
    for s in 0..3000u64 {
        let p = LayeredParams::new(s, [64, 64, 1250]);
        let m = p.seabed_min * 4.0;
        assert!(
            m >= SEABED_MIN_DEPTH_M[0] as f64
                && m < SEABED_MIN_DEPTH_M[1] as f64
                && m.fract() == 0.0
        );
        seen[(m as usize) - 20] = true;
    }
    assert!(seen.iter().all(|&x| x), "all 30 integer depths drawn");

    // Layer count ~ legacy: sediment column / mean thickness (6); crest
    // thinning adds a few.
    for (seed, nk) in [(1u64, 256usize), (2, 512), (3, 1250)] {
        let (_, nh) = layered_horizon_maps(seed, [48, 48, nk]);
        let p = LayeredParams::new(seed, [48, 48, nk]);
        let expect = (nk as f64 - p.seabed_min) / 6.0;
        eprintln!(
            "nk {nk}: {} layers, legacy-style expectation ~{expect:.0}",
            nh - 1
        );
        assert!(((nh - 1) as f64 - expect).abs() < 0.2 * expect + 3.0);
    }

    // Closure fluids: legacy `rng.integers(3)` per closure, ~1/3 each.
    let mut counts = [0usize; 3];
    for seed in 0..24u64 {
        let c = E2eConfig {
            rock_physics: RockPhysicsConfig {
                min_closure_voxels: 1,
                lithology: ToyLithology::Alternating,
                ..RockPhysicsConfig::default()
            },
            ..demo(seed, [32, 32, 128], 0)
        };
        for l in &rpm(&c).layers {
            for cl in l.fluids.iter().flat_map(|f| f.closures.iter()) {
                counts[cl.0 as usize] += 1;
            }
        }
    }
    let total: usize = counts.iter().sum();
    let e = total as f64 / 3.0;
    let chi2: f64 = counts.iter().map(|&o| (o as f64 - e).powi(2) / e).sum();
    eprintln!(
        "closure fluids over 24 seeds: {counts:?} (brine, oil, gas), chi2 {chi2:.2} (5% crit 5.99)"
    );
    assert!(total >= 60 && chi2 < 9.21, "{counts:?}");
}

/// Deterministic in the seed; different seeds give different structures;
/// per-column labels do not depend on the chunk shape.
#[test]
fn layered_is_deterministic_and_chunk_independent() {
    let a = layered_horizon_maps(11, [20, 24, 96]);
    assert_eq!(a, layered_horizon_maps(11, [20, 24, 96]));
    assert_ne!(a, layered_horizon_maps(12, [20, 24, 96]));
    let base = demo(11, [20, 24, 96], 2);
    let (l0, _) = generate_labels(&base);
    for chunks in [[1, 1, 96], [7, 5, 96], [20, 24, 32]] {
        let c = E2eConfig {
            chunk_shape: Some(chunks),
            ..base.clone()
        };
        assert_eq!(generate_labels(&c).0, l0);
    }
    // Horizons are ordered and the label volume uses every interval.
    let (maps, nh) = a;
    for col in maps.chunks(nh) {
        assert!(col.windows(2).all(|w| w[0] <= w[1]));
    }
}

/// RAM bound: the streaming working set stays within the per-chunk budget,
/// and the model's fixed maps grow like the labels (about 8 bytes per
/// horizon per column, i.e. ~1.3 bytes per voxel at 6-sample layers).
#[test]
fn layered_working_set_is_bounded() {
    let c = E2eConfig {
        chunk_shape: Some([8, 8, 256]),
        ..demo(7, [64, 64, 256], 4)
    };
    let (_, stats) = generate_chunked(&c);
    assert!(stats.is_bounded_by_chunk(64), "{stats:?}");
    let (labels, shape) = generate_labels(&c);
    let model = elastic_model(&c, &labels, shape);
    let per_voxel = model.model_bytes() as f64 / labels.len() as f64;
    let tile_budget = 8 * 8 * 256 * 32;
    eprintln!(
        "layered model bytes {} ({per_voxel:.2} B/voxel), peak temp {} B, tile budget {tile_budget} B",
        model.model_bytes(),
        stats.peak_temp_bytes
    );
    assert!(per_voxel < 4.0, "{per_voxel}");
    // Beyond the fixed model, per-tile scratch stays within one tile budget.
    assert!(
        stats.peak_temp_bytes <= model.model_bytes() + tile_budget,
        "{stats:?}"
    );
}
