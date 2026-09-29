//! Closures per sand unit (the default) vs the legacy generator, and the
//! `closures_per_layer` switch (`--closures-per-layer`, master 8b5988f).
//!
//! Fixture: `tests/fixtures/closure_units_reference.json` from
//! `tests/fixtures/generate_closure_units_reference.py`. The generator runs
//! the real legacy `Closures.find_top_lith_horizons` with the unit loop of
//! `create_closure_labels_from_depth_maps`, plus legacy `flood_fill_heap`.
//! * Bit-exact: sand-unit grouping and closure-unit selection for replayed
//!   legacy Markov facies sequences and edge cases.
//! * Bit-exact: per-column closure voxel counts of a two-layer sand unit
//!   with a pinched-out upper layer, closed on the unit top down to the unit
//!   base.
//! * Statistical: closure-unit counts, multi-layer units and unit thickness
//!   vs a legacy population of 3000 models, and the uniform brine / oil /
//!   gas split (legacy `rng.integers(3)` per closure).
use serde::Deserialize;
use synthoseis_core::lithology::{closure_units, interval_sand, ToyLithology};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, RockPhysicsConfig};
use synthoseis_core::rock_physics::{
    elastic_model, layer_fluids, unit_fluids, ElasticModel, RpmModel,
};
use synthoseis_core::{generate_chunked, generate_labels, ToyGeometry};

#[derive(Deserialize)]
struct UnitCase {
    sand: String,
    units: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
struct Population {
    n_units: Vec<usize>,
    n_multi: Vec<usize>,
    sand_in_units: Vec<usize>,
    unit_thickness: Vec<usize>,
}

#[derive(Deserialize)]
struct Placement {
    shape: [usize; 3],
    t: Vec<usize>,
    m: Vec<usize>,
    b: Vec<usize>,
    count_unit: Vec<usize>,
    count_layer: Vec<usize>,
}

#[derive(Deserialize)]
struct Fixture {
    units: Vec<UnitCase>,
    population: Population,
    placement: Placement,
}

fn fixture() -> Fixture {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/closure_units_reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn sand_of(s: &str) -> Vec<bool> {
    s.bytes().map(|c| c == b'1').collect()
}

/// Bit-exact: unit grouping and selection equal legacy
/// `find_top_lith_horizons` + `range(len(top_lith) - 1)` + `facies > 0`.
#[test]
fn closure_units_match_legacy() {
    let fx = fixture();
    assert!(fx.units.len() > 200);
    let mut multi = 0;
    for c in &fx.units {
        let got: Vec<[usize; 2]> = closure_units(&sand_of(&c.sand))
            .into_iter()
            .map(|(a, b)| [a, b])
            .collect();
        assert_eq!(got, c.units, "sand {}", c.sand);
        multi += c.units.iter().filter(|u| u[1] - u[0] > 1).count();
    }
    assert!(multi > 50, "fixture exercises multi-layer units ({multi})");
    // Documented edges: the deepest unit is skipped, the last entry never
    // starts a unit, consecutive sands merge.
    assert_eq!(closure_units(&sand_of("0110")), vec![]);
    assert_eq!(closure_units(&sand_of("01100")), vec![(1, 3)]);
    assert_eq!(closure_units(&sand_of("0101")), vec![(1, 2)]);
    assert_eq!(closure_units(&sand_of("0110110")), vec![(1, 3)]);
    assert_eq!(closure_units(&sand_of("01101100")), vec![(1, 3), (4, 6)]);
    assert_eq!(closure_units(&sand_of("01101")), vec![(1, 3)]);
    assert_eq!(closure_units(&sand_of("0111")), vec![]);
}

/// Two-sample Kolmogorov-Smirnov statistic.
fn ks2(a: &[f64], b: &[f64]) -> f64 {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
    }
    d
}

/// KS critical value at alpha = 0.001.
fn ks_crit(n: usize, m: usize) -> f64 {
    1.95 * (((n + m) as f64) / ((n * m) as f64)).sqrt()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Statistical: closure units of the Rust keyed Markov chain vs legacy
/// (fraction U(0.05, 0.25), thickness 2, 40 layers, 3000 models each).
#[test]
fn closure_unit_statistics_match_legacy_population() {
    let pop = fixture().population;
    let (mut n_units, mut n_multi, mut sand_in, mut thick) = (vec![], vec![], vec![], vec![]);
    for seed in 0..3000u64 {
        let u = closure_units(&interval_sand(ToyLithology::Markov, seed, 40, None, 2.0));
        n_units.push(u.len() as f64);
        n_multi.push(u.iter().filter(|(a, b)| b - a > 1).count() as f64);
        sand_in.push(u.iter().map(|(a, b)| b - a).sum::<usize>() as f64);
        thick.extend(u.iter().map(|(a, b)| (b - a) as f64));
    }
    let f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    for (name, rust, legacy) in [
        ("closure units / model", n_units, f(&pop.n_units)),
        ("multi-layer units / model", n_multi, f(&pop.n_multi)),
        (
            "sand layers in units / model",
            sand_in,
            f(&pop.sand_in_units),
        ),
        ("unit thickness (layers)", thick, f(&pop.unit_thickness)),
    ] {
        let d = ks2(&rust, &legacy);
        let crit = ks_crit(rust.len(), legacy.len());
        eprintln!(
            "{name}: rust mean {:.3} (n={}), legacy mean {:.3} (n={}), KS D {d:.4} (crit {crit:.4})",
            mean(&rust),
            rust.len(),
            mean(&legacy),
            legacy.len()
        );
        assert!(d < crit, "{name}: KS D {d} >= {crit}");
    }
}

/// Labels of the placement fixture: shale above `t`, the unit's upper layer
/// (label 1) in `[t, m)`, its lower layer (label 2) in `[m, b)`, shale
/// (label 3) below.
fn placement_labels(p: &Placement) -> Vec<u8> {
    let [ni, nj, nk] = p.shape;
    let mut labels = vec![0u8; ni * nj * nk];
    for c in 0..ni * nj {
        let col = &mut labels[c * nk..(c + 1) * nk];
        col[p.t[c]..p.m[c]].fill(1);
        col[p.m[c]..p.b[c]].fill(2);
        col[p.b[c]..].fill(3);
    }
    labels
}

/// Bit-exact: the closure of a two-layer sand unit is placed on the unit top
/// and extends down to the unit base (legacy `min(max(fill, top), base)`),
/// column by column. Per-layer closures stop at the internal horizon.
#[test]
fn unit_closure_placement_matches_legacy() {
    let p = fixture().placement;
    let [ni, nj, _] = p.shape;
    let labels = placement_labels(&p);
    let f = unit_fluids(&labels, p.shape, &[1, 2], 1, 11, 1e9, 1);
    let count = |contact: f32, top: usize, base: usize| {
        (top..base).filter(|&k| (k as f32) < contact).count()
    };
    let mut total = 0;
    for c in 0..ni * nj {
        let got = count(f.contact[c], p.t[c], p.b[c]);
        assert_eq!(got, p.count_unit[c], "column {c}");
        total += got;
    }
    let voxels: usize = f.closures.iter().map(|cl| cl.4).sum();
    assert_eq!(voxels, total);
    let legacy_unit: usize = p.count_unit.iter().sum();
    let legacy_layer: usize = p.count_layer.iter().sum();
    // Per layer: the upper layer alone (absent where pinched out) and the
    // lower layer on its own top.
    let up = layer_fluids(&labels, p.shape, 1, 1, 11, 1e9, 1);
    let lo = layer_fluids(&labels, p.shape, 2, 2, 11, 1e9, 1);
    let per_layer: usize = (0..ni * nj)
        .map(|c| count(up.contact[c], p.t[c], p.m[c]) + count(lo.contact[c], p.m[c], p.b[c]))
        .sum();
    eprintln!(
        "placement: unit closure voxels {total} (legacy {legacy_unit}); upper layer alone (legacy) {legacy_layer}; \
         per-layer mode {per_layer} in {} + {} closures",
        up.closures.len(),
        lo.closures.len()
    );
    assert!(total > legacy_layer);
    assert_eq!(f.closures.len(), 1);
}

fn layered(seed: u64, shape: [usize; 3], rp: RockPhysicsConfig) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some([16, 16, shape[2]]),
        faults: FaultConfig::with_count(4),
        filters: FilterConfig::default(),
        rock_physics: rp,
        geometry: ToyGeometry::Layered,
    }
}

fn sandy(per_layer: bool) -> RockPhysicsConfig {
    RockPhysicsConfig {
        sand_layer_fraction: Some(0.4),
        sand_layer_thickness: 3.0,
        closures_per_layer: per_layer,
        ..RockPhysicsConfig::default()
    }
}

/// A model with a multi-layer sand unit that closes on the dome.
const SEED: u64 = 6;
const SHAPE: [usize; 3] = [24, 20, 128];

fn model(c: &E2eConfig) -> Box<RpmModel> {
    let (labels, shape) = generate_labels(c);
    let ElasticModel::Rpm(m) = elastic_model(c, &labels, shape) else {
        panic!()
    };
    m
}

/// Model wiring: members of a sand unit share the unit's contact and fluid
/// maps; the closure list is on the shallowest member; the deepest unit and
/// shales have none; `closures_per_layer` restores per-layer closures.
#[test]
fn model_shares_unit_closures_across_member_layers() {
    let c = layered(SEED, SHAPE, sandy(false));
    let m = model(&c);
    let (labels, shape) = generate_labels(&c);
    let sand = interval_sand(ToyLithology::Markov, SEED, m.nh, Some(0.4), 3.0);
    let units = closure_units(&sand);
    let mut multi_with_closures = 0;
    for l in &m.layers {
        assert_eq!(l.fluids.is_some(), l.sand, "interval {}", l.interval);
    }
    for &(top, end) in &units {
        let members: Vec<_> = m
            .layers
            .iter()
            .filter(|l| l.interval >= top && l.interval < end)
            .collect();
        if members.is_empty() {
            continue;
        }
        let first = members[0].fluids.as_ref().unwrap();
        for l in &members[1..] {
            let f = l.fluids.as_ref().unwrap();
            assert_eq!(f.contact, first.contact);
            assert_eq!(f.fluid, first.fluid);
            assert!(f.closures.is_empty());
        }
        if members.len() > 1 && !first.closures.is_empty() {
            multi_with_closures += 1;
        }
    }
    assert!(multi_with_closures > 0, "a multi-layer unit with closures");
    // Sand layers outside every closure unit (the deepest unit) are brine.
    for l in m.layers.iter().filter(|l| l.sand) {
        if !units
            .iter()
            .any(|&(a, b)| l.interval >= a && l.interval < b)
        {
            let f = l.fluids.as_ref().unwrap();
            assert!(f.closures.is_empty() && f.contact.iter().all(|&x| x == f32::NEG_INFINITY));
        }
    }
    // Per layer: every sand label gets its own `layer_fluids`.
    let pl = model(&layered(SEED, SHAPE, sandy(true)));
    for (lab, l) in pl.layers.iter().enumerate() {
        if let Some(f) = &l.fluids {
            let want = layer_fluids(
                &labels,
                shape,
                lab as u8,
                l.interval,
                SEED,
                150.0 / 4.0,
                c.rock_physics.min_closure_voxels,
            );
            assert_eq!(f.contact, want.contact, "label {lab}");
            assert_eq!(f.closures, want.closures, "label {lab}");
        }
    }
    let h = |c: &E2eConfig| {
        let (v, _) = generate_chunked(c);
        v.angle_stack
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    assert_ne!(h(&c), h(&layered(SEED, SHAPE, sandy(true))));
}

/// Statistical: fluids are uniform over closures (legacy `rng.integers(3)`
/// per closure), and the effect of per-unit closures on closure counts.
#[test]
fn unit_closure_fluid_split_is_uniform() {
    // Many seeds on the placement unit: one closure per seed.
    let p = fixture().placement;
    let labels = placement_labels(&p);
    let mut by = [0f64; 3];
    let n = 3000;
    for seed in 0..n {
        let f = unit_fluids(&labels, p.shape, &[1, 2], 1, seed, 1e9, 1);
        by[f.closures[0].0 as usize] += 1.0;
    }
    let e = n as f64 / 3.0;
    let chi2: f64 = by.iter().map(|&o| (o - e) * (o - e) / e).sum();
    eprintln!("fluid split over {n} seeds (brine/oil/gas): {by:?}, chi2 {chi2:.2} (2 dof, crit 13.82 at 0.001)");
    assert!(chi2 < 13.82);

    // Models: closures per unit vs per layer.
    let (mut unit, mut layer) = ([0usize; 3], [0usize; 3]);
    for seed in 0..12u64 {
        for (per_layer, acc) in [(false, &mut unit), (true, &mut layer)] {
            let m = model(&layered(seed, SHAPE, sandy(per_layer)));
            for cl in m
                .layers
                .iter()
                .flat_map(|l| l.fluids.iter().flat_map(|f| f.closures.iter()))
            {
                acc[cl.0 as usize] += 1;
            }
        }
    }
    eprintln!("12 seeds 24x20x128 sand 0.4/3: closures brine/oil/gas per unit {unit:?}, per layer {layer:?}");
    assert!(unit.iter().sum::<usize>() > 0);
    assert!(unit.iter().sum::<usize>() < layer.iter().sum::<usize>());
}
