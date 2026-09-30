//! Legacy sand-fraction lithology (`--toy-lithology markov`, the default for
//! the layered geometry) vs the legacy generator.
//!
//! Fixture: `tests/fixtures/lithology_reference.json` from
//! `tests/fixtures/generate_lithology_reference.py`, which runs the real
//! legacy `Facies.sand_shale_facies_markov` / `MarkovChainFacies`.
//! * Bit-exact: given legacy's own draws (the initial state and the one
//!   uniform per `rng.choice(p=row)`), the Rust chain reproduces legacy facies
//!   exactly, and the transition matrix matches to the bit.
//! * Statistical: the Rust keyed draws vs a legacy population of 3000 models
//!   (fraction U(0.05, 0.25), sand unit 2 layers, 60 layers). Checked are the
//!   fraction distribution, the per-model sand proportion, and the sand / shale
//!   run lengths.
use serde::Deserialize;
use synthoseis_core::lithology::{
    interval_sand, legacy_choice, markov_states, sand_fraction, transition_matrix, transition_unit,
    validate, ToyLithology, SAND_LAYER_FRACTION, SAND_LAYER_THICKNESS,
};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, RockPhysicsConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid};
use synthoseis_core::{generate_chunked, generate_labels, ToyGeometry};

#[derive(Deserialize)]
struct Chain {
    seed: u64,
    initial: u8,
    // Floats as IEEE-754 bits: serde_json's default float parser is not
    // correctly rounded.
    sand_fraction_bits: u64,
    sand_thickness_bits: u64,
    u_bits: Vec<u64>,
    transition_bits: [[u64; 2]; 2],
    facies: Vec<u8>,
}

#[derive(Deserialize)]
struct Population {
    models: usize,
    layers: usize,
    fractions: Vec<f64>,
    sand_count: Vec<usize>,
    sand_run_hist: Vec<u64>,
    shale_run_hist: Vec<u64>,
    max_run_bin: usize,
}

#[derive(Deserialize)]
struct Fixture {
    sand_layer_fraction: [f64; 2],
    sand_layer_thickness: f64,
    chains: Vec<Chain>,
    population: Population,
}

fn fixture() -> Fixture {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/lithology_reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
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

/// 5 % critical value of the two-sample KS statistic.
fn ks2_crit(n: usize, m: usize) -> f64 {
    1.358 * (((n + m) as f64) / (n as f64 * m as f64)).sqrt()
}

fn runs(states: &[bool], value: bool, hist: &mut [u64]) {
    let mut k = 0;
    while k < states.len() {
        let start = k;
        while k < states.len() && states[k] == states[start] {
            k += 1;
        }
        if states[start] == value {
            let r = (k - start).min(hist.len());
            hist[r - 1] += 1;
        }
    }
}

/// Chi-square homogeneity of two histograms (bins with < 5 expected merged
/// into the tail). Returns (statistic, degrees of freedom).
fn chi2_homogeneity(a: &[u64], b: &[u64]) -> (f64, usize) {
    let (na, nb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let (mut ca, mut cb) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ca += x as f64;
        cb += y as f64;
        let t = ca + cb;
        if t * na.min(nb) / (na + nb) >= 5.0 {
            bins.push((ca, cb));
            ca = 0.0;
            cb = 0.0;
        }
    }
    if ca + cb > 0.0 {
        let last = bins.last_mut().unwrap();
        last.0 += ca;
        last.1 += cb;
    }
    let mut chi2 = 0.0;
    for &(x, y) in &bins {
        let t = x + y;
        let (ea, eb) = (t * na / (na + nb), t * nb / (na + nb));
        chi2 += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    (chi2, bins.len() - 1)
}

/// 5 % critical value of chi-square (Wilson-Hilferty).
fn chi2_crit(df: usize) -> f64 {
    let k = df as f64;
    k * (1.0 - 2.0 / (9.0 * k) + 1.645 * (2.0 / (9.0 * k)).sqrt()).powi(3)
}

#[test]
fn markov_chain_is_bit_exact_to_legacy_on_legacy_draws() {
    let fix = fixture();
    assert_eq!(fix.sand_layer_fraction, SAND_LAYER_FRACTION);
    assert_eq!(fix.sand_layer_thickness, SAND_LAYER_THICKNESS);
    let mut n = 0;
    for c in &fix.chains {
        let (f, th) = (
            f64::from_bits(c.sand_fraction_bits),
            f64::from_bits(c.sand_thickness_bits),
        );
        let u: Vec<f64> = c.u_bits.iter().map(|&b| f64::from_bits(b)).collect();
        let t = transition_matrix(f, th);
        for r in 0..2 {
            for k in 0..2 {
                assert_eq!(
                    t[r][k].to_bits(),
                    c.transition_bits[r][k],
                    "seed {} T[{r}][{k}]",
                    c.seed
                );
            }
        }
        let s = markov_states(c.initial, &u, f, th);
        assert_eq!(s, c.facies, "seed {} f {f} T {th}", c.seed);
        n += s.len();
    }
    eprintln!("{} legacy chains, {n} facies bit-exact", fix.chains.len());
    // numpy searchsorted(side="right"): u == cdf[0] goes to state 1.
    let row = [0.75, 0.25];
    assert_eq!(legacy_choice(row, 0.75), 1);
    assert_eq!(legacy_choice(row, 0.75f64.next_down_compat()), 0);
    assert_eq!(legacy_choice(row, 0.0), 0);
}

trait NextDown {
    fn next_down_compat(self) -> f64;
}
impl NextDown for f64 {
    fn next_down_compat(self) -> f64 {
        f64::from_bits(self.to_bits() - 1)
    }
}

#[test]
fn keyed_draws_match_legacy_population_statistics() {
    let fix = fixture();
    let pop = &fix.population;
    let seeds = 0..pop.models as u64;
    // Model sand fraction: legacy U(0.05, 0.25).
    let fr: Vec<f64> = seeds.clone().map(|s| sand_fraction(s, None)).collect();
    assert!(fr.iter().all(|f| (0.05..0.25).contains(f)));
    let d_frac = ks2(&fr, &pop.fractions);
    // One-sample KS against the uniform CDF too.
    let mut sorted = fr.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let d_unif = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let cdf = (x - 0.05) / 0.2;
            (cdf - i as f64 / n)
                .abs()
                .max(((i + 1) as f64 / n - cdf).abs())
        })
        .fold(0.0, f64::max);
    // Per-model sand proportion and run lengths over `layers` layers.
    let mut prop = Vec::with_capacity(pop.models);
    let mut sand_hist = vec![0u64; pop.max_run_bin];
    let mut shale_hist = vec![0u64; pop.max_run_bin];
    for s in seeds {
        let v = interval_sand(
            ToyLithology::Markov,
            s,
            pop.layers,
            None,
            SAND_LAYER_THICKNESS,
        );
        prop.push(v.iter().filter(|&&x| x).count() as f64 / v.len() as f64);
        runs(&v, true, &mut sand_hist);
        runs(&v, false, &mut shale_hist);
    }
    let legacy_prop: Vec<f64> = pop
        .sand_count
        .iter()
        .map(|&k| k as f64 / pop.layers as f64)
        .collect();
    let d_prop = ks2(&prop, &legacy_prop);
    let crit = ks2_crit(pop.models, pop.models);
    let (c_sand, df_sand) = chi2_homogeneity(&sand_hist, &pop.sand_run_hist);
    let (c_shale, df_shale) = chi2_homogeneity(&shale_hist, &pop.shale_run_hist);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "fraction: KS vs legacy {d_frac:.4}, vs U(0.05,0.25) {d_unif:.4} (5% crit {crit:.4} / {:.4})",
        1.358 / n.sqrt()
    );
    eprintln!(
        "sand proportion: mean {:.4} (legacy {:.4}), KS {d_prop:.4} (5% crit {crit:.4})",
        mean(&prop),
        mean(&legacy_prop)
    );
    eprintln!(
        "sand runs {sand_hist:?} vs legacy {:?}: chi2 {c_sand:.2} df {df_sand} (5% crit {:.2})",
        pop.sand_run_hist,
        chi2_crit(df_sand)
    );
    eprintln!(
        "shale runs {shale_hist:?} vs legacy {:?}: chi2 {c_shale:.2} df {df_shale} (5% crit {:.2})",
        pop.shale_run_hist,
        chi2_crit(df_shale)
    );
    assert!(d_frac < crit && d_unif < 1.358 / n.sqrt(), "fraction");
    assert!(d_prop < crit, "sand proportion");
    // 1 % level for the run histograms (two tests).
    assert!(c_sand < chi2_crit(df_sand) * 1.25 && c_shale < chi2_crit(df_shale) * 1.25);
}

#[test]
fn stationary_fraction_and_sand_unit_thickness() {
    // Legacy design: stationary sand fraction f, mean sand run T layers.
    for (f, t) in [(0.1, 2.0), (0.25, 2.0), (0.4, 3.0), (0.5, 1.0)] {
        let (mut sand, mut total, mut runs_n, mut run_len) = (0usize, 0usize, 0usize, 0usize);
        for seed in 0..400u64 {
            let v = interval_sand(ToyLithology::Markov, seed, 200, Some(f), t);
            sand += v.iter().filter(|&&x| x).count();
            total += v.len();
            let mut hist = vec![0u64; 200];
            runs(&v, true, &mut hist);
            runs_n += hist.iter().sum::<u64>() as usize;
            run_len += hist
                .iter()
                .enumerate()
                .map(|(i, &c)| (i + 1) * c as usize)
                .sum::<usize>();
        }
        let fr = sand as f64 / total as f64;
        let mean_run = run_len as f64 / runs_n as f64;
        eprintln!("f {f} T {t}: sand fraction {fr:.4}, mean sand run {mean_run:.3} layers");
        assert!((fr - f).abs() < 0.02, "{fr}");
        assert!((mean_run - t).abs() < 0.1 * t, "{mean_run}");
    }
}

#[test]
fn chain_is_prefix_stable_deterministic_and_validated() {
    let long = interval_sand(ToyLithology::Markov, 5, 200, None, 2.0);
    assert_eq!(
        &interval_sand(ToyLithology::Markov, 5, 17, None, 2.0)[..],
        &long[..17]
    );
    assert_eq!(long, interval_sand(ToyLithology::Markov, 5, 200, None, 2.0));
    assert_ne!(long, interval_sand(ToyLithology::Markov, 6, 200, None, 2.0));
    assert!(transition_unit(5, 0) != transition_unit(5, 1));
    let alt = interval_sand(ToyLithology::Alternating, 5, 6, None, 2.0);
    assert_eq!(alt, [false, true, false, true, false, true]);
    assert!(validate(0.25, 2.0).is_ok() && validate(2.0 / 3.0, 2.0).is_ok());
    assert!(
        validate(0.7, 2.0).is_err() && validate(0.0, 2.0).is_err() && validate(0.2, 0.5).is_err()
    );
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
        // Pinned master b4f4259 scenarios and goldens: no salt
        // (tests/salt.rs covers salt).
        rock_physics: RockPhysicsConfig { salt: false, ..rp },
        geometry: ToyGeometry::Layered,
    }
}

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

/// End to end: the model's sand layers are the Markov chain's; planar and
/// legacy toy stay alternating; alternating reproduces the previous
/// layered goldens; Markov goldens on the demo cube.
#[test]
fn model_uses_markov_lithology_end_to_end() {
    let c = layered(7, [64, 64, 256], RockPhysicsConfig::default());
    assert_eq!(c.effective_lithology(), ToyLithology::Markov);
    let (labels, shape) = generate_labels(&c);
    let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
        panic!()
    };
    let want = interval_sand(ToyLithology::Markov, 7, m.nh, None, 2.0);
    for l in &m.layers {
        assert_eq!(l.sand, want[l.interval], "interval {}", l.interval);
        assert_eq!(l.ng.is_empty(), !l.sand);
        assert_eq!(l.fluids.is_some(), l.sand);
    }
    let n_sand = m.layers.iter().filter(|l| l.sand).count();
    eprintln!(
        "64x64x256 seed 7 markov: fraction {:.3}, {n_sand}/{} sand layers",
        sand_fraction(7, None),
        m.layers.len()
    );

    let planar = E2eConfig {
        geometry: ToyGeometry::Planar,
        ..c.clone()
    };
    assert_eq!(planar.effective_lithology(), ToyLithology::Alternating);
    let toy = E2eConfig {
        rock_physics: RockPhysicsConfig::legacy_toy(),
        ..c.clone()
    };
    assert_eq!(toy.effective_lithology(), ToyLithology::Alternating);

    // Previous layered default (master after #30) under `alternating`.
    let demo = |rp: RockPhysicsConfig| layered(7, [64, 64, 128], rp);
    let alt = demo(RockPhysicsConfig {
        lithology: ToyLithology::Alternating,
        closures_unsegmented: true,
        ..RockPhysicsConfig::default()
    });
    let (v, _) = generate_chunked(&alt);
    assert_eq!(fnv(v.labels.iter().copied()), 0x021e_4d94_9085_f056);
    assert_eq!(ah(&v.angle_stack), 0x5d4c_ba89_7f5a_ef44);
    // Markov on the same cube with per-layer closures (master 8b5988f):
    // labels unchanged, stack differs from alternating.
    let (mv, _) = generate_chunked(&demo(RockPhysicsConfig {
        closures_per_layer: true,
        ..RockPhysicsConfig::default()
    }));
    eprintln!("markov demo stack15 {:#018x}", ah(&mv.angle_stack));
    assert_eq!(mv.labels, v.labels);
    assert_ne!(ah(&mv.angle_stack), ah(&v.angle_stack));
    assert_eq!(ah(&mv.angle_stack), MARKOV_DEMO_STACK15);
    // Closures per sand unit, unsegmented (master ef2dc42).
    let (uv, _) = generate_chunked(&demo(RockPhysicsConfig {
        closures_unsegmented: true,
        ..RockPhysicsConfig::default()
    }));
    assert_eq!(uv.labels, v.labels);
    assert_eq!(ah(&uv.angle_stack), MARKOV_UNIT_DEMO_STACK15);
    // Default: 3D-segmented closures per sand unit.
    let (sv, _) = generate_chunked(&demo(RockPhysicsConfig::default()));
    assert_eq!(ah(&sv.angle_stack), MARKOV_SEGMENTED_DEMO_STACK15);
}

/// Markov demo stack of master 8b5988f (closures per layer).
const MARKOV_DEMO_STACK15: u64 = 0xd952_d8da_8616_c9e3;
/// Markov demo stack with closures per sand unit.
const MARKOV_UNIT_DEMO_STACK15: u64 = 0x8270_8d88_0146_cf10;
/// Markov demo stack with 3D-segmented closures per sand unit (default):
/// on this cube no closure is split or joined, so it equals ef2dc42.
const MARKOV_SEGMENTED_DEMO_STACK15: u64 = MARKOV_UNIT_DEMO_STACK15;

/// With a richer sand fraction the Markov lithology still yields closures
/// with oil / gas / brine on the dome.
#[test]
fn markov_sands_form_closures_with_fluids() {
    let mut by_fluid = [0usize; 3];
    let mut sand_layers = 0;
    for seed in 0..6u64 {
        let c = layered(
            seed,
            [48, 48, 192],
            RockPhysicsConfig {
                min_closure_voxels: 1,
                ..RockPhysicsConfig::default()
            },
        );
        let (labels, shape) = generate_labels(&c);
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
            panic!()
        };
        sand_layers += m.layers.iter().filter(|l| l.sand).count();
        for l in &m.layers {
            for cl in l.fluids.iter().flat_map(|f| f.closures.iter()) {
                by_fluid[match cl.0 {
                    Fluid::Brine => 0,
                    Fluid::Oil => 1,
                    Fluid::Gas => 2,
                }] += 1;
            }
        }
    }
    eprintln!(
        "6 seeds 48x48x192 markov: {sand_layers} sand layers, closures brine/oil/gas {by_fluid:?}"
    );
    assert!(
        sand_layers > 0 && by_fluid[1] + by_fluid[2] > 0,
        "{by_fluid:?}"
    );
}
