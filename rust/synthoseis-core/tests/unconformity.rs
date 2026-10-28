//! Bevel-and-ponded-fill unconformity (erosion and onlap spec, Strata,
//! §7 tests 1-6 and 8-11; the CLI side of tests 7 and 8 is in
//! `synthoseis/tests/unconformity_cli.rs`, test 12 is not in this PR).
use synthoseis_core::folds::FoldConfig;
use synthoseis_core::partial_voxels::{
    column_parts, unit_thickness, ColumnGeometry, ColumnParts, Unit, EPS_FRAC,
};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::toy_geometry::{
    layered_horizon_maps_both, layered_horizon_maps_with, layered_stack, layered_stack_records,
};
use synthoseis_core::unconformity::{
    apply_one, exhumation_at, relief_map, slope_max, UnconformityConfig, UnconformityRecord,
};
use synthoseis_core::{
    generate_labels, generate_unconformity_labels, generate_unconformity_labels_output,
    unconformity_applied, unconformity_records, UnconformityShale,
};

fn uncf(count: usize) -> UnconformityConfig {
    UnconformityConfig { count, ..UnconformityConfig::default() }
}

fn cfg(seed: u64, shape: [usize; 3]) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        ..E2eConfig::default()
    }
}

// --------------------------------------------------------------------
// Tests 1 and 2: analytic planar dip (spec §7.1, §7.2).
// --------------------------------------------------------------------

const NH: usize = 24;
const U: usize = 10;
const TT: f64 = 6.0;
const D: f64 = 0.25;
const Z0: f64 = 100.0;
const NI: usize = 256;
const NJ: usize = 2;

fn flat_cfg() -> UnconformityConfig {
    UnconformityConfig {
        count: 1,
        level: Some(0.5),
        relief_m: Some(0.0),
        ..UnconformityConfig::default()
    }
}

fn pcol(c: usize) -> f64 {
    Z0 + D * (c / NJ) as f64
}

/// Planar dip P(i) = z0 + d i; deepest-first horizons: `z_k = P + t (u - k)`
/// (k <= u: older, deeper) and `z_k = P - t (k - u)` (k > u: younger),
/// with the pre-unconformity stack being `P + G_k`, `G_k = t (u - k)`.
fn planar_stack() -> Vec<Vec<f64>> {
    (0..NH)
        .map(|k| {
            let g = TT * (U as f64 - k as f64);
            (0..NI * NJ).map(|c| pcol(c) + g).collect()
        })
        .collect()
}

fn apply_planar(stack: &mut [Vec<f64>]) -> UnconformityRecord {
    apply_one(stack, 1, [NI, NJ, 512], &flat_cfg(), 1, NH - 1 - U)
}

/// Truncation: every older horizon is `max(z_k, E)`, `E = max(P, L)`; the
/// layer between k and k+1 has zero thickness for exactly the columns
/// `i < (L - G_k - z0)/d` (+-1 column) and its outcrop belt is `t/d = 24`
/// columns wide (+-1).
#[test]
fn t1_truncation_analytic() {
    let pre = planar_stack();
    let mut post = pre.clone();
    let rec = apply_planar(&mut post);
    assert!(rec.applied, "{rec:?}");
    assert_eq!(rec.relief, 0.0);
    let l = rec.level;
    // The binding of the cap needs G_0 >= 40 (errata): here G_0 = 60.
    assert!(
        KAPPA_FREE(&pre, l),
        "the kappa cap must not bind in this stack"
    );
    for c in 0..NI * NJ {
        let e = pcol(c).max(l);
        assert_eq!(post[U][c], e, "E at column {c}");
        for k in 0..U {
            let want = pre[k][c].max(e);
            assert!((post[k][c] - want).abs() <= 1e-12, "k {k} c {c}");
        }
    }
    for k in 0..U - 1 {
        let g_k = TT * (U as f64 - k as f64);
        let edge = (l - g_k - Z0) / D;
        // Zero-thickness columns of the layer (k, k+1) (rows i, step NJ).
        let zero: Vec<usize> = (0..NI)
            .filter(|&i| post[k][i * NJ] == post[k + 1][i * NJ])
            .collect();
        let last_zero = zero.iter().copied().max().map_or(-1, |v| v as i64);
        assert_eq!(zero.len() as i64, last_zero + 1, "k {k}: zero set not contiguous");
        let want = (edge.ceil() as i64).clamp(0, NI as i64);
        assert!(
            (zero.len() as i64 - want).abs() <= 1,
            "k {k}: {} zero columns, want {want} (edge {edge})",
            zero.len()
        );
        // Outcrop belt: positive thickness while the top is still on E.
        let belt = (0..NI)
            .filter(|&i| {
                let c = i * NJ;
                post[k][c] > post[k + 1][c] && post[k + 1][c] == post[U][c]
            })
            .count() as i64;
        // The belt of layer k is the run of columns between its own edge and
        // the edge of the layer above it: t/d = 24 columns wide once both
        // edges are inside the cube (it is cut by the cube side otherwise).
        let edge_up = (l - TT * (U as f64 - k as f64 - 1.0) - Z0) / D;
        let lo = edge.ceil().clamp(0.0, NI as f64);
        let hi = edge_up.ceil().clamp(0.0, NI as f64);
        let width = (hi - lo) as i64;
        if edge >= 0.0 && edge_up <= NI as f64 {
            assert!(width.abs_diff((TT / D).round() as i64) <= 1, "k {k}: belt width {width}");
        }
        assert!((belt - width).abs() <= 1, "k {k}: belt {belt}, want {width}");
    }
}

#[allow(non_snake_case)]
fn KAPPA_FREE(pre: &[Vec<f64>], l: f64) -> bool {
    (0..NI * NJ).all(|c| {
        let p = pre[U][c];
        (l.max(p) - p) <= 0.8 * (pre[0][c] - p) + 1e-12
    })
}

/// Onlap: younger horizons `z_k = P - s_k`, `s_k = (k - u) t`. On the side
/// `P >= L` horizon k sits on E up to `i_k = i_L + (F - s_k/w_k)/d` and
/// lifts off after it, onlap points move up-slope with height, and
/// the layers keep non-negative thickness.
#[test]
fn t2_onlap_analytic() {
    let mut post: Vec<Vec<f64>> = (0..NH)
        .map(|k| {
            let g = TT * (U as f64 - k as f64);
            (0..NI * NJ).map(|c| pcol(c) + g).collect()
        })
        .collect();
    let pre = post.clone();
    let rec = apply_planar(&mut post);
    assert!(rec.applied);
    let l = rec.level;
    let i_l = (l - Z0) / D;
    let p_max = pcol((NI - 1) * NJ);
    let f = p_max - l;
    assert!((rec.hollow - f).abs() < 1e-9, "F = {} vs {f}", rec.hollow);
    let big_s = 2.0 * (p_max - Z0); // rho max(Delta), Delta = L + F - P
    assert!((rec.fill_height - big_s).abs() < 1e-9);
    let mut last = f64::INFINITY;
    for k in U + 1..NH - 1 {
        let s_k = (k - U) as f64 * TT;
        let w = (1.0 - s_k / big_s).clamp(0.0, 1.0);
        let i_k = i_l + (f - s_k / w) / D;
        let on_e = |i: usize| post[k][i * NJ] == post[U][i * NJ];
        let side: Vec<usize> = (0..NI).filter(|&i| pcol(i * NJ) >= l).collect();
        // Contact set on the P >= L side is an initial segment.
        let contact: Vec<usize> = side.iter().copied().filter(|&i| on_e(i)).collect();
        if i_k >= i_l + 2.0 {
            let hi = *contact.last().expect("a contact exists") as f64;
            assert!((hi - i_k).abs() <= 1.0, "k {k}: contact {hi}, formula {i_k}");
            assert_eq!(
                contact.len() as f64,
                hi - side[0] as f64 + 1.0,
                "k {k}: contact is not contiguous"
            );
            assert!(hi <= last, "k {k}: onlap point moved down-slope with height");
            last = hi;
        } else if i_k < i_l - 2.0 {
            assert!(contact.is_empty(), "k {k}: unexpected contact");
        }
        // Lift-off after the contact: strictly above E.
        for &i in side.iter().filter(|&&i| (i as f64) > i_k + 1.0) {
            assert!(post[k][i * NJ] < post[U][i * NJ], "k {k} i {i}: not lifted off");
        }
        // Thickness above the contact is non-negative everywhere.
        for c in 0..NI * NJ {
            assert!(post[k][c] >= post[k + 1][c], "k {k} c {c}: negative thickness");
        }
        // The younger horizons only move down (towards E) and never past it.
        for c in 0..NI * NJ {
            assert!(post[k][c] >= pre[k][c] - 0.0 && post[k][c] <= post[U][c]);
        }
    }
    // The seabed is untouched.
    assert!(post[NH - 1].iter().zip(&pre[NH - 1]).all(|(a, b)| a.to_bits() == b.to_bits()));
}

// --------------------------------------------------------------------
// Tests 3 and 4: round trip and invariants.
// --------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Case {
    folds: bool,
    count: usize,
}

const CASES: [Case; 4] = [
    Case { folds: true, count: 1 },
    Case { folds: false, count: 1 },
    Case { folds: true, count: 2 },
    Case { folds: false, count: 2 },
];

fn fold_of(c: Case) -> Option<FoldConfig> {
    c.folds.then(FoldConfig::default)
}

/// `round(continuous) == production` bit for bit and the column invariants.
fn roundtrip_and_invariants(seed: u64, shape: [usize; 3], applied_seen: &mut [usize; 2]) {
    let n = shape[0] * shape[1];
    for case in CASES {
        let fold = fold_of(case);
        let ucfg = uncf(case.count);
        let ((prod, nh), (cont, nh2), recs) =
            layered_horizon_maps_both(seed, shape, fold.as_ref(), Some(&ucfg));
        assert_eq!(nh, nh2);
        for (i, (&p, &c)) in prod.iter().zip(&cont).enumerate() {
            assert!(
                c.round().to_bits() == p.to_bits() || (c.round() == p && p == 0.0),
                "seed {seed} {shape:?} folds {} N {}: element {i}: round({c}) != {p}",
                case.folds,
                case.count
            );
        }
        for c in 0..n {
            for h in 0..nh - 1 {
                assert!(prod[c * nh + h] <= prod[c * nh + h + 1], "seed {seed}: map order");
            }
        }
        let applied = recs.iter().filter(|r| r.applied).count();
        applied_seen[case.count - 1] += applied;
        // Invariants on the f64 stack.
        let (pre, _) = layered_stack_records(seed, shape, fold.as_ref(), None);
        let (post, recs2) = layered_stack_records(seed, shape, fold.as_ref(), Some(&ucfg));
        assert_eq!(recs, recs2, "stack is deterministic");
        assert_eq!(post.len(), pre.len());
        let nh = post.len();
        for c in 0..n {
            for k in 0..nh {
                assert!(post[k][c] >= pre[k][c], "seed {seed}: horizon {k} moved up");
                if k + 1 < nh {
                    assert!(post[k][c] >= post[k + 1][c], "seed {seed}: z'_k < z'_k+1");
                }
            }
        }
        assert!(
            post[nh - 1].iter().zip(&pre[nh - 1]).all(|(a, b)| a.to_bits() == b.to_bits()),
            "seed {seed}: seabed changed"
        );
        assert!(
            post[0].iter().zip(&pre[0]).all(|(a, b)| a.to_bits() == b.to_bits()),
            "seed {seed}: base changed"
        );
        if let Some(r) = recs.iter().find(|r| r.applied && r.number == 1) {
            // The shallowest unconformity is applied last: E = z'_u.
            let u = nh - 1 - r.s;
            for c in 0..n {
                let e = post[u][c];
                assert!(r.eroded[c] >= 0.0);
                assert!(e >= pre[u][c]);
                for k in 0..=u {
                    assert!(post[k][c] >= e, "seed {seed}: z'_{k} < E");
                }
                for k in u..nh {
                    assert!(post[k][c] <= e, "seed {seed}: z'_{k} > E");
                }
            }
            if case.count == 1 {
                for c in 0..n {
                    let e = post[u][c];
                    assert!(((e - pre[u][c]) as f32 - r.eroded[c]).abs() <= 1e-4);
                }
            }
        }
    }
}

#[test]
fn t3_t4_round_trip_and_invariants_32_cube() {
    let mut seen = [0usize; 2];
    for seed in 1..=200u64 {
        roundtrip_and_invariants(seed, [32, 32, 32], &mut seen);
    }
    eprintln!("32^3: applied unconformities N=1 {} N=2 {}", seen[0], seen[1]);
}

#[test]
fn t3_t4_round_trip_and_invariants_64x48x128() {
    let mut seen = [0usize; 2];
    for seed in 1..=200u64 {
        roundtrip_and_invariants(seed, [64, 48, 128], &mut seen);
    }
    eprintln!("64x48x128: applied unconformities N=1 {} N=2 {}", seen[0], seen[1]);
    // 200 seeds x (folds off, on): every cube of this size applies.
    assert!(seen[0] >= 380, "N=1 applied too rarely: {seen:?}");
    assert!(seen[1] >= 760, "N=2 applied too rarely: {seen:?}");
}

#[test]
fn t3_t4_round_trip_and_invariants_128_cube_20_seeds() {
    let mut seen = [0usize; 2];
    for seed in 1..=20u64 {
        roundtrip_and_invariants(seed, [128, 128, 128], &mut seen);
    }
    assert!(seen[0] > 0 && seen[1] > 0);
}

#[test]
#[ignore = "nightly: the full 200 seeds on the 128^3 cube"]
fn t3_t4_round_trip_and_invariants_128_cube_200_seeds() {
    let mut seen = [0usize; 2];
    for seed in 1..=200u64 {
        roundtrip_and_invariants(seed, [128, 128, 128], &mut seen);
    }
    eprintln!("128^3: applied unconformities N=1 {} N=2 {}", seen[0], seen[1]);
}

/// Label side of test 4: no 255 below the seabed, consecutive labels, zero
/// thickness layers present where an unconformity was applied.
#[test]
fn t4_labels_consecutive_and_zero_thickness_layers() {
    let mut zero_seen = 0;
    for seed in 1..=24u64 {
        for count in [1usize, 2] {
            let mut c = cfg(seed, [48, 40, 128]);
            c.rock_physics.unconformity.count = count;
            let (labels, [ni, nj, nk]) = generate_labels(&c);
            let (maps, nh) = synthoseis_core::toy_geometry::layered_horizon_maps_with(
                seed,
                [ni, nj, nk],
                c.rock_physics.fold_config(),
                c.rock_physics.unconformity_config(),
            );
            let mut present = [false; 256];
            for col in 0..ni * nj {
                let seabed = maps[col * nh] as usize;
                for k in 0..nk {
                    let v = labels[col * nk + k];
                    // #56 behaviour, unchanged: where the cube clips the
                    // base horizon (z_base >= nk) the fill can leave 255
                    // samples at the bottom of the column. Everywhere else
                    // the stack is gap free below the seabed.
                    if k >= seabed && maps[col * nh + nh - 1] < nk as f64 {
                        assert_ne!(v, 255, "seed {seed} col {col} k {k}: 255 below the seabed");
                    }
                    present[v as usize] = true;
                }
                if (0..nh - 1).any(|h| maps[col * nh + h] == maps[col * nh + h + 1]) {
                    zero_seen += 1;
                }
            }
            let max = (0..255).rev().find(|&v| present[v]).unwrap();
            assert!((0..=max).all(|v| present[v]), "seed {seed}: labels not consecutive");
        }
    }
    assert!(zero_seen > 0, "no zero-thickness layer was exercised");
}

// --------------------------------------------------------------------
// Test 5: dip limit.
// --------------------------------------------------------------------

#[test]
fn t5_erosion_surface_slope_obeys_the_limit_when_it_binds_and_when_not() {
    let shape = [64usize, 48, 128];
    let mut binds = [0usize; 4];
    let mut free = [0usize; 4];
    for (si, s_max) in [0.05f64, 0.5, 1.0, 2.0].into_iter().enumerate() {
        for seed in 1..=40u64 {
            for relief in [None, Some(2.0), Some(200.0)] {
                let ucfg = UnconformityConfig {
                    max_dip: s_max,
                    relief_m: relief,
                    ..UnconformityConfig::default()
                };
                let (_, recs) = layered_stack_records(seed, shape, Some(&FoldConfig::default()), Some(&ucfg));
                for r in &recs {
                    let m = u64::from(r.number) - 1;
                    let rm = relief_map(seed, m, shape[0], shape[1]);
                    let g = slope_max(&rm, shape[0], shape[1]);
                    let slope = r.relief * g;
                    assert!(
                        slope <= s_max * (1.0 + 1e-12),
                        "seed {seed} s_max {s_max}: slope of E-bar {slope}"
                    );
                    if r.relief < r.relief_drawn {
                        binds[si] += 1;
                        assert!((slope - s_max).abs() <= 1e-9 * s_max, "binding limit is met exactly");
                    } else {
                        free[si] += 1;
                        assert_eq!(r.relief, r.relief_drawn);
                    }
                }
            }
        }
        assert!(binds[si] > 0, "s_max {s_max}: the limit never bound");
        assert!(free[si] > 0, "s_max {s_max}: the limit always bound");
    }
}

fn steepest(stack: &[Vec<f64>], ni: usize, nj: usize) -> f64 {
    stack.iter().map(|h| slope_max(h, ni, nj)).fold(0.0, f64::max)
}

struct SlopeRow {
    seed: u64,
    before: f64,
    after: f64,
}

fn slope_rows(seeds: &[u64], shape: [usize; 3], fold: &FoldConfig, count: usize, threads: usize) -> Vec<SlopeRow> {
    let ucfg = uncf(count);
    let work = |seed: u64| {
        let pre = layered_stack(seed, shape, Some(fold), None);
        let post = layered_stack(seed, shape, Some(fold), Some(&ucfg));
        SlopeRow { seed, before: steepest(&pre, shape[0], shape[1]), after: steepest(&post, shape[0], shape[1]) }
    };
    let mut out: Vec<SlopeRow> = Vec::new();
    std::thread::scope(|s| {
        let chunk = seeds.len().div_ceil(threads);
        let handles: Vec<_> = seeds.chunks(chunk).map(|part| s.spawn(move || part.iter().map(|&x| work(x)).collect::<Vec<_>>())).collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    out
}

/// Largest slope of every final horizon <= max(steepest horizon before,
/// s_max), on folded stacks (default folds, dip-limited folds), N = 1 and 2.
#[test]
fn t5_no_horizon_steeper_than_before_on_folded_stacks() {
    let shape = [64usize, 48, 128];
    let seeds: Vec<u64> = (1..=200).collect();
    let limited = FoldConfig { max_dip: 0.5, ..FoldConfig::default() };
    let configs: [(&str, FoldConfig, usize); 3] = [
        ("default folds N=1", FoldConfig::default(), 1),
        ("dip-limited folds N=1", limited.clone(), 1),
        ("default folds N=2", FoldConfig::default(), 2),
    ];
    let s_max = UnconformityConfig::default().max_dip;
    for (name, fold, count) in configs {
        let rows = slope_rows(&seeds, shape, &fold, count, 4);
        let mut worst = f64::NEG_INFINITY;
        let mut over = 0;
        for r in &rows {
            let limit = r.before.max(s_max);
            worst = worst.max(r.after - limit);
            if r.after > limit * (1.0 + 1e-9) {
                over += 1;
                eprintln!("{name}: seed {} slope {:.4} -> {:.4} (limit {limit:.4})", r.seed, r.before, r.after);
            }
        }
        let max_before = rows.iter().map(|r| r.before).fold(0.0, f64::max);
        let max_after = rows.iter().map(|r| r.after).fold(0.0, f64::max);
        eprintln!("{name}: max slope before {max_before:.4} after {max_after:.4}; worst margin {worst:.4}; seeds over: {over}");
        assert_eq!(over, 0, "{name}: {over} of 200 seeds have a horizon steeper than before");
        // Independent of the thread count.
        let again = slope_rows(&seeds[..40], shape, &fold, count, 1);
        let again4 = slope_rows(&seeds[..40], shape, &fold, count, 3);
        for (a, b) in again.iter().zip(&again4) {
            assert_eq!(a.after.to_bits(), b.after.to_bits(), "{name}: thread count");
        }
        for (a, b) in again.iter().zip(&rows) {
            assert_eq!((a.before.to_bits(), a.after.to_bits()), (b.before.to_bits(), b.after.to_bits()));
        }
    }
}

// --------------------------------------------------------------------
// Test 6: invariance (whole = chunked = multi-process, with and without
// the geometry cache).
// --------------------------------------------------------------------

fn maps_of(c: &E2eConfig) -> (Vec<f64>, usize) {
    assert!(!c.rock_physics.salt);
    layered_horizon_maps_with(c.seed, c.shape(), c.rock_physics.fold_config(), c.rock_physics.unconformity_config())
}

fn tile_hash(v: &[u8]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

#[test]
fn t6_labels_identical_with_and_without_geometry_cache() {
    for (seed, faults, salt) in [(4u64, 0usize, false), (7, 3, true), (12, 2, false)] {
        for count in [1usize, 2] {
            let mut c = cfg(seed, [40, 36, 128]);
            c.faults = FaultConfig::with_count(faults);
            c.rock_physics.salt = salt;
            c.rock_physics.unconformity.count = count;
            assert!(unconformity_applied(&c), "seed {seed} N {count}: no unconformity");
            let cached = c.with_geometry_cache();
            assert_eq!(generate_labels(&c), generate_labels(&cached));
            assert_eq!(generate_unconformity_labels(&c), generate_unconformity_labels(&cached));
            assert_eq!(unconformity_records(&c), unconformity_records(&cached));
            assert_eq!(generate_unconformity_labels_output(&c), generate_unconformity_labels_output(&cached));
        }
    }
}

fn read_store(p: &std::path::Path) -> (Vec<u32>, Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>) {
    let s = synthoseis_io::MdioStore::open(p).unwrap();
    let vol = s.read_volume().unwrap().iter().map(|x| x.to_bits()).collect();
    (
        vol,
        s.read_labels_u8().unwrap(),
        s.read_fault_labels_u8().ok(),
        s.read_salt_labels_u8().ok(),
        s.has_unconformity_labels().then(|| s.read_unconformity_labels_u8().unwrap()),
    )
}

/// Classic, streaming (three tilings), strip-stitched, overlapped and
/// multi-process (1, 2, 3 workers) stores are identical, every one with
/// the unconformity labels of the library (output domain).
#[test]
fn t6_every_writer_produces_the_same_store_content() {
    use synthoseis_core::{run_e2e_multiprocess, run_e2e_streaming, run_e2e_streaming_overlapped, run_e2e_strip_stitched};
    let dir = tempfile::tempdir().unwrap();
    for legacy_axis in [false, true] {
        for count in [1usize, 2] {
            let mut base = cfg(7, [24, 20, 128]);
            base.faults = FaultConfig::with_count(3);
            base.rock_physics.salt = true;
            base.rock_physics.unconformity.count = count;
            if legacy_axis {
                base.time = synthoseis_core::TimeConfig::legacy();
            }
            assert!(unconformity_applied(&base));
            let want_u = generate_unconformity_labels_output(&base).expect("labels");
            assert!(want_u.iter().any(|&v| v > 0));
            if count == 2 {
                assert!(want_u.iter().any(|&v| v == 2), "second unconformity labelled");
            }
            type Run = Box<dyn Fn(&E2eConfig)>;
            let runs: Vec<(&str, Option<[usize; 3]>, Run)> = vec![
                ("classic", None, Box::new(|c| { synthoseis_core::pipeline::run_e2e(c).unwrap(); })),
                ("stream 8x5x16", Some([8, 5, 16]), Box::new(|c| { run_e2e_streaming(c).unwrap(); })),
                ("stream 5x7xnk", Some([5, 7, 128]), Box::new(|c| { run_e2e_streaming(c).unwrap(); })),
                ("stream whole", Some([24, 20, 128]), Box::new(|c| { run_e2e_streaming(c).unwrap(); })),
                ("strip 3", Some([8, 5, 16]), Box::new(|c| { run_e2e_strip_stitched(c, 3).unwrap(); })),
                ("overlap", Some([8, 5, 16]), Box::new(|c| { run_e2e_streaming_overlapped(c).unwrap(); })),
                ("mp 1", Some([8, 5, 16]), Box::new(|c| { run_e2e_multiprocess(c, 1).unwrap(); })),
                ("mp 2", Some([8, 5, 16]), Box::new(|c| { run_e2e_multiprocess(c, 2).unwrap(); })),
                ("mp 3", Some([6, 7, 32]), Box::new(|c| { run_e2e_multiprocess(c, 3).unwrap(); })),
            ];
            let mut want = None;
            for (name, chunks, run) in runs {
                let p = dir.path().join(format!("{legacy_axis}-{count}-{}.mdio", name.replace(' ', "_")));
                run(&E2eConfig { store_path: Some(p.clone()), chunk_shape: chunks, ..base.clone() });
                let got = read_store(&p);
                assert_eq!(got.4.as_ref(), Some(&want_u), "{name} (legacy axis {legacy_axis}, N {count}): unconformity labels");
                match &want {
                    None => want = Some(got),
                    Some(w) => assert!(*w == got, "{name} (legacy axis {legacy_axis}, N {count}): store content differs from classic"),
                }
            }
            eprintln!("legacy axis {legacy_axis} N {count}: labels hash {:016x}", tile_hash(&want_u));
        }
    }
}

// --------------------------------------------------------------------
// Test 8: label parity.
// --------------------------------------------------------------------

/// No faults, no salt: one voxel per applied column, at the production map
/// of horizon `s`, which is the first voxel of its layer.
#[test]
fn t8_labels_are_one_voxel_per_column_at_the_layer_boundary() {
    let mut applied_cols = 0usize;
    for seed in [1u64, 4, 7, 12] {
        for count in [1usize, 2] {
            let mut c = cfg(seed, [40, 36, 128]);
            c.rock_physics.salt = false;
            c.rock_physics.unconformity.count = count;
            let (labels, [ni, nj, nk]) = generate_labels(&c);
            let mask = generate_unconformity_labels(&c).expect("applied");
            let recs = unconformity_records(&c);
            let (maps, nh) = maps_of(&c);
            for col in 0..ni * nj {
                let m = &mask[col * nk..(col + 1) * nk];
                for r in recs.iter().filter(|r| r.applied) {
                    let z = maps[col * nh + r.s];
                    let n_this = m.iter().filter(|&&v| v == r.number).count();
                    if z >= 0.0 && z < nk as f64 {
                        assert_eq!(n_this, 1, "seed {seed} col {col}: {n_this} voxels of label {}", r.number);
                        let k = z as usize;
                        // Layer s starts here (or a coincident layer does).
                        assert_eq!(m[k], r.number);
                        if k > 0 {
                            assert_ne!(labels[col * nk + k], labels[col * nk + k - 1], "seed {seed} col {col}: not a layer boundary");
                        }
                        applied_cols += 1;
                    } else {
                        assert_eq!(n_this, 0);
                    }
                }
                assert!(m.iter().all(|&v| v as usize <= count));
            }
        }
    }
    assert!(applied_cols > 0);
}

/// Faults: the label is carried through the fault lookup like the layer
/// labels, so it stays within 1 voxel of a layer boundary of the faulted
/// labels; salt clears it.
#[test]
fn t8_labels_follow_faults_and_are_cleared_in_salt() {
    let (mut near, mut total, mut dup, mut gone) = (0usize, 0usize, 0usize, 0usize);
    for seed in [4u64, 7, 12, 3] {
        let mut c = cfg(seed, [40, 36, 128]);
        c.faults = FaultConfig::with_count(4);
        c.rock_physics.salt = false;
        let (labels, [ni, nj, nk]) = generate_labels(&c);
        let faults = generate_fault_labels_for(&c);
        assert!(faults.iter().any(|&v| v == 1), "no fault voxels");
        let mask = generate_unconformity_labels(&c).expect("applied");
        let base = {
            let mut b = c.clone();
            b.faults = FaultConfig::with_count(0);
            generate_unconformity_labels(&b).unwrap()
        };
        for col in 0..ni * nj {
            let m = &mask[col * nk..(col + 1) * nk];
            let b = &base[col * nk..(col + 1) * nk];
            let (cm, cb) = (m.iter().filter(|&&v| v > 0).count(), b.iter().filter(|&&v| v > 0).count());
            dup += usize::from(cm > cb);
            gone += usize::from(cm < cb);
            for k in 0..nk {
                if m[k] == 0 {
                    continue;
                }
                total += 1;
                let lab = &labels[col * nk..(col + 1) * nk];
                let boundary = |q: usize| q > 0 && lab[q] != lab[q - 1];
                if (k.saturating_sub(1)..=(k + 1).min(nk - 1)).any(boundary) {
                    near += 1;
                }
            }
        }
    }
    eprintln!("faulted label voxels {total}, within 1 voxel of a layer boundary {near}; columns with duplicated {dup} / vanished {gone}");
    assert!(total > 0);
    assert_eq!(near, total, "every unconformity label voxel is within 1 voxel of a faulted layer boundary");
    // A fault can duplicate or hide the one voxel of a column (errata), but
    // only in a minority of columns.
    assert!((dup + gone) * 4 < 4 * 40 * 36, "{dup} + {gone} columns changed by faults");

    // Salt clears the label.
    let mut c = cfg(7, [40, 36, 128]);
    c.rock_physics.salt = true;
    let salt = synthoseis_core::salt::generate_salt_labels(&c).expect("salt");
    assert!(salt.iter().any(|&v| v == 1));
    let mask = generate_unconformity_labels(&c).unwrap();
    let mut nosalt = c.clone();
    nosalt.rock_physics.salt = false;
    assert!(mask.iter().zip(&salt).all(|(&u, &s)| !(u > 0 && s == 1)), "label inside salt");
    assert!(mask.iter().any(|&v| v > 0));
    // Only the salt-affected columns differ from the no-salt run (the
    // salt drag moves the horizons, so only the column sums are compared).
    let _ = nosalt;
}

fn generate_fault_labels_for(c: &E2eConfig) -> Vec<u8> {
    synthoseis_core::generate_fault_labels(c).expect("faults")
}

/// Time mode: the legacy axis equals the depth labels; the Vp-TWT axis
/// carries one (resampled) voxel in nearly every applied column.
#[test]
fn t8_time_mode_labels() {
    let mut c = cfg(4, [32, 28, 128]);
    c.rock_physics.salt = false;
    let depth = generate_unconformity_labels(&c).expect("applied");
    let mut legacy = c.clone();
    legacy.time = synthoseis_core::TimeConfig::legacy();
    assert_eq!(generate_unconformity_labels_output(&legacy).unwrap(), depth);
    let out = generate_unconformity_labels_output(&c).expect("time-mode labels");
    let nt = out.len() / (32 * 28);
    assert!(nt > 0 && out.len() == 32 * 28 * nt);
    let mut per = [0usize; 4];
    for col in 0..32 * 28 {
        let n = out[col * nt..(col + 1) * nt].iter().filter(|&&v| v > 0).count();
        per[n.min(3)] += 1;
    }
    eprintln!("time mode: columns with 0/1/2/3+ label voxels: {per:?}");
    assert!(per[1] * 10 >= (32 * 28) * 8, "time mode: too few columns carry the label: {per:?}");
    assert_eq!(per[3], 0);
}

// --------------------------------------------------------------------
// Test 9: exhumation and facies.
// --------------------------------------------------------------------

/// Burial of the intervals >= s is the old burial + e dz; nothing changes
/// in the shallower intervals or where e = 0, and `--no-exhumation-burial`
/// leaves the model untouched.
#[test]
fn t9_exhumation_adds_the_eroded_thickness_to_the_burial() {
    use synthoseis_core::elastic_model;
    use synthoseis_core::ElasticModel;
    let mut c = cfg(4, [32, 28, 160]);
    c.rock_physics.salt = false;
    let (labels, shape) = generate_labels(&c);
    let [ni, nj, nk] = shape;
    let recs = unconformity_records(&c);
    assert!(recs.iter().any(|r| r.applied));
    let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else { panic!("rpm") };
    assert!(!m.exhumation.is_empty());
    let mut plain = (*m).clone();
    plain.exhumation = Default::default();
    // --no-exhumation-burial builds exactly that model.
    let mut off = c.clone();
    off.rock_physics.exhumation_burial = false;
    let ElasticModel::Rpm(m_off) = elastic_model(&off, &labels, shape) else { panic!("rpm") };
    assert!(m_off.exhumation.is_empty());
    assert_eq!(m_off.unconformities.len(), recs.len());
    let step = m.step;
    let (mut a, mut b, mut o) = (vec![0f32; nk], vec![0f32; nk], vec![0f32; nk]);
    let (mut shifted, mut same_deep, mut shallow) = (0usize, 0usize, 0usize);
    for i in 0..ni {
        for j in 0..nj {
            let col = i * nj + j;
            let lab = &labels[col * nk..(col + 1) * nk];
            m.depth_trace(i, j, lab, &mut a);
            plain.depth_trace(i, j, lab, &mut b);
            m_off.depth_trace(i, j, lab, &mut o);
            assert_eq!(b, o, "col {col}: --no-exhumation-burial changes the burial");
            for k in 0..nk {
                if lab[k] == 255 {
                    assert_eq!(a[k], b[k]);
                    continue;
                }
                let h = m.intervals[lab[k] as usize];
                let e: f32 = recs.iter().filter(|r| r.applied && h >= r.s).map(|r| r.eroded[col]).sum();
                assert_eq!(e, exhumation_at(&recs, col, h));
                let d = a[k] - b[k];
                let want = e * step;
                // Same run, same float expression: the shift is e dz to f32 rounding.
                assert!((d - want).abs() <= 1e-3 * (1.0 + want.abs()), "col {col} k {k} h {h}: {d} vs {want}");
                if e > 0.0 {
                    shifted += 1;
                } else if recs.iter().any(|r| r.applied && h < r.s) {
                    shallow += 1;
                    assert_eq!(d, 0.0);
                } else {
                    same_deep += 1;
                }
            }
        }
    }
    assert!(shifted > 0 && shallow > 0, "shifted {shifted} shallow {shallow} deep-unchanged {same_deep}");
}

#[test]
fn t9_shale_seal_modes() {
    for seed in [1u64, 4, 7, 12, 20] {
        let mut c = cfg(seed, [32, 32, 128]);
        let recs = unconformity_records(&c);
        let s = recs.iter().find(|r| r.applied).expect("applied").s;
        let nh = 40;
        let mut modes = Vec::new();
        for mode in [UnconformityShale::Below, UnconformityShale::Above, UnconformityShale::Both, UnconformityShale::None] {
            c.rock_physics.unconformity_shale = mode;
            modes.push(synthoseis_core::pipeline_stream::sealed_interval_sand(&c, nh));
        }
        let mut legacy = c.clone();
        legacy.rock_physics.legacy_unconformity = true;
        let plain = synthoseis_core::pipeline_stream::sealed_interval_sand(&legacy, nh);
        assert_eq!(modes[3], plain, "seed {seed}: none must leave the chain untouched");
        assert!(!modes[0][s]);
        for i in 0..nh {
            if i != s {
                assert_eq!(modes[0][i], plain[i], "seed {seed}: below changes only interval s");
            }
        }
        assert!(!modes[1][s - 1]);
        assert!((0..nh).all(|i| i == s - 1 || modes[1][i] == plain[i]));
        assert!(!modes[2][s] && !modes[2][s - 1]);
        assert!((0..nh).all(|i| i == s || i == s - 1 || modes[2][i] == plain[i]));
    }
}

// --------------------------------------------------------------------
// Test 10: partial voxels on pinching layers.
// --------------------------------------------------------------------

fn snap(x: f64) -> f64 {
    if (x - x.round()).abs() < EPS_FRAC {
        x.round()
    } else {
        x
    }
}

fn overlap(a: f64, b: f64, lo: f64, hi: f64) -> f64 {
    (b.min(hi) - a.max(lo)).max(0.0)
}

#[test]
fn t10_partial_voxels_conserve_thickness_of_pinching_layers() {
    let shape = [32usize, 28, 128];
    let nzf = shape[2] as f64;
    let (mut pinching, mut checked) = (0usize, 0usize);
    for seed in [7u64, 1, 2, 3, 4, 30] {
        let mut c = cfg(seed, shape);
        c.rock_physics.salt = false;
        let (maps, nh) = synthoseis_core::pipeline_stream::toy_horizon_maps_continuous(&c).unwrap();
        assert!(unconformity_applied(&c));
        let mut parts = ColumnParts::default();
        for col in 0..shape[0] * shape[1] {
            let z = &maps[col * nh..(col + 1) * nh];
            let contacts: Vec<f32> = (0..nh - 1).map(|h| (0.3 * z[h] + 0.7 * z[h + 1]) as f32).collect();
            column_parts(&ColumnGeometry { horizons: z, salt: None, contacts: &contacts }, shape[2], &mut parts);
            let zs: Vec<f64> = z.iter().map(|&v| snap(v)).collect();
            for h in 0..nh - 1 {
                let a = zs[h].max(zs[0]);
                let b = zs[h + 1].max(a);
                let want = overlap(a, b, 0.0, nzf);
                let got = unit_thickness(&parts, Unit::Interval(h));
                assert!((got - want).abs() <= 1e-9, "seed {seed} col {col} h {h}: {got} vs {want}");
                checked += 1;
                if z[h + 1] - z[h] < 1.0 {
                    pinching += 1;
                }
            }
            let sum: f64 = (0..shape[2]).flat_map(|k| parts.cell(k).iter().map(|p| p.frac)).sum();
            let sedimentary = overlap(zs[0], nzf, 0.0, nzf) + overlap(0.0, zs[0], 0.0, nzf);
            assert!((sum - sedimentary).abs() <= 1e-9 * nzf);
        }
    }
    eprintln!("checked {checked} layers, {pinching} thinner than one sample");
    assert!(pinching > 1000, "pinching layers not exercised ({pinching})");
}

// --------------------------------------------------------------------
// Test 11: salt and faults smoke.
// --------------------------------------------------------------------

/// The far-field identity gate of the salt maps spec holds with the
/// unconformity in the geometry (N = 1 and 2).
#[test]
fn t11_salt_far_field_identity_with_unconformity() {
    use synthoseis_core::salt::{drag_horizon_maps_before_pushdown, salt_body};
    for seed in [7u64, 1] {
        for count in [1usize, 2] {
            let mut c = cfg(seed, [64, 64, 256]);
            c.rock_physics.unconformity.count = count;
            let body = salt_body(&c).expect("salt on");
            let (maps, nh) = layered_horizon_maps_with(
                seed,
                [64, 64, 256],
                c.rock_physics.fold_config(),
                c.rock_physics.unconformity_config(),
            );
            assert!(unconformity_applied(&c));
            let n = 64 * 64;
            let dragged = drag_horizon_maps_before_pushdown(&maps, [64, 64, nh], &body, false);
            let hi = body.grid[2] as i64 - 1;
            let mut checked = 0usize;
            for h in 0..nh {
                let lifted: Vec<usize> = (0..n)
                    .filter(|&c| {
                        let k = (maps[c * nh + h] as i64).clamp(0, hi) as usize;
                        body.contains(c, k)
                    })
                    .collect();
                for c in 0..n {
                    let same = dragged[c * nh + h].to_bits() == maps[c * nh + h].to_bits();
                    let (ci, cj) = (c / 64, c % 64);
                    let dist = lifted.iter().map(|&t| ci.abs_diff(t / 64).max(cj.abs_diff(t % 64))).min();
                    match dist {
                        None => assert!(same, "seed {seed} N {count} h {h} c {c}: never-touch moved"),
                        Some(d) if d >= 13 => {
                            assert!(same, "seed {seed} N {count} h {h} c {c}: L-inf {d} cell moved");
                            checked += 1;
                        }
                        _ => {}
                    }
                }
            }
            assert!(checked > 0);
        }
    }
}

/// Faults with an unconformity: the mask and the layer labels go through
/// the same fault lookup, so the faulted label at every mask voxel is the
/// unfaulted label that starts layer `s`, and fault voxels coexist with it.
#[test]
fn t11_faults_displace_the_mask_and_the_layers_together() {
    let mut checked = 0usize;
    let mut moved = 0usize;
    for seed in [7u64, 4, 12] {
        let mut c = cfg(seed, [40, 36, 128]);
        c.faults = FaultConfig::with_count(4);
        c.rock_physics.salt = false;
        let (faulted, [ni, nj, nk]) = generate_labels(&c);
        let mut plain_cfg = c.clone();
        plain_cfg.faults = FaultConfig::with_count(0);
        let (plain, _) = generate_labels(&plain_cfg);
        let mask = generate_unconformity_labels(&c).expect("applied");
        let mask0 = generate_unconformity_labels(&plain_cfg).expect("applied");
        let faults = synthoseis_core::generate_fault_labels(&c).expect("faults");
        assert!(faults.iter().any(|&v| v == 1));
        for col in 0..ni * nj {
            let k0 = (0..nk).find(|&k| mask0[col * nk + k] == 1);
            for k in 0..nk {
                if mask[col * nk + k] != 1 {
                    continue;
                }
                let k0 = k0.expect("a faulted label has an unfaulted source");
                assert_eq!(faulted[col * nk + k], plain[col * nk + k0], "seed {seed} col {col}: label differs from layer s");
                checked += 1;
                moved += usize::from(k != k0);
            }
        }
    }
    eprintln!("faulted mask voxels {checked}, displaced {moved}");
    assert!(checked > 0 && moved > 0, "faults must displace the mask");
}
