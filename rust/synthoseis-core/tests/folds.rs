//! Validation tests of the analytic folds (spec "analytic folds" §8).

use std::f64::consts::TAU;
use synthoseis_core::folds::{self, Fold, FoldConfig, FoldSet, FoldStyle};
use synthoseis_core::toy_geometry::{
    layered_horizon_maps_continuous_with, layered_horizon_maps_with, layered_stack_unfolded,
};

fn cyl(wl_cols: f64, amp: f64, bin: f64) -> Fold {
    Fold::Cylindrical {
        x0: [0.0, 0.0],
        theta: 0.3,
        wavelength: wl_cols * bin,
        amp,
        phase: 0.4,
        tan_plunge: 0.0,
        pericline: f64::INFINITY,
        cot_dip: 0.0,
    }
}

fn set_of(f: Vec<Fold>, style: FoldStyle) -> FoldSet {
    let cfg = FoldConfig {
        style,
        growth: 0.0,
        ..FoldConfig::default()
    };
    FoldSet {
        folds: f,
        scaled: 0,
        factor: 1.0,
        dip_before: 0.0,
        dip_after: 0.0,
        cfg,
    }
}

/// Test 1: central differences of the analytic field against the exact
/// discrete derivatives of a sinusoid, `F' sin(kh)/(kh)` → `A k cos·sin(kh)/(kh)`
/// i.e. `A cos(ks+φ) sin(kh)/h` and `−A sin(ks+φ) (2 − 2cos kh)/h²`, to
/// 1e-12; and the error against the continuous §3.7 derivatives is second
/// order: λ = 8 → 32 columns divides it by about 16.
#[test]
fn analytic_dip_and_curvature() {
    let cfg = FoldConfig::default();
    let bin = cfg.bin_m;
    let mut errs = Vec::new();
    for wl in [8.0, 32.0] {
        let amp = 3.0;
        let s = set_of(vec![cyl(wl, amp, bin)], FoldStyle::Similar);
        let k = TAU / (wl * bin);
        let kh = k * bin;
        let (mut e_exact_d, mut e_exact_c) = (0.0f64, 0.0f64);
        let (mut e_cont_d, mut e_cont_c) = (0.0f64, 0.0f64);
        let (c, sn) = (0.3f64.cos(), 0.3f64.sin());
        for n in 0..200 {
            let r = n as f64 * bin * 0.37;
            let p = |q: f64| s.field((r + q) * c, (r + q) * sn, 0.0).0;
            let fd = (p(bin) - p(-bin)) / (2.0 * bin);
            let f2 = (p(bin) - 2.0 * p(0.0) + p(-bin)) / (bin * bin);
            let ph = k * r + 0.4;
            let exact_d = amp * ph.cos() * kh.sin() / bin;
            let exact_c = -amp * ph.sin() * (2.0 - 2.0 * kh.cos()) / (bin * bin);
            e_exact_d = e_exact_d.max((fd - exact_d).abs() / (amp * k));
            e_exact_c = e_exact_c.max((f2 - exact_c).abs() / (amp * k * k));
            e_cont_d = e_cont_d.max((fd - amp * k * ph.cos()).abs() / (amp * k));
            e_cont_c = e_cont_c.max((f2 + amp * k * k * ph.sin()).abs() / (amp * k * k));
        }
        assert!(e_exact_d < 1e-12, "λ={wl}: dip vs discrete {e_exact_d}");
        assert!(e_exact_c < 1e-12, "λ={wl}: curvature vs discrete {e_exact_c}");
        errs.push((e_cont_d, e_cont_c));
    }
    let (rd, rc) = (errs[0].0 / errs[1].0, errs[0].1 / errs[1].1);
    assert!((14.0..17.0).contains(&rd), "dip error ratio {rd}");
    assert!((14.0..17.0).contains(&rc), "curvature error ratio {rc}");
    // Dome: crest curvature A/R² and slope ring (A/R) e^{-1/2} at r = R.
    let (amp, r) = (5.0, 10.0 * bin);
    let d = set_of(
        vec![Fold::Dome {
            x0: [0.0, 0.0],
            theta: 0.0,
            sign: -1.0,
            amp,
            ra: r,
            rb: r,
            cot_dip: 0.0,
        }],
        FoldStyle::Similar,
    );
    let h = 1e-2;
    let f = |x: f64| d.field(x, 0.0, 0.0).0;
    let curv = (f(h) - 2.0 * f(0.0) + f(-h)) / (h * h);
    assert!(
        (curv - amp / (r * r)).abs() / (amp / (r * r)) < 1e-4,
        "{curv}"
    );
    let slope = d.field(r, 0.0, 0.0).1.abs();
    assert!((slope - amp / r * (-0.5f64).exp()).abs() < 1e-12);
    let ring = (1..400)
        .map(|n| d.field(n as f64 * r / 100.0, 0.0, 0.0).1.abs())
        .fold(0.0, f64::max);
    assert!(ring <= slope + 1e-12);
}

fn flat_stack(nh: usize, n: usize, dz: f64) -> Vec<Vec<f64>> {
    // deepest first, layer thickness `dz` samples, seabed at 10.
    (0..nh)
        .map(|s| vec![10.0 + (nh - 1 - s) as f64 * dz; n])
        .collect()
}

/// Test 2: style A with q = 0 preserves vertical thickness to 1e-12 below
/// the seabed; style B preserves true (normal) thickness to 1e-3 samples.
#[test]
fn thickness_preservation() {
    let (ni, nj) = (40, 30);
    let s = set_of(vec![cyl(24.0, 6.0, 12.5)], FoldStyle::Similar);
    let stack = flat_stack(6, ni * nj, 5.0);
    let sh = folds::shifts(&s, &stack, ni, nj);
    for h in 0..4 {
        for c in 0..ni * nj {
            let a = stack[h][c] + sh[h][c] - stack[h + 1][c] - sh[h + 1][c];
            assert!((a - 5.0).abs() < 1e-12);
        }
    }
    // Style B, on a 1-D profile across a pure cylinder (θ = 0): the closest
    // distance from each point of horizon h to horizon h+1 equals the flat
    // true thickness.
    let p = Fold::Cylindrical {
        x0: [0.0, 0.0],
        theta: 0.0,
        wavelength: 40.0 * 12.5,
        amp: 8.0,
        phase: 0.0,
        tan_plunge: 0.0,
        pericline: f64::INFINITY,
        cot_dip: 0.0,
    };
    let b = set_of(vec![p], FoldStyle::Parallel);
    let dz = b.cfg.dz_m;
    let kappa = b.curvature_bound();
    let curve = |t: f64, x: f64| {
        let (fx, _) = folds::foot_point(&b, x, 0.0, t);
        let (s, gx, _) = b.field(fx, 0.0, 0.0);
        let n = (1.0 + gx * gx).sqrt();
        (fx + t * gx / n, s - t / n)
    };
    for (t0, t1) in [(20.0, 40.0), (40.0, 60.0)] {
        assert!(t1 * kappa <= 0.8);
        for k in 0..60 {
            let x = k as f64 * 12.5;
            let (px, pz) = curve(t0, x);
            // closest distance to horizon t1, dense search + refinement
            let mut best = f64::INFINITY;
            for m in -4000..4000 {
                let (qx, qz) = curve(t1, x + m as f64 * 0.01);
                best = best.min(((qx - px).powi(2) + (qz - pz).powi(2)).sqrt());
            }
            assert!(((best - (t1 - t0)) / dz).abs() < 1e-3, "x={x}: {best}");
        }
    }
}

/// Distance (m) from `q` to the parallel-style horizon at true distance
/// `t` above the reference, parametrised directly by the foot point
/// `(x', y') ↦ (x', y', S) − t n` (no Newton solve), by a coarse grid then
/// a shrinking pattern search.
fn dist_to_horizon(set: &FoldSet, q: [f64; 3], t: f64) -> f64 {
    let pt = |x: f64, y: f64| {
        let (s, fx, fy) = set.field(x, y, 0.0);
        let n = (1.0 + fx * fx + fy * fy).sqrt();
        let p = [x + t * fx / n, y + t * fy / n, s - t / n];
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    let r = 2.0 * t + 50.0;
    let (mut bx, mut by, mut bd) = (q[0], q[1], f64::INFINITY);
    for a in -40..=40 {
        for b in -40..=40 {
            let (x, y) = (q[0] + a as f64 * r / 40.0, q[1] + b as f64 * r / 40.0);
            let d = pt(x, y);
            if d < bd {
                (bx, by, bd) = (x, y, d);
            }
        }
    }
    let mut step = r / 40.0;
    while step > 1e-7 {
        let mut moved = false;
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
            let d = pt(bx + dx * step, by + dy * step);
            if d < bd {
                (bx, by, bd) = (bx + dx * step, by + dy * step, d);
                moved = true;
            }
        }
        if !moved {
            step *= 0.5;
        }
    }
    bd
}

/// Test 3 (independent check, replaces the round trip): on real layered
/// stacks with domes and oblique, plunging folds, the parallel-style
/// horizons from [`folds::shifts`] (Newton foot points) lie on the analytic
/// offset surfaces, and the direct-search distance from each to the next
/// horizon is the unfolded true thickness, to 1e-3 samples. Applies to the
/// fold part only (fold 0, the unfolded dome and tilt, is vertical).
#[test]
fn parallel_thickness_on_real_stacks() {
    let cfg = FoldConfig {
        style: FoldStyle::Parallel,
        cylindrical: 2,
        domes: 2,
        ..FoldConfig::default()
    };
    let shape = [24, 20, 96];
    let mut obliques = 0;
    for seed in 1..=4u64 {
        let flat = layered_stack_unfolded(seed, shape);
        let tp = folds::taper(&flat, &cfg);
        let set = FoldSet::draw(seed, shape, &cfg, &tp);
        obliques += set
            .folds
            .iter()
            .filter(|f| matches!(f, Fold::Cylindrical { theta, .. } if (theta % std::f64::consts::FRAC_PI_2).abs() > 0.1))
            .count();
        let sh = folds::shifts(&set, &flat, shape[0], shape[1]);
        let nh = flat.len();
        let t_of = |s: usize| tp[0].0 - tp[s].0;
        let pairs = [(nh / 4, nh / 4 + 1), (nh / 2, nh / 2 + 3), (nh - 3, nh - 2)];
        for &(h0, h1) in &pairs {
            let (t0, t1) = (t_of(h0), t_of(h1));
            for col in (0..shape[0] * shape[1]).step_by(37) {
                let (i, j) = (col / shape[1], col % shape[1]);
                let (x, y) = (i as f64 * cfg.bin_m, j as f64 * cfg.bin_m);
                let q = [x, y, sh[h0][col] * cfg.dz_m - t0];
                let on = dist_to_horizon(&set, q, t0);
                assert!(on / cfg.dz_m < 1e-3, "seed {seed} h {h0}: off its surface {on}");
                let d = dist_to_horizon(&set, q, t1);
                assert!(
                    ((d - (t1 - t0)) / cfg.dz_m).abs() < 1e-3,
                    "seed {seed} h {h0}->{h1} col {col}: {d} vs {}",
                    t1 - t0
                );
            }
        }
    }
    assert!(obliques >= 4, "oblique folds exercised");
}

/// Largest |slope| of the added folds, samples per column, from the field
/// gradient directly (similar: every folded horizon at its burial and
/// taper; parallel: the reference surface).
fn grid_slope(s: &FoldSet, shape: [usize; 3], tp: &[(f64, f64)]) -> f64 {
    let levels: Vec<(f64, f64)> = match s.cfg.style {
        FoldStyle::Similar => tp[..tp.len() - 1].to_vec(),
        FoldStyle::Parallel => vec![(0.0, 1.0)],
    };
    let mut m = 0.0f64;
    for (d, t) in levels {
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let (_, fx, fy) = s.field(i as f64 * s.cfg.bin_m, j as f64 * s.cfg.bin_m, d);
                m = m.max(t * fx.hypot(fy));
            }
        }
    }
    m * s.cfg.bin_m / s.cfg.dz_m
}

/// Test 4: the measured slope on the grid is within the limit after
/// scaling, and a fold above the limit is scaled to it.
#[test]
fn limits_hold() {
    for style in [FoldStyle::Similar, FoldStyle::Parallel] {
        let cfg = FoldConfig {
            style,
            cylindrical: 4,
            domes: 6,
            ..FoldConfig::default()
        };
        assert_eq!(cfg.max_dip, 2.0);
        for seed in 1..=12u64 {
            for shape in [[64, 64, 256], [16, 12, 64], [120, 40, 128]] {
                let tp = folds::taper(&layered_stack_unfolded(seed, shape), &cfg);
                let s = FoldSet::draw(seed, shape, &cfg, &tp);
                let g = grid_slope(&s, shape, &tp);
                assert!(g <= cfg.max_dip + 1e-9, "{style:?} seed {seed}: {g}");
                assert!((g - s.dip_after).abs() < 1e-9);
                if style == FoldStyle::Parallel {
                    assert!(s.curvature_bound() * tp[0].0 <= 0.8 + 1e-12);
                }
                for f in &s.folds {
                    if let Fold::Cylindrical { wavelength, .. } = f {
                        assert!(*wavelength >= 8.0 * cfg.bin_m);
                    }
                }
            }
        }
    }
    // A steep fold scaled to the limit: the measured slope lands on it.
    let cfg = FoldConfig {
        cylindrical: 1,
        domes: 0,
        max_dip: 0.2,
        ..FoldConfig::default()
    };
    let shape = [64, 64, 512];
    let tp = folds::taper(&layered_stack_unfolded(3, shape), &cfg);
    let s = FoldSet::draw(3, shape, &cfg, &tp);
    assert_eq!(s.scaled, 1);
    assert!(s.factor < 1.0);
    let g = grid_slope(&s, shape, &tp);
    assert!(g <= 0.2 + 1e-12 && g > 0.2 - 1e-9, "{g}");
}

/// Test 7 (part) and 8: horizons stay ordered and below the seabed, and
/// `round(continuous) == production` on folded maps, seeds 1–20.
#[test]
fn folded_maps_ordered_and_consistent() {
    for style in [FoldStyle::Similar, FoldStyle::Parallel] {
        let cfg = FoldConfig {
            style,
            ..FoldConfig::default()
        };
        for seed in 1..=20u64 {
            let shape = [32, 28, 128];
            let (p, nh) = layered_horizon_maps_with(seed, shape, Some(&cfg), None);
            let (c, nh2) = layered_horizon_maps_continuous_with(seed, shape, Some(&cfg), None);
            assert_eq!(nh, nh2);
            let (_, nh0) = layered_horizon_maps_with(seed, shape, None, None);
            assert_eq!(nh, nh0, "layer count unchanged");
            for col in 0..shape[0] * shape[1] {
                for h in 0..nh {
                    assert_eq!(c[col * nh + h].round(), p[col * nh + h]);
                    if h > 0 {
                        assert!(p[col * nh + h] >= p[col * nh + h - 1]);
                        assert!(c[col * nh + h] >= c[col * nh]);
                    }
                }
                assert!(p[col * nh + nh - 1] >= shape[2] as f64 + 1.0);
            }
            // seabed unchanged
            let (l, _) = layered_horizon_maps_continuous_with(seed, shape, None, None);
            for col in 0..shape[0] * shape[1] {
                assert_eq!(c[col * nh], l[col * nh]);
            }
            assert_ne!(p, layered_horizon_maps_with(seed, shape, None, None).0);
        }
    }
}

/// Test 7 (fold sanity on the default), seed-4 demo 64×64×256 with 4
/// faults: no unfilled (255) sample below the seabed, every column has
/// sediment, and the closure census is pinned (regression guard; seed 4
/// keeps 5 closures folded and at 7e80f2af).
#[test]
fn seed4_demo_sanity() {
    use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
    use synthoseis_core::{closure_census, generate_labels};
    for (legacy, kept) in [(true, 5usize), (false, 5usize)] {
        let mut c = E2eConfig {
            seed: 4,
            inline_count: 64,
            crossline_count: 64,
            samples: 256,
            faults: FaultConfig::legacy_with_count(4),
            ..E2eConfig::default()
        };
        c.rock_physics.legacy_fold_geometry = legacy;
        let (labels, sh) = generate_labels(&c);
        for col in labels.chunks(sh[2]) {
            let first = col
                .iter()
                .position(|&l| l != 255)
                .expect("sediment in every column");
            assert!(
                col[first..].iter().all(|&l| l != 255),
                "unfilled sample below the seabed"
            );
        }
        assert_eq!(
            closure_census(&c, &labels, sh).unwrap().kept(),
            kept,
            "legacy={legacy}"
        );
    }
}

/// Test 7 (spec §8): over seeds 1–20 (64×64×256, 4 faults) the default
/// folds keep at least the 7e80f2af total of 76 closures (83 measured).
#[test]
fn closures_seeds_1_to_20() {
    use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
    use synthoseis_core::{closure_census, generate_labels};
    let kept = |legacy_unconformity: bool| {
        let mut total = 0;
        for seed in 1..=20u64 {
            let mut c = E2eConfig {
                seed,
                inline_count: 64,
                crossline_count: 64,
                samples: 256,
                faults: FaultConfig::legacy_with_count(4),
                ..E2eConfig::default()
            };
            c.rock_physics.legacy_unconformity = legacy_unconformity;
            let (labels, sh) = generate_labels(&c);
            total += closure_census(&c, &labels, sh).unwrap().kept();
        }
        total
    };
    // #56 head (`--legacy-unconformity`): 83. With the bevel: 78.
    let legacy = kept(true);
    assert!(legacy >= 76, "kept closures over seeds 1-20: {legacy}");
    assert_eq!(legacy, 83);
    let default = kept(false);
    assert!(default >= 76, "kept closures over seeds 1-20 with unconformity: {default}");
    assert_eq!(default, 78);
}

/// Build-once horizon maps: a run with the cached geometry is bit-identical
/// to one without, and a configuration changed after caching recomputes.
#[test]
fn geometry_cache_is_transparent() {
    use synthoseis_core::generate_labels;
    use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
    let mut c = E2eConfig {
        seed: 5,
        inline_count: 24,
        crossline_count: 20,
        samples: 96,
        faults: FaultConfig::legacy_with_count(2),
        ..E2eConfig::default()
    };
    c.rock_physics.salt = true;
    let cached = c.with_geometry_cache();
    assert_eq!(generate_labels(&c), generate_labels(&cached));
    let mut changed = cached.clone();
    changed.seed = 6;
    let mut plain = c.clone();
    plain.seed = 6;
    assert_eq!(generate_labels(&changed), generate_labels(&plain));
    changed.rock_physics.fold.style = FoldStyle::Parallel;
    plain.rock_physics.fold.style = FoldStyle::Parallel;
    assert_eq!(generate_labels(&changed), generate_labels(&plain));
}
