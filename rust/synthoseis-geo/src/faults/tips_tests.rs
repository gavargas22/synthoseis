//! Spec-fault-kinematics 7b tests (§9 rows 6–8). Rows 9–10 (tiles, halo)
//! run with tips on in `kinematics_tests.rs`.
use super::*;

const SHAPE: [usize; 3] = [64, 64, 96];

/// γ pinned at the top of the spec range: the shortest faults, so the tip
/// line lies inside the test cube.
fn short(taper: TipTaper) -> TipParams {
    TipParams {
        gamma_min: 0.1,
        gamma_max: 0.1,
        taper,
        ..TipParams::default()
    }
}

/// The first `n` single-fault heave models with tips (seeds 1, 2, …).
fn tipped(n: usize, tp: &TipParams) -> Vec<FaultModel> {
    (1..400u64)
        .filter_map(|s| {
            let params = sample_random_faults(
                SHAPE,
                &RandomFaultConfig {
                    count: 1,
                    ..RandomFaultConfig::default()
                },
                s,
            );
            let m = FaultModel::resolve(SHAPE, &params, &Seabed::Flat(4.0), s)
                .with_kinematics(FaultKinematics::Heave(HeaveParams::default()))
                .with_tips(s, tp);
            (m.faults().len() == 1).then_some(m)
        })
        .take(n)
        .collect()
}

/// §9 row 6: the output→source map keeps a positive Jacobian (no fold-over)
/// across the hanging wall, tips included. Central differences of the
/// gathered source on voxels whose whole stencil is in the hanging wall
/// with slip and unclamped sources. Kills (checked): `√(1 − r²)` taper
/// (infinite tip gradient).
#[test]
fn tip_taper_keeps_a_positive_jacobian() {
    assert!(jacobian_violations(TipTaper::Cos2).0 == 0);
}

fn jacobian_violations(taper: TipTaper) -> (usize, usize) {
    let [ni, nj, nk] = SHAPE;
    let top = (nk - 1) as f64;
    let p = HeaveParams::default();
    let (mut n, mut bad) = (0usize, 0usize);
    for m in tipped(6, &short(taper)) {
        let f = &m.faults()[0];
        let src: Vec<[f64; 3]> = (0..ni * nj * nk)
            .map(|q| m.gather(q / (nj * nk), (q / nk) % nj, q % nk).src)
            .collect();
        let at = |i: usize, j: usize, k: usize| src[(i * nj + j) * nk + k];
        let ok = |i: usize, j: usize, k: usize| {
            let (_, _, d) = stage_slip(f, &p, i as f64, j as f64, k, 0.0, top);
            let s = at(i, j, k);
            d > 0.0
                && s[0] > 0.0
                && s[1] > 0.0
                && s[2] > 0.0
                && s[0] < (ni - 1) as f64
                && s[1] < (nj - 1) as f64
                && s[2] < top
        };
        for i in 1..ni - 1 {
            for j in 1..nj - 1 {
                for k in 1..nk - 1 {
                    let st = [
                        (i, j, k),
                        (i - 1, j, k),
                        (i + 1, j, k),
                        (i, j - 1, k),
                        (i, j + 1, k),
                        (i, j, k - 1),
                        (i, j, k + 1),
                    ];
                    if !st.iter().all(|&(a, b, c)| ok(a, b, c)) {
                        continue;
                    }
                    let col = |a: [f64; 3], b: [f64; 3]| {
                        [
                            (b[0] - a[0]) / 2.0,
                            (b[1] - a[1]) / 2.0,
                            (b[2] - a[2]) / 2.0,
                        ]
                    };
                    let c0 = col(at(i - 1, j, k), at(i + 1, j, k));
                    let c1 = col(at(i, j - 1, k), at(i, j + 1, k));
                    let c2 = col(at(i, j, k - 1), at(i, j, k + 1));
                    let det = c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
                        - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
                    n += 1;
                    bad += !(det > 0.0) as usize;
                }
            }
        }
    }
    assert!(n > 10_000, "too few hanging-wall voxels checked: {n}");
    (bad, n)
}

#[test]
fn row6_mutant_sqrt_taper_folds_over() {
    let (bad, n) = jacobian_violations(TipTaper::SqrtMutant);
    assert!(bad > 0, "sqrt taper went undetected over {n} voxels");
}

/// §9 row 7: zero throw on and beyond the tip line, and no fault label at
/// any voxel whose throw is ≤ 1 sample. Also checks the label does reach
/// the tip region (the tip line cuts the label inside the cube). Kills
/// (checked): taper without the `r < 1` cutoff.
#[test]
fn throw_and_label_stop_at_the_tip_line() {
    let (beyond, label_out, cut) = tip_violations(TipTaper::Cos2);
    assert_eq!(beyond, 0, "nonzero throw beyond the tip line");
    assert_eq!(label_out, 0, "label where throw <= 1 sample");
    assert!(cut > 0, "the tip line never cut the surface in the cube");
}

fn tip_violations(taper: TipTaper) -> (usize, usize, usize) {
    let [ni, nj, nk] = SHAPE;
    let top = (nk - 1) as f64;
    let p = HeaveParams::default();
    let (mut beyond, mut label_out, mut cut) = (0, 0, 0);
    for m in tipped(6, &short(taper)) {
        let f = &m.faults()[0];
        let t = f.tip.as_ref().unwrap();
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let (_, _, d) = stage_slip(f, &p, x, y, k, 0.0, top);
                    let r = t.radius(x, y, z);
                    let hw = f.geometry.value_at(x, y, z) <= 1.0;
                    if r >= 1.0 && d != 0.0 {
                        beyond += 1;
                    }
                    if r >= 1.0 && hw && (f.geometry.value_at(x, y, z) - 1.0).abs() < 0.05 {
                        cut += 1;
                    }
                    // Throw of the surface through this voxel (the label
                    // band straddles the surface, footwall side included).
                    let throw = t.factor(x, y, z) * f.profile[k];
                    if m.gather(i, j, k).id != 0 && !(throw > 1.0) {
                        label_out += 1;
                    }
                }
            }
        }
    }
    (beyond, label_out, cut)
}

#[test]
fn row7_mutant_uncut_taper_leaks_past_the_tip() {
    let (beyond, label_out, _) = tip_violations(TipTaper::UncutMutant);
    assert!(beyond > 0 && label_out + beyond > 0);
}

/// §9 row 8: D_max ∝ Lⁿ with n = 1.00 ± 0.02 over 200 drawn faults. D_max
/// is `T·dz / sin δ_c` of the fault and L is measured from the tip line
/// (where the taper first reaches 0 along strike, by bisection), so the
/// slope of log(D_max/γ) on log L checks the implemented scaling. (log D on
/// log L alone has slope ≈ 0.4 here: the γ scatter over a decade in γ
/// dominates the narrow throw range — see docs/faults-port.md.) Kills
/// (checked): n = 2.
#[test]
fn max_displacement_scales_linearly_with_length() {
    let n = dl_slope(&TipParams::default());
    assert!((n - 1.0).abs() <= 0.02, "D-L exponent {n}");
}

fn dl_slope(tp: &TipParams) -> f64 {
    let p = HeaveParams::default();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for s in 1..400u64 {
        let params = sample_random_faults(
            SHAPE,
            &RandomFaultConfig {
                count: 4,
                ..RandomFaultConfig::default()
            },
            s,
        );
        let m = FaultModel::resolve(SHAPE, &params, &Seabed::Flat(4.0), s)
            .with_kinematics(FaultKinematics::Heave(p))
            .with_tips(s, tp);
        for f in m.faults() {
            let t = f.tip.as_ref().unwrap();
            assert!((tp.gamma_min..=tp.gamma_max).contains(&t.gamma));
            let d_max = f.params.throw * p.dz_m / t.dip_c_deg.to_radians().sin();
            // Half-length along strike: bisect the first zero of the taper.
            let at = |u: f64| {
                let v = u / p.bin_m;
                t.factor(t.c[0] + v * t.s_hat[0], t.c[1] + v * t.s_hat[1], t.c[2])
            };
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            while at(hi) > 0.0 {
                hi *= 2.0;
            }
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if at(mid) > 0.0 {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            xs.push((2.0 * hi).ln());
            ys.push((d_max / t.gamma).ln());
        }
        if xs.len() >= 200 {
            break;
        }
    }
    assert!(xs.len() >= 200);
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    sxy / sxx
}

#[test]
fn row8_mutant_quadratic_scaling_is_caught() {
    let n = dl_slope(&TipParams {
        dl_exponent: 2.0,
        ..TipParams::default()
    });
    assert!((n - 1.0).abs() > 0.02, "n = 2 mutant gave slope {n}");
}

/// γ comes from its own sub-stream: the legacy draws and the resolved
/// faults are untouched, and the tip is centred on the fault centre with
/// unit throw weight there.
#[test]
fn tips_leave_the_fault_draws_alone() {
    let params = sample_random_faults(SHAPE, &RandomFaultConfig::default(), 5);
    let m0 = FaultModel::resolve(SHAPE, &params, &Seabed::Flat(4.0), 5)
        .with_kinematics(FaultKinematics::Heave(HeaveParams::default()));
    let m1 = m0.clone().with_tips(5, &TipParams::default());
    assert_eq!(m0.faults().len(), m1.faults().len());
    for (a, b) in m0.faults().iter().zip(m1.faults()) {
        assert_eq!(a.params, b.params);
        assert_eq!(a.profile, b.profile);
        let t = b.tip.as_ref().unwrap();
        assert_eq!(t.factor(t.c[0], t.c[1], t.c[2]), 1.0);
    }
}
