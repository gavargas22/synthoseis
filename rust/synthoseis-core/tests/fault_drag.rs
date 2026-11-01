//! Optional fault drag (roadmap 7c, spec-fault-drag §4): pipeline bars
//! T4–T7, T9 and T13 on drag on vs drag off. References are computed here
//! from the drag-off run and the fault planes; nothing below calls the drag
//! code itself (only the pipeline entry points and the resolved `W_m`).
mod salt_order_metrics;
use salt_order_metrics as m;
use salt_order_metrics::Mode;
use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::{closure_census, generate_labels, DragParams, DragSense};

const SHAPE: [usize; 3] = [32, 32, 128];
const SALT_SHAPE: [usize; 3] = [128, 128, 256];

fn with_drag(c: &E2eConfig, sense: Option<DragSense>) -> E2eConfig {
    let mut c = c.clone();
    c.faults.drag = sense.map(|sense| DragParams {
        sense,
        ..DragParams::default()
    });
    c.with_geometry_cache()
}

fn base(seed: u64, shape: [usize; 3]) -> E2eConfig {
    m::cfg(seed, shape, 4, Mode::PostFault)
}

/// Columns (salt and unset excluded) with a deeper label above a shallower
/// one by more than 1 sample: a run of `≥ 2` cells of label `a` above a
/// cell of label `b < a` (single-cell rounding flips are the tolerance).
fn inversions(labels: &[u8], shape: [usize; 3], salt: Option<&synthoseis_core::salt::SaltBody>) -> (usize, usize) {
    let [ni, nj, nk] = shape;
    let (mut cols, mut tol) = (0, 0);
    for q in 0..ni * nj {
        let v: Vec<u8> = (0..nk)
            .filter(|&k| labels[q * nk + k] != 255 && !salt.is_some_and(|s| s.contains(q, k)))
            .map(|k| labels[q * nk + k])
            .collect();
        let (mut hard, mut soft) = (false, false);
        let mut run = 1;
        for w in 1..v.len() {
            if v[w] == v[w - 1] {
                run += 1;
                continue;
            }
            if v[w] < v[w - 1] {
                if run >= 2 { hard = true } else { soft = true }
            }
            run = 1;
        }
        cols += hard as usize;
        tol += (soft && !hard) as usize;
    }
    (cols, tol)
}

/// T4: horizon order. Faulted labels repeat strata across reverse faults,
/// so the bar is on drag-made inversions: columns inverted with drag on
/// that are not inverted with drag off.
fn t4(seed: u64, sense: DragSense) -> (usize, usize, usize) {
    let c = base(seed, SHAPE);
    let off = generate_labels(&c).0;
    let on = generate_labels(&with_drag(&c, Some(sense))).0;
    let (a, _) = inversions(&off, SHAPE, None);
    let (b, tol) = inversions(&on, SHAPE, None);
    (a, b, tol)
}

#[test]
#[ignore = "slow: 20 seeds x 3 senses; FAILS at the branch head (drag-made inversions on seeds 4, 7, 15, 18 and more under reverse), reported on the PR"]
fn t4_horizon_order_seeds_1_20() {
    let mut bad = Vec::new();
    for sense in [DragSense::Normal, DragSense::Reverse, DragSense::Auto] {
        for seed in 1u64..=20 {
            let (off, on, tol) = t4(seed, sense);
            eprintln!("T4 {sense:?} seed {seed}: inverted columns off {off} on {on} (tolerance used {tol})");
            if on > off {
                bad.push((sense, seed, off, on));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// T5: exposed hidden-lifted strata (the 7d bar) stay 0 with drag on.
fn t5(seed: u64) -> Option<usize> {
    let c = with_drag(&base(seed, SALT_SHAPE), Some(DragSense::Auto));
    let s = m::salt(&c)?;
    let r = m::run(&c);
    Some(m::exposed(&r, &s, &m::undragged(seed, SALT_SHAPE)))
}

#[test]
#[ignore = "slow: 128x128x256, seeds 1-20"]
fn t5_exposed_is_zero() {
    let mut bad = Vec::new();
    for seed in 1u64..=20 {
        let Some(e) = t5(seed) else {
            eprintln!("T5 seed {seed}: no salt");
            continue;
        };
        eprintln!("T5 seed {seed}: exposed {e}");
        if e > 0 {
            bad.push((seed, e));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// T6: kept closure volume with drag on against the #61 floors (7a
/// 98.14 %, 7b 97.73 %), measured as in `closure_minimum.rs` on the cube
/// and seeds the floors come from (32×32×128, 3 faults, seeds 1–12); seeds
/// 1–20 are printed too (drag off and on).
#[test]
#[ignore = "slow: 20 seeds x 2 profiles"]
fn t6_closure_floors() {
    let mut miss = Vec::new();
    for (legacy, floor) in [(true, 0.9814), (false, 0.9773)] {
        for (seeds, drag) in [(12u64, false), (12, true), (20, false), (20, true)] {
            let (mut kept, mut total) = (0usize, 0usize);
            for seed in 1u64..=seeds {
                let mut c = synthoseis_core::pipeline::E2eConfig {
                    seed,
                    inline_count: 32,
                    crossline_count: 32,
                    samples: 128,
                    faults: synthoseis_core::pipeline::FaultConfig::with_count(3),
                    ..E2eConfig::default()
                };
                c.faults.legacy_profile = legacy;
                if drag {
                    c.faults.drag = Some(DragParams::default());
                }
                let (labels, sh) = generate_labels(&c);
                let k = closure_census(&c, &labels, sh).unwrap();
                kept += k.sizes.iter().filter(|&&v| v >= k.minimum).sum::<usize>();
                total += k.sizes.iter().sum::<usize>();
            }
            let f = kept as f64 / total.max(1) as f64;
            eprintln!("T6 seeds 1-{seeds} legacy_profile={legacy} drag={drag}: kept {:.2} % (floor {:.2} %)", 100.0 * f, 100.0 * floor);
            if drag && seeds == 12 && f < floor {
                miss.push((legacy, f));
            }
        }
    }
    assert!(miss.is_empty(), "{miss:?}");
}

/// Local vertical stretch of the fault gather: output span over source
/// span of the gather lookup on a ±4-sample window, skipping windows that
/// touch a fault plane (the throw jump). Fraction above 2×.
fn stretch_over2(r: &m::Run) -> f64 {
    let [ni, nj, nk] = r.shape;
    let Some((_, lk)) = &r.src else { return 0.0 };
    let (mut n, mut over) = (0usize, 0usize);
    for q in 0..ni * nj {
        for k in 0..nk {
            let (lo, hi) = (k.saturating_sub(4), (k + 4).min(nk - 1));
            if (lo..=hi).any(|x| r.fault_id[q * nk + x] != 0) {
                continue;
            }
            n += 1;
            let span = (lk[q * nk + hi] - lk[q * nk + lo]) as f64;
            over += (span <= 0.0 || (hi - lo) as f64 / span > 2.0) as usize;
        }
    }
    over as f64 / n.max(1) as f64
}

#[test]
#[ignore = "slow: 20 seeds; FAILS on seeds 3, 4, 7, 10, 18 (seed 3 0.07 % -> 4.96 %); every stretched window is inside a drag zone, reported on the PR"]
fn t7_stretch_seeds_1_20() {
    let mut bad = Vec::new();
    for seed in 1u64..=20 {
        let c = base(seed, SHAPE);
        let off = stretch_over2(&m::run(&c));
        let on = stretch_over2(&m::run(&with_drag(&c, Some(DragSense::Auto))));
        eprintln!("T7 seed {seed}: stretch>2x off {:.4} % on {:.4} %", 100.0 * off, 100.0 * on);
        if on > off + 0.001 {
            bad.push((seed, off, on));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// T9: drag-on labels against a test-side resample of the drag-off,
/// unfaulted label volume through the drag-on gather source (nearest
/// sample, as the gather picks labels). Jaccard over non-fault voxels.
fn t9(seed: u64) -> (usize, usize, usize, usize) {
    t9_with(seed, Some(DragSense::Auto))
}

fn t9_with(seed: u64, sense: Option<DragSense>) -> (usize, usize, usize, usize) {
    let mut c = base(seed, SHAPE);
    c.rock_physics.salt = false;
    let c = with_drag(&c, sense);
    let r = m::run(&c);
    let und = m::undragged(seed, SHAPE);
    let [ni, nj, nk] = SHAPE;
    let Some((ij, lk)) = &r.src else { return (0, 0, 0, 0) };
    let (mut same, mut n, mut pad_n, mut pad_bad) = (0, 0, 0usize, 0usize);
    for v in 0..ni * nj * nk {
        if r.fault_id[v] != 0 {
            continue;
        }
        let near = |x: f32, n: usize| (x.round().max(0.0) as usize).min(n - 1);
        let w = (near(ij[v].0, ni) * nj + near(ij[v].1, nj)) * nk + near(lk[v], nk);
        if lk[v] >= nk as f32 - 0.5 {
            // Source in the bottom pad, which the test volume lacks: the
            // label must be set and at or below the column's bottom label.
            let b = und[w];
            pad_n += 1;
            pad_bad += (r.labels[v] == 255 || r.labels[v] < b) as usize;
            continue;
        }
        n += 1;
        let ok = r.labels[v] == und[w];
        same += ok as usize;
        if !ok && std::env::var("T9_DIAG").is_ok() {
            let k = v % nk;
            let col = v - k;
            let dk = (0..nk).filter(|&x| r.fault_id[col + x] != 0).map(|x| x.abs_diff(k)).min();
            eprintln!("T9 miss seed {seed} v {v} k {k} lab {} ref {} lk {:.3} ij {:?} nearest-plane dk {dk:?}", r.labels[v], und[w], lk[v], ij[v]);
        }
    }
    (same, n, pad_n, pad_bad)
}

#[test]
#[ignore = "diagnostic"]
fn t9_diag() {
    for seed in [3u64, 7] {
        eprintln!("T9 diag seed {seed}: drag off {:?} on {:?}", t9_with(seed, None), t9_with(seed, Some(DragSense::Auto)));
    }
}

#[test]
#[ignore = "slow: 20 seeds"]
fn t9_label_resample_parity_seeds_1_20() {
    let (mut s, mut n, mut pb) = (0, 0, 0);
    for seed in 1u64..=20 {
        let (a, b, pn, pbad) = t9(seed);
        eprintln!("T9 seed {seed}: label agreement {a}/{b} = {:.6}; pad-source voxels {pn}, unset or above the bottom label {pbad}", a as f64 / b.max(1) as f64);
        s += a;
        n += b;
        pb += pbad;
    }
    eprintln!("T9 all: {:.6}, pad misses {pb}", s as f64 / n.max(1) as f64);
    assert_eq!((s, pb), (n, 0));
}

/// T13 mask, from the spec geometry (spec-fault-drag §1), not from the
/// production labels: a voxel is in fault `n`'s zone when its metric
/// distance `rho = |f - 1| / |grad_m f|` to the ellipsoid surface is below
/// `W_n` and the local throw `D_loc` (tip or lateral weight times the
/// vertical profile) is non-zero, sub-sample tips included. Far = in no
/// zone.
fn far_mask(fm: &synthoseis_geo::faults::FaultModel, shape: [usize; 3], w_m: &[f64], bin_m: f64, dz_m: f64) -> Vec<bool> {
    let [ni, nj, nk] = shape;
    (0..ni * nj * nk)
        .map(|v| far_at(fm, w_m, bin_m, dz_m, ((v / nk) / nj) as f64, ((v / nk) % nj) as f64, v % nk))
        .collect()
}

fn far_at(fm: &synthoseis_geo::faults::FaultModel, w_m: &[f64], bin_m: f64, dz_m: f64, x: f64, y: f64, k: usize) -> bool {
    {
        {
            let z = k as f64;
            fm.faults().iter().zip(w_m).all(|(f, &w)| {
                let wt = match &f.tip {
                    Some(t) => t.factor(x, y, z),
                    None => f.lateral_at(x, y),
                };
                if wt * f.profile[k] == 0.0 {
                    return true;
                }
                let fv = f.geometry.value_at(x, y, z);
                let g = f.geometry.gradient_at(x, y, z);
                let gn = ((g[0] / bin_m).powi(2) + (g[1] / bin_m).powi(2) + (g[2] / dz_m).powi(2)).sqrt();
                !(gn > 0.0) || (fv - 1.0).abs() / gn >= w
            })
        }
    }
}

/// T13 needs a cube wider than the zones (W up to 400 m).
const T13_SHAPE: [usize; 3] = [96, 96, 128];

fn t13(seed: u64) -> (usize, usize) {
    let c = base(seed, T13_SHAPE);
    let off = m::run(&c);
    let cd = with_drag(&c, Some(DragSense::Auto));
    let on = m::run(&cd);
    let fm = synthoseis_core::fault_model(&cd).expect("faults");
    let dm = fm.drag().expect("drag");
    let w: Vec<f64> = dm.faults.iter().map(|f| f.w_m).collect();
    // Stages evaluate the zones at the stage position, so a voxel is far
    // only when it is outside every zone both at its output position and
    // at its drag-off gather source.
    let (ij, lk) = off.src.as_ref().expect("faults");
    let nk = off.shape[2];
    let mask: Vec<bool> = far_mask(&fm, off.shape, &w, dm.bin_m, dm.dz_m)
        .into_iter()
        .enumerate()
        .map(|(v, far)| far && far_at(&fm, &w, dm.bin_m, dm.dz_m, ij[v].0 as f64, ij[v].1 as f64, (lk[v].round().max(0.0) as usize).min(nk - 1)))
        .collect();
    let n = mask.iter().filter(|&&x| x).count();
    let diff = (0..mask.len()).filter(|&v| mask[v] && off.labels[v] != on.labels[v]).count();
    (diff, n)
}

#[test]
#[ignore = "slow: 20 seeds"]
fn t13_identical_outside_the_zones_seeds_1_20() {
    let mut bad = Vec::new();
    for seed in 1u64..=20 {
        let (d, n) = t13(seed);
        eprintln!("T13 seed {seed}: {d} of {n} far voxels differ");
        if d > 0 {
            bad.push((seed, d));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn t13_identical_outside_the_zones() {
    let (d, n) = t13(1);
    assert!(n > 0);
    assert_eq!(d, 0);
}
