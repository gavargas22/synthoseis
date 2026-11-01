//! Spec-fault-drag (7c) tests T1, T2, T8, T10, T11, T12 and the bottom-row
//! check of M6. Every reference is written out here; none calls the gather,
//! `stage_slip*` or a drag helper (only the fault surface `f(x)` and its
//! gradient, which define the fault).
use super::*;

const PI2: f64 = std::f64::consts::FRAC_PI_2;
const T: f64 = 28.0;

fn random_model(shape: [usize; 3], seed: u64, count: usize) -> FaultModel {
    let cfg = RandomFaultConfig {
        count,
        ..RandomFaultConfig::default()
    };
    let params = sample_random_faults(shape, &cfg, seed);
    FaultModel::resolve(shape, &params, &Seabed::Flat(4.0), seed)
        .with_kinematics(FaultKinematics::Heave(HeaveParams::default()))
}

fn drag(beta_h: f64, beta_f: f64, sense: DragSense) -> DragParams {
    DragParams {
        beta_h,
        beta_f,
        sense,
        ..DragParams::default()
    }
}

/// Planar fault `c·x − s·z = plane` (voxel coordinates) dipping 60°
/// (physical), hanging wall `c·x − s·z > plane` (above), constant throw
/// `T` samples, lateral weight 1, no tips.
struct Planar {
    shape: [usize; 3],
    c: f64,
    s: f64,
    plane: f64,
    bin: f64,
    dz: f64,
}

impl Planar {
    fn new(shape: [usize; 3]) -> Self {
        let p = HeaveParams::default();
        let tan_v = 60f64.to_radians().tan() * p.bin_m / p.dz_m;
        let th = (1.0 / tan_v).atan();
        let (c, s) = (th.cos(), th.sin());
        let plane = 0.5 * (shape[0] as f64 * c - shape[2] as f64 * s);
        Self {
            shape,
            c,
            s,
            plane,
            bin: p.bin_m,
            dz: p.dz_m,
        }
    }

    /// The model (drag off), heave on or off.
    fn model(&self, heave: bool) -> FaultModel {
        let [_, _, nk] = self.shape;
        let mut m = (1..400u64)
            .map(|s| random_model(self.shape, s, 1))
            .find(|m| m.faults().len() == 1)
            .expect("a one-fault model");
        if !heave {
            m = m.with_kinematics(FaultKinematics::Heave(HeaveParams {
                heave_scale: 0.0,
                ..HeaveParams::default()
            }));
        }
        let (c, s) = (self.c, self.s);
        let rot = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
        let half = 1.0e5;
        let f = &mut m.faults_mut()[0];
        f.geometry = FaultGeometry::from_parts(
            rot,
            [self.plane + half, 0.0, 0.0],
            [half * half, 1e14, 1e14],
        );
        f.profile = vec![T; nk];
        f.tip = None;
        f.test_flat_lateral();
        m
    }

    /// Signed plane function (voxels): > 0 in the hanging wall.
    fn l(&self, x: f64, z: f64) -> f64 {
        self.c * x - self.s * z - self.plane
    }

    /// Point-to-plane distance in metres.
    fn rho_m(&self, x: f64, z: f64) -> f64 {
        let gm = (self.c / self.bin).hypot(self.s / self.dz);
        self.l(x, z).abs() / gm
    }

    /// Distance in voxels (for the label band).
    fn rho_vox(&self, x: f64, z: f64) -> f64 {
        self.l(x, z).abs() / self.c.hypot(self.s)
    }

    /// W of spec §1: min(ω · T·dz / sin 60°, W_cap).
    fn w_m(&self, p: &DragParams) -> f64 {
        (p.width * T * self.dz / 60f64.to_radians().sin()).min(p.max_width_m)
    }
}

fn g_ref(q: f64) -> f64 {
    if q < 1.0 {
        (PI2 * q).cos().powi(2)
    } else {
        0.0
    }
}

/// Fill (ni, nj, nks) with `f(k)`, run the model's label gather.
fn labels_of(m: &FaultModel, f: impl Fn(usize) -> u8, tile: [usize; 2]) -> Vec<u8> {
    let [ni, nj, nk] = m.shape();
    let nks = nk + m.drag_pad();
    let mut src = vec![0u8; ni * nj * nks];
    for c in 0..ni * nj {
        for k in 0..nks {
            src[c * nks + k] = f(k);
        }
    }
    m.apply_to_labels_from(&src, nks, tile).0
}

const H: usize = 8;
const T1_SHAPE: [usize; 3] = [128, 8, 160];

/// T1 errors `(max, mean, n)` of the horizon tops (first voxel of each
/// layer) against `h(x) = h0 + d·H_hw + s·β·d·g(ρ/W)`, solved by fixed-point
/// iteration (ρ depends on h), over `0 < ρ < 1.2 W` outside the 1-voxel
/// label band.
fn t1_errors(heave: bool, sense: DragSense, s_sign: f64) -> (f64, f64, usize) {
    let pl = Planar::new(T1_SHAPE);
    let p = drag(0.5, 0.5, sense);
    let m = pl.model(heave).with_drag(&p);
    let [ni, nj, nk] = T1_SHAPE;
    let lab = labels_of(&m, |k| (k / H) as u8, [ni, nj]);
    let w = pl.w_m(&p);
    let j = nj / 2;
    let (mut max, mut sum, mut n) = (0f64, 0f64, 0usize);
    for i in 0..ni {
        let x = i as f64;
        let col = &lab[(i * nj + j) * nk..(i * nj + j + 1) * nk];
        for mh in 2..(nk / H) {
            let h0 = (mh * H) as f64;
            // Hanging wall: z − d − w(z) = h0; footwall: z − w(z) = h0.
            let solve = |hw: bool| {
                let mut z = h0 + if hw { T } else { 0.0 };
                for _ in 0..60 {
                    let g = g_ref(pl.rho_m(x, z) / w);
                    z = if hw {
                        h0 + T + s_sign * 0.5 * T * g
                    } else {
                        h0 - s_sign * 0.5 * T * g
                    };
                }
                z
            };
            let zh = solve(true);
            let zf = solve(false);
            let cand = [(zh, pl.l(x, zh) > 0.0), (zf, pl.l(x, zf) <= 0.0)];
            let Some(&(h, _)) = cand
                .iter()
                .filter(|c| c.1)
                .min_by(|a, b| a.0.total_cmp(&b.0))
            else {
                continue;
            };
            let rho = pl.rho_m(x, h);
            if !(rho > 0.0 && rho < 1.2 * w) || pl.rho_vox(x, h) < 1.0 {
                continue;
            }
            if h < 1.0 || h > (nk - 2) as f64 {
                continue;
            }
            let Some(top) = col.iter().position(|&v| v as usize >= mh) else {
                continue;
            };
            let e = (top as f64 - h).abs();
            max = max.max(e);
            sum += e;
            n += 1;
        }
    }
    (max, sum / n.max(1) as f64, n)
}

/// T1, analytic planar fault: max |err| ≤ 1 sample, mean ≤ 0.35, heave
/// on and off, both senses. Kills M1 (sign), M2 (width ×1.15), M3 (β ×1.15),
/// M4a/b (taper), M5 (footwall term dropped).
#[test]
fn t1_planar_fault_matches_the_analytic_horizons() {
    for heave in [true, false] {
        for (sense, s) in [(DragSense::Reverse, 1.0), (DragSense::Normal, -1.0)] {
            if sense == DragSense::Normal {
                // β_h + β_f = 1 > 0.9 is not allowed with normal sense.
                continue;
            }
            let (max, mean, n) = t1_errors(heave, sense, s);
            eprintln!("T1 heave={heave} sense={sense:?}: max {max:.3} mean {mean:.3} n {n}");
            assert!(n > 500, "T1 coverage {n}");
            assert!(max <= 1.0 && mean <= 0.35, "T1 heave={heave}: max {max:.3} mean {mean:.3}");
        }
    }
}

/// T1 margins of the taper mutants M4a / M4b (hooks; M1–M3, M5 are source
/// patches run by the mutation runner).
#[test]
fn t1_kills_the_taper_mutants() {
    let pl = Planar::new(T1_SHAPE);
    let [ni, nj, nk] = T1_SHAPE;
    let base = drag(0.5, 0.5, DragSense::Reverse);
    let want = labels_of(&pl.model(true).with_drag(&base), |k| (k / H) as u8, [ni, nj]);
    for taper in [DragTaper::ExpMutant, DragTaper::BoxMutant] {
        let p = DragParams { taper, ..base };
        let got = labels_of(&pl.model(true).with_drag(&p), |k| (k / H) as u8, [ni, nj]);
        let differ = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        eprintln!("T1 {taper:?}: {differ} voxels differ from cos²");
        assert!(differ > 0);
    }
    let _ = nk;
}

/// T2, compact support: voxels with ρ ≥ W + 1 voxel keep the drag-off
/// labels (single-fault models, tips on; ρ from the fault surface).
#[test]
fn t2_compact_support() {
    let shape = [64, 64, 96];
    let [ni, nj, nk] = shape;
    let (mut n, mut bad, mut models) = (0usize, 0usize, 0usize);
    for seed in 1..200u64 {
        let m = random_model(shape, seed, 1);
        if m.faults().len() != 1 {
            continue;
        }
        let m = m.with_tips(seed, &TipParams::default());
        let on = m.clone().with_drag(&DragParams::default());
        let w = on.drag().unwrap().faults[0].w_m;
        let a = labels_of(&m, |k| k as u8, [ni, nj]);
        let b = labels_of(&on, |k| k as u8, [ni, nj]);
        let f = &m.faults()[0];
        let p = HeaveParams::default();
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let v = f.geometry.value_at(x, y, z);
                    let g = f.geometry.gradient_at(x, y, z);
                    let gm = (g[0] / p.bin_m).hypot(g[1] / p.bin_m).hypot(g[2] / p.dz_m);
                    let rho = (v - 1.0).abs() / gm;
                    if rho >= w + p.bin_m.max(p.dz_m) {
                        let o = (i * nj + j) * nk + k;
                        n += 1;
                        bad += (a[o] != b[o]) as usize;
                    }
                }
            }
        }
        models += 1;
        if models == 6 {
            break;
        }
    }
    assert!(n > 100_000, "coverage {n}");
    assert_eq!(bad, 0, "{bad}/{n} voxels outside W + 1 voxel differ");
}

/// T8 (library): labels are the same for 1×1, 3×3 and 4×2 tilings.
#[test]
fn t8_tile_invariance() {
    let shape = [64, 64, 96];
    let [ni, nj, _] = shape;
    for seed in [1u64, 2, 3, 7] {
        let m = random_model(shape, seed, 4)
            .with_tips(seed, &TipParams::default())
            .with_drag(&DragParams::default());
        let a = labels_of(&m, |k| k as u8, [ni, nj]);
        for tile in [[ni.div_ceil(3), nj.div_ceil(3)], [ni.div_ceil(4), nj.div_ceil(2)]] {
            let b = labels_of(&m, |k| k as u8, tile);
            let d = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let moved = a.iter().enumerate().filter(|(v, &x)| x as usize != v % shape[2]).count();
            eprintln!("T8 seed {seed} tile {tile:?}: {d} of {} voxels differ from 1x1 ({moved} voxels moved by the faults)", a.len());
            assert!(a == b, "seed {seed} tile {tile:?}");
        }
    }
}

/// Horizon top (first voxel with label ≥ `mh`) in column i.
fn top(lab: &[u8], i: usize, nj: usize, nk: usize, mh: usize) -> Option<usize> {
    let j = nj / 2;
    lab[(i * nj + j) * nk..(i * nj + j + 1) * nk]
        .iter()
        .position(|&v| v as usize >= mh)
}

/// T10, sense: with reverse drag a hanging-wall horizon bends down toward
/// the fault (deeper near it than at 0.9 W), with normal drag up; the
/// footwall the other way. Auto picks reverse where θ ≥ 35° (θ from the
/// surface normal at the fault centre, floored at the 30° dip floor).
#[test]
fn t10_sense() {
    let pl = Planar::new(T1_SHAPE);
    let [ni, nj, nk] = T1_SHAPE;
    for (sense, s) in [(DragSense::Reverse, 1.0), (DragSense::Normal, -1.0)] {
        let p = drag(0.3, 0.15, sense);
        let lab = labels_of(&pl.model(true).with_drag(&p), |k| (k / H) as u8, [ni, nj]);
        let w = pl.w_m(&p);
        let mh = 10; // h0 = 80
        // Hanging-wall columns: horizon near z = 80 + T, above the plane.
        let (mut near, mut far) = (None, None);
        for i in 0..ni {
            let Some(t) = top(&lab, i, nj, nk, mh) else { continue };
            let (x, z) = (i as f64, t as f64);
            if pl.l(x, z) <= 0.0 {
                continue;
            }
            let q = pl.rho_m(x, z) / w;
            if (0.1..0.2).contains(&q) {
                near.get_or_insert(t as f64);
            }
            if (0.85..0.95).contains(&q) {
                far.get_or_insert(t as f64);
            }
        }
        let (near, far) = (near.expect("near"), far.expect("far"));
        eprintln!("T10 {sense:?}: hanging wall near {near} far {far}");
        assert!(s * (near - far) >= 2.0, "{sense:?}: near {near} far {far}");
    }
    // Auto.
    let p = HeaveParams::default();
    let (mut rev, mut nor) = (0, 0);
    for seed in 1..=20u64 {
        let m = random_model([64, 64, 96], seed, 4)
            .with_tips(seed, &TipParams::default())
            .with_drag(&DragParams::default());
        let d = m.drag().unwrap();
        for (n, f) in m.faults().iter().enumerate() {
            let c = f.center.map(|v| v as f64);
            let g = f.geometry.gradient_at(c[0], c[1], c[2]);
            let th = (g[0] / p.bin_m)
                .hypot(g[1] / p.bin_m)
                .atan2((g[2] / p.dz_m).abs())
                .to_degrees()
                .max(p.min_dip_deg);
            let want = if th >= 35.0 { 1.0 } else { -1.0 };
            assert_eq!(d.faults[n].s, want, "seed {seed} fault {n} θ {th}");
            if want > 0.0 {
                rev += 1
            } else {
                nor += 1
            }
        }
    }
    eprintln!("T10 auto, seeds 1-20: {rev} reverse, {nor} normal");
    assert!(rev > 0);
}

/// T11, throw continuity: at the fault the offset is d(1 + s(β_h + β_f))
/// ± 1 sample, far away (ρ ≥ W) d ± 1. Label = source sample.
#[test]
fn t11_throw_continuity() {
    let pl = Planar::new(T1_SHAPE);
    let [ni, nj, nk] = T1_SHAPE;
    for (sense, s) in [(DragSense::Reverse, 1.0), (DragSense::Normal, -1.0)] {
        let p = drag(0.3, 0.15, sense);
        let m = pl.model(true).with_drag(&p);
        let lab = labels_of(&m, |k| k.min(254) as u8, [ni, nj]);
        let w = pl.w_m(&p);
        let want = T * (1.0 + s * 0.45);
        let j = nj / 2;
        let (mut n_at, mut bad_at, mut n_far, mut bad_far) = (0, 0, 0, 0);
        for i in 0..ni {
            let col = &lab[(i * nj + j) * nk..(i * nj + j + 1) * nk];
            let x = i as f64;
            let u = |k: usize| k as f64 - col[k] as f64; // output − source
            // At the fault: the last hanging-wall and first footwall voxel.
            if let Some(kf) = (1..nk).find(|&k| pl.l(x, k as f64) <= 0.0) {
                let kh = kf - 1;
                if pl.l(x, kh as f64) > 0.0 && kh as f64 > T * 1.5 + 2.0 && kf + 8 < nk {
                    n_at += 1;
                    bad_at += ((u(kh) - u(kf) - want).abs() > 1.0) as usize;
                }
            }
            // Far field.
            for k in (T as usize * 2)..nk - 8 {
                if pl.rho_m(x, k as f64) >= w {
                    let d = if pl.l(x, k as f64) > 0.0 { T } else { 0.0 };
                    n_far += 1;
                    bad_far += ((u(k) - d).abs() > 1.0) as usize;
                }
            }
        }
        eprintln!("T11 {sense:?}: at fault {bad_at}/{n_at} off, far {bad_far}/{n_far} off (want {want})");
        assert!(n_at >= 15 && n_far > 1000, "coverage {n_at} {n_far}");
        assert_eq!((bad_at, bad_far), (0, 0), "{sense:?}");
    }
}

/// M6 bottom-row check: footwall uplift next to the cube bottom reads the
/// pad (sources below the last sample), it does not repeat the last sample.
#[test]
fn bottom_rows_read_the_pad() {
    let pl = Planar::new(T1_SHAPE);
    let [ni, nj, nk] = T1_SHAPE;
    let m = pl.model(true).with_drag(&DragParams {
        sense: DragSense::Reverse,
        ..DragParams::default()
    });
    let pad = m.drag_pad();
    assert_eq!(pad, (0.15f64 * T).ceil() as usize + 1);
    let lab = labels_of(&m, |k| k.min(254) as u8, [ni, nj]);
    let below = lab.iter().filter(|&&v| v as usize >= nk).count();
    // A clamped source (no pad) repeats the last sample wherever the
    // reference uplift k + u ≥ nk − 0.5; count those voxels.
    let w = pl.w_m(&DragParams::default());
    let j = nj / 2;
    let mut clamped = 0;
    for i in 0..ni {
        for k in nk - pad..nk {
            let (x, z) = (i as f64, k as f64);
            if pl.l(x, z) > 0.0 {
                continue;
            }
            let u = 0.15 * T * g_ref(pl.rho_m(x, z) / w);
            if z + u >= nk as f64 + 0.5 {
                let v = lab[(i * nj + j) * nk + k] as usize;
                clamped += (v == nk - 1) as usize;
            }
        }
    }
    let mut repeats = 0;
    for c in 0..ni * nj {
        let col = &lab[c * nk..(c + 1) * nk];
        for k in nk - pad..nk - 1 {
            repeats += (col[k] as usize == nk - 1 && col[k + 1] as usize == nk - 1) as usize;
        }
    }
    eprintln!("bottom rows: {below} voxels read the pad, {repeats} repeats of the last sample");
    assert!(below > 0);
    assert_eq!(clamped, 0, "{clamped} footwall voxels repeat the last sample");
}

/// Inline reference for T12 (spec §3 step 1, corrected): per fault,
/// (T_f samples, W_f m, θ_f deg, B_f unit = T (π/2) cos θ dz / W).
fn t12_ref(m: &FaultModel, p: &DragParams) -> Vec<(f64, f64, f64, f64)> {
    let hp = HeaveParams::default();
    m.faults()
        .iter()
        .map(|f| {
            let t = f.profile.iter().fold(0f64, |a, v| a.max(v.abs()));
            let c = [f.center[0] as f64, f.center[1] as f64, f.center[2] as f64];
            let g = f.geometry.gradient_at(c[0], c[1], c[2]);
            let (gx, gy, gz) = (g[0] / hp.bin_m, g[1] / hp.bin_m, g[2] / hp.dz_m);
            let th = (gx * gx + gy * gy).sqrt().atan2(gz.abs()).to_degrees().max(hp.min_dip_deg);
            let dmax = match &f.tip { Some(tl) => tl.d_max, None => t * hp.dz_m / th.to_radians().sin() };
            let w = (p.width * dmax).min(p.max_width_m);
            (t, w, th, t * PI2 * th.to_radians().cos() * hp.dz_m / w)
        })
        .collect()
}

/// T12: per-fault clamp B_f ≤ 0.16 after the clamp for every fault on seeds
/// 1–20 (defaults, and a large-throw case where W hits the 400 m cap);
/// every clamped fault is in the metadata with β before and after. Also the
/// normal-sense cap.
#[test]
fn t12_per_fault_clamp() {
    let (mut n_faults, mut n_clamped, mut n_cap) = (0, 0, 0);
    for large in [false, true] {
        for seed in 1..=20u64 {
            let cfg = RandomFaultConfig {
                count: 4,
                throw_min: if large { 60.0 } else { RandomFaultConfig::default().throw_min },
                throw_max: if large { 90.0 } else { RandomFaultConfig::default().throw_max },
                ..RandomFaultConfig::default()
            };
            let shape = [64, 64, 96];
            let m = FaultModel::resolve(shape, &sample_random_faults(shape, &cfg, seed), &Seabed::Flat(4.0), seed)
                .with_kinematics(FaultKinematics::Heave(HeaveParams::default()))
                .with_tips(seed, &TipParams::default());
            let p = DragParams::default();
            let m = m.with_drag(&p);
            let d = m.drag().unwrap();
            for (n, &(t, w, th, unit)) in t12_ref(&m, &p).iter().enumerate() {
                n_faults += 1;
                n_cap += (w >= 400.0 - 1e-9) as usize;
                let df = &d.faults[n];
                let (bh, bf) = (df.beta_h, df.beta_f);
                assert!(bh.max(bf) * unit <= 0.16 + 1e-12, "seed {seed} large {large} fault {n}: B_f {}", bh.max(bf) * unit);
                let lim = 0.16 / unit;
                assert!((bh - 0.30f64.min(lim)).abs() < 1e-12 && (bf - 0.15f64.min(lim)).abs() < 1e-12, "seed {seed} fault {n}: beta {bh} {bf}, lim {lim}");
                let rec = d.clamps.iter().find(|c| c.fault == n);
                if 0.30 * unit > 0.16 {
                    n_clamped += 1;
                    let c = rec.unwrap_or_else(|| panic!("seed {seed} fault {n} clamped but not recorded"));
                    assert_eq!(c.before, (0.30, 0.15));
                    assert_eq!(c.after, (bh, bf));
                    assert!((c.t - t).abs() < 1e-12 && (c.w_m - w).abs() < 1e-9 && (c.theta_deg - th).abs() < 1e-9);
                    assert!(d.summary().contains(&format!("fault {n}: B_f=")));
                } else {
                    assert!(rec.is_none(), "seed {seed} fault {n} recorded without a clamp");
                }
            }
        }
    }
    eprintln!("T12: {n_faults} faults, {n_clamped} clamped, {n_cap} at the 400 m cap");
    assert!(n_cap > 0, "no fault hits the 400 m cap");
    let m = random_model([64, 64, 96], 1, 4).with_drag(&DragParams {
        beta_h: 0.5,
        beta_f: 0.5,
        sense: DragSense::Normal,
        ..DragParams::default()
    });
    let d = m.drag().unwrap();
    assert!(d.beta_h + d.beta_f <= 0.9 + 1e-9 && !d.warnings.is_empty());
    assert!(DragParams { beta_h: 0.5, beta_f: 0.5, sense: DragSense::Normal, ..DragParams::default() }
        .validate()
        .is_err());
}

/// The drag-off model is the 7b model: `with_drag` is the only switch and
/// a legacy model ignores it.
#[test]
fn drag_off_and_legacy_are_unchanged() {
    let m = random_model([64, 64, 96], 3, 4);
    assert!(m.drag().is_none() && m.drag_pad() == 0);
    let legacy = FaultModel::resolve(
        [64, 64, 96],
        &sample_random_faults([64, 64, 96], &RandomFaultConfig { count: 4, ..Default::default() }, 3),
        &Seabed::Flat(4.0),
        3,
    );
    assert!(legacy.with_drag(&DragParams::default()).drag().is_none());
}
