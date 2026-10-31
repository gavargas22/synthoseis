//! Spec-fault-kinematics 7a tests (§9 rows 2–5 and 9–11). Row 1 (parity
//! floor) is the CLI byte-equality check; rows 12–13 live in
//! `synthoseis-core/tests/fault_heave.rs`.
use super::*;

const SHAPE: [usize; 3] = [64, 64, 96];

fn random_model(shape: [usize; 3], seed: u64, count: usize) -> FaultModel {
    let cfg = RandomFaultConfig {
        count,
        ..RandomFaultConfig::default()
    };
    let params = sample_random_faults(shape, &cfg, seed);
    FaultModel::resolve(shape, &params, &Seabed::Flat(4.0), seed)
}

fn heave(m: FaultModel) -> FaultModel {
    m.with_kinematics(FaultKinematics::Heave(HeaveParams::default()))
}

/// The first `n` single-fault models (one inserted fault) from seeds 1, 2, …
fn single_faults(n: usize) -> Vec<FaultModel> {
    (1..400u64)
        .map(|s| heave(random_model(SHAPE, s, 1)))
        .filter(|m| m.faults().len() == 1)
        .take(n)
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Every displaced voxel of a one-fault model: `(i, j, k, u, d, ∇f)` with
/// `u = output − source` (bins, bins, samples) from the gather. Voxels whose
/// source was clamped to the cube are left out.
fn displaced(m: &FaultModel) -> Vec<([usize; 3], [f64; 3], f64, [f64; 3])> {
    let f = &m.faults()[0];
    let [ni, nj, nk] = m.shape();
    let top = (nk - 1) as f64;
    let p = HeaveParams::default();
    let mut out = Vec::new();
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let (a, _, d) = stage_slip(f, &p, i as f64, j as f64, k, f.lateral(i, j), top);
                if d <= 1.0 || a <= 0.0 || a >= top {
                    continue;
                }
                let s = m.gather(i, j, k).src;
                if s[0] <= 0.0 || s[1] <= 0.0 || s[0] >= (ni - 1) as f64 || s[1] >= (nj - 1) as f64
                {
                    continue;
                }
                let u = [i as f64 - s[0], j as f64 - s[1], k as f64 - s[2]];
                let g = f.geometry.gradient_at(i as f64, j as f64, k as f64);
                out.push(([i, j, k], u, d, g));
            }
        }
    }
    out
}

/// §9 row 2: the slip vector lies in the fault surface (down-dip) where the
/// surface is steeper than the 30° floor, and is `d / tan δ_min` sideways
/// where it is shallower. At voxels whose physical dip is 60° (± 0.5°) the
/// heave is `T·dz / (tan 60°·bin)` bins (± 0.5 bin) and the vertical offset
/// is `T` (± 0.5 sample). Kills: heave sign flipped, heave = 0, tan ↔ sin.
#[test]
fn heave_is_down_dip_with_the_dip_floor() {
    let p = HeaveParams::default();
    let (mut steep, mut floor, mut at60) = (0usize, 0usize, 0usize);
    for m in single_faults(6) {
        for (_, u, d, g) in displaced(&m) {
            let nh = norm(&g[..2]);
            if nh < 1e-9 || g[2] == 0.0 {
                continue;
            }
            assert!(
                (u[2] - d).abs() < 1e-4,
                "vertical offset {} vs throw {d}",
                u[2]
            );
            let uh = norm(&u[..2]);
            let tan_v = nh / g[2].abs();
            if tan_v >= p.tan_min_v() {
                // ∇f · u = 0: the hanging wall slides along the surface.
                let dot = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];
                assert!(
                    dot.abs() <= 1e-4 * norm(&g) * norm(&u),
                    "slip leaves the surface: {dot}"
                );
                steep += 1;
            } else {
                assert!(
                    (uh - d / p.tan_min_v()).abs() < 1e-4,
                    "floor heave {uh} vs {}",
                    d / p.tan_min_v()
                );
                floor += 1;
            }
            let dip_deg = (tan_v * p.dz_m / p.bin_m).atan().to_degrees();
            if (dip_deg - 60.0).abs() <= 0.5 {
                let want = d * p.dz_m / (60f64.to_radians().tan() * p.bin_m);
                assert!((uh - want).abs() <= 0.5, "60° heave {uh} vs {want}");
                at60 += 1;
            }
        }
    }
    // Legacy surfaces are low-angle in physical units, so a 64×64×96 draw
    // has few voxels at 60°; the at-60° check runs wherever they occur.
    assert!(
        steep > 1000 && floor > 1000,
        "coverage steep={steep} floor={floor} at60={at60}"
    );
}

/// §9 row 3: the hanging wall moves away from the footwall in map view
/// (extension; a gap, not an overlap, in the horizon map). `∇f` points out
/// of the ellipsoid, i.e. towards the footwall, so `u_h · ∇_h f < 0`.
/// Kills: ŝ_d sign.
#[test]
fn hanging_wall_moves_away_from_the_footwall() {
    let (mut gap, mut overlap) = (0usize, 0usize);
    for m in single_faults(6) {
        for (_, u, _, g) in displaced(&m) {
            let dot = u[0] * g[0] + u[1] * g[1];
            if norm(&u[..2]) < 1e-9 {
                continue;
            }
            if dot < 0.0 {
                gap += 1
            } else {
                overlap += 1
            }
        }
    }
    assert!(gap > 10_000, "gap voxels {gap}");
    assert_eq!(overlap, 0, "{overlap} voxels move towards the footwall");
}

fn legacy_assemble(m: &FaultModel) -> (Vec<f32>, Vec<u8>) {
    let [ni, nj, nk] = m.shape();
    let t = m.compute_tile(0, ni, 0, nj);
    assert!(t.src_ij.is_none());
    (t.lookup, t.mask)
}

fn jaccard(a: &[u8], b: &[u8]) -> f64 {
    let inter = a
        .iter()
        .zip(b)
        .filter(|(x, y)| **x != 0 && **y != 0)
        .count();
    let uni = a
        .iter()
        .zip(b)
        .filter(|(x, y)| **x != 0 || **y != 0)
        .count();
    if uni == 0 {
        1.0
    } else {
        inter as f64 / uni as f64
    }
}

/// §9 row 4: with heave forced to 0 the gather reproduces the legacy throw
/// (max |Δ lookup| ≤ 1e-4 sample) and the closed-form label band overlaps
/// the legacy sobel label (spec §6.4: Jaccard ≥ 0.70 per seed and ≥ 0.83 on
/// the mean over ≥ 30 seeds, band 1.0). Kills: any change to the
/// vertical component.
#[test]
fn zero_heave_reproduces_the_legacy_throw_and_label() {
    let zero = FaultKinematics::Heave(HeaveParams {
        heave_scale: 0.0,
        ..HeaveParams::default()
    });
    let (mut checked, mut jac_sum) = (0, 0.0);
    for seed in 1..=60u64 {
        let legacy = random_model(SHAPE, seed, 2);
        if legacy.faults().is_empty() {
            continue;
        }
        let (look, mask) = legacy_assemble(&legacy);
        let m = legacy.clone().with_kinematics(zero);
        let [ni, nj, _] = SHAPE;
        let t = m.compute_tile(0, ni, 0, nj);
        let dmax = look
            .iter()
            .zip(&t.lookup)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(dmax <= 1e-4, "seed {seed}: max |Δ throw| {dmax}");
        let s = t.src_ij.as_ref().unwrap();
        assert!(s.iter().enumerate().all(|(o, &(x, y))| {
            let c = o / SHAPE[2];
            (x, y) == ((c / nj) as f32, (c % nj) as f32)
        }));
        let jac = jaccard(&mask, &t.mask);
        assert!(jac >= 0.70, "seed {seed}: label Jaccard {jac:.3}");
        jac_sum += jac;
        checked += 1;
    }
    assert!(checked >= 30, "only {checked} faulted seeds");
    let mean = jac_sum / checked as f64;
    assert!(
        mean >= 0.83,
        "mean label Jaccard {mean:.3} over {checked} seeds"
    );
    println!("label Jaccard mean {mean:.3} over {checked} faulted seeds");
}

/// §9 row 5: every output voxel gets a source (gather, no holes) and every
/// gathered label is one of the input labels.
#[test]
fn gather_leaves_no_voxel_unset() {
    for seed in [1u64, 7, 13] {
        let m = heave(random_model(SHAPE, seed, 4));
        let [ni, nj, nk] = SHAPE;
        let mut labels: Vec<u8> = (0..ni * nj * nk).map(|o| 1 + (o % nk / 8) as u8).collect();
        let before = labels.clone();
        m.apply_to_labels(&mut labels, [ni, nj]);
        assert!(labels.iter().all(|&l| l != 0), "seed {seed}: unset voxels");
        let max = *before.iter().max().unwrap();
        assert!(labels.iter().all(|&l| l <= max));
        assert_ne!(labels, before, "seed {seed}: nothing moved");
    }
}

/// §9 row 9: 4+ tiles equal one whole-cube tile, bit for bit (lookup, mask,
/// ids, src_ij, gathered labels and volume).
#[test]
fn heave_tiles_equal_the_whole_cube() {
    for seed in [1u64, 7] {
        let m = heave(random_model(SHAPE, seed, 4));
        let [ni, nj, nk] = SHAPE;
        let whole = m.compute_tile(0, ni, 0, nj);
        let mut n_tiles = 0;
        m.for_each_tile([23, 17], |t| {
            n_tiles += 1;
            for i in t.i0..t.i1 {
                for j in t.j0..t.j1 {
                    let (a, b) = (t.col_offset(i, j), whole.col_offset(i, j));
                    assert_eq!(t.lookup[a..a + nk], whole.lookup[b..b + nk]);
                    assert_eq!(t.mask[a..a + nk], whole.mask[b..b + nk]);
                    assert_eq!(t.segment_id[a..a + nk], whole.segment_id[b..b + nk]);
                    let (sa, sb) = (t.src_ij.as_ref().unwrap(), whole.src_ij.as_ref().unwrap());
                    assert_eq!(sa[a..a + nk], sb[b..b + nk]);
                }
            }
        });
        assert!(n_tiles >= 4);
        let base: Vec<u8> = (0..ni * nj * nk).map(|o| 1 + (o % nk / 8) as u8).collect();
        let (mut l1, mut l2) = (base.clone(), base.clone());
        let a = m.apply_to_labels(&mut l1, [ni, nj]);
        let b = m.apply_to_labels(&mut l2, [23, 17]);
        assert_eq!(l1, l2);
        assert_eq!(a, b);
        let vol: Vec<f32> = (0..ni * nj * nk)
            .map(|o| (o % nk) as f32 * 0.5 + (o / nk % 7) as f32)
            .collect();
        let (mut v1, mut v2) = (vol.clone(), vol.clone());
        m.apply_to_volume_f32(&mut v1, [ni, nj]);
        m.apply_to_volume_f32(&mut v2, [23, 17]);
        assert_eq!(v1, v2);
    }
}

/// §9 row 10: no source is further than H − 1 bins sideways from its output
/// column (H = ceil(T_max / tan δ_min,v) + 2 = 19), over seeds 1–20 with 4
/// faults at 128×128×256. Kills: dip floor removed.
#[test]
fn heave_stays_inside_the_halo_bound() {
    let p = HeaveParams::default();
    let h = p.halo(RandomFaultConfig::default().throw_max) as f64;
    assert_eq!(h, 19.0);
    let shape = [128, 128, 256];
    let mut worst = 0f64;
    for seed in 1..=20u64 {
        let m = heave(random_model(shape, seed, 4));
        for i in (0..128).step_by(2) {
            for j in (0..128).step_by(2) {
                for k in 0..256 {
                    let s = m.gather(i, j, k).src;
                    let r = (i as f64 - s[0]).abs().max((j as f64 - s[1]).abs());
                    worst = worst.max(r);
                }
            }
        }
    }
    assert!(
        worst <= h - 1.0,
        "max lateral pull {worst:.2} > H − 1 = {}",
        h - 1.0
    );
    assert!(worst > 1.0, "no heave at all ({worst})");
}

/// One stage by hand: the source of continuous `(x, y, z)` through fault `f`,
/// linear between the integer samples around `z` (spec §6.1/§6.3).
fn hand_stage(
    f: &ResolvedFault,
    p: &HeaveParams,
    x: f64,
    y: f64,
    z: f64,
    top: f64,
    ij_max: f64,
) -> [f64; 3] {
    let lat = f.lateral_at(x, y);
    let k0 = z.floor() as usize;
    let w = z - k0 as f64;
    let (a0, h0, _) = stage_slip(f, p, x, y, k0, lat, top);
    let (a1, h1, _) = stage_slip(f, p, x, y, (k0 + 1).min(top as usize), lat, top);
    let l = |a: f64, b: f64| a + w * (b - a);
    [
        (x - l(h0[0], h1[0])).clamp(0.0, ij_max),
        (y - l(h0[1], h1[1])).clamp(0.0, ij_max),
        l(a0, a1),
    ]
}

/// §9 row 11: two crossing faults compose youngest first (the last fault in
/// the list is applied first in the gather) and match a hand-built
/// composition to 1e-5 voxel; the oldest-first order does not. Kills:
/// evaluating oldest first.
#[test]
fn crossing_faults_compose_youngest_first() {
    let p = HeaveParams::default();
    let mut found = false;
    for seed in 1..60u64 {
        let m = heave(random_model(SHAPE, seed, 2));
        if m.faults().len() != 2 {
            continue;
        }
        let [ni, nj, nk] = SHAPE;
        let top = (nk - 1) as f64;
        let (mut both, mut differs) = (0usize, 0usize);
        for i in (0..ni).step_by(3) {
            for j in (0..nj).step_by(3) {
                for k in 0..nk {
                    let young =
                        hand_stage(&m.faults()[1], &p, i as f64, j as f64, k as f64, top, 63.0);
                    let z1 = young[2] as f32 as f64;
                    let want = hand_stage(&m.faults()[0], &p, young[0], young[1], z1, top, 63.0);
                    let got = m.gather(i, j, k).src;
                    if young[2] != k as f64 && (want[2] - z1).abs() > 0.0 {
                        both += 1;
                    }
                    for c in 0..3 {
                        assert!(
                            (got[c] - want[c] as f32 as f64).abs() <= 1e-5 + 1e-6 * want[c].abs(),
                            "seed {seed} ({i},{j},{k}) axis {c}: {} vs {}",
                            got[c],
                            want[c]
                        );
                    }
                    let old = m.gather_with(&p, i, j, k, true).src;
                    if (0..3).any(|c| (old[c] - got[c]).abs() > 1e-3) {
                        differs += 1;
                    }
                }
            }
        }
        if both > 50 {
            assert!(differs > 0, "seed {seed}: order does not matter here");
            found = true;
            break;
        }
    }
    assert!(found, "no seed with two crossing faults");
}
