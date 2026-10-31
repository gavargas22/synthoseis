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

/// §9 row 2, independent of the kernel (review S5, Strata's check A): the
/// offset is measured through `apply_to_volume_f32` on coordinate fields
/// and compared with `d / max(tan δ_v, tan δ_min,v)`, where `tan δ_v` comes
/// from a finite-difference normal of the surface function (not the
/// kernel's analytic gradient) and `d` is the measured vertical offset.
/// Within 0.5 bin at ≥ 99 % of the displaced voxels (the rest sit where the
/// slip changes between the two samples the gather interpolates); the dip
/// floor always holds and no voxel moves toward the footwall. Kills (checked):
/// heave sign flipped, dip floor removed (the planar test kills tan δ form).
#[test]
fn heave_matches_a_finite_difference_dip() {
    let shape = [64, 64, 96];
    let [ni, nj, nk] = shape;
    let p = HeaveParams::default();
    let (mut n, mut bad, mut floor_bad, mut toward, mut floored) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for m in single_faults(8) {
        let mk = |f: &dyn Fn(usize, usize, usize) -> f32| {
            let mut v = vec![0f32; ni * nj * nk];
            for i in 0..ni {
                for j in 0..nj {
                    for k in 0..nk {
                        v[(i * nj + j) * nk + k] = f(i, j, k);
                    }
                }
            }
            v
        };
        let (mut vx, mut vy, mut vz) = (
            mk(&|i, _, _| i as f32),
            mk(&|_, j, _| j as f32),
            mk(&|_, _, k| k as f32),
        );
        for v in [&mut vx, &mut vy, &mut vz] {
            m.apply_to_volume_f32(v, [16, 16]);
        }
        let f = &m.faults()[0];
        let e = 1e-3;
        for i in 2..ni - 2 {
            for j in 2..nj - 2 {
                for k in 2..nk - 2 {
                    let o = (i * nj + j) * nk + k;
                    let (sx, sy, sz) = (vx[o] as f64, vy[o] as f64, vz[o] as f64);
                    let d = k as f64 - sz;
                    if d <= 1.0
                        || sz <= 1.0
                        || sx <= 1.0
                        || sy <= 1.0
                        || sx >= (ni - 2) as f64
                        || sy >= (nj - 2) as f64
                    {
                        continue;
                    }
                    let v = |x: f64, y: f64, z: f64| f.geometry.value_at(x, y, z);
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let g = [
                        (v(x + e, y, z) - v(x - e, y, z)) / (2.0 * e),
                        (v(x, y + e, z) - v(x, y - e, z)) / (2.0 * e),
                        (v(x, y, z + e) - v(x, y, z - e)) / (2.0 * e),
                    ];
                    let gh = g[0].hypot(g[1]);
                    if gh < 1e-9 || g[2] == 0.0 {
                        continue;
                    }
                    let tan_v = gh / g[2].abs();
                    floored += (tan_v < p.tan_min_v()) as usize;
                    let want = d / tan_v.max(p.tan_min_v());
                    let uh = (x - sx).hypot(y - sy);
                    n += 1;
                    bad += ((uh - want).abs() > 0.5) as usize;
                    floor_bad += (uh > 0.5 && d / uh < p.tan_min_v() - 1e-3) as usize;
                    toward += ((x - sx) * g[0] + (y - sy) * g[1] > 1e-9) as usize;
                }
            }
        }
    }
    assert!(
        n > 5000 && floored > 500,
        "coverage n={n} floored={floored}"
    );
    assert!(
        bad as f64 <= 0.01 * n as f64,
        "{bad}/{n} voxels off d/tan δ by > 0.5 bin"
    );
    assert_eq!(floor_bad, 0, "dip floor violated");
    assert_eq!(toward, 0, "hanging wall moved toward the footwall");
}

/// §9 row 2, analytic (review S5): a planar fault dipping 60° (physical)
/// gives a heave of `T·dz / (tan 60°·bin)` bins (`T / tan 60°` in metres)
/// ± 0.5 bin and a vertical offset `T` ± 0.5 sample at every displaced
/// voxel, down-dip. The plane `x cos θ − z sin θ = c` (dipping towards +x) is the near face of
/// a slab-shaped ellipsoid (huge axes), so the kernel sees an exactly
/// planar surface; throw profile and lateral weight are a real fault's.
#[test]
fn planar_60_degree_fault_has_the_analytic_heave() {
    let p = HeaveParams::default();
    let tan_v = 60f64.to_radians().tan() * p.bin_m / p.dz_m;
    // Normal n = (cos θ, 0, sin θ) has tan δ_v = cos θ / sin θ.
    let th = (1.0 / tan_v).atan();
    let (c, s) = (th.cos(), th.sin());
    let rot = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
    let half = 1.0e3;
    let [ni, nj, nk] = SHAPE;
    let mut m = single_faults(1).remove(0);
    let plane = 0.5 * (ni as f64 * c - nk as f64 * s);
    m.faults_mut()[0].geometry =
        FaultGeometry::from_parts(rot, [plane + half, 0.0, 0.0], [half * half, 1e12, 1e12]);
    let f = m.faults()[0].clone();
    let (mut at60, mut worst) = (0usize, 0f64);
    for i in 1..ni - 1 {
        for j in 1..nj - 1 {
            for k in 1..nk - 1 {
                if f.geometry.value(i, j, k) > 1.0 {
                    continue;
                }
                let d = f.lateral(i, j) * f.profile[k];
                let s = m.gather(i, j, k).src;
                if d <= 1.0 || s[2] <= 0.0 || s[0] <= 0.0 || s[0] >= (ni - 1) as f64 {
                    continue;
                }
                let want = d * p.dz_m / (60f64.to_radians().tan() * p.bin_m);
                let uh = (i as f64 - s[0]).hypot(j as f64 - s[1]);
                assert!(
                    (uh - want).abs() <= 0.5,
                    "({i},{j},{k}): heave {uh} vs {want}"
                );
                assert!(
                    ((k as f64 - s[2]) - d).abs() <= 0.5,
                    "vertical {} vs {d}",
                    k as f64 - s[2]
                );
                // Down-dip: the hanging wall (above the plane) moves towards +x.
                assert!((i as f64 - s[0]) * c > 0.0, "heave toward the footwall");
                worst = worst.max((uh - want).abs());
                at60 += 1;
            }
        }
    }
    assert!(at60 > 1000, "at60 = {at60}");
    assert!(worst <= 0.5, "worst {worst}");
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
    tiles_equal_the_whole_cube(false);
}

/// Row 9 with 7b tip lines on (the pipeline default).
#[test]
fn tipped_tiles_equal_the_whole_cube() {
    tiles_equal_the_whole_cube(true);
}

fn tips(m: FaultModel, seed: u64, on: bool) -> FaultModel {
    if on {
        m.with_tips(seed, &TipParams::default())
    } else {
        m
    }
}

fn tiles_equal_the_whole_cube(on: bool) {
    for seed in [1u64, 7] {
        let m = tips(heave(random_model(SHAPE, seed, 4)), seed, on);
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
    stays_inside_the_halo_bound(false);
}

/// Row 10 with 7b tip lines on: the taper is ≤ 1, so the bound holds.
#[test]
fn tipped_heave_stays_inside_the_halo_bound() {
    stays_inside_the_halo_bound(true);
}

fn stays_inside_the_halo_bound(on: bool) {
    let p = HeaveParams::default();
    let h = p.halo(RandomFaultConfig::default().throw_max) as f64;
    assert_eq!(h, 19.0);
    let shape = [128, 128, 256];
    let mut worst = 0f64;
    for seed in 1..=20u64 {
        let m = tips(heave(random_model(shape, seed, 4)), seed, on);
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

/// Review S1/S2: the shared per-tile gather cache gives the uncached
/// tiles and column sources bit for bit (on a different tiling than the
/// cache grid), and the windowed in-place apply (tile + halo only, no
/// whole-cube copy) equals a reference that gathers from a full copy.
#[test]
fn gather_cache_and_halo_window_match_the_reference() {
    for seed in [1u64, 7, 13] {
        let plain = heave(random_model(SHAPE, seed, 4));
        if !plain.has_heave() {
            continue;
        }
        let cached = plain.clone().with_gather_cache([16, 16]);
        let [ni, nj, nk] = SHAPE;
        for (i0, i1, j0, j1) in [(0, ni, 0, nj), (5, 29, 11, 40), (48, 64, 0, 17)] {
            assert_eq!(
                cached.compute_tile(i0, i1, j0, j1),
                plain.compute_tile(i0, i1, j0, j1)
            );
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (i, j) in [(0, 0), (17, 33), (63, 63), (31, 8)] {
            cached.column_sources(i, j, &mut a);
            plain.column_sources(i, j, &mut b);
            assert_eq!(a, b);
            let g: Vec<(f64, f64)> = (0..nk)
                .map(|k| {
                    let s = plain.gather(i, j, k).src;
                    (s[0], s[1])
                })
                .collect();
            assert_eq!(a, g);
        }
        let base: Vec<u8> = (0..ni * nj * nk).map(|o| 1 + (o % nk / 8) as u8).collect();
        let whole = plain.compute_tile(0, ni, 0, nj);
        let mut want = base.clone();
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let [si, sj, sk] = whole.nearest_source(i, j, k, ni, nj);
                    want[(i * nj + j) * nk + k] = base[(si * nj + sj) * nk + sk];
                }
            }
        }
        for tile in [[16, 16], [7, 64], [64, 5]] {
            let mut got = base.clone();
            cached.apply_to_labels(&mut got, tile);
            assert_eq!(got, want, "seed {seed} tile {tile:?}");
        }
        let vol: Vec<f32> = (0..ni * nj * nk)
            .map(|o| (o % nk) as f32 * 0.5 + (o / nk % 7) as f32)
            .collect();
        let mut vwant = vol.clone();
        for i in 0..ni {
            for j in 0..nj {
                let l = whole.col_offset(i, j);
                for k in 0..nk {
                    let (x, y) = whole.source_ij(i, j, k);
                    vwant[(i * nj + j) * nk + k] =
                        trilinear(&vol, SHAPE, x, y, whole.lookup[l + k] as f64);
                }
            }
        }
        let mut vgot = vol.clone();
        cached.apply_to_volume_f32(&mut vgot, [9, 13]);
        assert_eq!(vgot, vwant);
    }
}

/// Review S1: the gather cache holds at most `GATHER_CACHE_TILES` tiles
/// whatever the cube size, and cached / uncached results are identical.
#[test]
fn gather_cache_memory_is_bounded_by_the_tile() {
    let shape = [96, 96, 48];
    let g = [16, 16];
    let plain = heave(random_model(shape, 3, 4));
    if !plain.has_heave() {
        return;
    }
    let cached = plain.clone().with_gather_cache(g);
    let bound = super::apply::GATHER_CACHE_TILES * g[0] * g[1] * shape[2] * 25;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for pass in 0..2 {
        for i0 in (0..shape[0]).step_by(g[0]) {
            for j0 in (0..shape[1]).step_by(g[1]) {
                let (i1, j1) = (i0 + g[0], j0 + g[1]);
                let (x, y) = (
                    cached.compute_tile(i0, i1, j0, j1),
                    plain.compute_tile(i0, i1, j0, j1),
                );
                assert_eq!(x.lookup, y.lookup);
                assert_eq!(x.src_ij, y.src_ij);
                assert_eq!(x.mask, y.mask);
                assert_eq!(x.segment_id, y.segment_id);
                assert!(cached.gather_cache_bytes() <= bound, "pass {pass}");
            }
        }
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                cached.column_sources(i, j, &mut a);
                plain.column_sources(i, j, &mut b);
                assert_eq!(a, b);
            }
        }
        assert!(cached.gather_cache_bytes() <= bound);
    }
    assert!(cached.gather_cache_bytes() > 0);
}
