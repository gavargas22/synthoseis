//! Salt pv-maps spec §8.1–8.3 (maps only): the partial-voxel maps with salt
//! round to the production maps in every cell, stay ordered and within ½
//! sample, and equal the no-salt continuous maps bit for bit far from the
//! salt body (lift-only drag).
use crate::pipeline::{E2eConfig, FaultConfig};
use crate::pipeline_stream::{toy_horizon_maps, toy_horizon_maps_continuous};

fn cfg(seed: u64, shape: [usize; 3], faults: usize, smooth_all: bool) -> E2eConfig {
    let mut c = E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(faults),
        ..E2eConfig::default()
    };
    c.rock_physics.salt_smooth_all_horizons = smooth_all;
    c
}

/// Chebyshev distance (columns) of every column to the salt body footprint
/// (`SaltBody::runs`, including the padded grid below the cube); `None`
/// without a body or with an empty footprint.
fn footprint_distance(c: &E2eConfig) -> Option<Vec<usize>> {
    let body = crate::salt::salt_body(c)?;
    let [ni, nj, _] = c.shape();
    let foot: Vec<(usize, usize)> = (0..ni * nj)
        .filter(|&q| body.runs[q].1 > body.runs[q].0)
        .map(|q| (q / nj, q % nj))
        .collect();
    if foot.is_empty() {
        return None;
    }
    Some(
        (0..ni * nj)
            .map(|q| {
                let (i, j) = (q / nj, q % nj);
                foot.iter().map(|&(a, b)| i.abs_diff(a).max(j.abs_diff(b))).min().unwrap()
            })
            .collect(),
    )
}

struct Gate {
    far: usize,
    near_tail_max: f64,
    fractional_far: usize,
}

/// Every invariant of §8.2, and (lift-only) the §8.3 map gate.
fn check(c: &E2eConfig) -> Gate {
    let what = format!(
        "seed {} shape {:?} faults {} smooth-all {}",
        c.seed,
        c.shape(),
        c.faults.count,
        c.rock_physics.salt_smooth_all_horizons
    );
    let (p, nh) = toy_horizon_maps(c);
    let (z, nh2) = toy_horizon_maps_continuous(c).unwrap();
    assert_eq!(nh, nh2);
    for (q, (&zv, &pv)) in z.iter().zip(&p).enumerate() {
        assert_eq!(zv.round(), pv, "{what}: round(z) != production at {q}");
        assert!((zv - pv).abs() <= 0.5, "{what}: |z − P| > ½ at {q}");
        let h = q % nh;
        if h >= 2 {
            assert!(z[q - 1] <= zv, "{what}: order at {q}");
        }
    }
    let mut g = Gate { far: 0, near_tail_max: 0.0, fractional_far: 0 };
    let Some(dist) = footprint_distance(c) else {
        return g;
    };
    if c.rock_physics.salt_smooth_all_horizons {
        return g;
    }
    let mut ns = c.clone();
    ns.rock_physics.salt = false;
    let (z0, _) = toy_horizon_maps_continuous(&ns).unwrap();
    for (col, d) in dist.iter().enumerate() {
        for h in 0..nh {
            let q = col * nh + h;
            if *d >= 13 {
                assert_eq!(z[q].to_bits(), z0[q].to_bits(), "{what}: far map at column {col} horizon {h}");
                g.far += 1;
                if z[q].fract() != 0.0 {
                    g.fractional_far += 1;
                }
            } else if *d >= 9 {
                let e = (z[q] - z0[q]).abs();
                assert!(e <= 0.1, "{what}: drag tail {e} at column {col} horizon {h}");
                g.near_tail_max = g.near_tail_max.max(e);
            }
        }
    }
    g
}

/// §8.1: far from the body the salt-on partial-voxel maps are the no-salt
/// continuous maps bit for bit, and they are sub-sample (not snapped).
#[test]
fn far_field_maps_equal_no_salt() {
    let g = check(&cfg(7, [32, 32, 128], 3, false));
    assert!(g.far > 0, "no far cells");
    assert!(g.fractional_far > 0, "far maps are all whole samples (snapped)");
    // The rounded legacy maps snap every far value.
    let mut c = cfg(7, [32, 32, 128], 3, false);
    c.rock_physics.salt_legacy_pv_maps = true;
    let (z, _) = toy_horizon_maps_continuous(&c).unwrap();
    let dist = footprint_distance(&c).unwrap();
    let nh = z.len() / dist.len();
    assert!(dist
        .iter()
        .enumerate()
        .filter(|(_, d)| **d >= 13)
        .all(|(col, _)| (0..nh).all(|h| z[col * nh + h].fract() == 0.0)));
}

/// §8.1 second case: a salt body with 0 voxels inside the cube (the 8³
/// default) no longer snaps the maps; they equal the no-salt continuous
/// maps outside the drag reach (every cell whose drag offset is 0).
#[test]
fn empty_body_maps_equal_no_salt() {
    let c = cfg(42, [8, 8, 8], 0, false);
    let body = crate::salt::salt_body(&c).expect("the 8³ default has a salt body");
    assert_eq!(body.voxels(8), 0);
    let (z, nh) = toy_horizon_maps_continuous(&c).unwrap();
    let mut ns = c.clone();
    ns.rock_physics.salt = false;
    let (z0, _) = toy_horizon_maps_continuous(&ns).unwrap();
    // Outside the drag reach = cells whose drag offset is exactly 0.
    let (r, _) = crate::toy_geometry::layered_horizon_maps(c.seed, c.shape());
    let [ni, nj, _] = c.shape();
    let d = crate::salt::drag_offsets(&r, [ni, nj, nh], &body, false);
    let (p, _) = toy_horizon_maps(&c);
    let mut same = 0;
    for q in 0..z.len() {
        assert_eq!(z[q].round(), p[q], "cell {q}");
        if d[q] == 0.0 {
            assert_eq!(z[q].to_bits(), z0[q].to_bits(), "cell {q}");
            same += 1;
        }
    }
    assert!(same > 0);
    assert!(z.iter().any(|v| v.fract() != 0.0), "maps snapped to whole samples");
}

/// §8.2 clamp invariant (and the §8.3 lift-only map gate): seeds 7, 1, 2,
/// 3, 30 × {32×32×128, 64×64×64} × faults {0, 3} × both drag modes.
#[test]
fn clamp_invariant_and_far_field_map_gate() {
    let mut far = 0;
    let mut tail: f64 = 0.0;
    for seed in [7u64, 1, 2, 3, 30] {
        for shape in [[32, 32, 128], [64, 64, 64]] {
            for faults in [0usize, 3] {
                for smooth_all in [false, true] {
                    let g = check(&cfg(seed, shape, faults, smooth_all));
                    far += g.far;
                    tail = tail.max(g.near_tail_max);
                }
            }
        }
    }
    println!("far map values bit-identical to --no-salt: {far}; drag tail at 9–12 columns: max {tail:.4} sample");
    assert!(far > 0);
}

/// Nightly sweep: seeds 1–30, 32×32×128, both drag modes.
#[test]
#[ignore = "nightly: clamp invariant and far-field map gate, seeds 1-30"]
fn clamp_invariant_seed_sweep() {
    for seed in 1u64..=30 {
        for smooth_all in [false, true] {
            check(&cfg(seed, [32, 32, 128], 3, smooth_all));
        }
    }
}
