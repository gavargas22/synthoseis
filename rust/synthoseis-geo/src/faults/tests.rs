use super::*;

/// Two faults from the committed Python fixture (`two_faults`, legacy units).
fn fixture_faults() -> Vec<FaultParams> {
    vec![
        FaultParams::from_legacy(
            15630.871845812155,
            129324.39734434821,
            5281125.689116923,
            -83.67823121884922,
            -65.21191981149121,
            -283.4972249587197,
            239.42875889968417,
            0.7388156621906314,
            10.0,
        )
        .with_profile(294.71916271887153, 4.803417853110496, 1.3643538502222659),
        FaultParams::from_legacy(
            124722.79736481395,
            54103.507564377636,
            5001372.2102915775,
            -330.3525104436583,
            -44.67621052975579,
            -242.2633461801118,
            105.94762908535289,
            0.6597227214441098,
            10.0,
        )
        .with_profile(241.6153411268524, 4.900705688526008, 1.4217406829944983),
    ]
}

const SHAPE: [usize; 3] = [32, 32, 48];

fn model() -> FaultModel {
    FaultModel::resolve(SHAPE, &fixture_faults(), &Seabed::Flat(4.0), 7)
}

fn assemble(m: &FaultModel, tile: [usize; 2]) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
    let [ni, nj, nk] = m.shape();
    let mut lookup = vec![0.0f32; ni * nj * nk];
    let mut mask = vec![0u8; ni * nj * nk];
    let mut ids = vec![0u8; ni * nj * nk];
    m.for_each_tile(tile, |t| {
        for i in t.i0..t.i1 {
            for j in t.j0..t.j1 {
                let g = (i * nj + j) * nk;
                let l = t.col_offset(i, j);
                lookup[g..g + nk].copy_from_slice(&t.lookup[l..l + nk]);
                mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                ids[g..g + nk].copy_from_slice(&t.segment_id[l..l + nk]);
            }
        }
    });
    (lookup, mask, ids)
}

#[test]
fn sampler_is_deterministic_and_in_range() {
    let cfg = RandomFaultConfig {
        count: 6,
        ..RandomFaultConfig::default()
    };
    let a = sample_random_faults([64, 64, 96], &cfg, 42);
    let b = sample_random_faults([64, 64, 96], &cfg, 42);
    let c = sample_random_faults([64, 64, 96], &cfg, 43);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.len(), 6);
    for p in &a {
        assert!(p.throw >= 5.0 && p.throw < 29.0, "throw {}", p.throw);
        assert!(!p.is_hockey_stick());
        assert!((100.0f64.powi(2)..600.0f64.powi(2)).contains(&p.a));
        assert!((0.1..0.75).contains(&p.tilt_pct));
        assert!((1.5..5.0).contains(&p.p) && (1.3..1.5).contains(&p.coef));
        assert!(p.center.is_none());
    }
    // Prefix stability: fault n does not depend on how many faults follow.
    let fewer = sample_random_faults(
        [64, 64, 96],
        &RandomFaultConfig {
            count: 2,
            ..cfg.clone()
        },
        42,
    );
    assert_eq!(&a[..2], &fewer[..]);
}

#[test]
fn empty_model_is_identity() {
    let m = FaultModel::empty([4, 5, 6]);
    let t = m.compute_tile(0, 4, 0, 5);
    for col in 0..20 {
        for k in 0..6 {
            assert_eq!(t.lookup[col * 6 + k], k as f32);
        }
    }
    assert!(t.mask.iter().all(|&v| v == 0));
    let mut labels: Vec<u8> = (0..120).map(|v| (v % 7) as u8).collect();
    let orig = labels.clone();
    let (mask, ids) = m.apply_to_labels(&mut labels, [2, 2]);
    assert_eq!(labels, orig);
    assert!(mask.iter().all(|&v| v == 0) && ids.iter().all(|&v| v == 0));
}

#[test]
fn resolve_matches_python_centres_when_explicit() {
    let ps: Vec<FaultParams> = fixture_faults()
        .into_iter()
        .zip([[22, 13, 37], [11, 13, 10]])
        .map(|(p, c)| p.with_center(Some(c)))
        .collect();
    let m = FaultModel::resolve(SHAPE, &ps, &Seabed::Flat(4.0), 0);
    assert_eq!(m.faults().len(), 2);
    assert_eq!(m.faults()[0].center, [22, 13, 37]);
    assert!(m.skipped().is_empty());
}

#[test]
fn seeded_centre_choice_is_deterministic() {
    let a = model();
    let b = model();
    assert_eq!(a.faults().len(), 2);
    let ca: Vec<_> = a.faults().iter().map(|f| f.center).collect();
    let cb: Vec<_> = b.faults().iter().map(|f| f.center).collect();
    assert_eq!(ca, cb);
}

#[test]
fn tiles_are_independent_of_tiling() {
    let m = model();
    let whole = assemble(&m, [SHAPE[0], SHAPE[1]]);
    assert!(whole.1.contains(&1), "fixture must produce fault voxels");
    assert!(whole.2.contains(&2), "second fault must label voxels");
    for tile in [[1, 1], [3, 5], [8, 8], [16, 32], [7, 32]] {
        assert_eq!(assemble(&m, tile), whole, "tile {tile:?}");
    }
    // ids exactly cover the mask
    for (&mk, &id) in whole.1.iter().zip(&whole.2) {
        assert_eq!(mk == 1, id != 0);
    }
}

#[test]
fn multi_worker_threads_match_single_worker() {
    let m = model();
    let [ni, nj, nk] = SHAPE;
    let single = assemble(&m, [SHAPE[0], SHAPE[1]]).1;
    // 4 workers own contiguous inline strips; each evaluates its own tiles.
    let workers = 4;
    let strips: Vec<(usize, usize)> = (0..workers)
        .map(|w| (w * ni / workers, (w + 1) * ni / workers))
        .collect();
    let parts: Vec<Vec<(usize, Vec<u8>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = strips
            .iter()
            .map(|&(a, b)| {
                let m = &m;
                s.spawn(move || {
                    let mut out = Vec::new();
                    let mut j0 = 0;
                    while j0 < nj {
                        let j1 = (j0 + 6).min(nj);
                        let t = m.compute_tile(a, b, j0, j1);
                        for i in a..b {
                            for j in j0..j1 {
                                let l = t.col_offset(i, j);
                                out.push(((i * nj + j) * nk, t.mask[l..l + nk].to_vec()));
                            }
                        }
                        j0 = j1;
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut multi = vec![0u8; ni * nj * nk];
    for part in parts {
        for (g, col) in part {
            multi[g..g + nk].copy_from_slice(&col);
        }
    }
    assert_eq!(multi, single);
}

#[test]
fn labels_are_displaced_downward_in_hanging_wall() {
    let m = model();
    let [ni, nj, nk] = SHAPE;
    let mut labels = vec![0u8; ni * nj * nk];
    for col in 0..ni * nj {
        for k in 0..nk {
            labels[col * nk + k] = (k / 6) as u8;
        }
    }
    let orig = labels.clone();
    let (mask, _) = m.apply_to_labels(&mut labels, [8, 8]);
    assert!(mask.contains(&1));
    assert_ne!(labels, orig);
    // Normal faulting: values are only pulled from shallower samples.
    for (a, b) in labels.iter().zip(&orig) {
        assert!(a <= b);
    }
}

#[test]
fn throw_variance_matches_python_lookup() {
    assert!((throw_variance(34.9) - 16000.0).abs() < 1e-9);
    let v5 = 16000.0 * (5.0f64 / 34.0).powf(1.3258);
    assert!((throw_variance(5.2) - v5).abs() / v5 < 1e-12);
    // clamped outside the legacy lookup
    assert_eq!(throw_variance(2.0), throw_variance(5.0));
    assert_eq!(throw_variance(50.0), throw_variance(34.0));
}

#[test]
fn vertical_profile_peaks_at_centre() {
    // Narrow profile, seabed far above: no taper roll.
    let (g, roll) = vertical_profile(64, 10.0, 3.0, 2.0, 30, -100.0);
    assert_eq!(roll, 0);
    let kmax = (0..64).max_by(|&a, &b| g[a].total_cmp(&g[b])).unwrap();
    assert!((28..=30).contains(&kmax), "peak at {kmax}");
    // even-length window: peak samples sit at n = ±0.5, just below throw
    assert!(g[kmax] <= 10.0 && g[kmax] > 9.9);
    // Seabed inside a wide profile: rolled down in steps of 5.
    let (_, roll) = vertical_profile(64, 10.0, 3.0, 2.0, 5, 6.0);
    assert!(roll > 0 && roll % 5 == 0);
}

#[test]
fn numpy_interp_semantics() {
    let fp = [0.0f32, 10.0, 20.0, 40.0];
    assert_eq!(interp_uniform(-1.0, &fp), 0.0);
    assert_eq!(interp_uniform(3.0, &fp), 40.0);
    assert_eq!(interp_uniform(9.0, &fp), 40.0);
    assert_eq!(interp_uniform(1.0, &fp), 10.0);
    assert!((interp_uniform(2.25, &fp) - 25.0).abs() < 1e-12);
    let age: Vec<f32> = (0..10).map(|k| k as f32 * 0.5).collect();
    assert!((horizon_depth_from_age(&age, 2.0) - 4.0).abs() < 1e-12);
    assert!((horizon_depth_from_age(&age, 1.25) - 2.5).abs() < 1e-12);
    assert_eq!(horizon_depth_from_age(&age, -3.0), 0.0);
    assert_eq!(horizon_depth_from_age(&age, 99.0), 9.0);
}

#[test]
fn segments_are_on_the_ellipsoid_surface() {
    let p = &fixture_faults()[0];
    let g = FaultGeometry::new(SHAPE, p);
    let mut seg = Vec::new();
    segment_block(&g, SHAPE, 0, SHAPE[0], 0, SHAPE[1], &mut seg);
    let nk = SHAPE[2];
    let mut n = 0;
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            for k in 0..nk {
                if seg[(i * SHAPE[1] + j) * nk + k] == SEG_ONE {
                    n += 1;
                    // a surface voxel has both inside and outside neighbours
                    let mut has_in = false;
                    let mut has_out = false;
                    for di in -1i64..=1 {
                        for dj in -1i64..=1 {
                            for dk in -1i64..=1 {
                                let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                                if a < 0 || b < 0 || c < 0 {
                                    continue;
                                }
                                let (a, b, c) = (a as usize, b as usize, c as usize);
                                if a >= SHAPE[0] || b >= SHAPE[1] || c >= nk {
                                    continue;
                                }
                                if g.inside(a, b, c) {
                                    has_in = true;
                                } else {
                                    has_out = true;
                                }
                            }
                        }
                    }
                    assert!(has_in && has_out);
                }
            }
        }
    }
    assert!(n > 0);
}
