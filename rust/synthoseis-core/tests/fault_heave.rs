//! Fault heave at the pipeline level (fault-kinematics spec 7a, §9 rows 12
//! and 13): the rock physics and the unconformity crossing read the source
//! column of every displaced voxel, and the defaults turn heave on.

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel};
use synthoseis_core::{fault_model, generate_labels, generate_unconformity_labels};
use synthoseis_geo::faults::DEFAULT_MIN_DIP_DEG;

fn cfg(seed: u64, shape: [usize; 3], legacy: bool) -> E2eConfig {
    let mut c = E2eConfig::tiny(seed);
    c.inline_count = shape[0];
    c.crossline_count = shape[1];
    c.samples = shape[2];
    c.faults = FaultConfig {
        count: 4,
        legacy_kinematics: legacy,
        ..FaultConfig::default()
    };
    // Salt and fluid contacts put Vp steps away from layer boundaries.
    c.rock_physics.salt = false;
    c.rock_physics.fluids = false;
    c
}

#[test]
fn heave_is_the_default() {
    let f = FaultConfig::default();
    assert!(!f.legacy_kinematics);
    assert_eq!(f.min_dip_deg, DEFAULT_MIN_DIP_DEG);
    assert_eq!(DEFAULT_MIN_DIP_DEG, 30.0);
    let m = fault_model(&cfg(7, [32, 32, 96], false)).expect("faults");
    assert!(m.has_heave());
    let m = fault_model(&cfg(7, [32, 32, 96], true)).expect("faults");
    assert!(!m.has_heave());
}

/// Share of Vp steps (> 2 % between neighbouring samples) with a layer-label
/// boundary within one voxel, over the displaced columns.
fn vp_step_match(c: &E2eConfig) -> (usize, usize) {
    let (labels, shape) = generate_labels(c);
    let [ni, nj, nk] = shape;
    let model = elastic_model(c, &labels, shape);
    let n = ni * nj * nk;
    let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    let (mut steps, mut matched) = (0, 0);
    for q in 0..ni * nj {
        let (l, v) = (&labels[q * nk..(q + 1) * nk], &vp[q * nk..(q + 1) * nk]);
        let edge = |k: usize| k >= 1 && k < nk && l[k] != l[k - 1];
        for k in 2..nk - 1 {
            if l[k - 1] == 255 || (v[k] - v[k - 1]).abs() <= 0.02 * v[k - 1] {
                continue;
            }
            steps += 1;
            matched += (edge(k - 1) || edge(k) || edge(k + 1)) as usize;
        }
    }
    (steps, matched)
}

/// §9 row 12: Vp across a fault follows the source layer. Every Vp step
/// (> 2 %) must sit within one voxel of a layer-label boundary as often as
/// under the legacy kinematics (to 0.5 %): the steps that are not on a
/// boundary are the in-layer ones (closures, lithology units) that legacy
/// has too, so the spec's absolute 99.9 % holds only where those are rare.
#[test]
fn vp_follows_the_source_layer() {
    for seed in [1u64, 7, 13] {
        let (steps, matched) = vp_step_match(&cfg(seed, [48, 48, 128], false));
        let (ls, lm) = vp_step_match(&cfg(seed, [48, 48, 128], true));
        let (share, legacy) = (matched as f64 / steps as f64, lm as f64 / ls as f64);
        assert!(steps > 1000, "seed {seed}: {steps} Vp steps");
        assert!(
            share >= 0.97,
            "seed {seed}: {matched}/{steps} Vp steps on a layer boundary"
        );
        assert!(
            share >= legacy - 5e-3,
            "seed {seed}: share {share:.4} vs legacy {legacy:.4}"
        );
    }
}

/// §9 row 13: each unconformity mark sits on a layer-label boundary of the
/// faulted labels (the crossing is evaluated at the source position).
/// Kills evaluating the surface at the output position.
#[test]
fn unconformity_crossing_is_evaluated_at_the_source() {
    let mut checked = 0;
    for seed in 1..=12u64 {
        let c = cfg(seed, [48, 48, 128], false);
        let Some(marks) = generate_unconformity_labels(&c) else {
            continue;
        };
        let (labels, [ni, nj, nk]) = generate_labels(&c);
        let (mut n, mut on) = (0usize, 0usize);
        for q in 0..ni * nj {
            for k in 1..nk {
                if marks[q * nk + k] != 0 {
                    n += 1;
                    on += (labels[q * nk + k] != labels[q * nk + k - 1]) as usize;
                }
            }
        }
        assert!(n > 1000, "seed {seed}: {n} marks");
        assert_eq!(on, n, "seed {seed}: marks off a layer boundary");
        checked += 1;
    }
    assert!(checked >= 3);
}

/// Config with a strongly varying net-to-gross map in every sand layer
/// (mean `U(0, 1)`, std 0.3), so a few bins of heave change the
/// net-to-gross a voxel should get.
fn ng_cfg(seed: u64) -> E2eConfig {
    let mut c = cfg(seed, [48, 48, 128], false);
    c.rock_physics.net_to_gross = synthoseis_core::rock_physics::NetToGross::Legacy {
        avg: [0.0, 1.0],
        stdev: [0.3, 0.3],
        octaves: 9,
    };
    c
}

/// Concordance of sand Vp with the net-to-gross of the source column and of
/// the output column: over pairs of interior sand voxels of one layer at one
/// sample whose source column is at least 2 bins from the output column, the
/// share of pairs where Vp moves the same way as each net-to-gross (pairs
/// with a net-to-gross difference > 0.1 under both readings).
fn ng_concordance(c: &E2eConfig, model_heave: bool) -> (f64, usize) {
    let (labels, shape) = generate_labels(c);
    let [ni, nj, nk] = shape;
    let mut model = elastic_model(c, &labels, shape);
    let synthoseis_core::rock_physics::ElasticModel::Rpm(m) = &mut model else {
        panic!("physical rock model expected");
    };
    if !model_heave {
        m.heave = None;
    }
    let n = ni * nj * nk;
    let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    let synthoseis_core::rock_physics::ElasticModel::Rpm(m) = &model else {
        unreachable!()
    };
    let faults = fault_model(c).expect("faults");
    let mut groups: std::collections::HashMap<(u8, usize), Vec<(f32, f32, f32)>> =
        Default::default();
    for i in 0..ni {
        for j in 0..nj {
            let q = (i * nj + j) * nk;
            for k in 1..nk - 1 {
                let lab = labels[q + k];
                if lab == 255 || labels[q + k - 1] != lab || labels[q + k + 1] != lab {
                    continue;
                }
                let layer = &m.layers[lab as usize];
                if !layer.sand || layer.ng.is_empty() {
                    continue;
                }
                let s = faults.gather(i, j, k).src;
                let (si, sj) = (s[0].round() as usize, s[1].round() as usize);
                if (si as f64 - i as f64).hypot(sj as f64 - j as f64) < 2.0 {
                    continue;
                }
                let ng_src = layer.ng[si * nj + sj];
                let ng_out = layer.ng[i * nj + j];
                groups
                    .entry((lab, k))
                    .or_default()
                    .push((vp[q + k], ng_src, ng_out));
            }
        }
    }
    // Only the pairs where the two readings disagree on which voxel has the
    // higher net-to-gross tell them apart.
    let (mut pairs, mut follow_src) = (0usize, 0usize);
    for g in groups.values() {
        for (x, a) in g.iter().enumerate() {
            for b in &g[x + 1..] {
                let (dv, ds, d_o) = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
                if dv == 0.0 || ds.abs() <= 0.05 || d_o.abs() <= 0.05 || (ds > 0.0) == (d_o > 0.0) {
                    continue;
                }
                pairs += 1;
                follow_src += ((dv > 0.0) == (ds > 0.0)) as usize;
            }
        }
    }
    (follow_src as f64 / pairs.max(1) as f64, pairs)
}

/// §9 row 12 (strong form): with a laterally varying net-to-gross, sand Vp
/// of a heaved voxel follows the net-to-gross of its source column, not of
/// its output column. Over the voxel pairs where the two readings disagree
/// on which voxel has the higher net-to-gross, Vp sides with the source
/// reading in >= 70 % of pairs (0.97 / 0.92 / 0.76 for seeds 1 / 7 / 13);
/// a model with its heave columns removed (every property, net-to-gross
/// included, read at the output column) sides with it in <= 20 %. This is
/// not the "same-column maps" mutant, which only moves the burial depth's
/// base horizon; [`burial_depth_reads_the_base_horizon_at_the_source_column`]
/// covers that one.
#[test]
fn sand_vp_follows_the_source_column_net_to_gross() {
    for seed in [1u64, 7, 13] {
        let c = ng_cfg(seed);
        let (share, pairs) = ng_concordance(&c, true);
        assert!(pairs > 10_000, "seed {seed}: {pairs} pairs");
        assert!(
            share >= 0.7,
            "seed {seed}: Vp follows the source column in {share:.3} of {pairs} pairs"
        );
        let (out, _) = ng_concordance(&c, false);
        assert!(
            out <= 0.2,
            "seed {seed}: output-column model still follows the source ({out:.3})"
        );
    }
}

/// Every layer run's burial depth comes from its base horizon read at the
/// run's *source* column: tv = zb(src) - z0 + (dk - dk0) with
/// dk = e - floor(zb(src)). The same-column mutant keeps the integer part
/// (dk absorbs it) but takes frac(zb) from the wrong column, so we require
/// it to the sub-sample on runs whose source moved by >= 1 bin and whose
/// two candidate bases differ in their fractional part by > 0.05 sample.
#[test]
fn burial_depth_reads_the_base_horizon_at_the_source_column() {
    let mut tested = 0usize;
    for seed in [1u64, 7, 13] {
        let mut c = E2eConfig::tiny(seed);
        (c.inline_count, c.crossline_count, c.samples) = (48, 48, 128);
        c.faults = FaultConfig {
            count: 4,
            ..FaultConfig::default()
        };
        c.rock_physics.salt = false;
        c.rock_physics.fluids = false;
        let (labels, shape) = generate_labels(&c);
        let [ni, nj, nk] = shape;
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
            panic!("rpm")
        };
        let hv = m.heave.as_ref().expect("heave on by default");
        let (mut src, mut depth) = (Vec::new(), vec![0f32; nk]);
        for i in 0..ni {
            for j in 0..nj {
                let col = &labels[(i * nj + j) * nk..(i * nj + j + 1) * nk];
                hv.column_sources(i, j, nk, &mut src);
                m.depth_trace_src(i, j, col, &src, &mut depth);
                let z0 = m.maps[(i * nj + j) * m.nh];
                let seabed = match col.iter().position(|&v| v != 255) {
                    Some(s) => s,
                    None => continue,
                };
                let dk0 = seabed as i64 - (z0.ceil() as i64).clamp(0, nk as i64);
                let mut k = seabed;
                while k < nk {
                    let mut e = k + 1;
                    while e < nk && col[e] == col[k] {
                        e += 1;
                    }
                    let h = m.intervals.get(col[k] as usize).copied();
                    if let Some(h) = h.filter(|&h| col[k] != 255 && h + 1 < m.nh) {
                        let (sx, sy) = src[e - 1];
                        let moved = (sx - i as f64).hypot(sy - j as f64) >= 1.0;
                        let zs = m.map_at(sx, sy, h + 1);
                        let zo = m.map_at(i as f64, j as f64, h + 1);
                        if moved && (zs.fract() - zo.fract()).abs() > 0.05 {
                            let dk = e as i64 - (zs.floor() as i64).clamp(0, nk as i64);
                            let want = (zs as f32 - z0 as f32 + (dk - dk0) as f32) * m.step;
                            assert!((depth[e - 1] - want).abs() <= 1e-3 * m.step.max(1.0),
                            "seed {seed} ({i},{j}) run {k}..{e}: depth {} want {want} (source base {zs:.3}, own-column base {zo:.3})", depth[e - 1]);
                            tested += 1;
                        }
                    }
                    k = e;
                }
            }
        }
    }
    assert!(tested > 200, "only {tested} displaced runs checked");
    println!("checked {tested} displaced runs");
}

/// Review S3: the salt body is stationary under heave. It is emplaced
/// after faulting (every output already keeps fault ∧ salt = 0), so the
/// salt kind is decided at the output column while the sediments around it
/// are gathered from their source columns: the body is the same with and
/// without heave, and a voxel is salt under heave exactly where the body
/// holds it. Kills deciding salt at the source column. (The staircase
/// fragments on the salt flanks of `seed1_inline` are sediment voxels whose
/// source lies inside the body, where the salt-dragged horizons are
/// normally hidden; see `docs/faults-port.md`.)
#[test]
fn salt_is_stationary_under_heave() {
    let (mut tested, mut moved) = (0, 0usize);
    for seed in [1u64, 7, 13] {
        let body = |legacy: bool| {
            let mut c = cfg(seed, [64, 64, 128], legacy);
            c.rock_physics.salt = true;
            c.with_geometry_cache()
        };
        let (c, cl) = (body(false), body(true));
        let (Some(s), Some(sl)) = (
            synthoseis_core::salt::salt_body(&c),
            synthoseis_core::salt::salt_body(&cl),
        ) else {
            continue;
        };
        assert_eq!(s.runs, sl.runs, "seed {seed}: salt body moved");
        let (labels, shape) = generate_labels(&c);
        let [ni, nj, nk] = shape;
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
            panic!("rpm")
        };
        assert!(m.heave.is_some());
        let mut scratch = synthoseis_core::rock_physics::ColumnScratch::default();
        let (mut rho, mut vp, mut vs) = (vec![0f32; nk], vec![0f32; nk], vec![0f32; nk]);
        let mut n = 0usize;
        for i in 0..ni {
            for j in 0..nj {
                let q = i * nj + j;
                let col = &labels[q * nk..(q + 1) * nk];
                m.column(i, j, col, &mut scratch, &mut rho, &mut vp, &mut vs);
                for k in 0..nk {
                    let is_salt = format!("{:?}", scratch.kinds[k]) == "Salt";
                    assert_eq!(is_salt, s.contains(q, k), "seed {seed} ({i},{j},{k})");
                    n += is_salt as usize;
                    let (x, y) = scratch.src_ij[k];
                    moved += (is_salt && ((x - i as f64).abs() > 0.5 || (y - j as f64).abs() > 0.5))
                        as usize;
                }
            }
        }
        assert!(n > 1000, "seed {seed}: {n} salt voxels");
        tested += 1;
    }
    assert!(tested >= 2, "{tested} seeds with salt");
    assert!(moved > 100, "{moved} heaved salt voxels: test is vacuous");
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (r.floor() as usize, r.ceil() as usize);
    v[lo] + (r - lo as f64) * (v[hi] - v[lo])
}

/// Spec §2 / §6.4 figures (review S6): heave at peak throw over the surface
/// voxels of seeds 1-20 at 128×128×256 with 4 faults is p50 4.9, p95 11.3,
/// max 14.5 bins after the 30° floor (± 0.1). Slow: `--ignored`.
#[test]
#[ignore]
fn heave_at_peak_throw_matches_the_spec() {
    use synthoseis_geo::faults::{heave_at_peak_throw, FaultKinematics};
    let mut h = Vec::new();
    let mut faults = 0;
    for seed in 1..=20 {
        let mut c = cfg(seed, [128, 128, 256], false);
        c.rock_physics = E2eConfig::default().rock_physics;
        // The spec §2 figures are for the 7a (Gaussian) throw; the 7b tip
        // taper gives p5/50/95/max 1.19/4.25/11.12/14.47 bins.
        c.faults.legacy_profile = true;
        let m = fault_model(&c).expect("faults");
        let FaultKinematics::Heave(p) = m.kinematics() else {
            panic!("heave")
        };
        faults += m.faults().len();
        h.extend(heave_at_peak_throw(&m, &p));
    }
    let (p50, p95, max) = (pct(&mut h, 50.0), pct(&mut h, 95.0), pct(&mut h, 100.0));
    eprintln!(
        "faults {faults} voxels {} p5 {:.2} p50 {p50:.2} p95 {p95:.2} max {max:.2}",
        h.len(),
        pct(&mut h, 5.0)
    );
    for (got, want) in [(p50, 4.9), (p95, 11.3), (max, 14.5)] {
        assert!(
            (got - want).abs() <= 0.1,
            "heave at peak throw {got:.2} vs spec {want}"
        );
    }
}

/// Salt-flank fragments (review S3 follow-up): non-salt voxels whose
/// source position (nearest voxel of the gathered `(i, j, k)`; legacy: the
/// vertical lookup) lies inside the stationary salt body.
fn salt_fragments(seed: u64, shape: [usize; 3], legacy: bool) -> Option<usize> {
    let mut c = cfg(seed, shape, legacy);
    c.rock_physics.salt = true;
    let c = c.with_geometry_cache();
    let s = synthoseis_core::salt::salt_body(&c)?;
    let m = fault_model(&c)?;
    let [ni, nj, nk] = shape;
    let t = m.compute_tile(0, ni, 0, nj);
    let near = |x: f32, n: usize| (x.round().max(0.0) as usize).min(n - 1);
    let mut n = 0;
    for i in 0..ni {
        for j in 0..nj {
            let q = i * nj + j;
            for k in 0..nk {
                let v = q * nk + k;
                if s.contains(q, k) {
                    continue;
                }
                let (x, y) = t.src_ij.as_ref().map_or((i as f32, j as f32), |p| p[v]);
                let sq = near(x, ni) * nj + near(y, nj);
                n += s.contains(sq, near(t.lookup[v], nk)) as usize;
            }
        }
    }
    Some(n)
}

/// The salt-flank fragments stay a small fraction of the cube and within
/// 1.6× of the legacy vertical lookup's (seeds 1, 7, 13 at 128×128×256;
/// numbers in `docs/faults-port.md`). Guards against the fragments growing
/// until the drag-before-faulting follow-up lands.
#[test]
fn salt_flank_fragments_stay_bounded() {
    let shape = [128, 128, 256];
    let cube = (shape[0] * shape[1] * shape[2]) as f64;
    let mut tested = 0;
    for seed in [1u64, 7, 13] {
        let (Some(h), Some(l)) = (
            salt_fragments(seed, shape, false),
            salt_fragments(seed, shape, true),
        ) else {
            continue;
        };
        eprintln!("seed {seed}: heave {h} legacy {l}");
        assert!(
            (h as f64) <= 0.025 * cube,
            "seed {seed}: {h} fragment voxels"
        );
        assert!(
            (h as f64) <= 1.6 * l as f64,
            "seed {seed}: heave {h} vs legacy {l}"
        );
        tested += 1;
    }
    assert!(tested >= 2, "{tested} seeds with salt");
}
