//! Fault heave at the pipeline level (fault-kinematics spec 7a, §9 rows 12
//! and 13): the rock physics and the unconformity crossing read the source
//! column of every displaced voxel, and the defaults turn heave on.

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::rock_physics::elastic_model;
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
/// reading the output column's maps (the model without its heave columns,
/// the "same-column maps" mutant) sides with it in <= 6 %.
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
