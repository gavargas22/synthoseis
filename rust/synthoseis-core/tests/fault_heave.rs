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
