//! Salt pv-maps spec §8.3–8.6: the continuous salt partial-voxel maps
//! (default) against `--no-salt` far from salt, against the rounded legacy
//! maps (`salt_legacy_pv_maps`), depth-label identity and the partial-voxel
//! counters. PR run: 32×32×128 seed 7; nightly: the 64×64×256 demo.
use synthoseis_core::partial_model::partial_voxel_summary;
use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::{
    elastic_model, generate_angle_stack_from_labels, generate_labels, generate_output_labels,
    FaultConfig, TimeConfig,
};

fn demo(seed: u64, shape: [usize; 3], faults: usize) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::legacy_with_count(faults),
        ..E2eConfig::default()
    }
}

fn variant(c: &E2eConfig, v: &str) -> E2eConfig {
    let mut c = c.clone();
    match v {
        "default" => {}
        "rounded" => c.rock_physics.salt_legacy_pv_maps = true,
        "no-salt" => c.rock_physics.salt = false,
        _ => unreachable!(),
    }
    c
}

/// `true` for columns more than 8 columns (Chebyshev) from the salt body
/// footprint (`SaltBody::runs`, including the padded grid).
fn far_columns(c: &E2eConfig) -> Vec<bool> {
    let body = synthoseis_core::salt::salt_body(c).expect("salt body");
    let [ni, nj, _] = c.shape();
    let foot: Vec<(usize, usize)> = (0..ni * nj)
        .filter(|&q| body.runs[q].1 > body.runs[q].0)
        .map(|q| (q / nj, q % nj))
        .collect();
    (0..ni * nj)
        .map(|q| {
            let (i, j) = (q / nj, q % nj);
            foot.iter().all(|&(a, b)| i.abs_diff(a).max(j.abs_diff(b)) > 8)
        })
        .collect()
}

struct Out {
    stack: Vec<f32>,
    time_labels: Vec<u8>,
    depth_labels: Vec<u8>,
    nt: usize,
}

fn render(c: &E2eConfig) -> Out {
    let (labels, shape) = generate_labels(c);
    let (vols, _) = generate_angle_stack_from_labels(c, &labels, shape, 15.0);
    let ol = generate_output_labels(c, &labels, &elastic_model(c, &labels, shape));
    Out { nt: vols.shape[2], stack: vols.angle_stack, time_labels: ol.labels, depth_labels: labels }
}

/// (changed cells %, rel. RMS) of `a` vs reference `b` over the selected
/// columns.
fn diff(a: &[f32], b: &[f32], nt: usize, cols: &[bool]) -> (f64, f64) {
    let (mut n, mut ch, mut e, mut r) = (0usize, 0usize, 0.0f64, 0.0f64);
    for (col, &keep) in cols.iter().enumerate() {
        if !keep {
            continue;
        }
        for t in col * nt..(col + 1) * nt {
            let d = a[t] as f64 - b[t] as f64;
            n += 1;
            ch += usize::from(a[t].to_bits() != b[t].to_bits());
            e += d * d;
            r += (b[t] as f64).powi(2);
        }
    }
    (100.0 * ch as f64 / n.max(1) as f64, (e / r.max(f64::MIN_POSITIVE)).sqrt())
}

fn label_changes(a: &[u8], b: &[u8], nt: usize, cols: &[bool]) -> usize {
    cols.iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(col, _)| (col * nt..(col + 1) * nt).filter(|&t| a[t] != b[t]).count())
        .sum()
}

/// §8.3 far-field gate vs `--no-salt`, §8.5 output-change gate vs the
/// rounded maps, on one cube; returns the measured numbers.
fn gates(c: &E2eConfig, far_rms_max: f64, far_labels_max: usize, check_change: bool) {
    let d = render(&variant(c, "default"));
    let r = render(&variant(c, "rounded"));
    let n = render(&variant(c, "no-salt"));
    let far = far_columns(c);
    let all = vec![true; far.len()];
    let nt = d.nt;
    let (_, far_today) = diff(&r.stack, &n.stack, nt, &far);
    let (_, far_fix) = diff(&d.stack, &n.stack, nt, &far);
    let far_lab_today = label_changes(&r.time_labels, &n.time_labels, nt, &far);
    let far_lab_fix = label_changes(&d.time_labels, &n.time_labels, nt, &far);
    let (chg, rms) = diff(&d.stack, &r.stack, nt, &all);
    let lab = label_changes(&d.time_labels, &r.time_labels, nt, &all);
    let lab_pct = 100.0 * lab as f64 / d.time_labels.len() as f64;
    println!(
        "seed {} {:?}: far columns {}/{}; far 15° rel. RMS vs --no-salt {far_today:.4} -> {far_fix:.4}; far time labels vs --no-salt {far_lab_today} -> {far_lab_fix}; default vs rounded: {chg:.1} % cells, rel. RMS {rms:.3}, time labels {lab} ({lab_pct:.2} %)",
        c.seed,
        c.shape(),
        far.iter().filter(|v| **v).count(),
        far.len()
    );
    assert!(far.iter().any(|v| *v), "no far columns");
    assert!(far_fix <= far_rms_max, "far-field rel. RMS {far_fix}");
    assert!(far_lab_fix <= far_labels_max, "far-field time labels {far_lab_fix}");
    assert!(far_fix < far_today, "the fix must reduce the far-field error");
    if check_change {
        assert!(chg >= 50.0, "output change {chg} % (no-op?)");
        assert!((0.3..=0.6).contains(&rms), "output change rel. RMS {rms}");
        assert!((0.5..=4.0).contains(&lab_pct), "time labels changed {lab_pct} %");
    }
    // Depth labels never move.
    assert_eq!(d.depth_labels, r.depth_labels, "depth labels");
}

/// §8.3 (PR run) + §8.5: 32×32×128 seed 7, 3 faults.
#[test]
fn far_field_and_output_change_gates_32() {
    gates(&demo(7, [32, 32, 128], 3), 0.03, 10, true);
}

/// §8.3 nightly: the 64×64×256 demo (seed 7, 4 faults): far-field rel.
/// RMS ≤ 0.005 and 0 far time-label changes vs `--no-salt`.
#[test]
#[ignore = "nightly: salt pv-maps far-field gate on the 64x64x256 demo"]
fn far_field_gate_demo_64() {
    let mut c = demo(7, [64, 64, 256], 4);
    c.rock_physics.legacy_rock_physics = true;
    c.rock_physics.legacy_fold_geometry = true;
    gates(&c, 0.005, 0, false);
}

/// [`far_field_gate_demo_64`] under the physical rock physics (default):
/// the burial-depth dependent end-members move 3 far time labels vs
/// `--no-salt` (of 2466 far columns x 256 samples), rel. RMS 0.0002. The
/// 0-label gate above holds under `--legacy-rock-physics`.
///
/// Folded default (folds review, dip limit 2 samples/column): the salt now
/// walls closure compartments that reach the far columns, so with fluids
/// on the far field differs from `--no-salt` by rel. RMS 0.0205 and 70
/// time labels, all of it fluid fill: with `fluids = false` it is rel. RMS
/// 0.0001 and 0 labels (fluids on at the old limit 1: also 0.0001 and 0).
/// The pv-maps gate is therefore checked with fluids off.
#[test]
#[ignore = "nightly: salt pv-maps far-field gate on the 64x64x256 demo, physical rock physics"]
fn far_field_gate_demo_64_physical() {
    let mut c = demo(7, [64, 64, 256], 4);
    c.rock_physics.fluids = false;
    gates(&c, 0.005, 3, false);
}

/// §8.4 depth-label identity: default vs rounded maps, seeds 7 and 1,
/// faults 0 and 3, time mode and the legacy axis: depth labels and the
/// legacy-axis output labels / `salt_labels` / `fault_labels` identical.
#[test]
fn depth_labels_identical_to_rounded_maps() {
    for seed in [7u64, 1] {
        for faults in [0usize, 3] {
            for legacy_axis in [false, true] {
                let mut c = demo(seed, [24, 24, 96], faults);
                if legacy_axis {
                    c.time = TimeConfig::legacy();
                    c.rock_physics.partial_voxels = synthoseis_core::partial_voxels::PartialVoxelConfig::with(
                        synthoseis_core::partial_voxels::PvReflectivity::Cell,
                    );
                }
                let d = variant(&c, "default");
                let r = variant(&c, "rounded");
                let (ld, sh) = generate_labels(&d);
                let (lr, _) = generate_labels(&r);
                assert_eq!(ld, lr, "seed {seed} faults {faults}: depth labels");
                if legacy_axis {
                    let a = generate_output_labels(&d, &ld, &elastic_model(&d, &ld, sh));
                    let b = generate_output_labels(&r, &lr, &elastic_model(&r, &lr, sh));
                    assert_eq!(a.labels, b.labels, "seed {seed}: legacy-axis labels");
                    assert_eq!(a.faults, b.faults, "seed {seed}: legacy-axis fault labels");
                    assert_eq!(a.salt, b.salt, "seed {seed}: legacy-axis salt labels");
                }
            }
        }
    }
}

/// §8.6 counter sanity on the 32² demo: the mixed-cell share is within 1
/// point of `--no-salt`, the label guard stays < 1 % of sediment cells, and
/// the rounded maps read lower (partial voxels effectively off).
#[test]
fn partial_voxel_counters_return_to_no_salt_level() {
    let c = demo(7, [32, 32, 128], 3);
    let pct = |v: &str| {
        let s = partial_voxel_summary(&variant(&c, v)).unwrap().stats;
        (100.0 * s.mixed as f64 / s.cells as f64, 100.0 * s.label_guard as f64 / s.sediment as f64)
    };
    let (d, dg) = pct("default");
    let (r, _) = pct("rounded");
    let (n, _) = pct("no-salt");
    println!("mixed cells: rounded {r:.1} %, default {d:.1} %, --no-salt {n:.1} %; label guard {dg:.3} %");
    assert!((d - n).abs() <= 1.0, "mixed {d} vs no-salt {n}");
    assert!(dg < 1.0, "label guard {dg} %");
    assert!(r < d, "rounded maps {r} >= default {d}");
}

