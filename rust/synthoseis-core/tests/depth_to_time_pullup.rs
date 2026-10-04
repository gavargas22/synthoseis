//! Salt pull-up measured at sub-sample resolution on the wavelet-free
//! time-mode reflectivity (the production time fuse with `NO_WAVELET`, not
//! the Ricker / bandpass stack written to the store), against the pull-up
//! the velocities predict (spec §5.3, demo part; Strata's review of PR B).
//! Picks on the deliverable stack agree within 0.25 ms (evidence script and
//! Strata's independent check), but this test does not use the stack.
//!
//! Seed 1, 32 × 32 × 128, default config, salt run vs `--no-salt` run.
//!
//! * Horizon: the sub-salt layer top L (below the salt base, inside both
//!   traces) with the most salt-footprint columns whose event *dominates*
//!   the raw time reflectivity of both runs: max |x| of samples n−1, n (n =
//!   first output sample of layer L in the time label cube) ≥ 2× every other
//!   |x| within ±6 samples. Amplitude QC only, no timing information; it
//!   excludes weak contrasts next to strong ones, which a peak pick would
//!   lock onto.
//! * Trace: the production time-mode fuse without a wavelet
//!   (`fuse_tile_local` with `NO_WAVELET`, default incidence).
//! * Pick: #37's salt-test pick: polarity from samples n−1, n; coarse
//!   extremum in n−2 … n+1; extremum of the 16× FFT-upsampled trace (direct
//!   DFT, independent of the insertion kernel) within ±1 sample; parabola
//!   through the three best fine samples.
//! * Measured pull-up = t_nosalt − t_salt; predicted = T_nosalt(z_L) −
//!   T_salt(z_L) from each run's Vp (cell-boundary TWT; includes the salt
//!   drag of the horizon).
//!
//! The evidence script (`scripts/depth_to_time_evidence/`) runs the same
//! measurement on the 64 × 64 × 256 cubes of seeds 1, 2, 3, 7 and 30.

use synthoseis_core::pipeline::E2eConfig;

/// The opt-out (`PartialVoxelConfig::whole_voxels`, = the d8b96e69 library
/// default): partial voxels are the library default since PR B2, and this
/// golden asserts master (whole-voxel) output.
fn whole(c: &E2eConfig) -> E2eConfig {
    let mut c = c.clone();
    c.rock_physics.partial_voxels =
        synthoseis_core::partial_voxels::PartialVoxelConfig::whole_voxels();
    c
}

const UP: usize = 16;
const DOM_MIN: f64 = 2.0;
const DOM_HALF: usize = 6;

struct Run {
    labels: Vec<u8>,
    salt: Option<Vec<u8>>,
    time_labels: Vec<u8>,
    twt: Vec<f64>,
    rfc: Vec<f32>,
    nz: usize,
    nt: usize,
    dt: f64,
}

fn run(salt: bool) -> Run {
    let mut cfg = E2eConfig {
        seed: 1,
        inline_count: 32,
        crossline_count: 32,
        samples: 128,
        ..E2eConfig::default()
    };
    // Whole voxels: the prediction is the cell-boundary TWT of the first
    // labelled cell, while partial voxels (the PR B2 default) place the
    // event at the exact sub-cell horizon time.
    cfg = whole(&cfg);
    cfg.rock_physics.salt = salt;
    assert!(cfg.time_enabled());
    let (labels, shape) = synthoseis_core::pipeline_stream::generate_labels(&cfg);
    let model = synthoseis_core::rock_physics::elastic_model(&cfg, &labels, shape);
    let axis = cfg.time_axis().unwrap();
    let [ni, nj, nz] = shape;
    let twt = synthoseis_core::time_mode::tile_twt(&model, &labels, shape, 0, ni, 0, nj, &axis);
    let mut rfc = vec![0.0f32; ni * nj * axis.nt];
    synthoseis_core::pipeline_stream::fuse_tile_local(
        &labels,
        shape,
        0,
        ni,
        0,
        nj,
        &model,
        synthoseis_gpu::NO_WAVELET,
        synthoseis_core::pipeline_stream::DEFAULT_INCIDENCE_DEG,
        &mut rfc,
        &mut synthoseis_core::pipeline_stream::WorkingSetStats::default(),
    );
    let time_labels = synthoseis_core::time_mode::generate_output_labels(&cfg, &labels, &model).labels;
    Run {
        salt: synthoseis_core::salt::generate_salt_labels(&cfg),
        labels,
        time_labels,
        twt,
        rfc,
        nz,
        nt: axis.nt,
        dt: axis.dt_ms,
    }
}

fn first(col: &[u8], l: u8) -> Option<usize> {
    col.iter().position(|&v| v == l)
}

/// 16× band-limited upsampling (spectrum zero padding, Nyquist bin split),
/// as #37's `fft_upsample_16`.
fn fft_upsample_16(y: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = y.len();
    assert!(n % 2 == 0);
    let spec: Vec<(f64, f64)> = (0..=n / 2)
        .map(|k| {
            y.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, &v)| {
                let w = -2.0 * PI * (k * t) as f64 / n as f64;
                (re + v * w.cos(), im + v * w.sin())
            })
        })
        .collect();
    (0..UP * n)
        .map(|s| {
            let u = s as f64 / UP as f64;
            let mut acc = spec[0].0;
            for (k, &(re, im)) in spec.iter().enumerate().skip(1) {
                let w = 2.0 * PI * k as f64 * u / n as f64;
                let scale = if k == n / 2 { 1.0 } else { 2.0 };
                acc += scale * (re * w.cos() - im * w.sin());
            }
            acc / n as f64
        })
        .collect()
}

/// Event dominance around label sample `n` (0 when too close to an end).
fn dominance(x: &[f32], n: usize) -> f64 {
    if n < DOM_HALF + 2 || n + DOM_HALF + 2 > x.len() {
        return 0.0;
    }
    let a = |k: usize| (x[k] as f64).abs();
    let ev = a(n - 1).max(a(n));
    let nb = (n - 1 - DOM_HALF..n - 1).chain(n + 1..n + 1 + DOM_HALF).map(a).fold(0.0, f64::max);
    ev / nb.max(1e-30)
}

/// Sub-sample pick (samples) and polarity of the event at label sample `n`.
fn pick(x: &[f32], n: usize) -> (f64, f64) {
    let y: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let p = if y[n - 1].abs() > y[n].abs() { y[n - 1].signum() } else { y[n].signum() };
    let c = (n - 2..n + 2).max_by(|&a, &b| (p * y[a]).total_cmp(&(p * y[b]))).unwrap();
    let fine = fft_upsample_16(&y);
    for (k, &v) in y.iter().enumerate() {
        assert!((fine[UP * k] - v).abs() <= 1e-9, "upsample reproduces sample {k}");
    }
    let s = (UP * (c - 1) + 1..UP * (c + 1)).max_by(|&a, &b| (p * fine[a]).total_cmp(&(p * fine[b]))).unwrap();
    let (a, b, d) = (p * fine[s - 1], p * fine[s], p * fine[s + 1]);
    ((s as f64 + 0.5 * (a - d) / (a - 2.0 * b + d)) / UP as f64, p)
}

#[test]
fn salt_pull_up_subsample_matches_velocities() {
    let (s, n) = (run(true), run(false));
    let (nz, nt, dt) = (s.nz, s.nt, s.dt);
    let salt = s.salt.as_ref().expect("seed 1 has salt");
    let cols = s.labels.len() / nz;
    // Salt footprint columns and their salt base (deepest salt cell).
    let base: Vec<Option<usize>> = (0..cols).map(|c| salt[c * nz..(c + 1) * nz].iter().rposition(|&v| v == 1)).collect();
    let max_label = *s.labels.iter().filter(|&&v| v != 255).max().unwrap();
    let qc = |l: u8, c: usize| -> Option<(usize, usize, usize, usize)> {
        let b = base[c]?;
        let zs = first(&s.labels[c * nz..(c + 1) * nz], l)?;
        let zn = first(&n.labels[c * nz..(c + 1) * nz], l)?;
        let ts = first(&s.time_labels[c * nt..(c + 1) * nt], l)?;
        let tn = first(&n.time_labels[c * nt..(c + 1) * nt], l)?;
        let d = dominance(&s.rfc[c * nt..(c + 1) * nt], ts).min(dominance(&n.rfc[c * nt..(c + 1) * nt], tn));
        (zs > b && d >= DOM_MIN).then_some((zs, zn, ts, tn))
    };
    let l = (1..=max_label).max_by_key(|&l| (0..cols).filter(|&c| qc(l, c).is_some()).count()).unwrap();
    let mut resid = Vec::new();
    let (mut pred_max, mut meas_at_max) = (0.0f64, 0.0f64);
    for c in 0..cols {
        let Some((zs, zn, ts, tn)) = qc(l, c) else { continue };
        let (ps, pol_s) = pick(&s.rfc[c * nt..(c + 1) * nt], ts);
        let (pn, pol_n) = pick(&n.rfc[c * nt..(c + 1) * nt], tn);
        if pol_s != pol_n {
            continue;
        }
        let pred = n.twt[c * (nz + 1) + zn] - s.twt[c * (nz + 1) + zs];
        let meas = (pn - ps) * dt;
        if pred > pred_max {
            (pred_max, meas_at_max) = (pred, meas);
        }
        resid.push(meas - pred);
    }
    let k = resid.len() as f64;
    let mean = resid.iter().sum::<f64>() / k;
    let std = (resid.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / k).sqrt();
    let max_abs = resid.iter().fold(0.0f64, |m, r| m.max(r.abs()));
    println!(
        "seed 1 32x32x128, horizon {l}: {} columns, max predicted pull-up {pred_max:.3} ms (measured {meas_at_max:.3} ms); residual mean {mean:.3} std {std:.3} max |.| {max_abs:.3} ms",
        resid.len()
    );
    assert!(resid.len() >= 100, "too few measurable columns: {}", resid.len());
    assert!(pred_max > 50.0, "pull-up too small to be a test: {pred_max}");
    assert!(mean.abs() <= 0.25, "residual mean {mean} ms");
    assert!(std <= 0.6, "residual std {std} ms");
    assert!(max_abs <= 1.5, "residual max |.| {max_abs} ms");
}
