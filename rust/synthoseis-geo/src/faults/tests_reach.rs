//! Unit tests for the vertical-reach modes (`ReachMode`).

use super::*;

/// Worst legacy draw: sigma = 300, p = 1.5, throw just under 35 samples.
fn worst_reach() -> f64 {
    vertical_reach(34.99, 300.0, 1.5)
}

#[test]
fn worst_case_reach_is_about_577_samples() {
    let r = worst_reach();
    assert!((576.0..578.0).contains(&r), "{r}");
    // Reach grows with throw and sigma and shrinks with p.
    assert!(vertical_reach(10.0, 100.0, 2.0) < vertical_reach(20.0, 100.0, 2.0));
    assert!(vertical_reach(10.0, 100.0, 2.0) < vertical_reach(10.0, 150.0, 2.0));
    assert!(vertical_reach(10.0, 100.0, 3.0) < vertical_reach(10.0, 100.0, 2.0));
    assert_eq!(vertical_reach(1.0, 100.0, 2.0), 0.0);
}

#[test]
fn legacy_taper_succeeds_on_tall_columns() {
    // Extreme draws on a 640-sample column with the seabed at 4 samples.
    for &(throw, sigma, p) in &[(34.99, 300.0, 1.5), (29.0, 300.0, 5.0), (5.0, 1.0, 1.5)] {
        for center_k in [5usize, 100, 320, 639] {
            let t = taper(640, throw, sigma, p, center_k, 4.0);
            assert!(t.ok, "throw {throw} sigma {sigma} p {p} centre {center_k}");
        }
    }
}

#[test]
fn fit_column_is_legacy_when_taper_succeeds() {
    for &(nk, throw, sigma, p, c, wb) in &[
        (640usize, 20.0, 250.0, 2.0, 300usize, 4.0),
        (64, 10.0, 3.0, 2.0, 30, 2.0),
        (700, 29.0, 300.0, 1.5, 50, 10.5),
    ] {
        let leg = vertical_profile_mode(nk, throw, sigma, p, c, wb, ReachMode::Legacy);
        let fit = vertical_profile_mode(nk, throw, sigma, p, c, wb, ReachMode::FitColumn);
        assert!(leg.seabed_ok);
        assert_eq!(leg, fit);
        assert!(!fit.rescued);
        let (prof, roll) = vertical_profile(nk, throw, sigma, p, c, wb);
        assert_eq!((prof, roll), (leg.profile, leg.roll));
    }
}

#[test]
fn fit_column_rescues_short_columns() {
    // The legacy loop gives up on a 64-sample column for sigma = 250.
    let (nk, throw, sigma, p, c, wb) = (64usize, 20.0, 250.0, 2.0, 30usize, 4.0);
    let leg = vertical_profile_mode(nk, throw, sigma, p, c, wb, ReachMode::Legacy);
    assert!(!leg.seabed_ok);
    let fit = vertical_profile_mode(nk, throw, sigma, p, c, wb, ReachMode::FitColumn);
    assert!(fit.seabed_ok && fit.rescued);
    assert!(fit.sigma < sigma);
    assert!(fit.sigma <= fit_sigma_to_column(nk, throw, p, c, wb).unwrap());
    // Throw at and above the seabed is <= 1; peak throw is still reached
    // somewhere below it (or rolled below the cube bottom).
    assert!(fit.profile[..=wb as usize].iter().all(|&v| v <= 1.0));
    assert!(fit.profile.iter().all(|&v| v <= throw + 1e-9));
    // The rescue is deterministic.
    let again = vertical_profile_mode(nk, throw, sigma, p, c, wb, ReachMode::FitColumn);
    assert_eq!(fit, again);
}

#[test]
fn fit_sigma_rule() {
    // d_max = (c - 1.5 - wb) + 5 * floor((nk - wb) / 5)
    let s = fit_sigma_to_column(64, 20.0, 2.0, 30, 4.0).unwrap();
    let d_max = (30.0 - 1.5 - 4.0) + 60.0;
    let want = 0.98 * d_max / (2.0 * 20f64.ln()).powf(0.25);
    assert!((s - want).abs() < 1e-12);
    assert!((vertical_reach(20.0, s, 2.0) - 0.98 * d_max).abs() < 1e-9);
    assert!(fit_sigma_to_column(64, 1.0, 2.0, 30, 4.0).is_none());
    assert!(fit_sigma_to_column(64, 20.0, 2.0, 0, 70.0).is_none());
}

#[test]
fn argmax_matches_numpy_first_index_on_plateaus() {
    // sigma = 280, p = 4.69: g has a plateau of samples exactly == throw.
    // np.roll by `centre + g.argmax() + roll_int` (argmax = FIRST plateau
    // index) puts the plateau's last sample at `centre - 1`, like Python.
    // (Using the analytic centre instead would push it 5 samples deeper.)
    let throw = 29.41483010465063;
    let t = taper(640, throw, 280.27642083952276, 4.689747331569834, 626, 4.0);
    assert!(t.ok && t.roll == 0);
    let on: Vec<usize> = (0..640).filter(|&k| t.profile[k] == throw).collect();
    assert!(on.len() > 1, "plateau expected");
    assert_eq!(*on.last().unwrap(), 625);
    // Non-plateau case: unique max, still at centre - 1.
    let t = taper(640, 20.0, 50.0, 1.5, 300, 4.0);
    let peak = (0..640)
        .max_by(|&a, &b| t.profile[a].total_cmp(&t.profile[b]))
        .unwrap();
    assert_eq!(peak, 299 + t.roll);
}
