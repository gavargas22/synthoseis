//! SNR / Hilterman noise-weight helpers from `Seismic.add_weighted_noise`
//! and `tests/test_seismic_noise.py`.

/// `std_ratio = sqrt(10 ** (sn_db / 10))` used to scale noise vs signal std.
pub fn snr_std_ratio(sn_db: f64) -> f64 {
    (10.0_f64).powf(sn_db / 10.0).sqrt()
}

/// Hilterman near/far weights for an incident angle in **degrees**.
///
/// Returns `(cos²(θ), sin²(θ))` with θ in radians — see
/// `tests/test_seismic_noise.py`.
pub fn hilterman_noise_weights(angle_deg: f64) -> (f64, f64) {
    let th = angle_deg.to_radians();
    let c = th.cos();
    let s = th.sin();
    (c * c, s * s)
}
