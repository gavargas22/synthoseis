//! Seismic modeling ports from `datagenerator/Seismic.py`,
//! `datagenerator/wavelets.py`, and `datagenerator/zoeppritz_kernel.py`.
//!
//! # First landed kernels
//! - [`zoeppritz_pp`] / [`compute_rfc_volumes`] — Zoeppritz PP reflectivity
//! - [`ricker`] / [`hanflat`] — wavelet construction helpers
//! - [`convolve_same_1d`] / [`apply_wavelet_traces`] — trace convolution
//! - [`snr_std_ratio`] / [`hilterman_noise_weights`] — noise SNR scaling
//! - [`WeightedNoise`] / [`philox4x32_10`] / [`RunningStats`] — deterministic,
//!   counter-based replacement for legacy `add_weighted_noise`
//! - [`butterworth_bandpass`] / [`IirFilter::filtfilt_f32`] — legacy Butterworth
//!   bandpass (scipy `butter` + `filtfilt`)
//! - [`lateral_uniform_tile`] / [`lateral_uniform_volume`] — legacy lateral
//!   filter (scipy `uniform_filter`, reflect), tile-invariant with a halo
//!
//! Golden fixtures: `tests/fixtures/seismic_kernels.json`
//! (regenerate via `tests/fixtures/generate_seismic_kernels.py`).
//!
//! Parity = angle-stack MAE / max-abs via `synthoseis-core::parity`
//! (not bit-identical full seismic). CPU path only; GPU flags later.

mod kernels;
#[cfg(test)]
#[path = "tests_kernels.rs"]
mod tests_kernels;

pub use kernels::{
    apply_wavelet_traces, butterworth_bandpass, compute_rfc_volumes, convolve_same_1d,
    cumsum_traces_f32, hanflat, hilterman_noise_weights, laplace_pair, lateral_source_range,
    lateral_uniform_tile, lateral_uniform_volume, legacy_degree_noise_weights,
    legacy_digitisation_ms, legacy_noise_mask_threshold, lfilter_zi, noise_key,
    noise_mask_threshold, philox4x32_10, reflect_index, ricker, snr_std_ratio, splitmix64,
    uniform_window, weighted_laplace_std, zoeppritz_pp, FilterError, IirFilter, RunningStats,
    WeightedNoise,
};
