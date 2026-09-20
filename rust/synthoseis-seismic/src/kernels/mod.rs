//! Bounded seismic kernels ported from `datagenerator/`.

mod noise;
mod wavelets;
mod zoeppritz;

pub use noise::{hilterman_noise_weights, snr_std_ratio};
pub use wavelets::{apply_wavelet_traces, convolve_same_1d, hanflat, ricker};
pub use zoeppritz::{compute_rfc_volumes, zoeppritz_pp};
