//! Bounded seismic kernels ported from `datagenerator/`.

mod filters;
mod noise;
mod wavelets;
mod zoeppritz;

pub use filters::{
    butterworth_bandpass, cumsum_traces_f32, lateral_source_range, lateral_uniform_tile,
    lateral_uniform_volume, legacy_digitisation_ms, lfilter_zi, reflect_index, uniform_window,
    FilterError, IirFilter,
};
pub use noise::{hilterman_noise_weights, snr_std_ratio};
pub use wavelets::{apply_wavelet_traces, convolve_same_1d, hanflat, ricker};
pub use zoeppritz::{compute_rfc_volumes, zoeppritz_pp};
