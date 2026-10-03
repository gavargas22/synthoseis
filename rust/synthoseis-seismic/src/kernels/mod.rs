//! Bounded seismic kernels ported from `datagenerator/`.

mod filters;
mod noise;
mod subcell;
mod twt;
mod wavelets;
mod zoeppritz;

pub use filters::{
    butterworth_bandpass, cumsum_traces_f32, lateral_source_range, lateral_uniform_tile,
    lateral_uniform_volume, legacy_digitisation_ms, lfilter_zi, reflect_index, uniform_window,
    FilterError, IirFilter,
};
pub use noise::{
    hilterman_noise_weights, laplace_pair, legacy_degree_noise_weights,
    legacy_noise_mask_threshold, noise_key, noise_mask_threshold, philox4x32_10, snr_std_ratio,
    splitmix64, weighted_laplace_std, RunningStats, WeightedNoise,
};
pub use wavelets::{apply_wavelet_traces, convolve_same_1d, hanflat, ricker};
pub use zoeppritz::{
    compute_rfc_volumes, compute_rfc_volumes_form, zoeppritz_pp, zoeppritz_pp_complex,
    zoeppritz_pp_exact, zoeppritz_pp_form, ZoeppritzForm,
};
pub use subcell::{subcell_column, subcell_reflectivity, SubLayer, SubcellColumn};
pub use twt::{
    default_twt_samples, depth_staircase_ok, insert_spikes, linear_split, output_nyquist_ok,
    point_sample_labels, reflectivity_time_column, reflectivity_time_column_with_twt, twt_column, KaiserSinc, TwtKernel, TwtScratch,
    SINC_HALF_WIDTH, SINC_KAISER_BETA, SINC_PHASES, TWT_REFERENCE_VELOCITY_M_S,
};
