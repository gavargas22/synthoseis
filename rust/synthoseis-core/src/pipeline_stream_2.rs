/// Fuse one spatial tile: elastic props → Zoeppritz RFC → wavelet convolution.
///
/// Delegates to [`synthoseis_gpu::fuse_tile_dispatch`] so `--gpu` / prefer-gpu
/// can select the auto path. Default remains the CPU software adapter
/// (bit-identical to the historical inline implementation).
pub fn fuse_tile_local(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
    stats: &mut WorkingSetStats,
) {
    let nk = shape[2];
    stats.observe(synthoseis_gpu::fuse_tile_scratch_bytes(nk));
    let _backend = synthoseis_gpu::fuse_tile_dispatch(
        labels,
        shape,
        i0,
        i1,
        j0,
        j1,
        trends,
        wavelet,
        angle_deg,
        tile_out,
    );
}
