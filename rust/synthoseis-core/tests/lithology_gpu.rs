//! GPU vs CPU on the Markov lithology. Kept in its own test binary because
//! `set_prefer_gpu` is process-global.
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, RockPhysicsConfig};
use synthoseis_core::ToyGeometry;

fn layered(seed: u64, shape: [usize; 3], rp: RockPhysicsConfig) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some([16, 16, shape[2]]),
        faults: FaultConfig::with_count(4),
        filters: FilterConfig::default(),
        rock_physics: rp,
        geometry: ToyGeometry::Layered,
    }
}

/// GPU fuse path (WGSL when an adapter exists, else the CPU fallback) vs CPU
/// on the Markov lithology.
#[test]
fn markov_gpu_matches_cpu() {
    let c = layered(
        3,
        [24, 20, 160],
        RockPhysicsConfig {
            sand_layer_fraction: Some(0.4),
            ..RockPhysicsConfig::default()
        },
    );
    let cpu = synthoseis_core::generate_reflectivity(&c, 30.0);
    synthoseis_gpu::set_prefer_gpu(true);
    let gpu = synthoseis_core::generate_reflectivity(&c, 30.0);
    synthoseis_gpu::set_prefer_gpu(false);
    let gap = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "markov rfc30 gpu vs cpu: {gap:e} ({})",
        synthoseis_gpu::backend_status()
    );
    assert!(gap < 1e-5, "{gap}");
}
