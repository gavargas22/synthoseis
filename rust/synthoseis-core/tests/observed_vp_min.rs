//! The per-run records of #54 (the lowest sediment Vp of the end-of-run
//! depth-staircase warning and the NaN-guard count) are reset at every run
//! entry, so a process that runs several models reports each run on its
//! own. This file holds a single test because the records are
//! process-global.

use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::{run_e2e_chunked, RockPhysicsConfig};

/// `slow`: higher Athy surface porosity for both lithologies, so every
/// end-member, and the run's minimum, is slower.
fn cfg(slow: bool) -> E2eConfig {
    let mut rock_physics = RockPhysicsConfig::default();
    if slow {
        rock_physics.phys.sand.phi0 = 0.6;
        rock_physics.phys.shale.phi0 = 0.75;
    }
    E2eConfig {
        seed: 7,
        inline_count: 12,
        crossline_count: 10,
        samples: 64,
        rock_physics,
        ..E2eConfig::default()
    }
}

fn run_min(c: &E2eConfig) -> f32 {
    run_e2e_chunked(c).expect("run");
    synthoseis_rpm::observed_vp_min().expect("layer voxels were evaluated")
}

#[test]
fn second_run_records_are_independent_of_the_first() {
    let fast = cfg(false);
    let slow = cfg(true);
    let fast_alone = run_min(&fast);
    let slow_min = run_min(&slow);
    // The slow model's minimum is lower, so carrying its minimum over would
    // show in the next fast run.
    assert!(slow_min < fast_alone, "slow {slow_min} vs fast {fast_alone}");
    let fast_after = run_min(&fast);
    assert_eq!(fast_after.to_bits(), fast_alone.to_bits(), "fast after slow {fast_after} vs alone {fast_alone}");
    // And the warning text follows the run's own minimum.
    let mut b2 = fast.clone();
    b2.time.dt_ms = 2.0;
    run_e2e_chunked(&b2).expect("run");
    let w = b2.time_warning_after_run().expect("dz 4 m at dt 2 ms warns");
    assert!(w.contains(&format!("Vp_min {} m/s", (fast_alone as f64).floor())), "{w}");

    // NaN-guard count: the default model never guards, so after a run the
    // count is 0 even if an earlier run in this process left guards behind
    // (simulated by bumping the counter as a guarded run would).
    assert_eq!(synthoseis_core::guarded_voxels(), 0);
    synthoseis_rpm::GUARDED_VOXELS.fetch_add(5, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(synthoseis_core::guarded_voxels(), 5);
    run_e2e_chunked(&fast).expect("run");
    assert_eq!(synthoseis_core::guarded_voxels(), 0, "guarded count carried over from an earlier run");
}
