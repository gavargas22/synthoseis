//! Every run entry that calls `reset_run_records()` gets its own
//! slow-run-then-fast-run check: after a slow run (lower minimum Vp) and a
//! simulated guarded run (GUARDED_VOXELS bumped), a fast run through the
//! same entry must report exactly the minimum Vp and guard count it reports
//! when run alone. The records are process-global, so the tests take a lock.
//!
//! The default models never trip the NaN guard, so the guarded run is
//! simulated by bumping the counter, as a guarded run would.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use synthoseis_core::pipeline::{run_e2e, E2eConfig};
use synthoseis_core::{
    prepare_multiprocess_store, run_e2e_chunked, run_e2e_geometry_once_seismic_many,
    run_e2e_multiprocess, run_e2e_streaming, run_e2e_streaming_overlapped,
    run_e2e_strip_stitched, run_worker_partition, RockPhysicsConfig,
};

static LOCK: Mutex<()> = Mutex::new(());

fn cfg(slow: bool, store: Option<&Path>) -> E2eConfig {
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
        chunk_shape: Some([6, 5, 64]),
        rock_physics,
        store_path: store.map(|p| p.to_path_buf()),
        ..E2eConfig::default()
    }
}

/// Run `entry` fast alone, slow, (simulated guards), fast again; assert the
/// second fast run's records equal the first's.
fn check(name: &str, needs_store: bool, entry: impl Fn(&E2eConfig)) {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let mut n = 0;
    let mut run = |slow: bool| {
        n += 1;
        let p = dir.path().join(format!("{name}_{n}.mdio"));
        entry(&cfg(slow, needs_store.then_some(p.as_path())));
        (
            synthoseis_rpm::observed_vp_min().expect("layer voxels were evaluated"),
            synthoseis_core::guarded_voxels(),
        )
    };
    let (fast_alone, guards_alone) = run(false);
    let (slow_min, _) = run(true);
    assert!(slow_min < fast_alone, "{name}: slow {slow_min} vs fast {fast_alone}");
    synthoseis_rpm::GUARDED_VOXELS.fetch_add(5, Ordering::Relaxed);
    let (fast_after, guards_after) = run(false);
    let min_ok = fast_after.to_bits() == fast_alone.to_bits();
    let guards_ok = guards_after == guards_alone;
    assert!(
        min_ok && guards_ok,
        "{name}: records carried over: min Vp fast-after-slow {fast_after} vs alone {fast_alone} \
         (ok={min_ok}); GUARDED_VOXELS {guards_after} vs alone {guards_alone} (ok={guards_ok})"
    );
}

#[test]
fn run_e2e_resets() {
    check("run_e2e", true, |c| { run_e2e(c).unwrap(); });
}

#[test]
fn run_e2e_chunked_resets() {
    check("chunked", false, |c| { run_e2e_chunked(c).unwrap(); });
}

#[test]
fn run_e2e_streaming_resets() {
    check("streaming", true, |c| { run_e2e_streaming(c).unwrap(); });
}

#[test]
fn run_e2e_streaming_overlapped_resets() {
    check("overlap", true, |c| { run_e2e_streaming_overlapped(c).unwrap(); });
}

#[test]
fn run_e2e_strip_stitched_resets() {
    // workers > 1, so the strip path runs (1 worker delegates to streaming).
    check("strip", true, |c| { run_e2e_strip_stitched(c, 2).unwrap(); });
}

#[test]
fn run_e2e_geometry_once_seismic_many_resets() {
    check("geo_many", false, |c| { run_e2e_geometry_once_seismic_many(c, &[0.0, 15.0]).unwrap(); });
}

#[test]
fn run_e2e_multiprocess_resets() {
    check("mp", true, |c| { run_e2e_multiprocess(c, 2).unwrap(); });
}

#[test]
fn run_worker_partition_resets() {
    // One worker of an in-process multiprocess run, called directly.
    check("worker", true, |c| {
        let (path, plan, _) = prepare_multiprocess_store(c, 2, c.chunk_shape).unwrap();
        run_worker_partition(c, &plan, 0, &path).unwrap();
    });
}
