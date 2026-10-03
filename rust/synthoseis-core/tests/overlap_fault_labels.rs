//! The overlapped writer (`run_e2e_streaming_overlapped`, library only)
//! writes depth-axis `data/fault_labels` equal to the classic (`run_e2e`) and
//! streaming (`run_e2e_streaming`) stores, salt-masked like them
//! (`fault AND NOT salt`, or unmasked with `fault_labels_through_salt`).
//! `FaultConfig::overlap_legacy_no_fault_labels` reproduces master
//! (f3720fb2 / d51ab237: no depth-axis `fault_labels` array) byte for byte.
//! In time mode (#39) the overlapped writer keeps #39's output-domain labels;
//! the depth pass is guarded off.
use std::path::Path;

use synthoseis_core::pipeline::{run_e2e, E2eConfig, FaultConfig, RockPhysicsConfig};
use synthoseis_core::{
    generate_fault_labels, generate_fault_labels_output, run_e2e_streaming,
    run_e2e_streaming_overlapped, TimeConfig, ToyGeometry,
};
use synthoseis_io::MdioStore;
use tempfile::tempdir;

fn case(
    geometry: ToyGeometry,
    seed: u64,
    shape: [usize; 3],
    faults: usize,
    sand: Option<f64>,
) -> E2eConfig {
    E2eConfig {
        // Depth axis: these cases compare against depth-domain labels and the
        // master hashes (time mode has its own test below).
        time: TimeConfig::legacy(),
        geometry,
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(faults),
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: sand,
            ..RockPhysicsConfig::default()
        },
        ..E2eConfig::default()
    }
}

fn faults_of(p: &Path) -> Option<Vec<u8>> {
    MdioStore::open(p).unwrap().read_fault_labels_u8().ok()
}

fn count(v: &[u8]) -> usize {
    v.iter().map(|&x| x as usize).sum()
}

/// FNV-1a over every file of a store (sorted relative paths and contents),
/// ignoring the `"created"` timestamp lines of `.zattrs` / `.zmetadata`.
fn store_dir_hash(root: &Path) -> u64 {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    files.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for f in files {
        let rel = f.strip_prefix(root).unwrap().to_string_lossy().into_owned();
        eat(rel.as_bytes());
        let bytes = std::fs::read(&f).unwrap();
        if rel.ends_with(".zattrs") || rel.ends_with(".zmetadata") {
            let text = String::from_utf8(bytes).unwrap();
            for line in text.lines().filter(|l| !l.contains("\"created\"")) {
                eat(line.as_bytes());
            }
        } else {
            eat(&bytes);
        }
    }
    h
}

/// Overlap fault labels equal the classic and streaming stores (and the
/// library reference) across chunk shapes: fault-only (planar), faults +
/// salt (seed 4, 24x24x128) and seed 7, masked and through the salt.
#[test]
fn overlap_fault_labels_match_classic_and_streaming() {
    let dir = tempdir().unwrap();
    // (name, config, chunk shapes, fault voxels, fault voxels inside salt
    // with `fault_labels_through_salt`)
    let cases = [
        (
            "fault_only",
            case(ToyGeometry::Planar, 7, [24, 20, 64], 3, None),
            0usize,
        ),
        (
            "salt_seed4",
            case(ToyGeometry::Layered, 4, [24, 24, 128], 4, Some(0.4)),
            1,
        ),
        (
            "salt_seed7",
            case(ToyGeometry::Layered, 7, [32, 32, 128], 4, Some(0.5)),
            1,
        ),
    ];
    for (name, base, needs_salt) in cases {
        let nk = base.samples;
        for through in [false, true] {
            if through && needs_salt == 0 {
                continue;
            }
            let mut base = base.clone();
            base.rock_physics.fault_labels_through_salt = through;
            let reference = generate_fault_labels(&base).expect("faults");
            assert!(count(&reference) > 0, "{name}: no fault voxels");
            if needs_salt > 0 {
                let salt = synthoseis_core::salt::generate_salt_labels(&base).unwrap();
                let inside = reference
                    .iter()
                    .zip(&salt)
                    .filter(|(a, b)| **a & **b == 1)
                    .count();
                assert_eq!(
                    inside > 0,
                    through,
                    "{name}: fault voxels inside salt (through={through})"
                );
            } else {
                assert!(!base.effective_salt());
            }
            let classic = dir.path().join(format!("{name}-{through}-classic.mdio"));
            run_e2e(&E2eConfig {
                store_path: Some(classic.clone()),
                ..base.clone()
            })
            .unwrap();
            assert_eq!(
                faults_of(&classic).as_ref(),
                Some(&reference),
                "{name}: classic"
            );
            for chunks in [[8, 5, 16], [5, 7, nk], [24, 24, nk]] {
                let with = |p: &Path| E2eConfig {
                    store_path: Some(p.to_path_buf()),
                    chunk_shape: Some(chunks),
                    ..base.clone()
                };
                let s = dir
                    .path()
                    .join(format!("{name}-{through}-{chunks:?}-s.mdio"));
                run_e2e_streaming(&with(&s)).unwrap();
                let o = dir
                    .path()
                    .join(format!("{name}-{through}-{chunks:?}-o.mdio"));
                run_e2e_streaming_overlapped(&with(&o)).unwrap();
                assert_eq!(
                    faults_of(&s).as_ref(),
                    Some(&reference),
                    "{name} {chunks:?}: streaming"
                );
                assert_eq!(
                    faults_of(&o).as_ref(),
                    Some(&reference),
                    "{name} {chunks:?}: overlap"
                );
            }
        }
    }
}

/// Time mode (default since #39): the overlapped writer's `fault_labels`
/// are #39's masked output-domain (time) labels, equal to the classic and
/// streaming stores. The depth-mode pass of this PR is guarded off in time
/// mode; without the guard the depth tiles would overwrite the time labels
/// and this test fails.
#[test]
fn overlap_fault_labels_time_mode_match_classic_and_streaming() {
    let dir = tempdir().unwrap();
    let base = E2eConfig {
        time: TimeConfig::default(),
        ..case(ToyGeometry::Layered, 7, [24, 24, 128], 4, Some(0.5))
    };
    assert!(base.time_enabled() && base.effective_salt());
    assert!(!base.rock_physics.fault_labels_through_salt);
    let reference = generate_fault_labels_output(&base).expect("faults");
    assert!(count(&reference) > 0, "no time-domain fault voxels");
    // The depth labels differ from the time labels, so a depth overwrite
    // would be caught.
    let depth = generate_fault_labels(&E2eConfig {
        time: TimeConfig::legacy(),
        ..base.clone()
    })
    .expect("faults");
    assert_ne!(depth, reference, "depth and time labels coincide");
    let classic = dir.path().join("time-classic.mdio");
    run_e2e(&E2eConfig {
        store_path: Some(classic.clone()),
        ..base.clone()
    })
    .unwrap();
    assert_eq!(faults_of(&classic).as_ref(), Some(&reference), "classic");
    let nt = base.time_axis().expect("time axis").nt;
    for chunks in [[8, 5, 16], [5, 7, nt], [24, 24, nt]] {
        let with = |p: &Path| E2eConfig {
            store_path: Some(p.to_path_buf()),
            chunk_shape: Some(chunks),
            ..base.clone()
        };
        let s = dir.path().join(format!("time-{chunks:?}-s.mdio"));
        run_e2e_streaming(&with(&s)).unwrap();
        let o = dir.path().join(format!("time-{chunks:?}-o.mdio"));
        run_e2e_streaming_overlapped(&with(&o)).unwrap();
        assert_eq!(
            faults_of(&s).as_ref(),
            Some(&reference),
            "{chunks:?}: streaming"
        );
        assert_eq!(
            faults_of(&o).as_ref(),
            Some(&reference),
            "{chunks:?}: overlap"
        );
    }
}

/// `overlap_legacy_no_fault_labels` reproduces master overlap output: on
/// the depth axis (`TimeConfig::legacy()`) no `fault_labels` array, and the
/// whole store is byte-identical (except the creation timestamp) to the store
/// written by a master build with the same config. The hashes were recorded
/// with `store_dir_hash` on f3720fb2 and are unchanged on d51ab237 (#39 left
/// legacy depth-mode overlap output alone). The default differs only by the
/// new `fault_labels` array. In time mode the switch is a no-op: #39 writes
/// the output-domain labels and the store equals d51ab237 either way.
#[test]
fn overlap_legacy_no_fault_labels_reproduces_master() {
    let dir = tempdir().unwrap();
    for (name, base, master) in [
        (
            "salt_seed4",
            case(ToyGeometry::Layered, 4, [24, 24, 128], 4, Some(0.4)),
            MASTER_SALT_SEED4,
        ),
        (
            "fault_only",
            case(ToyGeometry::Planar, 7, [24, 20, 64], 3, None),
            MASTER_FAULT_ONLY,
        ),
    ] {
        let with = |p: &Path, legacy: bool| E2eConfig {
            store_path: Some(p.to_path_buf()),
            chunk_shape: Some([8, 5, 16]),
            faults: FaultConfig {
                overlap_legacy_no_fault_labels: legacy,
                ..base.faults.clone()
            },
            ..base.clone()
        };
        let legacy = dir.path().join(format!("{name}-legacy.mdio"));
        run_e2e_streaming_overlapped(&with(&legacy, true)).unwrap();
        assert!(!legacy.join("data").join("fault_labels").exists(), "{name}");
        assert_eq!(store_dir_hash(&legacy), master, "{name}: legacy vs master");
        let new = dir.path().join(format!("{name}-new.mdio"));
        run_e2e_streaming_overlapped(&with(&new, false)).unwrap();
        assert!(new.join("data").join("fault_labels").is_dir(), "{name}");
        let (a, b) = (
            MdioStore::open(&legacy).unwrap(),
            MdioStore::open(&new).unwrap(),
        );
        assert_eq!(a.read_volume().unwrap(), b.read_volume().unwrap());
        assert_eq!(a.read_labels_u8().unwrap(), b.read_labels_u8().unwrap());
        assert_ne!(store_dir_hash(&new), master);
    }
    // Time mode: identical to master d51ab237 with or without the switch.
    let base = E2eConfig {
        time: TimeConfig::default(),
        ..case(ToyGeometry::Layered, 4, [24, 24, 128], 4, Some(0.4))
    };
    for legacy in [false, true] {
        let p = dir.path().join(format!("time-{legacy}.mdio"));
        let mut c = base.clone();
        c.store_path = Some(p.clone());
        c.chunk_shape = Some([8, 5, 16]);
        c.faults.overlap_legacy_no_fault_labels = legacy;
        run_e2e_streaming_overlapped(&c).unwrap();
        assert_eq!(
            store_dir_hash(&p),
            MASTER_D51AB237_TIME_SEED4,
            "time mode legacy={legacy} vs master d51ab237"
        );
    }
}

// `store_dir_hash` of the overlapped stores written by master builds (chunks
// [8, 5, 16]). Depth axis: identical on f3720fb2 and d51ab237.
const MASTER_SALT_SEED4: u64 = 0x0e16_90c2_9b5c_3cfc;
const MASTER_FAULT_ONLY: u64 = 0x7b1d_3001_07f2_6899;
// Time mode (default since #39), d51ab237.
const MASTER_D51AB237_TIME_SEED4: u64 = 0x0e5d_3ca0_a7bb_ba0a;
