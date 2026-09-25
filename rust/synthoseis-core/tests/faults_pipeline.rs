//! Pipeline-level fault tests: disabled-by-default bit identity, determinism,
//! chunk/tiling/worker invariance of `fault_labels`, and MDIO round-trips.

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::{
    fault_model, generate_chunked, generate_fault_labels, generate_labels, run_e2e_multiprocess,
    run_e2e_streaming, run_e2e_strip_stitched,
};
use synthoseis_io::MdioStore;
use tempfile::tempdir;

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn cfg(seed: u64, shape: [usize; 3], chunks: [usize; 3]) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some(chunks),
        faults: FaultConfig::default(),
    }
}

/// Hashes of `generate_chunked` recorded on master 9c0614c (before the fault
/// port). With faults disabled (the default) output must stay bit-identical.
#[test]
fn faults_disabled_by_default_is_bit_identical_to_master() {
    assert_eq!(FaultConfig::default().count, 0);
    assert!(!E2eConfig::tiny(1).faults.enabled());
    type Golden = (u64, [usize; 3], [usize; 3], u64, u64);
    let golden: [Golden; 3] = [
        (
            42,
            [8, 8, 8],
            [4, 4, 8],
            0x0f2b_c73c_ab3b_0874,
            0x3e27_e417_f473_649f,
        ),
        (
            3,
            [32, 32, 64],
            [8, 8, 64],
            0x1ead_b5de_789c_e42c,
            0xfa3e_c777_f070_0b7e,
        ),
        (
            7,
            [48, 40, 56],
            [16, 8, 56],
            0xb8f6_ae8a_248c_c882,
            0x1843_7da7_9be1_bec4,
        ),
    ];
    for (seed, shape, chunks, hl, ha) in golden {
        let c = cfg(seed, shape, chunks);
        assert!(fault_model(&c).is_none());
        assert!(generate_fault_labels(&c).is_none());
        let (v, _) = generate_chunked(&c);
        assert_eq!(fnv(v.labels.iter().copied()), hl, "labels hash seed {seed}");
        assert_eq!(
            fnv(v.angle_stack.iter().flat_map(|x| x.to_bits().to_le_bytes())),
            ha,
            "angle hash seed {seed}"
        );
    }
}

const SHAPE: [usize; 3] = [32, 32, 48];
const SEED: u64 = 4;

fn faulted(chunks: [usize; 3]) -> E2eConfig {
    E2eConfig {
        faults: FaultConfig::with_count(4),
        ..cfg(SEED, SHAPE, chunks)
    }
}

#[test]
fn faulted_labels_differ_and_are_deterministic() {
    let c = faulted([8, 8, 48]);
    let model = fault_model(&c).expect("faults enabled");
    assert!(
        !model.faults().is_empty(),
        "seed {SEED} should insert faults"
    );
    let (plain, _) = generate_labels(&cfg(SEED, SHAPE, [8, 8, 48]));
    let (a, _) = generate_labels(&c);
    let (b, _) = generate_labels(&c);
    assert_eq!(a, b);
    assert_ne!(a, plain, "faulting must change the labels");
    let mask = generate_fault_labels(&c).unwrap();
    assert!(mask.iter().all(|&v| v <= 1));
    let n: usize = mask.iter().map(|&v| v as usize).sum();
    assert!(n > 0, "fault mask empty");
    // Chunked generation (labels + angle stack) is also deterministic.
    let (v1, _) = generate_chunked(&c);
    let (v2, _) = generate_chunked(&c);
    assert_eq!(v1.labels, a);
    assert_eq!(v1.labels, v2.labels);
    assert_eq!(v1.angle_stack, v2.angle_stack);
}

#[test]
fn fault_labels_invariant_to_chunk_shape() {
    let reference_mask = generate_fault_labels(&faulted([32, 32, 48])).unwrap();
    let (reference_labels, _) = generate_labels(&faulted([32, 32, 48]));
    for chunks in [[8, 8, 48], [5, 7, 48], [1, 32, 48], [16, 3, 24]] {
        let c = faulted(chunks);
        assert_eq!(
            generate_fault_labels(&c).unwrap(),
            reference_mask,
            "{chunks:?}"
        );
        assert_eq!(generate_labels(&c).0, reference_labels, "{chunks:?}");
        let (v, _) = generate_chunked(&c);
        assert_eq!(v.labels, reference_labels, "chunked {chunks:?}");
    }
}

#[test]
fn fault_labels_streaming_strip_and_multiprocess_match_reference() {
    let dir = tempdir().unwrap();
    let reference = generate_fault_labels(&faulted([8, 8, 48])).unwrap();
    let (ref_labels, _) = generate_labels(&faulted([8, 8, 48]));

    let stream_path = dir.path().join("stream.mdio");
    let c = E2eConfig {
        store_path: Some(stream_path.clone()),
        ..faulted([8, 8, 16])
    };
    run_e2e_streaming(&c).expect("streaming");
    let store = MdioStore::open(&stream_path).unwrap();
    assert_eq!(store.read_fault_labels_u8().unwrap(), reference);
    assert_eq!(store.read_labels_u8().unwrap(), ref_labels);

    let strip_path = dir.path().join("strip.mdio");
    let c = E2eConfig {
        store_path: Some(strip_path.clone()),
        ..faulted([8, 8, 48])
    };
    let (report, _) = run_e2e_strip_stitched(&c, 4).expect("strip stitch");
    assert!(report.parity.passes_defaults());
    let store = MdioStore::open(&strip_path).unwrap();
    assert_eq!(store.read_fault_labels_u8().unwrap(), reference);

    let mp_path = dir.path().join("mp.mdio");
    let c = E2eConfig {
        store_path: Some(mp_path.clone()),
        ..faulted([8, 16, 48])
    };
    let (report, _) = run_e2e_multiprocess(&c, 3).expect("multiprocess");
    assert!(report.parity.passes_defaults());
    let store = MdioStore::open(&mp_path).unwrap();
    assert_eq!(store.read_fault_labels_u8().unwrap(), reference);
    assert_eq!(report.volumes.labels, ref_labels);
}

#[test]
fn faults_disabled_store_has_no_fault_labels() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("plain.mdio");
    let c = E2eConfig {
        store_path: Some(path.clone()),
        ..cfg(SEED, [16, 16, 24], [8, 8, 24])
    };
    run_e2e_streaming(&c).expect("streaming");
    assert!(!path.join("data").join("fault_labels").exists());
    let store = MdioStore::open(&path).unwrap();
    assert!(store.read_fault_labels_u8().is_err());
}

/// Fault voxels with `k < seabed(i, j)` (in the water column).
fn above_seabed(c: &E2eConfig, mask: &[u8]) -> usize {
    let wb = synthoseis_core::pipeline_stream::fault_seabed(c);
    let nk = c.samples;
    wb.iter()
        .enumerate()
        .map(|(col, &w)| {
            (0..nk)
                .filter(|&k| (k as f64) < w && mask[col * nk + k] == 1)
                .count()
        })
        .sum()
}

fn with_reach(c: &E2eConfig, legacy_reach: bool) -> E2eConfig {
    E2eConfig {
        faults: FaultConfig {
            legacy_reach,
            ..c.faults.clone()
        },
        ..c.clone()
    }
}

/// On a cube tall enough for the legacy seabed taper (sub-seabed column
/// >= ~582 samples) the default reach mode is bit-identical to legacy.
#[test]
fn default_reach_is_legacy_exact_on_tall_cubes() {
    assert!(!FaultConfig::default().legacy_reach);
    for seed in [4u64, 10] {
        let base = E2eConfig {
            faults: FaultConfig::with_count(4),
            ..cfg(seed, [24, 24, 704], [12, 12, 704])
        };
        let legacy = with_reach(&base, true);
        let model = fault_model(&base).unwrap();
        assert!(!model.faults().is_empty(), "seed {seed}");
        assert_eq!(model.reach_rescued(), 0, "seed {seed}");
        assert!(model.faults().iter().all(|f| f.seabed_ok), "seed {seed}");
        let m_def = generate_fault_labels(&base).unwrap();
        let m_leg = generate_fault_labels(&legacy).unwrap();
        assert_eq!(m_def, m_leg, "mask seed {seed}");
        assert_eq!(generate_labels(&base).0, generate_labels(&legacy).0);
        assert_eq!(above_seabed(&base, &m_leg), 0, "legacy above seabed");
    }
}

/// On a short cube the legacy taper gives up and faults the water column; the
/// default mode fits sigma to the column and keeps the water column clean,
/// deterministically and independent of chunking.
#[test]
fn default_reach_keeps_water_column_clean_on_short_cubes() {
    let c = faulted([8, 8, 48]);
    let legacy = with_reach(&c, true);
    let m_def = generate_fault_labels(&c).unwrap();
    let m_leg = generate_fault_labels(&legacy).unwrap();
    let model = fault_model(&c).unwrap();
    assert!(model.reach_rescued() > 0);
    assert!(model.faults().iter().all(|f| f.seabed_ok));
    assert!(fault_model(&legacy)
        .unwrap()
        .faults()
        .iter()
        .any(|f| !f.seabed_ok));
    assert_eq!(above_seabed(&c, &m_def), 0);
    assert!(m_def.contains(&1));
    assert_eq!(generate_fault_labels(&faulted([5, 7, 48])).unwrap(), m_def);
    println!(
        "short cube: legacy voxels={} (above seabed {}), default voxels={}",
        m_leg.iter().map(|&v| v as usize).sum::<usize>(),
        above_seabed(&legacy, &m_leg),
        m_def.iter().map(|&v| v as usize).sum::<usize>()
    );
}
