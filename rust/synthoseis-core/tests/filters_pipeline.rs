//! Pipeline-level seismic filter tests (port of the legacy post-convolution
//! Butterworth bandpass + lateral filter): disabled-by-default bit identity,
//! halo correctness, and bit-identical output for every chunk shape, strip
//! worker count and multi-process partition.

use synthoseis_core::pipeline::{
    generate_tiny_cube, run_e2e, E2eConfig, FaultConfig, FilterConfig,
};
use synthoseis_core::{
    generate_chunked, run_e2e_chunked, run_e2e_geometry_once_seismic_many, run_e2e_multiprocess,
    run_e2e_streaming, run_e2e_streaming_overlapped, run_e2e_strip_stitched, seismic_filters,
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

fn angle_hash(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn cfg(seed: u64, shape: [usize; 3], chunks: [usize; 3], faults: usize) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some(chunks),
        faults: FaultConfig::with_count(faults),
        filters: FilterConfig::default(),
    }
}

fn filtered(chunks: [usize; 3], filters: FilterConfig) -> E2eConfig {
    E2eConfig {
        filters,
        ..cfg(10, [24, 20, 64], chunks, 3)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Hashes of `generate_chunked` recorded on master ca11a457 (before the
/// filter port). Filters are off by default, so output must not change.
#[test]
fn filters_disabled_by_default_is_bit_identical_to_master() {
    assert!(!FilterConfig::default().enabled());
    assert!(!E2eConfig::tiny(1).filters.enabled());
    assert!(seismic_filters(&E2eConfig::tiny(1)).is_none());
    type Golden = (u64, [usize; 3], [usize; 3], usize, u64, u64);
    let golden: [Golden; 3] = [
        (
            4,
            [32, 32, 48],
            [8, 8, 48],
            4,
            0xf81da1d9647b7ae4,
            0xd3bc399056319827,
        ),
        (
            10,
            [24, 20, 64],
            [8, 5, 64],
            3,
            0x193e13e29a1ba848,
            0x3af5b029bb944ced,
        ),
        (
            3,
            [32, 32, 64],
            [8, 8, 64],
            0,
            0x1eadb5de789ce42c,
            0xfa3ec777f0700b7e,
        ),
    ];
    for (seed, shape, chunks, faults, labels_h, angle_h) in golden {
        let c = cfg(seed, shape, chunks, faults);
        let (v, _) = generate_chunked(&c);
        assert_eq!(
            fnv(v.labels.iter().copied()),
            labels_h,
            "labels seed {seed}"
        );
        assert_eq!(angle_hash(&v.angle_stack), angle_h, "angle seed {seed}");
        // Explicit "off" configs are identical to the default.
        for off in [
            FilterConfig {
                lateral_size: 0,
                ..Default::default()
            },
            FilterConfig {
                bandpass_order: 2,
                ..Default::default()
            },
        ] {
            assert!(!off.enabled());
            let (w, _) = generate_chunked(&E2eConfig {
                filters: off,
                ..c.clone()
            });
            assert_eq!(angle_hash(&w.angle_stack), angle_h);
        }
    }
}

fn configs() -> Vec<FilterConfig> {
    vec![
        FilterConfig::legacy(4.0, 30.0, 3),
        FilterConfig::legacy(5.5, 22.0, 5),
        FilterConfig {
            bandpass_hz: Some([3.0, 35.0]),
            bandpass_order: 2,
            lateral_size: 1,
        },
        FilterConfig {
            bandpass_hz: None,
            bandpass_order: 4,
            lateral_size: 4,
        },
    ]
}

/// The halo-tiled pipeline equals the whole-volume filter applied to the
/// unfiltered stack (i.e. halos are complete, including reflected borders).
#[test]
fn filtered_stack_equals_whole_volume_filter() {
    let (raw, _) = generate_chunked(&cfg(10, [24, 20, 64], [8, 5, 64], 3));
    for fc in configs() {
        let c = filtered([8, 5, 64], fc.clone());
        let (v, _) = generate_chunked(&c);
        assert_ne!(bits(&v.angle_stack), bits(&raw.angle_stack), "{fc:?}");
        let mut whole = raw.angle_stack.clone();
        synthoseis_core::pipeline_stream::apply_filters_to_volume(&c, &mut whole);
        assert_eq!(bits(&v.angle_stack), bits(&whole), "{fc:?}");
        assert_eq!(v.labels, raw.labels, "filters never touch labels");
    }
}

#[test]
fn filtered_stack_invariant_to_chunk_shape() {
    let chunk_shapes: [[usize; 3]; 7] = [
        [24, 20, 64],
        [8, 5, 64],
        [5, 7, 64],
        [1, 20, 64],
        [24, 1, 16],
        [7, 3, 32],
        [2, 2, 64],
    ];
    for fc in configs() {
        let (reference, _) = generate_chunked(&filtered(chunk_shapes[0], fc.clone()));
        for chunks in &chunk_shapes[1..] {
            let (v, _) = generate_chunked(&filtered(*chunks, fc.clone()));
            assert_eq!(
                bits(&v.angle_stack),
                bits(&reference.angle_stack),
                "{fc:?} chunks {chunks:?}"
            );
        }
        // Classic whole-cube path (generate_tiny_cube / run_e2e).
        let classic = generate_tiny_cube(&filtered([8, 5, 64], fc.clone()));
        assert_eq!(
            bits(&classic.angle_stack),
            bits(&reference.angle_stack),
            "classic {fc:?}"
        );
    }
}

/// Streaming, overlapped streaming, strip-stitch (2-4 workers), multi-process
/// (1-3 workers) and geometry-once all write bit-identical filtered stacks.
#[test]
fn filtered_stack_invariant_to_workers_and_paths() {
    let dir = tempdir().unwrap();
    for (n, fc) in configs().into_iter().take(2).enumerate() {
        let (reference, _) = generate_chunked(&filtered([24, 20, 64], fc.clone()));
        let want = bits(&reference.angle_stack);
        let read = |p: &std::path::Path| bits(&MdioStore::open(p).unwrap().read_volume().unwrap());

        for (k, chunks) in [[8, 5, 16], [5, 7, 64], [3, 20, 32]]
            .into_iter()
            .enumerate()
        {
            let p = dir.path().join(format!("stream_{n}_{k}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..filtered(chunks, fc.clone())
            };
            let (report, _) = run_e2e_streaming(&c).expect("streaming");
            assert!(report.parity.passes_defaults());
            assert_eq!(read(&p), want, "streaming {fc:?} {chunks:?}");

            let p = dir.path().join(format!("overlap_{n}_{k}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..filtered(chunks, fc.clone())
            };
            run_e2e_streaming_overlapped(&c).expect("overlap");
            assert_eq!(read(&p), want, "overlap {fc:?} {chunks:?}");
        }

        for workers in [2, 3, 4] {
            let p = dir.path().join(format!("strip_{n}_{workers}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..filtered([5, 7, 64], fc.clone())
            };
            let (report, _) = run_e2e_strip_stitched(&c, workers).expect("strip stitch");
            assert!(report.parity.passes_defaults());
            assert_eq!(read(&p), want, "strip {fc:?} workers {workers}");
        }

        for workers in [1, 2, 3] {
            let p = dir.path().join(format!("mp_{n}_{workers}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..filtered([8, 7, 64], fc.clone())
            };
            let (report, _) = run_e2e_multiprocess(&c, workers).expect("multiprocess");
            assert!(report.parity.passes_defaults());
            assert_eq!(bits(&report.volumes.angle_stack), want);
            assert_eq!(read(&p), want, "multiprocess {fc:?} workers {workers}");
        }

        let (geo, _) =
            run_e2e_geometry_once_seismic_many(&filtered([5, 7, 64], fc.clone()), &[0.0, 15.0])
                .expect("geometry once");
        let at15 = synthoseis_core::generate_chunked_at_angle(&filtered([24, 20, 64], fc), 15.0).0;
        assert_eq!(
            bits(&geo.stacks[1].volumes.angle_stack),
            bits(&at15.angle_stack)
        );
    }
}

#[test]
fn invalid_filter_config_is_an_error() {
    let dir = tempdir().unwrap();
    let short = E2eConfig {
        store_path: Some(dir.path().join("short.mdio")),
        filters: FilterConfig::legacy(4.0, 30.0, 1),
        ..cfg(1, [8, 8, 27], [4, 4, 27], 0)
    };
    assert!(run_e2e_streaming(&short)
        .unwrap_err()
        .contains("27 samples"));
    assert!(run_e2e_chunked(&short).is_err());
    assert!(run_e2e(&short).is_err());
    assert!(run_e2e_strip_stitched(&short, 2).is_err());
    assert!(run_e2e_multiprocess(&short, 2).is_err());
    assert!(run_e2e_streaming_overlapped(&short).is_err());
    let nyquist = E2eConfig {
        filters: FilterConfig::legacy(4.0, 125.0, 1),
        ..cfg(1, [8, 8, 64], [4, 4, 64], 0)
    };
    assert!(run_e2e_chunked(&nyquist).unwrap_err().contains("Nyquist"));
    // 28 samples (> padlen 27) is accepted.
    let ok = E2eConfig {
        filters: FilterConfig::legacy(4.0, 30.0, 3),
        ..cfg(1, [8, 8, 28], [4, 4, 28], 0)
    };
    assert!(run_e2e_chunked(&ok).is_ok());
}
