#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::generate_tiny_cube;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn chunked_matches_tiny_bit_identical() {
        let cfg = E2eConfig::tiny(42);
        let classic = generate_tiny_cube(&cfg);
        let (chunked, stats) = generate_chunked(&cfg);
        assert_eq!(classic.labels, chunked.labels);
        assert_eq!(classic.angle_stack, chunked.angle_stack);
        assert!(stats.tiles_processed >= 1);
    }

    #[test]
    fn chunked_mdio_uses_subvolume_chunks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunked.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            store_path: Some(path.clone()),
            chunk_shape: Some([4, 4, 8]),
            ..E2eConfig::tiny(7)
        };
        let (report, _stats) = run_e2e_chunked(&cfg).expect("chunked e2e");
        assert_eq!(report.status, "ok-e2e-chunked");
        let zarray = fs::read_to_string(path.join("data").join("chunked_012").join(".zarray"))
            .expect(".zarray");
        let compact: String = zarray.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            compact.contains("\"chunks\":[4,4,8]"),
            "chunks not sub-volume: {zarray}"
        );
        assert!(path.join("data").join("chunked_012").join("0.0.0").is_file());
        assert!(
            path.join("data").join("chunked_012").join("1.0.0").is_file()
                || path.join("data").join("chunked_012").join("0.1.0").is_file()
        );
    }

    #[test]
    fn larger_cube_chunked_parity_and_bound() {
        let cfg = E2eConfig {
            faults: Default::default(),
            seed: 3,
            inline_count: 32,
            crossline_count: 32,
            samples: 64,
            store_path: None,
            chunk_shape: Some([8, 8, 64]),
        };
        let (a, stats) = generate_chunked(&cfg);
        let (b, _) = generate_chunked(&cfg);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.angle_stack, b.angle_stack);
        assert_eq!(a.shape, [32, 32, 64]);
        let n = 32 * 32 * 64;
        let one_elastic = n * 4;
        assert!(
            stats.peak_temp_bytes < one_elastic,
            "peak_temp_bytes {} should be << one elastic volume {}",
            stats.peak_temp_bytes,
            one_elastic
        );
        assert!(
            stats.is_bounded_by_chunk(64),
            "stats not bounded: {:?}",
            stats
        );
    }

    #[test]
    fn streaming_write_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            seed: 11,
            inline_count: 16,
            crossline_count: 16,
            samples: 32,
            store_path: Some(path.clone()),
            chunk_shape: Some([8, 8, 32]),
        };
        let (report, stats) = run_e2e_streaming(&cfg).expect("stream");
        assert_eq!(report.status, "ok-e2e-chunked");
        assert!(path.join("data").join("chunked_012").join("0.0.0").is_file());
        assert!(path.join("data").join("chunked_012").join("1.1.0").is_file());
        let n = 16 * 16 * 32;
        assert!(stats.peak_temp_bytes < n * 4);
    }

    #[test]
    fn default_chunks_are_subvolume_for_8() {
        assert_eq!(default_subvolume_chunks([8, 8, 8]), [4, 4, 8]);
        assert_eq!(default_subvolume_chunks([32, 32, 64]), [8, 8, 64]);
    }
    #[test]
    fn strip_stitch_workers4_parity_vs_single() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("strip.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            seed: 42,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
            store_path: Some(path.clone()),
            // chunk_i=2 → 4 i-chunks → one per worker
            chunk_shape: Some([2, 4, 8]),
        };
        let (report, stats) = run_e2e_strip_stitched(&cfg, 4).expect("strip-stitch");
        assert_eq!(report.status, "ok-e2e-strip-stitch");
        assert!(report.parity.passes_defaults());
        assert!((report.parity.label_iou - 1.0).abs() < 1e-12);
        assert_eq!(report.parity.angle_mae, 0.0);
        assert!(stats.tiles_processed >= 4);

        let (single, _) = generate_chunked(&cfg);
        assert_eq!(report.volumes.labels, single.labels);
        assert_eq!(report.volumes.angle_stack, single.angle_stack);
    }

    #[test]
    fn strip_stitch_16x16x32_workers4_near_parity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("strip16.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            seed: 7,
            inline_count: 16,
            crossline_count: 16,
            samples: 32,
            store_path: Some(path.clone()),
            chunk_shape: Some([4, 4, 32]),
        };
        let (report, _) = run_e2e_strip_stitched(&cfg, 4).expect("strip-stitch 16");
        assert_eq!(report.status, "ok-e2e-strip-stitch");
        assert!(report.parity.passes_defaults());
        let (single, _) = generate_chunked(&cfg);
        assert_eq!(report.volumes.labels, single.labels);
        assert_eq!(report.volumes.angle_stack, single.angle_stack);
    }

    #[test]
    fn strip_stitch_workers_one_matches_chunked() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("one.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            seed: 3,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
            store_path: Some(path),
            chunk_shape: Some([4, 4, 8]),
        };
        let (a, _) = run_e2e_strip_stitched(&cfg, 1).expect("w1");
        assert_eq!(a.status, "ok-e2e-chunked");
        assert!(a.parity.passes_defaults());
    }

}
