//! The bandpass trailing-sample fix (legacy parity bug fix).
//!
//! The Rust fuse writes `nk` reflectivity samples per trace; the last one is
//! always 0 (legacy produces only `nk - 1`: `rfc_raw` has one sample fewer
//! than the elastic cube). With the Ricker skipped (bandpass on, legacy
//! chain) the default filters now bandpass only the first `nk - 1` samples
//! and write 0 to the trailing sample, as legacy never has one;
//! `--bandpass-trailing-sample` (`FilterConfig::bandpass_trailing_sample`)
//! restores the old whole-trace bandpass bit for bit.
//!
//! Golden hashes: FNV-1a 64 over the f32 little-endian bits of the 15° angle
//! stack from `generate_chunked` (equal to `generate_tiny_cube`). "Old"
//! hashes were recorded on the #35 head c2371fa (= master for these paths)
//! before the fix; Ricker-only hashes are identical before and after.
//! End-to-end parity against the real legacy generator is in
//! `angle_stack_legacy_e2e.rs`.

use synthoseis_core::pipeline::{generate_tiny_cube, E2eConfig, FaultConfig, FilterConfig, NoiseConfig, RockPhysicsConfig};
use synthoseis_core::{generate_chunked, generate_reflectivity, run_e2e_chunked};

fn angle_hash(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Planar model (as `filters_pipeline.rs`), current default rock physics.
fn planar(filters: FilterConfig) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Planar,
        seed: 10,
        inline_count: 24,
        crossline_count: 20,
        samples: 64,
        chunk_shape: Some([8, 5, 64]),
        faults: FaultConfig::with_count(3),
        filters,
        ..E2eConfig::default()
    }
}

/// Default (folded) geometry with more faults, without the salt body
/// (`--no-salt`), so these pins stay the pre-salt #36 values and show the
/// bandpass fix is isolated from the salt change.
fn rich(filters: FilterConfig) -> E2eConfig {
    E2eConfig {
        rock_physics: RockPhysicsConfig { salt: false, ..Default::default() },
        ..rich_salt(filters)
    }
}

/// [`rich`] with the default salt body (1472 salt voxels on this model).
fn rich_salt(filters: FilterConfig) -> E2eConfig {
    E2eConfig {
        seed: 7,
        inline_count: 32,
        crossline_count: 32,
        samples: 96,
        chunk_shape: Some([8, 8, 96]),
        faults: FaultConfig::with_count(4),
        filters,
        ..E2eConfig::default()
    }
}

fn noise() -> NoiseConfig {
    NoiseConfig {
        snr_db: Some(12.5),
        seed: Some(7),
        ..Default::default()
    }
}

fn o2() -> FilterConfig {
    FilterConfig {
        bandpass_hz: Some([3.0, 35.0]),
        bandpass_order: 2,
        lateral_size: 1,
        ..Default::default()
    }
}

fn with_trailing(fc: FilterConfig, on: bool) -> FilterConfig {
    FilterConfig {
        bandpass_trailing_sample: on,
        ..fc
    }
}

/// Assert both the chunked and the classic whole-cube path hash to `want`.
fn check(c: &E2eConfig, want: u64, what: &str) {
    let (v, _) = generate_chunked(c);
    assert_eq!(angle_hash(&v.angle_stack), want, "chunked {what}: {:#018x}", angle_hash(&v.angle_stack));
    assert_eq!(angle_hash(&generate_tiny_cube(c).angle_stack), want, "classic {what}");
}

/// Bandpass-on (Ricker skipped) configs: (name, config, old hash, fixed hash).
fn skip_cases() -> Vec<(&'static str, E2eConfig, u64, u64)> {
    vec![
        ("legacy 4-30 lateral 3", planar(FilterConfig::legacy(4.0, 30.0, 3)), 0x073373b70d064c1d, 0xe42ddd563b1e2a1f),
        ("legacy 5.5-22 lateral 5", planar(FilterConfig::legacy(5.5, 22.0, 5)), 0x645b1355eefd1d49, 0x8d6204781cf32cf1),
        ("3-35 order 2", planar(o2()), 0x130f13e9bc396103, 0x10432a7896a0012d),
        (
            "legacy 4-30 lateral 3 + noise",
            planar(FilterConfig { noise: noise(), ..FilterConfig::legacy(4.0, 30.0, 3) }),
            0xe65b7ffea3e169fa,
            0x8f43672d8d481386,
        ),
        ("rich legacy 4-30 lateral 3", rich(FilterConfig::legacy(4.0, 30.0, 3)), 0xe8c489d00daf35c1, 0x02c5ac46fac91025),
    ]
}

/// `--bandpass-trailing-sample` reproduces the pre-fix output bit for bit;
/// the default (fixed) output is pinned too.
#[test]
fn both_bandpass_modes_are_pinned() {
    for (name, c, old, fixed) in skip_cases() {
        assert!(c.filters.skips_ricker() && c.filters.bandpass_excludes_trailing_sample());
        check(&c, fixed, &format!("fixed {name}"));
        let flagged = E2eConfig { filters: with_trailing(c.filters.clone(), true), ..c.clone() };
        assert!(!flagged.filters.bandpass_excludes_trailing_sample());
        check(&flagged, old, &format!("--bandpass-trailing-sample {name}"));
        assert_ne!(old, fixed);
    }
}

/// Paths without the Ricker skip (filters off, lateral only, `keep_ricker`,
/// noise without a bandpass) are unchanged by the fix and ignore the flag.
#[test]
fn ricker_paths_are_unchanged_under_both_flag_values() {
    let cases: Vec<(&str, E2eConfig, u64)> = vec![
        ("filters off", planar(FilterConfig::default()), 0xb39847a478012314),
        ("lateral 4", planar(FilterConfig { lateral_size: 4, ..Default::default() }), 0x04d841a689a4ff3f),
        (
            "keep_ricker 4-30 lateral 3",
            planar(FilterConfig { keep_ricker: true, ..FilterConfig::legacy(4.0, 30.0, 3) }),
            0xe4bdd9147912f734,
        ),
        (
            "lateral 3 + noise",
            planar(FilterConfig { lateral_size: 3, noise: noise(), ..Default::default() }),
            0x69a7ed34a05ccd38,
        ),
        (
            "keep_ricker + noise",
            planar(FilterConfig { keep_ricker: true, noise: noise(), ..FilterConfig::legacy(4.0, 30.0, 3) }),
            0xe45557ae0f5f1a74,
        ),
        ("rich filters off", rich(FilterConfig::default()), 0xae7820aa7a26c460),
    ];
    for (name, c, want) in cases {
        assert!(!c.filters.skips_ricker(), "{name}");
        for on in [false, true] {
            // The flag is only accepted with a bandpass by the CLI; the core
            // simply ignores it when the Ricker is kept.
            let c = E2eConfig { filters: with_trailing(c.filters.clone(), on), ..c.clone() };
            assert!(!c.filters.bandpass_excludes_trailing_sample());
            check(&c, want, &format!("{name} (flag {on})"));
        }
    }
}

/// The salt-default variant of the `rich` pins: the salt body changes the
/// model (so every hash differs from the `--no-salt` pins above), and the
/// bandpass modes behave the same way on top of it.
#[test]
fn salt_default_rich_is_pinned() {
    assert!(rich_salt(FilterConfig::default()).effective_salt());
    assert!(!rich(FilterConfig::default()).effective_salt());
    let fixed = FilterConfig::legacy(4.0, 30.0, 3);
    let cases: Vec<(&str, E2eConfig, u64)> = vec![
        ("filters off", rich_salt(FilterConfig::default()), 0x0427bd22f58bb87e),
        ("legacy 4-30 lateral 3 (fixed)", rich_salt(fixed.clone()), 0x5cac37c5c26d06e1),
        ("legacy 4-30 lateral 3 --bandpass-trailing-sample", rich_salt(with_trailing(fixed, true)), 0xf47229efbffaa6c7),
    ];
    for (name, c, want) in cases {
        check(&c, want, &format!("salt-default rich {name}"));
        let no_salt = E2eConfig { rock_physics: RockPhysicsConfig { salt: false, ..c.rock_physics.clone() }, ..c.clone() };
        assert_ne!(angle_hash(&generate_chunked(&no_salt).0.angle_stack), want, "{name}: salt changes the output");
    }
}

/// Fixed mode = the whole-trace filter applied to the first `nk - 1`
/// reflectivity samples of every trace, with the trailing sample 0.
#[test]
fn fixed_mode_filters_only_the_first_nk_minus_1_samples() {
    for fc in [FilterConfig::legacy(4.0, 30.0, 3), FilterConfig::legacy(5.5, 22.0, 5), o2()] {
        let c = planar(fc.clone());
        let nk = c.samples;
        let (v, _) = generate_chunked(&c);
        assert!(v.angle_stack.chunks_exact(nk).all(|t| t[nk - 1].to_bits() == 0), "{fc:?}");

        let rfc = generate_reflectivity(&c, 15.0);
        assert!(rfc.chunks_exact(nk).all(|t| t[nk - 1] == 0.0), "fuse trailing sample");
        let mut short: Vec<f32> = rfc.chunks_exact(nk).flat_map(|t| t[..nk - 1].to_vec()).collect();
        let short_cfg = E2eConfig { samples: nk - 1, filters: with_trailing(fc.clone(), true), ..c.clone() };
        synthoseis_core::pipeline_stream::apply_filters_to_volume(&short_cfg, &mut short);
        let fixed_short: Vec<f32> = v.angle_stack.chunks_exact(nk).flat_map(|t| t[..nk - 1].to_vec()).collect();
        assert_eq!(bits(&fixed_short), bits(&short), "{fc:?}");

        // The old mode changes every trace's deep part.
        let (old, _) = generate_chunked(&E2eConfig { filters: with_trailing(fc.clone(), true), ..c.clone() });
        assert_ne!(bits(&old.angle_stack), bits(&v.angle_stack));
    }
}

/// With noise the fuse output has noise on the trailing sample too; legacy
/// has no sample there, so fixed mode writes 0.
#[test]
fn fixed_mode_zeroes_the_trailing_sample_with_noise() {
    let c = planar(FilterConfig { noise: noise(), ..FilterConfig::legacy(4.0, 30.0, 3) });
    let nk = c.samples;
    let (v, _) = generate_chunked(&c);
    assert!(v.angle_stack.chunks_exact(nk).all(|t| t[nk - 1].to_bits() == 0));
    let (old, _) = generate_chunked(&E2eConfig { filters: with_trailing(c.filters.clone(), true), ..c });
    assert!(old.angle_stack.chunks_exact(nk).any(|t| t[nk - 1] != 0.0));
}

/// The padlen check counts the filtered samples: `nk - 1` in fixed mode.
#[test]
fn padlen_check_counts_filtered_samples() {
    let short = |samples: usize, on: bool| E2eConfig {
        samples,
        chunk_shape: Some([8, 5, samples]),
        ..planar(with_trailing(FilterConfig::legacy(4.0, 30.0, 1), on))
    };
    // Order 4: padlen 27.
    assert!(run_e2e_chunked(&short(29, false)).is_ok());
    let err = run_e2e_chunked(&short(28, false)).unwrap_err();
    assert!(err.contains("got 27 of 28 samples"), "{err}");
    assert!(run_e2e_chunked(&short(28, true)).is_ok());
    assert!(run_e2e_chunked(&short(27, true)).is_err());
}
