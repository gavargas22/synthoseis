//! Zoeppritz `det` -> `d` fix: the textbook expression is the default,
//! `legacy_zoeppritz` (`--legacy-zoeppritz`) reproduces master 33a3a93 (the
//! rock-physics default before the fix) bit for bit, `legacy_toy_depth`
//! implies it (master 10f4dcd guarantee), normal incidence is unchanged, and
//! the size of the change on the demo cube is reported.

use synthoseis_core::pipeline::{
    generate_tiny_cube, E2eConfig, FaultConfig, FilterConfig, NoiseConfig, RockPhysicsConfig,
};
use synthoseis_core::rock_physics::{elastic_model, MixingMethod, ZoeppritzForm};
use synthoseis_core::{generate_chunked, generate_chunked_at_angle, generate_labels, generate_reflectivity};

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn ah(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn demo(rock_physics: RockPhysicsConfig) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Planar,
        seed: 7,
        inline_count: 64,
        crossline_count: 64,
        samples: 128,
        store_path: None,
        chunk_shape: Some([16, 16, 128]),
        faults: FaultConfig::with_count(4),
        filters: FilterConfig::default(),
        rock_physics,
    }
}

fn rich(rock: RockPhysicsConfig) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Planar,
        seed: 10,
        inline_count: 24,
        crossline_count: 20,
        samples: 64,
        store_path: None,
        chunk_shape: Some([8, 5, 64]),
        faults: FaultConfig::with_count(3),
        filters: FilterConfig {
            noise: NoiseConfig {
                snr_db: Some(12.5),
                seed: Some(3),
                legacy_angle_weights: false,
                legacy_seabed: false,
            },
            // Golden from before the bandpass trailing-sample fix.
            bandpass_trailing_sample: true,
            ..FilterConfig::legacy(4.0, 30.0, 3)
        },
        rock_physics: RockPhysicsConfig {
            mixing: MixingMethod::BackusModuli,
            first_random_layer: 0,
            layer_shift_samples: Some(6),
            property_shift_samples: Some(3),
            min_closure_voxels: 1,
            ..rock
        },
    }
}

fn legacy_z() -> RockPhysicsConfig {
    RockPhysicsConfig {
        legacy_zoeppritz: true,
        ..RockPhysicsConfig::default()
    }
}

/// Hashes recorded on master 33a3a93 (default rock physics, legacy
/// Zoeppritz) with the same configurations.
#[test]
fn legacy_zoeppritz_reproduces_master_33a3a93() {
    let a = demo(legacy_z());
    let (v, _) = generate_chunked(&a);
    assert_eq!(fnv(v.labels.iter().copied()), 0x22fa_1389_4f70_0192);
    assert_eq!(ah(&v.angle_stack), 0x262f_9caa_ac02_c346);
    assert_eq!(ah(&generate_reflectivity(&a, 15.0)), 0x803c_6752_f97f_a3b5);
    assert_eq!(
        ah(&generate_chunked_at_angle(&a, 30.0).0.angle_stack),
        0x013c_dec9_3d4e_3315
    );
    assert_eq!(
        ah(&generate_chunked(&rich(legacy_z())).0.angle_stack),
        0x7e27_0df1_f8c1_f40b
    );
    let tiny = E2eConfig {
        rock_physics: legacy_z(),
        geometry: synthoseis_core::ToyGeometry::Planar,
        ..E2eConfig::tiny(42)
    };
    assert_eq!(
        ah(&generate_tiny_cube(&tiny).angle_stack),
        0x7e4e_878d_f8b5_3e79
    );

    // The corrected default differs at 15 / 30 degrees.
    let d = demo(RockPhysicsConfig::default());
    assert_ne!(ah(&generate_chunked(&d).0.angle_stack), 0x262f_9caa_ac02_c346);
    assert_ne!(ah(&generate_chunked(&rich(RockPhysicsConfig::default())).0.angle_stack), 0x7e27_0df1_f8c1_f40b);
}

#[test]
fn legacy_toy_depth_implies_legacy_zoeppritz() {
    assert_eq!(RockPhysicsConfig::default().zoeppritz_form(), ZoeppritzForm::Exact);
    assert_eq!(legacy_z().zoeppritz_form(), ZoeppritzForm::Legacy);
    assert_eq!(RockPhysicsConfig::legacy_toy().zoeppritz_form(), ZoeppritzForm::Legacy);
    let toy_only = RockPhysicsConfig {
        legacy_toy_depth: true,
        ..RockPhysicsConfig::default()
    };
    assert_eq!(toy_only.zoeppritz_form(), ZoeppritzForm::Legacy);
    // master 10f4dcd raw reflectivity at 15 degrees, with or without the flag.
    for rp in [toy_only, RockPhysicsConfig::legacy_toy()] {
        assert_eq!(ah(&generate_reflectivity(&demo(rp), 15.0)), 0xee47_4d76_0e21_eb09);
    }
    let c = demo(RockPhysicsConfig::default());
    let (labels, shape) = generate_labels(&c);
    assert_eq!(elastic_model(&c, &labels, shape).zoeppritz_form(), ZoeppritzForm::Exact);
}

struct Delta {
    max_abs: f64,
    rel_rms: f64,
    changed: f64,
    ref_max: f64,
}

fn delta(new: &[f32], old: &[f32]) -> Delta {
    let (mut s2, mut r2, mut m, mut rm, mut ch) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    for (&a, &b) in new.iter().zip(old) {
        let d = (a as f64 - b as f64).abs();
        s2 += d * d;
        r2 += (b as f64).powi(2);
        m = m.max(d);
        rm = rm.max((b as f64).abs());
        ch += usize::from(a.to_bits() != b.to_bits());
    }
    Delta {
        max_abs: m,
        rel_rms: (s2 / r2.max(1e-300)).sqrt(),
        changed: ch as f64 / new.len() as f64,
        ref_max: rm,
    }
}

/// Size of the fix on the demo cube (64x64x128, seed 7, 4 faults):
/// identical at normal incidence, small but visible at 15 / 30 degrees.
#[test]
fn fix_size_on_demo_cube() {
    let fixed = demo(RockPhysicsConfig::default());
    let old = demo(legacy_z());
    for ang in [0.0, 15.0, 30.0, 45.0] {
        let rf = generate_reflectivity(&fixed, ang);
        let ro = generate_reflectivity(&old, ang);
        let d = delta(&rf, &ro);
        let (sf, _) = generate_chunked_at_angle(&fixed, ang);
        let (so, _) = generate_chunked_at_angle(&old, ang);
        let s = delta(&sf.angle_stack, &so.angle_stack);
        eprintln!(
            "angle {ang:>4}: raw rfc max|d| {:.3e} (ref max {:.3}), rel rms {:.3e}, changed {:.3}%;  \
             stack max|d| {:.3e} (ref max {:.3}), rel rms {:.3e}",
            d.max_abs,
            d.ref_max,
            d.rel_rms,
            100.0 * d.changed,
            s.max_abs,
            s.ref_max,
            s.rel_rms
        );
        if ang == 0.0 {
            assert_eq!(d.max_abs, 0.0);
            assert_eq!(s.max_abs, 0.0);
        } else {
            assert!(d.max_abs > 0.0 && s.max_abs > 0.0, "angle {ang}");
            assert!(d.rel_rms < 0.5, "angle {ang}: {}", d.rel_rms);
        }
    }
}
