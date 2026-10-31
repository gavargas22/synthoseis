//! Per-stage timings, legacy vs heave kinematics (spec-fault-kinematics 7a
//! runtime breakdown), on the library's whole-cube stage functions:
//!
//! ```text
//! cargo run --release -p synthoseis-core --example heave_profile -- 7 128 128 256 4
//! ```
use std::time::Instant;
use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::rock_physics::elastic_model;
use synthoseis_core::{
    fault_model, generate_angle_stack_from_labels, generate_fault_labels, generate_labels,
    DEFAULT_INCIDENCE_DEG,
};

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let (seed, shape, count) = (a[0] as u64, [a[1], a[2], a[3]], a[4]);
    let mut rows: Vec<(&str, [f64; 2])> = Vec::new();
    let mut put =
        |name: &'static str, kin: usize, t: f64| match rows.iter_mut().find(|r| r.0 == name) {
            Some(r) => r.1[kin] = t,
            None => {
                let mut v = [0.0; 2];
                v[kin] = t;
                rows.push((name, v));
            }
        };
    for (kin, legacy) in [(0, true), (1, false)] {
        let c = E2eConfig {
            seed,
            inline_count: shape[0],
            crossline_count: shape[1],
            samples: shape[2],
            faults: FaultConfig {
                count,
                legacy_kinematics: legacy,
                ..FaultConfig::default()
            },
            ..Default::default()
        };
        let [ni, nj, nk] = shape;
        let t = Instant::now();
        let m = fault_model(&c).expect("faults");
        put("fault model build", kin, t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut acc = 0.0;
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let g = m.gather(i, j, k);
                    acc += g.src[0] + g.src[2] + g.id as f64;
                }
            }
        }
        put("gather walk, all voxels", kin, t.elapsed().as_secs_f64());
        std::hint::black_box(acc);
        let t = Instant::now();
        let (labels, sh) = generate_labels(&c);
        put(
            "generate_labels (horizons + faulting)",
            kin,
            t.elapsed().as_secs_f64(),
        );
        let t = Instant::now();
        std::hint::black_box(generate_fault_labels(&c));
        put("generate_fault_labels", kin, t.elapsed().as_secs_f64());
        let t = Instant::now();
        let model = elastic_model(&c, &labels, sh);
        put("elastic_model build", kin, t.elapsed().as_secs_f64());
        let n = ni * nj * nk;
        let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let t = Instant::now();
        model.tile_properties(&labels, sh, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
        put(
            "rock physics, whole cube as one tile",
            kin,
            t.elapsed().as_secs_f64(),
        );
        let t = Instant::now();
        let out = generate_angle_stack_from_labels(&c, &labels, sh, DEFAULT_INCIDENCE_DEG);
        put(
            "angle stack (elastic_model + per-tile fuse)",
            kin,
            t.elapsed().as_secs_f64(),
        );
        std::hint::black_box(out);
        let t = Instant::now();
        std::hint::black_box(synthoseis_core::time_mode::generate_output_masks(&c));
        put("output masks (store write)", kin, t.elapsed().as_secs_f64());
    }
    println!("seed {seed}: stage | legacy s | heave s | heave/legacy");
    for (n, v) in rows {
        println!(
            "{n} | {:.2} | {:.2} | {:.2}",
            v[0],
            v[1],
            v[1] / v[0].max(1e-9)
        );
    }
}
