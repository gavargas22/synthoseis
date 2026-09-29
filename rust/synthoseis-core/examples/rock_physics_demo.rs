//! Rock-physics demo: the corrected legacy model (default) vs the master toy
//! model (`--legacy-toy-depth`) on one (faulted) cube. Dumps raw volumes for
//! `examples/plot_rock_physics_demo.py` and prints reflectivity statistics.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example rock_physics_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=128]
//! ```
//!
//! Writes little-endian C-order `(ni, nj, nk)` arrays: `labels.u8`,
//! `depth.f32` (default model, metres below the seabed) and, per model
//! `M` in `toy`, `rpm` (inverse velocity), `backus`: `M_vp.f32`, `M_vs.f32`,
//! `M_rho.f32`, `M_rfc15.f32` (raw 15° reflectivity) and `M_stack{0,15,30}.f32`
//! (Ricker angle stacks). `meta.json` holds shape, closures and the GPU vs CPU
//! gap of the default model (when a wgpu adapter exists).

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, MixingMethod, RockPhysicsConfig};
use synthoseis_core::{generate_chunked_at_angle, generate_labels, generate_reflectivity};

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn write(path: PathBuf, bytes: &[u8]) {
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(bytes))
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn gap(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn stats(v: &[f32]) -> String {
    let n = v.len() as f64;
    let (mut lo, mut hi, mut sum, mut big, mut nz) = (f32::MAX, f32::MIN, 0.0f64, 0usize, 0usize);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
        sum += x as f64;
        big += usize::from(x.abs() > 1.0);
        nz += usize::from(x != 0.0);
    }
    format!(
        "{{\"min\":{lo:e},\"max\":{hi:e},\"mean\":{:e},\"abs_gt_1_pct\":{:.4},\"nonzero_pct\":{:.3}}}",
        sum / n,
        100.0 * big as f64 / n,
        100.0 * nz as f64 / n
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().cloned().unwrap_or_else(|| "rock_physics_demo".into()));
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 128);
    std::fs::create_dir_all(&out).expect("create out dir");

    let base = E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        store_path: None,
        chunk_shape: Some([16, 16, nk]),
        faults: FaultConfig::with_count(count),
        filters: FilterConfig::default(),
        rock_physics: RockPhysicsConfig::default(),
    };
    let models = [
        ("toy", RockPhysicsConfig::legacy_toy()),
        ("rpm", RockPhysicsConfig::default()),
        (
            "backus",
            RockPhysicsConfig {
                mixing: MixingMethod::BackusModuli,
                ..RockPhysicsConfig::default()
            },
        ),
    ];
    let (labels, shape) = generate_labels(&base);
    write(out.join("labels.u8"), &labels);
    let n = ni * nj * nk;
    let mut meta = format!("{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"depth_step_m\":4.0");
    for (name, rp) in &models {
        let cfg = E2eConfig {
            rock_physics: rp.clone(),
            ..base.clone()
        };
        let model = elastic_model(&cfg, &labels, shape);
        let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
        write(out.join(format!("{name}_vp.f32")), &f32_bytes(&vp));
        write(out.join(format!("{name}_vs.f32")), &f32_bytes(&vs));
        write(out.join(format!("{name}_rho.f32")), &f32_bytes(&rho));
        let rfc = generate_reflectivity(&cfg, 15.0);
        write(out.join(format!("{name}_rfc15.f32")), &f32_bytes(&rfc));
        meta += &format!(",\"{name}_rfc15\":{}", stats(&rfc));
        for a in [0.0, 15.0, 30.0] {
            let (v, _) = generate_chunked_at_angle(&cfg, a);
            write(out.join(format!("{name}_stack{a:.0}.f32")), &f32_bytes(&v.angle_stack));
        }
        if let ElasticModel::Rpm(m) = &model {
            if *name == "rpm" {
                let mut depth = vec![0.0f32; n];
                for i in 0..ni {
                    for j in 0..nj {
                        let g = (i * nj + j) * nk;
                        m.depth_trace(i, j, &labels[g..g + nk], &mut depth[g..g + nk]);
                    }
                }
                write(out.join("depth.f32"), &f32_bytes(&depth));
                let closures: Vec<String> = m
                    .layers
                    .iter()
                    .filter_map(|l| l.fluids.as_ref().map(|f| (l.interval, f)))
                    .flat_map(|(h, f)| {
                        f.closures.iter().map(move |c| {
                            format!(
                                "{{\"layer\":{h},\"fluid\":\"{}\",\"crest\":{},\"contact\":{},\"columns\":{},\"voxels\":{}}}",
                                c.0.as_str(),
                                c.1,
                                c.2,
                                c.3,
                                c.4
                            )
                        })
                    })
                    .collect();
                meta += &format!(",\"closures\":[{}]", closures.join(","));
                // GPU vs CPU on the default model (software adapter in CI).
                synthoseis_gpu::set_prefer_gpu(true);
                let status = synthoseis_gpu::backend_status();
                let gpu_rfc = generate_reflectivity(&cfg, 15.0);
                let (gpu_stack, _) = generate_chunked_at_angle(&cfg, 15.0);
                synthoseis_gpu::set_prefer_gpu(false);
                let (cpu_stack, _) = generate_chunked_at_angle(&cfg, 15.0);
                meta += &format!(
                    ",\"gpu_backend\":\"{status}\",\"gpu_available\":{},\"gpu_cpu_max_abs_rfc15\":{:e},\"gpu_cpu_max_abs_stack15\":{:e}",
                    synthoseis_gpu::gpu_device_available(),
                    gap(&gpu_rfc, &rfc),
                    gap(&gpu_stack.angle_stack, &cpu_stack.angle_stack)
                );
            }
        }
    }
    meta += "}";
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}
