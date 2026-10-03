//! Zoeppritz fix demo: textbook (default) vs legacy `det` typo
//! (`--legacy-zoeppritz`) on the rock-physics demo cube, the GPU vs CPU gap
//! for both forms, and AVO curves of real interfaces from the cube. Dumps
//! data for `examples/plot_zoeppritz_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example zoeppritz_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=128]
//! ```
//!
//! Writes little-endian C-order `(ni, nj, nk)` f32 arrays
//! `{exact,legacy}_stack{15,30,45}.f32` and `meta.json` (shape, the change
//! per angle, GPU vs CPU gaps, and a few interfaces with their AVO curves).

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, RockPhysicsConfig};
use synthoseis_core::{generate_chunked_at_angle, generate_labels, generate_reflectivity};
use synthoseis_seismic::{zoeppritz_pp_form, ZoeppritzForm};

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

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        r += (y as f64).powi(2);
    }
    (d / r.max(1e-300)).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().cloned().unwrap_or_else(|| "zoeppritz_demo".into()));
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 128);
    std::fs::create_dir_all(&out).expect("create out dir");
    let exact = E2eConfig {
        time: Default::default(),
        geometry: synthoseis_core::ToyGeometry::Planar,
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
    let legacy = E2eConfig {
        rock_physics: RockPhysicsConfig {
            legacy_zoeppritz: true,
            ..RockPhysicsConfig::default()
        },
        ..exact.clone()
    };
    let mut meta = format!("{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"change\":[");
    let mut first = true;
    for a in [0.0, 15.0, 30.0, 45.0] {
        let (e, _) = generate_chunked_at_angle(&exact, a);
        let (l, _) = generate_chunked_at_angle(&legacy, a);
        let re = generate_reflectivity(&exact, a);
        let rl = generate_reflectivity(&legacy, a);
        if a > 0.0 {
            write(out.join(format!("exact_stack{a:.0}.f32")), &f32_bytes(&e.angle_stack));
            write(out.join(format!("legacy_stack{a:.0}.f32")), &f32_bytes(&l.angle_stack));
        }
        // GPU (WGSL mode 1) vs CPU for both forms.
        synthoseis_gpu::set_prefer_gpu(true);
        let ge = generate_reflectivity(&exact, a);
        let gl = generate_reflectivity(&legacy, a);
        let (gse, _) = generate_chunked_at_angle(&exact, a);
        synthoseis_gpu::set_prefer_gpu(false);
        meta += &format!(
            "{}{{\"angle\":{a},\"rfc_max_abs\":{:e},\"rfc_rel_rms\":{:e},\"stack_max_abs\":{:e},\"stack_rel_rms\":{:e},\
             \"ref_rfc_max\":{:e},\"gpu_cpu_rfc_exact\":{:e},\"gpu_cpu_rfc_legacy\":{:e},\"gpu_cpu_stack_exact\":{:e}}}",
            if first { "" } else { "," },
            gap(&re, &rl),
            rel_rms(&re, &rl),
            gap(&e.angle_stack, &l.angle_stack),
            rel_rms(&e.angle_stack, &l.angle_stack),
            rl.iter().fold(0.0f32, |m, x| m.max(x.abs())),
            gap(&ge, &re),
            gap(&gl, &rl),
            gap(&gse.angle_stack, &e.angle_stack),
        );
        first = false;
    }
    meta += &format!(
        "],\"gpu_backend\":\"{}\",\"gpu_available\":{}",
        synthoseis_gpu::backend_status(),
        synthoseis_gpu::gpu_device_available()
    );

    // Interfaces of the centre column (distinct property pairs).
    let (labels, shape) = generate_labels(&exact);
    let model = elastic_model(&exact, &labels, shape);
    let (ci, cj) = (ni / 2, nj / 2);
    let (mut vp, mut vs, mut rho) = (vec![0.0f32; nk], vec![0.0f32; nk], vec![0.0f32; nk]);
    model.tile_properties(&labels, shape, ci, ci + 1, cj, cj + 1, &mut vp, &mut vs, &mut rho);
    let mut itfs = Vec::new();
    for k in 0..nk - 1 {
        if (vp[k], vs[k], rho[k]) != (vp[k + 1], vs[k + 1], rho[k + 1]) {
            let p = [vp[k], vs[k], rho[k], vp[k + 1], vs[k + 1], rho[k + 1]].map(|x| x as f64);
            let curve = |f| {
                (0..=50)
                    .map(|a| format!("{:e}", zoeppritz_pp_form(p[0], p[1], p[2], p[3], p[4], p[5], a as f64, f)))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            itfs.push(format!(
                "{{\"k\":{k},\"props\":[{}],\"exact\":[{}],\"legacy\":[{}]}}",
                p.map(|x| format!("{x}")).join(","),
                curve(ZoeppritzForm::Exact),
                curve(ZoeppritzForm::Legacy)
            ));
        }
    }
    meta += &format!(",\"column\":[{ci},{cj}],\"interfaces\":[{}]}}", itfs.join(","));
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}
