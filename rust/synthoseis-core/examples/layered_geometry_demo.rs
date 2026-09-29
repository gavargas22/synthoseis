//! Layered toy geometry demo (the default since this change) vs the master
//! planar geometry: labels, per-voxel facies/fluid, Vp, 15 deg reflectivity,
//! 0/15/30 deg angle stacks, what the random depth shifts and the closure
//! fluids each change. Dumps data for `examples/plot_layered_geometry_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example layered_geometry_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=256]
//! ```
//!
//! Writes little-endian C-order `(ni, nj, nk)` arrays (`*.u8`, `*.f32`),
//! `(ni, nj, nh)` f64 horizon maps and `meta.json`.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid, RockPhysicsConfig};
use synthoseis_core::toy_geometry::LayeredParams;
use synthoseis_core::{
    generate_chunked_at_angle, generate_labels, generate_reflectivity, ToyGeometry,
};

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

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        r += (y as f64).powi(2);
    }
    (d / r.max(1e-300)).sqrt()
}

fn changed(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as f64 / a.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "layered_demo".into()),
    );
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 256);
    std::fs::create_dir_all(&out).expect("create out dir");
    let layered = E2eConfig {
        geometry: ToyGeometry::Layered,
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
    let planar = E2eConfig {
        geometry: ToyGeometry::Planar,
        ..layered.clone()
    };
    let no_shifts = E2eConfig {
        rock_physics: RockPhysicsConfig {
            first_random_layer: 10_000,
            ..RockPhysicsConfig::default()
        },
        ..layered.clone()
    };
    let no_fluids = E2eConfig {
        rock_physics: RockPhysicsConfig {
            fluids: false,
            ..RockPhysicsConfig::default()
        },
        ..layered.clone()
    };

    let (labels, shape) = generate_labels(&layered);
    let (plabels, _) = generate_labels(&planar);
    write(out.join("labels.u8"), &labels);
    write(out.join("planar_labels.u8"), &plabels);
    let ElasticModel::Rpm(m) = elastic_model(&layered, &labels, shape) else {
        panic!("rpm model")
    };
    write(
        out.join("maps.f64"),
        &m.maps
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );

    // Vp and per-voxel facies: 0 water, 1 shale, 2 brine sand, 3 oil, 4 gas.
    let n = ni * nj * nk;
    let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    ElasticModel::Rpm(m.clone())
        .tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    write(out.join("vp.f32"), &f32_bytes(&vp));
    let mut facies = vec![0u8; n];
    for c in 0..ni * nj {
        let col = &labels[c * nk..(c + 1) * nk];
        let Some(seabed) = col.iter().position(|&l| l != 255) else {
            continue;
        };
        for k in seabed..nk {
            let lab = col[k];
            let Some(l) = m.layers.get(lab as usize).filter(|_| lab != 255) else {
                continue;
            };
            facies[c * nk + k] = if !l.sand {
                1
            } else {
                match &l.fluids {
                    Some(f) if (k as f32) < f.contact[c] => match f.fluid[c] {
                        Fluid::Brine => 2,
                        Fluid::Oil => 3,
                        Fluid::Gas => 4,
                    },
                    _ => 2,
                }
            };
        }
    }
    write(out.join("facies.u8"), &facies);

    let r15 = generate_reflectivity(&layered, 15.0);
    let pr15 = generate_reflectivity(&planar, 15.0);
    let ns15 = generate_reflectivity(&no_shifts, 15.0);
    write(out.join("rfc15.f32"), &f32_bytes(&r15));
    write(out.join("planar_rfc15.f32"), &f32_bytes(&pr15));
    write(out.join("noshift_rfc15.f32"), &f32_bytes(&ns15));
    for a in [0.0, 15.0, 30.0] {
        let (v, _) = generate_chunked_at_angle(&layered, a);
        write(
            out.join(format!("stack{a:.0}.f32")),
            &f32_bytes(&v.angle_stack),
        );
    }
    let (p15, _) = generate_chunked_at_angle(&planar, 15.0);
    let (nf15, _) = generate_chunked_at_angle(&no_fluids, 15.0);
    let (l15, _) = generate_chunked_at_angle(&layered, 15.0);
    write(out.join("planar_stack15.f32"), &f32_bytes(&p15.angle_stack));
    write(
        out.join("nofluid_stack15.f32"),
        &f32_bytes(&nf15.angle_stack),
    );

    let p = LayeredParams::new(seed, shape);
    let layers: Vec<String> = m
        .layers
        .iter()
        .map(|l| {
            let closures: Vec<String> = l
                .fluids
                .iter()
                .flat_map(|f| f.closures.iter())
                .map(|c| {
                    format!(
                        "{{\"fluid\":\"{:?}\",\"crest\":{:.2},\"contact\":{:.2},\"columns\":{},\"voxels\":{}}}",
                        c.0, c.1, c.2, c.3, c.4
                    )
                })
                .collect();
            format!(
                "{{\"interval\":{},\"sand\":{},\"shift\":{},\"closures\":[{}]}}",
                l.interval,
                l.sand,
                l.shifts.layer,
                closures.join(",")
            )
        })
        .collect();
    let nz = |r: &[f32]| r.iter().filter(|&&x| x != 0.0).count() as f64 / r.len() as f64;
    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"nh\":{},\
         \"params\":{{\"seabed_min\":{},\"dome_center\":[{:.3},{:.3}],\"dome_radius\":{:.3},\"dome_amp\":{:.3},\"tilt\":[{:.5},{:.5}],\"growth\":{:.5}}},\
         \"model_bytes\":{},\"rfc15_nonzero\":{:.5},\"planar_rfc15_nonzero\":{:.5},\
         \"shift_effect\":{{\"rfc15_rel_rms\":{:.5},\"changed\":{:.5}}},\
         \"fluid_effect\":{{\"stack15_rel_rms\":{:.5},\"changed\":{:.5}}},\
         \"planar_labels_distinct\":{},\"layers\":[{}]}}",
        m.nh,
        p.seabed_min,
        p.dome_center[0],
        p.dome_center[1],
        p.dome_radius,
        p.dome_amp,
        p.tilt[0],
        p.tilt[1],
        p.growth,
        ElasticModel::Rpm(m.clone()).model_bytes(),
        nz(&r15),
        nz(&pr15),
        rel_rms(&ns15, &r15),
        changed(&ns15, &r15),
        rel_rms(&nf15.angle_stack, &l15.angle_stack),
        changed(&nf15.angle_stack, &l15.angle_stack),
        {
            let mut s: Vec<u8> = plabels.iter().copied().filter(|&l| l != 255).collect();
            s.sort_unstable();
            s.dedup();
            s.len()
        },
        layers.join(",")
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}
