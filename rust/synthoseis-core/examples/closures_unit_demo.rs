//! Closures per sand unit (default) vs per sand layer
//! (`--closures-per-layer`, master 8b5988f) on the layered geometry with
//! the Markov lithology. Dumps data for
//! `examples/plot_closures_unit_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example closures_unit_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=256] [fraction=legacy] [thickness=2]
//! ```
//!
//! Writes little-endian C-order `(ni, nj, nk)` arrays per mode
//! (`{layer,unit}_{facies.u8,stack15.f32}`) and `meta.json`. Facies codes
//! are 0 water, 1 shale, 2 brine sand, 3 oil sand, 4 gas sand.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::lithology::{closure_units, interval_sand, sand_fraction, ToyLithology};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid, RockPhysicsConfig};
use synthoseis_core::{generate_chunked_at_angle, generate_labels, ToyGeometry};

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn write(path: PathBuf, bytes: &[u8]) {
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(bytes))
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "closures_unit_demo".into()),
    );
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 256);
    let fraction: Option<f64> = args.get(6).and_then(|s| s.parse().ok());
    let thickness: f64 = arg(&args, 7, 2.0);
    std::fs::create_dir_all(&out).expect("create out dir");
    let base = E2eConfig {
        geometry: ToyGeometry::Layered,
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        store_path: None,
        chunk_shape: Some([16, 16, nk]),
        faults: FaultConfig::with_count(count),
        filters: FilterConfig::default(),
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: fraction,
            sand_layer_thickness: thickness,
            ..RockPhysicsConfig::default()
        },
    };
    let (labels, shape) = generate_labels(&base);
    let mut parts = Vec::new();
    let mut nh = 0;
    for (name, per_layer) in [("layer", true), ("unit", false)] {
        let c = E2eConfig {
            rock_physics: RockPhysicsConfig {
                closures_per_layer: per_layer,
                ..base.rock_physics.clone()
            },
            ..base.clone()
        };
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
            panic!("rpm")
        };
        nh = m.nh;
        let mut facies = vec![0u8; labels.len()];
        for col in 0..ni * nj {
            let lc = &labels[col * nk..(col + 1) * nk];
            let Some(seabed) = lc.iter().position(|&l| l != 255) else {
                continue;
            };
            for k in seabed..nk {
                let lab = lc[k];
                let Some(l) = m.layers.get(lab as usize).filter(|_| lab != 255) else {
                    continue;
                };
                facies[col * nk + k] = if !l.sand {
                    1
                } else {
                    match &l.fluids {
                        Some(f) if (k as f32) < f.contact[col] => match f.fluid[col] {
                            Fluid::Brine => 2,
                            Fluid::Oil => 3,
                            Fluid::Gas => 4,
                        },
                        _ => 2,
                    }
                };
            }
        }
        write(out.join(format!("{name}_facies.u8")), &facies);
        let (v, _) = generate_chunked_at_angle(&c, 15.0);
        write(
            out.join(format!("{name}_stack15.f32")),
            &v.angle_stack
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let mut cl = [0usize; 3];
        // (interval of the closure's layer / unit top, fluid code, voxels)
        let mut list = Vec::new();
        for l in &m.layers {
            for x in l.fluids.iter().flat_map(|f| f.closures.iter()) {
                cl[x.0 as usize] += 1;
                list.push(format!("[{},{},{}]", l.interval, x.0 as u8, x.4));
            }
        }
        parts.push(format!(
            "\"{name}\":{{\"closures\":{cl:?},\"list\":[{}],\"hc_voxels\":{}}}",
            list.join(","),
            facies.iter().filter(|&&f| f >= 3).count()
        ));
    }
    let sand = interval_sand(ToyLithology::Markov, seed, nh, fraction, thickness);
    let units: Vec<String> = closure_units(&sand)
        .iter()
        .map(|(a, b)| format!("[{a},{b}]"))
        .collect();
    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"sand_fraction\":{:.5},\"thickness\":{thickness},\"sand\":[{}],\"units\":[{}],{}}}",
        sand_fraction(seed, fraction),
        sand.iter().map(|&s| (s as u8).to_string()).collect::<Vec<_>>().join(","),
        units.join(","),
        parts.join(",")
    );
    write(out.join("meta.json"), meta.as_bytes());
    // Rust closure-unit population (as in tests/closure_units.rs).
    let (mut n_units, mut thick) = (Vec::new(), Vec::new());
    for s in 0..3000u64 {
        let u = closure_units(&interval_sand(ToyLithology::Markov, s, 40, None, 2.0));
        n_units.push(u.len().to_string());
        thick.extend(u.iter().map(|(a, b)| (b - a).to_string()));
    }
    write(
        out.join("rust_population.json"),
        format!(
            "{{\"n_units\":[{}],\"unit_thickness\":[{}]}}",
            n_units.join(","),
            thick.join(",")
        )
        .as_bytes(),
    );
    println!("{meta}");
}
