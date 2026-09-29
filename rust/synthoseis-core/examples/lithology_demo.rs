//! Toy lithology demo: the previous alternating shale/sand layers
//! (`--toy-lithology alternating`) vs the legacy sand-fraction Markov chain
//! (default) on the layered geometry. Dumps data for
//! `examples/plot_lithology_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example lithology_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=256]
//! ```
//!
//! Writes little-endian C-order `(ni, nj, nk)` arrays per lithology
//! (`{alt,markov}_{facies.u8,rfc15.f32,stack15.f32}`) and `meta.json`.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::lithology::{interval_sand, sand_fraction, ToyLithology};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid, RockPhysicsConfig};
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "lithology_demo".into()),
    );
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 256);
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
        rock_physics: RockPhysicsConfig::default(),
    };
    let (labels, shape) = generate_labels(&base);
    write(out.join("labels.u8"), &labels);
    let mut parts = Vec::new();
    for (name, lith) in [
        ("alt", ToyLithology::Alternating),
        ("markov", ToyLithology::Markov),
    ] {
        let c = E2eConfig {
            rock_physics: RockPhysicsConfig {
                lithology: lith,
                ..RockPhysicsConfig::default()
            },
            ..base.clone()
        };
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
            panic!("rpm")
        };
        // 0 water, 1 shale, 2 brine sand, 3 oil sand, 4 gas sand.
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
        write(
            out.join(format!("{name}_rfc15.f32")),
            &f32_bytes(&generate_reflectivity(&c, 15.0)),
        );
        for a in [15.0, 30.0] {
            let (v, _) = generate_chunked_at_angle(&c, a);
            write(
                out.join(format!("{name}_stack{a:.0}.f32")),
                &f32_bytes(&v.angle_stack),
            );
        }
        let sand: Vec<String> = m
            .layers
            .iter()
            .map(|l| (l.sand as u8).to_string())
            .collect();
        let mut cl = [0usize; 3];
        for l in &m.layers {
            for x in l.fluids.iter().flat_map(|f| f.closures.iter()) {
                cl[x.0 as usize] += 1;
            }
        }
        let sand_vox = facies.iter().filter(|&&f| f >= 2).count();
        let sed_vox = facies.iter().filter(|&&f| f >= 1).count();
        parts.push(format!(
            "\"{name}\":{{\"sand\":[{}],\"closures\":{cl:?},\"sand_voxel_fraction\":{:.5},\"hc_voxels\":{}}}",
            sand.join(","),
            sand_vox as f64 / sed_vox.max(1) as f64,
            facies.iter().filter(|&&f| f >= 3).count()
        ));
    }
    // Population draws for the plot (same as tests/lithology.rs).
    let pop: Vec<String> = (0..3000u64)
        .map(|s| {
            let v = interval_sand(ToyLithology::Markov, s, 60, None, 2.0);
            v.iter().filter(|&&x| x).count().to_string()
        })
        .collect();
    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"sand_fraction\":{:.5},{},\"population_sand_count\":[{}]}}",
        sand_fraction(seed, None),
        parts.join(","),
        pop.join(",")
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!(
        "{}",
        &meta[..meta.find("\"population").unwrap_or(meta.len())]
    );
}
