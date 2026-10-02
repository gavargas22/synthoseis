//! Salt bodies (default) vs `--no-salt` (master b4f4259) on the layered
//! geometry. Dumps data for `examples/plot_salt_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example salt_demo -- OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=256]
//! cargo run --release -p synthoseis-core --example salt_demo -- stats OUT_DIR [models=16000]
//! ```
//!
//! Demo mode writes little-endian C-order `(ni, nj, nk)` arrays per mode
//! (`{nosalt,salt}_facies.u8`, `{nosalt,salt}_labels.u8`,
//! `{nosalt,salt}_stack{0,15,30}.f32`), `salt_mask.u8` and `meta.json`.
//! Facies codes are 0 water, 1 shale, 2 brine sand, 3 oil sand, 4 gas sand,
//! 5 salt.
//!
//! Stats mode writes `population.json`: 10 shape statistics of `models`
//! Rust salt bodies on the legacy example grid (64 x 64 x 1250 + pad, flat
//! horizon 1 at 20 samples), keyed like `tests/salt.rs`.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, RockPhysicsConfig, RpmModel};
use synthoseis_core::salt::{keyed_draws, salt_body, salt_geometry, SALT_PAD};
use synthoseis_core::{generate_chunked_at_angle, generate_labels, ToyGeometry};

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

fn config(seed: u64, faults: usize, shape: [usize; 3], salt: bool) -> E2eConfig {
    E2eConfig {
        geometry: ToyGeometry::Layered,
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some([16, 16, shape[2]]),
        faults: FaultConfig::with_count(faults),
        filters: FilterConfig::default(),
        rock_physics: RockPhysicsConfig {
            salt,
            ..RockPhysicsConfig::default()
        },
    }
}

/// Facies codes (5 = salt).
fn facies(labels: &[u8], shape: [usize; 3], m: &RpmModel) -> Vec<u8> {
    let [ni, nj, nk] = shape;
    let mut f = vec![0u8; labels.len()];
    for col in 0..ni * nj {
        let lc = &labels[col * nk..(col + 1) * nk];
        let seabed = lc.iter().position(|&l| l != 255).unwrap_or(nk);
        for k in 0..nk {
            let g = col * nk + k;
            if m.salt.as_ref().is_some_and(|s| s.contains(col, k)) {
                f[g] = 5;
                continue;
            }
            if k < seabed {
                continue;
            }
            let lab = lc[k];
            let Some(l) = m.layers.get(lab as usize).filter(|_| lab != 255) else {
                continue;
            };
            f[g] = if !l.sand {
                1
            } else {
                match &l.fluids {
                    Some(x) if (k as f32) < x.contact[col] => 2 + x.fluid[col] as u8,
                    _ => 2,
                }
            };
        }
    }
    f
}

fn demo(args: &[String]) {
    let out = PathBuf::from(args.first().cloned().unwrap_or_else(|| "salt_demo".into()));
    let seed: u64 = arg(args, 1, 7);
    let faults: usize = arg(args, 2, 4);
    let shape = [arg(args, 3, 64usize), arg(args, 4, 64), arg(args, 5, 256)];
    std::fs::create_dir_all(&out).expect("create out dir");
    let [ni, nj, nk] = shape;
    let mut parts = Vec::new();
    let mut stacks: Vec<Vec<Vec<f32>>> = Vec::new();
    for (name, salt) in [("nosalt", false), ("salt", true)] {
        let c = config(seed, faults, shape, salt);
        let (labels, sh) = generate_labels(&c);
        write(out.join(format!("{name}_labels.u8")), &labels);
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, sh) else {
            panic!("rpm")
        };
        let f = facies(&labels, sh, &m);
        write(out.join(format!("{name}_facies.u8")), &f);
        let mut cl = [0usize; 3];
        for x in m
            .layers
            .iter()
            .flat_map(|l| l.fluids.iter().flat_map(|f| f.closures.iter()))
        {
            cl[x.0 as usize] += 1;
        }
        let mut extra = String::new();
        if let Some(b) = salt_body(&c) {
            let mut mask = vec![0u8; ni * nj * nk];
            for col in 0..ni * nj {
                b.column_mask(col, &mut mask[col * nk..(col + 1) * nk]);
            }
            write(out.join("salt_mask.u8"), &mask);
            // Sand-unit tops that fall inside salt (legacy horizon gaps).
            extra = format!(
                ",\"salt_top\":{:.3},\"salt_radius\":{:.3},\"salt_voxels\":{},\"salt_columns\":{}",
                b.top,
                b.radius,
                b.voxels(nk),
                b.columns(nk)
            );
        }
        parts.push(format!(
            "\"{name}\":{{\"horizons\":{},\"closures\":{cl:?},\"hc_voxels\":{},\"oil_voxels\":{},\"gas_voxels\":{}{extra}}}",
            m.nh,
            f.iter().filter(|&&x| x == 3 || x == 4).count(),
            f.iter().filter(|&&x| x == 3).count(),
            f.iter().filter(|&&x| x == 4).count(),
        ));
        let mut st = Vec::new();
        for angle in [0.0, 15.0, 30.0] {
            let s = generate_chunked_at_angle(&c, angle).0.angle_stack;
            write(
                out.join(format!("{name}_stack{}.f32", angle as u32)),
                &f32_bytes(&s),
            );
            st.push(s);
        }
        stacks.push(st);
    }
    let rel: Vec<String> = (0..3)
        .map(|a| {
            let (u, s) = (&stacks[0][a], &stacks[1][a]);
            let d: f64 = u.iter().zip(s).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let e: f64 = u.iter().map(|x| (*x as f64).powi(2)).sum();
            let changed = u
                .iter()
                .zip(s)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            format!(
                "{{\"rel_rms\":{:.6},\"changed_samples\":{changed}}}",
                (d / e).sqrt()
            )
        })
        .collect();
    let meta = format!(
        "{{\"shape\":{shape:?},\"seed\":{seed},\"faults\":{faults},{},\"stack_change\":{{\"0\":{},\"15\":{},\"30\":{}}}}}",
        parts.join(","),
        rel[0],
        rel[1],
        rel[2]
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}

/// Same statistics as `tests/salt.rs::rust_population`.
fn stats(args: &[String]) {
    let out = PathBuf::from(args.first().cloned().unwrap_or_else(|| "salt_stats".into()));
    let models: u64 = arg(args, 1, 16_000);
    std::fs::create_dir_all(&out).expect("create out dir");
    let grid = [64, 64, 1250 + SALT_PAD];
    let h1 = vec![20.0; 64 * 64];
    let names = [
        "radius", "top", "tip", "cx", "cy", "r1", "base", "bx", "by", "r2",
    ];
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 10];
    for s in 0..models {
        let mut draw = keyed_draws(0x5A17_0000 + s);
        let (radius, top, p) = salt_geometry(&h1, grid, 1.0, &mut draw);
        let st = |c: &[[f64; 3]]| {
            let mx = c.iter().map(|q| q[0]).sum::<f64>() / c.len() as f64;
            let my = c.iter().map(|q| q[1]).sum::<f64>() / c.len() as f64;
            let r = c.iter().map(|q| (q[0] - mx).hypot(q[1] - my)).sum::<f64>() / c.len() as f64;
            let z = c.iter().map(|q| q[2]).sum::<f64>() / c.len() as f64;
            (mx, my, r, z)
        };
        let (cx, cy, r1, _) = st(&p[0..36]);
        let (bx, by, r2, base) = st(&p[109..145]);
        for (k, v) in [radius, top, p[108][2], cx, cy, r1, base, bx, by, r2]
            .into_iter()
            .enumerate()
        {
            cols[k].push(v);
        }
    }
    let body: Vec<String> = names
        .iter()
        .zip(&cols)
        .map(|(n, v)| format!("\"{n}\":{:?}", v))
        .collect();
    write(
        out.join("population.json"),
        format!("{{{}}}", body.join(",")).as_bytes(),
    );
    println!(
        "wrote {} ({models} models)",
        out.join("population.json").display()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("stats") {
        stats(&args[1..]);
    } else {
        demo(&args);
    }
}
