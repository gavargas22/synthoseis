//! Before / after evidence of the analytic folds (folds spec §7, §8 test 7):
//! closure counts and brine / oil / gas closure voxels, and inline sections
//! of the continuous horizons, labels and 15° stack for plotting
//! (`examples/plot_folds_demo.py`).
//!
//! `cargo run --release -p synthoseis-core --example folds_demo -- OUT`
use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::pipeline_stream::{generate_chunked, toy_horizon_maps_continuous};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid};
use synthoseis_core::{closure_census, generate_labels, FoldStyle};

fn f32s(path: PathBuf, v: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn sweep() {
    let (mut a, mut b, mut more, mut less) = (0usize, 0usize, 0, 0);
    for seed in 1..=20u64 {
        let k = |legacy: bool| {
            let mut c = E2eConfig {
                seed,
                inline_count: 64,
                crossline_count: 64,
                samples: 256,
                faults: FaultConfig::with_count(4),
                ..E2eConfig::default()
            };
            c.rock_physics.legacy_fold_geometry = legacy;
            let (labels, sh) = generate_labels(&c);
            closure_census(&c, &labels, sh).unwrap().kept()
        };
        let (l, f) = (k(true), k(false));
        println!("seed {seed}: kept closures legacy {l} folded {f}");
        a += l;
        b += f;
        if f > l {
            more += 1
        }
        if f < l {
            less += 1
        }
    }
    println!("total kept: legacy {a}, folded {b}; folded more on {more} seeds, fewer on {less}");
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("sweep") {
        return sweep();
    }
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "folds_demo".into()),
    );
    std::fs::create_dir_all(&out).unwrap();
    let mut json = Vec::new();
    for (name, seed, shape, faults) in [
        ("seed4", 4u64, [64usize, 64, 256], 4usize),
        ("seed7", 7, [64, 64, 128], 4),
    ] {
        for (tag, legacy, style) in [
            ("legacy", true, FoldStyle::Similar),
            ("similar", false, FoldStyle::Similar),
            ("parallel", false, FoldStyle::Parallel),
        ] {
            let mut c = E2eConfig {
                seed,
                inline_count: shape[0],
                crossline_count: shape[1],
                samples: shape[2],
                faults: FaultConfig::with_count(faults),
                ..E2eConfig::default()
            };
            c.rock_physics.legacy_fold_geometry = legacy;
            c.rock_physics.fold.style = style;
            let (labels, sh) = generate_labels(&c);
            let census = closure_census(&c, &labels, sh).unwrap();
            let (mut brine, mut oil, mut gas, mut ncl) = (0usize, 0usize, 0usize, 0usize);
            if let ElasticModel::Rpm(m) = elastic_model(&c, &labels, sh) {
                for l in &m.layers {
                    if let Some(f) = &l.fluids {
                        for cl in &f.closures {
                            ncl += 1;
                            match cl.0 {
                                Fluid::Brine => brine += cl.4,
                                Fluid::Oil => oil += cl.4,
                                Fluid::Gas => gas += cl.4,
                            }
                        }
                    }
                }
            }
            let sediment = labels.iter().filter(|&&v| v != 255).count();
            let tot = (brine + oil + gas).max(1) as f64;
            println!(
                "{name} {tag}: closures kept {} of {} compartments; fluid closures {ncl}; closure voxels brine {brine} oil {oil} gas {gas} = {:.1} / {:.1} / {:.1} % of closure voxels; HC {:.3} % of sediment",
                census.kept(),
                census.sizes.len(),
                100.0 * brine as f64 / tot,
                100.0 * oil as f64 / tot,
                100.0 * gas as f64 / tot,
                100.0 * (oil + gas) as f64 / sediment as f64
            );
            json.push(format!(
                "\"{name}_{tag}\":{{\"kept\":{},\"compartments\":{},\"brine\":{brine},\"oil\":{oil},\"gas\":{gas},\"sediment\":{sediment}}}",
                census.kept(),
                census.sizes.len()
            ));
            // Inline section i = ni/2: continuous horizons (nj × nh), labels
            // (nj × nk), 15° stack (nj × nt).
            let i = shape[0] / 2;
            let (maps, nh) = toy_horizon_maps_continuous(&c).unwrap();
            let nj = shape[1];
            let hz: Vec<f32> = (0..nj)
                .flat_map(|j| (0..nh).map(move |h| (i * nj + j) * nh + h))
                .map(|x| maps[x] as f32)
                .collect();
            f32s(
                out.join(format!("{name}_{tag}_horizons_{nj}x{nh}.f32")),
                &hz,
            );
            let nk = sh[2];
            let lab: Vec<f32> = labels[i * nj * nk..(i + 1) * nj * nk]
                .iter()
                .map(|&v| v as f32)
                .collect();
            f32s(out.join(format!("{name}_{tag}_labels_{nj}x{nk}.f32")), &lab);
            let (v, _) = generate_chunked(&c);
            let nt = v.angle_stack.len() / (shape[0] * nj);
            f32s(
                out.join(format!("{name}_{tag}_stack_{nj}x{nt}.f32")),
                &v.angle_stack[i * nj * nt..(i + 1) * nj * nt],
            );
        }
    }
    std::fs::write(out.join("summary.json"), format!("{{{}}}", json.join(","))).unwrap();
}
