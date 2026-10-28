//! Before / after evidence of the bevel-and-ponded-fill unconformity
//! (erosion and onlap spec §7, §8): closure counts and brine / oil / gas
//! closure voxels, the largest horizon slope before and after the bevel,
//! and inline sections / maps for plotting (`examples/plot_unconformity_demo.py`).
//!
//! `cargo run --release -p synthoseis-core --example unconformity_demo -- sweep|slopes|OUT`
use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::pipeline_stream::{generate_chunked, toy_horizon_maps_continuous};
use synthoseis_core::rock_physics::{elastic_model, ElasticModel, Fluid};
use synthoseis_core::toy_geometry::layered_stack_records;
use synthoseis_core::unconformity::slope_max;
use synthoseis_core::{closure_census, generate_labels, unconformity_records};

fn f32s(path: PathBuf, v: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn fluids(c: &E2eConfig, labels: &[u8], sh: [usize; 3]) -> [usize; 3] {
    let mut v = [0usize; 3];
    if let ElasticModel::Rpm(m) = elastic_model(c, labels, sh) {
        for l in &m.layers {
            if let Some(f) = &l.fluids {
                for cl in &f.closures {
                    v[match cl.0 {
                        Fluid::Brine => 0,
                        Fluid::Oil => 1,
                        Fluid::Gas => 2,
                    }] += cl.4;
                }
            }
        }
    }
    v
}

fn pct(v: [usize; 3]) -> String {
    let t = v.iter().sum::<usize>().max(1) as f64;
    format!(
        "{:.1} / {:.1} / {:.1} %",
        100.0 * v[0] as f64 / t,
        100.0 * v[1] as f64 / t,
        100.0 * v[2] as f64 / t
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// 7e80f2af geometry (`--legacy-fold-geometry`).
    Legacy7e80,
    /// #56 geometry (`--legacy-unconformity`).
    Folds56,
    /// This PR's default.
    Unconf,
}

fn cfg(seed: u64, shape: [usize; 3], faults: usize, mode: Mode) -> E2eConfig {
    let mut c = E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(faults),
        ..E2eConfig::default()
    };
    c.rock_physics.legacy_fold_geometry = mode == Mode::Legacy7e80;
    c.rock_physics.legacy_unconformity = mode == Mode::Folds56;
    c
}

/// Largest slope (samples per column) of any horizon, before and after the bevel.
fn slopes() {
    println!("seed | shape | applied | s (from seabed) | steepest horizon before | after | erosion-surface relief m / drawn m | max eroded m");
    for (seed, nk) in [
        (1u64, 256usize),
        (4, 256),
        (7, 128),
        (12, 256),
        (2, 256),
        (3, 256),
        (5, 256),
        (6, 256),
    ] {
        let shape = [64, 64, nk];
        let c = cfg(seed, shape, 0, Mode::Unconf);
        let fold = c.rock_physics.fold_config();
        let u = c.rock_physics.unconformity_config();
        let (pre, _) = layered_stack_records(seed, shape, fold, None);
        let (post, recs) = layered_stack_records(seed, shape, fold, u);
        let g = |st: &[Vec<f64>]| st.iter().map(|h| slope_max(h, 64, 64)).fold(0.0, f64::max);
        let r = &recs[0];
        println!(
            "{seed} | 64x64x{nk} | {} | {} | {:.4} | {:.4} | {:.1} / {:.1} | {:.1}",
            r.applied,
            r.s,
            g(&pre),
            g(&post),
            r.relief,
            r.relief_drawn,
            r.max_eroded
        );
    }
}

fn sweep() {
    let mut tot = [[0usize; 3]; 3];
    let mut kept = [0usize; 3];
    let modes = [Mode::Legacy7e80, Mode::Folds56, Mode::Unconf];
    for seed in 1..=20u64 {
        let mut row = [0usize; 3];
        for (m, mode) in modes.iter().enumerate() {
            let c = cfg(seed, [64, 64, 256], 4, *mode);
            let (labels, sh) = generate_labels(&c);
            let v = fluids(&c, &labels, sh);
            for x in 0..3 {
                tot[m][x] += v[x];
            }
            row[m] = closure_census(&c, &labels, sh).unwrap().kept();
            kept[m] += row[m];
        }
        println!(
            "seed {seed}: kept closures 7e80 {} / #56 {} / unconformity {}",
            row[0], row[1], row[2]
        );
    }
    println!(
        "total kept: 7e80 {}, #56 {}, unconformity {}",
        kept[0], kept[1], kept[2]
    );
    for (m, n) in ["7e80", "#56", "unconformity"].iter().enumerate() {
        println!(
            "closure voxels brine / oil / gas, seeds 1-20, {n}: {:?} = {}",
            tot[m],
            pct(tot[m])
        );
    }
    for (seed, nk) in [(4u64, 256usize), (7, 128)] {
        for (m, mode) in modes.iter().enumerate() {
            let c = cfg(seed, [64, 64, nk], 4, *mode);
            let (labels, sh) = generate_labels(&c);
            let census = closure_census(&c, &labels, sh).unwrap();
            let v = fluids(&c, &labels, sh);
            println!(
                "seed {seed} 64x64x{nk} {}: kept {} of {} compartments; brine/oil/gas {:?} = {}",
                ["7e80", "#56", "unconformity"][m],
                census.kept(),
                census.sizes.len(),
                v,
                pct(v)
            );
        }
    }
}

/// Wall time of the label volume, the closure census and the stack for seed 4.
fn phases() {
    for (name, mode) in [("#56", Mode::Folds56), ("unconformity", Mode::Unconf)] {
        let c = cfg(4, [64, 64, 256], 4, mode);
        let t = std::time::Instant::now();
        let (labels, sh) = generate_labels(&c);
        let t_lab = t.elapsed();
        let t = std::time::Instant::now();
        let _ = closure_census(&c, &labels, sh).unwrap();
        let t_cen = t.elapsed();
        let t = std::time::Instant::now();
        let m = elastic_model(&c, &labels, sh);
        let t_mod = t.elapsed();
        let t = std::time::Instant::now();
        let _ = synthoseis_core::generate_unconformity_labels_output(&c);
        let t_ul = t.elapsed();
        let t = std::time::Instant::now();
        let _ = synthoseis_core::time_mode::generate_output_labels(&c, &labels, &m);
        let t_ol = t.elapsed();
        let t = std::time::Instant::now();
        let _ = generate_chunked(&c);
        let t_all = t.elapsed();
        println!("{name}: labels {t_lab:?}, census {t_cen:?}, elastic_model {t_mod:?}, unconf_labels_output {t_ul:?}, output_labels {t_ol:?}, generate_chunked {t_all:?}");
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("sweep") => return sweep(),
        Some("slopes") => return slopes(),
        Some("phases") => return phases(),
        _ => {}
    }
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "unconformity_demo".into()),
    );
    std::fs::create_dir_all(&out).unwrap();
    let mut json = Vec::new();
    for (name, seed, shape) in [
        ("seed4", 4u64, [64usize, 64, 256]),
        ("seed7", 7, [64, 64, 128]),
    ] {
        let srec = {
            let c = cfg(seed, shape, 4, Mode::Unconf);
            unconformity_records(&c)[0].s
        };
        for (tag, mode) in [("before", Mode::Folds56), ("after", Mode::Unconf)] {
            let c = cfg(seed, shape, 4, mode);
            let (labels, sh) = generate_labels(&c);
            let census = closure_census(&c, &labels, sh).unwrap();
            let v = fluids(&c, &labels, sh);
            json.push(format!(
                "\"{name}_{tag}\":{{\"kept\":{},\"compartments\":{},\"brine\":{},\"oil\":{},\"gas\":{},\"s\":{srec}}}",
                census.kept(),
                census.sizes.len(),
                v[0],
                v[1],
                v[2]
            ));
            println!(
                "{name} {tag}: kept {} of {}; brine/oil/gas {:?} = {}",
                census.kept(),
                census.sizes.len(),
                v,
                pct(v)
            );
            let [ni, nj, _] = shape;
            let i = ni / 2;
            let (maps, nh) = toy_horizon_maps_continuous(&c).unwrap();
            let hz: Vec<f32> = (0..nj)
                .flat_map(|j| (0..nh).map(move |h| (i * nj + j) * nh + h))
                .map(|x| maps[x] as f32)
                .collect();
            f32s(
                out.join(format!("{name}_{tag}_horizons_{nj}x{nh}.f32")),
                &hz,
            );
            let surf: Vec<f32> = (0..ni * nj).map(|c| maps[c * nh + srec] as f32).collect();
            f32s(
                out.join(format!("{name}_{tag}_map_surface_{ni}x{nj}.f32")),
                &surf,
            );
            let deep: Vec<f32> = (0..ni * nj)
                .map(|c| maps[c * nh + srec + 6] as f32)
                .collect();
            f32s(
                out.join(format!("{name}_{tag}_map_below_{ni}x{nj}.f32")),
                &deep,
            );
            let nk = sh[2];
            let lab: Vec<f32> = labels[i * nj * nk..(i + 1) * nj * nk]
                .iter()
                .map(|&v| v as f32)
                .collect();
            f32s(out.join(format!("{name}_{tag}_labels_{nj}x{nk}.f32")), &lab);
            let (vol, _) = generate_chunked(&c);
            let nt = vol.angle_stack.len() / (ni * nj);
            f32s(
                out.join(format!("{name}_{tag}_stack_{nj}x{nt}.f32")),
                &vol.angle_stack[i * nj * nt..(i + 1) * nj * nt],
            );
            if mode == Mode::Unconf {
                let recs = unconformity_records(&c);
                let r = &recs[0];
                let er: Vec<f32> = r.eroded.clone();
                f32s(out.join(format!("{name}_eroded_{ni}x{nj}.f32")), &er);
                println!("{name}: {}", r.summary());
            }
        }
    }
    std::fs::write(out.join("summary.json"), format!("{{{}}}", json.join(","))).unwrap();
}
