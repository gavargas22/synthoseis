//! 3D closure segmentation across faults (default) vs unsegmented closures
//! per sand unit (`--closures-unsegmented`, master ef2dc42) on the layered
//! geometry with the Markov lithology. Dumps data for
//! `examples/plot_closure_segments_demo.py`.
//!
//! ```text
//! cargo run --release -p synthoseis-core --example closure_segments_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=256] [fraction=legacy] [thickness=default]
//! cargo run --release -p synthoseis-core --example closure_segments_demo -- stats OUT_DIR [models=2000]
//! ```
//!
//! Demo mode writes little-endian C-order `(ni, nj, nk)` arrays per mode
//! (`{unseg,seg}_{facies.u8,comp.i32}`, `{unseg,seg}_stack15.f32`) and
//! `meta.json`. Facies codes are 0 water, 1 shale, 2 brine sand, 3 oil sand,
//! 4 gas sand; `comp` is the closure id of a closure voxel (unseg: 2D region
//! per unit; seg: 3D compartment), else -1.
//!
//! Stats mode writes `stats.json`: fluid counts of split-off compartment
//! draws (direct, 2M keys; with the primary draw of the same closure), and
//! of the kept compartments of faulted sandy models.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::closure_segments::{
    segment_runs, segmented_sand_unit_fluids, split_compartment_fluid, unit_closure_runs,
};
use synthoseis_core::lithology::{closure_units, interval_sand};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};
use synthoseis_core::rock_physics::{
    closure_fluid, elastic_model, ElasticModel, RockPhysicsConfig, RpmModel,
};
use synthoseis_core::{generate_chunked_at_angle, generate_labels, ToyGeometry};

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn write(path: PathBuf, bytes: &[u8]) {
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(bytes))
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn config(
    seed: u64,
    faults: usize,
    shape: [usize; 3],
    fraction: Option<f64>,
    thickness: f64,
) -> E2eConfig {
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
            sand_layer_fraction: fraction,
            sand_layer_thickness: thickness,
            ..RockPhysicsConfig::default()
        },
    }
}

/// Closure runs of every closure unit, with a unit index per run.
fn all_runs(
    labels: &[u8],
    shape: [usize; 3],
    m: &RpmModel,
    sand: &[bool],
    max_column: f64,
) -> Vec<(usize, synthoseis_core::closure_segments::ClosureRun)> {
    let mut out = Vec::new();
    for (u, (top, end)) in closure_units(sand).into_iter().enumerate() {
        let mut members: Vec<(usize, usize)> = m
            .intervals
            .iter()
            .enumerate()
            .filter(|&(lab, &h)| lab < 255 && h >= top && h < end)
            .map(|(lab, &h)| (h, lab))
            .collect();
        members.sort_unstable();
        let ids: Vec<u8> = members.iter().map(|&(_, lab)| lab as u8).collect();
        if ids.is_empty() {
            continue;
        }
        for r in unit_closure_runs(labels, shape, &ids, top, max_column) {
            out.push((u, r));
        }
    }
    out
}

/// Facies codes, and whether each voxel is in a (kept) closure.
fn facies(labels: &[u8], shape: [usize; 3], m: &RpmModel) -> (Vec<u8>, Vec<bool>) {
    let [ni, nj, nk] = shape;
    let mut f = vec![0u8; labels.len()];
    let mut inside = vec![false; labels.len()];
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
            f[col * nk + k] = if !l.sand {
                1
            } else {
                match &l.fluids {
                    Some(x) if (k as f32) < x.contact[col] => {
                        inside[col * nk + k] = true;
                        2 + x.fluid[col] as u8
                    }
                    _ => 2,
                }
            };
        }
    }
    (f, inside)
}

fn demo(args: &[String]) {
    let out = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "closure_segments_demo".into()),
    );
    let seed: u64 = arg(args, 1, 7);
    let faults: usize = arg(args, 2, 4);
    let shape = [arg(args, 3, 64usize), arg(args, 4, 64), arg(args, 5, 256)];
    let fraction: Option<f64> = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .filter(|&f: &f64| f > 0.0);
    let thickness: f64 = arg(args, 7, RockPhysicsConfig::default().sand_layer_thickness);
    std::fs::create_dir_all(&out).expect("create out dir");
    let base = config(seed, faults, shape, fraction, thickness);
    let (labels, sh) = generate_labels(&base);
    let rp = base.rock_physics.clone();
    let max_column = rp.max_column_m / rp.depth_step_m;
    let n = labels.len();
    let nk = shape[2];
    let mut parts = Vec::new();
    let mut stacks: Vec<Vec<Vec<f32>>> = Vec::new();
    for (name, unseg) in [("unseg", true), ("seg", false)] {
        let c = E2eConfig {
            rock_physics: RockPhysicsConfig {
                closures_unsegmented: unseg,
                ..rp.clone()
            },
            ..base.clone()
        };
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, sh) else {
            panic!("rpm")
        };
        let sand = interval_sand(c.effective_lithology(), seed, m.nh, fraction, thickness);
        let (f, inside) = facies(&labels, sh, &m);
        write(out.join(format!("{name}_facies.u8")), &f);
        let runs = all_runs(&labels, sh, &m, &sand, max_column);
        let ids: Vec<usize> = if unseg {
            // 2D region per unit.
            let mut keys: Vec<(usize, u64)> = runs.iter().map(|(_, r)| (r.layer, r.rank)).collect();
            keys.sort_unstable();
            keys.dedup();
            runs.iter()
                .map(|(_, r)| keys.binary_search(&(r.layer, r.rank)).unwrap())
                .collect()
        } else {
            segment_runs(
                &runs
                    .iter()
                    .map(|(_, r)| (r.col, r.k0, r.k1))
                    .collect::<Vec<_>>(),
                sh[0],
                sh[1],
            )
        };
        let mut comp = vec![-1i32; n];
        for ((_, r), &id) in runs.iter().zip(&ids) {
            // Only closure voxels that carry a closure fluid (kept).
            for k in r.k0..r.k1 {
                if inside[r.col * nk + k] {
                    comp[r.col * nk + k] = id as i32;
                }
            }
        }
        write(
            out.join(format!("{name}_comp.i32")),
            &comp
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let mut cl = [0usize; 3];
        for x in m
            .layers
            .iter()
            .flat_map(|l| l.fluids.iter().flat_map(|f| f.closures.iter()))
        {
            cl[x.0 as usize] += 1;
        }
        let mut extra = String::new();
        if !unseg {
            let (_, comps) = segmented_sand_unit_fluids(
                &labels,
                sh,
                &m.intervals,
                &sand,
                seed,
                max_column,
                rp.min_closure_voxels,
            );
            let kept: Vec<_> = comps.iter().filter(|k| k.kept).collect();
            let mut fl = [0usize; 3];
            for k in &kept {
                fl[k.fluid as usize] += 1;
            }
            extra = format!(
                ",\"compartments\":{},\"compartment_fluids\":{fl:?},\"split_off\":{},\"multi_piece\":{},\"multi_unit\":{},\"removed\":{}",
                kept.len(),
                kept.iter().filter(|k| !k.primary).count(),
                kept.iter().filter(|k| k.pieces > 1).count(),
                kept.iter().filter(|k| k.units > 1).count(),
                comps.len() - kept.len()
            );
        }
        let regions = {
            let mut keys: Vec<(usize, u64)> = runs.iter().map(|(_, r)| (r.layer, r.rank)).collect();
            keys.sort_unstable();
            keys.dedup();
            keys.len()
        };
        let hc = f.iter().filter(|&&x| x >= 3).count();
        let closure_vox = inside.iter().filter(|&&x| x).count();
        parts.push(format!(
            "\"{name}\":{{\"closures\":{cl:?},\"regions_2d\":{regions},\"hc_voxels\":{hc},\"oil_voxels\":{},\"gas_voxels\":{},\"closure_voxels\":{closure_vox}{extra}}}",
            f.iter().filter(|&&x| x == 3).count(),
            f.iter().filter(|&&x| x == 4).count(),
        ));
        let mut st = Vec::new();
        for angle in [0.0, 15.0, 30.0] {
            st.push(generate_chunked_at_angle(&c, angle).0.angle_stack);
        }
        write(
            out.join(format!("{name}_stack15.f32")),
            &st[1]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
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
        "{{\"shape\":{shape:?},\"seed\":{seed},\"faults\":{faults},\"fraction\":{},\"thickness\":{thickness},{},\"stack_change\":{{\"0\":{},\"15\":{},\"30\":{}}}}}",
        fraction.map_or("null".into(), |f| f.to_string()),
        parts.join(","),
        rel[0],
        rel[1],
        rel[2]
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}

fn stats(args: &[String]) {
    let out = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "closure_segments_stats".into()),
    );
    let models: u64 = arg(args, 1, 2000);
    std::fs::create_dir_all(&out).expect("create out dir");
    // Direct draws: 2M split-off keys.
    let mut split = [0u64; 3];
    let mut table = [[0u64; 3]; 3];
    for seed in 0..50u64 {
        for layer in 0..40usize {
            for rank in 0..10u64 {
                for col in (0..4096usize).step_by(41) {
                    let s = split_compartment_fluid(seed, layer, rank, col) as usize;
                    split[s] += 1;
                    table[closure_fluid(seed, layer, rank) as usize][s] += 1;
                }
            }
        }
    }
    // Model level: kept compartments of faulted sandy models.
    let (mut all, mut split_off, mut multi_unit, mut multi_piece) =
        ([0u64; 3], [0u64; 3], [0u64; 3], [0u64; 3]);
    for seed in 0..models {
        let c = config(seed, 4, [24, 20, 128], Some(0.5), 1.0);
        let (labels, sh) = generate_labels(&c);
        let ElasticModel::Rpm(m) = elastic_model(&c, &labels, sh) else {
            panic!()
        };
        let sand = interval_sand(c.effective_lithology(), seed, m.nh, Some(0.5), 1.0);
        let rp = &c.rock_physics;
        let (_, comps) = segmented_sand_unit_fluids(
            &labels,
            sh,
            &m.intervals,
            &sand,
            seed,
            rp.max_column_m / rp.depth_step_m,
            rp.min_closure_voxels,
        );
        for k in comps.iter().filter(|k| k.kept) {
            let f = k.fluid as usize;
            all[f] += 1;
            if !k.primary {
                split_off[f] += 1;
            }
            if k.units > 1 {
                multi_unit[f] += 1;
            }
            if k.pieces > 1 {
                multi_piece[f] += 1;
            }
        }
    }
    let s = format!(
        "{{\"direct_split\":{split:?},\"split_vs_primary\":{table:?},\"models\":{models},\"model\":{{\"shape\":[24,20,128],\"faults\":4,\"fraction\":0.5,\"thickness\":1,\"all\":{all:?},\"split_off\":{split_off:?},\"multi_unit\":{multi_unit:?},\"multi_piece\":{multi_piece:?}}}}}"
    );
    write(out.join("stats.json"), s.as_bytes());
    println!("{s}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("stats") {
        stats(&args[1..]);
    } else {
        demo(&args);
    }
}
