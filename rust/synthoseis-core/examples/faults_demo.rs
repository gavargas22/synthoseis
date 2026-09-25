//! Faulted-cube demo: runs the chunked e2e pipeline with faults enabled and
//! dumps raw volumes for plotting (`examples/plot_faults_demo.py`).
//!
//! ```text
//! cargo run --release -p synthoseis-core --example faults_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=96] [nj=96] [nk=128]
//! ```
//!
//! Writes little-endian raw arrays (C order, shape in `meta.json`):
//! `labels.u8` (faulted layer labels), `labels_unfaulted.u8`, `fault_mask.u8`,
//! `fault_ids.u8`, `angle_stack.f32` (15° fused stack from faulted labels),
//! `age.f32` (a layer-cake age volume faulted by the same model, for display).

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::{fault_model, fault_tile, generate_chunked, generate_labels};

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
            .unwrap_or_else(|| "faults_demo".into()),
    );
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 96);
    let nj: usize = arg(&args, 4, 96);
    let nk: usize = arg(&args, 5, 128);
    std::fs::create_dir_all(&out).expect("create out dir");

    let base = E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        store_path: None,
        chunk_shape: Some([16, 16, nk]),
        faults: FaultConfig::default(),
    };
    let cfg = E2eConfig {
        faults: FaultConfig::with_count(count),
        ..base.clone()
    };

    let t0 = std::time::Instant::now();
    let model = fault_model(&cfg).expect("faults enabled");
    let t_resolve = t0.elapsed();
    let (unfaulted, _) = generate_labels(&base);
    let t1 = std::time::Instant::now();
    let (vols, stats) = generate_chunked(&cfg);
    let t_gen = t1.elapsed();

    let mut age: Vec<f32> = Vec::with_capacity(ni * nj * nk);
    for i in 0..ni {
        for j in 0..nj {
            let spacing = 6.0 + 0.02 * i as f64 - 0.015 * j as f64;
            for k in 0..nk {
                age.push(((k as f64 - 4.0) / spacing) as f32);
            }
        }
    }
    let mut ids = vec![0u8; ni * nj * nk];
    let mut scratch = unfaulted.clone();
    let (mask, fault_ids) = model.apply_to_labels(&mut scratch, fault_tile(&cfg));
    ids.copy_from_slice(&fault_ids);
    let _ = model.apply_to_volume_f32(&mut age, fault_tile(&cfg));
    assert_eq!(
        scratch, vols.labels,
        "pipeline labels == model-applied labels"
    );

    write(out.join("labels.u8"), &vols.labels);
    write(out.join("labels_unfaulted.u8"), &unfaulted);
    write(out.join("fault_mask.u8"), &mask);
    write(out.join("fault_ids.u8"), &ids);
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    write(out.join("angle_stack.f32"), &f32s(&vols.angle_stack));
    write(out.join("age.f32"), &f32s(&age));

    let voxels: usize = mask.iter().map(|&v| v as usize).sum();
    let faults: Vec<String> = model
        .faults()
        .iter()
        .map(|f| {
            format!(
                "{{\"center\":{:?},\"throw\":{:.3},\"seabed_roll\":{}}}",
                f.center, f.params.throw, f.seabed_roll
            )
        })
        .collect();
    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"requested\":{count},\"inserted\":{},\"skipped\":{},\"fault_voxels\":{voxels},\"resolve_ms\":{},\"generate_ms\":{},\"peak_temp_bytes\":{},\"faults\":[{}]}}",
        model.faults().len(),
        model.skipped().len(),
        t_resolve.as_millis(),
        t_gen.as_millis(),
        stats.peak_temp_bytes,
        faults.join(",")
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
}
