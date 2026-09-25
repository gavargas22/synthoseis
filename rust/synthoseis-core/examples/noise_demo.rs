//! Seismic-noise demo: dumps the deterministic noise fields and a noisy,
//! filtered cube for the legacy comparison and plots
//! (`examples/plot_noise_demo.py`, `tests/fixtures/generate_seismic_noise.py`).
//!
//! ```text
//! cargo run --release -p synthoseis-core --example noise_demo -- \
//!     OUT_DIR [seed=7] [ni=32] [nj=32] [nk=96] [snr_db=12.5] [seeds=16] \
//!     [low=4] [high=30] [lateral=3]
//! ```
//!
//! Writes little-endian raw arrays in C order (shapes in `meta.json`):
//! - `rfc_{5,15,25}.f32`: raw reflectivity `(ni, nj, nk)` (legacy `rfc_raw`
//!   input, last sample is the Rust pad sample),
//! - `seabed.f64`: seabed map `(ni, nj)` in samples,
//! - `noise_{radians,legacy}_{5,15,25}.f32`: noise fields `(seeds, ni, nj,
//!   nk)` for noise seeds `1..=seeds` with the default (radian) and the
//!   legacy (degree) angle weights,
//! - `chain_noise_bp.f32` / `chain_noise_bp_alt.f32`: noise seed 1 + bandpass
//!   + lateral filter at 15 deg (Ricker skipped), 16x16 vs 5x7 tiles,
//! - `chain_noise_keep.f32`: same with `keep_ricker` (noise, Ricker, bandpass).

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, NoiseConfig};
use synthoseis_core::{
    fault_model, generate_chunked_at_angle, generate_labels, generate_noise, generate_reflectivity,
    noise_signal_std,
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
    let out = PathBuf::from(args.first().cloned().unwrap_or_else(|| "noise_demo".into()));
    let seed: u64 = arg(&args, 1, 7);
    let ni: usize = arg(&args, 2, 32);
    let nj: usize = arg(&args, 3, 32);
    let nk: usize = arg(&args, 4, 96);
    let snr_db: f64 = arg(&args, 5, 12.5);
    let seeds: u64 = arg(&args, 6, 16);
    let low: f64 = arg(&args, 7, 4.0);
    let high: f64 = arg(&args, 8, 30.0);
    let lateral: usize = arg(&args, 9, 3);
    std::fs::create_dir_all(&out).expect("create out dir");
    let angles = [5.0, 15.0, 25.0];

    let base = E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        store_path: None,
        chunk_shape: Some([16, 16, nk]),
        faults: FaultConfig::default(),
        filters: FilterConfig::default(),
    };
    assert!(fault_model(&base).is_none(), "demo runs without faults");
    for a in angles {
        write(
            out.join(format!("rfc_{a}.f32")),
            &f32_bytes(&generate_reflectivity(&base, a)),
        );
    }
    let seabed: Vec<u8> = synthoseis_core::pipeline_stream::fault_seabed(&base)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    write(out.join("seabed.f64"), &seabed);
    let (labels, shape) = generate_labels(&base);
    let data_std = noise_signal_std(&base, &labels, shape);

    let t0 = std::time::Instant::now();
    for (mode, legacy) in [("radians", false), ("legacy", true)] {
        for a in angles {
            let mut all = Vec::with_capacity(seeds as usize * ni * nj * nk * 4);
            for s in 1..=seeds {
                let c = E2eConfig {
                    filters: FilterConfig {
                        noise: NoiseConfig {
                            snr_db: Some(snr_db),
                            seed: Some(s),
                            legacy_angle_weights: legacy,
                        },
                        ..FilterConfig::default()
                    },
                    ..base.clone()
                };
                all.extend(f32_bytes(&generate_noise(&c, a).expect("noise on")));
            }
            write(out.join(format!("noise_{mode}_{a}.f32")), &all);
        }
    }
    let t_noise = t0.elapsed();

    let noisy = FilterConfig {
        noise: NoiseConfig {
            snr_db: Some(snr_db),
            seed: Some(1),
            legacy_angle_weights: false,
        },
        ..FilterConfig::legacy(low, high, lateral)
    };
    let chain = |chunks: [usize; 3], fc: FilterConfig| {
        let c = E2eConfig {
            chunk_shape: Some(chunks),
            filters: fc,
            ..base.clone()
        };
        generate_chunked_at_angle(&c, 15.0).0.angle_stack
    };
    let t1 = std::time::Instant::now();
    let bp = chain([16, 16, nk], noisy.clone());
    let t_chain = t1.elapsed();
    let bp_alt = chain([5, 7, nk], noisy.clone());
    let keep = chain(
        [16, 16, nk],
        FilterConfig {
            keep_ricker: true,
            ..noisy.clone()
        },
    );
    let identical = bp.iter().zip(&bp_alt).all(|(a, b)| a.to_bits() == b.to_bits());
    write(out.join("chain_noise_bp.f32"), &f32_bytes(&bp));
    write(out.join("chain_noise_bp_alt.f32"), &f32_bytes(&bp_alt));
    write(out.join("chain_noise_keep.f32"), &f32_bytes(&keep));

    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"snr_db\":{snr_db},\"noise_seeds\":{seeds},\
         \"angles\":[5.0,15.0,25.0],\"norm_angle_deg\":15.0,\"digi_ms\":4.0,\
         \"data_std\":{data_std:e},\"low\":{low},\"high\":{high},\"order\":4,\
         \"lateral\":{lateral},\"chain_noise_seed\":1,\
         \"tiles_16x16_vs_5x7_bit_identical\":{identical},\"t_noise_fields_s\":{:.3},\
         \"t_chain_s\":{:.3}}}",
        t_noise.as_secs_f64(),
        t_chain.as_secs_f64()
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
    assert!(identical, "noisy filtered stack must not depend on the tile shape");
}
