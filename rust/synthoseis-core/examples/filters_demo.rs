//! Filtered-cube demo: runs the chunked e2e pipeline on a (faulted) cube with
//! and without the legacy post-convolution filters and dumps raw volumes for
//! the legacy parity check and plots (`examples/plot_filters_demo.py`).
//!
//! ```text
//! cargo run --release -p synthoseis-core --example filters_demo -- \
//!     OUT_DIR [seed=7] [faults=4] [ni=64] [nj=64] [nk=128] [low=4] [high=30] [lateral=3]
//! ```
//!
//! Writes little-endian raw arrays (C order `(ni, nj, nk)`, shape in
//! `meta.json`): `angle_raw.f32` (unfiltered 15° Ricker stack),
//! `angle_rfc.f32` (raw 15° reflectivity, no wavelet: the legacy `rfc_raw`
//! input to `postprocess_rfc_cubes`), `angle_filtered.f32` (bandpass +
//! lateral filter with the Ricker skipped, i.e. the legacy chain, 16x16
//! tiles), `angle_filtered_alt.f32` (same filters, 5x7 tiles),
//! `angle_filtered_keep.f32` (`keep_ricker`: Ricker, then bandpass + lateral,
//! the #25 behaviour) and `labels.u8`. Plots: `plot_filters_demo.py`,
//! `plot_ricker_skip.py`.

use std::io::Write;
use std::path::PathBuf;

use synthoseis_core::{generate_chunked, generate_reflectivity};
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig};

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
            .unwrap_or_else(|| "filters_demo".into()),
    );
    let seed: u64 = arg(&args, 1, 7);
    let count: usize = arg(&args, 2, 4);
    let ni: usize = arg(&args, 3, 64);
    let nj: usize = arg(&args, 4, 64);
    let nk: usize = arg(&args, 5, 128);
    let low: f64 = arg(&args, 6, 4.0);
    let high: f64 = arg(&args, 7, 30.0);
    let lateral: usize = arg(&args, 8, 3);
    std::fs::create_dir_all(&out).expect("create out dir");

    let raw_cfg = E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        store_path: None,
        chunk_shape: Some([16, 16, nk]),
        faults: FaultConfig::with_count(count),
        filters: FilterConfig::default(),
    };
    let cfg = E2eConfig {
        filters: FilterConfig::legacy(low, high, lateral),
        ..raw_cfg.clone()
    };
    let alt_cfg = E2eConfig {
        chunk_shape: Some([5, 7, nk]),
        ..cfg.clone()
    };

    let t0 = std::time::Instant::now();
    let (raw, _) = generate_chunked(&raw_cfg);
    let t_raw = t0.elapsed();
    let t1 = std::time::Instant::now();
    let (filt, stats) = generate_chunked(&cfg);
    let t_filt = t1.elapsed();
    let (alt, _) = generate_chunked(&alt_cfg);
    let keep_cfg = E2eConfig {
        filters: FilterConfig {
            keep_ricker: true,
            ..cfg.filters.clone()
        },
        ..cfg.clone()
    };
    let (keep, _) = generate_chunked(&keep_cfg);
    let rfc = generate_reflectivity(&raw_cfg, 15.0);
    let identical = filt
        .angle_stack
        .iter()
        .zip(&alt.angle_stack)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    write(out.join("angle_raw.f32"), &f32_bytes(&raw.angle_stack));
    write(
        out.join("angle_filtered.f32"),
        &f32_bytes(&filt.angle_stack),
    );
    write(
        out.join("angle_filtered_alt.f32"),
        &f32_bytes(&alt.angle_stack),
    );
    write(
        out.join("angle_filtered_keep.f32"),
        &f32_bytes(&keep.angle_stack),
    );
    write(out.join("angle_rfc.f32"), &f32_bytes(&rfc));
    write(out.join("labels.u8"), &raw.labels);
    let meta = format!(
        "{{\"shape\":[{ni},{nj},{nk}],\"seed\":{seed},\"faults\":{count},\"low\":{low},\"high\":{high},\
         \"order\":4,\"lateral\":{lateral},\"digi_ms\":4.0,\"angle_deg\":15.0,\
         \"ricker_skipped\":{},\"ricker_hz\":40.0,\
         \"tiles_16x16_vs_5x7_bit_identical\":{identical},\"t_raw_s\":{:.3},\"t_filtered_s\":{:.3},\
         \"peak_temp_bytes\":{}}}",
        cfg.filters.skips_ricker(),
        t_raw.as_secs_f64(),
        t_filt.as_secs_f64(),
        stats.peak_temp_bytes
    );
    write(out.join("meta.json"), meta.as_bytes());
    println!("{meta}");
    assert!(
        identical,
        "filtered stack must not depend on the tile shape"
    );
}
