//! Rock physics port: parity with the legacy property chain, the
//! `--legacy-toy-depth` switch (master 10f4dcd goldens), statistics of the
//! random parts, the faulted-cube approximation, and invariance of the
//! default model across chunk shapes, workers, processes and paths.
//!
//! Fixture: `tests/fixtures/rock_physics.json` from
//! `tests/fixtures/generate_rock_physics.py` (real legacy
//! `build_faulted_property_geomodels` + `build_property_models_randomised_depth`).

use serde::Deserialize;
use serde_json::Value;
use synthoseis_core::pipeline::{
    generate_tiny_cube, E2eConfig, FaultConfig, FilterConfig, NoiseConfig, RockPhysicsConfig,
};
use synthoseis_core::rock_physics::{
    elastic_model, keyed_unit, label_intervals, layer_depth_trace, layer_fluids, layer_shifts,
    legacy_shift, net_to_gross_map, shift_half_ranges, ElasticModel, Fluid, MixingMethod,
    NetToGross,
};
use synthoseis_core::{
    generate_chunked, generate_chunked_at_angle, generate_labels, generate_reflectivity,
    run_e2e_geometry_once_seismic_many, run_e2e_multiprocess, run_e2e_streaming,
    run_e2e_streaming_overlapped, run_e2e_strip_stitched,
};
use synthoseis_geo::faults::{horizon_depth_from_age, FaultModel, FaultParams, ReachMode, Seabed};
use synthoseis_geo::fill_layer_labels;
use synthoseis_io::MdioStore;
use synthoseis_rpm::{legacy_column_properties, LayerShifts, VoxelKind};
use tempfile::tempdir;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_path("rock_physics.json")).unwrap())
        .unwrap()
}

/// Integers of a fixture array, plain or run-length encoded as
/// `{"rle": [v0, n0, v1, n1, ...]}`.
fn ints(v: &Value) -> Vec<i64> {
    let int = |x: &Value| x.as_i64().unwrap();
    match v.get("rle") {
        Some(r) => {
            let r = r.as_array().unwrap();
            r.chunks(2)
                .flat_map(|p| std::iter::repeat_n(int(&p[0]), int(&p[1]) as usize))
                .collect()
        }
        None => v.as_array().unwrap().iter().map(int).collect(),
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    ints(v).into_iter().map(|x| x as u32).collect()
}

fn f32s_from_bits(v: &Value) -> Vec<f32> {
    u32s(v).into_iter().map(f32::from_bits).collect()
}

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn angle_hash(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn shape_of(fx: &Value) -> [usize; 3] {
    let s: Vec<usize> = fx["meta"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    [s[0], s[1], s[2]]
}

/// Depth model: the per-layer TVDML of every labelled voxel is bit-identical
/// to the legacy `faulted_depth` cube wherever both assign the same layer.
#[test]
fn depth_model_bit_exact_vs_legacy() {
    let fx = fixture();
    let [ni, nj, nk] = shape_of(&fx);
    let nh = 4;
    let maps: Vec<f64> = f32s_from_bits(&fx["maps_bits"]).into_iter().map(f64::from).collect();
    let legacy = f32s_from_bits(&fx["legacy_depth_bits"]);
    let layer = ints(&fx["legacy_layer"]);
    let labels = fill_layer_labels(&maps, [ni, nj, nh], nk);
    let intervals = label_intervals(&maps, ni, nj, nh, nk);
    assert_eq!(intervals, vec![0, 1, 2]);
    let (mut compared, mut same_layer, mut sediment) = (0usize, 0usize, 0usize);
    let mut depth = vec![0.0f32; nk];
    for c in 0..ni * nj {
        let col = &labels[c * nk..(c + 1) * nk];
        let seabed = layer_depth_trace(col, &maps[c * nh..(c + 1) * nh], &intervals, 4.0, &mut depth);
        for k in 0..nk {
            let g = c * nk + k;
            if k < seabed {
                assert_eq!(depth[k], 0.0);
            }
            if layer[g] >= 1 {
                sediment += 1;
            }
            if col[k] != 255 && layer[g] == col[k] as i64 + 1 {
                same_layer += 1;
                compared += 1;
                assert_eq!(
                    depth[k].to_bits(),
                    legacy[g].to_bits(),
                    "column {c} sample {k}: rust {} legacy {}",
                    depth[k],
                    legacy[g]
                );
            }
        }
    }
    let agreement = same_layer as f64 / sediment as f64;
    eprintln!(
        "depth parity: {compared} voxels bit-identical (layer agreement {:.4} of legacy sediment voxels)",
        agreement
    );
    assert!(compared > ni * nj * nk / 2);
    // Rust labels [ceil z_h, floor z_{h+1}) (with a one-sample 255 gap at a
    // non-integer horizon) vs legacy rounding: only boundary samples differ.
    assert!(agreement > 0.85, "layer agreement {agreement}");
}

fn legacy_kinds(fx: &Value, fluids: bool) -> (Vec<f32>, Vec<VoxelKind>) {
    let depth = f32s_from_bits(&fx["legacy_depth_bits"]);
    let layer = ints(&fx["legacy_layer"]);
    let age = ints(&fx["age"]);
    let ng = f32s_from_bits(&fx["ng_bits"]);
    let oil = u32s(&fx["oil"]);
    let gas = u32s(&fx["gas"]);
    let max_age = *age.iter().max().unwrap();
    let kinds = (0..depth.len())
        .map(|g| {
            if layer[g] < 0 {
                VoxelKind::Water
            } else if age[g] == max_age {
                VoxelKind::Unfilled
            } else {
                let fluid = match (fluids, oil[g], gas[g]) {
                    (true, 1, 0) => Fluid::Oil,
                    (true, 0, 1) => Fluid::Gas,
                    _ => Fluid::Brine,
                };
                VoxelKind::Layer {
                    layer: (age[g] - 1) as usize,
                    ng: ng[g],
                    fluid,
                }
            }
        })
        .collect();
    (depth, kinds)
}

fn legacy_shifts(v: &Value) -> Vec<LayerShifts> {
    (1..=3)
        .map(|z| match v.get(z.to_string()) {
            Some(s) => {
                let p = s["props"].as_array().unwrap();
                let mut props = [[0i64; 3]; 4];
                for (r, row) in props.iter_mut().enumerate() {
                    for (c, x) in row.iter_mut().enumerate() {
                        *x = p[r][c].as_i64().unwrap();
                    }
                }
                LayerShifts {
                    layer: s["layer"].as_i64().unwrap(),
                    props,
                }
            }
            None => LayerShifts::default(),
        })
        .collect()
}

/// Golden test of the property builder (trends, shifts, water, sand/shale
/// mixing for brine / oil / gas, base forward-fill) against legacy, for
/// inverse-velocity and Backus mixing: every voxel bit-identical.
#[test]
fn properties_bit_exact_vs_legacy_mixing() {
    let fx = fixture();
    let [ni, nj, nk] = shape_of(&fx);
    let (depth, kinds) = legacy_kinds(&fx, true);
    for (key, method) in [
        ("inv_vel", MixingMethod::InverseVelocity),
        ("backus", MixingMethod::BackusModuli),
    ] {
        let case = &fx["properties"][key];
        let shifts = legacy_shifts(&case["shifts"]);
        assert!(shifts.iter().any(|s| s.layer != 0 || s.props != [[0; 3]; 4]));
        let want_rho = u32s(&case["props"]["rho"]);
        let want_vp = u32s(&case["props"]["vp"]);
        let want_vs = u32s(&case["props"]["vs"]);
        let (mut rho, mut vp, mut vs) = (vec![0.0f32; nk], vec![0.0f32; nk], vec![0.0f32; nk]);
        let mut n = 0usize;
        for c in 0..ni * nj {
            let r = c * nk..(c + 1) * nk;
            legacy_column_properties(&depth[r.clone()], &kinds[r.clone()], &shifts, method, &mut rho, &mut vp, &mut vs);
            for k in 0..nk {
                let g = c * nk + k;
                assert_eq!(rho[k].to_bits(), want_rho[g], "{key} rho col {c} k {k}");
                assert_eq!(vp[k].to_bits(), want_vp[g], "{key} vp col {c} k {k}");
                assert_eq!(vs[k].to_bits(), want_vs[g], "{key} vs col {c} k {k}");
                n += 1;
            }
        }
        eprintln!("{key}: {n} voxels x 3 properties bit-identical to legacy");
    }
    // Oil and gas voxels are present and differ from brine.
    let (_, brine_kinds) = legacy_kinds(&fx, false);
    let shifts = legacy_shifts(&fx["properties"]["inv_vel"]["shifts"]);
    let (mut a, mut b) = (vec![0.0f32; nk], vec![0.0f32; nk]);
    let (mut s1, mut s2, mut s3) = (vec![0.0f32; nk], vec![0.0f32; nk], vec![0.0f32; nk]);
    let mut differs = 0;
    for c in 0..ni * nj {
        let r = c * nk..(c + 1) * nk;
        legacy_column_properties(&depth[r.clone()], &kinds[r.clone()], &shifts, MixingMethod::InverseVelocity, &mut a, &mut s1, &mut s2);
        legacy_column_properties(&depth[r.clone()], &brine_kinds[r], &shifts, MixingMethod::InverseVelocity, &mut b, &mut s3, &mut s2);
        differs += a.iter().zip(&b).filter(|(x, y)| x != y).count();
    }
    assert!(differs > 100, "fluid substitution changed {differs} voxels");
}

/// Statistical checks where legacy is random: shift draws, shift half
/// ranges, fluid draws and net-to-gross maps against the legacy draws.
#[test]
fn random_parts_match_legacy_statistics() {
    let fx = fixture();
    let st = &fx["statistics"];
    // int(uniform(-5, 5)): same pmf (0 is twice as likely: truncation).
    let n = 40_000u64;
    let mut counts = [0usize; 11];
    for t in 0..n {
        counts[(legacy_shift(5, &[1, 2, t]) + 5) as usize] += 1;
    }
    for v in -5i64..=5 {
        let legacy = st["shift5_pmf"][v.to_string()].as_f64().unwrap();
        let rust = counts[(v + 5) as usize] as f64 / n as f64;
        assert!((rust - legacy).abs() < 0.01, "shift {v}: rust {rust} legacy {legacy}");
    }
    assert_eq!(counts[0] + counts[10], 0);
    // Half ranges: int(triangular(35, 75, 125)) and int(triangular(5, 11, 20)).
    let (mut lay, mut prop) = (Vec::new(), Vec::new());
    for seed in 0..20_000u64 {
        let (l, p) = shift_half_ranges(seed, &RockPhysicsConfig::default());
        lay.push(l as f64);
        prop.push(p as f64);
    }
    for (name, v) in [("layer_half_range", &lay), ("property_half_range", &prop)] {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let want = st[name]["mean"].as_f64().unwrap();
        let sd = st[name]["sd"].as_f64().unwrap();
        assert!((m - want).abs() < 4.0 * sd / (v.len() as f64).sqrt() * 2f64.sqrt() + 0.05, "{name} mean {m} want {want}");
    }
    // Fluids: uniform over brine / oil / gas.
    let mut fl = [0usize; 3];
    for t in 0..30_000u64 {
        fl[((keyed_unit(&[9, 4, 1, t]) * 3.0) as usize).min(2)] += 1;
    }
    for (c, &cnt) in fl.iter().enumerate() {
        let legacy = st["fluid_freq"][c].as_f64().unwrap();
        assert!((cnt as f64 / 30_000.0 - legacy).abs() < 0.015, "fluid {c}");
    }
    // Net-to-gross maps (32 x 32, 200 seeds).
    let ngs = &st["ng_maps"];
    let (mut means, mut sds, mut lag1) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for seed in 0..200u64 {
        let m = net_to_gross_map(seed, &NetToGross::default(), 32, 32, 1);
        let v: Vec<f64> = m.iter().map(|&x| x as f64).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
        let mut ac = 0.0;
        for i in 1..32 {
            for j in 0..32 {
                ac += (v[i * 32 + j] - mean) * (v[(i - 1) * 32 + j] - mean);
            }
        }
        lag1.push(ac / (31.0 * 32.0) / var.max(1e-30));
        means.push(mean);
        sds.push(var.sqrt());
        lo = lo.min(m.iter().cloned().fold(f32::MAX, f32::min));
        hi = hi.max(m.iter().cloned().fold(f32::MIN, f32::max));
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mm, ms, ml) = (avg(&means), avg(&sds), avg(&lag1));
    eprintln!(
        "N/G maps rust: mean {mm:.4} sd {ms:.4} lag1 {ml:.3} range [{lo}, {hi}]; legacy: mean {:.4} sd {:.4} lag1 {:.3} range [{}, {}]",
        ngs["mean_of_means"].as_f64().unwrap(),
        ngs["mean_of_sds"].as_f64().unwrap(),
        ngs["lag1_autocorr_mean"].as_f64().unwrap(),
        ngs["min"].as_f64().unwrap(),
        ngs["max"].as_f64().unwrap()
    );
    let se = ngs["sd_of_means"].as_f64().unwrap() * (2.0f64 / 200.0).sqrt();
    assert!((mm - ngs["mean_of_means"].as_f64().unwrap()).abs() < 5.0 * se, "N/G mean");
    let se_sd = ngs["sd_of_sds"].as_f64().unwrap() * (2.0f64 / 200.0).sqrt();
    assert!((ms - ngs["mean_of_sds"].as_f64().unwrap()).abs() < 5.0 * se_sd + 0.002, "N/G sd");
    assert!(lo >= 0.45 && hi <= 0.9);
    assert!(ml > 0.5, "N/G maps must be laterally smooth (lag-1 autocorr {ml})");
}

/// Layer shifts are keyed by (seed, layer): only layers deeper than
/// `first_random_layer` move, draws do not depend on anything else.
#[test]
fn layer_shifts_are_keyed_and_thresholded() {
    let rp = RockPhysicsConfig::default();
    for l in 0..20 {
        assert_eq!(layer_shifts(3, &rp, l), LayerShifts::default(), "layer {l}");
    }
    let deep: Vec<_> = (20..60).map(|l| layer_shifts(3, &rp, l)).collect();
    assert!(deep.iter().any(|s| s.layer != 0));
    assert_eq!(deep, (20..60).map(|l| layer_shifts(3, &rp, l)).collect::<Vec<_>>());
    assert_ne!(deep, (20..60).map(|l| layer_shifts(4, &rp, l)).collect::<Vec<_>>());
    let (lh, ph) = shift_half_ranges(3, &rp);
    for s in &deep {
        assert!(s.layer.unsigned_abs() < lh as u64);
        assert!(s.props.iter().flatten().all(|v| v.unsigned_abs() < ph as u64));
    }
}

/// Spill-point closure on a dome: the fluid column is capped by the spill
/// point and `max_column`, the fluid is keyed by (seed, layer, closure).
#[test]
fn dome_closure_selects_fluid_above_contact() {
    let [ni, nj, nk] = [21usize, 19, 60];
    let mut labels = vec![255u8; ni * nj * nk];
    let mut top = vec![0usize; ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            let r2 = (i as f64 - 10.0).powi(2) + (j as f64 - 9.0).powi(2);
            let t = (20.0 + 0.12 * r2).min(34.0) as usize;
            top[i * nj + j] = t;
            let g = (i * nj + j) * nk;
            labels[g + 2..g + t].fill(0);
            labels[g + t..g + 50].fill(1);
            labels[g + 50..g + nk].fill(2);
        }
    }
    let f = layer_fluids(&labels, [ni, nj, nk], 1, 1, 11, 1000.0, 1);
    assert_eq!(f.closures.len(), 1, "{:?}", f.closures);
    let (fluid, crest, contact, cols, _) = f.closures[0];
    assert_eq!(crest, 20.0);
    // Spill point: the shallowest boundary column (j = 0 / nj-1 at i = 10,
    // top = 20 + 0.12 * 81 -> sample 29).
    assert_eq!(contact, 29.0);
    assert!(cols > 50);
    let c = 10 * nj + 9;
    assert_eq!(f.contact[c], 29.0);
    assert_eq!(f.fluid[c], fluid);
    // Column cap: 5 samples below the crest.
    let g = layer_fluids(&labels, [ni, nj, nk], 1, 1, 11, 5.0, 1);
    assert_eq!(g.closures[0].2, 25.0);
    // Keyed: same seed, same fluid; the fluid set over seeds covers all three.
    let fluids: std::collections::HashSet<_> = (0..30)
        .map(|s| layer_fluids(&labels, [ni, nj, nk], 1, 1, s, 1000.0, 1).closures[0].0)
        .collect();
    assert_eq!(fluids.len(), 3);
    // min voxels filter.
    assert!(layer_fluids(&labels, [ni, nj, nk], 1, 1, 11, 1000.0, 1_000_000).closures.is_empty());
}

fn cfg(seed: u64, shape: [usize; 3], chunks: [usize; 3], faults: usize) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Planar,
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some(chunks),
        faults: FaultConfig::with_count(faults),
        filters: FilterConfig::default(),
        rock_physics: RockPhysicsConfig::default(),
    }
}

fn legacy(c: E2eConfig) -> E2eConfig {
    E2eConfig {
        rock_physics: RockPhysicsConfig::legacy_toy(),
        ..c
    }
}

/// `--legacy-toy-depth` reproduces master 10f4dcd bit for bit (hashes
/// recorded on 10f4dcd with `examples/master_hashes`-style runs).
#[test]
fn legacy_toy_depth_switch_reproduces_master_10f4dcd() {
    let a = legacy(cfg(7, [64, 64, 128], [16, 16, 128], 4));
    let (v, _) = generate_chunked(&a);
    assert_eq!(fnv(v.labels.iter().copied()), 0x22fa_1389_4f70_0192);
    assert_eq!(angle_hash(&v.angle_stack), 0x60cf_db87_1073_766c);
    assert_eq!(angle_hash(&generate_reflectivity(&a, 15.0)), 0xee47_4d76_0e21_eb09);
    let (d, _) = generate_chunked_at_angle(&a, 30.0);
    assert_eq!(angle_hash(&d.angle_stack), 0xefda_0400_7f40_7851);
    let b = legacy(E2eConfig {
        filters: FilterConfig {
            noise: NoiseConfig {
                snr_db: Some(12.5),
                seed: Some(3),
                legacy_angle_weights: false,
                legacy_seabed: false,
            },
            ..FilterConfig::legacy(4.0, 30.0, 3)
        },
        ..cfg(10, [24, 20, 64], [8, 5, 64], 3)
    });
    let (w, _) = generate_chunked(&b);
    assert_eq!(angle_hash(&w.angle_stack), 0xbec6_ff4c_0daf_5786);
    assert_eq!(
        angle_hash(&generate_tiny_cube(&legacy(E2eConfig::tiny(42))).angle_stack),
        0x3e27_e417_f473_649f
    );
    // The default model differs.
    let (x, _) = generate_chunked(&cfg(7, [64, 64, 128], [16, 16, 128], 4));
    assert_eq!(x.labels, v.labels);
    assert_ne!(angle_hash(&x.angle_stack), 0x60cf_db87_1073_766c);
}

/// Default model sanity on the root-cause cube: no post-critical spikes, no
/// per-sample DC ramp; water above the seabed; per-layer constant properties.
#[test]
fn default_model_reflectivity_is_physical() {
    let c = cfg(7, [64, 64, 128], [16, 16, 128], 4);
    let r = generate_reflectivity(&c, 15.0);
    let (lo, hi) = r.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let nz = r.iter().filter(|&&x| x != 0.0).count() as f64 / r.len() as f64;
    let mean = r.iter().map(|&x| x as f64).sum::<f64>() / r.len() as f64;
    eprintln!("default rfc15: min {lo:e} max {hi:e} mean {mean:e} nonzero {:.3}%", 100.0 * nz);
    assert!(lo > -1.0 && hi < 1.0);
    assert!(nz < 0.1, "sparse reflectivity expected, got {nz}");
    let t = legacy(c.clone());
    let rt = generate_reflectivity(&t, 15.0);
    assert!(rt.iter().any(|x| x.abs() > 1.0), "master toy has post-critical spikes");

    let (labels, shape) = generate_labels(&c);
    let model = elastic_model(&c, &labels, shape);
    let ElasticModel::Rpm(m) = &model else { panic!("default must be the rpm model") };
    let [ni, nj, nk] = shape;
    let n = ni * nj * nk;
    let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    for g in (0..n).step_by(7) {
        if labels[g] == 255 && labels[g - g % nk..g].iter().all(|&l| l == 255) {
            assert_eq!((vp[g], vs[g], rho[g]), (1500.0, 1000.0, 1.028f32));
        }
    }
    assert_eq!(m.step, 4.0);
}

/// Layered, rich filters, 4 faults, sand fraction 0.4 / thickness 3: seed 6
/// has a multi-layer sand unit that closes on the dome.
fn unit_case(closures_per_layer: bool) -> E2eConfig {
    let r = rich([8, 5, 128]);
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Layered,
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: Some(0.4),
            sand_layer_thickness: 3.0,
            closures_per_layer,
            // Pinned master b4f4259 scenario (tests/closure_units.rs).
            salt: false,
            ..RockPhysicsConfig::default()
        },
        filters: r.filters.clone(),
        ..cfg(6, [24, 20, 128], [8, 5, 128], 4)
    }
}

/// Sandy thin-unit layered cube with 4 faults (seed 7): segmentation joins
/// closures across faults; `unsegmented` is the ef2dc42 model.
fn segmented_case(unsegmented: bool) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Layered,
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: Some(0.5),
            sand_layer_thickness: 1.0,
            closures_unsegmented: unsegmented,
            // Pinned master b4f4259 scenario (tests/closure_segments.rs).
            salt: false,
            ..RockPhysicsConfig::default()
        },
        ..cfg(7, [24, 20, 128], [8, 5, 128], 4)
    }
}

/// Salt body (default on) with 4 faults and a closure beside the salt
/// (seed 30); salt labels are checked by every MDIO path.
fn salt_case(salt: bool) -> E2eConfig {
    E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Layered,
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: Some(0.4),
            salt,
            ..RockPhysicsConfig::default()
        },
        ..cfg(30, [24, 20, 128], [8, 5, 128], 4)
    }
}

fn rich(chunks: [usize; 3]) -> E2eConfig {
    E2eConfig {
        filters: FilterConfig {
            noise: NoiseConfig::snr(15.0),
            ..FilterConfig::legacy(4.0, 30.0, 3)
        },
        rock_physics: RockPhysicsConfig {
            mixing: MixingMethod::BackusModuli,
            first_random_layer: 0,
            layer_shift_samples: Some(6),
            property_shift_samples: Some(3),
            min_closure_voxels: 1,
            ..RockPhysicsConfig::default()
        },
        ..cfg(10, [24, 20, 64], chunks, 3)
    }
}

/// Default model (plus Backus, random shifts, closures, filters, noise and
/// faults) is bit-identical across chunk shapes, the classic path,
/// streaming, overlap, strip-stitch 2/3/4, multi-process 1/2/3 and
/// geometry-once; the legacy switches (`--legacy-toy-depth`,
/// `--legacy-zoeppritz`) too, and the layered default toy geometry (with
/// default shifts active in the 160-sample case) with its Markov lithology
/// (default and with explicit sand fraction / unit thickness).
#[test]
fn default_model_invariant_to_tiling_workers_and_paths() {
    let dir = tempdir().unwrap();
    let read = |p: &std::path::Path| bits(&MdioStore::open(p).unwrap().read_volume().unwrap());
    let layered = |c: E2eConfig| E2eConfig {
        geometry: synthoseis_core::ToyGeometry::Layered,
        ..c
    };
    let legacy_zoeppritz = |c: E2eConfig| E2eConfig {
        rock_physics: RockPhysicsConfig { legacy_zoeppritz: true, ..c.rock_physics.clone() },
        ..c
    };
    for (n, base) in [
        rich([8, 5, 64]),
        cfg(4, [24, 20, 48], [8, 5, 48], 2),
        legacy(rich([8, 5, 64])),
        legacy_zoeppritz(rich([8, 5, 64])),
        layered(rich([8, 5, 64])),
        // Plain default model on the layered default geometry, deep enough
        // (> 20 layers) for the default random depth shifts.
        layered(cfg(4, [24, 20, 160], [8, 5, 160], 2)),
        // Legacy sand-fraction lithology with explicit options (the two
        // layered cases above use the default Markov chain).
        E2eConfig {
            rock_physics: RockPhysicsConfig {
                sand_layer_fraction: Some(0.4),
                sand_layer_thickness: 3.0,
                ..rich([8, 5, 64]).rock_physics
            },
            ..layered(rich([8, 5, 64]))
        },
        // Closures per sand unit with a multi-layer unit closing on the dome
        // (tests/closure_units.rs), then the per-layer switch.
        unit_case(false),
        unit_case(true),
        // 3D closure segmentation where faults join closures of different
        // sand units (tests/closure_segments.rs).
        segmented_case(false),
        // Salt body (tests/salt.rs).
        salt_case(true),
    ]
    .into_iter()
        .enumerate()
    {
        let with = |chunks: [usize; 3], store: Option<std::path::PathBuf>| E2eConfig {
            chunk_shape: Some(chunks),
            store_path: store,
            ..base.clone()
        };
        if n >= 4 {
            assert_eq!(base.effective_lithology(), synthoseis_core::ToyLithology::Markov);
        }
        if n == 6 {
            let (labels, shape) = generate_labels(&base);
            let ElasticModel::Rpm(m) = elastic_model(&base, &labels, shape) else { panic!() };
            let n_sand = m.layers.iter().filter(|l| l.sand).count();
            assert!(n_sand > 0 && n_sand < m.layers.len(), "mixed lithology {n_sand}/{}", m.layers.len());
        }
        if n == 7 {
            let (labels, shape) = generate_labels(&base);
            let ElasticModel::Rpm(m) = elastic_model(&base, &labels, shape) else { panic!() };
            let sand = synthoseis_core::lithology::interval_sand(
                synthoseis_core::ToyLithology::Markov, base.seed, m.nh, Some(0.4), 3.0,
            );
            let multi = synthoseis_core::lithology::closure_units(&sand).into_iter().any(|(a, b)| {
                b - a > 1
                    && m.layers.iter().any(|l| {
                        l.interval == a && l.fluids.as_ref().is_some_and(|f| !f.closures.is_empty())
                    })
            });
            assert!(multi, "multi-layer sand unit with closures");
            assert_ne!(
                bits(&generate_chunked(&base).0.angle_stack),
                bits(&generate_chunked(&unit_case(true)).0.angle_stack),
                "per unit vs per layer"
            );
        }
        if n == 9 {
            assert_ne!(
                bits(&generate_chunked(&base).0.angle_stack),
                bits(&generate_chunked(&segmented_case(true)).0.angle_stack),
                "segmented vs unsegmented"
            );
        }
        if n == 10 {
            let body = synthoseis_core::salt::salt_body(&base).expect("salt");
            assert!(body.voxels(base.samples) > 500);
            let (labels, shape) = generate_labels(&base);
            let ElasticModel::Rpm(m) = elastic_model(&base, &labels, shape) else { panic!() };
            assert!(m.salt.is_some());
            assert!(m.layers.iter().any(|l| l.fluids.as_ref().is_some_and(|f| !f.closures.is_empty())));
            assert_ne!(
                bits(&generate_chunked(&base).0.angle_stack),
                bits(&generate_chunked(&salt_case(false)).0.angle_stack),
                "salt vs no salt"
            );
        }
        if n == 5 {
            let (labels, shape) = generate_labels(&base);
            let ElasticModel::Rpm(m) = elastic_model(&base, &labels, shape) else {
                panic!()
            };
            assert!(
                m.layers
                    .iter()
                    .any(|l| l.interval >= 20 && l.shifts.layer != 0),
                "default shifts active"
            );
        }
        let (reference, _) = generate_chunked(&base);
        let want = bits(&reference.angle_stack);
        for chunks in [[1, 1, base.samples], [5, 7, base.samples], [24, 20, base.samples], [3, 20, 16]] {
            let (v, _) = generate_chunked(&with(chunks, None));
            assert_eq!(bits(&v.angle_stack), want, "case {n} chunks {chunks:?}");
        }
        assert_eq!(bits(&generate_tiny_cube(&base).angle_stack), want, "case {n} classic");
        for (k, chunks) in [[8, 5, 16], [5, 7, base.samples]].into_iter().enumerate() {
            let p = dir.path().join(format!("s{n}_{k}.mdio"));
            run_e2e_streaming(&with(chunks, Some(p.clone()))).unwrap();
            assert_eq!(read(&p), want, "case {n} streaming {chunks:?}");
            let p = dir.path().join(format!("o{n}_{k}.mdio"));
            run_e2e_streaming_overlapped(&with(chunks, Some(p.clone()))).unwrap();
            assert_eq!(read(&p), want, "case {n} overlap {chunks:?}");
        }
        for workers in [2, 3, 4] {
            let p = dir.path().join(format!("t{n}_{workers}.mdio"));
            run_e2e_strip_stitched(&with([5, 7, base.samples], Some(p.clone())), workers).unwrap();
            assert_eq!(read(&p), want, "case {n} strip {workers}");
        }
        for workers in [1, 2, 3] {
            let p = dir.path().join(format!("m{n}_{workers}.mdio"));
            let (r, _) = run_e2e_multiprocess(&with([8, 7, base.samples], Some(p.clone())), workers).unwrap();
            assert_eq!(bits(&r.volumes.angle_stack), want);
            assert_eq!(read(&p), want, "case {n} multiprocess {workers}");
        }
        let (geo, _) = run_e2e_geometry_once_seismic_many(&base, &[0.0, 15.0, 30.0]).unwrap();
        assert_eq!(geo.labels_generated, 1);
        let s15 = geo.stacks.iter().find(|s| s.angle_deg == 15.0).unwrap();
        assert_eq!(bits(&s15.volumes.angle_stack), want, "case {n} geometry-once");
        for s in &geo.stacks {
            let (v, _) = generate_chunked_at_angle(&base, s.angle_deg);
            assert_eq!(bits(&s.volumes.angle_stack), bits(&v.angle_stack), "case {n} angle {}", s.angle_deg);
        }
    }
}

/// The model is a pure function of (cfg, labels): rebuilding it gives the
/// same model, and mixing / N/G / shifts change the output.
#[test]
fn model_options_take_effect() {
    let base = rich([8, 5, 64]);
    let (labels, shape) = generate_labels(&base);
    assert_eq!(elastic_model(&base, &labels, shape), elastic_model(&base, &labels, shape));
    let h = |c: &E2eConfig| angle_hash(&generate_chunked(c).0.angle_stack);
    let reference = h(&base);
    let variants = [
        RockPhysicsConfig { mixing: MixingMethod::InverseVelocity, ..base.rock_physics.clone() },
        RockPhysicsConfig { net_to_gross: NetToGross::Constant(1.0), ..base.rock_physics.clone() },
        RockPhysicsConfig { first_random_layer: 20, ..base.rock_physics.clone() },
        RockPhysicsConfig::default(),
    ];
    for rp in variants {
        let c = E2eConfig { rock_physics: rp.clone(), ..base.clone() };
        assert_ne!(h(&c), reference, "{rp:?}");
    }
}

// ---------------------------------------------------------------------------
// Faulted-cube approximation vs legacy.

#[derive(Deserialize)]
struct FaultFixture {
    cases: Vec<FaultCase>,
}

#[derive(Deserialize)]
struct AgeCfg {
    z_top: f64,
    spacing_base: f64,
    spacing_di: f64,
    spacing_dj: f64,
}

#[derive(Deserialize)]
struct FaultJson {
    a: f64,
    b: f64,
    c: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    throw: f64,
    tilt_pct: f64,
    center: Option<[usize; 3]>,
    sigma: Option<f64>,
    p: Option<f64>,
    coef: Option<f64>,
}

#[derive(Deserialize)]
struct FaultExpected {
    horizon_depths: Vec<f64>,
}

#[derive(Deserialize)]
struct FaultCase {
    name: String,
    shape: [usize; 3],
    infill_factor: f64,
    wb_const: f64,
    age: AgeCfg,
    horizons: Vec<i64>,
    horizon_stride: usize,
    faults: Vec<FaultJson>,
    expected: FaultExpected,
}

/// Legacy non-partial-voxel layer fill of one column from (faulted) horizon
/// depths `fh` (seabed duplicated as legacy layer 0): legacy layer per sample
/// (`-1` water / unfilled).
fn legacy_fill(fh: &[f32], nk: usize) -> Vec<i64> {
    let mut h = vec![fh[0]];
    h.extend_from_slice(fh);
    let mut out = vec![-1i64; nk];
    for i in (1..h.len() - 1).rev() {
        let top = h[i] as i64;
        let base = h[i + 1] as i64 + 1;
        let thick = base - top;
        let b = (0.5f32 + h[i + 1].clamp(0.0, (nk - 1) as f32)) as i64;
        for k in 0..thick.max(0) {
            let s = b - k;
            if (0..nk as i64).contains(&s) {
                out[s as usize] = i as i64;
            }
        }
    }
    out
}

/// Quantifies the label-run approximation (depth from post-fault label runs
/// on unfaulted maps) against legacy (depth from horizon maps re-picked on
/// the faulted age volume) on the legacy fault fixture; also reports the
/// naive "fault the depth cube" variant (no run shift).
#[test]
fn faulted_cube_approximation_vs_legacy() {
    let text = std::fs::read_to_string(fixture_path("fault_cubes.json")).unwrap();
    let fx: FaultFixture = serde_json::from_str(&text).unwrap();
    let step = 4.0f32;
    let (mut n_all, mut n_cmp, mut exact, mut within, mut sum, mut sum_naive, mut max_e, mut max_naive) =
        (0usize, 0usize, 0usize, 0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut errs = Vec::new();
    let mut max_horizon_vs_fixture = 0.0f64;
    for case in &fx.cases {
        let [ni, nj, nk] = case.shape;
        let params: Vec<FaultParams> = case
            .faults
            .iter()
            .map(|f| {
                let p = FaultParams::from_legacy(f.a, f.b, f.c, f.x0, f.y0, f.z0, f.throw, f.tilt_pct, case.infill_factor);
                let p = match (f.sigma, f.p, f.coef) {
                    (Some(s), Some(pp), Some(c)) => p.with_profile(s, pp, c),
                    _ => p,
                };
                p.with_center(f.center)
            })
            .collect();
        let model = FaultModel::resolve_with_mode(case.shape, &params, &Seabed::Flat(case.wb_const), 0, ReachMode::Legacy);
        let nh = case.horizons.len();
        let mut maps = vec![0.0f64; ni * nj * nh];
        let mut age = vec![0.0f32; ni * nj * nk];
        for i in 0..ni {
            for j in 0..nj {
                let sp = case.age.spacing_base + case.age.spacing_di * i as f64 + case.age.spacing_dj * j as f64;
                for (n, &h) in case.horizons.iter().enumerate() {
                    maps[(i * nj + j) * nh + n] = case.age.z_top + h as f64 * sp;
                }
                for k in 0..nk {
                    age[(i * nj + j) * nk + k] = ((k as f64 - case.age.z_top) / sp) as f32;
                }
            }
        }
        let mut labels = fill_layer_labels(&maps, [ni, nj, nh], nk);
        let intervals = label_intervals(&maps, ni, nj, nh, nk);
        model.apply_to_labels(&mut labels, [8, 8]);
        model.apply_to_volume_f32(&mut age, [8, 8]);
        let st = case.horizon_stride.max(1);
        let nj_s = nj.div_ceil(st);
        let mut depth = vec![0.0f32; nk];
        for i in 0..ni {
            for j in 0..nj {
                let c = i * nj + j;
                let acol = &age[c * nk..(c + 1) * nk];
                let fh: Vec<f32> = case.horizons.iter().map(|&h| horizon_depth_from_age(acol, h as f64) as f32).collect();
                if i % st == 0 && j % st == 0 {
                    for (n, &v) in fh.iter().enumerate() {
                        let e = case.expected.horizon_depths[((i / st) * nj_s + j / st) * nh + n];
                        max_horizon_vs_fixture = max_horizon_vs_fixture.max((v as f64 - e).abs());
                    }
                }
                let lay = legacy_fill(&fh, nk);
                let col = &labels[c * nk..(c + 1) * nk];
                let mc = &maps[c * nh..(c + 1) * nh];
                layer_depth_trace(col, mc, &intervals, step, &mut depth);
                for k in 0..nk {
                    if lay[k] < 1 {
                        continue;
                    }
                    n_all += 1;
                    let l = col[k];
                    if l == 255 || lay[k] != l as i64 + 1 {
                        continue;
                    }
                    let h = l as usize;
                    let want = (fh[h + 1] - fh[0]) * step;
                    let naive = (mc[h + 1] as f32 - mc[0] as f32) * step;
                    let e = (depth[k] - want).abs() as f64;
                    let en = (naive - want).abs() as f64;
                    n_cmp += 1;
                    exact += usize::from(e == 0.0);
                    within += usize::from(e <= step as f64);
                    sum += e;
                    sum_naive += en;
                    max_e = max_e.max(e);
                    max_naive = max_naive.max(en);
                    errs.push(e);
                }
            }
        }
        eprintln!("case {}: {} faults", case.name, model.faults().len());
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| errs[((errs.len() - 1) as f64 * q) as usize];
    let (mean, mean_naive) = (sum / n_cmp as f64, sum_naive / n_cmp as f64);
    eprintln!(
        "faulted approximation (label runs) vs legacy re-picked horizons: {n_cmp} voxels \
         ({:.2}% of legacy sediment, same layer); mean |dz| {mean:.3} m, p50 {:.3} m, p95 {:.3} m, \
         p99 {:.3} m, max {max_e:.3} m, exact {:.1}%, within one sample (4 m) {:.2}%; \
         naive depth-cube faulting: mean {mean_naive:.3} m, max {max_naive:.3} m; \
         horizon re-pick vs legacy fixture max {max_horizon_vs_fixture:.2e} samples",
        100.0 * n_cmp as f64 / n_all as f64,
        p(0.5),
        p(0.95),
        p(0.99),
        100.0 * exact as f64 / n_cmp as f64,
        100.0 * within as f64 / n_cmp as f64
    );
    assert!(max_horizon_vs_fixture < 0.05);
    assert!(mean < 0.2 * mean_naive, "label runs must beat the naive depth-cube faulting");
    assert!(mean < 1.5, "mean |dz| {mean} m");
    assert!(within as f64 / n_cmp as f64 > 0.9);
}
