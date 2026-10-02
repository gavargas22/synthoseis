//! End-to-end angle-stack parity against the REAL legacy Python generator.
//!
//! Fixture: `tests/fixtures/angle_stack_e2e.json` plus
//! `angle_stack_e2e_seed<N>.bin`, written by
//! `tests/fixtures/generate_angle_stack_e2e.py`. That script runs the legacy
//! pipeline (`datagenerator`: horizons → facies → faults → closures →
//! `build_elastic_properties`) on two small full models (16 × 16 × 510, fixed
//! seeds), then the legacy seismic chain on the legacy elastic cubes:
//! `create_rfc_volumes` (the Numba Zoeppritz kernel) →
//! `postprocess_rfc_cubes(rfc_raw, "noise_free")` (`apply_bandlimits` →
//! `apply_lateral_filter` → `apply_cumsum`), captured before legacy output
//! scaling. Legacy angles: 0, 7, 15, 24 and 45° (`incident_angles` 7/15/24
//! plus the `model_qc_volumes` 0° and 45°).
//!
//! Rust cannot regenerate the legacy geology (the toy model draws from its own
//! RNG), so the test feeds the legacy Vp / Vs / rho through the Rust
//! production chain from elastic properties onward:
//!
//! * reflectivity: `synthoseis_gpu::fuse_props_tile_cpu` (the CPU fuse of the
//!   default rock-physics path) with `NO_WAVELET` (the bandpass-on chain skips
//!   the Ricker) and the Zoeppritz form of `--legacy-zoeppritz`;
//! * filters: `synthoseis_core::pipeline_stream::apply_filters_to_volume`
//!   (the production bandpass + lateral filter design and kernels) with the
//!   legacy model's own corners, order and lateral size;
//! * cumsum: `cumsum_traces_f32` + the 2–100 Hz bandpass (kernel only; the
//!   relative-impedance deliverable is not wired into the Rust pipeline).
//!
//! Legacy stores `nk - 1` reflectivity samples per trace; the Rust fuse
//! stores `nk` with a trailing 0. The parity tests run the Rust filters on
//! the legacy `nk - 1` grid; `known_gap_trailing_sample_moves_bandpass_edge`
//! measures what the Rust production `nk` convention does instead.
//!
//! Results and tolerances are discussed in `docs/angle-stack-e2e-parity.md`.

use std::io::Read;
use std::path::PathBuf;

use serde_json::Value;
use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::pipeline_stream::apply_filters_to_volume;
use synthoseis_core::{FilterConfig, RockPhysicsConfig, ZoeppritzForm};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// FNV-1a 64 over the little-endian bytes of an f32 slice (same as the
/// generator's `fnv1a64_f32`).
fn fnv_f32(v: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

struct Case {
    j: Value,
    seed: u64,
    shape: [usize; 3],
    angles: Vec<f64>,
    vp: Vec<f32>,
    vs: Vec<f32>,
    rho: Vec<f32>,
    columns: Vec<(usize, usize)>,
    /// Legacy sampled traces `[column][angle]`, `nk - 1` samples each.
    s_rfc: Vec<Vec<Vec<f32>>>,
    s_stack: Vec<Vec<Vec<f32>>>,
    s_cumsum: Vec<Vec<Vec<f32>>>,
}

impl Case {
    fn nk(&self) -> usize {
        self.shape[2]
    }
    fn legacy_hash(&self, kind: &str, a: usize) -> &str {
        self.j["legacy"][kind]["fnv1a64"][a].as_str().unwrap()
    }
    fn legacy_stat(&self, kind: &str, a: usize, stat: &str) -> f64 {
        self.j["legacy"][kind]["stats"][a][stat].as_f64().unwrap()
    }
    /// Offset of trace `(i, j)` in an `(ni, nj, nk - 1)` legacy-grid cube.
    fn trace(&self, i: usize, j: usize) -> usize {
        (i * self.shape[1] + j) * (self.nk() - 1)
    }
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

fn load_cases() -> Vec<Case> {
    let dir = fixtures();
    let text = std::fs::read_to_string(dir.join("angle_stack_e2e.json")).expect("fixture json");
    let root: Value = serde_json::from_str(&text).expect("json");
    root["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let s = usizes(&c["shape"]);
            let shape = [s[0], s[1], s[2]];
            let n = shape.iter().product::<usize>();
            let angles: Vec<f64> =
                c["angles_deg"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let columns: Vec<(usize, usize)> = c["legacy"]["sample_columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = usizes(p);
                    (p[0], p[1])
                })
                .collect();
            let raw = std::fs::read(dir.join(c["props_blob"].as_str().unwrap())).expect("blob");
            let mut bytes = Vec::new();
            flate2::read::ZlibDecoder::new(&raw[..]).read_to_end(&mut bytes).expect("zlib");
            let f: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let (nc, na, nk1) = (columns.len(), angles.len(), shape[2] - 1);
            let block = nc * na * nk1;
            assert_eq!(f.len(), 3 * n + 3 * block, "blob layout");
            let samples = |off: usize| -> Vec<Vec<Vec<f32>>> {
                (0..nc)
                    .map(|ci| {
                        (0..na)
                            .map(|a| {
                                let o = off + (ci * na + a) * nk1;
                                f[o..o + nk1].to_vec()
                            })
                            .collect()
                    })
                    .collect()
            };
            Case {
                seed: c["seed"].as_u64().unwrap(),
                shape,
                vp: f[..n].to_vec(),
                vs: f[n..2 * n].to_vec(),
                rho: f[2 * n..3 * n].to_vec(),
                s_rfc: samples(3 * n),
                s_stack: samples(3 * n + block),
                s_cumsum: samples(3 * n + 2 * block),
                columns,
                angles,
                j: c.clone(),
            }
        })
        .collect()
}

/// Rust production reflectivity for the whole cube, `(ni, nj, nk)`, with the
/// Rust trailing sample 0.
fn rust_reflectivity(c: &Case, angle: f64, form: ZoeppritzForm) -> Vec<f32> {
    let mut out = vec![0.0f32; c.vp.len()];
    synthoseis_gpu::fuse_props_tile_cpu(
        &c.vp,
        &c.vs,
        &c.rho,
        c.nk(),
        synthoseis_gpu::NO_WAVELET,
        angle,
        form,
        &mut out,
    );
    out
}

/// Drop the last sample of every trace: `(ni, nj, nk)` → `(ni, nj, nk - 1)`.
fn legacy_grid(v: &[f32], nk: usize) -> Vec<f32> {
    v.chunks_exact(nk).flat_map(|t| t[..nk - 1].iter().copied()).collect()
}

/// Production filter config with the legacy model's own drawn parameters.
fn filter_cfg(c: &Case, samples: usize) -> E2eConfig {
    let bp: Vec<f64> =
        c.j["bandpass_hz"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(c.j["digi_ms"].as_f64().unwrap(), synthoseis_core::pipeline::TINY_DIGI);
    E2eConfig {
        inline_count: c.shape[0],
        crossline_count: c.shape[1],
        samples,
        filters: FilterConfig {
            bandpass_hz: Some([bp[0], bp[1]]),
            bandpass_order: c.j["bandpass_order"].as_u64().unwrap() as usize,
            lateral_size: c.j["lateral_filter_size"].as_u64().unwrap() as usize,
            ..FilterConfig::default()
        },
        ..E2eConfig::default()
    }
}

/// The form `--legacy-zoeppritz` selects.
fn legacy_form() -> ZoeppritzForm {
    RockPhysicsConfig { legacy_zoeppritz: true, ..RockPhysicsConfig::default() }.zoeppritz_form()
}

/// Rust filtered stack on the legacy `nk - 1` grid (bandpass + lateral).
fn stack_on_legacy_grid(c: &Case, rfc_legacy_grid: &[f32]) -> Vec<f32> {
    let mut v = rfc_legacy_grid.to_vec();
    apply_filters_to_volume(&filter_cfg(c, c.nk() - 1), &mut v);
    v
}

/// Legacy `apply_cumsum`: float32 cumsum along z, then the 2–100 Hz bandpass
/// (legacy `cfg.order`).
fn cumsum_on_legacy_grid(c: &Case, stack: &[f32]) -> Vec<f32> {
    let mut v = stack.to_vec();
    synthoseis_seismic::cumsum_traces_f32(&mut v, c.nk() - 1);
    let f = synthoseis_seismic::butterworth_bandpass(
        2.0,
        100.0,
        synthoseis_seismic::legacy_digitisation_ms(c.j["digi_ms"].as_f64().unwrap()),
        c.j["bandpass_order"].as_u64().unwrap() as usize,
    )
    .unwrap();
    f.filtfilt_traces_f32(&mut v, c.nk() - 1).unwrap();
    v
}

#[derive(Default, Debug, Clone, Copy)]
struct Diff {
    n: usize,
    exact: usize,
    max_abs: f64,
    sq_err: f64,
    sq_ref: f64,
}

impl Diff {
    fn add(&mut self, rust: f32, legacy: f32) {
        self.n += 1;
        if rust.to_bits() == legacy.to_bits() {
            self.exact += 1;
        }
        let d = (rust as f64 - legacy as f64).abs();
        self.max_abs = self.max_abs.max(d);
        self.sq_err += d * d;
        self.sq_ref += (legacy as f64).powi(2);
    }
    fn rel_rms(&self) -> f64 {
        if self.sq_ref == 0.0 {
            0.0
        } else {
            (self.sq_err / self.sq_ref).sqrt()
        }
    }
    fn exact_pct(&self) -> f64 {
        100.0 * self.exact as f64 / self.n.max(1) as f64
    }
}

/// Compare a Rust legacy-grid cube with the legacy sampled columns.
fn diff_samples(c: &Case, rust: &[f32], legacy: &[Vec<Vec<f32>>], a: usize) -> Diff {
    let mut d = Diff::default();
    for (ci, &(i, j)) in c.columns.iter().enumerate() {
        let o = c.trace(i, j);
        for (k, &l) in legacy[ci][a].iter().enumerate() {
            d.add(rust[o + k], l);
        }
    }
    d
}

fn std_f64(v: &[f32]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / n).sqrt()
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()))
}

/// f32 ulp at magnitude `x`.
fn ulp_at(x: f64) -> f64 {
    let x = x as f32;
    (f32::from_bits(x.to_bits() + 1) - x) as f64
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// Samples whose legacy magnitude is at most this are round-off residue of an
/// exactly zero reflection (identical elastic properties across the
/// interface): legacy's Numba `fastmath` real path and the Rust complex path
/// leave different residues (1e-31 … 3e-16).
const RESIDUE: f64 = 1e-6;

#[test]
fn fixture_is_the_documented_legacy_run() {
    let cases = load_cases();
    assert_eq!(cases.len(), 2);
    for c in &cases {
        assert_eq!(c.shape, [16, 16, 510]);
        assert_eq!(c.angles, vec![0.0, 7.0, 15.0, 24.0, 45.0]);
        assert!(c.j["lateral_filter_size"].as_u64().unwrap() > 1, "lateral filter exercised");
        assert!(c.vp.iter().chain(&c.vs).chain(&c.rho).all(|x| x.is_finite() && *x > 0.0));
        // Water column: legacy `water_properties` (1.028 g/cc, 1500 m/s, 1000 m/s).
        assert_eq!((c.vp[0], c.vs[0], c.rho[0]), (1500.0, 1000.0, 1.028));
    }
    let gas = &cases[0].j["closure_voxels"]["gas"];
    assert!(gas.as_u64().unwrap() > 0, "seed 25 has gas closures (strong contrasts)");
}

/// Reflectivity: Rust `--legacy-zoeppritz` vs the legacy Numba kernel.
///
/// * 0°: the whole cube is bit-identical (FNV hash of all 130,304 samples).
/// * Other angles: every sample with |r| > `RESIDUE` is within 1 f32 ulp
///   (the full-cube offline comparison finds 2 one-ulp samples in 1.3 M;
///   the sampled columns are bit-exact there); residue samples differ by
///   < 1e-12 (measured ≤ 3e-16).
/// * Full-cube std and max |r| agree to 1e-6 relative.
/// * Sensitivity: the textbook form (the default) differs from legacy by
///   > 1e-3 on these columns at every non-zero angle, so the fixture pins the
///   legacy `det` form.
#[test]
fn legacy_reflectivity_matches_rust_legacy_zoeppritz() {
    for c in load_cases() {
        let nk = c.nk();
        for (a, &angle) in c.angles.iter().enumerate() {
            let full = rust_reflectivity(&c, angle, legacy_form());
            assert!(full.chunks_exact(nk).all(|t| t[nk - 1] == 0.0), "Rust trailing sample");
            let rfc = legacy_grid(&full, nk);
            let hash_ok = fnv_f32(&rfc) == c.legacy_hash("rfc_raw", a);
            let d = diff_samples(&c, &rfc, &c.s_rfc, a);
            let (mut max_ulp, mut max_residue) = (0u32, 0.0f64);
            for (ci, &(i, j)) in c.columns.iter().enumerate() {
                let o = c.trace(i, j);
                for (k, &l) in c.s_rfc[ci][a].iter().enumerate() {
                    let r = rfc[o + k];
                    if (l as f64).abs() > RESIDUE {
                        assert_eq!(r.is_sign_negative(), l.is_sign_negative());
                        max_ulp = max_ulp.max((r.to_bits() as i64 - l.to_bits() as i64).unsigned_abs() as u32);
                    } else {
                        max_residue = max_residue.max((r as f64 - l as f64).abs());
                    }
                }
            }
            let tb = legacy_grid(&rust_reflectivity(&c, angle, ZoeppritzForm::Exact), nk);
            let d_tb = diff_samples(&c, &tb, &c.s_rfc, a);
            println!(
                "seed {} {angle:>2}°: rfc full-cube hash {} | columns exact {:.2}% max |Δ| {:.2e} max ulp (|r|>1e-6) {max_ulp} max residue Δ {max_residue:.1e} | textbook-vs-legacy max {:.3e}",
                c.seed,
                if hash_ok { "bit-identical" } else { "differs" },
                d.exact_pct(),
                d.max_abs,
                d_tb.max_abs
            );
            if angle == 0.0 {
                assert!(hash_ok, "0° reflectivity must be bit-identical to legacy");
                assert_eq!(d_tb.max_abs, 0.0, "forms coincide at normal incidence");
            } else {
                assert!(d_tb.max_abs > 1e-3, "fixture must distinguish det from d");
            }
            assert!(max_ulp <= 1, "reflectivity > 1 ulp from legacy");
            assert!(max_residue < 1e-12, "zero-reflection residue {max_residue}");
            assert!(rel(std_f64(&rfc), c.legacy_stat("rfc_raw", a, "std")) < 1e-6);
            assert!(rel(max_abs(&rfc), c.legacy_stat("rfc_raw", a, "max_abs")) < 1e-6);
        }
    }
}

/// Noise-free angle stacks (bandpass + lateral) and the cumsum deliverable on
/// the legacy `nk - 1` grid vs legacy `postprocess_rfc_cubes`.
///
/// Not bit-exact over the full cube: 99.4–99.7 % of stack samples and
/// 96.3–98.6 % of cumsum samples are identical, the rest differ at f32
/// rounding level (full cube: stack max |Δ| 3.7e-9 against peaks of
/// 0.06–0.09, cumsum 1.5e-8 against 0.18–0.26). Two
/// causes, both at rounding level:
/// 1. the reflectivity residues above feed the IIR filter;
/// 2. `lfilter_zi`: the fixture was generated with the repository's locked
///    scipy 1.17.1, which solves `(I − A) zi = B` with `linalg.solve`; the Rust
///    port follows the closed form of scipy ≥ 1.18. The two `zi` differ by
///    ≤ 1.6e-9 relative. With the Rust `zi` substituted, scipy's filtfilt
///    reproduces the Rust bandpass bit for bit, and the lateral filter is
///    bit-exact (see the doc).
///
/// Tolerance: max |Δ| ≤ 8 ulp(peak) and relative RMS ≤ 1e-7, i.e. rounding
/// noise; any modelling difference (wrong corner, order, padding, lateral
/// size, Zoeppritz form) is orders of magnitude larger.
#[test]
fn legacy_noise_free_stacks_match_on_legacy_grid() {
    for c in load_cases() {
        let nk = c.nk();
        for (a, &angle) in c.angles.iter().enumerate() {
            let rfc = legacy_grid(&rust_reflectivity(&c, angle, legacy_form()), nk);
            let stack = stack_on_legacy_grid(&c, &rfc);
            let cumsum = cumsum_on_legacy_grid(&c, &stack);
            for (kind, rust, legacy) in
                [("stack", &stack, &c.s_stack), ("cumsum", &cumsum, &c.s_cumsum)]
            {
                let d = diff_samples(&c, rust, legacy, a);
                let peak = c.legacy_stat(kind, a, "max_abs");
                let hash_ok = fnv_f32(rust) == c.legacy_hash(kind, a);
                println!(
                    "seed {} {angle:>2}° {kind:>6}: full-cube hash {} | columns exact {:.2}% max |Δ| {:.2e} ({:.2} ulp of peak {:.3e}) rel RMS {:.2e} | full-cube std rel {:.1e}",
                    c.seed,
                    if hash_ok { "bit-identical" } else { "differs" },
                    d.exact_pct(),
                    d.max_abs,
                    d.max_abs / ulp_at(peak),
                    peak,
                    d.rel_rms(),
                    rel(std_f64(rust), c.legacy_stat(kind, a, "std")),
                );
                assert!(d.max_abs <= 8.0 * ulp_at(peak), "{kind} max |Δ| {}", d.max_abs);
                assert!(d.rel_rms() <= 1e-7, "{kind} rel RMS {}", d.rel_rms());
                assert!(rel(std_f64(rust), c.legacy_stat(kind, a, "std")) < 1e-6);
                assert!(rel(max_abs(rust), peak) < 1e-6);
            }
        }
    }
}

/// KNOWN DISCREPANCY (not fixed here; this PR changes no output).
///
/// The Rust fuse writes `nk` reflectivity samples per trace with a trailing
/// 0; legacy has `nk - 1`. The production filters bandpass the `nk`-sample
/// trace, and `filtfilt`'s odd extension about the last sample (0 instead of
/// legacy's last reflectivity) plus the extra sample change the deepest part
/// of every trace. Relative to legacy, over the first `nk - 1` samples:
/// ≈ 1.2–1.8e-2 max |Δ| in the last 10 samples (stack peaks 0.06–0.09),
/// ~1e-3 at 50–100 samples above the base, ≤ 1.4e-5 beyond 200 samples,
/// relative RMS 5–16 % over the cube. Legacy itself is edge-affected there
/// too (its 10 pad samples), but differently.
///
/// This test pins the size and depth extent of the gap so that a fix (or a
/// regression) shows up here; update it and `docs/angle-stack-e2e-parity.md`
/// together.
#[test]
fn known_gap_trailing_sample_moves_bandpass_edge() {
    const BANDS: [(usize, usize); 6] = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, usize::MAX)];
    for c in load_cases() {
        let nk = c.nk();
        for (a, &angle) in c.angles.iter().enumerate() {
            let full = rust_reflectivity(&c, angle, legacy_form());
            let parity = stack_on_legacy_grid(&c, &legacy_grid(&full, nk));
            let mut production = full.clone();
            apply_filters_to_volume(&filter_cfg(&c, nk), &mut production);
            let production = legacy_grid(&production, nk);

            // Full cube, against the parity stack (= legacy to rounding, above).
            let mut all = Diff::default();
            let mut band = [0.0f64; 6];
            for t in 0..c.shape[0] * c.shape[1] {
                for k in 0..nk - 1 {
                    let (p, q) = (production[t * (nk - 1) + k], parity[t * (nk - 1) + k]);
                    all.add(p, q);
                    let from_base = nk - 2 - k;
                    let b = BANDS.iter().position(|&(lo, hi)| from_base >= lo && from_base < hi).unwrap();
                    band[b] = band[b].max((p as f64 - q as f64).abs());
                }
            }
            // Sampled columns, directly against legacy.
            let d_leg = diff_samples(&c, &production, &c.s_stack, a);
            let peak = c.legacy_stat("stack", a, "max_abs");
            println!(
                "seed {} {angle:>2}°: production vs legacy columns max {:.3e} rel RMS {:.3e} | full cube max {:.3e} rel RMS {:.3e} (peak {:.3e}) | max |Δ| by samples above base [0,10) {:.1e} [10,25) {:.1e} [25,50) {:.1e} [50,100) {:.1e} [100,200) {:.1e} [200,) {:.1e}",
                c.seed, d_leg.max_abs, d_leg.rel_rms(), all.max_abs, all.rel_rms(), peak,
                band[0], band[1], band[2], band[3], band[4], band[5],
            );
            // The gap exists and is concentrated at the base ...
            assert!(band[0] > 5e-3 && band[0] < 5e-2, "basal gap {}", band[0]);
            assert!(all.rel_rms() > 1e-2 && all.rel_rms() < 0.3, "rel RMS {}", all.rel_rms());
            // ... and decays with distance from it.
            assert!(band[3] < 2e-3, "50-100 samples above base {}", band[3]);
            assert!(band[4] < 5e-4, "100-200 samples above base {}", band[4]);
            assert!(band[5] < 5e-5, ">= 200 samples above base {}", band[5]);
        }
    }
}
