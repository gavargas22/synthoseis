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
//! stores `nk` with a trailing 0. The Rust filters (default mode) bandpass
//! only the first `nk - 1` samples and leave the trailing sample 0, so the
//! tests feed the full `nk`-sample fuse output through the production
//! filters and compare the first `nk - 1` samples with legacy.
//! `trailing_sample_bandpass_edge_matches_legacy_to_the_base` pins the fix
//! and the size of the gap `--bandpass-trailing-sample` (the old whole-trace
//! behaviour) reproduces.
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

/// Production filter config with the legacy model's own drawn parameters
/// (default mode: the bandpass skips the trailing sample).
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
        time: synthoseis_core::TimeConfig::legacy(),
        ..E2eConfig::default()
    }
}

/// The form `--legacy-zoeppritz` selects.
fn legacy_form() -> ZoeppritzForm {
    RockPhysicsConfig { legacy_zoeppritz: true, ..RockPhysicsConfig::default() }.zoeppritz_form()
}

/// Rust production filtered stack (bandpass + lateral) of a full
/// `(ni, nj, nk)` fuse output, `trailing_sample` selecting
/// `--bandpass-trailing-sample`. Returns the `(ni, nj, nk)` cube.
fn production_stack(c: &Case, rfc_full: &[f32], trailing_sample: bool) -> Vec<f32> {
    let mut cfg = filter_cfg(c, c.nk());
    cfg.filters.bandpass_trailing_sample = trailing_sample;
    let mut v = rfc_full.to_vec();
    apply_filters_to_volume(&cfg, &mut v);
    v
}

/// Default production stack on the legacy `nk - 1` grid. Asserts that the
/// trailing sample the bandpass leaves out is 0 (the fuse's value, and what
/// the lateral filter keeps for an all-zero depth slice).
fn stack_on_legacy_grid(c: &Case, rfc_full: &[f32]) -> Vec<f32> {
    let nk = c.nk();
    let v = production_stack(c, rfc_full, false);
    assert!(v.chunks_exact(nk).all(|t| t[nk - 1].to_bits() == 0), "trailing sample not 0");
    legacy_grid(&v, nk)
}

/// Independent reference for the fix: the filters with the old whole-trace
/// bandpass run on the legacy `nk - 1` grid (what #35's parity tests did).
fn reference_on_legacy_grid(c: &Case, rfc_full: &[f32]) -> Vec<f32> {
    let mut cfg = filter_cfg(c, c.nk() - 1);
    cfg.filters.bandpass_trailing_sample = true;
    let mut v = legacy_grid(rfc_full, c.nk());
    apply_filters_to_volume(&cfg, &mut v);
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

/// Noise-free production angle stacks (bandpass + lateral, default mode, fed
/// the full `nk`-sample fuse output) and the cumsum deliverable, on the
/// legacy `nk - 1` grid, vs legacy `postprocess_rfc_cubes`.
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
        for (a, &angle) in c.angles.iter().enumerate() {
            let stack = stack_on_legacy_grid(&c, &rust_reflectivity(&c, angle, legacy_form()));
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

/// FIXED: the bandpass trailing-sample edge now matches legacy to the base.
///
/// The Rust fuse writes `nk` reflectivity samples per trace with a trailing
/// 0; legacy has `nk - 1`. Before the fix the production filters bandpassed
/// the whole `nk`-sample trace, and `filtfilt`'s odd extension about the last
/// sample (0 instead of legacy's last reflectivity) plus the extra sample
/// changed the deepest part of every trace (≈ 1.2–1.8e-2 max |Δ| in the last
/// 10 samples against stack peaks of 0.06–0.09, relative RMS 5–16 %).
///
/// Default mode now bandpasses only the first `nk - 1` samples, so:
/// * the default production stack equals the whole-trace filter run on the
///   legacy `nk - 1` grid bit for bit (full cube), with the trailing sample 0;
/// * against legacy, every depth band down to the last sample is within the
///   fixture tolerance (8 ulp of the peak; the residual is the `lfilter_zi`
///   and reflectivity-residue rounding of the parity test above);
/// * `--bandpass-trailing-sample` still reproduces the old gap (pinned by
///   its size and depth extent, so a regression in either mode shows here).
///
/// Update together with `docs/angle-stack-e2e-parity.md`.
#[test]
fn trailing_sample_bandpass_edge_matches_legacy_to_the_base() {
    const BANDS: [(usize, usize); 6] = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, usize::MAX)];
    let band_of = |from_base: usize| BANDS.iter().position(|&(lo, hi)| from_base >= lo && from_base < hi).unwrap();
    for c in load_cases() {
        let nk = c.nk();
        for (a, &angle) in c.angles.iter().enumerate() {
            let full = rust_reflectivity(&c, angle, legacy_form());
            let reference = reference_on_legacy_grid(&c, &full);
            let fixed = stack_on_legacy_grid(&c, &full);
            let old = legacy_grid(&production_stack(&c, &full, true), nk);
            let peak = c.legacy_stat("stack", a, "max_abs");
            let tol = 8.0 * ulp_at(peak);

            // Fixed mode: bit-identical to the reference over the full cube.
            assert_eq!(fnv_f32(&fixed), fnv_f32(&reference), "seed {} {angle}°", c.seed);
            assert!(fixed.iter().zip(&reference).all(|(x, y)| x.to_bits() == y.to_bits()));

            // Old mode, full cube vs the reference (= legacy to rounding).
            let mut old_all = Diff::default();
            let mut old_band = [0.0f64; 6];
            for t in 0..c.shape[0] * c.shape[1] {
                for k in 0..nk - 1 {
                    let (p, q) = (old[t * (nk - 1) + k], reference[t * (nk - 1) + k]);
                    old_all.add(p, q);
                    let b = band_of(nk - 2 - k);
                    old_band[b] = old_band[b].max((p as f64 - q as f64).abs());
                }
            }
            // Both modes, sampled columns directly against legacy, by depth band.
            let mut fixed_band = [0.0f64; 6];
            let mut old_leg_band = [0.0f64; 6];
            for (ci, &(i, j)) in c.columns.iter().enumerate() {
                let o = c.trace(i, j);
                for (k, &l) in c.s_stack[ci][a].iter().enumerate() {
                    let b = band_of(nk - 2 - k);
                    fixed_band[b] = fixed_band[b].max((fixed[o + k] as f64 - l as f64).abs());
                    old_leg_band[b] = old_leg_band[b].max((old[o + k] as f64 - l as f64).abs());
                }
            }
            let d_fixed = diff_samples(&c, &fixed, &c.s_stack, a);
            let d_old = diff_samples(&c, &old, &c.s_stack, a);
            let f = |v: &[f64; 6]| v.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" ");
            println!(
                "seed {} {angle:>2}° (peak {peak:.3e}, 8 ulp {tol:.1e}) columns vs legacy: fixed max {:.2e} rel RMS {:.2e} | old max {:.2e} rel RMS {:.2e} | max |Δ| by samples above base [0,10) [10,25) [25,50) [50,100) [100,200) [200,): fixed {} | old {} | old full cube {} (rel RMS {:.3e})",
                c.seed, d_fixed.max_abs, d_fixed.rel_rms(), d_old.max_abs, d_old.rel_rms(),
                f(&fixed_band), f(&old_leg_band), f(&old_band), old_all.rel_rms(),
            );
            // Fixed: rounding-level parity in every band, base included.
            for (b, &m) in fixed_band.iter().enumerate() {
                assert!(m <= tol, "fixed band {:?}: max |Δ| {m:e} > {tol:e}", BANDS[b]);
            }
            assert!(d_fixed.rel_rms() <= 1e-7, "fixed rel RMS {}", d_fixed.rel_rms());

            // Old (--bandpass-trailing-sample): the gap exists and is
            // concentrated at the base ...
            assert!(old_band[0] > 5e-3 && old_band[0] < 5e-2, "basal gap {}", old_band[0]);
            assert!(old_all.rel_rms() > 1e-2 && old_all.rel_rms() < 0.3, "rel RMS {}", old_all.rel_rms());
            assert!(old_leg_band[0] > 1e3 * tol, "old basal gap vs legacy {}", old_leg_band[0]);
            // ... and decays with distance from it.
            assert!(old_band[3] < 2e-3, "50-100 samples above base {}", old_band[3]);
            assert!(old_band[4] < 5e-4, "100-200 samples above base {}", old_band[4]);
            assert!(old_band[5] < 5e-5, ">= 200 samples above base {}", old_band[5]);
        }
    }
}

/// Depth-to-time at a uniform 2000 m/s (spec §5.2), on the legacy fixture.
///
/// The legacy axis implies a constant 2000 m/s (one 4 m cell = 4 ms), so
/// time mode with every column's T built from 2000 m/s (the
/// constant-velocity test hook; Zoeppritz still sees the legacy Vp / Vs /
/// rho) must be the legacy axis moved down one sample:
///
/// * raw reflectivity: the production time-mode fuse
///   (`time_mode::fuse_props_tile_time`, sinc insertion) equals the
///   production depth fuse (`fuse_props_tile_cpu`, the actual depth-path
///   output) one sample down, bit for bit (`r_k` lands on `T_{k+1}`, an
///   exact sample). No Zoeppritz is recomputed here.
/// * stacks: the production time-mode filters (bandpass designed at dt with
///   the dead last sample, lateral filter) on `nt = nk + 1` samples, read
///   one sample down, against #35's legacy Python stack with #35's
///   tolerance (max |Δ| ≤ 8 ulp of the peak, relative RMS ≤ 1e-7).
///
///   The one physical difference is the top edge: legacy puts the first
///   interface at t = 0 and `filtfilt` pads by odd extension about that
///   sample; the time trace has the correct zero sample above it. The two
///   paddings agree when the first `padlen = 3 (2·order + 1)` = 27 samples
///   are reflection-free (water column) across the lateral filter's
///   footprint. Those "clean" sampled columns (3 of seed 3's, 2 of seed
///   25's) meet the tolerance on the whole trace, at the same rounding level
///   as the depth path. Seed 25's other two sampled columns have the seabed
///   reflection at sample 21 < 27, so they carry a decaying top-edge
///   transient (2.2–2.8e-3, ≤ 4 % of the peak, in the first 50 samples;
///   2–3e-9 below sample 400), pinned here by depth band; they meet the
///   tolerance from sample 400 down. The 4 % is specific to these columns,
///   not a general bound: the transient's size depends on where the first
///   reflection sits inside `padlen` (Strata's synthetic reaches 30 % with
///   the seabed at sample 26–27), and it vanishes once the first reflection
///   is at or below `padlen`.
#[test]
fn legacy_fixture_time_mode_uniform_2000() {
    const BANDS: [usize; 9] = [50, 100, 150, 200, 250, 300, 350, 400, usize::MAX];
    for c in load_cases() {
        let nk = c.nk();
        let nt = nk + 1;
        let axis = synthoseis_core::TimeAxis {
            dt_ms: 4.0,
            nt,
            dz: 4.0,
            kernel: synthoseis_seismic::TwtKernel::Sinc,
            constant_twt_vp: Some(2000.0),
        };
        let time_cfg = E2eConfig {
            time: synthoseis_core::TimeConfig {
                samples: Some(nt),
                constant_twt_vp: Some(2000.0),
                ..synthoseis_core::TimeConfig::default()
            },
            ..filter_cfg(&c, nk)
        };
        assert!(time_cfg.time_enabled() && time_cfg.output_samples() == nt);
        let padlen = 3 * (2 * time_cfg.filters.bandpass_order + 1);
        for (a, &angle) in c.angles.iter().enumerate() {
            let depth = rust_reflectivity(&c, angle, legacy_form());
            let mut time = vec![0.0f32; c.shape[0] * c.shape[1] * nt];
            synthoseis_core::time_mode::fuse_props_tile_time(
                &c.vp,
                &c.vs,
                &c.rho,
                nk,
                &axis,
                synthoseis_gpu::NO_WAVELET,
                angle,
                legacy_form(),
                &mut time,
            );
            for (d, t) in depth.chunks_exact(nk).zip(time.chunks_exact(nt)) {
                assert_eq!(t[0].to_bits(), 0);
                assert!(
                    t[1..].iter().zip(d).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "seed {} {angle}°: time reflectivity != depth fuse one sample down",
                    c.seed
                );
            }
            // First reflection per column (|r| above the residue level).
            let first: Vec<usize> = depth
                .chunks_exact(nk)
                .map(|t| t.iter().position(|x| (*x as f64).abs() > RESIDUE).unwrap_or(nk))
                .collect();
            // Production time-mode filters, then back onto the legacy grid.
            apply_filters_to_volume(&time_cfg, &mut time);
            assert!(time.chunks_exact(nt).all(|t| t[nt - 1].to_bits() == 0), "dead last sample");
            let stack: Vec<f32> =
                time.chunks_exact(nt).flat_map(|t| t[1..nk].iter().copied()).collect();
            let peak = c.legacy_stat("stack", a, "max_abs");
            let tol = 8.0 * ulp_at(peak);
            let half = time_cfg.filters.lateral_size / 2;
            let (ni, nj) = (c.shape[0], c.shape[1]);
            // Edge columns: a reflection within padlen samples of the top
            // anywhere in the lateral filter's footprint.
            let edge_column = |i: usize, j: usize| {
                (i.saturating_sub(half)..(i + half + 1).min(ni))
                    .flat_map(|a| (j.saturating_sub(half)..(j + half + 1).min(nj)).map(move |b| (a, b)))
                    .any(|(a, b)| first[a * nj + b] < padlen)
            };
            // Optional evidence dump (PR description figures): per sampled
            // column, #35's legacy Python stack then the time-mode stack on
            // the legacy grid, `nk - 1` f32 each.
            if let Some(dir) = std::env::var_os("SYNTHOSEIS_D2T_EVIDENCE_DIR") {
                let mut out = Vec::new();
                for (ci, &(i, j)) in c.columns.iter().enumerate() {
                    let o = c.trace(i, j);
                    let traces = c.s_stack[ci][a].iter().chain(&stack[o..o + nk - 1]);
                    out.extend(traces.flat_map(|x| x.to_le_bytes()));
                }
                let name = format!("uniform2000_seed{}_angle{angle}.f32", c.seed);
                std::fs::write(PathBuf::from(dir).join(name), out).expect("evidence dump");
            }
            let (mut clean, mut edge_top, mut edge_below) = (Diff::default(), Diff::default(), Diff::default());
            let mut band = [0.0f64; 9];
            let mut n_edge = 0;
            for (ci, &(i, j)) in c.columns.iter().enumerate() {
                let o = c.trace(i, j);
                let is_edge = edge_column(i, j);
                n_edge += is_edge as usize;
                for (k, &l) in c.s_stack[ci][a].iter().enumerate() {
                    let r = stack[o + k];
                    if !is_edge {
                        clean.add(r, l);
                        continue;
                    }
                    let b = BANDS.iter().position(|&h| k < h).unwrap();
                    band[b] = band[b].max((r as f64 - l as f64).abs());
                    if k >= 400 { edge_below.add(r, l) } else { edge_top.add(r, l) }
                }
            }
            println!(
                "seed {} {angle:>2}° time-mode 2000 m/s stack vs legacy (8 ulp {tol:.1e}): {} clean columns max |Δ| {:.2e} rel RMS {:.2e} | {n_edge} top-edge columns (reflection above sample {padlen}) max |Δ| by samples [0,50) [50,100) … [350,400) [400,): {}",
                c.seed,
                c.columns.len() - n_edge,
                clean.max_abs,
                clean.rel_rms(),
                band.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" ")
            );
            assert!(clean.n > 0, "seed {}: no clean column", c.seed);
            assert!(clean.max_abs <= tol, "seed {} {angle}°: clean max |Δ| {}", c.seed, clean.max_abs);
            assert!(clean.rel_rms() <= 1e-7, "seed {} {angle}°: clean rel RMS {}", c.seed, clean.rel_rms());
            if n_edge > 0 {
                assert!(band[0] <= 0.04 * peak, "seed {} {angle}°: top-edge transient {} (seed-specific pin)", c.seed, band[0]);
                assert!(band.windows(2).all(|w| w[1] <= w[0]), "transient must decay: {band:?}");
                assert!(edge_below.max_abs <= tol, "seed {} {angle}°: below 400 max |Δ| {}", c.seed, edge_below.max_abs);
                // Relative to the edge columns' whole legacy energy (#35's
                // relative-RMS normalisation).
                let rms = (edge_below.sq_err / (edge_below.sq_ref + edge_top.sq_ref)).sqrt();
                assert!(rms <= 1e-7, "seed {} {angle}°: below 400 rel RMS {rms}", c.seed);
            }
        }
    }
}
