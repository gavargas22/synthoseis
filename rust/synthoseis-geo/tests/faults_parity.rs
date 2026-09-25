//! Parity of the Rust fault port against the legacy Python generator.
//!
//! Fixture: `tests/fixtures/fault_cubes.json`, produced by
//! `tests/fixtures/generate_fault_cubes.py`, which runs the real
//! `Faults.build_faults` + `apply_xyz_displacement` with recorded random draws.
//! Rust is fed the identical explicit parameters (random draws bypassed).
//!
//! The committed 32×32×48 fixture is too short for the legacy seabed taper
//! (sigma draws up to 300 samples), so it is compared in `ReachMode::Legacy`.
//! `tests/fixtures/fault_cubes_tall.json` (16×16×640, taper always succeeds)
//! is compared in the default `ReachMode::FitColumn`, which must be
//! bit-identical to legacy there.

use serde::Deserialize;
use synthoseis_geo::faults::{
    horizon_depth_from_age, middle_candidates, survey_surface, FaultGeometry, FaultModel,
    FaultParams, ReachMode, Seabed,
};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
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
    py_lateral_offset: Option<[f64; 2]>,
    sigma: Option<f64>,
    p: Option<f64>,
    coef: Option<f64>,
}

#[derive(Deserialize)]
struct Expected {
    /// Alternating 0/1 run lengths (C order), starting with a run of 0s.
    fault_mask_runs: Vec<u64>,
    n_fault_voxels: u64,
    horizon_depths: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: [usize; 3],
    infill_factor: f64,
    wb_const: f64,
    age: AgeCfg,
    horizons: Vec<i64>,
    #[serde(default = "one")]
    horizon_stride: usize,
    faults: Vec<FaultJson>,
    expected: Expected,
}

/// Parity metrics for one case.
#[derive(Debug)]
pub struct CaseParity {
    pub mask_agreement: f64,
    pub mask_iou: f64,
    pub py_fault_voxels: u64,
    pub rs_fault_voxels: u64,
    pub horizon_max_abs: f64,
    pub horizon_mean_abs: f64,
    pub horizon_frac_within_0p01: f64,
}

fn one() -> usize {
    1
}

fn load() -> Fixture {
    // SYNTHOSEIS_FAULT_FIXTURE overrides the committed fixture (e.g. a larger
    // `generate_fault_cubes.py --sweep N` run that is not committed).
    let path = std::env::var("SYNTHOSEIS_FAULT_FIXTURE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| fixture_path("fault_cubes.json"));
    load_from(&path)
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn load_from(path: &std::path::Path) -> Fixture {
    let text = std::fs::read_to_string(path).expect("read fault fixture");
    serde_json::from_str(&text).expect("parse fault fixture")
}

fn decode_runs(runs: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    for (idx, &n) in runs.iter().enumerate() {
        out.extend(std::iter::repeat((idx % 2) as u8).take(n as usize));
    }
    out
}

fn params(case: &Case) -> Vec<FaultParams> {
    params_with(case, false)
}

/// `replay_ties = true` also replays Python's rotated-argmax half-pixel
/// offset (only differs from the analytic choice on exact 4-way ties).
fn params_with(case: &Case, replay_ties: bool) -> Vec<FaultParams> {
    case.faults
        .iter()
        .map(|f| {
            let p = FaultParams::from_legacy(
                f.a,
                f.b,
                f.c,
                f.x0,
                f.y0,
                f.z0,
                f.throw,
                f.tilt_pct,
                case.infill_factor,
            );
            let p = match (f.sigma, f.p, f.coef) {
                (Some(s), Some(pp), Some(c)) => p.with_profile(s, pp, c),
                _ => p,
            };
            let p = p.with_center(f.center);
            if replay_ties {
                p.with_lateral_offset(f.py_lateral_offset)
            } else {
                p
            }
        })
        .collect()
}

fn age_volume(case: &Case) -> Vec<f32> {
    let [ni, nj, nk] = case.shape;
    let a = &case.age;
    let mut v = vec![0.0f32; ni * nj * nk];
    for i in 0..ni {
        for j in 0..nj {
            let spacing = a.spacing_base + a.spacing_di * i as f64 + a.spacing_dj * j as f64;
            for k in 0..nk {
                v[(i * nj + j) * nk + k] = ((k as f64 - a.z_top) / spacing) as f32;
            }
        }
    }
    v
}

fn run_case(case: &Case, tile: [usize; 2]) -> (Vec<u8>, Vec<f32>, FaultModel) {
    run_case_with(case, tile, false, ReachMode::Legacy)
}

fn run_case_with(
    case: &Case,
    tile: [usize; 2],
    replay_ties: bool,
    mode: ReachMode,
) -> (Vec<u8>, Vec<f32>, FaultModel) {
    let ps = params_with(case, replay_ties);
    let model =
        FaultModel::resolve_with_mode(case.shape, &ps, &Seabed::Flat(case.wb_const), 0, mode);
    let mut age = age_volume(case);
    let mask = model.apply_to_volume_f32(&mut age, tile);
    (mask, age, model)
}

fn parity(case: &Case, mask: &[u8], age: &[f32]) -> CaseParity {
    let [ni, nj, nk] = case.shape;
    let py = decode_runs(&case.expected.fault_mask_runs);
    assert_eq!(py.len(), mask.len());
    let (mut same, mut inter, mut uni) = (0u64, 0u64, 0u64);
    for (&a, &b) in py.iter().zip(mask) {
        same += u64::from(a == b);
        inter += u64::from(a == 1 && b == 1);
        uni += u64::from(a == 1 || b == 1);
    }
    let nh = case.horizons.len();
    let st = case.horizon_stride.max(1);
    let (ni_s, nj_s) = (ni.div_ceil(st), nj.div_ceil(st));
    let (mut max_abs, mut sum_abs, mut within) = (0.0f64, 0.0f64, 0u64);
    for (a, i) in (0..ni).step_by(st).enumerate() {
        for (b, j) in (0..nj).step_by(st).enumerate() {
            let col = &age[(i * nj + j) * nk..(i * nj + j + 1) * nk];
            for (n, &h) in case.horizons.iter().enumerate() {
                let d = horizon_depth_from_age(col, h as f64);
                let e = case.expected.horizon_depths[(a * nj_s + b) * nh + n];
                let diff = (d - e).abs();
                max_abs = max_abs.max(diff);
                sum_abs += diff;
                within += u64::from(diff <= 0.01);
            }
        }
    }
    let nhz = (ni_s * nj_s * nh) as f64;
    CaseParity {
        mask_agreement: same as f64 / py.len() as f64,
        mask_iou: if uni == 0 {
            1.0
        } else {
            inter as f64 / uni as f64
        },
        py_fault_voxels: case.expected.n_fault_voxels,
        rs_fault_voxels: mask.iter().map(|&v| v as u64).sum(),
        horizon_max_abs: max_abs,
        horizon_mean_abs: sum_abs / nhz,
        horizon_frac_within_0p01: within as f64 / nhz,
    }
}

#[test]
fn fault_parity_vs_python_fixture() {
    let fx = load();
    let sweep = std::env::var("SYNTHOSEIS_FAULT_FIXTURE").is_ok();
    assert!(!fx.cases.is_empty());
    let mut worst_iou = 1.0f64;
    let (mut worst_h, mut worst_h_replay) = (0.0f64, 0.0f64);
    for case in &fx.cases {
        let (mask, age, model) = run_case(case, [8, 8]);
        // Faults Python skipped must be skipped by Rust as well.
        let py_active = case.faults.iter().filter(|f| f.center.is_some()).count();
        assert_eq!(
            model.faults().len(),
            py_active,
            "{}: active faults",
            case.name
        );
        let p = parity(case, &mask, &age);
        let (mask_r, age_r, _) = run_case_with(case, [8, 8], true, ReachMode::Legacy);
        let pr = parity(case, &mask_r, &age_r);
        println!("{}: analytic {p:?}", case.name);
        println!("{}: replay   {pr:?}", case.name);
        worst_iou = worst_iou.min(p.mask_iou).min(pr.mask_iou);
        worst_h = worst_h.max(p.horizon_max_abs);
        worst_h_replay = worst_h_replay.max(pr.horizon_max_abs);
        if let Ok(dir) = std::env::var("SYNTHOSEIS_FAULT_DUMP") {
            let d = std::path::Path::new(&dir);
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join(format!("{}_rs_mask.u8", case.name)), &mask).unwrap();
            let bytes: Vec<u8> = age.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(d.join(format!("{}_rs_faulted_age.f32", case.name)), bytes).unwrap();
        }
        // Labels: must match to high agreement in every mode.
        assert!(p.mask_iou >= 0.95, "{}: mask IoU {p:?}", case.name);
        assert!(
            p.mask_agreement >= 0.999,
            "{}: mask agreement {p:?}",
            case.name
        );
        // Displaced horizons: exact replay must be within f32 round-off +
        // fixture rounding; analytic mode is checked on the committed fixture.
        assert!(
            pr.horizon_frac_within_0p01 >= 0.999,
            "{}: replay horizons {pr:?}",
            case.name
        );
        if !sweep {
            assert!(
                p.horizon_frac_within_0p01 >= 0.99,
                "{}: horizons {p:?}",
                case.name
            );
        }
    }
    println!(
        "SUMMARY cases={} worst_mask_iou={worst_iou} worst_horizon_max_abs(analytic)={worst_h} worst_horizon_max_abs(replay)={worst_h_replay}",
        fx.cases.len()
    );
}

#[test]
fn python_centre_is_a_rust_middle_candidate() {
    // The Python run picked `center` with rng.choice over get_middle_z's final
    // subset; the Rust port of get_middle_z must contain that voxel.
    let fx = load();
    for case in &fx.cases {
        for (n, (f, p)) in case.faults.iter().zip(params(case)).enumerate() {
            let geom = FaultGeometry::new(case.shape, &p);
            let (cands, _) =
                survey_surface(case.shape, &geom, &Seabed::Flat(case.wb_const), [8, 8]);
            match f.center {
                Some(c) => {
                    let sub = middle_candidates(&cands, case.shape[0]).expect("candidates");
                    let c32 = [c[0] as u32, c[1] as u32, c[2] as u32];
                    assert!(
                        sub.contains(&c32),
                        "{} fault {n}: {c:?} not in {sub:?}",
                        case.name
                    );
                }
                None => assert!(
                    middle_candidates(&cands, case.shape[0]).is_none(),
                    "{} fault {n}: Python skipped but Rust found candidates",
                    case.name
                ),
            }
        }
    }
}

#[test]
fn fault_parity_tiling_invariant() {
    let fx = load();
    let case = &fx.cases[fx.cases.len() - 1];
    let (m1, a1, _) = run_case(case, [case.shape[0], case.shape[1]]);
    let (m2, a2, _) = run_case(case, [5, 7]);
    let (m3, a3, _) = run_case(case, [1, 1]);
    assert_eq!(m1, m2);
    assert_eq!(m1, m3);
    assert_eq!(a1, a2);
    assert_eq!(a1, a3);
}

/// Legacy parity + default-mode exactness on a cube tall enough for the
/// legacy seabed taper (16×16×640, wb = 4): `FitColumn` must not rescue any
/// fault and must be bit-identical to `Legacy`, hence match Python equally.
#[test]
fn tall_fixture_default_mode_is_legacy_exact() {
    // SYNTHOSEIS_FAULT_TALL_FIXTURE: e.g. `generate_fault_cubes.py --tall
    // --tall-seeds 1,2,...,30 --out /tmp/tall.json` (not committed).
    let path = std::env::var("SYNTHOSEIS_FAULT_TALL_FIXTURE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| fixture_path("fault_cubes_tall.json"));
    let sweep = std::env::var("SYNTHOSEIS_FAULT_TALL_FIXTURE").is_ok();
    let fx = load_from(&path);
    assert!(!fx.cases.is_empty());
    for case in &fx.cases {
        for replay in [false, true] {
            let (m_leg, a_leg, leg) = run_case_with(case, [8, 8], replay, ReachMode::Legacy);
            let (m_def, a_def, def) = run_case_with(case, [8, 8], replay, ReachMode::FitColumn);
            let py_active = case.faults.iter().filter(|f| f.center.is_some()).count();
            assert_eq!(
                def.faults().len(),
                py_active,
                "{}: active faults",
                case.name
            );
            assert!(
                leg.faults().iter().all(|f| f.seabed_ok),
                "{}: legacy taper",
                case.name
            );
            assert_eq!(def.reach_rescued(), 0, "{}: rescued", case.name);
            assert_eq!(m_def, m_leg, "{}: mask default vs legacy", case.name);
            assert_eq!(a_def, a_leg, "{}: age default vs legacy", case.name);
            let p = parity(case, &m_def, &a_def);
            println!("{} (default mode, replay_ties={replay}): {p:?}", case.name);
            assert!(p.mask_iou >= 0.95, "{}: mask IoU {p:?}", case.name);
            assert!(p.mask_agreement >= 0.999, "{}: agreement {p:?}", case.name);
            if replay && !sweep {
                // Analytic mode differs from Python only on exact 4-way argmax
                // ties of the lateral gaussian (tall_5 fault 0 has one). In
                // sweeps, numpy's SIMD `exp` can also move the vertical
                // plateau argmax by one sample (see docs/faults-port.md).
                assert!(
                    p.horizon_frac_within_0p01 >= 0.999,
                    "{}: horizons {p:?}",
                    case.name
                );
            }
            let wb = case.wb_const;
            let [ni, nj, nk] = case.shape;
            let above = (0..ni * nj)
                .flat_map(|c| (0..nk).map(move |k| (c, k)))
                .filter(|&(c, k)| (k as f64) < wb && m_def[c * nk + k] == 1)
                .count();
            assert_eq!(above, 0, "{}: fault voxels above seabed", case.name);
        }
    }
}

/// On the short fixture the legacy taper gives up; `FitColumn` rescues it:
/// every seabed taper succeeds, no fault voxel sits above the seabed, and the
/// result stays tiling-invariant and deterministic.
#[test]
fn short_fixture_fit_column_keeps_water_column_clean() {
    let fx = load();
    for case in &fx.cases {
        let (m, a, model) = run_case_with(case, [8, 8], false, ReachMode::FitColumn);
        assert!(model.faults().iter().all(|f| f.seabed_ok), "{}", case.name);
        let (m2, a2, _) = run_case_with(case, [5, 7], false, ReachMode::FitColumn);
        assert_eq!(m, m2, "{}: tiling", case.name);
        assert_eq!(a, a2, "{}: tiling", case.name);
        let [ni, nj, nk] = case.shape;
        let wb = case.wb_const;
        for c in 0..ni * nj {
            for k in 0..nk {
                if (k as f64) < wb {
                    assert_eq!(m[c * nk + k], 0, "{}: mask above seabed", case.name);
                }
            }
        }
        let (m_leg, _, _) = run_case(case, [8, 8]);
        let n = |v: &[u8]| v.iter().map(|&x| x as u64).sum::<u64>();
        println!(
            "{}: legacy voxels={} fit_column voxels={} rescued={}",
            case.name,
            n(&m_leg),
            n(&m),
            model.reach_rescued()
        );
    }
}
