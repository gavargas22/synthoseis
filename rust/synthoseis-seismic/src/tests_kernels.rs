use super::*;
use serde::Deserialize;
use synthoseis_core::parity;

#[derive(Debug, Deserialize)]
struct FixtureFile {
    meta: Meta,
    scalar_zoeppritz: Vec<ScalarCase>,
    props: Props,
    rfc: Vec<f32>,
    ricker_30hz_dt4_c1: Vec<f64>,
    ricker_25hz_dt4_c2_len: usize,
    ricker_25hz_dt4_c2_peak_idx: usize,
    ricker_25hz_dt4_c2_peak: f64,
    hanflat_ones8_pct050: Vec<f64>,
    convolve_same: ConvolveBlock,
    snr: SnrBlock,
    hilterman_weights: Vec<HiltermanCase>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    shape_props: [usize; 3],
    #[allow(dead_code)]
    shape_rfc: [usize; 4],
    angles_deg: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ScalarCase {
    vp1: f64,
    vs1: f64,
    rho1: f64,
    vp2: f64,
    vs2: f64,
    rho2: f64,
    angle_deg: f64,
    rpp: f32,
}

#[derive(Debug, Deserialize)]
struct Props {
    vp: Vec<f32>,
    vs: Vec<f32>,
    rho: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ConvolveBlock {
    trace: Vec<f64>,
    wavelet: Vec<f64>,
    out: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct SnrBlock {
    sn_db: Vec<f64>,
    std_ratio: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct HiltermanCase {
    angle_deg: f64,
    near: f64,
    far: f64,
}

fn load_fixture() -> FixtureFile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/seismic_kernels.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture: {e}"))
}

#[test]
fn zoeppritz_pp_matches_python_scalar_goldens() {
    let fix = load_fixture();
    for c in &fix.scalar_zoeppritz {
        let got = zoeppritz_pp(c.vp1, c.vs1, c.rho1, c.vp2, c.vs2, c.rho2, c.angle_deg);
        assert!(
            (got - c.rpp).abs() < 1e-5,
            "zoeppritz angle={} rust={got} python={}",
            c.angle_deg,
            c.rpp
        );
    }
}

#[test]
fn compute_rfc_volumes_matches_python_and_parity_mae() {
    let fix = load_fixture();
    let got = compute_rfc_volumes(
        &fix.props.vp,
        &fix.props.vs,
        &fix.props.rho,
        fix.meta.shape_props,
        &fix.meta.angles_deg,
    );
    assert_eq!(got.len(), fix.rfc.len());
    for (i, (g, e)) in got.iter().zip(fix.rfc.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "rfc[{i}] rust={g} python={e}"
        );
    }
    // Wire angle stacks through core parity harness.
    let report = parity::compare_volumes(&[0u8; 1], &[0u8; 1], &got, &fix.rfc);
    assert!(
        report.angle_mae <= parity::ANGLE_MAE_MAX,
        "mae={}",
        report.angle_mae
    );
    assert!(
        report.angle_max_abs <= parity::ANGLE_MAX_ABS_MAX,
        "max_abs={}",
        report.angle_max_abs
    );
    assert!(report.passes_defaults(), "{report:?}");
}

#[test]
fn ricker_and_hanflat_match_python_goldens() {
    let fix = load_fixture();
    let ones = vec![1.0f64; 8];
    let hf = hanflat(&ones, 0.50);
    assert_eq!(hf.len(), fix.hanflat_ones8_pct050.len());
    for (g, e) in hf.iter().zip(fix.hanflat_ones8_pct050.iter()) {
        assert!((g - e).abs() < 1e-9, "hanflat rust={g} python={e}");
    }

    let s1 = ricker(30.0, 4.0, 1);
    assert_eq!(s1.len(), fix.ricker_30hz_dt4_c1.len());
    for (i, (g, e)) in s1.iter().zip(fix.ricker_30hz_dt4_c1.iter()).enumerate() {
        assert!((g - e).abs() < 1e-8, "ricker_c1[{i}] rust={g} python={e}");
    }

    let s2 = ricker(25.0, 4.0, 2);
    assert_eq!(s2.len(), fix.ricker_25hz_dt4_c2_len);
    let peak_idx = s2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(peak_idx, fix.ricker_25hz_dt4_c2_peak_idx);
    assert!((s2[peak_idx] - fix.ricker_25hz_dt4_c2_peak).abs() < 1e-8);
}

#[test]
fn convolve_same_1d_matches_python_golden() {
    let fix = load_fixture();
    let got = convolve_same_1d(&fix.convolve_same.trace, &fix.convolve_same.wavelet);
    assert_eq!(got.len(), fix.convolve_same.out.len());
    for (g, e) in got.iter().zip(fix.convolve_same.out.iter()) {
        assert!((g - e).abs() < 1e-10, "convolve rust={g} python={e}");
    }
}

#[test]
fn apply_wavelet_traces_preserves_shape_and_peak() {
    // Tiny 2x2x8 cube with a spike; wavelet [0.25,0.5,0.25].
    let shape = [2usize, 2, 8];
    let mut cube = vec![0.0f32; 2 * 2 * 8];
    cube[2] = 1.0; // first trace spike
    let wav = [0.25f64, 0.5, 0.25];
    let out = apply_wavelet_traces(&cube, shape, &wav);
    assert_eq!(out.len(), cube.len());
    // Same-mode convolve of spike at index 2 with [0.25,0.5,0.25] -> peak 0.5 at 2.
    assert!((out[2] - 0.5).abs() < 1e-6);
    assert!((out[1] - 0.25).abs() < 1e-6);
    assert!((out[3] - 0.25).abs() < 1e-6);
}

#[test]
fn snr_and_hilterman_match_python_goldens() {
    let fix = load_fixture();
    for (db, expected) in fix.snr.sn_db.iter().zip(fix.snr.std_ratio.iter()) {
        let got = snr_std_ratio(*db);
        assert!((got - expected).abs() < 1e-9, "snr rust={got} python={expected}");
    }
    for c in &fix.hilterman_weights {
        let (near, far) = hilterman_noise_weights(c.angle_deg);
        assert!((near - c.near).abs() < 1e-9);
        assert!((far - c.far).abs() < 1e-9);
        assert!((near + far - 1.0).abs() < 1e-12);
    }
}
