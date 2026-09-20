use super::*;
use serde::Deserialize;
use synthoseis_core::parity;

const ATOL: f64 = 1e-9;

#[derive(Debug, Deserialize)]
struct FixtureFile {
    plane: PlaneBlock,
    rotate_point: RotateBlock,
    horizon_clip: ClipBlock,
    labels: LabelBlock,
}

#[derive(Debug, Deserialize)]
struct PlaneBlock {
    xyz: Vec<[f64; 3]>,
    abc: [f64; 3],
    eval_8x8: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct RotateBlock {
    cases: Vec<RotateCase>,
}

#[derive(Debug, Deserialize)]
struct RotateCase {
    x: f64,
    y: f64,
    deg: f64,
    out: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct ClipBlock {
    input: Vec<f64>,
    output: Vec<f64>,
    shape: [usize; 3],
}

#[derive(Debug, Deserialize)]
struct LabelBlock {
    reference: Vec<u8>,
}

fn load_fixture() -> FixtureFile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/parity_cubes_8.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture: {e}"))
}

#[test]
fn fit_plane_lsq_matches_python_golden() {
    let fix = load_fixture();
    let abc = fit_plane_lsq(&fix.plane.xyz);
    for k in 0..3 {
        assert!(
            (abc[k] - fix.plane.abc[k]).abs() < 1e-9,
            "abc[{k}] rust={} python={}",
            abc[k],
            fix.plane.abc[k]
        );
    }
}

#[test]
fn eval_plane_matches_python_golden() {
    let fix = load_fixture();
    let [a, b, c] = fix.plane.abc;
    let got = eval_plane(8, 8, a, b, c);
    assert_eq!(got.len(), fix.plane.eval_8x8.len());
    for (i, (g, e)) in got.iter().zip(fix.plane.eval_8x8.iter()).enumerate() {
        assert!(
            (g - e).abs() < ATOL,
            "eval_plane[{i}] rust={g} python={e}"
        );
    }
}

#[test]
fn rotate_point_matches_python_golden() {
    let fix = load_fixture();
    for case in &fix.rotate_point.cases {
        let (x1, y1) = rotate_point(case.x, case.y, case.deg);
        assert!((x1 - case.out[0]).abs() < ATOL);
        assert!((y1 - case.out[1]).abs() < ATOL);
    }
}

#[test]
fn enforce_nonnegative_thicknesses_matches_python_golden() {
    let fix = load_fixture();
    let mut maps = fix.horizon_clip.input.clone();
    enforce_nonnegative_thicknesses(&mut maps, fix.horizon_clip.shape);
    assert_eq!(maps.len(), fix.horizon_clip.output.len());
    for (i, (g, e)) in maps.iter().zip(fix.horizon_clip.output.iter()).enumerate() {
        assert!(
            (g - e).abs() < ATOL,
            "horizon_clip[{i}] rust={g} python={e}"
        );
    }
}

#[test]
fn fill_layer_labels_matches_python_and_parity_iou() {
    let fix = load_fixture();
    let mut maps = fix.horizon_clip.input.clone();
    enforce_nonnegative_thicknesses(&mut maps, fix.horizon_clip.shape);
    let labels = fill_layer_labels(&maps, fix.horizon_clip.shape, 8);
    assert_eq!(labels, fix.labels.reference);

    // Wire geo labels into the core parity harness (self IoU = 1).
    let report = parity::compare_volumes(
        &labels,
        &fix.labels.reference,
        &[0.0f32; 1],
        &[0.0f32; 1],
    );
    assert!((report.label_iou - 1.0).abs() < 1e-12);
    assert!((report.label_agreement - 1.0).abs() < 1e-12);
}
