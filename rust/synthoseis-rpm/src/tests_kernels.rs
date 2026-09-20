use super::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureFile {
    example: TrendBlock,
    tagilsk: TagilskBlock,
}

#[derive(Debug, Deserialize)]
struct TrendBlock {
    z: Vec<f64>,
    shale_rho: Vec<f64>,
    shale_vp: Vec<f64>,
    shale_vs: Vec<f64>,
    brine_sand_rho: Vec<f64>,
    brine_sand_vp: Vec<f64>,
    brine_sand_vs: Vec<f64>,
    oil_sand_rho: Vec<f64>,
    oil_sand_vp: Vec<f64>,
    oil_sand_vs: Vec<f64>,
    gas_sand_rho: Vec<f64>,
    gas_sand_vp: Vec<f64>,
    gas_sand_vs: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct TagilskBlock {
    z: Vec<f64>,
    shale_rho: Vec<f64>,
    shale_vp: Vec<f64>,
    shale_vs: Vec<f64>,
    brine_sand_rho: Vec<f64>,
    brine_sand_vp: Vec<f64>,
    brine_sand_vs: Vec<f64>,
    gas_sand_vp: Vec<f64>,
    gas_sand_vs: Vec<f64>,
}

fn load_fixture() -> FixtureFile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rpm_trends.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture: {e}"))
}

fn assert_close(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label} len");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = (1e-6_f64).max(e.abs() * 1e-9);
        assert!(
            (g - e).abs() < tol,
            "{label}[{i}] rust={g} python={e} tol={tol}"
        );
    }
}

#[test]
fn rpm_example_trends_match_python_goldens() {
    let fix = load_fixture();
    let z = &fix.example.z;
    assert_close(&RpmExampleTrends::shale_rho(z), &fix.example.shale_rho, "shale_rho");
    assert_close(&RpmExampleTrends::shale_vp(z), &fix.example.shale_vp, "shale_vp");
    assert_close(&RpmExampleTrends::shale_vs(z), &fix.example.shale_vs, "shale_vs");
    assert_close(
        &RpmExampleTrends::brine_sand_rho(z),
        &fix.example.brine_sand_rho,
        "brine_sand_rho",
    );
    assert_close(
        &RpmExampleTrends::brine_sand_vp(z),
        &fix.example.brine_sand_vp,
        "brine_sand_vp",
    );
    assert_close(
        &RpmExampleTrends::brine_sand_vs(z),
        &fix.example.brine_sand_vs,
        "brine_sand_vs",
    );
    assert_close(
        &RpmExampleTrends::oil_sand_rho(z),
        &fix.example.oil_sand_rho,
        "oil_sand_rho",
    );
    assert_close(
        &RpmExampleTrends::oil_sand_vp(z),
        &fix.example.oil_sand_vp,
        "oil_sand_vp",
    );
    assert_close(
        &RpmExampleTrends::oil_sand_vs(z),
        &fix.example.oil_sand_vs,
        "oil_sand_vs",
    );
    assert_close(
        &RpmExampleTrends::gas_sand_rho(z),
        &fix.example.gas_sand_rho,
        "gas_sand_rho",
    );
    assert_close(
        &RpmExampleTrends::gas_sand_vp(z),
        &fix.example.gas_sand_vp,
        "gas_sand_vp",
    );
    assert_close(
        &RpmExampleTrends::gas_sand_vs(z),
        &fix.example.gas_sand_vs,
        "gas_sand_vs",
    );
}

#[test]
fn tagilsk_trends_match_python_goldens() {
    let fix = load_fixture();
    let z = &fix.tagilsk.z;
    assert_close(&tagilsk_shale_rho(z), &fix.tagilsk.shale_rho, "tag_shale_rho");
    assert_close(&tagilsk_shale_vp(z), &fix.tagilsk.shale_vp, "tag_shale_vp");
    assert_close(&tagilsk_shale_vs(z), &fix.tagilsk.shale_vs, "tag_shale_vs");
    assert_close(
        &tagilsk_brine_sand_rho(z),
        &fix.tagilsk.brine_sand_rho,
        "tag_brine_rho",
    );
    assert_close(
        &tagilsk_brine_sand_vp(z),
        &fix.tagilsk.brine_sand_vp,
        "tag_brine_vp",
    );
    assert_close(
        &tagilsk_brine_sand_vs(z),
        &fix.tagilsk.brine_sand_vs,
        "tag_brine_vs",
    );
    assert_close(&tagilsk_gas_sand_vp(z), &fix.tagilsk.gas_sand_vp, "tag_gas_vp");
    assert_close(&tagilsk_gas_sand_vs(z), &fix.tagilsk.gas_sand_vs, "tag_gas_vs");
}

#[test]
fn polyval_matches_numpy_ordering() {
    // numpy.polyval([1, 2, 3], 4) = 1*16 + 2*4 + 3 = 27
    assert!((polyval(&[1.0, 2.0, 3.0], 4.0) - 27.0).abs() < 1e-12);
}
