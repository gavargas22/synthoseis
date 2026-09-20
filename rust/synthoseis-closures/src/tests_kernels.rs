use super::*;
use serde::Deserialize;
use synthoseis_core::parity;

#[derive(Debug, Deserialize)]
struct FixtureFile {
    labels_gappy_rle: Vec<[i32; 2]>,
    bincount: Vec<i64>,
    relabel: RelabelBlock,
    fluid: FluidBlock,
    filter_min5: FilterBlock,
    closure_codes: CodesBlock,
    top_of_closure: TopBlock,
    bbox: BboxBlock,
    flood_fill_2d: FloodBlock,
}

#[derive(Debug, Deserialize)]
struct RelabelBlock {
    label_values: Vec<i32>,
    labels_rle: Vec<[i32; 2]>,
}

#[derive(Debug, Deserialize)]
struct FluidBlock {
    fluid_type_code: Vec<i32>,
    oil_rle: Vec<[i32; 2]>,
    gas_rle: Vec<[i32; 2]>,
    brine_rle: Vec<[i32; 2]>,
}

#[derive(Debug, Deserialize)]
struct FilterBlock {
    input_rle: Vec<[i32; 2]>,
    output_rle: Vec<[i32; 2]>,
    min_voxels: i64,
    sizes_before: Vec<i64>,
    sizes_after: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct CodesBlock {
    labels_rle: Vec<[i32; 2]>,
    num: i32,
    code: f32,
    hc_flat: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TopBlock {
    input_rle: Vec<[i32; 2]>,
    output_rle: Vec<[i32; 2]>,
    pad_up: usize,
    pad_down: usize,
}

#[derive(Debug, Deserialize)]
struct BboxBlock {
    label_id: i32,
    fault_block_val: f32,
    fault_tol: f32,
    pad: usize,
    slices: Option<[[usize; 2]; 3]>,
}

#[derive(Debug, Deserialize)]
struct FloodBlock {
    shape: [usize; 2],
    input: Vec<f64>,
    output: Vec<f64>,
}

fn expand_rle_i32(pairs: &[[i32; 2]]) -> Vec<i32> {
    let mut out = Vec::new();
    for [value, count] in pairs {
        out.extend(std::iter::repeat(*value).take(*count as usize));
    }
    out
}

fn expand_rle_u8(pairs: &[[i32; 2]]) -> Vec<u8> {
    expand_rle_i32(pairs).into_iter().map(|v| v as u8).collect()
}

fn load_fixture() -> FixtureFile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/closure_cubes_8.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture: {e}"))
}

#[test]
fn bincount_label_sizes_matches_python_golden() {
    let fix = load_fixture();
    let labels = expand_rle_i32(&fix.labels_gappy_rle);
    assert_eq!(bincount_label_sizes(&labels), fix.bincount);
}

#[test]
fn relabel_consecutive_matches_python_golden() {
    let fix = load_fixture();
    let labels = expand_rle_i32(&fix.labels_gappy_rle);
    let (vals, rel) = relabel_consecutive(&labels);
    assert_eq!(vals, fix.relabel.label_values);
    assert_eq!(rel, expand_rle_i32(&fix.relabel.labels_rle));
}

#[test]
fn filter_labels_by_min_voxels_and_sizes_match_python() {
    let fix = load_fixture();
    let input = expand_rle_i32(&fix.filter_min5.input_rle);
    let got = filter_labels_by_min_voxels(&input, fix.filter_min5.min_voxels);
    assert_eq!(got, expand_rle_i32(&fix.filter_min5.output_rle));
    let (s, t) = closure_size_filter_sizes(&input, &got);
    assert_eq!(s, fix.filter_min5.sizes_before);
    assert_eq!(t, fix.filter_min5.sizes_after);
}

#[test]
fn parse_closure_codes_matches_python_golden() {
    let fix = load_fixture();
    let labels = expand_rle_i32(&fix.closure_codes.labels_rle);
    let hc = vec![0.0f32; labels.len()];
    let got = parse_closure_codes(&hc, &labels, fix.closure_codes.num, fix.closure_codes.code);
    assert_eq!(got.len(), fix.closure_codes.hc_flat.len());
    for (i, (g, e)) in got.iter().zip(fix.closure_codes.hc_flat.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "closure_codes[{i}] rust={g} python={e}"
        );
    }
}

#[test]
fn assign_fluid_types_matches_python_and_parity_iou() {
    let fix = load_fixture();
    let labels = expand_rle_i32(&fix.relabel.labels_rle);
    let segs: Vec<f32> = labels.iter().map(|&v| if v > 0 { 1.0 } else { 0.0 }).collect();
    let (oil, gas, brine) =
        assign_fluid_types(&labels, &segs, &fix.fluid.fluid_type_code);
    assert_eq!(oil, expand_rle_u8(&fix.fluid.oil_rle));
    assert_eq!(gas, expand_rle_u8(&fix.fluid.gas_rle));
    assert_eq!(brine, expand_rle_u8(&fix.fluid.brine_rle));

    // Wire closure HC labels into the core parity harness (self IoU = 1).
    let mut hc = oil.clone();
    for (h, g) in hc.iter_mut().zip(gas.iter()) {
        *h = (*h).saturating_add(*g);
    }
    let report = parity::compare_volumes(&hc, &hc, &[0.0f32; 1], &[0.0f32; 1]);
    assert!((report.label_iou - 1.0).abs() < 1e-12);
    assert!((report.label_agreement - 1.0).abs() < 1e-12);
    assert!(report.passes_defaults());
}

#[test]
fn get_top_of_closure_matches_python_golden() {
    let fix = load_fixture();
    let input_i = expand_rle_i32(&fix.top_of_closure.input_rle);
    let input: Vec<f32> = input_i.iter().map(|&v| v as f32).collect();
    let got = get_top_of_closure(
        &input,
        [8, 8, 8],
        fix.top_of_closure.pad_up,
        fix.top_of_closure.pad_down,
    );
    let expected: Vec<f32> = expand_rle_i32(&fix.top_of_closure.output_rle)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn bbox_for_label_and_fault_matches_python_golden() {
    let fix = load_fixture();
    let labels = expand_rle_i32(&fix.filter_min5.input_rle);
    let mut fault = vec![0.0f32; labels.len()];
    // Reconstruct fault_throw: 1.0 where label==3
    for (f, &lab) in fault.iter_mut().zip(labels.iter()) {
        if lab == 3 {
            *f = 1.0;
        }
    }
    let got = bbox_for_label_and_fault(
        &labels,
        [8, 8, 8],
        fix.bbox.label_id,
        &fault,
        fix.bbox.fault_block_val,
        fix.bbox.fault_tol,
        fix.bbox.pad,
    );
    assert_eq!(got, fix.bbox.slices);
}

#[test]
fn flood_fill_heap_2d_matches_python_golden() {
    let fix = load_fixture();
    let got = flood_fill_heap_2d(&fix.flood_fill_2d.input, fix.flood_fill_2d.shape, 1.0e22);
    assert_eq!(got.len(), fix.flood_fill_2d.output.len());
    for (i, (g, e)) in got.iter().zip(fix.flood_fill_2d.output.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-9,
            "flood_fill[{i}] rust={g} python={e}"
        );
    }
}

#[test]
fn filtered_closure_labels_parity_near_identity() {
    let fix = load_fixture();
    let input = expand_rle_i32(&fix.filter_min5.input_rle);
    let filtered = filter_labels_by_min_voxels(&input, fix.filter_min5.min_voxels);
    // Cast to u8 for parity (labels are small ints).
    let a: Vec<u8> = filtered.iter().map(|&v| v as u8).collect();
    let b: Vec<u8> = expand_rle_i32(&fix.filter_min5.output_rle)
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let report = parity::compare_volumes(&a, &b, &[0.0f32; 1], &[0.0f32; 1]);
    assert!(report.passes_defaults(), "{report:?}");
    assert!((report.label_iou - 1.0).abs() < 1e-12);
}
