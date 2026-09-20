//! Real parity harness: label IoU / agreement and angle-stack MAE / max-abs.
//!
//! # Product locks
//! - Parity = **labels + angle stacks**, not bit-identical full seismic.
//! - Fixed-seed 8^3 fixtures live in `tests/fixtures/parity_cubes_8.json`
//!   (regenerate via `tests/fixtures/generate_parity_cubes.py`).
//!
//! # Documented tolerances (default)
//! | Metric | Threshold |
//! |--------|-----------|
//! | Label macro IoU | >= 0.99 |
//! | Label agreement | >= 0.99 |
//! | Angle-stack MAE | <= 1e-3 |
//! | Angle-stack max-abs | <= 5e-3 |

use serde::Deserialize;

/// Historical fixture tag; angle stacks are now fully deterministic (no RNG).
pub const GOLDEN_SEED: u64 = 0x5EED_CAFE;

/// Default acceptance thresholds for near-parity checks.
pub const LABEL_IOU_MIN: f64 = 0.99;
pub const LABEL_AGREEMENT_MIN: f64 = 0.99;
pub const ANGLE_MAE_MAX: f64 = 1e-3;
pub const ANGLE_MAX_ABS_MAX: f64 = 5e-3;

/// Unset / background label in fixtures (outside horizon intervals).
pub const LABEL_UNSET: u8 = 255;

/// Mean absolute error over equal-length float slices.
pub fn mean_absolute_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "MAE length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Maximum absolute difference over equal-length float slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "max-abs length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0_f64, f64::max)
}

/// Per-class IoU for integer labels. Empty union -> 1.0 (vacuously equal).
pub fn label_iou(pred: &[u8], truth: &[u8], class_id: u8) -> f64 {
    assert_eq!(pred.len(), truth.len(), "IoU length mismatch");
    let mut inter = 0usize;
    let mut union = 0usize;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let pm = *p == class_id;
        let tm = *t == class_id;
        if pm && tm {
            inter += 1;
        }
        if pm || tm {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// Macro-mean IoU over all classes present in either volume, excluding `skip`.
pub fn macro_label_iou(pred: &[u8], truth: &[u8], skip: u8) -> f64 {
    assert_eq!(pred.len(), truth.len());
    let mut present = [false; 256];
    for v in pred.iter().chain(truth.iter()) {
        present[*v as usize] = true;
    }
    present[skip as usize] = false;
    let mut sum = 0.0;
    let mut n = 0usize;
    for (cid, is_present) in present.iter().enumerate() {
        if *is_present {
            sum += label_iou(pred, truth, cid as u8);
            n += 1;
        }
    }
    if n == 0 {
        1.0
    } else {
        sum / n as f64
    }
}

/// Fraction of voxels with identical labels.
pub fn label_agreement(pred: &[u8], truth: &[u8]) -> f64 {
    assert_eq!(pred.len(), truth.len());
    if pred.is_empty() {
        return 1.0;
    }
    let matches = pred
        .iter()
        .zip(truth.iter())
        .filter(|(a, b)| a == b)
        .count();
    matches as f64 / pred.len() as f64
}

/// Combined check against documented tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityReport {
    pub label_iou: f64,
    pub label_agreement: f64,
    pub angle_mae: f64,
    pub angle_max_abs: f64,
}

impl ParityReport {
    pub fn passes_defaults(self) -> bool {
        self.label_iou >= LABEL_IOU_MIN
            && self.label_agreement >= LABEL_AGREEMENT_MIN
            && self.angle_mae <= ANGLE_MAE_MAX
            && self.angle_max_abs <= ANGLE_MAX_ABS_MAX
    }
}

/// Compare label + angle-stack volumes with default tolerances.
pub fn compare_volumes(
    labels_a: &[u8],
    labels_b: &[u8],
    angles_a: &[f32],
    angles_b: &[f32],
) -> ParityReport {
    ParityReport {
        label_iou: macro_label_iou(labels_a, labels_b, LABEL_UNSET),
        label_agreement: label_agreement(labels_a, labels_b),
        angle_mae: mean_absolute_error(angles_a, angles_b),
        angle_max_abs: max_abs_diff(angles_a, angles_b),
    }
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    labels: LabelBlock,
    angle_stack: AngleBlock,
}

#[derive(Debug, Deserialize)]
struct LabelBlock {
    reference_rle: Vec<[u8; 2]>,
    perturbed_rle: Vec<[u8; 2]>,
}

#[derive(Debug, Deserialize)]
struct AngleBlock {
    shape: [usize; 3],
    /// Documented as `0.1*i + 0.05*j + 0.02*k` (deterministic; no RNG).
    #[allow(dead_code)]
    #[serde(default)]
    formula: Option<String>,
    pert_delta: f32,
}

fn expand_rle(pairs: &[[u8; 2]]) -> Vec<u8> {
    let mut out = Vec::new();
    for [value, count] in pairs {
        out.extend(std::iter::repeat(*value).take(*count as usize));
    }
    out
}

fn synth_angle_stack(shape: [usize; 3], pert_delta: f32) -> (Vec<f32>, Vec<f32>) {
    let [ni, nj, nk] = shape;
    let mut reference = Vec::with_capacity(ni * nj * nk);
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                reference.push(0.1 * i as f32 + 0.05 * j as f32 + 0.02 * k as f32);
            }
        }
    }
    let perturbed: Vec<f32> = reference.iter().map(|v| v + pert_delta).collect();
    (reference, perturbed)
}

/// Load the checked-in 8^3 fixture from `tests/fixtures/parity_cubes_8.json`.
///
/// Labels are stored RLE-compressed; angle stacks are synthesized from the
/// documented deterministic formula (`0.1*i + 0.05*j + 0.02*k`).
pub fn load_parity_cubes_8() -> Result<(Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>), String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/parity_cubes_8.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let fix: FixtureFile =
        serde_json::from_str(&text).map_err(|e| format!("parse fixture: {e}"))?;
    let labels_ref = expand_rle(&fix.labels.reference_rle);
    let labels_pert = expand_rle(&fix.labels.perturbed_rle);
    let (angles_ref, angles_pert) =
        synth_angle_stack(fix.angle_stack.shape, fix.angle_stack.pert_delta);
    Ok((labels_ref, labels_pert, angles_ref, angles_pert))
}
