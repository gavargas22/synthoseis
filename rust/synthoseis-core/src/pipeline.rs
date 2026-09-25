//! End-to-end single-worker tiny-cube pipeline.
//!
//! # Shape
//! MDIO create → geo (horizons/labels) → closures (relabel/filter) → RPM
//! (elastic props) → seismic (Zoeppritz + wavelet) → MDIO write (labels +
//! angle stack) → parity check (second deterministic pass).
//!
//! # Product locks
//! - Single local worker, CPU path only
//! - Parity = labels + angle stacks (not bit-identical full seismic)
//! - Tiny fixed dims (default 8³); Python `main.py` generator stays untouched

use std::path::{Path, PathBuf};

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::{
    enforce_nonnegative_thicknesses, eval_plane, fill_layer_labels, fit_plane_lsq,
};
use synthoseis_io::{CreateConfig, DeliverableWriter, Dimension, MdioStore};
use synthoseis_rpm::RpmExampleTrends;
use synthoseis_seismic::{apply_wavelet_traces, compute_rfc_volumes, ricker};

use crate::parity::{self, ParityReport};

/// Default tiny-cube edge length used by the e2e smoke path.
pub const TINY_DIM: usize = 8;

/// Default digi (ms) for the tiny-cube store.
pub const TINY_DIGI: f64 = 4.0;

/// Synthetic depth scale (m per sample index) for RPM trends.
pub(crate) const DEPTH_PER_SAMPLE: f64 = 100.0;

/// Configuration for the single-worker e2e pipeline.
#[derive(Debug, Clone)]
pub struct E2eConfig {
    pub seed: u64,
    pub inline_count: usize,
    pub crossline_count: usize,
    pub samples: usize,
    /// Optional MDIO root; when set, labels + angle stack are written.
    pub store_path: Option<PathBuf>,
    /// Optional MDIO / fused-generation chunk shape `[ci, cj, ck]`.
    ///
    /// When `None`, [`crate::pipeline_stream::resolve_chunk_shape`] picks a
    /// sub-volume default (never full-array when the grid allows). Chunk keys
    /// are strip-friendly for a later multi-worker partition.
    pub chunk_shape: Option<[usize; 3]>,
    /// Optional fault modelling (port of `datagenerator/Faults.py`).
    ///
    /// Default is disabled (`count == 0`): geology, labels and angle stacks
    /// are bit-identical to the pre-fault pipeline.
    pub faults: FaultConfig,
}

/// Fault settings for the e2e pipeline (random-mode draw, seeded from
/// [`E2eConfig::seed`]). See `docs/faults-port.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultConfig {
    /// Number of faults to draw (`0` = faulting disabled).
    pub count: usize,
    /// Minimum throw in samples (Python `low_fault_throw / infill_factor`).
    pub throw_min: f64,
    /// Maximum throw in samples. Default `29.0` keeps below the hockey-stick
    /// threshold (`0.85 * 35`), whose drag zone is deferred in the port.
    pub throw_max: f64,
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            count: 0,
            throw_min: 5.0,
            throw_max: 29.0,
        }
    }
}

impl FaultConfig {
    /// `count` faults with default throw range.
    pub fn with_count(count: usize) -> Self {
        Self {
            count,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.count > 0
    }
}

impl Default for E2eConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            inline_count: TINY_DIM,
            crossline_count: TINY_DIM,
            samples: TINY_DIM,
            store_path: None,
            chunk_shape: None,
            faults: FaultConfig::default(),
        }
    }
}

impl E2eConfig {
    pub fn tiny(seed: u64) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        [self.inline_count, self.crossline_count, self.samples]
    }
}

/// Volumes produced by one deterministic pipeline pass.
#[derive(Debug, Clone)]
pub struct E2eVolumes {
    pub labels: Vec<u8>,
    pub angle_stack: Vec<f32>,
    pub shape: [usize; 3],
}

/// Full e2e report: volumes, parity vs second pass, optional store path.
#[derive(Debug, Clone)]
pub struct E2eReport {
    pub volumes: E2eVolumes,
    pub parity: ParityReport,
    pub store_path: Option<PathBuf>,
    pub status: &'static str,
}

/// Generate labels + angle stack for a tiny cube (deterministic for fixed seed).
pub fn generate_tiny_cube(cfg: &E2eConfig) -> E2eVolumes {
    let [ni, nj, nk] = cfg.shape();
    assert!(nk >= 2, "need at least 2 samples for reflectivity");

    // --- geo: two dipping planes from seed-derived control points ---
    let seed_f = cfg.seed as f64;
    let a0 = 0.05 + (seed_f % 7.0) * 0.01;
    let b0 = 0.03 + ((seed_f / 3.0) % 5.0) * 0.01;
    let c0 = 0.5;
    let a1 = a0 * 0.5;
    let b1 = b0 * 0.5;
    let c1 = (nk as f64) * 0.55;

    // Fit planes from three points each (exercises fit_plane_lsq), then eval.
    let pts0 = [[0.0, 0.0, c0], [1.0, 0.0, a0 + c0], [0.0, 1.0, b0 + c0]];
    let pts1 = [[0.0, 0.0, c1], [1.0, 0.0, a1 + c1], [0.0, 1.0, b1 + c1]];
    let [fa, fb, fc] = fit_plane_lsq(&pts0);
    let [ga, gb, gc] = fit_plane_lsq(&pts1);
    let z0 = eval_plane(ni, nj, fa, fb, fc);
    let z1 = eval_plane(ni, nj, ga, gb, gc);
    // Bottom horizon: flat near base.
    let z2 = vec![(nk as f64) - 0.5; ni * nj];

    let nh = 3usize;
    let mut maps = vec![0.0f64; ni * nj * nh];
    for n in 0..(ni * nj) {
        maps[n * nh] = z0[n];
        maps[n * nh + 1] = z1[n];
        maps[n * nh + 2] = z2[n];
    }
    enforce_nonnegative_thicknesses(&mut maps, [ni, nj, nh]);
    let mut labels = fill_layer_labels(&maps, [ni, nj, nh], nk);

    // --- closures: treat unset as background, relabel, drop tiny bodies ---
    let as_i32: Vec<i32> = labels
        .iter()
        .map(|&v| if v == 255 { 0 } else { (v as i32) + 1 })
        .collect();
    let (_vals, relabeled) = relabel_consecutive(&as_i32);
    let filtered = filter_labels_by_min_voxels(&relabeled, 1);
    for (dst, &src) in labels.iter_mut().zip(filtered.iter()) {
        *dst = if src == 0 {
            255
        } else {
            (src - 1).clamp(0, 254) as u8
        };
    }

    // --- faults (optional; no-op when cfg.faults.count == 0) ---
    crate::pipeline_stream::apply_faults_to_labels(cfg, &maps, nh, &mut labels);

    // --- RPM: elastic props from depth trends by layer class ---
    let mut vp = vec![0.0f32; ni * nj * nk];
    let mut vs = vec![0.0f32; ni * nj * nk];
    let mut rho = vec![0.0f32; ni * nj * nk];
    let depths: Vec<f64> = (0..nk).map(|k| k as f64 * DEPTH_PER_SAMPLE).collect();
    let shale_vp = RpmExampleTrends::shale_vp(&depths);
    let shale_vs = RpmExampleTrends::shale_vs(&depths);
    let shale_rho = RpmExampleTrends::shale_rho(&depths);
    let brine_vp = RpmExampleTrends::brine_sand_vp(&depths);
    let brine_vs = RpmExampleTrends::brine_sand_vs(&depths);
    let brine_rho = RpmExampleTrends::brine_sand_rho(&depths);
    let oil_vp = RpmExampleTrends::oil_sand_vp(&depths);
    let oil_vs = RpmExampleTrends::oil_sand_vs(&depths);
    let oil_rho = RpmExampleTrends::oil_sand_rho(&depths);

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let idx = (i * nj + j) * nk + k;
                let lab = labels[idx];
                let (v_p, v_s, r) = match lab {
                    0 => (shale_vp[k], shale_vs[k], shale_rho[k]),
                    1 => (brine_vp[k], brine_vs[k], brine_rho[k]),
                    _ => (oil_vp[k], oil_vs[k], oil_rho[k]),
                };
                vp[idx] = v_p as f32;
                vs[idx] = v_s as f32;
                rho[idx] = r as f32;
            }
        }
    }

    // --- seismic: single mid-angle RFC + Ricker wavelet → angle stack ---
    let angles = [15.0_f64];
    let rfc = compute_rfc_volumes(&vp, &vs, &rho, [ni, nj, nk], &angles);
    // rfc shape: (1, ni, nj, nk-1) — pad last sample with 0 to match nk.
    let mut angle_cube = vec![0.0f32; ni * nj * nk];
    let zm1 = nk - 1;
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..zm1 {
                let src = ((0 * ni + i) * nj + j) * zm1 + k;
                let dst = (i * nj + j) * nk + k;
                angle_cube[dst] = rfc[src];
            }
        }
    }
    // Short wavelet: higher frequency keeps support reasonable for tiny nk.
    let wavelet = ricker(40.0, TINY_DIGI, 1);
    let angle_stack = apply_wavelet_traces(&angle_cube, [ni, nj, nk], &wavelet);

    E2eVolumes {
        labels,
        angle_stack,
        shape: [ni, nj, nk],
    }
}

/// Write labels + angle stack into an MDIO store (create or overwrite path).
pub fn write_e2e_mdio(path: &Path, cfg: &E2eConfig, volumes: &E2eVolumes) -> Result<(), String> {
    let [ni, nj, nk] = volumes.shape;
    let create = CreateConfig {
        dimensions: [
            Dimension::sized("inline", ni),
            Dimension::sized("crossline", nj),
            Dimension::sized("sample", nk),
        ],
        chunks: Some(crate::pipeline_stream::resolve_chunk_shape(cfg)),
        digi: TINY_DIGI,
        seed: cfg.seed,
        units: "ms".into(),
        name: "synthoseis-e2e".into(),
    };
    let store = MdioStore::create_empty(path, &create).map_err(|e| e.to_string())?;
    DeliverableWriter::write_volume(&store, &volumes.angle_stack).map_err(|e| e.to_string())?;
    DeliverableWriter::write_labels(&store, &volumes.labels).map_err(|e| e.to_string())?;
    if let Some(mask) = crate::pipeline_stream::generate_fault_labels(cfg) {
        store
            .write_fault_labels_u8(&mask)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Run the full e2e path: generate → (optional MDIO write) → second-pass parity.
pub fn run_e2e(cfg: &E2eConfig) -> Result<E2eReport, String> {
    let volumes = generate_tiny_cube(cfg);
    let second = generate_tiny_cube(cfg);
    let parity = parity::compare_volumes(
        &volumes.labels,
        &second.labels,
        &volumes.angle_stack,
        &second.angle_stack,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "e2e self-parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    let mut store_path = None;
    if let Some(ref path) = cfg.store_path {
        write_e2e_mdio(path, cfg, &volumes)?;
        // Round-trip check from MDIO.
        let opened = MdioStore::open(path).map_err(|e| e.to_string())?;
        let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
        let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
        let mdio_parity = parity::compare_volumes(
            &volumes.labels,
            &back_labels,
            &volumes.angle_stack,
            &back_angles,
        );
        if !mdio_parity.passes_defaults() {
            return Err(format!(
                "e2e MDIO round-trip parity failed: {mdio_parity:?}"
            ));
        }
        store_path = Some(path.clone());
    }

    Ok(E2eReport {
        volumes,
        parity,
        store_path,
        status: "ok-e2e",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn e2e_self_parity_exact() {
        let cfg = E2eConfig::tiny(42);
        let a = generate_tiny_cube(&cfg);
        let b = generate_tiny_cube(&cfg);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.angle_stack, b.angle_stack);
        let report = parity::compare_volumes(&a.labels, &b.labels, &a.angle_stack, &b.angle_stack);
        assert!(report.passes_defaults());
        assert!((report.label_iou - 1.0).abs() < 1e-12);
        assert!(report.angle_mae == 0.0);
    }

    #[test]
    fn e2e_mdio_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("e2e.mdio");
        let cfg = E2eConfig {
            faults: Default::default(),
            store_path: Some(path.clone()),
            chunk_shape: None,
            ..E2eConfig::tiny(7)
        };
        let report = run_e2e(&cfg).expect("e2e");
        assert_eq!(report.status, "ok-e2e");
        assert!(path
            .join("data")
            .join("chunked_012")
            .join(".zarray")
            .is_file());
        assert!(path.join("data").join("labels").join(".zarray").is_file());
        assert_eq!(report.volumes.shape, [8, 8, 8]);
        assert_eq!(report.volumes.labels.len(), 512);
        assert_eq!(report.volumes.angle_stack.len(), 512);
    }

    #[test]
    fn e2e_seed_changes_output() {
        let a = generate_tiny_cube(&E2eConfig::tiny(1));
        let b = generate_tiny_cube(&E2eConfig::tiny(99));
        // Different seeds should change plane coefficients → labels or angles.
        assert!(a.labels != b.labels || a.angle_stack != b.angle_stack);
    }
}
