use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use synthoseis_closures::{filter_labels_by_min_voxels, relabel_consecutive};
use synthoseis_geo::{
    enforce_nonnegative_thicknesses, eval_plane, fill_layer_labels, fit_plane_lsq,
};
use synthoseis_io::{CreateConfig, Dimension, MdioStore};
use synthoseis_rpm::RpmExampleTrends;
use synthoseis_seismic::ricker;

use crate::parity;
use crate::pipeline::{
    E2eConfig, E2eReport, E2eVolumes, DEPTH_PER_SAMPLE, TINY_DIGI,
};

/// Peak temporary buffer bytes during fused generation / streaming write.
#[derive(Debug, Clone, Default)]
pub struct WorkingSetStats {
    pub peak_temp_bytes: usize,
    pub chunk_shape: [usize; 3],
    pub volume_shape: [usize; 3],
    pub tiles_processed: usize,
}

impl WorkingSetStats {
    pub(crate) fn observe(&mut self, bytes: usize) {
        self.peak_temp_bytes = self.peak_temp_bytes.max(bytes);
    }

    /// Peak temps stay within a slack multiple of one chunk of scratch bytes.
    pub fn is_bounded_by_chunk(&self, slack: usize) -> bool {
        let [ci, cj, ck] = self.chunk_shape;
        let tile = ci.saturating_mul(cj).saturating_mul(ck).max(1);
        let budget = tile.saturating_mul(32).saturating_mul(slack.max(1));
        let nk = self.volume_shape[2];
        let floor = 9 * nk * 8 + 4096;
        self.peak_temp_bytes <= budget.max(floor)
    }
}

/// Call counter for [`generate_labels`] — used by geometry-once tests.
pub static GENERATE_LABELS_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Resolve MDIO / generation chunk shape.
///
/// Prefer explicit `cfg.chunk_shape`. Otherwise pick a proper **sub-volume**
/// tile so e2e never writes `chunks == [ni,nj,nk]` when the grid allows.
pub fn resolve_chunk_shape(cfg: &E2eConfig) -> [usize; 3] {
    let [ni, nj, nk] = cfg.shape();
    if let Some(c) = cfg.chunk_shape {
        return [
            c[0].clamp(1, ni.max(1)),
            c[1].clamp(1, nj.max(1)),
            c[2].clamp(1, nk.max(1)),
        ];
    }
    default_subvolume_chunks([ni, nj, nk])
}

/// Default sub-volume chunks: `min(8, max(1, n/2))` spatially × full samples.
pub fn default_subvolume_chunks(shape: [usize; 3]) -> [usize; 3] {
    let [ni, nj, nk] = shape;
    let ci = if ni <= 1 {
        1
    } else {
        (ni / 2).clamp(1, 8).min(ni)
    };
    let cj = if nj <= 1 {
        1
    } else {
        (nj / 2).clamp(1, 8).min(nj)
    };
    [ci, cj, nk.max(1)]
}

/// Label generation — bit-identical to the geo+closures section of
/// [`crate::pipeline::generate_tiny_cube`].
pub fn generate_labels(cfg: &E2eConfig) -> (Vec<u8>, [usize; 3]) {
    GENERATE_LABELS_CALLS.fetch_add(1, Ordering::SeqCst);
    let [ni, nj, nk] = cfg.shape();
    assert!(nk >= 2, "need at least 2 samples for reflectivity");

    let seed_f = cfg.seed as f64;
    let a0 = 0.05 + (seed_f % 7.0) * 0.01;
    let b0 = 0.03 + ((seed_f / 3.0) % 5.0) * 0.01;
    let c0 = 0.5;
    let a1 = a0 * 0.5;
    let b1 = b0 * 0.5;
    let c1 = (nk as f64) * 0.55;

    let pts0 = [
        [0.0, 0.0, c0],
        [1.0, 0.0, a0 + c0],
        [0.0, 1.0, b0 + c0],
    ];
    let pts1 = [
        [0.0, 0.0, c1],
        [1.0, 0.0, a1 + c1],
        [0.0, 1.0, b1 + c1],
    ];
    let [fa, fb, fc] = fit_plane_lsq(&pts0);
    let [ga, gb, gc] = fit_plane_lsq(&pts1);
    let z0 = eval_plane(ni, nj, fa, fb, fc);
    let z1 = eval_plane(ni, nj, ga, gb, gc);
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
    (labels, [ni, nj, nk])
}

pub(crate) fn depth_trends(nk: usize) -> [Vec<f64>; 9] {
    let depths: Vec<f64> = (0..nk).map(|k| k as f64 * DEPTH_PER_SAMPLE).collect();
    [
        RpmExampleTrends::shale_vp(&depths),
        RpmExampleTrends::shale_vs(&depths),
        RpmExampleTrends::shale_rho(&depths),
        RpmExampleTrends::brine_sand_vp(&depths),
        RpmExampleTrends::brine_sand_vs(&depths),
        RpmExampleTrends::brine_sand_rho(&depths),
        RpmExampleTrends::oil_sand_vp(&depths),
        RpmExampleTrends::oil_sand_vs(&depths),
        RpmExampleTrends::oil_sand_rho(&depths),
    ]
}
