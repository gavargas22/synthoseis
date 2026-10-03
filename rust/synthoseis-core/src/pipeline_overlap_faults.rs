//! `data/fault_labels` for the overlapped writer
//! ([`crate::run_e2e_streaming_overlapped`]), on the depth-sample axis.
//!
//! Master f3720fb2 and earlier never wrote fault labels from the overlapped
//! writer (library path only: the CLI rejects `--faults` with `--overlap`).
//! This pass writes them after the angle-stack / label chunks have been
//! flushed, one fault tile per MDIO chunk footprint (memory: one tile), with
//! the same tile evaluation and salt mask as the streaming path
//! (`pipeline_stream_3`): `fault AND NOT salt`
//! ([`crate::salt::mask_fault_tile_salt`], after drag and the seabed clamp),
//! or unmasked with `fault_labels_through_salt`. It then checks the store
//! against [`crate::generate_fault_labels`].
//!
//! [`crate::FaultConfig::overlap_legacy_no_fault_labels`] skips it
//! (master f3720fb2 output).

use synthoseis_io::MdioStore;

use crate::pipeline::E2eConfig;
use crate::pipeline_stream::{fault_model, fault_tile_chunk, WorkingSetStats};
use crate::rock_physics::ElasticModel;

/// Write and verify the depth-axis `data/fault_labels` of an overlapped run
/// into `store` (no-op without faults or with the legacy switch).
pub(crate) fn write_overlap_fault_labels(
    store: &MdioStore,
    cfg: &E2eConfig,
    trends: &ElasticModel,
    shape: [usize; 3],
    chunks: [usize; 3],
    stats: &mut WorkingSetStats,
) -> Result<(), String> {
    if !cfg.faults.enabled() || cfg.faults.overlap_legacy_no_fault_labels {
        return Ok(());
    }
    let Some(model) = fault_model(cfg) else {
        return Ok(());
    };
    let salt = crate::salt::fault_label_salt(cfg, trends);
    let [ni, nj, nk] = shape;
    let [ci, cj, ck] = chunks;
    store
        .ensure_fault_labels_array()
        .map_err(|e| e.to_string())?;
    let mut chunk = Vec::new();
    for (i_chunk, i0) in (0..ni).step_by(ci).enumerate() {
        let i1 = (i0 + ci).min(ni);
        for (j_chunk, j0) in (0..nj).step_by(cj).enumerate() {
            let j1 = (j0 + cj).min(nj);
            let mut t = model.compute_tile(i0, i1, j0, j1);
            if let Some(s) = salt {
                crate::salt::mask_fault_tile_salt(&mut t, s);
            }
            stats.observe(t.lookup.capacity() * 4 + t.mask.capacity() * 2 + ci * cj * ck);
            for (k_chunk, k0) in (0..nk).step_by(ck).enumerate() {
                let k1 = (k0 + ck).min(nk);
                fault_tile_chunk(&t, k0, k1, &mut chunk);
                store
                    .write_fault_labels_chunk([i_chunk, j_chunk, k_chunk], &chunk)
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    if let Some(reference) = crate::generate_fault_labels(cfg) {
        let back = store.read_fault_labels_u8().map_err(|e| e.to_string())?;
        if back != reference {
            return Err("overlapped fault_labels diverged from the tile-wise reference".into());
        }
    }
    Ok(())
}
