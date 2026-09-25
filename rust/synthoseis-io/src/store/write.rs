//! write_chunk / open / write_volume
use super::MdioStore;
use crate::zarr::*;
use crate::{
    err, CreateConfig, Dimension, Result, FAULT_LABELS_VARIABLE, LABELS_VARIABLE, PRIMARY_VARIABLE,
};
use serde_json::Value;
use std::path::Path;

impl MdioStore {
    pub fn write_chunk(&self, chunk_indices: [usize; 3], samples: &[f32]) -> Result<()> {
        let shape = self.shape();
        let chunks = self.config.chunks_or_shape();
        let start = [
            chunk_indices[0] * chunks[0],
            chunk_indices[1] * chunks[1],
            chunk_indices[2] * chunks[2],
        ];
        if start[0] >= shape[0] || start[1] >= shape[1] || start[2] >= shape[2] {
            return Err(err(format!(
                "chunk indices {:?} out of range",
                chunk_indices
            )));
        }
        let end = [
            (start[0] + chunks[0]).min(shape[0]),
            (start[1] + chunks[1]).min(shape[1]),
            (start[2] + chunks[2]).min(shape[2]),
        ];
        let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
        let expected = cshape[0] * cshape[1] * cshape[2];
        if samples.len() != expected {
            return Err(err(format!(
                "chunk {:?} expects {expected} samples, got {}",
                chunk_indices,
                samples.len()
            )));
        }
        let bytes = f32_slice_to_le_bytes(samples);
        write_chunk_bytes(
            &self.root.join("data").join(PRIMARY_VARIABLE),
            &chunk_key(&chunk_indices),
            &bytes,
        )
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let attrs: Value = read_json(root.join(".zattrs"))?;
        if attrs.get("synthoseis_mdio").and_then(|v| v.as_bool()) != Some(true)
            && attrs.get("api_version").is_none()
        {
            return Err(err("path does not look like an MDIO / synthoseis store"));
        }
        let dims = parse_dimensions(&attrs)?;
        let zarray: Value = read_json(root.join("data").join(PRIMARY_VARIABLE).join(".zarray"))?;
        let shape = parse_usize3(&zarray["shape"])?;
        let chunks = parse_usize3(&zarray["chunks"])?;
        let mut dimensions = dims;
        for i in 0..3 {
            if dimensions[i].size() != shape[i] {
                dimensions[i] = Dimension::sized(&dimensions[i].name, shape[i]);
            }
        }
        let config = CreateConfig {
            dimensions,
            chunks: Some(chunks),
            digi: attrs.get("digi").and_then(|v| v.as_f64()).unwrap_or(0.0),
            seed: attrs.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
            units: attrs
                .get("units")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            name: attrs
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("synthoseis")
                .to_string(),
        };
        Ok(Self { root, config })
    }

    pub fn write_volume(&self, samples: &[f32]) -> Result<()> {
        let shape = self.shape();
        let expected = shape[0] * shape[1] * shape[2];
        if samples.len() != expected {
            return Err(err(format!(
                "shape mismatch: expected {expected} samples, got {}",
                samples.len()
            )));
        }
        let chunks = self.config.chunks_or_shape();
        let array_dir = self.root.join("data").join(PRIMARY_VARIABLE);

        let n0 = ceildiv(shape[0], chunks[0]);
        let n1 = ceildiv(shape[1], chunks[1]);
        let n2 = ceildiv(shape[2], chunks[2]);
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let start = [i0 * chunks[0], i1 * chunks[1], i2 * chunks[2]];
                    let end = [
                        (start[0] + chunks[0]).min(shape[0]),
                        (start[1] + chunks[1]).min(shape[1]),
                        (start[2] + chunks[2]).min(shape[2]),
                    ];
                    let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                    let mut buf = vec![0f32; cshape[0] * cshape[1] * cshape[2]];
                    let mut bi = 0;
                    for x0 in start[0]..end[0] {
                        for x1 in start[1]..end[1] {
                            for x2 in start[2]..end[2] {
                                let li = (x0 * shape[1] + x1) * shape[2] + x2;
                                buf[bi] = samples[li];
                                bi += 1;
                            }
                        }
                    }
                    let bytes = f32_slice_to_le_bytes(&buf);
                    write_chunk_bytes(&array_dir, &chunk_key(&[i0, i1, i2]), &bytes)?;
                }
            }
        }

        let live_shape = self.config.spatial_shape();
        let live_n = live_shape[0] * live_shape[1];
        write_chunk_bytes(
            &self.root.join("metadata").join("live_mask"),
            &chunk_key(&[0, 0]),
            &vec![1u8; live_n],
        )?;
        update_root_attr_u64(&self.root, "trace_count", live_n as u64)?;
        update_stats(&self.root, samples)?;
        // Refresh consolidated metadata after attr / live_mask updates.
        write_consolidated_metadata(&self.root)?;
        Ok(())
    }

    /// Write a uint8 label volume under `data/labels` (same shape as primary).
    ///
    /// Creates the Zarr array on first write. Does not replace `chunked_012`.
    pub fn write_labels_u8(&self, labels: &[u8]) -> Result<()> {
        self.write_u8_variable(&LABELS, labels)
    }

    /// Ensure `data/labels` Zarr array exists with the store's chunk shape.
    pub fn ensure_labels_array(&self) -> Result<()> {
        self.ensure_u8_variable(&LABELS)
    }

    /// Write one uint8 labels chunk (same indexing as [`Self::write_chunk`]).
    pub fn write_labels_chunk(&self, chunk_indices: [usize; 3], labels: &[u8]) -> Result<()> {
        self.write_u8_variable_chunk(&LABELS, chunk_indices, labels)
    }

    /// Write the binary fault-label volume under `data/fault_labels`.
    pub fn write_fault_labels_u8(&self, labels: &[u8]) -> Result<()> {
        self.write_u8_variable(&FAULT_LABELS, labels)
    }

    /// Ensure `data/fault_labels` exists with the store's chunk shape.
    pub fn ensure_fault_labels_array(&self) -> Result<()> {
        self.ensure_u8_variable(&FAULT_LABELS)
    }

    /// Write one `data/fault_labels` chunk (same indexing as [`Self::write_chunk`]).
    pub fn write_fault_labels_chunk(&self, chunk_indices: [usize; 3], labels: &[u8]) -> Result<()> {
        self.write_u8_variable_chunk(&FAULT_LABELS, chunk_indices, labels)
    }

    fn ensure_u8_variable(&self, var: &U8Variable) -> Result<()> {
        let shape = self.shape();
        let chunks = self.config.chunks_or_shape();
        let array_dir = self.root.join("data").join(var.name);
        std::fs::create_dir_all(&array_dir)?;
        write_zarray(
            array_dir.join(".zarray"),
            &shape,
            &chunks,
            "|u1",
            serde_json::json!(var.fill),
        )?;
        write_json(
            array_dir.join(".zattrs"),
            &serde_json::json!({
                "long_name": var.long_name,
                "synthoseis_deliverable": var.deliverable
            }),
        )?;
        Ok(())
    }

    fn write_u8_variable(&self, var: &U8Variable, labels: &[u8]) -> Result<()> {
        let shape = self.shape();
        let expected = shape[0] * shape[1] * shape[2];
        if labels.len() != expected {
            return Err(err(format!(
                "{} shape mismatch: expected {expected}, got {}",
                var.name,
                labels.len()
            )));
        }
        let chunks = self.config.chunks_or_shape();
        self.ensure_u8_variable(var)?;
        let array_dir = self.root.join("data").join(var.name);

        let n0 = ceildiv(shape[0], chunks[0]);
        let n1 = ceildiv(shape[1], chunks[1]);
        let n2 = ceildiv(shape[2], chunks[2]);
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let start = [i0 * chunks[0], i1 * chunks[1], i2 * chunks[2]];
                    let end = [
                        (start[0] + chunks[0]).min(shape[0]),
                        (start[1] + chunks[1]).min(shape[1]),
                        (start[2] + chunks[2]).min(shape[2]),
                    ];
                    let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                    let mut buf = vec![0u8; cshape[0] * cshape[1] * cshape[2]];
                    let mut bi = 0;
                    for x0 in start[0]..end[0] {
                        for x1 in start[1]..end[1] {
                            for x2 in start[2]..end[2] {
                                let li = (x0 * shape[1] + x1) * shape[2] + x2;
                                buf[bi] = labels[li];
                                bi += 1;
                            }
                        }
                    }
                    write_chunk_bytes(&array_dir, &chunk_key(&[i0, i1, i2]), &buf)?;
                }
            }
        }
        write_consolidated_metadata(&self.root)?;
        Ok(())
    }

    fn write_u8_variable_chunk(
        &self,
        var: &U8Variable,
        chunk_indices: [usize; 3],
        labels: &[u8],
    ) -> Result<()> {
        self.ensure_u8_variable(var)?;
        let shape = self.shape();
        let chunks = self.config.chunks_or_shape();
        let start = [
            chunk_indices[0] * chunks[0],
            chunk_indices[1] * chunks[1],
            chunk_indices[2] * chunks[2],
        ];
        if start[0] >= shape[0] || start[1] >= shape[1] || start[2] >= shape[2] {
            return Err(err(format!(
                "{} chunk indices {:?} out of range",
                var.name, chunk_indices
            )));
        }
        let end = [
            (start[0] + chunks[0]).min(shape[0]),
            (start[1] + chunks[1]).min(shape[1]),
            (start[2] + chunks[2]).min(shape[2]),
        ];
        let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
        let expected = cshape[0] * cshape[1] * cshape[2];
        if labels.len() != expected {
            return Err(err(format!(
                "{} chunk {:?} expects {expected} samples, got {}",
                var.name,
                chunk_indices,
                labels.len()
            )));
        }
        write_chunk_bytes(
            &self.root.join("data").join(var.name),
            &chunk_key(&chunk_indices),
            labels,
        )
    }

    /// Mark all traces live and refresh root stats after chunked primary writes.
    ///
    /// Call once after streaming [`Self::write_chunk`] / [`Self::write_labels_chunk`].
    /// `samples` may be a running concatenation of written chunk samples (order
    /// does not matter for min/max/mean/std/rms).
    pub fn finalize_after_chunked_write(&self, samples: &[f32]) -> Result<()> {
        let live_shape = self.config.spatial_shape();
        let live_n = live_shape[0] * live_shape[1];
        write_chunk_bytes(
            &self.root.join("metadata").join("live_mask"),
            &chunk_key(&[0, 0]),
            &vec![1u8; live_n],
        )?;
        update_root_attr_u64(&self.root, "trace_count", live_n as u64)?;
        if !samples.is_empty() {
            update_stats(&self.root, samples)?;
        }
        write_consolidated_metadata(&self.root)?;
        Ok(())
    }
}

/// A uint8 deliverable array under `data/<name>`.
struct U8Variable {
    name: &'static str,
    long_name: &'static str,
    deliverable: &'static str,
    fill: u8,
}

const LABELS: U8Variable = U8Variable {
    name: LABELS_VARIABLE,
    long_name: "layer_labels",
    deliverable: "labels",
    fill: 255,
};

const FAULT_LABELS: U8Variable = U8Variable {
    name: FAULT_LABELS_VARIABLE,
    long_name: "fault_labels",
    deliverable: "fault_labels",
    fill: 0,
};

pub struct DeliverableWriter;

impl DeliverableWriter {
    pub fn write_volume(store: &MdioStore, samples: &[f32]) -> Result<()> {
        store.write_volume(samples)
    }

    pub fn write_smoke_volume(store: &MdioStore, samples: &[f32]) -> Result<()> {
        Self::write_volume(store, samples)
    }

    pub fn write_labels(store: &MdioStore, labels: &[u8]) -> Result<()> {
        store.write_labels_u8(labels)
    }
}
