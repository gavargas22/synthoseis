//! write_chunk / open / write_volume
use super::MdioStore;
use crate::zarr::*;
use crate::{
    err, CreateConfig, Dimension, PRIMARY_VARIABLE, Result,
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
        let zarray: Value =
            read_json(root.join("data").join(PRIMARY_VARIABLE).join(".zarray"))?;
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
}

pub struct DeliverableWriter;

impl DeliverableWriter {
    pub fn write_volume(store: &MdioStore, samples: &[f32]) -> Result<()> {
        store.write_volume(samples)
    }

    pub fn write_smoke_volume(store: &MdioStore, samples: &[f32]) -> Result<()> {
        Self::write_volume(store, samples)
    }
}
