//! read_volume / read_live_mask
use super::MdioStore;
use crate::zarr::*;
use crate::{err, Result, FAULT_LABELS_VARIABLE, LABELS_VARIABLE, PRIMARY_VARIABLE};
use std::fs;

impl MdioStore {
    pub fn read_volume(&self) -> Result<Vec<f32>> {
        let shape = self.shape();
        let chunks = self.config.chunks_or_shape();
        let mut out = vec![0f32; shape[0] * shape[1] * shape[2]];
        let array_dir = self.root.join("data").join(PRIMARY_VARIABLE);

        let n0 = ceildiv(shape[0], chunks[0]);
        let n1 = ceildiv(shape[1], chunks[1]);
        let n2 = ceildiv(shape[2], chunks[2]);
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let path = array_dir.join(chunk_key(&[i0, i1, i2]));
                    if !path.is_file() {
                        continue;
                    }
                    let bytes = fs::read(&path)?;
                    let start = [i0 * chunks[0], i1 * chunks[1], i2 * chunks[2]];
                    let end = [
                        (start[0] + chunks[0]).min(shape[0]),
                        (start[1] + chunks[1]).min(shape[1]),
                        (start[2] + chunks[2]).min(shape[2]),
                    ];
                    let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                    let expected = cshape[0] * cshape[1] * cshape[2] * 4;
                    if bytes.len() != expected {
                        return Err(err(format!(
                            "chunk {:?} size {}: expected {expected}",
                            [i0, i1, i2],
                            bytes.len()
                        )));
                    }
                    let mut bi = 0;
                    for x0 in start[0]..end[0] {
                        for x1 in start[1]..end[1] {
                            for x2 in start[2]..end[2] {
                                let mut le = [0u8; 4];
                                le.copy_from_slice(&bytes[bi..bi + 4]);
                                bi += 4;
                                let li = (x0 * shape[1] + x1) * shape[2] + x2;
                                out[li] = f32::from_le_bytes(le);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn read_live_mask(&self) -> Result<Vec<u8>> {
        let live_shape = self.config.spatial_shape();
        let path = self
            .root
            .join("metadata")
            .join("live_mask")
            .join(chunk_key(&[0, 0]));
        let mut buf = fs::read(path)?;
        let n = live_shape[0] * live_shape[1];
        if buf.len() != n {
            return Err(err(format!("live_mask length {}: expected {n}", buf.len())));
        }
        for b in &mut buf {
            *b = if *b == 0 { 0 } else { 1 };
        }
        Ok(buf)
    }

    /// Read the uint8 label volume from `data/labels` (same shape as primary).
    pub fn read_labels_u8(&self) -> Result<Vec<u8>> {
        self.read_u8_variable(LABELS_VARIABLE, 255)
    }

    /// Read the binary fault-label volume from `data/fault_labels`.
    pub fn read_fault_labels_u8(&self) -> Result<Vec<u8>> {
        self.read_u8_variable(FAULT_LABELS_VARIABLE, 0)
    }

    fn read_u8_variable(&self, name: &str, fill: u8) -> Result<Vec<u8>> {
        let shape = self.shape();
        let chunks = self.config.chunks_or_shape();
        let mut out = vec![fill; shape[0] * shape[1] * shape[2]];
        let array_dir = self.root.join("data").join(name);
        if !array_dir.join(".zarray").is_file() {
            return Err(err(format!("{name} array not present in store")));
        }

        let n0 = ceildiv(shape[0], chunks[0]);
        let n1 = ceildiv(shape[1], chunks[1]);
        let n2 = ceildiv(shape[2], chunks[2]);
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let path = array_dir.join(chunk_key(&[i0, i1, i2]));
                    if !path.is_file() {
                        continue;
                    }
                    let bytes = fs::read(&path)?;
                    let start = [i0 * chunks[0], i1 * chunks[1], i2 * chunks[2]];
                    let end = [
                        (start[0] + chunks[0]).min(shape[0]),
                        (start[1] + chunks[1]).min(shape[1]),
                        (start[2] + chunks[2]).min(shape[2]),
                    ];
                    let cshape = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                    let expected = cshape[0] * cshape[1] * cshape[2];
                    if bytes.len() != expected {
                        return Err(err(format!(
                            "{name} chunk {:?} size {}: expected {expected}",
                            [i0, i1, i2],
                            bytes.len()
                        )));
                    }
                    let mut bi = 0;
                    for x0 in start[0]..end[0] {
                        for x1 in start[1]..end[1] {
                            for x2 in start[2]..end[2] {
                                let li = (x0 * shape[1] + x1) * shape[2] + x2;
                                out[li] = bytes[bi];
                                bi += 1;
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}
