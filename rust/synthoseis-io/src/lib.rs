//! MDIO-only I/O stubs for the Rust rewrite.
//!
//! # Why not crates.io `mdio`?
//! The `mdio` crate is an Ethernet PHY driver, unrelated to seismic MDIO
//! ([mdio.dev](https://mdio.dev)), which is a Zarr-based volume format.
//!
//! # Smoke store
//! This crate writes a **minimal Zarr v2-like** on-disk layout that captures the
//! *intent* of an MDIO working store: chunked array data plus JSON metadata
//! attributes (`dims`, `digi`, `seed`, `units`). Full MDIO / Python interop and
//! a production Zarr crate (`zarrs` or similar) are follow-ups.
//!
//! Layout created under `<root>/`:
//! ```text
//! .zgroup
//! .zattrs                 # digi, seed, units, dims
//! volume/
//!   .zarray               # shape, chunks, dtype, order
//!   0.0.0                 # raw little-endian f32 chunk
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("shape mismatch: expected {expected} samples, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, IoError>;

/// Metadata attrs stored alongside the chunked volume (MDIO-intent).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreMeta {
    pub dims: [String; 3],
    pub shape: [usize; 3],
    pub digi: f64,
    pub seed: u64,
    pub units: String,
}

impl Default for StoreMeta {
    fn default() -> Self {
        Self {
            dims: ["inline".into(), "crossline".into(), "time".into()],
            shape: [2, 2, 4],
            digi: 4.0,
            seed: 42,
            units: "ms".into(),
        }
    }
}

/// Handle to an on-disk MDIO-intent store root.
#[derive(Debug, Clone)]
pub struct MdioStore {
    root: PathBuf,
    meta: StoreMeta,
}

impl MdioStore {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn meta(&self) -> &StoreMeta {
        &self.meta
    }

    /// Create a new store directory with Zarr-v2-like group metadata.
    pub fn create(root: impl AsRef<Path>, meta: &StoreMeta) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("volume"))?;

        write_json(
            root.join(".zgroup"),
            &serde_json::json!({ "zarr_format": 2 }),
        )?;
        write_json(
            root.join(".zattrs"),
            &serde_json::json!({
                "dims": meta.dims,
                "digi": meta.digi,
                "seed": meta.seed,
                "units": meta.units,
                "mdio_intent": true,
                "note": "Minimal Zarr-v2-like layout; full MDIO/Python interop is a follow-up."
            }),
        )?;

        let chunks = meta.shape; // single chunk for smoke
        write_json(
            root.join("volume").join(".zarray"),
            &serde_json::json!({
                "zarr_format": 2,
                "shape": meta.shape,
                "chunks": chunks,
                "dtype": "<f4",
                "compressor": null,
                "fill_value": 0.0,
                "order": "C",
                "filters": null,
                "dimension_separator": "."
            }),
        )?;

        Ok(Self {
            root,
            meta: meta.clone(),
        })
    }

    /// Load `.zattrs` and confirm the store looks like our smoke layout.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let attrs: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join(".zattrs"))?)?;
        let dims_val = attrs
            .get("dims")
            .and_then(|d| d.as_array())
            .ok_or_else(|| {
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing dims in .zattrs",
                ))
            })?;
        let dims = [
            dims_val
                .first()
                .and_then(|v| v.as_str())
                .unwrap_or("inline")
                .to_string(),
            dims_val
                .get(1)
                .and_then(|v| v.as_str())
                .unwrap_or("crossline")
                .to_string(),
            dims_val
                .get(2)
                .and_then(|v| v.as_str())
                .unwrap_or("time")
                .to_string(),
        ];
        let zarray: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join("volume").join(".zarray"))?)?;
        let shape = [
            zarray["shape"][0].as_u64().unwrap_or(0) as usize,
            zarray["shape"][1].as_u64().unwrap_or(0) as usize,
            zarray["shape"][2].as_u64().unwrap_or(0) as usize,
        ];
        let meta = StoreMeta {
            dims,
            shape,
            digi: attrs.get("digi").and_then(|v| v.as_f64()).unwrap_or(0.0),
            seed: attrs.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
            units: attrs
                .get("units")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        };
        Ok(Self { root, meta })
    }
}

/// Deliverable write API stub (MDIO-intent only).
pub struct DeliverableWriter;

impl DeliverableWriter {
    /// Write a tiny contiguous f32 volume as a single Zarr chunk `0.0.0`.
    pub fn write_smoke_volume(store: &MdioStore, samples: &[f32]) -> Result<()> {
        let expected = store.meta.shape.iter().product::<usize>();
        if samples.len() != expected {
            return Err(IoError::ShapeMismatch {
                expected,
                got: samples.len(),
            });
        }
        let chunk_path = store.root.join("volume").join("0.0.0");
        let mut file = fs::File::create(&chunk_path)?;
        for v in samples {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}

fn write_json(path: PathBuf, value: &serde_json::Value) -> Result<()> {
    let mut file = fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn smoke_writes_on_disk_store_with_metadata() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("smoke.mdio");
        let meta = StoreMeta::default();
        let store = MdioStore::create(&root, &meta).expect("create");

        let n = meta.shape.iter().product::<usize>();
        let mut data = vec![0.0_f32; n];
        data[0] = 1.5;
        data[n - 1] = -2.25;
        DeliverableWriter::write_smoke_volume(&store, &data).expect("write");

        assert!(root.join(".zgroup").is_file());
        assert!(root.join(".zattrs").is_file());
        assert!(root.join("volume").join(".zarray").is_file());
        assert!(root.join("volume").join("0.0.0").is_file());

        let opened = MdioStore::open(&root).expect("open");
        assert_eq!(opened.meta().dims, meta.dims);
        assert_eq!(opened.meta().shape, meta.shape);
        assert_eq!(opened.meta().digi, meta.digi);
        assert_eq!(opened.meta().seed, meta.seed);
        assert_eq!(opened.meta().units, meta.units);

        let attrs: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join(".zattrs")).unwrap()).unwrap();
        assert_eq!(attrs["dims"][0], "inline");
        assert_eq!(attrs["dims"][1], "crossline");
        assert_eq!(attrs["dims"][2], "time");
        assert!(attrs["mdio_intent"].as_bool().unwrap_or(false));

        let chunk = fs::read(root.join("volume").join("0.0.0")).unwrap();
        assert_eq!(chunk.len(), n * 4);
    }
}
