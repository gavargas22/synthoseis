//! Honest MDIO working-store I/O (Zarr v2 hierarchy).
//!
//! There is no seismic MDIO crate on crates.io (`mdio` is Ethernet PHY).
//! MDIO ([mdio.dev](https://mdio.dev) / mdio-python) is Zarr-backed.
//! This crate writes **Zarr v2** matching mdio-python `create_empty`:
//! `metadata/` + `data/`, `live_mask`, and `chunked_012` (uncompressed LE f32).
//! We emit Zarr v2 JSON + raw chunks directly (`zarrs` is V3-first / high MSRV).
//!
//! Interop: Python `mdio` (multidimio) opens this hierarchy via `MDIOReader` (0.9.x).
//! We write consolidated `.zmetadata` and a stub `chunked_012_trace_headers` array so
//! open succeeds. Known gaps: no Blosc/ZFP, headers are stub (not SEG-Y-faithful),
//! no cloud backends. mdio 1.x `open_mdio` (xarray-flat) is a later slice.


mod zarr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PRIMARY_VARIABLE: &str = "chunked_012";

/// Declared API version attribute (synthoseis Rust MDIO writer).
pub const API_VERSION: &str = "0.1.0-synthoseis-rust";

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, IoError>;

fn err(msg: impl Into<String>) -> IoError {
    IoError::Msg(msg.into())
}

/// One grid axis: name + coordinate vector (mdio-python `Dimension.to_dict`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dimension {
    pub name: String,
    pub coords: Vec<f64>,
}

impl Dimension {
    pub fn sized(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            coords: (0..size).map(|i| i as f64).collect(),
        }
    }

    pub fn size(&self) -> usize {
        self.coords.len()
    }
}

/// Configuration for [`MdioStore::create_empty`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreateConfig {
    /// Three dimensions, last is the sample/time axis.
    pub dimensions: [Dimension; 3],
    /// Chunk shape for `chunked_012` (C-order). Defaults to full array if `None`.
    pub chunks: Option<[usize; 3]>,
    /// Sample interval / digi (stored as root attr for synthoseis).
    pub digi: f64,
    pub seed: u64,
    pub units: String,
    /// Dataset display name (root attr).
    pub name: String,
}

impl CreateConfig {
    pub fn shape(&self) -> [usize; 3] {
        [
            self.dimensions[0].size(),
            self.dimensions[1].size(),
            self.dimensions[2].size(),
        ]
    }

    pub fn chunks_or_shape(&self) -> [usize; 3] {
        self.chunks.unwrap_or_else(|| self.shape())
    }

    pub fn spatial_shape(&self) -> [usize; 2] {
        let s = self.shape();
        [s[0], s[1]]
    }
}

impl Default for CreateConfig {
    fn default() -> Self {
        Self {
            dimensions: [
                Dimension::sized("inline", 2),
                Dimension::sized("crossline", 2),
                Dimension::sized("time", 4),
            ],
            chunks: None,
            digi: 4.0,
            seed: 42,
            units: "ms".into(),
            name: "synthoseis".into(),
        }
    }
}

/// Backward-compatible smoke metadata used by the CLI skeleton.
///
/// Prefer [`CreateConfig`] for new code (explicit coords + chunking).
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

impl From<&StoreMeta> for CreateConfig {
    fn from(m: &StoreMeta) -> Self {
        Self {
            dimensions: [
                Dimension::sized(&m.dims[0], m.shape[0]),
                Dimension::sized(&m.dims[1], m.shape[1]),
                Dimension::sized(&m.dims[2], m.shape[2]),
            ],
            chunks: Some(m.shape),
            digi: m.digi,
            seed: m.seed,
            units: m.units.clone(),
            name: "synthoseis".into(),
        }
    }
}



mod store;

pub use store::{DeliverableWriter, MdioStore};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zarr::read_json;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn create_empty_writes_mdio_hierarchy() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("demo.mdio");
        let cfg = CreateConfig::default();
        let store = MdioStore::create_empty(&root, &cfg).unwrap();

        assert!(root.join(".zgroup").is_file());
        assert!(root.join(".zattrs").is_file());
        assert!(root.join("metadata").join(".zgroup").is_file());
        assert!(root.join("metadata").join("live_mask").join(".zarray").is_file());
        assert!(root
            .join("data")
            .join(PRIMARY_VARIABLE)
            .join(".zarray")
            .is_file());
        assert!(root.join(".zmetadata").is_file());
        assert!(root
            .join("metadata")
            .join(format!("{PRIMARY_VARIABLE}_trace_headers"))
            .join(".zarray")
            .is_file());

        let attrs: Value = read_json(root.join(".zattrs")).unwrap();
        assert_eq!(attrs["api_version"], API_VERSION);
        assert_eq!(attrs["dimension"][0]["name"], "inline");
        assert_eq!(attrs["dimension"][1]["name"], "crossline");
        assert_eq!(attrs["dimension"][2]["name"], "time");
        assert_eq!(attrs["digi"], 4.0);
        assert_eq!(attrs["seed"], 42);
        assert_eq!(attrs["units"], "ms");
        assert_eq!(store.shape(), [2, 2, 4]);

        let mask = store.read_live_mask().unwrap();
        assert_eq!(mask, vec![0, 0, 0, 0]);
    }

    #[test]
    fn write_read_volume_round_trip_updates_live_mask() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("rt.mdio");
        let cfg = CreateConfig {
            chunks: Some([2, 2, 4]),
            ..CreateConfig::default()
        };
        let store = MdioStore::create_empty(&root, &cfg).unwrap();
        let n = 2 * 2 * 4;
        let mut data = vec![0.0f32; n];
        data[0] = 1.5;
        data[n - 1] = -2.25;
        DeliverableWriter::write_volume(&store, &data).unwrap();

        let opened = MdioStore::open(&root).unwrap();
        assert_eq!(opened.shape(), [2, 2, 4]);
        let back = opened.read_volume().unwrap();
        assert_eq!(back.len(), n);
        assert_eq!(back[0], 1.5);
        assert_eq!(back[n - 1], -2.25);
        for (a, b) in data.iter().zip(back.iter()) {
            assert_eq!(a, b);
        }

        let mask = opened.read_live_mask().unwrap();
        assert_eq!(mask, vec![1, 1, 1, 1]);

        let attrs: Value = read_json(root.join(".zattrs")).unwrap();
        assert_eq!(attrs["trace_count"], 4);
        assert!(attrs["min"].as_f64().unwrap() <= -2.25);
        assert!(attrs["max"].as_f64().unwrap() >= 1.5);

        // Chunk file present (single chunk for this config).
        assert!(root
            .join("data")
            .join(PRIMARY_VARIABLE)
            .join("0.0.0")
            .is_file());
    }

    #[test]
    fn multi_chunk_round_trip() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("chunked.mdio");
        let cfg = CreateConfig {
            dimensions: [
                Dimension::sized("inline", 4),
                Dimension::sized("crossline", 4),
                Dimension::sized("sample", 8),
            ],
            chunks: Some([2, 2, 4]),
            digi: 2.0,
            seed: 7,
            units: "ms".into(),
            name: "multi".into(),
        };
        let store = MdioStore::create_empty(&root, &cfg).unwrap();
        let n = 4 * 4 * 8;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        store.write_volume(&data).unwrap();
        let back = store.read_volume().unwrap();
        assert_eq!(data, back);
        assert_eq!(store.read_live_mask().unwrap().iter().filter(|&&b| b == 1).count(), 16);
    }
}
