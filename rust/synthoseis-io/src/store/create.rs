//! create_empty / create
use super::MdioStore;
use crate::zarr::*;
use crate::{
    err, API_VERSION, CreateConfig, PRIMARY_VARIABLE, Result, StoreMeta,
};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

impl MdioStore {
    pub fn create_empty(path: impl AsRef<Path>, config: &CreateConfig) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir_all(root.join("metadata").join("live_mask"))?;
        fs::create_dir_all(root.join("data").join(PRIMARY_VARIABLE))?;

        let shape = config.shape();
        let chunks = config.chunks_or_shape();
        for (i, &c) in chunks.iter().enumerate() {
            if c == 0 || c > shape[i] {
                return Err(err(format!(
                    "invalid chunk[{i}]={c} for shape {:?}",
                    shape
                )));
            }
        }

        write_json(root.join(".zgroup"), &json!({ "zarr_format": 2 }))?;
        let dims_json: Vec<Value> = config
            .dimensions
            .iter()
            .map(|d| json!({ "name": d.name, "coords": d.coords }))
            .collect();
        let created = iso8601_now_approx();
        write_json(
            root.join(".zattrs"),
            &json!({
                "name": config.name,
                "created": created,
                "api_version": API_VERSION,
                "dimension": dims_json,
                "trace_count": 0,
                "mean": 0.0,
                "std": 0.0,
                "rms": 0.0,
                "min": 0.0,
                "max": 0.0,
                "digi": config.digi,
                "seed": config.seed,
                "units": config.units,
                "synthoseis_mdio": true,
                "zarr_format_note": "Zarr v2 to match mdio-python create_empty"
            }),
        )?;

        write_json(
            root.join("metadata").join(".zgroup"),
            &json!({ "zarr_format": 2 }),
        )?;
        write_json(
            root.join("metadata").join(".zattrs"),
            &json!({
                "text_header": [],
                "binary_header": {}
            }),
        )?;

        let live_shape = config.spatial_shape();
        let live_chunks = live_shape;
        write_zarray(
            root.join("metadata").join("live_mask").join(".zarray"),
            &live_shape,
            &live_chunks,
            "|b1",
            json!(false),
        )?;
        let live_n = live_shape[0] * live_shape[1];
        write_chunk_bytes(
            &root.join("metadata").join("live_mask"),
            &chunk_key(&[0, 0]),
            &vec![0u8; live_n],
        )?;

        write_json(root.join("data").join(".zgroup"), &json!({ "zarr_format": 2 }))?;
        write_zarray(
            root.join("data").join(PRIMARY_VARIABLE).join(".zarray"),
            &shape,
            &chunks,
            "<f4",
            json!(0.0),
        )?;

        Ok(Self {
            root,
            config: config.clone(),
        })
    }

    pub fn create(path: impl AsRef<Path>, meta: &StoreMeta) -> Result<Self> {
        Self::create_empty(path, &CreateConfig::from(meta))
    }
}
