//! Internal Zarr v2 JSON + chunk helpers.
use crate::{err, Dimension, Result};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) fn ceildiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub(crate) fn chunk_key(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

pub(crate) fn write_json(path: PathBuf, value: &Value) -> Result<()> {
    let mut f = File::create(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}

pub(crate) fn read_json(path: PathBuf) -> Result<Value> {
    let s = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&s)?)
}

pub(crate) fn write_zarray(
    path: PathBuf,
    shape: &[usize],
    chunks: &[usize],
    dtype: &str,
    fill_value: Value,
) -> Result<()> {
    write_json(
        path,
        &json!({
            "zarr_format": 2,
            "shape": shape,
            "chunks": chunks,
            "dtype": dtype,
            "compressor": null,
            "fill_value": fill_value,
            "order": "C",
            "filters": null,
            "dimension_separator": "."
        }),
    )
}

pub(crate) fn write_chunk_bytes(array_dir: &Path, key: &str, bytes: &[u8]) -> Result<()> {
    let mut f = File::create(array_dir.join(key))?;
    f.write_all(bytes)?;
    Ok(())
}

pub(crate) fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub(crate) fn parse_usize3(v: &Value) -> Result<[usize; 3]> {
    let arr = v
        .as_array()
        .ok_or_else(|| err("expected length-3 array"))?;
    if arr.len() != 3 {
        return Err(err("expected length-3 array"));
    }
    Ok([
        arr[0].as_u64().unwrap_or(0) as usize,
        arr[1].as_u64().unwrap_or(0) as usize,
        arr[2].as_u64().unwrap_or(0) as usize,
    ])
}

pub(crate) fn parse_dimensions(attrs: &Value) -> Result<[Dimension; 3]> {
    let dims = attrs
        .get("dimension")
        .and_then(|d| d.as_array())
        .ok_or_else(|| err("missing dimension in .zattrs"))?;
    if dims.len() < 3 {
        return Err(err("need at least 3 dimensions"));
    }
    let mut out = [
        Dimension::sized("inline", 0),
        Dimension::sized("crossline", 0),
        Dimension::sized("time", 0),
    ];
    for i in 0..3 {
        let name = dims[i]
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(["inline", "crossline", "time"][i])
            .to_string();
        let coords = dims[i]
            .get("coords")
            .and_then(|c| c.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        out[i] = Dimension { name, coords };
    }
    Ok(out)
}

pub(crate) fn update_root_attr_u64(root: &Path, key: &str, value: u64) -> Result<()> {
    let path = root.join(".zattrs");
    let mut attrs: Value = read_json(path.clone())?;
    attrs[key] = json!(value);
    write_json(path, &attrs)
}

pub(crate) fn update_stats(root: &Path, samples: &[f32]) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }
    let mut min = samples[0];
    let mut max = samples[0];
    let mut sum = 0f64;
    let mut sumsq = 0f64;
    for &s in samples {
        min = min.min(s);
        max = max.max(s);
        let x = s as f64;
        sum += x;
        sumsq += x * x;
    }
    let n = samples.len() as f64;
    let mean = sum / n;
    let rms = (sumsq / n).sqrt();
    let var = (sumsq / n) - mean * mean;
    let std = var.max(0.0).sqrt();
    let path = root.join(".zattrs");
    let mut attrs: Value = read_json(path.clone())?;
    attrs["mean"] = json!(mean);
    attrs["std"] = json!(std);
    attrs["rms"] = json!(rms);
    attrs["min"] = json!(min);
    attrs["max"] = json!(max);
    write_json(path, &attrs)
}

pub(crate) fn iso8601_now_approx() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    const DAY: u64 = 86400;
    let days = secs / DAY;
    let rem = secs % DAY;
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Write Zarr v2 consolidated `.zmetadata` so mdio-python `MDIOReader` can open.
///
/// Classic format: `{"zarr_consolidated_format": 1, "metadata": { "<path>": <json>, ... }}`.
pub(crate) fn write_consolidated_metadata(root: &Path) -> Result<()> {
    use std::collections::BTreeMap;

    let mut metadata: BTreeMap<String, Value> = BTreeMap::new();

    fn collect(dir: &Path, prefix: &str, out: &mut BTreeMap<String, Value>) -> Result<()> {
        for name in [".zgroup", ".zattrs", ".zarray"] {
            let path = dir.join(name);
            if path.is_file() {
                let key = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{prefix}/{name}")
                };
                out.insert(key, read_json(path)?);
            }
        }
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let child_prefix = if prefix.is_empty() {
                        name
                    } else {
                        format!("{prefix}/{name}")
                    };
                    collect(&path, &child_prefix, out)?;
                }
            }
        }
        Ok(())
    }

    collect(root, "", &mut metadata)?;
    let doc = json!({
        "zarr_consolidated_format": 1,
        "metadata": metadata,
    });
    write_json(root.join(".zmetadata"), &doc)
}
