//! Bounded closure / trap kernels ported from
//! `datagenerator/_closures_vectorised.py` and `datagenerator/Closures.py`.


/// Per-label voxel counts: length `max(labels)+1`, index 0 = background.
///
/// Port of `datagenerator._closures_vectorised.bincount_label_sizes`.
pub fn bincount_label_sizes(labels: &[i32]) -> Vec<i64> {
    if labels.is_empty() {
        return vec![0];
    }
    let mut max_v = 0i32;
    for &v in labels {
        assert!(v >= 0, "bincount_label_sizes: negative label {v}");
        if v > max_v {
            max_v = v;
        }
    }
    let mut counts = vec![0i64; (max_v as usize) + 1];
    for &v in labels {
        counts[v as usize] += 1;
    }
    counts
}

/// Relabel to consecutive ints `0..N` (0 = background preserved if present).
///
/// Returns `(label_values, new_labels)` where `label_values` is `[1..N]` when
/// background 0 is present — matching `relabel_consecutive`.
pub fn relabel_consecutive(labels: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut unique: Vec<i32> = labels.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let mut remap = std::collections::HashMap::with_capacity(unique.len());
    for (new_id, &old) in unique.iter().enumerate() {
        remap.insert(old, new_id as i32);
    }
    let new_labels: Vec<i32> = labels.iter().map(|v| remap[v]).collect();
    let label_values = if unique.first().copied() == Some(0) {
        (1..unique.len() as i32).collect()
    } else {
        (0..unique.len() as i32).collect()
    };
    (label_values, new_labels)
}

/// Zero labels whose voxel count is strictly below `min_voxels` (0 untouched).
///
/// Voxel-threshold helper used by closure size filters / `remove_small_objects`
/// after connected-component labelling.
pub fn filter_labels_by_min_voxels(labels: &[i32], min_voxels: i64) -> Vec<i32> {
    let counts = bincount_label_sizes(labels);
    let mut out = labels.to_vec();
    for (lid, &c) in counts.iter().enumerate() {
        if lid == 0 {
            continue;
        }
        if c < min_voxels {
            let id = lid as i32;
            for v in &mut out {
                if *v == id {
                    *v = 0;
                }
            }
        }
    }
    out
}

/// Size lists before/after a size filter: `bincount[1:]` for each volume.
///
/// Port of `closure_size_filter_sizes`.
pub fn closure_size_filter_sizes(before: &[i32], after: &[i32]) -> (Vec<i64>, Vec<i64>) {
    let s = bincount_label_sizes(before);
    let t = bincount_label_sizes(after);
    (s[1..].to_vec(), t[1..].to_vec())
}

/// Vectorised `parse_closure_codes`: add `code + size` into `hc` for labels `1..=num`.
///
/// Port of `parse_closure_codes_vectorised`. Mutates a copy of `hc` and returns it.
pub fn parse_closure_codes(hc: &[f32], labels: &[i32], num: i32, code: f32) -> Vec<f32> {
    assert_eq!(hc.len(), labels.len());
    let mut out = hc.to_vec();
    if num <= 0 {
        return out;
    }
    let counts = bincount_label_sizes(labels);
    let lut_size = counts.len().max((num as usize) + 1);
    let mut remap = vec![0.0f32; lut_size];
    let upto = ((num as usize) + 1).min(counts.len());
    for i in 1..upto {
        if counts[i] > 0 {
            remap[i] = code + counts[i] as f32;
        }
    }
    if lut_size > (num as usize) + 1 {
        for i in ((num as usize) + 1)..lut_size {
            remap[i] = i as f32;
        }
    }
    for (o, &lab) in out.iter_mut().zip(labels.iter()) {
        let idx = lab as usize;
        if idx < remap.len() {
            *o += remap[idx];
        }
    }
    out
}

/// Assign oil / gas / brine uint8 masks from consecutive labels + fluid codes.
///
/// Port of `assign_fluid_types_vectorised`.
/// `fluid_type_code[i]` ∈ {0=brine, 1=oil, 2=gas}; length ≥ `max(labels)+1`.
pub fn assign_fluid_types(
    labels_clean: &[i32],
    closure_segments: &[f32],
    fluid_type_code: &[i32],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(labels_clean.len(), closure_segments.len());
    let n = labels_clean.len();
    let mut oil = vec![0u8; n];
    let mut gas = vec![0u8; n];
    let mut brine = vec![0u8; n];
    for i in 0..n {
        let lab = labels_clean[i];
        if lab <= 0 {
            continue;
        }
        let code = fluid_type_code[lab as usize];
        match code {
            0 if closure_segments[i] > 0.0 => brine[i] = 1,
            1 => oil[i] = 1,
            2 => gas[i] = 1,
            _ => {}
        }
    }
    (oil, gas, brine)
}
