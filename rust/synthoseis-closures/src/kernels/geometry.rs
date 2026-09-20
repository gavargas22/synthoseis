//! Trap crest + fault bbox helpers.

/// Mask retaining only the shallowest (first nonzero along z) sample per (x,y).
///
/// Port of `Closures.get_top_of_closure`. `shape = [nx, ny, nz]`, row-major.
pub fn get_top_of_closure(volume: &[f32], shape: [usize; 3], pad_up: usize, pad_down: usize) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    assert_eq!(volume.len(), nx * ny * nz);
    let mut out = vec![0.0f32; volume.len()];
    for x in 0..nx {
        for y in 0..ny {
            let mut top: Option<usize> = None;
            for z in 0..nz {
                if volume[(x * ny + y) * nz + z] != 0.0 {
                    top = Some(z);
                    break;
                }
            }
            if let Some(z) = top {
                if z == 0 {
                    // Python: `t > 0` gate — skip column if top is at index 0
                    continue;
                }
                let zmin = z.saturating_sub(pad_up);
                let zmax = (z + pad_down + 1).min(nz);
                let base = (x * ny + y) * nz;
                for zz in zmin..zmax {
                    out[base + zz] = 1.0;
                }
            }
        }
    }
    out
}

/// Tight AABB around `(labels == label_id) & |fault_throw - val| < tol`, padded.
///
/// Port of `bbox_for_label_and_fault`. Returns `[ [i0,i1), [j0,j1), [k0,k1) ]`
/// or `None` if the mask is empty. `shape = [ni, nj, nk]`.
pub fn bbox_for_label_and_fault(
    labels: &[i32],
    shape: [usize; 3],
    label_id: i32,
    fault_throw: &[f32],
    fault_block_val: f32,
    fault_tol: f32,
    pad: usize,
) -> Option<[[usize; 2]; 3]> {
    let [ni, nj, nk] = shape;
    assert_eq!(labels.len(), ni * nj * nk);
    assert_eq!(fault_throw.len(), labels.len());
    let mut min_i = usize::MAX;
    let mut min_j = usize::MAX;
    let mut min_k = usize::MAX;
    let mut max_i = 0usize;
    let mut max_j = 0usize;
    let mut max_k = 0usize;
    let mut any = false;
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let idx = (i * nj + j) * nk + k;
                if labels[idx] != label_id {
                    continue;
                }
                if (fault_throw[idx] - fault_block_val).abs() >= fault_tol {
                    continue;
                }
                any = true;
                min_i = min_i.min(i);
                min_j = min_j.min(j);
                min_k = min_k.min(k);
                max_i = max_i.max(i);
                max_j = max_j.max(j);
                max_k = max_k.max(k);
            }
        }
    }
    if !any {
        return None;
    }
    Some([
        [min_i.saturating_sub(pad), (max_i + 1 + pad).min(ni)],
        [min_j.saturating_sub(pad), (max_j + 1 + pad).min(nj)],
        [min_k.saturating_sub(pad), (max_k + 1 + pad).min(nk)],
    ])
}
