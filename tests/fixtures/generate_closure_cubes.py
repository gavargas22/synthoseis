#!/usr/bin/env python3
"""Regenerate tests/fixtures/closure_cubes_8.json (closure kernel goldens).

Mirrors datagenerator._closures_vectorised + Closures.get_top_of_closure /
flood_fill_heap (border-as-edge core). No scipy required.
"""
from __future__ import annotations
import heapq, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def bincount_label_sizes(labels):
    return np.bincount(np.ascontiguousarray(labels).ravel().astype(np.int64, copy=False))

def relabel_consecutive(labels_clean):
    unique_vals, inverse = np.unique(labels_clean, return_inverse=True)
    new_labels = inverse.reshape(labels_clean.shape).astype(labels_clean.dtype)
    label_values = list(range(1, unique_vals.size)) if unique_vals.size and unique_vals[0] == 0 else list(range(0, unique_vals.size))
    return label_values, new_labels

def parse_closure_codes_vectorised(hc, labels, num, code=0.1):
    if num <= 0: return hc
    counts = bincount_label_sizes(labels)
    lut_size = max(counts.size, int(num) + 1)
    remap = np.zeros(lut_size, dtype=np.float32)
    upto = min(int(num) + 1, counts.size)
    nonzero = counts[1:upto] > 0
    remap[1:upto] = np.where(nonzero, code + counts[1:upto].astype(np.float32), 0.0)
    if lut_size > int(num) + 1:
        remap[int(num) + 1 : lut_size] = np.arange(int(num) + 1, lut_size, dtype=np.float32)
    return hc + remap[labels.astype(np.int64, copy=False)]

def assign_fluid_types_vectorised(labels_clean, closure_segments, fluid_type_code):
    labels_idx = labels_clean.astype(np.int64, copy=False)
    fg = labels_idx > 0
    fluid = fluid_type_code[labels_idx]
    brine = ((fluid == 0) & fg & (closure_segments > 0)).astype(np.uint8)
    oil = ((fluid == 1) & fg).astype(np.uint8)
    gas = ((fluid == 2) & fg).astype(np.uint8)
    return oil, gas, brine

def closure_size_filter_sizes(a, b):
    return bincount_label_sizes(a)[1:].tolist(), bincount_label_sizes(b)[1:].tolist()

def bbox_for_label_and_fault(labels_clean, label_id, fault_throw, fault_block_val, fault_tol=0.25, pad=32):
    mask = (labels_clean == label_id) & (np.abs(fault_throw - fault_block_val) < fault_tol)
    if not mask.any(): return None
    idx = np.argwhere(mask)
    mins, maxs = idx.min(0), idx.max(0) + 1
    return tuple(slice(max(0, int(mn) - pad), min(int(sh), int(mx) + pad)) for mn, mx, sh in zip(mins, maxs, labels_clean.shape))

def get_top_of_closure(inarray, pad_up=0, pad_down=0):
    mask = inarray != 0
    t = np.where(mask.any(axis=-1), mask.argmax(axis=-1), -1)
    xy, z = np.argwhere(t > 0), t[t > 0]
    out = np.zeros_like(inarray)
    for (x, y), zz in zip(xy, z):
        out[x, y, (zz - pad_up):(zz + pad_down + 1)] = 1
    return out

def filter_labels_by_min_voxels(labels, min_voxels):
    counts = bincount_label_sizes(labels)
    out = labels.copy()
    for lid, c in enumerate(counts):
        if lid and c < min_voxels: out[out == lid] = 0
    return out

def flood_fill_heap_core(test_array, empty_value=1.0e22):
    input_array = np.copy(test_array).astype(np.float64)
    input_array[np.isnan(input_array)] = empty_value
    h_max = float(np.max(input_array * 2.0))
    nr, nc = input_array.shape
    inside = np.ones((nr, nc), dtype=bool)
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    inside[input_array >= empty_value / 2] = False
    output = np.copy(input_array); output[inside] = h_max
    heap = [(float(output[r, c]), int(r), int(c), 1) for r, c in np.transpose(np.where(~inside))]
    heapq.heapify(heap)
    while True:
        try: h_crt, tr, tc, edge = heapq.heappop(heap)
        except IndexError: break
        for nr_, nc_ in ((tr-1, tc), (tr+1, tc), (tr, tc-1), (tr, tc+1)):
            if edge:
                if nr_ < 0 or nc_ < 0 or nr_ >= nr or nc_ >= nc: continue
                if not inside[nr_, nc_]: continue
            if output[nr_, nc_] == h_max:
                output[nr_, nc_] = max(h_crt, float(input_array[nr_, nc_]))
                heapq.heappush(heap, (float(output[nr_, nc_]), nr_, nc_, 0))
    output[output == empty_value] = np.nan
    return output

def rle_i32(arr):
    flat = [int(x) for x in np.asarray(arr).reshape(-1)]
    out, cur, n = [], flat[0], 1
    for x in flat[1:]:
        if x == cur and n < 255: n += 1
        else: out.append([cur, n]); cur, n = x, 1
    out.append([cur, n]); return out

def main():
    labels = np.zeros((8, 8, 8), dtype=np.int32)
    labels[2:4, 2:4, 2] = 1; labels[4:7, 4:7, 3:5] = 3
    gappy = labels.copy(); gappy[1, 1, 1] = 7
    counts = bincount_label_sizes(gappy).tolist()
    label_values, relabeled = relabel_consecutive(gappy)
    fcode = np.zeros(int(relabeled.max()) + 1, dtype=np.int64); fcode[1], fcode[2], fcode[3] = 1, 2, 0
    oil, gas, brine = assign_fluid_types_vectorised(relabeled, (relabeled > 0).astype(np.float32), fcode)
    filtered = filter_labels_by_min_voxels(labels, 5)
    hc = parse_closure_codes_vectorised(np.zeros_like(labels, dtype=np.float32), labels, int(labels.max()), 0.1)
    binary = (labels == 3).astype(np.float32); top = get_top_of_closure(binary)
    s, t = closure_size_filter_sizes(labels, filtered)
    fault = np.zeros((8, 8, 8), dtype=np.float32); fault[4:7, 4:7, 3:5] = 1.0
    bbox = bbox_for_label_and_fault(labels.astype(np.float32), 3, fault, 1.0, 0.25, 1)
    bbox_list = [[sl.start, sl.stop] for sl in bbox] if bbox else None
    surf = np.array([[10,10,7,10,10],[10,5,4,5,10],[10,4,2,4,10],[10,5,4,5,10],[10,10,10,10,10]], dtype=float)
    ff = flood_fill_heap_core(surf)
    out = {
        "meta": {"shape": [8,8,8], "ported_python": [
            "datagenerator._closures_vectorised.bincount_label_sizes",
            "datagenerator._closures_vectorised.relabel_consecutive",
            "datagenerator._closures_vectorised.assign_fluid_types_vectorised",
            "datagenerator._closures_vectorised.parse_closure_codes_vectorised",
            "datagenerator._closures_vectorised.closure_size_filter_sizes",
            "datagenerator._closures_vectorised.bbox_for_label_and_fault",
            "datagenerator.Closures.get_top_of_closure",
            "datagenerator.Closures.flood_fill_heap (core priority flood; border=edge)",
            "filter_labels_by_min_voxels (voxel threshold helper)",
        ]},
        "labels_gappy_rle": rle_i32(gappy), "bincount": counts,
        "relabel": {"label_values": label_values, "labels_rle": rle_i32(relabeled)},
        "fluid": {"fluid_type_code": fcode.tolist(), "oil_rle": rle_i32(oil), "gas_rle": rle_i32(gas), "brine_rle": rle_i32(brine)},
        "filter_min5": {"input_rle": rle_i32(labels), "output_rle": rle_i32(filtered), "min_voxels": 5, "sizes_before": s, "sizes_after": t},
        "closure_codes": {"labels_rle": rle_i32(labels), "num": int(labels.max()), "code": 0.1, "hc_flat": [round(float(x), 6) for x in hc.reshape(-1)]},
        "top_of_closure": {"input_rle": rle_i32(binary.astype(np.int32)), "output_rle": rle_i32(top.astype(np.int32)), "pad_up": 0, "pad_down": 0},
        "bbox": {"label_id": 3, "fault_block_val": 1.0, "fault_tol": 0.25, "pad": 1, "slices": bbox_list},
        "flood_fill_2d": {"shape": [5,5], "input": surf.reshape(-1).tolist(), "output": [round(float(x), 6) for x in ff.reshape(-1)]},
    }
    dest = ROOT / "tests" / "fixtures" / "closure_cubes_8.json"
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
