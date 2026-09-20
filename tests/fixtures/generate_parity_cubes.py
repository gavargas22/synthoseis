#!/usr/bin/env python3
"""Regenerate fixed-seed parity cubes (labels + angle stacks) for CI docs.

Produces tests/fixtures/parity_cubes_8.json (minified).
Mirrors datagenerator.Horizons plane / thickness-clip helpers used by the Rust ports.

Angle stacks are fully deterministic (no RNG): angle[i,j,k] = 0.1*i + 0.05*j + 0.02*k.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fit_plane_lsq(xyz: np.ndarray) -> list[float]:
    """Mirror Horizons.fit_plane_lsq (Z = aX + bY + c)."""
    rows = xyz.shape[0]
    g = np.ones((rows, 3))
    g[:, 0] = xyz[:, 0]
    g[:, 1] = xyz[:, 1]
    z = xyz[:, 2]
    (a, b, c), _, _, _ = np.linalg.lstsq(g, z, rcond=-1)
    return [float(a), float(b), float(c)]


def eval_plane(nx: int, ny: int, a: float, b: float, c: float) -> np.ndarray:
    """Mirror Horizons.eval_plane."""
    z = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            z[i, j] = a * i + b * j + c
    return z


def rotate_point(x: float, y: float, angle_in_degrees: float) -> tuple[float, float]:
    angle = angle_in_degrees * np.pi / 180.0
    x1 = math.cos(angle) * x + math.sin(angle) * y
    y1 = -math.sin(angle) * x + math.cos(angle) * y
    return float(x1), float(y1)


def enforce_nonnegative_thicknesses(maps: np.ndarray) -> np.ndarray:
    """Thickness clip from Horizons.insert_feature_into_horizon_stack."""
    new_maps = maps.copy()
    for i in range(new_maps.shape[-1] - 1, 1, -1):
        layer_thickness = new_maps[..., i] - new_maps[..., i - 1]
        if np.min(layer_thickness) < 0:
            np.clip(layer_thickness, 0, a_max=None, out=layer_thickness)
            new_maps[..., i - 1] = new_maps[..., i] - layer_thickness
    return new_maps


def fill_layer_labels(depth_maps: np.ndarray, n_samples: int) -> np.ndarray:
    """Discrete labels between successive horizons (no partial voxels)."""
    ni, nj, nh = depth_maps.shape
    labels = np.full((ni, nj, n_samples), 255, dtype=np.uint8)
    for i in range(ni):
        for j in range(nj):
            for h in range(nh - 1):
                z0 = int(np.ceil(depth_maps[i, j, h]))
                z1 = int(np.floor(depth_maps[i, j, h + 1]))
                z0 = max(0, min(n_samples, z0))
                z1 = max(0, min(n_samples, z1))
                if z1 > z0:
                    labels[i, j, z0:z1] = h
    return labels


def _r6(xs):
    return [round(float(x), 6) for x in xs]


def main() -> None:
    ni = nj = ns = 8

    xyz = np.array(
        [
            [0.0, 0.0, -1.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [2.0, 4.0, 1.0],
            [1.0, 1.0, -0.25],
        ],
        dtype=float,
    )
    abc = fit_plane_lsq(xyz)
    plane = eval_plane(8, 8, *abc)

    base = plane - plane.min() + 1.0
    h0 = np.zeros((ni, nj))
    h1 = base * 0.35 + 1.5
    h2 = base * 0.70 + 3.0
    h3 = np.full((ni, nj), float(ns - 1))
    depth_maps = np.dstack([h0, h1, h2, h3])
    depth_maps_glitch = depth_maps.copy()
    depth_maps_glitch[..., 1] = depth_maps_glitch[..., 2] + 1.0
    clipped = enforce_nonnegative_thicknesses(depth_maps_glitch)
    labels = fill_layer_labels(clipped, ns)

    ii, jj, kk = np.meshgrid(
        np.arange(ni), np.arange(nj), np.arange(ns), indexing="ij"
    )
    # Fully deterministic angle stack (no RNG) for compact checked-in goldens.
    angle_stack = (0.1 * ii + 0.05 * jj + 0.02 * kk).astype(np.float32)

    labels_pert = labels.copy()
    present = [int(c) for c in sorted(set(labels.flatten()) - {255})]
    assert len(present) >= 2, present
    idxs = list(zip(*np.where(labels == present[0])))
    i, j, k = idxs[0]
    labels_pert[i, j, k] = present[1]
    angle_pert = angle_stack + 1e-4

    out = {
        "meta": {
            "seed": "deterministic-no-rng",
            "shape": [ni, nj, ns],
            "metrics": {
                "labels": "IoU (macro mean over present classes) + per-voxel agreement",
                "angle_stacks": "MAE + max-abs",
                "tolerances": {
                    "label_iou_min": 0.99,
                    "label_agreement_min": 0.99,
                    "angle_mae_max": 1e-3,
                    "angle_max_abs_max": 5e-3,
                },
            },
            "ported_python": [
                "datagenerator.Horizons.Horizons.fit_plane_lsq",
                "datagenerator.Horizons.Horizons.eval_plane",
                "datagenerator.Horizons.Horizons.insert_feature_into_horizon_stack (nonneg thickness clip)",
                "fill_layer_labels (discrete labels between horizons)",
            ],
        },
        "plane": {
            "xyz": xyz.tolist(),
            "abc": [round(x, 12) for x in abc],
            "eval_8x8": _r6(plane.reshape(-1)),
        },
        "rotate_point": {
            "cases": [
                {
                    "x": 1.0,
                    "y": 0.0,
                    "deg": 90.0,
                    "out": list(rotate_point(1.0, 0.0, 90.0)),
                },
                {
                    "x": 3.0,
                    "y": 4.0,
                    "deg": 45.0,
                    "out": list(rotate_point(3.0, 4.0, 45.0)),
                },
            ]
        },
        "horizon_clip": {
            "input": _r6(depth_maps_glitch.reshape(-1)),
            "output": _r6(clipped.reshape(-1)),
            "shape": list(depth_maps_glitch.shape),
        },
        "labels": {
            "reference": labels.reshape(-1).astype(int).tolist(),
            "perturbed": labels_pert.reshape(-1).astype(int).tolist(),
        },
        "angle_stack": {
            "reference": _r6(angle_stack.reshape(-1)),
            "perturbed": _r6(angle_pert.reshape(-1)),
        },
    }

    dest = ROOT / "tests" / "fixtures" / "parity_cubes_8.json"
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
