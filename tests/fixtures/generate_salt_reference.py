"""Generate the salt-body reference fixture from the legacy code.

Uses the *real* legacy functions:

* ``datagenerator.Salt.SaltModel`` (``compute_salt_body_segmentation`` →
  ``insertSalt3D`` → ``salt_circle_points`` / ``create_circular_pointcloud``
  → ``util.is_it_in_hull``), with a stub ``cfg``. The stub's
  ``horizon_ss.spawn`` returns a fixed child ``SeedSequence``, so a second
  generator on the same child replays the model's ``rng`` stream.
* ``SaltModel.update_depth_maps_with_salt_segments_drag`` and
  ``push_down_remove_negative_thickness`` (horizon drag against the salt).
* ``Closures._flood_fill`` plus the top / base clamp of
  ``create_closure_labels_from_depth_maps``, with salt gaps in the top map
  (legacy NaN → 0 → ``/digi`` → ``-1`` with ``partial_voxels``).

Sections:

* ``geometry``: cubes, the horizon-1 map, the model's first 1000 unit draws
  (IEEE bits), the legacy point cloud (bits) and the legacy salt mask as one
  ``[k0, k1)`` run per column (the generator asserts each column's salt is
  contiguous, as it must be for a convex hull).
* ``drag``: horizon maps (bits), a salt mask (runs) and the legacy dragged
  maps (bits).
* ``fill``: faulted domes with a salt gap region; per column ``2 * cd`` as
  an integer (``cd`` is a multiple of 0.5).
* ``population``: 4000 legacy salt bodies on a 64×64×1250 cube (legacy
  ``example.json``, pad 10) with a flat horizon-1 map at 20 samples. The
  hull test is stubbed out here (only the draws are compared); recorded:
  radius, top, tip depth, first-circle centre and radius, base-circle depth,
  centre and radius.

Usage (repo root; numpy, scipy, scikit-image installed)::

    python tests/fixtures/generate_salt_reference.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import datagenerator.Salt as salt_mod  # noqa: E402
from datagenerator.Closures import _flood_fill  # noqa: E402
from datagenerator.Salt import SaltModel  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
PAD = 10


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64).ravel().tolist()


def stub_cfg(shape, h1, child):
    ni, nj, nk = shape
    store = {"faulted_depth_maps": np.repeat(h1[:, :, None], 3, axis=2)}
    return types.SimpleNamespace(
        cube_shape=(ni, nj, nk),
        pad_samples=PAD,
        horizon_ss=types.SimpleNamespace(spawn=lambda n: [child]),
        create_array=lambda name, shape: np.zeros(shape),
        model_store=store,
    )


def runs_of(mask):
    ni, nj, nk = mask.shape
    out = []
    for i in range(ni):
        for j in range(nj):
            idx = np.flatnonzero(mask[i, j])
            if idx.size == 0:
                out.append([0, 0])
            else:
                assert idx[-1] - idx[0] + 1 == idx.size, "salt must be contiguous per column"
                out.append([int(idx[0]), int(idx[-1]) + 1])
    return out


def mask_of(runs, shape):
    ni, nj, nk = shape
    m = np.zeros(shape, dtype=np.int16)
    for c, (k0, k1) in enumerate(runs):
        m[c // nj, c % nj, k0:k1] = 1
    return m


def geometry_case(seed, shape, h1):
    child = SeedSequence(seed)
    model = SaltModel(stub_cfg(shape, h1, child))
    with contextlib.redirect_stdout(io.StringIO()):
        model.compute_salt_body_segmentation()
    draws = default_rng(child).random(1000)
    m = np.asarray(model.salt_segments) > 0
    return {
        "shape": list(shape),
        "h1": bits(h1),
        "draws": bits(draws),
        "points": bits(np.asarray(model.points)),
        "runs": runs_of(m),
    }, m


def drag_case(maps, salt_runs, shape):
    ni, nj, nk = shape
    salt = mask_of(salt_runs, (ni, nj, nk + PAD))
    store = {
        "faulted_depth_maps": maps.copy(),
        "faulted_depth_maps_gaps": maps.copy(),
        "salt_segments": salt,
        "faulted_depth": np.zeros((ni, nj, nk + PAD), dtype=np.float32),
    }
    model = SaltModel.__new__(SaltModel)
    model.cfg = types.SimpleNamespace(cube_shape=(ni, nj, nk), model_store=store, model_qc_volumes=False)
    with contextlib.redirect_stdout(io.StringIO()):
        dragged, _ = model.update_depth_maps_with_salt_segments_drag()
    return {
        "shape": [ni, nj, nk],
        "nh": int(maps.shape[2]),
        "maps": bits(maps),
        "runs": salt_runs,
        "dragged": bits(dragged),
    }


def legacy_closure_depth(t, b, max_column):
    with contextlib.redirect_stdout(io.StringIO()):
        cd = _flood_fill(t.astype(float), max_column_height=max_column)
    tt = np.maximum(t, 0.0)
    cd[cd == 0] = tt[cd == 0]
    cd[cd == 1] = tt[cd == 1]
    cd[cd == 1e5] = tt[cd == 1e5]
    cd = np.max(np.dstack((cd, t)), axis=-1)
    return np.min(np.dstack((cd, b)), axis=-1)


def fill_case(rng):
    ni, nj = int(rng.integers(44, 64)), int(rng.integers(44, 64))
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    plateau = 90.0
    # Salt: a disc of gap cells with horizons dragged up around it.
    si, sj = rng.uniform(14, ni - 14), rng.uniform(14, nj - 14)
    sr = rng.uniform(3, 7)
    r = np.sqrt((ii - si) ** 2 + (jj - sj) ** 2)
    gap = r < sr
    t = np.full((ni, nj), plateau)
    t = np.minimum(t, np.floor(rng.uniform(25, 45) + rng.uniform(1.5, 4) * np.maximum(r - sr, 0)))
    for _ in range(int(rng.integers(0, 3))):
        ci, cj = rng.uniform(10, ni - 10), rng.uniform(10, nj - 10)
        dome = np.floor(rng.uniform(25, 50) + rng.uniform(0.2, 0.7) * ((ii - ci) ** 2 + (jj - cj) ** 2))
        t = np.minimum(t, dome)
    for _ in range(int(rng.integers(0, 2))):
        th = rng.uniform(0, np.pi)
        side = (ii - ni / 2) * np.cos(th) + (jj - nj / 2) * np.sin(th) > rng.uniform(-8, 8)
        t = t + np.where(side, float(rng.integers(3, 15)), 0.0)
    t = np.minimum(t, plateau)
    t[:5, :] = t[-5:, :] = plateau
    t[:, :5] = t[:, -5:] = plateau
    b = t + rng.integers(2, 14) + (ii + 2 * jj) % 3
    t[gap] = -1.0
    return t, b.astype(float), gap


def check_fill(t, gap, cd):
    if not gap.any():
        return False
    ring = ndimage.binary_dilation(gap, structure=np.ones((3, 3))) & ~gap
    if ring[:5, :].any() or ring[-5:, :].any() or ring[:, :5].any() or ring[:, -5:].any():
        return False
    closed = cd > np.maximum(t, 0)
    lab4, n4 = ndimage.label(closed)
    lab8, n8 = ndimage.label(closed, structure=np.ones((3, 3)))
    if n4 != n8 or n4 == 0:
        return False
    if np.bincount(lab8.ravel())[1:].min() < 50:
        return False
    # Some closure must touch the ring (a trap against the salt flank).
    return bool((ndimage.binary_dilation(ring, structure=np.ones((3, 3))) & closed).any())


def population(n):
    shape = (64, 64, 1250)
    h1 = np.full((64, 64), 20.0)
    real = salt_mod.is_it_in_hull
    salt_mod.is_it_in_hull = lambda points, xyz: np.zeros(len(xyz), dtype=bool)
    out = {k: [] for k in ("radius", "top", "tip", "cx", "cy", "r1", "base", "bx", "by", "r2")}
    try:
        for s in range(n):
            child = SeedSequence(900_000 + s)
            cfg = stub_cfg(shape, h1, child)
            model = SaltModel(cfg)
            # Record the radius / top draws the same way the model makes them.
            rec = default_rng(child)
            radius = rec.triangular(shape[0] / 6, shape[0] / 5, shape[0] / 4)
            top = 20.0 + rec.uniform(150, 300)
            model.cfg.cube_shape = shape
            with contextlib.redirect_stdout(io.StringIO()):
                model.compute_salt_body_segmentation()
            p = np.asarray(model.points)
            c1 = p[0:36]
            c4 = p[109:145]
            out["radius"].append(radius)
            out["top"].append(top)
            out["tip"].append(p[108, 2])
            out["cx"].append(c1[:, 0].mean())
            out["cy"].append(c1[:, 1].mean())
            out["r1"].append(np.hypot(c1[:, 0] - c1[:, 0].mean(), c1[:, 1] - c1[:, 1].mean()).mean())
            out["base"].append(c4[:, 2].mean())
            out["bx"].append(c4[:, 0].mean())
            out["by"].append(c4[:, 1].mean())
            out["r2"].append(np.hypot(c4[:, 0] - c4[:, 0].mean(), c4[:, 1] - c4[:, 1].mean()).mean())
    finally:
        salt_mod.is_it_in_hull = real
    return {k: [float(x) for x in v] for k, v in out.items()}


def main():
    rng = default_rng(20260930)
    geometry = []
    masks = []
    for n, shape in enumerate([(40, 36, 380), (32, 32, 420), (48, 40, 400), (36, 48, 390),
                               (30, 30, 360), (44, 44, 410), (28, 40, 370), (40, 40, 440)]):
        ni, nj, _ = shape
        ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
        h1 = 8 + rng.uniform(4, 20) * np.exp(-((ii - ni / 2) ** 2 + (jj - nj / 2) ** 2) / (0.1 * ni * nj)) \
            + rng.uniform(0, 3, (ni, nj))
        g, m = geometry_case(1000 + n, shape, h1)
        geometry.append(g)
        masks.append(m)
    rng = default_rng([20260930, 1])
    drag = []
    for n, (g, m) in enumerate(zip(geometry[:4], masks[:4])):
        ni, nj, nk = g["shape"]
        nh = int(rng.integers(10, 16))
        th = rng.uniform(0.6, 1.4, (ni, nj, nh)) * (nk / nh)
        maps = 5 + np.cumsum(th, axis=2)
        if n % 2 == 0:
            maps = np.round(maps)
        drag.append(drag_case(maps, g["runs"], (ni, nj, nk)))
    # Small maps (fewer cells than the Gaussian radius) and a box of salt.
    for ni, nj in ((5, 7), (1, 9), (13, 3), (2, 2)):
        nk, nh = 60, 12
        maps = 3 + np.cumsum(rng.uniform(2, 8, (ni, nj, nh)), axis=2)
        salt = np.zeros((ni, nj, nk + PAD), dtype=bool)
        salt[: max(1, ni // 2), :, 20:] = True
        drag.append(drag_case(maps, runs_of(salt), (ni, nj, nk)))
    rng = default_rng([20260930, 2])
    fill = []
    tries = 0
    while len(fill) < 24:
        tries += 1
        t, b, gap = fill_case(rng)
        max_column = float(rng.choice([12.0, 20.0, 37.5]))
        cd = legacy_closure_depth(t, b, max_column)
        if not check_fill(t, gap, cd):
            continue
        assert (2 * cd == np.round(2 * cd)).all()
        fill.append({
            "shape": list(t.shape),
            "max_column": max_column,
            "t": t.astype(int).ravel().tolist(),
            "b": b.astype(int).ravel().tolist(),
            "gap": gap.astype(int).ravel().tolist(),
            "cd2": (2 * cd).astype(np.int64).ravel().tolist(),
        })
    pop = population(4000)
    out = {
        "meta": {
            "generator": "tests/fixtures/generate_salt_reference.py",
            "legacy": "Salt.SaltModel (geometry, drag), util.is_it_in_hull, Closures._flood_fill with salt gaps",
            "pad": PAD,
            "population": {"n": 4000, "shape": [64, 64, 1250], "h1": 20.0},
        },
        "geometry": geometry,
        "drag": drag,
        "fill": fill,
        "population": pop,
    }
    path = Path(__file__).with_name("salt_reference.json")
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    dragged = sum(len(d["maps"]) for d in drag)
    salt_vox = sum(int(m.sum()) for m in masks)
    print(f"wrote {path} ({path.stat().st_size} bytes); geometry {len(geometry)} ({salt_vox} salt voxels), "
          f"drag {len(drag)} ({dragged} map values), fill {len(fill)} ({tries} tries), population {len(pop['radius'])}")


if __name__ == "__main__":
    main()
