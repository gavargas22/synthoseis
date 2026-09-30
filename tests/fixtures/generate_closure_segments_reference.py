"""Generate the 3D closure segmentation reference fixture from the legacy code.

Uses the *real* legacy functions from ``datagenerator/Closures.py``:

* ``_flood_fill`` plus the clamp of ``create_closure_labels_from_depth_maps``
  (``cd = min(max(fill, top), base)``) for the per-column closure map.
* ``Closures.segment_closures`` (clip, ``3x3x1`` opening, restrict to sand,
  ``measure.label(connectivity=2)``, ``remove_small_objects``), called with a
  stub ``self`` (``faults.faulted_lithology`` all sand, ``cfg``, and the
  skimage modules that ``Closures.__init__`` attaches).

Sections:

* ``fill``: faulted domes. Each case is a unit top map ``t`` cut by one or
  two faults (straight cliffs with throw), a base map ``b`` (unit thickness
  varies, so faults can offset the unit by more than its thickness), and a
  max column (in samples). Recorded per column: ``2 * cd`` as an integer
  (``cd`` is always a multiple of 0.5 here, so this is exact). Cases are
  drawn so that legacy details the port leaves out cannot matter: all depths
  are >= 1 (no fault-gap walls), a flat deep plateau of >= 5 cells touches
  every edge (the 3-cell border zeroing does not change any spill level),
  and every closed region has >= 50 cells and is 4-connected exactly when it
  is 8-connected (legacy caps 8-connected regions of >= 50 cells; the port
  caps every 4-connected region). The generator asserts all three.
* ``segment``: voxel run sets ``(col, k0, k1)`` on a grid. Footprints are
  unions of aligned 3x3 cell blocks with one interval per block, so the
  legacy ``3x3x1`` opening leaves them unchanged. Recorded per run: the
  legacy component label at the run's voxels after ``remove_small_objects``
  (0 = removed).

Usage (repo root; numpy, scipy, scikit-image installed)::

    python tests/fixtures/generate_closure_segments_reference.py
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
from numpy.random import default_rng
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datagenerator.Closures import Closures, _flood_fill, flood_fill_heap  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def legacy_closure_depth(t, b, max_column):
    with contextlib.redirect_stdout(io.StringIO()):
        cd = _flood_fill(t.astype(float), max_column_height=max_column)
    # create_closure_labels_from_depth_maps
    cd[cd == 0] = t[cd == 0]
    cd[cd == 1] = t[cd == 1]
    cd[cd == 1e5] = t[cd == 1e5]
    cd = np.max(np.dstack((cd, t)), axis=-1)
    return np.min(np.dstack((cd, b)), axis=-1)


def fill_case(rng, ni, nj):
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    plateau = 90.0
    t = np.full((ni, nj), plateau)
    for _ in range(int(rng.integers(1, 5))):
        ci, cj = rng.uniform(10, ni - 10), rng.uniform(10, nj - 10)
        crest = float(rng.integers(20, 45))
        slope = rng.uniform(0.2, 0.7)
        ell = rng.uniform(0.6, 1.6)
        dome = np.floor(crest + slope * ((ii - ci) ** 2 * ell + (jj - cj) ** 2 / ell))
        t = np.minimum(t, dome)
    for _ in range(int(rng.integers(1, 3))):
        th = rng.uniform(0, np.pi)
        d = rng.uniform(-8, 8)
        side = (ii - ni / 2) * np.cos(th) + (jj - nj / 2) * np.sin(th) > d
        t = t + np.where(side, float(rng.integers(3, 22)), 0.0)
    t = np.minimum(t, plateau)
    # >= 5-cell plateau margin on every edge.
    t[:5, :] = t[-5:, :] = plateau
    t[:, :5] = t[:, -5:] = plateau
    thick = rng.integers(2, 14) + (ii + 2 * jj) % 3
    b = t + thick
    return t, b.astype(float)


def check_fill(t, cd, max_column):
    heap = flood_fill_heap(t.astype(float))
    closed = heap > t
    lab4, n4 = ndimage.label(closed)
    lab8, n8 = ndimage.label(closed, structure=np.ones((3, 3)))
    if n4 != n8 or n4 == 0:
        return False
    sizes = np.bincount(lab8.ravel())[1:]
    if sizes.min() < 50:
        return False
    # Border zeroing / plateau: legacy uncapped fill equals the heap fill.
    with contextlib.redirect_stdout(io.StringIO()):
        raw = _flood_fill(t.astype(float), max_column_height=1e9)
    inner = np.zeros_like(closed)
    inner[4:-4, 4:-4] = True
    if not np.array_equal((raw > t) & inner, closed):
        return False
    if not np.array_equal(np.where(closed, raw, 0), np.where(closed, heap, 0)):
        return False
    return (t >= 1).all()


def segment_case(rng):
    bi, bj, nk = int(rng.integers(4, 9)), int(rng.integers(4, 9)), 40
    ni, nj = 3 * bi, 3 * bj
    runs = []
    for a in range(bi):
        for c in range(bj):
            ivs = []
            for _ in range(int(rng.choice([0, 1, 1, 2]))):
                k0 = int(rng.integers(2, 30))
                k1 = k0 + int(rng.integers(1, 6))
                if all(k1 < p0 or k0 > p1 for p0, p1 in ivs):
                    ivs.append((k0, k1))
            for k0, k1 in ivs:
                for di in range(3):
                    for dj in range(3):
                        runs.append(((3 * a + di) * nj + 3 * c + dj, k0, k1))
    vol = np.zeros((ni, nj, nk), dtype=float)
    for col, k0, k1 in runs:
        vol[col // nj, col % nj, k0:k1] = 1.0
    min_voxels = int(rng.choice([1, 20, 60, 150]))
    stub = types.SimpleNamespace(
        faults=types.SimpleNamespace(faulted_lithology=np.ones((ni, nj, nk))),
        cfg=types.SimpleNamespace(closure_min_voxels=min_voxels, verbose=False, infill_factor=1),
    )
    from skimage import measure, morphology

    stub._measure, stub._morphology = measure, morphology
    stub.remove_small_objects = types.MethodType(Closures.remove_small_objects, stub)
    labels, opened = Closures.segment_closures(stub, vol)
    assert np.array_equal(opened > 0, vol > 0), "opening must be a no-op"
    comp = []
    for col, k0, k1 in runs:
        ls = set(labels[col // nj, col % nj, k0:k1].tolist())
        assert len(ls) == 1
        comp.append(int(ls.pop()))
    return {"shape": [ni, nj, nk], "min_voxels": min_voxels,
            "runs": [list(r) for r in runs], "label": comp}


def main():
    rng = default_rng(20260929)
    fill = []
    tries = 0
    while len(fill) < 24:
        tries += 1
        ni, nj = int(rng.integers(44, 64)), int(rng.integers(44, 64))
        t, b = fill_case(rng, ni, nj)
        max_column = float(rng.choice([12.0, 20.0, 37.5, 1e9]))
        if not check_fill(t, None, max_column):
            continue
        cd = legacy_closure_depth(t, b, max_column)
        assert (2 * cd == np.round(2 * cd)).all()
        fill.append({
            "shape": [ni, nj],
            "max_column": max_column,
            "t": t.astype(int).ravel().tolist(),
            "b": b.astype(int).ravel().tolist(),
            "cd2": (2 * cd).astype(np.int64).ravel().tolist(),
        })
    segment = [segment_case(rng) for _ in range(40)]
    out = {
        "meta": {
            "generator": "tests/fixtures/generate_closure_segments_reference.py",
            "legacy": "Closures._flood_fill + create_closure_labels_from_depth_maps clamp; Closures.segment_closures",
        },
        "fill": fill,
        "segment": segment,
    }
    path = Path(__file__).with_name("closure_segments_reference.json")
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    closed = sum(int((np.array(c["cd2"]) > 2 * np.array(c["t"])).sum()) for c in fill)
    print(f"wrote {path} ({path.stat().st_size} bytes); fill cases {len(fill)} "
          f"({tries} tries, {closed} closed columns); segment cases {len(segment)}, "
          f"runs {sum(len(s['runs']) for s in segment)}")


if __name__ == "__main__":
    main()
