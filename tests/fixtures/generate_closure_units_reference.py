"""Generate the closures-per-sand-unit reference fixture from the legacy code.

Uses the *real* legacy functions:

* ``Closures.find_top_lith_horizons``, called with a stub ``self``
  (``facies``, ``onlap_list = []``, ``cfg.verbose = False``). The toy model
  has no onlaps. It is combined with the unit loop of
  ``create_closure_labels_from_depth_maps``: ``for ihorizon in
  range(len(top_lith) - 1)``, keeping ``top_lith_facies > 0``. The unit's
  top map is ``top_lith_indices[k]`` and its base map is
  ``top_lith_indices[k + 1]``.
* ``Facies.sand_shale_facies_markov`` / ``MarkovChainFacies``, for the
  facies sequences.
* ``flood_fill_heap``, the core of legacy ``_flood_fill``, for closure
  placement.

Sections:

* ``units``: facies sequences from legacy Markov chains over many seeds,
  fractions and thicknesses, plus edge cases. Each has the legacy sand units
  as Rust interval ranges ``[top, end)``. Legacy facies index ``i`` is Rust
  interval ``i - 1``, and legacy ``facies[0]`` is water.
* ``population``: legacy statistics over many models, with the sand fraction
  drawn from U(0.05, 0.25), thickness 2 and 40 layers. Recorded per model:
  the number of closure units, the number of multi-layer closure units, and
  the number of sand layers covered by closure units.
* ``placement``: integer depth maps of a two-layer sand unit (top ``t``,
  internal horizon ``m``, base ``b``), with the upper layer pinched out in
  places. Recorded per column: the legacy closure voxel count ``int(cd) -
  t`` with ``cd = min(max(fill, t), b)`` and ``fill = flood_fill_heap(t)``,
  for the whole unit (base ``b``) and for the upper layer alone (base
  ``m``). The legacy closure cube labels samples ``t + 1 .. int(cd)``. Rust's
  first unit sample is its top, so the counts are what match.

Usage (repo root; numpy, scipy, opensimplex installed)::

    python tests/fixtures/generate_closure_units_reference.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datagenerator.Closures import Closures, flood_fill_heap  # noqa: E402
from datagenerator.Horizons import Facies  # noqa: E402

N_LAYERS = 40
N_MODELS = 3000


def legacy_facies(seed: int, frac: float, thick, n: int) -> np.ndarray:
    cfg = types.SimpleNamespace(
        horizon_ss=SeedSequence(seed), sand_layer_pct=frac, sand_layer_thickness=thick
    )
    f = Facies(cfg, n, [], None)
    f.sand_shale_facies_markov()
    return f.facies


def legacy_units(facies: np.ndarray):
    """Rust interval ranges of the sand units legacy computes closures for."""
    stub = types.SimpleNamespace(
        facies=np.asarray(facies, dtype=float),
        onlap_list=[],
        cfg=types.SimpleNamespace(verbose=False),
    )
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        Closures.find_top_lith_horizons(stub)
    tops = [int(t) for t in stub.top_lith_indices]
    fac = stub.top_lith_facies
    units = []
    for ihorizon in range(len(tops) - 1):
        if fac[ihorizon] > 0:
            units.append([tops[ihorizon] - 1, tops[ihorizon + 1] - 1])
    return units


def sand_string(facies) -> str:
    return "".join("1" if f > 0 else "0" for f in facies[1:])


def main():
    units = []
    for seed in range(200):
        rng = default_rng(10_000 + seed)
        frac = float(rng.choice([0.1, 0.25, 0.4, 0.5, 0.6]))
        thick = int(rng.choice([1, 2, 3, 5]))
        n = int(rng.integers(3, 60))
        # MarkovChainFacies needs a < 1: frac / (thick (1 - frac)) < 1.
        if frac / (thick * (1 - frac)) >= 1:
            thick += 2
        f = legacy_facies(seed, frac, thick, n)
        units.append({"sand": sand_string(f), "units": legacy_units(f)})
    edge = ["1", "0", "11", "10", "01", "00", "111", "101", "010", "110", "011",
            "1100", "0110", "0011", "1111", "0000", "10101", "01010", "111011",
            "0111011100", "1110001111", "0001110000"]
    for s in edge:
        f = np.hstack(([-1.0], [float(c) for c in s]))
        units.append({"sand": s, "units": legacy_units(f)})

    rng = default_rng(20260928)
    pop = {"n_units": [], "n_multi": [], "sand_in_units": [], "unit_thickness": []}
    for m in range(N_MODELS):
        frac = float(rng.uniform(0.05, 0.25))
        f = legacy_facies(1_000_000 + m, frac, 2, N_LAYERS)
        u = legacy_units(f)
        pop["n_units"].append(len(u))
        pop["n_multi"].append(sum(1 for a, b in u if b - a > 1))
        pop["sand_in_units"].append(sum(b - a for a, b in u))
        pop["unit_thickness"].extend(b - a for a, b in u)

    # Placement: dome with a pinched-out upper layer.
    ni, nj = 40, 36
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    prng = default_rng(7)
    r2 = (ii - 19.3) ** 2 + (jj - 16.8) ** 2
    t = np.floor(30 + 0.05 * r2 + prng.integers(0, 2, (ni, nj))).astype(np.int64)
    t = np.minimum(t, 90)
    thick_a = np.maximum(0, np.floor(4 + 3 * np.sin(ii / 5.0) - 0.08 * jj)).astype(np.int64)
    m_ = t + thick_a  # upper layer [t, m); pinched out where thick_a == 0
    b = m_ + 6 + (jj % 3)
    fill = flood_fill_heap(t.astype(float))
    cd_unit = np.minimum(np.maximum(fill, t), b)
    cd_layer = np.minimum(np.maximum(fill, t), m_)
    count_unit = cd_unit.astype(int) - t
    count_layer = cd_layer.astype(int) - t
    assert (count_unit >= 0).all() and count_unit.sum() > count_layer.sum() > 0
    assert (thick_a == 0).any()
    placement = {
        "shape": [ni, nj, 128],
        "t": t.ravel().tolist(),
        "m": m_.ravel().tolist(),
        "b": b.ravel().tolist(),
        "count_unit": count_unit.ravel().tolist(),
        "count_layer": count_layer.ravel().tolist(),
    }

    out = {
        "meta": {
            "generator": "tests/fixtures/generate_closure_units_reference.py",
            "legacy": "Closures.find_top_lith_horizons + create_closure_labels_from_depth_maps unit loop; flood_fill_heap",
            "population": {"n_models": N_MODELS, "n_layers": N_LAYERS, "sand_fraction": [0.05, 0.25], "thickness": 2},
        },
        "units": units,
        "population": pop,
        "placement": placement,
    }
    path = Path(__file__).with_name("closure_units_reference.json")
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    print("mean n_units", np.mean(pop["n_units"]), "mean multi", np.mean(pop["n_multi"]),
          "mean thickness", np.mean(pop["unit_thickness"]))
    print("placement voxels unit", int(count_unit.sum()), "layer", int(count_layer.sum()))


if __name__ == "__main__":
    main()
