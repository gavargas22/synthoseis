"""Generate the lithology (sand/shale) reference fixture from the legacy code.

Calls the *real* legacy classes ``datagenerator.Horizons.Facies`` (via
``sand_shale_facies_markov``, as ``create_facies_array`` does) and
``MarkovChainFacies``.

* ``chains``: for several (seed, sand fraction, sand thickness, layers) the
  legacy facies, plus the replayed draws: the initial ``rng.choice(2, 1)``
  state and the one ``rng.random()`` double that each ``rng.choice(p=row)``
  consumes. The generator asserts that replaying ``cdf = cumsum(row) /
  sum(row); state = searchsorted(cdf, u, "right")`` reproduces legacy, so
  Rust can be checked bit for bit on the same draws.
* ``population``: legacy statistics over many models. Per model, the sand
  fraction is legacy ``Parameters``' ``rng.uniform(sand_layer_fraction.min,
  .max)`` (0.05, 0.25 in config/example.json), ``sand_layer_thickness`` is 2,
  and there are 60 layers. Recorded: the drawn fractions, the per-model sand
  proportion and the sand / shale run-length histograms.

Usage (repo root; numpy, scipy, opensimplex installed)::

    python tests/fixtures/generate_lithology_reference.py
"""

from __future__ import annotations

import json
import sys
import types
from itertools import groupby
from pathlib import Path

import numpy as np
from numpy.random import SeedSequence, default_rng

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datagenerator.Horizons import Facies, MarkovChainFacies  # noqa: E402

SAND_FRACTION = (0.05, 0.25)  # config/example.json sand_layer_fraction
SAND_THICKNESS = 2  # config/example.json sand_layer_thickness
N_LAYERS = 60
N_MODELS = 3000
MAX_RUN = 12  # histogram bins 1..MAX_RUN-1 and a >= MAX_RUN bin


def legacy_facies(seed: int, frac: float, thick, n: int) -> np.ndarray:
    cfg = types.SimpleNamespace(
        horizon_ss=SeedSequence(seed), sand_layer_pct=frac, sand_layer_thickness=thick
    )
    f = Facies(cfg, n, [], None)
    f.sand_shale_facies_markov()
    return f.facies


def replay(seed: int, frac: float, thick, n: int):
    rng = default_rng(SeedSequence(seed).spawn(1)[0])
    init = int(rng.choice(2, 1)[0])
    u = [float(rng.random()) for _ in range(n)]
    mk = MarkovChainFacies(frac, thick, (0, 1))
    t = mk.transition
    s, out = init, []
    for x in u:
        row = t[s]
        cdf = np.cumsum(row)
        cdf /= cdf[-1]
        s = int(np.searchsorted(cdf, x, side="right"))
        out.append(s)
    return init, u, out, t


def bits(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


def runs(states, value):
    return [len(list(g)) for k, g in groupby(states) if k == value]


def main():
    chains = []
    for seed, frac, thick, n in [
        (0, 0.05, 2, 200),
        (1, 0.15, 2, 200),
        (2, 0.25, 2, 200),
        (3, 0.2, 3, 200),
        (4, 0.4, 2, 200),
        (5, 0.6, 2, 200),
        (6, 0.1, 1, 200),
        (7, 0.3, 5, 200),
        (8, 0.6666, 2, 200),
        (9, 0.12345678901234, 2, 200),
    ]:
        legacy = legacy_facies(seed, frac, thick, n)
        init, u, out, t = replay(seed, frac, thick, n)
        assert legacy[0] == -1.0 and len(legacy) == n + 1
        assert [int(v) for v in legacy[1:]] == out, (seed, frac, thick)
        chains.append(
            {
                "seed": seed,
                "sand_fraction": frac,
                "sand_thickness": float(thick),
                "initial": init,
                "u": u,
                "transition": t.tolist(),
                # IEEE-754 bits: exact regardless of the JSON float parser.
                "sand_fraction_bits": bits(frac),
                "sand_thickness_bits": bits(float(thick)),
                "u_bits": [bits(x) for x in u],
                "transition_bits": [[bits(x) for x in row] for row in t.tolist()],
                "facies": out,
            }
        )

    fractions, proportions, sand_counts = [], [], []
    sand_hist = [0] * MAX_RUN
    shale_hist = [0] * MAX_RUN
    for m in range(N_MODELS):
        rng = default_rng(10_000 + m)
        frac = float(rng.uniform(low=SAND_FRACTION[0], high=SAND_FRACTION[1]))
        fac = legacy_facies(20_000 + m, frac, SAND_THICKNESS, N_LAYERS)[1:].astype(int)
        fractions.append(frac)
        proportions.append(float(fac.mean()))
        sand_counts.append(int(fac.sum()))
        for r in runs(fac, 1):
            sand_hist[min(r, MAX_RUN) - 1] += 1
        for r in runs(fac, 0):
            shale_hist[min(r, MAX_RUN) - 1] += 1
    fix = {
        "sand_layer_fraction": SAND_FRACTION,
        "sand_layer_thickness": SAND_THICKNESS,
        "chains": chains,
        "population": {
            "models": N_MODELS,
            "layers": N_LAYERS,
            "fractions": fractions,
            "sand_proportion": proportions,
            "sand_count": sand_counts,
            "sand_run_hist": sand_hist,
            "shale_run_hist": shale_hist,
            "max_run_bin": MAX_RUN,
        },
    }
    out = ROOT / "tests" / "fixtures" / "lithology_reference.json"
    out.write_text(json.dumps(fix, separators=(",", ":")))
    print(f"wrote {out}: {len(chains)} chains, {N_MODELS} models")
    print(f"legacy mean sand proportion {np.mean(proportions):.4f}, mean fraction {np.mean(fractions):.4f}")
    print("sand runs", sand_hist, "shale runs", shale_hist)


if __name__ == "__main__":
    main()
