"""Reference statistics for the Rust seismic-noise port from the legacy code.

Runs the *real* legacy ``SeismicVolume.add_weighted_noise`` (and its
``noise_3d``) from ``datagenerator/Seismic.py`` on the raw reflectivity that
the Rust pipeline produces (``noise_demo`` example output: ``rfc_{5,15,25}.f32``
and ``seabed.f64``) for many legacy seeds, and records per-angle noise
statistics: mean, std, excess kurtosis, a banded amplitude spectrum and the
inter-angle correlation, plus the legacy ``data_std``. The Rust test
``rust/synthoseis-core/tests/noise_pipeline.rs`` regenerates the same
reflectivity and compares its deterministic noise against these numbers.

Two legacy variants are recorded:

* ``legacy_degrees``: legacy exactly as written (``math.cos(ang)`` with the
  angle in degrees), compared with Rust ``legacy_angle_weights = true``;
* ``radians``: legacy with the documented radian fix
  (``tests/test_seismic_noise.py``), compared with the Rust default.

Usage (repo root; numpy + scipy + zarr + xarray + tqdm installed)::

    cargo run --release -p synthoseis-core --example noise_demo -- /tmp/noise_demo
    python tests/fixtures/generate_seismic_noise.py /tmp/noise_demo [--seeds 64]
        # writes tests/fixtures/seismic_noise.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_seismic_filters import SeismicVolume  # noqa: E402  (stubs numba etc.)

REPO = Path(__file__).resolve().parents[2]
ANGLES = (5.0, 15.0, 25.0)
N_BANDS = 6


class _Cfg:
    """Attributes read by ``add_weighted_noise``."""

    def __init__(self, sn_db, digi=4.0):
        self.verbose = False
        self.sn_db = sn_db
        self.digi = digi
        self.model_qc_volumes = False

    @staticmethod
    def create_array(name, shape, dtype="float32"):
        return np.zeros(shape, dtype=dtype)


def load_demo(demo: Path):
    meta = json.loads((demo / "meta.json").read_text())
    ni, nj, nk = meta["shape"]
    rfc = np.stack(
        [np.fromfile(demo / f"rfc_{a:g}.f32", "<f4").reshape(ni, nj, nk) for a in ANGLES]
    )[..., : nk - 1]  # legacy rfc_raw has nk - 1 Zoeppritz samples
    seabed = np.fromfile(demo / "seabed.f64", "<f8").reshape(ni, nj)
    return meta, np.ascontiguousarray(rfc, dtype=np.float32), seabed


@contextlib.contextmanager
def radian_weights():
    """Temporarily apply the radian fix (``add_weighted_noise`` does
    ``from math import sin, cos`` at call time)."""
    cos, sin = math.cos, math.sin
    math.cos = lambda a: cos(math.radians(a))
    math.sin = lambda a: sin(math.radians(a))
    try:
        yield
    finally:
        math.cos, math.sin = cos, sin


def legacy_add_noise(rfc, seabed, sn_db, seed, digi=4.0, radians=False):
    """Run legacy ``add_weighted_noise``; returns ``(noise, rfc_noise_added)``
    with ``noise = rfc_noise_added - rfc_raw`` (f64)."""
    sv = object.__new__(SeismicVolume)
    sv.cfg = _Cfg(sn_db, digi)
    sv.rng = np.random.default_rng(seed)
    sv.angles = ANGLES
    sv.rfc_raw = rfc.copy()
    sv.rfc_noise_added = np.zeros_like(rfc)
    depth_maps = (seabed * digi)[..., None]  # legacy depth maps are in digi units
    with radian_weights() if radians else contextlib.nullcontext():
        sv.add_weighted_noise(depth_maps)
    noisy = np.asarray(sv.rfc_noise_added)
    return noisy.astype(np.float64) - rfc.astype(np.float64), noisy


def legacy_data_std(rfc, seabed, digi=4.0):
    """Legacy ``data_std`` (lines copied from ``add_weighted_noise``)."""
    from datagenerator.util import mute_above_seafloor

    wb = seabed * digi
    thr = wb / (digi + 15) * digi
    norm = rfc[1]
    return float(norm[mute_above_seafloor(thr, np.ones(norm.shape, "float")) != 0.0].std())


def field_stats(n):
    """Statistics of one noise field ``(ni, nj, nk - 1)``."""
    x = n.astype(np.float64)
    mean = float(x.mean())
    std = float(x.std())
    kurt = float(((x - mean) ** 4).mean() / std**4 - 3.0)
    traces = x.reshape(-1, x.shape[-1])
    amp = np.abs(np.fft.rfft(traces, axis=-1)).mean(axis=0) / (std * math.sqrt(x.shape[-1]))
    bands = [float(b.mean()) for b in np.array_split(amp[1:], N_BANDS)]
    return {"mean": mean, "std": std, "kurtosis": kurt, "bands": bands}


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


def summarize(samples):
    """Mean and seed-to-seed std of every statistic."""
    out = {}
    for key in samples[0]:
        v = np.array([s[key] for s in samples], dtype=np.float64)
        out[key] = {"mean": v.mean(axis=0).tolist(), "sd": v.std(axis=0, ddof=1).tolist()}
    return out


def reference(rfc, seabed, sn_db, seeds, radians):
    per_angle = {a: [] for a in ANGLES}
    corrs = []
    for s in seeds:
        noise, _ = legacy_add_noise(rfc, seabed, sn_db, s, radians=radians)
        for x, a in enumerate(ANGLES):
            per_angle[a].append(field_stats(noise[x]))
        corrs.append({"corr_5_25": corr(noise[0], noise[2]), "corr_5_15": corr(noise[0], noise[1])})
    return {
        "angles": {str(a): summarize(v) for a, v in per_angle.items()},
        "inter_angle": summarize(corrs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", type=Path)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--out", type=Path, default=REPO / "tests/fixtures/seismic_noise.json")
    a = ap.parse_args()
    meta, rfc, seabed = load_demo(a.demo)
    sn_db = meta["snr_db"]
    seeds = list(range(1, a.seeds + 1))
    fixture = {
        "meta": {
            "shape": meta["shape"],
            "seed": meta["seed"],
            "faults": 0,
            "snr_db": sn_db,
            "digi_ms": 4.0,
            "angles": list(ANGLES),
            "legacy_seeds": a.seeds,
            "n_bands": N_BANDS,
            "std_ratio": math.sqrt(10 ** (sn_db / 10.0)),
            "generator": "tests/fixtures/generate_seismic_noise.py",
            "numpy": np.__version__,
        },
        "legacy_data_std": legacy_data_std(rfc, seabed),
        "legacy_degrees": reference(rfc, seabed, sn_db, seeds, radians=False),
        "radians": reference(rfc, seabed, sn_db, seeds, radians=True),
    }
    a.out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {a.out}: data_std={fixture['legacy_data_std']:.9e}")


if __name__ == "__main__":
    main()
