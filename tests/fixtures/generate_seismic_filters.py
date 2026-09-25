"""Generate seismic-filter parity fixtures from the legacy Python generator.

Calls the *real* legacy code in ``datagenerator/Seismic.py``:

* ``derive_butterworth_bandpass`` (``scipy.signal.butter(..., output="ba")``)
  and ``scipy.signal.lfilter_zi`` for the filter coefficients,
* ``SeismicVolume.apply_bandlimits`` (``apply_butterworth_bandpass`` =
  ``filtfilt(method="pad")`` on every trace of a 4-D float32 array),
* ``SeismicVolume.apply_lateral_filter`` (``uniform_filter(size=(0, n, n, 0))``),
* ``SeismicVolume.apply_cumsum`` (float32 cumsum + 2-100 Hz bandpass),

on small deterministic float32 volumes, with a stub ``cfg`` holding only the
attributes these methods read. ``numba`` and ``rockphysics.RockPropertyModels``
are stubbed only so the module imports without those unrelated packages.

Usage (from repo root, numpy + scipy + tqdm + zarr + xarray installed)::

    python tests/fixtures/generate_seismic_filters.py   # writes seismic_filters.json
    python tests/fixtures/generate_seismic_filters.py --filter-raw IN.f32 NI NJ NK \
        --low 4 --high 30 --lateral 3 --out OUT.f32     # filter a raw Rust stack
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import lfilter_zi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")


def _stub_unrelated_imports():
    if "numba" not in sys.modules:
        try:
            import numba  # noqa: F401
        except ImportError:
            nb = types.ModuleType("numba")

            def njit(*a, **k):
                if len(a) == 1 and callable(a[0]) and not k:
                    return a[0]
                return lambda f: f

            nb.njit = nb.jit = njit
            nb.prange = range
            sys.modules["numba"] = nb
    try:
        import rockphysics.RockPropertyModels  # noqa: F401
    except ImportError:
        rpm = types.ModuleType("rockphysics.RockPropertyModels")
        rpm.select_rpm = rpm.RockProperties = rpm.EndMemberMixing = None
        sys.modules["rockphysics.RockPropertyModels"] = rpm


_stub_unrelated_imports()
from datagenerator.Seismic import SeismicVolume, derive_butterworth_bandpass  # noqa: E402


class _Cfg:
    """Attributes read by apply_bandlimits / apply_lateral_filter / apply_cumsum."""

    def __init__(self, shape, low, high, order, lateral, digi=4.0):
        self.verbose = False
        self.digi = digi
        self.infill_factor = 10
        self.pad_samples = 0
        self.cube_shape = tuple(shape)
        self.lowfreq = low
        self.highfreq = high
        self.order = order
        self.lateral_filter_size = lateral


class _Seis:
    apply_bandlimits = SeismicVolume.apply_bandlimits
    apply_lateral_filter = SeismicVolume.apply_lateral_filter
    apply_cumsum = SeismicVolume.apply_cumsum

    def __init__(self, cfg):
        self.cfg = cfg


def legacy_filter(stack3d, low, high, order, lateral, digi=4.0):
    """Legacy postprocess_rfc_cubes filtering of one (ni, nj, nk) float32 stack."""
    cfg = _Cfg(stack3d.shape, low, high, order, lateral, digi)
    s = _Seis(cfg)
    data = stack3d[np.newaxis].astype(np.float32).copy()
    band = s.apply_bandlimits(data)
    if lateral > 1:
        band = s.apply_lateral_filter(band)
    return band[0]


def test_volume(shape, seed):
    rng = np.random.default_rng(seed)
    ni, nj, nk = shape
    v = rng.standard_normal(shape).astype(np.float32) * np.float32(0.05)
    k = np.arange(nk)
    for i in range(ni):
        for j in range(nj):
            for c in rng.integers(4, nk - 4, 3):
                v[i, j, c] += np.float32(rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0))
            v[i, j] += np.float32(0.3) * np.sin(2 * np.pi * (k / (8.0 + i + 0.5 * j))).astype(np.float32)
    return v.astype(np.float32)


def flat(a):
    """float64 values as floats; float32 values as their shortest f32 decimal."""
    a = np.asarray(a).ravel()
    if a.dtype == np.float32:
        out = []
        for x in a:
            v = float(np.format_float_positional(x, unique=True, trim="-"))
            assert np.float32(v) == x  # f64 parse -> f32 round-trips exactly
            out.append(v)
        return out
    return [float(x) for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("seismic_filters.json")))
    ap.add_argument("--filter-raw", nargs=4, metavar=("IN", "NI", "NJ", "NK"))
    ap.add_argument("--low", type=float, default=4.0)
    ap.add_argument("--high", type=float, default=30.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--lateral", type=int, default=3)
    ap.add_argument("--digi", type=float, default=4.0)
    args = ap.parse_args()

    if args.filter_raw:
        path, ni, nj, nk = args.filter_raw
        shape = (int(ni), int(nj), int(nk))
        stack = np.fromfile(path, dtype="<f4").reshape(shape)
        out = legacy_filter(stack, args.low, args.high, args.order, args.lateral, args.digi)
        out.astype("<f4").tofile(args.out)
        print(args.out)
        return

    designs = []
    for low, high, digi, order in [
        (3.0, 20.0, 4.0, 4), (6.0, 35.0, 4.0, 4), (4.5, 27.3, 4.0, 4), (2.0, 100.0, 4.0, 4),
        (4.0, 90.0, 4.0, 4), (5.0, 30.0, 2.0, 4), (3.7, 24.1, 4.0, 2), (3.7, 24.1, 4.0, 3),
        (3.7, 24.1, 4.0, 6), (10.0, 60.0, 1.0, 1),
    ]:
        dt = digi / 1000.0
        b, a = derive_butterworth_bandpass(low, high, dt * 1000.0, order=order)
        designs.append({"low": low, "high": high, "digi": digi, "order": order,
                        "b": flat(b), "a": flat(a), "zi": flat(lfilter_zi(b, a))})

    shape = (3, 2, 64)
    vol = test_volume(shape, 7)
    bandpass = []
    for low, high, order in [(3.0, 20.0, 4), (6.0, 35.0, 3)]:
        cfg = _Cfg(shape, low, high, order, 1)
        out = _Seis(cfg).apply_bandlimits(vol[np.newaxis].copy())[0]
        bandpass.append({"low": low, "high": high, "order": order, "output": flat(out)})

    lateral = []
    for lshape, size, seed in [((5, 4, 3), 3, 1), ((5, 4, 3), 5, 2), ((4, 3, 2), 4, 3),
                               ((2, 1, 3), 5, 4)]:
        v = np.random.default_rng(seed).standard_normal(lshape).astype(np.float32)
        cfg = _Cfg(lshape, 3.0, 20.0, 4, size)
        out = _Seis(cfg).apply_lateral_filter(v[np.newaxis].copy())[0]
        lateral.append({"shape": list(lshape), "size": size, "input": flat(v), "output": flat(out)})

    cfg = _Cfg(shape, 3.0, 20.0, 4, 1)
    cumsum = _Seis(cfg).apply_cumsum(vol[np.newaxis].copy())[0]
    chain = legacy_filter(vol, 4.5, 27.3, 4, 3)

    fixture = {
        "meta": {"numpy": np.__version__, "scipy": scipy.__version__,
                 "generator": "tests/fixtures/generate_seismic_filters.py"},
        "designs": designs,
        "volume": {"shape": list(shape), "input": flat(vol)},
        "bandpass": bandpass,
        "cumsum_order4": flat(cumsum),
        "chain_4p5_27p3_o4_lat3": flat(chain),
        "lateral": lateral,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
    print(args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
