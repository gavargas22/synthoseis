"""Generate rock-physics parity fixtures from the legacy Python generator.

Calls the *real* legacy code:

* ``Faults.build_faulted_property_geomodels`` (``partial_voxels=False``) for
  the per-layer depth-below-mudline cube (legacy ``faulted_depth``);
* ``Seismic.build_property_models_randomised_depth`` with ``RPMExample`` and
  ``EndMemberMixing`` (inverse velocity and Backus moduli) for Vp / Vs / rho,
  including water, per-layer random depth shifts (recorded), closure fluids and
  ``fix_zero_values_at_base``;
* ``Horizons.create_random_net_over_gross_map`` and the legacy draws
  (``int(rng.uniform(-h, h))``, ``triangular`` half ranges,
  ``rng.integers(3)`` fluids) for the statistical checks.

Legacy runs in float32. numpy's float32 ``z**3`` goes through SVML on AVX-512
hosts (up to 1 ULP from libm ``powf``), so the goldens are generated with
``NPY_DISABLE_CPU_FEATURES`` set to the AVX-512 groups (portable libm
semantics, what Rust ``f32::powf`` calls); the AVX-512 variant is measured
and its deviation recorded under ``avx512``.

Usage (repo root; numpy, scipy, zarr, opensimplex, bruges installed)::

    python tests/fixtures/generate_rock_physics.py   # writes rock_physics.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

AVX512 = "X86_V4 AVX512_ICL AVX512_SPR AVX512_SKX AVX512_CLX AVX512_CNL AVX512F AVX512CD"
if os.environ.get("RP_FIXTURE_CHILD") is None and "NPY_DISABLE_CPU_FEATURES" not in os.environ:
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=AVX512)
    sys.exit(subprocess.call([sys.executable, __file__, *sys.argv[1:]], env=env))

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")


def _stub_numba():
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


_stub_numba()
from datagenerator.Faults import Faults  # noqa: E402
from datagenerator.Seismic import SeismicVolume  # noqa: E402
from rockphysics.rpm_example import RPMExample  # noqa: E402

SHAPE = (8, 6, 48)
DIGI = 4


class _Cfg:
    def __init__(self, shape):
        self.cube_shape = tuple(shape)
        self.partial_voxels = False
        self.variable_shale_ng = False
        self.digi = DIGI
        self.verbose = False
        self.qc_plots = False
        self.include_salt = False
        self.model_qc_volumes = False
        self.project = "example"
        self.model_store = {}
        self.rpm_scaling_factors = {
            "layershiftsamples": 3,
            "RPshiftsamples": 2,
            "shalerho_factor": 1.0,
            "shalevp_factor": 1.0,
            "shalevs_factor": 1.0,
            "sandrho_factor": 1.0,
            "sandvp_factor": 1.0,
            "sandvs_factor": 1.0,
        }

    def write_to_logfile(self, *a, **k):
        pass


def rust_maps(shape):
    """Rust horizon maps `(ni, nj, 4)` (seabed + 3 layer bases), float32 values."""
    ni, nj, nk = shape
    i, j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    z0 = 3.3 + 0.11 * i + 0.07 * j
    z1 = 14.6 + 0.37 * i - 0.21 * j + 1.7 * np.sin(0.9 * i) * np.cos(0.6 * j)
    z2 = 29.2 + 0.19 * i + 0.33 * j
    z3 = np.full_like(z0, nk - 0.5)
    return np.stack([z0, z1, z2, z3], axis=-1).astype(np.float32)


def legacy_depth_and_lith(maps):
    """Real `build_faulted_property_geomodels`; legacy layer i+1 = Rust layer i."""
    ni, nj, nk = SHAPE
    f = Faults.__new__(Faults)
    f.cfg = _Cfg(SHAPE)
    # Legacy horizon 0 is the top of the (water-coded) layer 0: duplicate the seabed.
    f.faulted_depth_maps = np.concatenate([maps[..., :1], maps], axis=-1).astype(np.float32)
    f.faulted_age_volume = np.zeros(SHAPE, np.float32)
    f.vols = types.SimpleNamespace(geologic_age=np.zeros(SHAPE, np.float32))
    for name in ("faulted_lithology", "faulted_net_to_gross", "faulted_depth", "reservoir"):
        setattr(f, name, np.zeros(SHAPE, np.float32))
    f.write_cube_to_disk = lambda *a, **k: None
    # Distinct facies markers identify the legacy layer in the lith cube
    # (none equals 1.0, so no random N/G map is drawn here).
    facies = np.array([-1.0, 0.25, 0.5, 0.75])
    f.build_faulted_property_geomodels(facies)
    lith = np.asarray(f.faulted_lithology)
    layer = np.full(SHAPE, -1, np.int32)
    for n, v in enumerate(facies[1:], start=1):
        layer[lith == np.float32(v)] = n
    return np.asarray(f.faulted_depth, np.float32), layer


def rle(values):
    """Run-length encode a flat integer sequence as {"rle": [v0, n0, v1, n1, ...]}.

    Depth, layers, net-to-gross and properties are piecewise constant along
    each trace (one depth per layer), so this keeps the fixture small."""
    out = []
    for v in values:
        v = int(v)
        if out and out[-2] == v:
            out[-1] += 1
        else:
            out += [v, 1]
    return {"rle": out}


def unrle(obj):
    if isinstance(obj, dict):
        r = obj["rle"]
        return [v for v, n in zip(r[::2], r[1::2]) for _ in range(n)]
    return obj


def bits(a):
    return rle(np.ascontiguousarray(a, np.float32).view(np.uint32).ravel().tolist())


def legacy_properties(depth, layer, ng, oil, gas, mixing, seed=5):
    """Real `build_property_models_randomised_depth` (recording the shifts)."""
    s = SeismicVolume.__new__(SeismicVolume)
    s.cfg = _Cfg(SHAPE)
    s.first_random_lyr = 1  # randomise legacy layers 2 and 3
    s.rng = np.random.default_rng(seed)
    lith = np.where(layer < 0, -1.0, 0.0).astype(np.float32)
    age = np.where(layer < 0, 0, layer).astype(np.float32)
    age[..., -1] = 4.0  # age == max is skipped by the legacy loop -> forward-filled
    s.cfg.model_store = {
        "faulted_depth": depth,
        "oil_closures": oil,
        "gas_closures": gas,
        "faulted_age_volume": age,
    }
    s.faults = types.SimpleNamespace(faulted_lithology=lith, faulted_net_to_gross=ng)
    for name in ("rho", "vp", "vs", "rho_ff", "vp_ff", "vs_ff"):
        setattr(s, name, np.zeros(SHAPE, np.float32))
    shifts = {}
    orig_layer, orig_props = s.get_delta_z_layer, s.get_delta_z_properties

    def rec_layer(z, half, cells):
        v = orig_layer(z, half, cells)
        shifts.setdefault(int(z), {"layer": 0, "props": []})["layer"] = int(v)
        return v

    def rec_props(z, half):
        v = orig_props(z, half)
        shifts.setdefault(int(z), {"layer": 0, "props": []})["props"].append([int(x) for x in v])
        return v

    s.get_delta_z_layer, s.get_delta_z_properties = rec_layer, rec_props
    s.build_property_models_randomised_depth(RPMExample(s.cfg), mixing_method=mixing)
    return (
        {"rho": bits(s.rho), "vp": bits(s.vp), "vs": bits(s.vs)},
        {str(k): v for k, v in shifts.items()},
        age,
    )


def statistics():
    """Legacy random draws for the statistical checks."""
    rng = np.random.default_rng(2024)
    out = {}
    d = np.array([int(rng.uniform(-5, 5)) for _ in range(40000)])
    out["shift5_pmf"] = {str(v): float((d == v).mean()) for v in range(-5, 6)}
    lay = np.array([int(rng.triangular(35, 75, 125)) for _ in range(20000)])
    prop = np.array([int(rng.triangular(5, 11, 20)) for _ in range(20000)])
    out["layer_half_range"] = {"mean": float(lay.mean()), "sd": float(lay.std()),
                               "q10": float(np.quantile(lay, 0.1)), "q90": float(np.quantile(lay, 0.9))}
    out["property_half_range"] = {"mean": float(prop.mean()), "sd": float(prop.std()),
                                  "q10": float(np.quantile(prop, 0.1)), "q90": float(np.quantile(prop, 0.9))}
    fl = rng.integers(3, size=30000)
    out["fluid_freq"] = [float((fl == c).mean()) for c in range(3)]
    # Net-to-gross maps (real Horizons.create_random_net_over_gross_map).
    h = Faults.__new__(Faults)
    h.cfg = _Cfg((32, 32, 8))
    means, sds, mins, maxs, lag1 = [], [], [], [], []
    for seed in range(200):
        h.rng = np.random.default_rng(seed)
        m = h.create_random_net_over_gross_map()
        means.append(m.mean()); sds.append(m.std()); mins.append(m.min()); maxs.append(m.max())
        a = m - m.mean()
        lag1.append(float((a[1:, :] * a[:-1, :]).mean() / max(a.var(), 1e-30)))
    out["ng_maps"] = {
        "shape": [32, 32], "n": 200,
        "mean_of_means": float(np.mean(means)), "sd_of_means": float(np.std(means)),
        "mean_of_sds": float(np.mean(sds)), "sd_of_sds": float(np.std(sds)),
        "min": float(np.min(mins)), "max": float(np.max(maxs)),
        "lag1_autocorr_mean": float(np.mean(lag1)),
    }
    return out


def main():
    ni, nj, nk = SHAPE
    maps = rust_maps(SHAPE)
    depth, layer = legacy_depth_and_lith(maps)

    rng = np.random.default_rng(99)
    ng = np.zeros(SHAPE, np.float32)
    ng_l2 = rng.uniform(0.45, 0.9, (ni, nj)).astype(np.float32)
    ng_l3 = rng.uniform(0.45, 0.9, (ni, nj)).astype(np.float32)
    ng_l3[::3, ::2] = 0.0  # some pure-shale columns in a sand layer
    ng[layer == 2] = np.broadcast_to(ng_l2[..., None], SHAPE)[layer == 2]
    ng[layer == 3] = np.broadcast_to(ng_l3[..., None], SHAPE)[layer == 3]
    kk = np.broadcast_to(np.arange(nk), SHAPE)
    ii = np.broadcast_to(np.arange(ni)[:, None, None], SHAPE)
    jj = np.broadcast_to(np.arange(nj)[None, :, None], SHAPE)
    oil = ((layer == 2) & (ii < 4) & (kk < 26)).astype(np.float32)
    gas = ((layer == 3) & (jj >= 3) & (kk < 40)).astype(np.float32)

    props = {}
    for mixing in ("inv_vel", "backus"):
        p, shifts, age = legacy_properties(depth, layer, ng, oil, gas, mixing)
        props[mixing] = {"props": p, "shifts": shifts}

    fixture = {
        "meta": {
            "shape": list(SHAPE), "digi": DIGI, "first_random_lyr": 1,
            "layershiftsamples": 3, "RPshiftsamples": 2,
            "numpy": np.__version__,
            "npy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
            "note": "legacy layer n (1..3) = Rust layer n-1; age == 4 at the last sample is unfilled",
        },
        "maps_bits": np.ascontiguousarray(maps, np.float32).view(np.uint32).ravel().tolist(),
        "legacy_depth_bits": bits(depth),
        "legacy_layer": rle(layer.ravel()),
        "ng_bits": bits(ng),
        "oil": rle(oil.astype(np.uint8).ravel()),
        "gas": rle(gas.astype(np.uint8).ravel()),
        "age": rle(age.astype(np.int32).ravel()),
        "properties": props,
        "statistics": statistics(),
    }
    out = Path(__file__).with_name("rock_physics.json")
    out.write_text(json.dumps(fixture, separators=(",", ":")))
    if os.environ.get("RP_AVX512_PROBE") is None:
        # Deviation of the AVX-512 (SVML) float32 power from the golden.
        env = dict(os.environ, RP_AVX512_PROBE="1", RP_FIXTURE_CHILD="1")
        env.pop("NPY_DISABLE_CPU_FEATURES", None)
        probe = subprocess.run([sys.executable, __file__, "--probe"], env=env,
                               capture_output=True, text=True, check=True)
        fixture["avx512"] = json.loads(probe.stdout)
    out.write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size} bytes); avx512={fixture.get('avx512')}")


def probe():
    """Run the property builder with numpy's native (AVX-512) dispatch and
    report the ULP deviation from the portable golden."""
    maps = rust_maps(SHAPE)
    depth, layer = legacy_depth_and_lith(maps)
    gold = json.loads(Path(__file__).with_name("rock_physics.json").read_text()) \
        if Path(__file__).with_name("rock_physics.json").exists() else None
    ng = np.array(unrle(gold["ng_bits"]), np.uint32).view(np.float32).reshape(SHAPE) if gold else None
    from numpy._core._multiarray_umath import __cpu_features__ as feats
    out = {"avx512_features": sorted(k for k, v in feats.items() if v and "AVX512" in k)}
    if gold is None:
        print(json.dumps(out)); return
    oil = np.array(unrle(gold["oil"]), np.float32).reshape(SHAPE)
    gas = np.array(unrle(gold["gas"]), np.float32).reshape(SHAPE)
    p, _, _ = legacy_properties(depth, layer, ng, oil, gas, "inv_vel")
    for key in ("rho", "vp", "vs"):
        a = np.array(unrle(p[key]), np.int64)
        b = np.array(unrle(gold["properties"]["inv_vel"]["props"][key]), np.int64)
        out[key] = {"mismatch_frac": float((a != b).mean()), "max_ulp": int(np.abs(a - b).max())}
    print(json.dumps(out))


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        main()
