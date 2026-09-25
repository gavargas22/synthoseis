"""Generate fault-port parity fixtures from the legacy Python generator.

Runs the *real* ``datagenerator.Faults.Faults.build_faults`` +
``apply_xyz_displacement`` code path on a small cube with a stubbed
``Parameters`` / ``Geomodel`` (no zarr store, no QC volumes) and records:

* the explicit fault parameters actually used (random draws are recorded, so
  the Rust port can be fed identical parameters and bypass its own sampler),
* the final binary fault mask (``fault_planes`` after thresholding),
* the faulted geologic-age cube reduced to displaced horizon depths
  (``np.interp(h, faulted_age[i, j, :], arange(nk))`` — the core of
  ``Faults.improve_depth_maps_post_faulting``).

Hockey-stick drag (throw >= 0.85 * 35 samples) is excluded on purpose: the
Rust port defers it (see docs/faults-port.md), so throws are redrawn below
29 samples.

Usage (from repo root, with numpy/scipy/matplotlib/scikit-image/tqdm/zarr/xarray/
opensimplex installed)::

    python tests/fixtures/generate_fault_cubes.py            # writes fault_cubes.json
    python tests/fixtures/generate_fault_cubes.py --dump DIR # also dumps .npy volumes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ.setdefault("MPLBACKEND", "Agg")

from datagenerator.Faults import Faults  # noqa: E402

INFILL = 10
HOCKEY_MIN_THROW = 0.85 * 35.0


class _Cfg:
    """Minimal stand-in for ``datagenerator.Parameters`` used by build_faults."""

    def __init__(self, shape, n_faults, work):
        self.cube_shape = tuple(shape)
        self.pad_samples = 0
        self.infill_factor = INFILL
        self.number_faults = n_faults
        self.verbose = False
        self.model_qc_volumes = False
        self.qc_plots = False
        self.include_salt = False
        self.include_channels = False
        self.work_subfolder = work
        self.fault_ss = np.random.SeedSequence(0)
        self.fmode = "explicit"
        self.fnoise = "none"
        self.low_fault_throw = 5.0 * INFILL
        self.high_fault_throw = 35.0 * INFILL

    def create_array(self, name, shape, dtype="float32"):
        return np.zeros(shape, dtype=dtype)

    def remove_array(self, name):
        pass

    def write_to_logfile(self, *args, **kwargs):
        pass


class _Vols:
    def __init__(self, age):
        self.geologic_age = age
        self.onlap_segments = np.zeros_like(age)


class _RecordingRng:
    """Wrap a numpy Generator and record every draw (so Rust can replay them)."""

    def __init__(self, seed):
        self.g = np.random.default_rng(seed)
        self.log = []

    def uniform(self, low=0.0, high=1.0, size=None):
        v = self.g.uniform(low, high, size)
        self.log.append(("uniform", low, high, v))
        return v

    def triangular(self, left, mode, right, size=None):
        v = self.g.triangular(left, mode, right, size)
        self.log.append(("triangular", v))
        return v

    def choice(self, a, size=None, *args, **kwargs):
        v = self.g.choice(a, size, *args, **kwargs)
        self.log.append(("choice", v))
        return v


def age_cube(shape, age_cfg):
    ni, nj, nk = shape
    i = np.arange(ni, dtype=float)[:, None, None]
    j = np.arange(nj, dtype=float)[None, :, None]
    k = np.arange(nk, dtype=float)[None, None, :]
    spacing = age_cfg["spacing_base"] + age_cfg["spacing_di"] * i + age_cfg["spacing_dj"] * j
    return ((k - age_cfg["z_top"]) / spacing).astype("float32")


def draw_fault_params(obj, rng, n_faults):
    """Python's random-mode draw, redrawn until no hockey-stick throws."""
    obj.rng = rng
    while True:
        fp = obj._fault_params_random()
        if np.all(fp["throw"] / INFILL < HOCKEY_MIN_THROW):
            return fp


def run_case(name, shape, n_faults, seed, age_cfg, wb_const, dump=None, stride=6):
    work = tempfile.mkdtemp(prefix=f"faults_{name}_")
    cfg = _Cfg(shape, n_faults, work)
    age = age_cube(shape, age_cfg)
    depth_maps = np.full((shape[0], shape[1], 2), wb_const * INFILL, dtype=float)
    depth_maps[..., 1] = (shape[2] - 1) * INFILL
    obj = Faults(cfg, depth_maps, np.array([]), _Vols(age), np.array([]), np.array([]))

    rng = _RecordingRng(seed)
    fp = draw_fault_params(obj, rng, n_faults)
    rng.log.clear()

    # Record the fault-centre choice (get_middle_z's rng.choice over the
    # middle-of-surface candidates) so Rust can replay it exactly.
    centres = {}
    orig_centre = obj.get_fault_centre

    def recording_centre(ellipsoid, wb, seg, index):
        z = orig_centre(ellipsoid, wb, seg, index)
        centres[index] = [int(v) for v in z] if np.size(z) else None
        return z

    obj.get_fault_centre = recording_centre
    obj.rng = rng

    # Capture scipy.ndimage.rotate outputs inside xyz_dis (drag field, then the
    # lateral displacement field) to record the rotated-argmax half-pixel
    # offset. When the four centre pixels tie analytically (rotation by a
    # multiple of 90 deg) spline round-off picks the winner; Rust can replay it.
    import scipy.ndimage.interpolation as _interp

    captured = []
    _orig_rotate = _interp.rotate

    def _capturing_rotate(*a, **k):
        r = _orig_rotate(*a, **k)
        captured.append(k.get("output"))
        return r

    _interp.rotate = _capturing_rotate
    try:
        obj.build_faults(fp)
    finally:
        _interp.rotate = _orig_rotate
    offsets = []
    for arr in captured[1::2]:
        r, c = np.unravel_index(np.argmax(arr), arr.shape)
        offsets.append([float(r - (arr.shape[0] - 1) / 2), float(c - (arr.shape[1] - 1) / 2)])
    offsets = iter(offsets)
    faulted_age = obj.apply_xyz_displacement(age).astype("float32")

    # Replay the per-fault xyz_dis draws: uniform(shear) [+ triangular], then
    # (only if a centre was found) sigma, p, coef.
    draws = [e for e in rng.log if e[0] == "uniform"]
    faults = []
    it = iter(draws)
    for i in range(n_faults):
        next(it)  # shear_zone_width (unused by the displacement field)
        entry = {
            k: float(fp[k][i])
            for k in ("a", "b", "c", "x0", "y0", "z0", "throw", "tilt_pct")
        }
        entry["center"] = centres.get(i)
        if entry["center"] is not None:
            entry["py_lateral_offset"] = next(offsets)
            entry["sigma"] = float(next(it)[3])
            entry["p"] = float(next(it)[3])
            entry["coef"] = float(next(it)[3])
        faults.append(entry)

    mask = (obj.fault_planes[:] > 0.5).astype(np.uint8)
    nk = shape[2]
    origtime = np.arange(nk, dtype=float)
    h_lo = int(np.ceil(age.min())) + 1
    h_hi = int(np.floor(age.max())) - 1
    horizons = list(range(h_lo, h_hi + 1))
    ii_s = list(range(0, shape[0], stride))
    jj_s = list(range(0, shape[1], stride))
    depths = np.zeros((len(ii_s), len(jj_s), len(horizons)))
    for a, ii in enumerate(ii_s):
        for b, jj in enumerate(jj_s):
            for n, h in enumerate(horizons):
                depths[a, b, n] = np.interp(h, faulted_age[ii, jj, :], origtime)

    if dump:
        d = Path(dump)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{name}_mask.npy", mask)
        np.save(d / f"{name}_faulted_age.npy", faulted_age)
        np.save(d / f"{name}_age.npy", age)
        np.save(d / f"{name}_lookup.npy", obj.displacement_vectors[:])
        np.save(d / f"{name}_horizon_depths.npy", depths)

    return {
        "name": name,
        "shape": list(shape),
        "infill_factor": INFILL,
        "seed": seed,
        "wb_const": wb_const,
        "age": age_cfg,
        "horizons": horizons,
        "horizon_stride": stride,
        "faults": faults,
        "expected": {
            "fault_mask_runs": mask_runs(mask.ravel()),
            "n_fault_voxels": int(mask.sum()),
            "horizon_depths": [round(float(v), 4) for v in depths.ravel()],
            "faulted_age_sum": float(faulted_age.astype(float).sum()),
        },
    }


def mask_runs(arr):
    """Alternating run lengths over a flat C-order 0/1 array, starting with 0s.

    ``[3, 2, 5]`` means three 0s, two 1s, five 0s. A leading run may be 0.
    """
    out = []
    cur = 0
    run = 0
    for v in arr:
        v = 1 if v else 0
        if v == cur:
            run += 1
        else:
            out.append(run)
            cur, run = v, 1
    out.append(run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", default=str(Path(__file__).with_name("fault_cubes.json")))
    ap.add_argument("--sweep", type=int, default=0,
                    help="instead of the committed cases, run N random seeds (48x48x64, 1-3 faults)")
    args = ap.parse_args()
    age_cfg = {"z_top": 2.0, "spacing_base": 6.0, "spacing_di": 0.03, "spacing_dj": -0.02}
    if args.sweep:
        import contextlib
        import io
        cases = []
        for seed in range(args.sweep):
            with contextlib.redirect_stdout(io.StringIO()):
                c = run_case(f"sweep_{seed}", (48, 48, 64), 1 + seed % 3, 1000 + seed,
                             age_cfg, 4.0, None, stride=1)
            cases.append(c)
            print(c["name"], "voxels", c["expected"]["n_fault_voxels"], flush=True)
        with open(args.out, "w") as f:
            json.dump({"cases": cases}, f, separators=(",", ":"))
        return
    cases = [
        run_case("one_fault", (32, 32, 48), 1, 11, age_cfg, 4.0, args.dump),
        run_case("two_faults", (32, 32, 48), 2, 29, age_cfg, 4.0, args.dump),
        run_case("two_faults_b", (32, 32, 48), 2, 58, age_cfg, 4.0, args.dump),
        run_case("three_faults_skips", (32, 32, 48), 3, 5, age_cfg, 4.0, args.dump),
    ]
    with open(args.out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    for c in cases:
        print(c["name"], "fault voxels:", c["expected"]["n_fault_voxels"],
              "centres:", [f["center"] for f in c["faults"]])


if __name__ == "__main__":
    main()
