#!/usr/bin/env python3
"""End-to-end angle-stack fixture from the REAL legacy generator.

Runs the legacy Python pipeline (``datagenerator``: horizons, facies,
geomodels, faults, closures, ``SeismicVolume.build_elastic_properties``) on
small full models with fixed seeds, then the legacy seismic chain on the
legacy elastic cubes:

1. ``SeismicVolume.create_rfc_volumes`` (the Numba kernel
   ``datagenerator/zoeppritz_kernel.py``) -> ``rfc_raw`` (angles, ni, nj, nk-1);
2. ``SeismicVolume.postprocess_rfc_cubes(rfc_raw, "noise_free")``, i.e. the
   ``model_qc_volumes`` noise-free branch of ``build_seismic_volumes``:
   ``apply_bandlimits`` (Butterworth ``filtfilt``) -> ``apply_lateral_filter``
   -> ``apply_cumsum``. ``write_final_cubes_to_disk`` is intercepted so the
   stacks are captured *before* legacy output scaling (``_scale_seismic``
   and the rpm near/mid/far factors, which the Rust port does not do).

Noise (``add_weighted_noise``) is skipped: it draws from numpy's stream and the
Rust port is only statistically equivalent (see ``docs/filters-port.md``).

Rust cannot regenerate the legacy *geology* (its toy geometry, lithology and
rock physics draw from its own RNG), so the fixture stores the legacy elastic
cubes (Vp, Vs, rho; float32, zlib) and the Rust test feeds them through its
production reflectivity -> bandpass -> lateral chain. Legacy outputs are pinned
by FNV-1a 64 hashes of the full float32 cubes plus a few sampled traces.

It also replays the ORIGINAL upstream reflectivity path
(``Seismic.RFC.zoeppritz_reflectivity`` driven exactly like
``create_rfc_volumes`` in upstream commit 6e4c2e30: per trace, complex64
storage, real part) on the same cubes, and the textbook (``d``) form, to
document which legacy Zoeppritz path(s) carry the ``det`` typo.

Usage (repo root, locked ``uv sync --frozen`` environment)::

    uv run --frozen python tests/fixtures/generate_angle_stack_e2e.py

Writes ``tests/fixtures/angle_stack_e2e.json`` and
``tests/fixtures/angle_stack_e2e_seed<N>.bin`` (zlib; see ``BLOB_LAYOUT``).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")
OUT = REPO / "tests" / "fixtures"

# (seed, test_mode size, min_closure_voxels_simple). Seeds chosen so both
# legacy lateral filter sizes > 1 are covered (seed 25: n = 5 with oil and gas
# closures; seed 3: n = 3, brine only). min_closure_voxels_simple is lowered
# from 500 so a 16 x 16 cube keeps its closures (legacy config knob).
CASES = [(25, 16, 20), (3, 16, 20)]
NZ = 500  # cube_shape[2]; legacy needs ~500 samples (onlap LUT), + pad 10
SAMPLE_COLUMNS = [(0, 0), (8, 7), (15, 15)]  # + one gas-closure column

BLOB_LAYOUT = (
    "zlib( vp | vs | rho | sample_rfc_raw | sample_stack | sample_cumsum ), "
    "little-endian float32; vp/vs/rho (ni, nj, nk) C order; each sample block "
    "(n_columns, n_angles, nk - 1) for legacy.sample_columns"
)


def fnv1a64_f32(a: np.ndarray) -> str:
    """FNV-1a 64 over the little-endian bytes of a float32 array (C order)."""
    data = np.ascontiguousarray(a, dtype="<f4").tobytes()
    h = 0xCBF29CE484222325
    # vectorised enough for ~1 MB arrays
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def run_legacy(seed: int, size: int, min_vox: int):
    from datagenerator.Closures import Closures
    from datagenerator.Faults import Faults
    from datagenerator.Geomodels import Geomodel
    from datagenerator.Horizons import build_unfaulted_depth_maps, create_facies_array
    from datagenerator.Parameters import Parameters
    from datagenerator.Seismic import SeismicVolume

    tmp = Path(tempfile.mkdtemp(prefix="stackparity_"))
    cfg = json.loads((REPO / "config" / "example.json").read_text())
    cfg.update(
        project="example",  # selects rockphysics.rpm_example.RPMExample
        project_folder=str(tmp / "proj"),
        work_folder=str(tmp / "work"),
        cube_shape=[size, size, NZ],
        include_salt=False,
        extra_qc_plots=False,
        verbose=False,
        model_store_in_memory=True,
        cleanup_intermediates=True,
        closure_types=["simple"],
        min_closure_voxels_simple=min_vox,
    )
    cfg_path = tmp / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    with contextlib.redirect_stdout(io.StringIO()):
        p = Parameters(str(cfg_path), test_mode=size)
        p.setup_model(seed=seed)
        p.setup_model_store(in_memory=True)
        dm, onlaps, fans, fan_thk = build_unfaulted_depth_maps(p)
        facies = create_facies_array(p, dm, onlaps, fans)
        geo = Geomodel(p, dm, onlaps, facies)
        geo.build_unfaulted_geomodels()
        faults = Faults(p, dm, onlaps, geo, fans, fan_thk)
        faults.apply_faulting_to_geomodels_and_depth_maps()
        faults.build_faulted_property_geomodels(facies)
        closures = Closures(p, faults, facies, onlaps)
        closures.create_closures()
        seis = SeismicVolume(p, faults, closures)
        seis.build_elastic_properties("inv_vel")

        captured = {}

        def capture(dat, name):
            captured[name] = np.array(dat, dtype=np.float32, copy=True)

        seis.write_final_cubes_to_disk = capture
        seis.write_cube_to_disk = capture
        seis.create_rfc_volumes()
        rfc = np.array(seis.rfc_raw[:], dtype=np.float32)
        seis.postprocess_rfc_cubes(seis.rfc_raw[:], "noise_free", bb=False)

    return dict(
        p=p,
        seis=seis,
        faults=faults,
        vp=np.array(seis.vp[:], dtype=np.float32),
        vs=np.array(seis.vs[:], dtype=np.float32),
        rho=np.array(seis.rho[:], dtype=np.float32),
        rfc=rfc,
        stack=captured["seismicCubes_RFC_noise_free_"],
        cumsum=captured["seismicCubes_cumsum_noise_free_"],
        oil=int(np.sum(closures.oil_closures[:])),
        gas=int(np.sum(closures.gas_closures[:])),
        brine=int(np.sum(closures.brine_closures[:])),
    )


def upstream_rfc(vp, vs, rho, angles):
    """Upstream 6e4c2e30 ``create_rfc_volumes``: ``Seismic.RFC`` per trace."""
    from datagenerator.Seismic import RFC

    theta = np.asanyarray(angles).reshape((-1, 1))
    ni, nj, nk = vp.shape
    zoep = np.zeros((ni, nj, nk - 1, theta.size), dtype="complex64")
    for i in range(ni):
        for j in range(nj):
            r = RFC(vp[i, j, :-1], vs[i, j, :-1], rho[i, j, :-1],
                    vp[i, j, 1:], vs[i, j, 1:], rho[i, j, 1:], theta)
            zoep[i, j, :, :] = r.zoeppritz_reflectivity().T
    return np.moveaxis(np.real(zoep).astype("float64"), -1, 0).astype(np.float32)


def textbook_rfc(vp, vs, rho, angles):
    """bruges ``zoeppritz_rpp`` (textbook ``d``), float32 like the kernels."""
    from bruges.reflection import zoeppritz_rpp

    a = [vp[..., :-1], vs[..., :-1], rho[..., :-1], vp[..., 1:], vs[..., 1:], rho[..., 1:]]
    a = [x.astype(np.float64) for x in a]
    return np.stack([np.real(zoeppritz_rpp(*a, theta1=ang)).astype(np.float32) for ang in angles])


def stats(x):
    return dict(std=float(np.std(x, dtype=np.float64)), max_abs=float(np.max(np.abs(x))))


def main() -> None:
    import numba
    import scipy

    cases = []
    for seed, size, min_vox in CASES:
        r = run_legacy(seed, size, min_vox)
        p, seis = r["p"], r["seis"]
        angles = [float(a) for a in seis.angles]
        vp, vs, rho = r["vp"], r["vs"], r["rho"]
        ni, nj, nk = vp.shape
        seabed = r["faults"].faulted_depth_maps[..., 0] / p.digi
        gas_cols = np.argwhere(np.asarray(seis.traps.gas_closures[:]).any(axis=-1))
        cols = list(SAMPLE_COLUMNS)
        if len(gas_cols):
            cols.append(tuple(int(v) for v in gas_cols[len(gas_cols) // 2]))
        samp = [np.stack([r[kind][:, i, j, :] for i, j in cols]) for kind in ("rfc", "stack", "cumsum")]
        raw = b"".join(np.ascontiguousarray(x, "<f4").tobytes() for x in (vp, vs, rho, *samp))
        blob = zlib.compress(raw, 9)
        blob_name = f"angle_stack_e2e_seed{seed}.bin"
        (OUT / blob_name).write_bytes(blob)

        # Which legacy Zoeppritz path(s) carry the typo: compare on this model.
        up = upstream_rfc(vp, vs, rho, seis.angles)
        tb = textbook_rfc(vp, vs, rho, seis.angles)
        zpaths = []
        for a, ang in enumerate(angles):
            dn = np.abs(up[a].astype(np.float64) - r["rfc"][a])
            dt = np.abs(tb[a].astype(np.float64) - r["rfc"][a])
            zpaths.append(dict(
                angle=ang,
                numba_vs_rfc_class_max_abs=float(dn.max()),
                numba_vs_rfc_class_bit_identical_frac=float(np.mean(up[a].view("<u4") == r["rfc"][a].view("<u4"))),
                numba_vs_textbook_max_abs=float(dt.max()),
                rfc_class_vs_textbook_max_abs=float(np.abs(up[a].astype(np.float64) - tb[a]).max()),
            ))

        case = dict(
            seed=seed,
            test_mode=size,
            min_closure_voxels_simple=min_vox,
            shape=[ni, nj, nk],
            angles_deg=angles,
            digi_ms=float(p.digi),
            bandpass_hz=[float(p.lowfreq), float(p.highfreq)],
            bandpass_order=int(p.order),
            lateral_filter_size=int(p.lateral_filter_size),
            closure_voxels=dict(oil=r["oil"], gas=r["gas"], brine=r["brine"]),
            seabed_samples=dict(min=float(np.nanmin(seabed)), max=float(np.nanmax(seabed))),
            props_blob=blob_name,
            blob_sha256=hashlib.sha256(raw).hexdigest(),
            legacy=dict(
                rfc_raw=dict(shape=list(r["rfc"].shape[1:]),
                             fnv1a64=[fnv1a64_f32(r["rfc"][a]) for a in range(len(angles))],
                             stats=[stats(r["rfc"][a]) for a in range(len(angles))]),
                stack=dict(shape=list(r["stack"].shape[1:]),
                           fnv1a64=[fnv1a64_f32(r["stack"][a]) for a in range(len(angles))],
                           stats=[stats(r["stack"][a]) for a in range(len(angles))]),
                cumsum=dict(shape=list(r["cumsum"].shape[1:]),
                            fnv1a64=[fnv1a64_f32(r["cumsum"][a]) for a in range(len(angles))],
                            stats=[stats(r["cumsum"][a]) for a in range(len(angles))]),
                sample_columns=[list(c) for c in cols],
            ),
            zoeppritz_paths=zpaths,
        )
        cases.append(case)
        print(f"seed {seed}: shape {vp.shape} angles {angles} lat {p.lateral_filter_size} "
              f"bp {p.lowfreq:.3f}-{p.highfreq:.3f} closures oil {r['oil']} gas {r['gas']}")
        for z in zpaths:
            print("   ", z)

    out = dict(
        meta=dict(
            generator="tests/fixtures/generate_angle_stack_e2e.py",
            legacy_chain=[
                "datagenerator main.build_model steps up to SeismicVolume.build_elastic_properties('inv_vel')",
                "SeismicVolume.create_rfc_volumes (Numba datagenerator/zoeppritz_kernel.py)",
                "SeismicVolume.postprocess_rfc_cubes(rfc_raw, 'noise_free'): apply_bandlimits -> apply_lateral_filter -> apply_cumsum",
            ],
            captured_before="write_final_cubes_to_disk (_scale_seismic + rpm factors not applied)",
            config="config/example.json with cube_shape [test_mode, test_mode, 500], include_salt false, closure_types ['simple'], project 'example'",
            blob_layout=BLOB_LAYOUT,
            hash="FNV-1a 64 over little-endian float32 bytes, C order, one cube per angle",
            versions=dict(numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                          python=sys.version.split()[0]),
        ),
        cases=cases,
    )
    (OUT / "angle_stack_e2e.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print("wrote", OUT / "angle_stack_e2e.json")


if __name__ == "__main__":
    main()
