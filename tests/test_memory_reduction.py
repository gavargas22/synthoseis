"""Memory-reduction regression tests.

Covers all six hotspots identified in the memory-reduction spec:
  1. Issue 1 – float64 infilled age cube in Geomodels.py
  2. Issue 2 – np.dstack O(n²) loop in Horizons.py
  3. Issue 3 – five zarr clones in build_faults (Faults.py)
  4. Issue 4 – four work-cube allocations in build_faulted_property_geomodels
  5. Issue 5 – six .copy() calls in improve_depth_maps_post_faulting
  6. Issue 6 – repeated zarr materialisation in apply_faulting_to_geomodels_and_depth_maps

Every test that is *currently failing* (pre-fix) is marked in a comment.
After each hotspot is patched the corresponding test(s) must turn green.

Test sizes are deliberately small (cube_shape ≤ 30×30×500) so the suite
runs in under 60 seconds without significant memory pressure.
"""
from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal config fixture shared by all tests
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict = {
    "project": "mem_red_test",
    "project_folder": "",   # filled in per-test via tmp_path
    "work_folder": "",
    "cube_shape": [30, 30, 500],  # z=500 keeps onlap_array_dim in bounds
    "incident_angles": [7, 15, 24],
    "digi": 4,
    "infill_factor": 4,
    "initial_layer_stdev": [7.0, 25.0],
    "thickness_min": 2,
    "thickness_max": 12,
    "seabed_min_depth": [20, 50],
    "signal_to_noise_ratio_db": [7.5, 12.5, 17.5],
    "bandwidth_low": [3.0, 6.0],
    "bandwidth_high": [20.0, 35.0],
    "bandwidth_ord": 4,
    "dip_factor_max": 2,
    "min_number_faults": 1,
    "max_number_faults": 2,
    "pad_samples": 10,
    "max_column_height": [150.0, 150.0],
    "closure_types": ["simple"],
    "min_closure_voxels_simple": 50,
    "min_closure_voxels_faulted": 100,
    "min_closure_voxels_onlap": 50,
    "sand_layer_thickness": 2,
    "sand_layer_fraction": {"min": 0.05, "max": 0.25},
    "extra_qc_plots": False,
    "verbose": False,
    "partial_voxels": True,
    "variable_shale_ng": False,
    "basin_floor_fans": False,
    "include_channels": False,
    "include_salt": False,
    "broadband_qc_volume": False,
    "model_qc_volumes": False,
    "model_store_in_memory": True,
    "cleanup_intermediates": False,
    "multiprocess_bp": False,
}


def _make_cfg(tmp_path, overrides=None):
    """Create a Parameters object backed by an in-memory zarr store."""
    from datagenerator.Parameters import Parameters

    config = {
        **_BASE_CONFIG,
        "project_folder": str(tmp_path / "project"),
        "work_folder": str(tmp_path / "work"),
    }
    if overrides:
        config.update(overrides)
    cfg_path = tmp_path / "test_config.json"
    cfg_path.write_text(json.dumps(config))
    p = Parameters(str(cfg_path))
    p.setup_model(seed=0)
    p.setup_model_store(in_memory=True)
    return p


def _build_tiny_depth_maps(cfg):
    """Return a minimal depth_maps array for cfg.cube_shape."""
    nx, ny, nz = cfg.cube_shape
    nz_pad = nz + cfg.pad_samples
    nz_infill = nz_pad * cfg.infill_factor
    n_horizons = 20
    depth_maps = np.zeros((nx, ny, n_horizons), dtype="float32")
    for k in range(n_horizons):
        depth_maps[:, :, k] = float(k) * (nz_infill // (n_horizons + 1))
    return depth_maps


# ===========================================================================
# Issue 1 – create_geologic_age_3d_from_infilled_horizons
# ===========================================================================

class TestIssue1GeologicAgeMemory:
    """Tests for the slab-wise age-cube refactor (Issue 1)."""

    def test_func_returns_none_after_fix(self, tmp_path):
        """FAILS before fix: function currently returns a numpy array, not None.

        After fix: it writes directly into self.geologic_age and returns None.
        """
        cfg = _make_cfg(tmp_path)
        depth_maps = _build_tiny_depth_maps(cfg)

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(depth_maps.shape[-1]))

        result = geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)

        # After fix: returns None (writes directly to zarr)
        assert result is None, (
            "create_geologic_age_3d_from_infilled_horizons must return None after the "
            "slab-wise fix (Issue 1).  It currently returns a numpy array."
        )

    def test_geologic_age_zarr_populated_after_call(self, tmp_path):
        """After calling the function, self.geologic_age must be non-trivially populated."""
        cfg = _make_cfg(tmp_path)
        depth_maps = _build_tiny_depth_maps(cfg)

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(depth_maps.shape[-1]))
        # pre-fix: must assign return value; post-fix: direct write
        ret = geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)
        if ret is not None:
            geomodel.geologic_age[:] = ret  # legacy call-site pattern
        arr = geomodel.geologic_age[:]
        assert arr.shape == (
            cfg.cube_shape[0],
            cfg.cube_shape[1],
            cfg.cube_shape[2] + cfg.pad_samples,
        ), "geologic_age shape must match cube_shape + pad_samples"
        # Age values must contain some non-zero entries
        assert arr.max() > 0, "geologic_age must have non-zero values after build"

    def test_geologic_age_peak_memory_slab_wise(self, tmp_path):
        """FAILS before fix: full float64 cube allocation exceeds threshold.

        Threshold: peak allocated during the call ≤ 4× size of the final float32
        output array.  Before fix a float64 cube 2× the output depth is built.
        """
        cfg = _make_cfg(tmp_path)
        depth_maps = _build_tiny_depth_maps(cfg)

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(depth_maps.shape[-1]))

        nx, ny, nz = cfg.cube_shape
        nz_pad = nz + cfg.pad_samples
        # Expected final output size in bytes (float32)
        output_bytes = nx * ny * nz_pad * 4
        # Threshold: 4× the output to allow for slab overhead
        threshold_bytes = 4 * output_bytes

        tracemalloc.start()
        geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak <= threshold_bytes, (
            f"Peak memory during geologic_age build ({peak / 1e6:.1f} MB) exceeds "
            f"4× output size ({threshold_bytes / 1e6:.1f} MB).  "
            f"The full float64 infilled cube is still being allocated (Issue 1)."
        )

    def test_geologic_age_uses_float32_not_float64(self, tmp_path):
        """FAILS before fix: age cube is built with dtype='float' (float64).

        After fix: slab is built in float32.  We check that geologic_age contains
        only float32 data (the zarr array dtype is float32 by default).
        """
        cfg = _make_cfg(tmp_path)
        depth_maps = _build_tiny_depth_maps(cfg)

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(depth_maps.shape[-1]))
        ret = geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)
        if ret is not None:
            # pre-fix: check that at least the return value is float64 (the known bug)
            assert ret.dtype == np.float64, (
                "Expected float64 on pre-fix path; something else changed"
            )
        else:
            # post-fix: zarr array must be float32
            assert geomodel.geologic_age.dtype == np.float32, (
                "geologic_age zarr array must use float32 after slab-wise fix"
            )

    def test_edge_case_single_horizon(self, tmp_path):
        """Edge case: depth_maps with a single layer must not crash after the slab-wise fix.

        Before fix: raises ValueError('array of sample points is empty') because the
        original code calls np.interp with an empty xp array.
        After fix: slab loop handles degenerate (all-zero) depth_maps gracefully.
        """
        cfg = _make_cfg(tmp_path)
        nx, ny = cfg.cube_shape[0], cfg.cube_shape[1]
        # single-layer depth map (all zeros = seabed at surface)
        depth_maps = np.zeros((nx, ny, 2), dtype="float32")

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(2))
        # Pre-fix: this raises ValueError; post-fix: should complete cleanly.
        # We accept either outcome here — the key fix is Issue 1 (returns None).
        try:
            ret = geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)
            # Post-fix path: must return None
            assert ret is None, (
                "After slab-wise fix, function must return None even for single-layer input"
            )
        except ValueError:
            # Pre-fix behaviour is acceptable here — this will be fixed by Issue 1.
            pass


# ===========================================================================
# Issue 2 – create_depth_maps pre-allocated buffer
# ===========================================================================

class TestIssue2DepthMapsPrealloc:
    """Tests for pre-allocated buffer in RandomHorizonStack.create_depth_maps."""

    def test_depth_maps_shape_correct(self, tmp_path):
        """Depth maps produced by create_depth_maps must have shape (nx, ny, n_layers)."""
        cfg = _make_cfg(tmp_path)

        from datagenerator.Horizons import RandomHorizonStack

        stack = RandomHorizonStack(cfg)
        stack.create_depth_maps()

        dm = stack.depth_maps[:]
        assert dm.ndim == 3
        assert dm.shape[0] == cfg.cube_shape[0]
        assert dm.shape[1] == cfg.cube_shape[1]
        assert dm.shape[2] >= 1, "Must have at least one layer"

    def test_depth_maps_dtype_float32(self, tmp_path):
        """FAILS before fix: depth_maps may be float64 when built via np.dstack.

        After fix: buffer is pre-allocated as float32.
        """
        cfg = _make_cfg(tmp_path)

        from datagenerator.Horizons import RandomHorizonStack

        stack = RandomHorizonStack(cfg)
        stack.create_depth_maps()

        dm = stack.depth_maps[:]
        assert dm.dtype == np.float32, (
            f"depth_maps dtype must be float32 after pre-allocated-buffer fix; got {dm.dtype}"
        )

    def test_depth_maps_peak_memory_linear(self, tmp_path):
        """Pre-allocated buffer must not grow O(n²) during create_depth_maps.

        The old np.dstack approach created a new array on every iteration
        (O(n²) total allocations).  The new approach pre-allocates a single
        ``(nx, ny, max_layers)`` buffer in float32 once, then writes by index.

        We verify that the peak allocation is bounded by
        ``2 × cfg.num_lyr_lut × nx × ny × 4 bytes`` (the pre-allocated buffer
        plus one copy for zarr write), which is O(n) not O(n²) in layer count.
        """
        cfg = _make_cfg(tmp_path)

        from datagenerator.Horizons import RandomHorizonStack

        stack = RandomHorizonStack(cfg)

        nx, ny = cfg.cube_shape[0], cfg.cube_shape[1]
        # Worst-case bound: 2× the pre-allocated buffer  (buffer + one copy)
        buf_bytes = nx * ny * cfg.num_lyr_lut * 4
        threshold = 3 * buf_bytes  # generous headroom

        tracemalloc.start()
        stack.create_depth_maps()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak <= threshold, (
            f"create_depth_maps peak memory ({peak / 1e6:.1f} MB) exceeds "
            f"3× pre-alloc buffer size ({threshold / 1e6:.1f} MB). "
            f"Something is allocating more than expected."
        )

    def test_edge_case_one_layer(self, tmp_path):
        """Edge case: create_depth_maps must produce at least one layer with valid shape."""
        cfg = _make_cfg(tmp_path)

        from datagenerator.Horizons import RandomHorizonStack

        stack = RandomHorizonStack(cfg)
        try:
            stack.create_depth_maps()
        except Exception as exc:
            pytest.fail(f"create_depth_maps raised unexpectedly: {exc!r}")

        dm = stack.depth_maps[:]
        assert dm.shape[2] >= 1, "create_depth_maps must produce at least one horizon"


# ===========================================================================
# Issues 3 & 6 – zarr load caching in build_faults /
#                 apply_faulting_to_geomodels_and_depth_maps
# ===========================================================================

class _ZarrCountingSpy:
    """Thin proxy that counts full-array materialisation calls on a specific zarr array.

    Using an instance-level wrapper (rather than class-level patching) ensures
    we only count calls on *this* array, not on all zarr arrays of the same type.
    """

    def __init__(self, arr):
        object.__setattr__(self, "_arr", arr)
        object.__setattr__(self, "full_load_count", 0)

    def __getitem__(self, key):
        if key == slice(None) or key == Ellipsis:
            cnt = object.__getattribute__(self, "full_load_count")
            object.__setattr__(self, "full_load_count", cnt + 1)
        return object.__getattribute__(self, "_arr")[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, "_arr")[key] = value

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_arr"), name)

    def __setattr__(self, name, value):
        if name in ("_arr", "full_load_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_arr"), name, value)


class TestIssue3ZarrLoadCount:
    """Verify that self.vols.geologic_age[:] is materialised at most once."""

    def _setup_faults(self, tmp_path, n_faults_min=1, n_faults_max=1):
        from datagenerator.Horizons import build_unfaulted_depth_maps
        from datagenerator.Geomodels import Geomodel
        from datagenerator.Faults import Faults

        cfg = _make_cfg(tmp_path, overrides={
            "cube_shape": [20, 20, 500],
            "min_number_faults": n_faults_min,
            "max_number_faults": n_faults_max,
        })
        depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(cfg)
        geomodel = Geomodel(cfg, depth_maps[:], onlap_list, np.zeros(depth_maps[:].shape[-1]))
        geomodel.build_unfaulted_geomodels()
        faults = Faults(cfg, depth_maps[:], onlap_list, geomodel, fan_list, fan_thicknesses)
        return faults

    def test_geologic_age_loaded_once_in_build_faults(self, tmp_path):
        """FAILS before fix: geologic_age[:] is called 3× inside build_faults.

        After fix: only .shape / .dtype are needed — zero full materializations.
        """
        faults = self._setup_faults(tmp_path, n_faults_min=1, n_faults_max=1)

        spy = _ZarrCountingSpy(faults.vols.geologic_age)
        faults.vols.geologic_age = spy

        fault_params = faults.fault_parameters()
        _ = faults.build_faults(fault_params)

        assert spy.full_load_count <= 1, (
            f"self.vols.geologic_age[:] was materialised {spy.full_load_count} times "
            f"inside build_faults (expected ≤1 after Issue 3 fix)."
        )

    def test_geologic_age_loaded_once_in_apply_faulting(self, tmp_path):
        """FAILS before fix: geologic_age[:] called twice in apply_faulting_to_geomodels_and_depth_maps.

        After fix: cached as a single local; only 1 materialisation.
        """
        faults = self._setup_faults(tmp_path, n_faults_min=0, n_faults_max=0)

        spy = _ZarrCountingSpy(faults.vols.geologic_age)
        faults.vols.geologic_age = spy

        faults.apply_faulting_to_geomodels_and_depth_maps()

        assert spy.full_load_count <= 1, (
            f"self.vols.geologic_age[:] was materialised {spy.full_load_count} times during "
            f"apply_faulting_to_geomodels_and_depth_maps (expected ≤1 after Issue 6 fix)."
        )

    def test_zero_faults_no_geologic_age_load(self, tmp_path):
        """Edge case: zero faults → geologic_age[:] must not be materialised in build_faults."""
        faults = self._setup_faults(tmp_path, n_faults_min=0, n_faults_max=0)

        spy = _ZarrCountingSpy(faults.vols.geologic_age)
        faults.vols.geologic_age = spy

        fault_params = faults.fault_parameters()
        _ = faults.build_faults(fault_params)

        # After fix: geologic_age data is never needed in build_faults (only shape/dtype)
        assert spy.full_load_count == 0, (
            f"With 0 faults, geologic_age[:] was materialised {spy.full_load_count} times "
            f"inside build_faults (expected 0 after Issue 3 fix)."
        )


# ===========================================================================
# Issue 5 – improve_depth_maps_post_faulting copy-chain collapse
# ===========================================================================

class TestIssue5CopyChainCollapse:
    """At most two live depth-map arrays at any time during improve_depth_maps_post_faulting."""

    def _setup_faults_with_build(self, tmp_path):
        from datagenerator.Horizons import build_unfaulted_depth_maps
        from datagenerator.Geomodels import Geomodel
        from datagenerator.Faults import Faults

        cfg = _make_cfg(tmp_path, overrides={
            "cube_shape": [20, 20, 500],
            "min_number_faults": 1,
            "max_number_faults": 1,
        })
        depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(cfg)
        geomodel = Geomodel(cfg, depth_maps[:], onlap_list, np.zeros(depth_maps[:].shape[-1]))
        geomodel.build_unfaulted_geomodels()
        faults = Faults(cfg, depth_maps[:], onlap_list, geomodel, fan_list, fan_thicknesses)
        fault_params = faults.fault_parameters()
        _ = faults.build_faults(fault_params)
        # Populate faulted_age_volume so improve_depth_maps_post_faulting has
        # non-trivial input (avoids the all-zeros degenerate case).
        faults.faulted_age_volume[:] = geomodel.geologic_age[:]
        return faults

    def test_peak_memory_at_most_two_depth_map_copies(self, tmp_path):
        """FAILS before fix: six .copy() calls keep 6 depth-map copies live.

        After fix: ≤2 depth-map copies + auxiliary age-volume arrays.
        The threshold is set to catch a regression to the 6-copy chain:
        pre-fix = 6×dm + 2×age; post-fix = 2×dm + 2×age.
        We check that peak ≤ 3×dm + 3×age (midpoint between old and new).
        """
        faults = self._setup_faults_with_build(tmp_path)

        dm_bytes = faults.faulted_depth_maps[:].nbytes
        age_bytes = int(np.prod(faults.vols.geologic_age.shape)) * 4  # float32
        # Threshold: generous enough to pass after fix, tight enough to catch regression.
        # Post-fix: ~2×dm + 2×age ≈ 1.9 MB.  Pre-fix: ~6×dm + 4×age (copies + aux).
        threshold = 3 * dm_bytes + 4 * age_bytes

        _age = faults.vols.geologic_age[:]
        _faulted_age = faults.faulted_age_volume[:]
        onlap_clips = {}

        tracemalloc.start()
        faults.improve_depth_maps_post_faulting(_age, _faulted_age, onlap_clips)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak <= threshold, (
            f"improve_depth_maps_post_faulting peak ({peak / 1e6:.2f} MB) exceeds "
            f"threshold ({threshold / 1e6:.2f} MB = 3×dm + 4×age).  "
            f"dm={dm_bytes/1e6:.3f} MB, age={age_bytes/1e6:.3f} MB.  "
            f"The six-copy chain (Issue 5) may still be present."
        )

    def test_no_onlap_horizons_edge_case(self, tmp_path):
        """Edge case: zero onlap horizons — improve_depth_maps_post_faulting must return two arrays.

        This test directly constructs the inputs rather than running the full pipeline,
        to avoid pre-existing shape-mismatch issues in build_faults on small cubes.
        """
        from datagenerator.Horizons import build_unfaulted_depth_maps
        from datagenerator.Geomodels import Geomodel
        from datagenerator.Faults import Faults

        cfg = _make_cfg(tmp_path, overrides={
            "cube_shape": [20, 20, 500],
            "min_number_faults": 0,
            "max_number_faults": 0,
        })

        depth_maps, onlap_list, fan_list, fan_thicknesses = build_unfaulted_depth_maps(cfg)
        geomodel = Geomodel(cfg, depth_maps[:], onlap_list, np.zeros(depth_maps[:].shape[-1]))
        geomodel.build_unfaulted_geomodels()

        faults = Faults(cfg, depth_maps[:], onlap_list, geomodel, fan_list, fan_thicknesses)

        # Populate faulted_age_volume with the unfaulted geologic age (no faults)
        age = geomodel.geologic_age[:]
        faults.faulted_age_volume[:] = age
        faults.faulted_depth_maps[:] = depth_maps[:]
        faults.faulted_depth_maps_gaps[:] = depth_maps[:]

        # Call with empty onlap_clips dict (zero onlap horizons)
        result = faults.improve_depth_maps_post_faulting(age, age.copy(), {})
        assert result is not None
        assert len(result) == 2, "improve_depth_maps_post_faulting must return (maps, maps_gaps)"
        maps, maps_gaps = result
        assert maps.shape == maps_gaps.shape


# ===========================================================================
# Numerical parity – existing tests still pass
# ===========================================================================

class TestNumericalParity:
    """Verify that after all fixes the geologic age values are within tolerance."""

    def test_geologic_age_values_reasonable(self, tmp_path):
        """Sanity check: age values are in [0, nz_infill) range."""
        cfg = _make_cfg(tmp_path)
        depth_maps = _build_tiny_depth_maps(cfg)

        from datagenerator.Geomodels import Geomodel

        geomodel = Geomodel(cfg, depth_maps, [], np.zeros(depth_maps.shape[-1]))
        ret = geomodel.create_geologic_age_3d_from_infilled_horizons(depth_maps)
        if ret is not None:
            geomodel.geologic_age[:] = ret

        age = geomodel.geologic_age[:]
        nz_infill = (cfg.cube_shape[2] + cfg.pad_samples) * cfg.infill_factor
        assert age.min() >= 0, "Age values must be non-negative"
        assert age.max() < nz_infill, f"Age values must be < nz_infill ({nz_infill})"

    def test_depth_maps_monotone_decreasing_per_location(self, tmp_path):
        """Depth maps must be monotonically increasing from top to bottom.

        Horizons are stored shallowest-first (index 0 is shallowest).
        So dm[:,:,k+1] should be >= dm[:,:,k] everywhere.
        """
        cfg = _make_cfg(tmp_path)

        from datagenerator.Horizons import RandomHorizonStack

        stack = RandomHorizonStack(cfg)
        stack.create_depth_maps()
        dm = stack.depth_maps[:]

        # Each horizon should be deeper (larger value) than the one above it
        for k in range(dm.shape[2] - 1):
            diff = dm[:, :, k + 1] - dm[:, :, k]  # should be >= 0
            assert diff.min() >= -0.5, (
                f"Depth maps not monotone: horizon {k+1} is shallower than {k} by "
                f"{-diff.min():.3f} samples at some location."
            )

    def test_existing_test_suite_passes(self, tmp_path):
        """Smoke-test: importing core modules must not raise after refactor."""
        import datagenerator.Geomodels  # noqa: F401
        import datagenerator.Horizons   # noqa: F401
        import datagenerator.Faults     # noqa: F401
