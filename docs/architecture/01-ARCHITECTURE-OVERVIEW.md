# Synthoseis Architecture Overview

## 1. System Purpose

Synthoseis generates pseudo-random synthetic 3D seismic data and associated geological labels for training deep learning networks. It simulates the full chain from geological structure through rock physics to seismic wave propagation, producing realistic labeled training data without requiring real-world acquisition.

The system models:
- Sedimentary layer deposition with realistic thickness distributions
- Structural deformation (faulting) with multiple fault patterns
- Stratigraphic features (onlaps, basin floor fans)
- Salt diapir intrusion
- Hydrocarbon trap identification and fluid assignment
- Elastic property modeling via depth-dependent rock physics
- Angle-dependent seismic reflectivity via Zoeppritz equations
- Realistic noise, filtering, and geophysical augmentation

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI (main.py)                               │
│   argparse → sequential loop over num_runs → build_model()          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATION                            │
│                       build_model()                                  │
│                                                                     │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │Parameters│→ │ Horizons │→ │Geomodels│→ │  Faults  │→ │Closures│ │
│  └─────────┘  └──────────┘  └─────────┘  └──────────┘  └────────┘ │
│       │                                        │             │      │
│       │            ┌──────────┐                │             │      │
│       │            │   Salt   │────────────────┘             │      │
│       │            └──────────┘                              │      │
│       │                                                      │      │
│       │       ┌──────────────────┐                           │      │
│       └──────→│  SeismicVolume   │←──────────────────────────┘      │
│               └──────────────────┘                                  │
│                        │                                            │
│               ┌────────┴─────────┐                                  │
│               │  RockPhysics RPM │                                  │
│               └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER (Zarr v3)                           │
│               StorageClient (mdio_backend.py)                       │
│        create_dataset / get_dataset / list / remove / close         │
│              Default chunks: (128, 128, 128)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Execution Order

The `build_model()` function in `main.py` executes these stages strictly sequentially. Each stage depends on outputs of previous stages:

| Stage | Module | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| 1 | `Parameters` | JSON config file | Shared state object `p` | Load config, create directories, open storage |
| 2 | `Horizons` | `p` | `depth_maps`, `facies`, `onlap_list`, `fan_list` | Generate unfaulted layer boundaries |
| 3 | `Geomodels` | `depth_maps`, `facies`, `onlap_list` | `geologic_age` 3D volume, `onlap_segments` | Convert 2D horizons to 3D age model |
| 4 | `Faults` | `depth_maps`, `geo_models`, `fan_list` | Faulted depth maps, faulted property volumes | Apply structural deformation |
| 5 | `Salt` (optional) | `depth_maps`, `salt_segments` | Modified depth maps with drag | Generate salt diapir bodies |
| 6 | `Closures` | Faulted data, `facies`, `onlap_list` | Closure volumes (oil/gas/brine by type) | Identify hydrocarbon traps |
| 7 | `SeismicVolume` | Faulted data, closures | RFC volumes, seismic cubes | Generate synthetic seismic data |
| 8 | Cleanup | — | — | Close storage, delete temp files |

---

## 4. Core Design Patterns

### 4.1 The Borg Pattern (Shared State Singleton)

The `Parameters` class uses the Borg pattern (Alex Martelli) where all instances share the same `__dict__`. This means every module that creates a `Parameters` instance (or receives one) accesses identical state:

```
class _Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Parameters(_Borg):
    # All attributes are shared across all instances
```

**Implications:**
- Configuration is globally accessible without explicit passing
- State is reset between runs via `self._shared_state = {}`
- Tight coupling: all modules implicitly depend on shared state
- No dependency injection; modules cannot be tested in isolation easily

### 4.2 Sequential Pipeline with Mutable Shared State

Each pipeline stage receives the `Parameters` object plus intermediate results from previous stages. Stages mutate shared arrays in-place and store results to the Zarr backend.

There is no pipeline abstraction, dependency graph, or event system. The orchestration is a linear sequence of constructor calls and method invocations in `build_model()`.

### 4.3 Storage Abstraction

The `StorageClient` wraps Zarr v3 with a simple CRUD API:
- `create_dataset(name, data=, shape=)` — Write array
- `get_dataset(name, lazy=, use_dask=)` — Read array (eager or lazy)
- `list_datasets(prefix=)` — Enumerate stored data
- `remove_dataset(name)` — Delete array
- `close()` — Flush and close

Default chunk size: `(128, 128, 128)`. Default dtype: `float32`.

### 4.4 Dynamic Rock Physics Model Loading

Rock physics models are selected at runtime based on the `"project"` field in the JSON config. The `select_rpm()` function uses conditional imports:

```
if cfg.project == "example":
    from rockphysics.rpm_example import RPMExample
    rpm = RPMExample(cfg)
```

Each RPM must implement four methods returning `RockProperties` objects:
- `calc_shale_properties(z_rho, z_vp, z_vs)`
- `calc_brine_sand_properties(z_rho, z_vp, z_vs)`
- `calc_oil_sand_properties(z_rho, z_vp, z_vs)`
- `calc_gas_sand_properties(z_rho, z_vp, z_vs)`

---

## 5. Module Dependency Graph

```
main.py
├── Parameters.py ─────────────── synthoseis/storage/mdio_backend.py (StorageClient)
│   └── reads JSON config
│   └── creates directories, logfiles
│   └── manages Zarr storage
│
├── Horizons.py
│   ├── simplexNoise.py (Perlin noise)
│   ├── perlin-noise library
│   └── scipy.interpolate (griddata, spline)
│
├── Geomodels.py
│   ├── dask (parallel trace interpolation)
│   └── scipy.ndimage (maximum_filter)
│
├── Faults.py
│   ├── scipy.linalg (matrix exponential for 3D rotation)
│   ├── scipy.signal (fftconvolve for hockey stick)
│   ├── scipy.ndimage (Sobel, maximum_filter, Gaussian, rotate)
│   ├── skimage.measure (label for intersections)
│   ├── dask (parallel trace interpolation)
│   └── Salt.py (optional)
│       └── scipy.spatial (Delaunay for convex hull)
│
├── Closures.py
│   ├── skimage.measure (label for connected components)
│   ├── skimage.morphology (remove_small_objects, dilation)
│   └── scipy.ndimage (grey_dilation)
│
├── Seismic.py
│   ├── rockphysics/RockPropertyModels.py
│   │   ├── bruges.rockphysics.moduli (elastic moduli)
│   │   └── rockphysics/rpm_example.py (depth trends)
│   ├── wavelets.py (Ricker, FFT, spectral analysis)
│   ├── histogram_equalizer.py (amplitude normalization)
│   ├── Augmentation.py (tz_stretch, uniform_stretch)
│   ├── dask (chunk-based Zoeppritz, noise, filtering)
│   └── scipy.signal (butter, filtfilt for bandpass)
│
└── util.py (plotting, QC, math helpers)
    ├── plotly (3D fault/closure visualization)
    ├── matplotlib (2D cross-sections)
    └── scipy.spatial (Delaunay for hull test)
```

---

## 6. Data Flow and Array Shapes

### 6.1 Primary Data Structures

| Data | Shape | Type | Description |
|------|-------|------|-------------|
| `depth_maps` | `(X, Y, num_horizons)` | float32 | 2D horizon depth surfaces |
| `facies` | `(num_horizons+1,)` | float | Per-layer lithology code (-1=water, 0=shale, 1=sand) |
| `geologic_age` | `(X, Y, Z+pad)` | float32 | 3D interpolated age model |
| `onlap_segments` | `(X, Y, Z+pad)` | float32 | Onlap proximity labels |
| `fault_planes` | `(X, Y, Z+pad)` | float32 | Binary fault plane segmentation |
| `displacement_vectors` | `(X, Y, Z+pad)` | float32 | Z-displacement per voxel |
| `faulted_age_volume` | `(X, Y, Z+pad)` | float32 | Geologic age after faulting |
| `faulted_lithology` | `(X, Y, Z+pad)` | float32 | Rock type after faulting |
| `faulted_depth` | `(X, Y, Z+pad)` | float32 | Depth below mudline |
| `closure_segments` | `(X, Y, Z+pad)` | float32 | Closure identification labels |
| `oil_closures` | `(X, Y, Z+pad)` | uint8 | Binary oil-bearing voxels |
| `gas_closures` | `(X, Y, Z+pad)` | uint8 | Binary gas-bearing voxels |
| `salt_segments` | `(X, Y, Z+pad)` | int16 | Binary salt body |
| `rho`, `vp`, `vs` | `(X, Y, Z+pad)` | float32 | Elastic properties |
| `rfc_raw` | `(n_angles, X, Y, Z-1)` | float32 | Raw reflection coefficients |
| `rfc_noise_added` | `(n_angles, X, Y, Z-1)` | float32 | RFC with noise |

### 6.2 Coordinate Convention

```
cube[i, j, k]
  i = inline (X)        — ranges [0, cube_shape[0]-1]
  j = crossline (Y)     — ranges [0, cube_shape[1]-1]
  k = sample/depth (Z)  — ranges [0, cube_shape[2]+pad_samples-1]

Depth increases with k (deeper = larger k values).
Horizons are stacked: index 0 = shallowest, last = deepest.
```

### 6.3 Infill Factor

Internal computations use an "infilled" resolution that is `infill_factor` times finer in the Z-direction. This provides sub-sample precision for layer boundaries:

```
Infilled shape: (X, Y, (Z + pad) * infill_factor)
Output shape:   (X, Y, Z + pad)
Downsampling:   Every infill_factor-th sample (stride slicing)
```

---

## 7. Configuration System

### 7.1 JSON Configuration

All model parameters are defined in JSON files under `config/`. The `Parameters` class reads these and exposes them as instance attributes on the Borg-shared state.

**Key parameter groups:**
- **Geometry:** `cube_shape`, `digi`, `infill_factor`, `pad_samples`
- **Horizons:** `initial_layer_stdev`, `thickness_min/max`, `seabed_min_depth`
- **Faults:** `min/max_number_faults`, `dip_factor_max`
- **Closures:** `closure_types`, `min_closure_voxels_*`, `max_column_height`
- **Seismic:** `incident_angles`, `signal_to_noise_ratio_db`, `bandwidth_low/high`
- **Sand:** `sand_layer_thickness`, `sand_layer_fraction`
- **Toggles:** `include_salt`, `basin_floor_fans`, `partial_voxels`
- **QC:** `extra_qc_plots`, `model_qc_volumes`, `verbose`

### 7.2 Randomization

Many parameters are randomized per model run using distributions:
- **Triangular:** SNR, fault throw, layer shift samples
- **Uniform:** Dip angles, azimuths, frequency bands, sand percentage
- **Gamma:** Layer thicknesses
- **Power:** Layer dip magnitudes (skewed toward small values)
- **Binomial:** Channel occurrence (3% probability), sand/shale facies

### 7.3 RPM Scaling Factors

Per-run factors randomize rock property depth lookups:
- `layershiftsamples`: Shift depth lookup for all layers (triangular 35-125)
- `RPshiftsamples`: Per-property depth offset (triangular 5-20)
- `shale/sand_rho/vp/vs_factor`: Multiplicative scaling (default 1.0)
- `near/mid/far_factor`: Per-angle amplitude scaling (default 1.0)

---

## 8. Parallelization Strategy

| Component | Technology | Granularity |
|-----------|-----------|-------------|
| Age model interpolation | Dask delayed (threads) | Per-trace (i,j) |
| Fault displacement | Dask delayed (threads) | Per-trace (i,j) |
| Depth map re-interpolation | Dask delayed (threads) | Per-trace (i,j) |
| Zoeppritz RFC computation | Dask map_blocks | Per-chunk |
| Noise generation | Dask random | Per-chunk |
| Bandpass filtering | Dask map_blocks | Per-chunk |
| Cumulative sum | Dask cumsum | Full array |
| Sequential model runs | Python for-loop | Sequential (NOT parallel) |
| Fault application | Sequential loop | Per-fault (order matters) |

**Cubed** is available as an import but not actively used in the current pipeline.

---

## 9. Output Structure

Each model run produces:

```
{project_folder}/seismic__{timestamp}_{runid}/
├── seismicCube_{timestamp}.key          # Survey geometry metadata
├── model_parameters_{timestamp}.txt      # Parameter log
├── sql_log.txt                          # SQL-formatted parameter dump
├── parameters.db                        # SQLite database
├── model_data.zarr/                     # Main Zarr store
│   ├── depth_maps                       # Horizon data
│   ├── facies                           # Lithology codes
│   ├── geologic_age                     # 3D age model
│   ├── fault_planes                     # Fault segmentation
│   ├── oil_closures / gas_closures      # Trap labels
│   ├── rho / vp / vs                    # Elastic properties
│   ├── rfc_raw                          # Raw reflectivity
│   ├── rfc_noise_added                  # Noisy reflectivity
│   ├── seismic_*_degrees                # Final seismic cubes
│   └── ... (40+ datasets)              # Various intermediate results
└── *.png                                # QC plots (if enabled)
```

---

## 10. Known Architectural Issues

1. **Borg pattern creates tight coupling.** All modules implicitly share mutable state through the `Parameters` object. Testing any module in isolation requires constructing the full shared state.

2. **No pipeline abstraction.** The build_model() function is a monolithic procedural script. There is no way to run stages independently, skip stages, or resume from checkpoints.

3. **No error recovery.** If any stage fails, cleanup code (storage close, temp deletion) does not execute because the cleanup block is not in a finally clause.

4. **Sequential multi-run execution.** The `num_runs` loop is sequential despite models being independent. No multiprocessing or distributed execution.

5. **Redundant storage initialization.** `storage_setup()` is called twice: once inside `setup_model()` and once directly in `build_model()`.

6. **Mixed abstraction levels.** Some modules (Closures) inherit from multiple parent classes (Horizons, Geomodel, Parameters), while others use composition. No consistent pattern.

7. **Hard-coded constants.** Threshold values (0.25, 0.45, 50 pixels), salt properties (2.17 g/cc, 4500 m/s), water properties (1.028 g/cc, 1500 m/s) are embedded in code rather than configuration.

8. **os.system() for shell operations.** Uses `os.system("rm -rf ...")` and `os.system("chmod ...")` instead of Python's shutil/os.chmod, creating potential command injection risks.

9. **No input validation.** The configuration JSON has no schema validation. Invalid parameter combinations fail silently or at runtime deep in the pipeline.

10. **Deprecated code retained.** Channel simulation (fluvsim/meanderpy) is disabled but all code remains, adding ~1600 lines of dead code.

---

## 11. External Dependencies

### Critical Dependencies

| Package | Version | Usage |
|---------|---------|-------|
| numpy | >=2 | Array operations throughout |
| scipy | — | Interpolation, filtering, spatial, ndimage, signal |
| scikit-image | — | Morphology, measure (flood-fill, labeling) |
| dask[distributed] | >=2025.9.1 | Parallel array computation |
| zarr | — | Chunked array storage (v3) |
| bruges | >=0.5.4 | Zoeppritz equations, elastic moduli |
| numba | >=0.62.0 | JIT compilation (meanderpy migration) |
| perlin-noise | >=1.12 | Stochastic noise fields |
| matplotlib | — | Plotting and QC visualization |
| plotly | — | Interactive 3D visualization |

### Optional Dependencies

| Package | Usage |
|---------|-------|
| cubed | Larger-than-memory operations (imported but unused) |
| intel-fortran-rt | Fortran channel simulation (deprecated feature) |
