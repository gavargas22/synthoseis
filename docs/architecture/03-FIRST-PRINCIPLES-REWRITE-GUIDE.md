# Synthoseis First-Principles Rewrite Guide

## Purpose of This Document

This document captures the essential domain knowledge, algorithms, and design requirements extracted from the existing Synthoseis codebase. It is written as a specification for an agentic development system to rewrite the application from scratch without porting any existing code.

The existing codebase (~15,000 LOC) is functional but architecturally convoluted, with tight coupling via the Borg pattern, no test coverage of core logic, mixed abstraction levels, and accumulated technical debt. A clean rewrite should preserve the scientific correctness while adopting modern software engineering practices.

---

## Part 1: What Synthoseis Does

### 1.1 Domain Definition

Synthoseis is a **synthetic seismic data generator** for deep learning training. It produces:
- 3D seismic amplitude volumes (realistic waveforms at multiple angle stacks)
- Labeled 3D volumes for supervised learning:
  - Fault segmentation masks
  - Hydrocarbon closure masks (by trap type and fluid type)
  - Lithology/facies volumes
  - Geological age models
  - Horizon depth maps

### 1.2 Core Pipeline (Invariant Sequence)

The generation process MUST follow this order because each stage depends on previous outputs:

```
1. Configure      → Load parameters, set up storage
2. Build Horizons → Create 2D layer boundary surfaces (depth maps)
3. Assign Facies  → Label each layer as sand or shale
4. Build 3D Model → Convert 2D horizons to 3D age volume
5. Apply Faults   → Deform the 3D model with structural discontinuities
6. Insert Salt    → (Optional) Add salt diapir and modify horizons
7. Find Closures  → Identify hydrocarbon traps via flood-fill
8. Compute Physics → Calculate elastic properties (Vp, Vs, Rho)
9. Generate Seismic → Apply Zoeppritz equations, noise, filtering
10. Output         → Write final volumes and metadata
```

### 1.3 What Varies Between Runs

Every run produces a unique geological realization through controlled randomization of:
- Layer count, thicknesses, dip angles, and azimuths
- Fault count, positions, orientations, throws, and patterns
- Sand/shale distribution (Markov chain with random parameters)
- Onlap episode positions (1-7 tilting events)
- Basin floor fan locations and geometries (optional, 1-3 fans)
- Salt body position, radius, and tilt (optional)
- Seismic noise level, frequency band, and noise type
- Rock property depth offsets and scaling factors

---

## Part 2: Domain Concepts

### 2.1 Horizons and Depth Maps

A **horizon** is a 2D surface representing a geological layer boundary in depth (or time). The collection of horizons forms a **depth map stack** of shape `(X, Y, N_horizons)`.

**Key properties:**
- Horizons are ordered: shallowest (youngest) at index 0, deepest (oldest) at last index
- Layer thickness = depth[k+1] - depth[k]; must be non-negative
- Horizons have lateral variation: each (x,y) position has its own depth value
- The number of horizons is variable (depends on thickness distribution and target seabed depth)

### 2.2 Facies

**Facies** are lithological categories assigned per layer:
- `-1` = water (above seabed)
- `0` = shale (impermeable mudstone)
- `1` = sand (permeable sandstone, potential reservoir)

Facies assignment uses a two-state Markov chain controlled by target sand fraction and average sand bed thickness. This produces realistic stratigraphic alternations.

### 2.3 Geologic Age Model

A 3D volume where each voxel holds a continuous **age value** (float). Age 0 = youngest (top), increasing downward. Ages are interpolated between horizon depths at each lateral position, creating smooth transitions within layers.

### 2.4 Faults

**Faults** are structural discontinuities where rock has been displaced. Each fault has:
- A **fault plane** (3D surface defined by a rotated ellipsoid)
- A **throw** (maximum vertical displacement)
- A **displacement field** (3D volume describing how each voxel moves)

Multiple fault patterns are supported: random, self-branching, staircase, relay ramp, horst-and-graben.

Faults must be applied sequentially because each fault deforms the cumulative state.

### 2.5 Salt Bodies

**Salt** is a mobile evaporite rock that forms diapirs (vertical intrusions). Modeled as:
- Two concentric multi-ring point clouds (top and base) defining a tapered column
- Convex hull of all points defines the salt boundary
- Salt body mutes horizons (NaN) and applies downward drag to surrounding strata
- Fixed elastic properties: Rho=2.17 g/cc, Vp=4500 m/s, Vs=2250 m/s

### 2.6 Closures (Hydrocarbon Traps)

**Closures** are structural or stratigraphic geometries that can trap hydrocarbons:
- **Simple (4-way):** Anticlines sealed by dip on all sides
- **Faulted:** Traps bounded by one or more fault planes
- **Onlap:** Traps sealed by younger strata terminating against older surfaces
- **Salt-bounded:** Traps sealed by salt body
- **False:** Apparent closures that cannot actually trap hydrocarbons (downlap)

Identification uses a **priority-queue flood-fill algorithm** on each horizon surface, with column height limits and minimum voxel count filtering.

### 2.7 Elastic Properties

Rock properties at each voxel, depth-dependent via polynomial trends:
- **Rho** (density, g/cc): Increases with depth (compaction)
- **Vp** (P-wave velocity, m/s): Increases with depth (lithification)
- **Vs** (S-wave velocity, m/s): Increases with depth

Different trends for shale, brine-sand, oil-sand, and gas-sand. Sand voxels are mixed with shale via net-to-gross weighting using either inverse-velocity or Backus moduli averaging.

### 2.8 Seismic Reflectivity

At each interface between layers, the **Zoeppritz equations** compute angle-dependent reflection coefficients from the contrast in elastic properties above and below. This produces realistic amplitude-versus-offset (AVO) behavior essential for deep learning of seismic interpretation.

### 2.9 Onlap Episodes

Tilting events where existing strata are rotated (1-7 per model). Creates angular unconformities where younger, flatter layers progressively bury older, tilted layers.

### 2.10 Basin Floor Fans

Submarine fan deposits with teardrop-shaped thickness maps. Generated from parametric surfaces with random scale, aspect ratio, asymmetry, and orientation. 1-3 fans per model when enabled.

---

## Part 3: Algorithms to Implement

### 3.1 Horizon Generation

**Input:** Configuration parameters (cube dimensions, thickness range, seabed depth, dip factor)
**Output:** 3D array `depth_maps(X, Y, N_horizons)` of horizon depths

**Algorithm:**

1. **Pre-generate lookup tables** for N layers:
   - Thicknesses: Gamma(alpha=4, beta=2) distribution
   - Dip angles: Power distribution skewed toward small values
   - Azimuths: Uniform [0, 360)

2. **Create base layer** (deepest horizon):
   - Fit a dipping plane to 3 control points (random azimuth, dip 0-75 deg)
   - Generate residual surface from Halton quasi-random points with normally-distributed elevation, interpolated via cubic spline
   - Combine: base = dipping_plane + residual

3. **Stack layers upward** (iterate):
   - Generate Perlin noise map for lateral thickness variation (random octave, bounds)
   - Create dipping plane for this layer
   - Compute thickness: (base_thickness - dip_component) * infill_factor * noise_factor
   - new_depth = previous_depth - thickness
   - Stop when shallowest point reaches seabed target

4. **Insert onlap episodes** (angular unconformities):
   - For each selected horizon: generate random dip and azimuth
   - Apply dipping offset to all shallower horizons' thicknesses

5. **Insert basin floor fans** (optional):
   - Generate parametric teardrop surface, decimate, interpolate, smooth
   - Expand depth map stack to accommodate new layers

6. **Insert seafloor** at shallowest horizon

### 3.2 Facies Generation

**Input:** Number of layers, target sand fraction, sand thickness, onlap/fan lists
**Output:** 1D array of facies codes per layer

**Algorithm:**
1. Initialize 2-state Markov chain:
   - P(shale→sand) = sand_fraction / (sand_thickness * (1 - sand_fraction))
   - P(sand→shale) = 1 / sand_thickness
2. Generate sequence of states (sand=1, shale=0)
3. Force shale below onlap horizons (seal)
4. Force sand at fan horizons; shale above/below fans

### 3.3 3D Age Model

**Input:** depth_maps(X, Y, N_horizons)
**Output:** geologic_age(X, Y, Z) — continuous age values

**Algorithm:**
For each (i,j) position independently (parallelizable):
1. Extract horizon depths at this location
2. Use `np.interp()` to interpolate age values at all Z samples
3. X-values: full Z-range; X-points: horizon depths; Y-points: integer indices

Use infill factor for sub-sample precision, then downsample.

### 3.4 Fault Generation

**Input:** Configuration (fault count, mode, throw range), geomodel, depth_maps
**Output:** Displaced geomodel, fault planes, fault intersections

**Algorithm per fault:**

1. **Generate fault parameters** based on mode:
   - Ellipsoid semi-axes, center position, tilt, throw
   - Five patterns: random, self-branching, staircase, relay, horst-graben

2. **Create rotated ellipsoid:**
   - Apply 3D rotation using matrix exponential (Lie algebra)
   - Extract fault surface via Sobel edge detection

3. **Compute displacement field:**
   - Vertical: General Gaussian at fault-cube intersection depth
   - Horizontal: 2D multivariate normal rotated by fault strike angle
   - Hockey stick: FFT convolution for large faults (drag zone)

4. **Apply displacement** to geologic age via interpolation

5. **Update fault planes:** Apply previous faulting, add new, threshold

6. **Detect intersections:** Values > 1 after fault addition

**After all faults:**
- Re-interpolate horizons from faulted age cube (parallelizable per-trace)
- Restore zero-thickness layers at onlaps/fans
- Create depth_maps_gaps with NaN at faults

### 3.5 Salt Generation

**Input:** Configuration, depth_maps
**Output:** Binary salt volume, modified depth maps

**Algorithm:**
1. Generate 218 boundary points: two multi-ring point clouds (top/base) at random positions
2. Compute Delaunay triangulation; test all voxels for hull membership
3. Apply vertical drag: shift horizons in salt zone by 2 samples per layer
4. Smooth, remove negative thicknesses, mute horizons inside salt

### 3.6 Closure Identification

**Input:** Faulted depth maps, facies, fault planes, onlap segments
**Output:** Classified closure volumes (by type and fluid)

**Algorithm per horizon:**
1. Mark fault gaps, dilate, create boundary walls
2. Run priority-queue flood-fill from edges inward
3. Limit closure heights; label connected components; remove small regions
4. Convert 2D closures to 3D voxels
5. Classify by intersection with faults/onlaps/salt
6. Assign fluid type (oil/gas/brine)

### 3.7 Elastic Properties

**Input:** Faulted lithology, depth, net-to-gross, closure fluid types
**Output:** 3D volumes of Rho, Vp, Vs

**Algorithm per layer:**
1. Compute shale properties from depth trend at randomized depth
2. For sand voxels: compute sand properties, determine fluid type, mix with shale
3. Mixing methods: inverse-velocity (harmonic mean) or Backus moduli
4. Override salt voxels with fixed values
5. Apply scaling factors

### 3.8 Seismic Generation

**Input:** Rho, Vp, Vs volumes; angle list; noise/filter parameters
**Output:** Final seismic cubes per angle

**Algorithm:**
1. **Zoeppritz:** At each interface, for each angle, compute PP reflection coefficient from the full 4×4 equation system
2. **Noise:** Generate 3D exponential noise, weight by angle (Hilterman: cos^2 + sin^2 mixing)
3. **Bandpass:** Butterworth filter, zero-phase (filtfilt), configurable order and cutoffs
4. **Lateral filter:** Uniform box filter in XY plane
5. **Cumsum:** Integrate reflectivity along Z, apply 2-100 Hz bandpass
6. **Augmentation:** Time-depth stretch (differential compaction) and uniform random squeeze
7. **RMO:** Residual moveout via spline-interpolated velocity perturbations
8. **Normalization:** Histogram equalization to standard normal distribution

---

## Part 4: Configuration Schema

The system must accept a JSON configuration file with these parameter groups:

### 4.1 Required Parameters

```json
{
  "project": "string — RPM model name for dynamic loading",
  "project_folder": "string — output directory path",
  "work_folder": "string — temporary work directory",
  "cube_shape": [300, 300, 1250],
  "incident_angles": [7, 15, 24],
  "digi": 4,
  "infill_factor": 10,
  "initial_layer_stdev": [7.0, 25.0],
  "thickness_min": 2,
  "thickness_max": 12,
  "seabed_min_depth": [20, 50],
  "signal_to_noise_ratio_db": [7.5, 12.5, 17.5],
  "bandwidth_low": [3.0, 6.0],
  "bandwidth_high": [20.0, 35.0],
  "bandwidth_ord": 4,
  "dip_factor_max": 2.0,
  "min_number_faults": 1,
  "max_number_faults": 6,
  "pad_samples": 10,
  "closure_types": ["simple", "faulted", "onlap"],
  "min_closure_voxels_simple": 500,
  "min_closure_voxels_faulted": 2500,
  "min_closure_voxels_onlap": 500,
  "max_column_height": [150.0, 150.0],
  "sand_layer_thickness": 2,
  "sand_layer_fraction": {"min": 0.05, "max": 0.25}
}
```

### 4.2 Feature Toggles (Boolean)

```json
{
  "include_salt": false,
  "basin_floor_fans": false,
  "partial_voxels": true,
  "variable_shale_ng": false,
  "extra_qc_plots": false,
  "model_qc_volumes": false,
  "broadband_qc_volume": false,
  "verbose": false,
  "multiprocess_bp": false
}
```

### 4.3 Randomized Parameters (Computed from Config Ranges)

| Parameter | Distribution | Config Source |
|-----------|-------------|---------------|
| `initial_layer_stdev` | Uniform | `initial_layer_stdev[0..1]` |
| `lateral_filter_size` | Choice {1,3,5} | None |
| `sn_db` | Triangular | `signal_to_noise_ratio_db[0..2]` |
| `sand_layer_pct` | Uniform | `sand_layer_fraction.min/max` |
| `seabed_min_depth` | Uniform int | `seabed_min_depth[0..1]` |
| `lowfreq` | Uniform | `bandwidth_low[0..1]` |
| `highfreq` | Uniform | `bandwidth_high[0..1]` |
| `noise_type` | Choice | {"random", "coherent_frowns", "coherent_smiles"} |
| `fault_mode` | Choice {0,1,2} | None |
| `fault_clustering` | Choice {0,1,2} | None |
| `number_faults` | Mode-dependent range | `min/max_number_faults` |

---

## Part 5: Architecture Recommendations for Rewrite

### 5.1 Replace Borg Pattern with Dependency Injection

The Borg pattern creates implicit global state that is hard to test and reason about. Instead:

- Define a **frozen dataclass** or **Pydantic model** for configuration
- Pass configuration explicitly to each pipeline stage
- Each stage receives only what it needs (interface segregation)
- No mutable shared state between stages

### 5.2 Define a Pipeline Abstraction

Replace the monolithic `build_model()` with a composable pipeline:

```
Pipeline = [
    ConfigStage,
    HorizonStage,
    FaciesStage,
    GeomodelStage,
    FaultStage,
    SaltStage,        # optional
    ClosureStage,
    ElasticStage,
    SeismicStage,
    OutputStage,
]
```

Each stage should:
- Declare its inputs and outputs as typed interfaces
- Be independently testable
- Support checkpointing (save/resume from any stage)
- Support dry-run (validate configuration without execution)

### 5.3 Validate Configuration

Implement schema validation on the JSON config:
- Use Pydantic or JSON Schema
- Validate parameter ranges (thickness_min < thickness_max, etc.)
- Validate feature toggle dependencies (closure_types vs. onlap enable)
- Fail fast with clear error messages

### 5.4 Use Proper Error Handling

- Wrap pipeline execution in try/finally for cleanup
- Define custom exception hierarchy for domain errors
- Log errors with context (which stage, which parameters)
- Support partial cleanup on failure

### 5.5 Make Rock Physics Models Pluggable

Replace conditional imports with a proper plugin system:
- Define an abstract base class / protocol for RPM
- Use entry points or a registry pattern
- Allow external RPM packages

### 5.6 Separate Concerns

| Concern | Current | Recommended |
|---------|---------|-------------|
| Configuration | Borg + JSON reading + directory creation + storage setup all in Parameters | Separate config parser, directory manager, storage factory |
| Logging | write_to_logfile + sqldict + write_sqldict_to_db mixed into Parameters | Separate logging and metrics collection |
| Storage | Zarr-specific API throughout | Abstract storage interface with Zarr implementation |
| Visualization | QC plots embedded in pipeline stages | Separate visualization module invoked post-pipeline |
| Randomization | Inline np.random calls everywhere | Centralized RNG with seed management for reproducibility |

### 5.7 Manage Random Seeds

The current system has no seed management. For reproducibility:
- Accept optional global seed in config
- Derive per-stage seeds deterministically from global seed
- Record all seeds in output metadata
- Support exact reproduction of any previous run

### 5.8 Support Parallel Multi-Run Execution

The current `num_runs` loop is sequential. Since runs are independent:
- Support multiprocessing/distributed execution
- Each run gets its own storage instance and output directory
- Orchestration layer manages run scheduling

### 5.9 Modernize I/O

Replace:
- `os.system("rm -rf ...")` → `shutil.rmtree()`
- `os.system("chmod ...")` → `os.chmod()` with recursive walk
- `os.stat()` existence checks → `Path.exists()`
- String path manipulation → `pathlib.Path`

### 5.10 Type Safety

- Use type hints throughout (the existing code uses them sparingly)
- Define typed data structures for intermediate pipeline products
- Use Protocol classes for duck-typed interfaces

---

## Part 6: Data Structures Specification

### 6.1 Core Arrays

| Name | Shape | Type | Range | Description |
|------|-------|------|-------|-------------|
| `depth_maps` | (X, Y, N_hz) | float32 | [0, Z*infill] | Horizon depth surfaces in sample units |
| `facies` | (N_hz+1,) | int8 | {-1, 0, 1} | Per-layer lithology code |
| `geologic_age` | (X, Y, Z) | float32 | [0, N_hz] | Continuous age per voxel |
| `onlap_segments` | (X, Y, Z) | float32 | [0, N_onlaps] | Onlap proximity label |
| `fault_planes` | (X, Y, Z) | uint8 | {0, 1} | Binary fault segmentation |
| `fault_intersections` | (X, Y, Z) | uint8 | {0, 1} | Multiple fault overlap |
| `displacement_vectors` | (X, Y, Z) | float32 | [-max, +max] | Z-displacement field |
| `faulted_age` | (X, Y, Z) | float32 | [0, N_hz] | Age after faulting |
| `faulted_lithology` | (X, Y, Z) | float32 | {-1,0,1,2} | Lithology after faulting (2=salt) |
| `net_to_gross` | (X, Y, Z) | float32 | [0, 1] | Sand fraction per voxel |
| `faulted_depth` | (X, Y, Z) | float32 | [0, max_depth] | Depth below mudline |
| `salt_segments` | (X, Y, Z) | uint8 | {0, 1} | Binary salt body |
| `rho` | (X, Y, Z) | float32 | [1.0, 2.5] | Density (g/cc) |
| `vp` | (X, Y, Z) | float32 | [1500, 5000] | P-wave velocity (m/s) |
| `vs` | (X, Y, Z) | float32 | [0, 3000] | S-wave velocity (m/s) |
| `rfc` | (N_ang, X, Y, Z-1) | float32 | [-1, 1] | Reflection coefficients |
| `oil_closures` | (X, Y, Z) | uint8 | {0, 1} | Oil-bearing voxels |
| `gas_closures` | (X, Y, Z) | uint8 | {0, 1} | Gas-bearing voxels |
| `seismic` | (N_ang, X, Y, Z) | float32 | normalized | Final seismic output |

### 6.2 Coordinate Convention

```
Array indexing: array[inline, crossline, depth_sample]
- inline (i):    0 to X-1, physical X direction
- crossline (j): 0 to Y-1, physical Y direction
- depth (k):     0 to Z-1, depth increases with k
```

### 6.3 Z-axis Padding

All 3D volumes include `pad_samples` extra samples in the Z direction to prevent boundary artifacts during filtering and convolution. The actual output cube is `cube_shape`, but internal arrays are `cube_shape + (0, 0, pad_samples)`.

---

## Part 7: Key Constants and Physical Values

### 7.1 Water Properties
- Density: 1.028 g/cc
- Vp: 1500 m/s
- Vs: 1000 m/s (note: physically Vs=0 in water; value used for numerical stability)

### 7.2 Salt Properties
- Density: 2.17 g/cc (halite)
- Vp: 4500 m/s
- Vs: 2250 m/s

### 7.3 Threshold Values
- Fault plane edge detection: Sobel values < 0.5 zeroed
- Binary segmentation threshold: 0.45
- Fault intersection detection: values > 1.1
- Flood-fill minimum region size: 50 pixels
- Sentinel depth value for flood-fill: 1.0e5

### 7.4 Default Storage
- Chunk size: (128, 128, 128)
- Default dtype: float32
- Storage format: Zarr v3

---

## Part 8: Probability Distributions Reference

| Distribution | Parameters | Usage |
|-------------|-----------|-------|
| **Gamma(4, 2)** | alpha=4, beta=2 | Layer thicknesses |
| **Power(100)** | `(1-U^0.01)*scale` | Dip angles (skewed small) |
| **Triangular(a,b,c)** | left, mode, right | SNR, fault throw, layer shift, salt radius |
| **Uniform(a,b)** | min, max | Azimuths, frequencies, sand fraction, depth offsets |
| **Binomial(1, p)** | p=probability | Channel occurrence (p=0.03), sand/shale |
| **Normal(0, sigma)** | mean=0, std=sigma | Residual elevation for structure maps |
| **Exponential(lambda)** | lambda=1/100 | Seismic noise amplitudes |
| **Markov(P)** | Transition matrix P | Facies sequence generation |

---

## Part 9: Dependencies for Rewrite

### 9.1 Required Scientific Libraries

| Library | Purpose | Alternatives |
|---------|---------|-------------|
| **numpy** | Array operations | — (fundamental) |
| **scipy** | Interpolation, filtering, spatial, ndimage | — (covers too many needs) |
| **scipy.signal** | Butterworth design, filtfilt | — |
| **scipy.spatial** | Delaunay triangulation for convex hull | — |
| **scipy.ndimage** | Sobel, maximum_filter, uniform_filter, Gaussian | — |
| **scikit-image** | Connected components (label), morphology | — |

### 9.2 Optional/Replaceable Libraries

| Library | Current Use | Can Replace With |
|---------|------------|-----------------|
| **bruges** | Zoeppritz equations, elastic moduli | Implement directly (well-documented equations) |
| **perlin-noise** | Perlin noise generation | Implement with numpy (simplex noise) |
| **numba** | JIT for channel migration | Not needed if channels deprecated |
| **dask** | Parallel array computation | Could use multiprocessing or native chunking |
| **zarr** | Chunked storage | HDF5, NetCDF, or custom |
| **plotly** | 3D visualization | matplotlib 3D or pyvista |

### 9.3 New Recommended Libraries

| Library | Purpose |
|---------|---------|
| **pydantic** | Configuration validation and typing |
| **typer** or **click** | Modern CLI framework |
| **structlog** | Structured logging |
| **pytest** | Unit and integration testing |
| **xarray** | Labeled multi-dimensional arrays (natural fit for seismic data) |

---

## Part 10: Test Strategy for Rewrite

### 10.1 Unit Tests

Each algorithm should have dedicated unit tests:

| Component | Test Strategy |
|-----------|--------------|
| Horizon generation | Verify depth ordering, non-negative thickness, seabed target reached |
| Facies Markov chain | Verify transition probabilities match target sand fraction |
| Age model interpolation | Verify monotonicity, boundary values at horizons |
| Fault displacement | Verify displacement field symmetry, throw magnitude |
| Flood-fill closures | Test on known geometries (dome, half-graben) |
| Zoeppritz equations | Compare against published values for known media |
| Butterworth filter | Verify frequency response at design points |
| Mixing models | Verify end-member behavior (ng=0 → pure shale, ng=1 → pure sand) |
| Salt convex hull | Verify all boundary points inside hull, points outside are outside |

### 10.2 Integration Tests

| Test | Verification |
|------|-------------|
| Minimal pipeline (32^3 cube) | Runs without error, produces all expected outputs |
| Configuration validation | Rejects invalid configs with clear messages |
| Deterministic output | Same seed → identical results |
| Multi-run isolation | Run N models; verify no state leakage between runs |
| Storage round-trip | Write → read → compare for all array types |

### 10.3 Property-Based Tests

| Property | Assertion |
|----------|-----------|
| All depths in depth_maps | Non-decreasing along Z axis at each (x,y) |
| All layer thicknesses | Non-negative |
| Facies codes | All values in {-1, 0, 1} (or {-1, 0, 1, 2} with salt) |
| Elastic properties | Vp > 0, Vs >= 0, Rho > 0 everywhere |
| Poisson's ratio | 0 < PR < 0.5 (physically valid) |
| RFC values | |R| <= 1 for all angles |
| Fault planes | Binary {0, 1} after thresholding |
| Closures | Oil + gas + brine mutually exclusive per voxel |

---

## Part 11: Glossary of Domain Terms

| Term | Definition |
|------|-----------|
| **AVO** | Amplitude Versus Offset — variation of seismic reflection amplitude with angle |
| **Acoustic Impedance (AI)** | Product of density and P-wave velocity (Rho * Vp) |
| **Azimuth** | Compass direction of maximum geological dip (0-360 degrees) |
| **Backus averaging** | Method for computing effective elastic properties of layered media |
| **Basin floor fan** | Submarine sediment deposit with teardrop-shaped morphology |
| **Butterworth filter** | Maximally-flat magnitude response bandpass filter |
| **Closure** | Structural or stratigraphic geometry that can trap hydrocarbons |
| **Convex hull** | Smallest convex shape enclosing a set of points |
| **Digi** | Vertical sampling interval (typically 4 milliseconds) |
| **Dip** | Angle of geological surface from horizontal |
| **Facies** | Rock type classification (e.g., sand, shale) |
| **Fault** | Structural discontinuity where rock has been displaced |
| **Fault throw** | Maximum vertical displacement across a fault |
| **Flood-fill** | Algorithm to identify enclosed regions on a surface |
| **Graben** | Structural low bounded by parallel normal faults |
| **Halton sequence** | Low-discrepancy quasi-random point distribution |
| **Hockey stick** | Nonlinear displacement pattern near fault terminations |
| **Horizon** | 2D surface representing a geological layer boundary |
| **Horst** | Structural high bounded by parallel normal faults |
| **Infill factor** | Vertical resolution multiplier for sub-sample precision |
| **Lame parameters** | Fundamental elastic constants (lambda, mu) |
| **Net-to-gross (N/G)** | Fraction of reservoir-quality rock in a layer |
| **Onlap** | Younger strata terminating against an older tilted surface |
| **Perlin noise** | Gradient-based coherent noise function for spatial variation |
| **Poisson's ratio** | Ratio of lateral strain to axial strain |
| **Ricker wavelet** | Second derivative of Gaussian; standard seismic source pulse |
| **RMO** | Residual Moveout — velocity-dependent time shifts between angle stacks |
| **RPM** | Rock Property Model — defines elastic property trends with depth |
| **Salt diapir** | Vertical intrusion of mobile salt rock |
| **Seabed** | Ocean floor; shallowest geological horizon |
| **Shear impedance (SI)** | Product of density and S-wave velocity (Rho * Vs) |
| **Simplex noise** | Improved variant of Perlin noise |
| **Spill point** | Lowest point on a closure boundary where hydrocarbons escape |
| **Strike** | Compass direction perpendicular to dip |
| **Vp** | P-wave (compressional) velocity |
| **Vs** | S-wave (shear) velocity |
| **Zoeppritz equations** | Exact solution for reflection/transmission at elastic interface |

---

## Part 12: File Outputs Specification

Each model run must produce:

### 12.1 Seismic Data
- One 3D volume per angle stack (e.g., 7°, 15°, 24°)
- Both raw and normalized versions
- Cumulative sum volumes
- Optional: broadband QC volume, RMO-applied volumes

### 12.2 Labels
- Fault plane segmentation (binary)
- Fault intersection zones (binary)
- Oil closure volume (binary)
- Gas closure volume (binary)
- Brine closure volume (binary)
- Closure type classification (simple, faulted, onlap, salt, false)
- Lithology/facies volume
- Salt body segmentation (binary, if enabled)

### 12.3 Geological Models
- Horizon depth maps (faulted and unfaulted)
- Geologic age volume
- Net-to-gross volume
- Depth below mudline volume
- Onlap segment volume

### 12.4 Elastic Properties (Optional QC)
- Rho, Vp, Vs volumes
- Raw reflection coefficients per angle

### 12.5 Metadata
- Configuration parameters used (including all random values)
- Survey geometry keyfile (SEG-Y compatible)
- Elapsed time and processing log
- Parameter database (SQLite or equivalent)
