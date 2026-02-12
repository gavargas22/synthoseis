# Synthoseis Pipeline Processes — Complete Reference

This document describes every process in the Synthoseis generation pipeline, from configuration loading through final seismic output. Each section documents the algorithms, mathematical models, data transformations, and design decisions for one pipeline stage.

---

## Process 1: Configuration and Initialization

### 1.1 Entry Point

`main.py` parses CLI arguments:
- `-c / --config_file` (required): Path to JSON configuration
- `-n / --num_runs` (default: 1): Number of sequential model runs
- `-r / --run_id` (default: None): Identifier appended to output folder names
- `-t / --test_mode` (default: None): Integer to reduce cube dimensions

### 1.2 Run Loop

For each run in `range(num_runs)`:
1. Generate RPM scaling factors (random layershift, RPshift; all multipliers default to 1.0)
2. Call `build_model(config_file, run_id, test_mode, rpm_factors)`

Runs are sequential. Each run gets unique timestamp-based output directories.

### 1.3 Parameters Initialization

The `Parameters` class uses the Borg pattern (shared `__dict__`). On construction:

1. **Reset shared state** — `self._shared_state = {}` clears previous run
2. **Read JSON config** — Loads all parameters from file
3. **Create directory structure:**
   - `project_folder/seismic__{timestamp}_{runid}/` — Final output
   - `work_folder/temp_folder__{timestamp}/` — Temporary data
4. **Write key file** — SEG-Y-compatible geometry metadata
5. **Open storage** — `StorageClient.open(path, mode="a")` creates Zarr v3 store
6. **Randomize parameters:**
   - `initial_layer_stdev` — Uniform within configured range
   - `lateral_filter_size` — Random choice from {1, 3, 5}
   - `sn_db` — Triangular distribution within configured bounds
   - `sand_layer_pct` — Uniform within configured fraction range
   - `seabed_min_depth` — Random integer if range provided
   - `lowfreq`, `highfreq` — Uniform within configured bands
   - Noise type — Random selection: "random", "random_coherent_frowns", "random_coherent_smiles"
7. **Randomize fault settings:**
   - `mode` — Random 0/1/2 (random, clustered, horst-graben)
   - `clustering` — Random 0/1/2 (self-branching, staircase, relay)
   - `number_faults` — Mode-dependent range
   - Throw ranges — Computed from infill_factor and mode
8. **Initialize RPM scaling factors** — Layer shift, property shift, all factor multipliers

### 1.4 Storage Backend

The `StorageClient` wraps Zarr v3:
- Path: `temp_folder/model_data.zarr`
- Chunking: `(128, 128, 128)` default, clamped to array dimensions
- API: `create_dataset`, `get_dataset` (numpy/dask/lazy), `list_datasets`, `remove_dataset`, `close`
- Context manager support (`with` statement)

---

## Process 2: Horizon Generation

### 2.1 Overview

Horizons are 2D depth surfaces representing geological layer boundaries. The system builds horizons from bottom up, stacking layers with random thicknesses and dips until reaching the configured seabed depth.

### 2.2 Random Horizon Stack Creation

**Algorithm: `RandomHorizonStack.create_depth_maps()`**

**Step 1: Generate lookup tables.** Pre-compute random parameters for up to `num_lyr_lut` layers:
- **Thicknesses** — Gamma distribution (alpha=4, beta=2): produces realistic sedimentary thicknesses skewed toward moderate values
- **Dips** — Power distribution: `(1 - U^0.01) * 7.0 * dip_factor_max` where U ~ Uniform[0,1]. Most dips are small; few are steep.
- **Azimuths** — Uniform [0, 360) degrees
- **Onlap flags** — 1-7 random layer indices for tilting episodes
- **Channel flags** — Binomial(1, 0.03) per layer (3% probability, deprecated)

**Step 2: Create base layer.** Generate a random depth structure map for the deepest horizon:
1. Create dipping plane via least-squares fit to 3 control points (random azimuth, dip 0-75 degrees)
2. Generate residual structure: scatter 2-25 Halton quasi-random points with normally-distributed elevation offsets
3. Interpolate residual surface via cubic spline (scipy.griddata) with 15% grid padding
4. Combine: `base_layer = dipping_plane + residual_structure`

**Step 3: Stack layers upward.** Loop up to 20,000 iterations:
1. Generate random thickness factor map using Perlin noise (octave from triangular 1.3-5.25; bounds from triangular distributions)
2. Create thickness map: `thickness = (base_thickness - dipping_plane_component) * infill_factor * thickness_factor`
3. Clip to [0, max_thickness * 1.5]
4. Compute current depth: `current_depth = previous_depth - thickness_map`
5. Stack via `np.dstack()` — shallowest at index 0
6. **Stop when** minimum depth reaches seabed target: `current_depth.min() <= seabed_min_depth * infill_factor / digi`

**Output:** `depth_maps` array of shape `(X, Y, num_horizons)` in sample units.

### 2.3 Onlap (Tilting) Episodes

**Algorithm: `Onlaps.insert_tilting_episodes()`**

For each selected onlap horizon (1-7 horizons), iterate from deep to shallow:
1. Generate random dip (5-20 degrees) and azimuth (0-360 degrees)
2. Create dipping plane at the onlap horizon
3. For all shallower horizons: add dipping plane offset to thickness, clip to [0, max]
4. This creates angular unconformities where younger layers progressively bury older tilted layers

**Geological meaning:** Represents tectonic tilting events followed by renewed deposition.

### 2.4 Basin Floor Fans

**Algorithm: `BasinFloorFans.insert_fans_into_horizons()`**

Randomly selects 1-3 layers for fan insertion. For each:

1. **Generate parametric fan shape:** 180x180 points on parametric surface:
   ```
   x = 0.5 * (1 - cos(theta)) * sin(theta) * cos(phi)
   y = cos(theta)
   z = 0.5 * (1 - cos(theta)) * sin(theta) * sin(phi)
   ```
2. **Scale** by random dimensions (length 50-200, aspect 1.5-4.0)
3. **Apply asymmetry** via rotation and stretch (factor -1 to 1)
4. **Decimate** from 32,400 to ~130 points with added noise
5. **Interpolate** sparse points onto grid via cubic griddata
6. **Smooth** with Gaussian filter and grey closing morphology
7. **Insert** into horizon stack by expanding Z-dimension and shifting horizons below

**Optional fan pairs:** 50% chance of creating a second fan nearby with rotated offset.

### 2.5 Facies Assignment

**Algorithm: `Facies.sand_shale_facies_markov()`**

Uses a 2-state Markov chain to generate realistic sand/shale alternations:

**Transition matrix:**
```
P = | 1-alpha   alpha  |     where alpha = sand_pct / (sand_thickness * (1 - sand_pct))
    | beta      1-beta |           beta  = 1 / sand_thickness
```

- `alpha` = probability of shale-to-sand transition
- `beta` = probability of sand-to-shale transition
- Larger `sand_thickness` → longer sand sequences

**Post-processing:**
- Layers below onlap horizons forced to shale (unconformity seal)
- Fan layers set to sand; layers above/below fans set to shale

**Output:** 1D array of shape `(num_horizons+1,)` with values {-1=water, 0=shale, 1=sand}.

### 2.6 Seafloor Insertion

Sets the shallowest horizon (index 0) to water bottom depth:
```
wb_time_map = second_horizon_depth - 1.5 samples
```
Converts to physical units via `digi / infill_factor`.

---

## Process 3: Geomodel Construction

### 3.1 Overview

Converts 2D horizon depth maps into a 3D volumetric representation where each voxel carries a continuous geological age value.

### 3.2 Age Model Creation

**Algorithm: `create_geologic_age_3d_from_infilled_horizons()`**

**Infilled resolution:** Internal computation uses Z-dimension multiplied by `infill_factor` for sub-sample precision.

For each lateral position (i,j) — parallelized via Dask delayed:

1. Find the deepest valid horizon index at this location
2. Extract horizon depths as 1D array
3. Interpolate ages across all Z samples using `np.interp()`:
   - X-values (output): full Z-range as linspace
   - X-points (input): horizon depths at this location
   - Y-points (input): integer horizon indices (0, 1, 2, ...)
4. Assign interpolated trace to age volume

**Result:** Continuous age values between horizon boundaries, creating smooth transitions within layers.

### 3.3 Anti-aliasing

After interpolation at infilled resolution:
- **Simple filter:** Stride-slice every `infill_factor`-th sample in Z: `cube[..., ::infill_factor]`
- **Advanced filter (defined but unused):** Apply `maximum_filter` before downsampling

### 3.4 Onlap Surface Labeling

For each onlap horizon, label a neighborhood of voxels (±1.5 * infill_factor samples) using fancy indexing with meshgrids. Values are additive — multiple nearby onlaps accumulate.

**Output:** `onlap_segments` volume where non-zero values indicate proximity to tilting events.

---

## Process 4: Fault Generation and Displacement

### 4.1 Overview

Faults are modeled as 3D rotated ellipsoids. Displacement is computed per-fault and applied sequentially to the geologic age volume. After all faults, horizons are re-interpolated from the faulted age model.

### 4.2 Fault Plane Geometry

**Representation:** Each fault is a rotated 3D ellipsoid defined by:
- Semi-axes: `a`, `b`, `c` (squared values)
- Center: `(x0, y0, z0)`
- Tilt percentage: `tilt_pct` (0-1)
- Maximum throw (vertical displacement)

**3D Rotation:** Uses scipy `linalg.expm()` (matrix exponential) applied to a strike unit vector via Lie algebra representation. The rotation parameters are:
- `theta = atan2(tilt * sqrt(distance), cube_z)`
- `dip_angle = atan2(y0 - center_y, x0 - center_x)`

**Edge Detection:** Sobel operator applied on all three axes extracts the fault surface from the ellipsoid volume. Values < 0.5 are zeroed.

### 4.3 Fault Parameter Generation Modes

Five fault pattern modes (selected based on random mode + clustering):

| Mode | Pattern | Semi-axis Ranges | Throw Range |
|------|---------|-----------------|-------------|
| Random | Uncorrelated positions | a,b: U(100,600)^2 | U(low, high) |
| Self-branching | Clustered in segments | Similar to random | 5-15 * infill |
| Staircase | Linear progression in X,Y | Segment-based | Mode-dependent |
| Relay ramps | Overlapping segments | Similar to branching | 5-15 * infill |
| Horst-graben | Regularly spaced parallel | a: U(200,800)^2; b: U(50,200)^2 | Variable |

### 4.4 Displacement Algorithm

For each fault:

**Vertical displacement (Z):**
- General Gaussian: `sigma` ~ U(10*throw-50, 300), `p` ~ U(1.5, 5)
- Positioned at fault-cube intersection depth
- Tapered to prevent breaching seafloor

**Horizontal displacement (XY):**
- 2D multivariate normal distribution
- Variance from throw-to-length scaling: `fault_length = 0.0013 * throw^1.3258`
- Rotated by alpha = `atan2(y_idx - center_y, x_idx - center_x)`

**Hockey stick effect (for large faults):**
- Applied when throw >= 85% of maximum
- Convolution of drag zone with fault plane classification via `signal.fftconvolve()`
- Creates nonlinear displacement near fault planes

**Final 3D displacement:**
```
stretch_cube = base_indices - xy_displacement * z_shift_function(z)
```
Applied to geologic age via interpolation.

### 4.5 Sequential Fault Application

Faults MUST be applied sequentially — each fault deforms the cumulative result of all previous faults:
1. Compute displacement for fault N
2. Apply displacement to current faulted_depths (from faults 0..N-1)
3. Update fault_planes: apply faulting to existing planes, add new fault, threshold
4. Detect intersections: values > 1.1 after addition indicate overlap
5. Accumulate max_fault_throw

### 4.6 Post-Faulting Horizon Re-interpolation

After all faults applied:
1. Re-interpolate horizons using the faulted geologic age cube (Dask parallel, per-trace)
2. For each trace: if age range > 0, use `np.interp()` to find new horizon depths
3. Restore zero-thickness onlap and fan layers
4. Create `depth_maps_gaps` with NaN at fault locations

### 4.7 Faulted Property Models

Iterate through depth maps layer by layer:
1. Determine fractional layer coverage for each voxel (partial voxels)
2. Assign lithology from facies array using age model lookup
3. Compute net-to-gross maps (Perlin noise based)
4. Compute depth below mudline

---

## Process 5: Salt Body Generation

### 5.1 Overview

Salt bodies are modeled as convex hulls around multi-level point clouds, creating tapering diapir-like geometries. Salt modifies depth maps through vertical drag.

### 5.2 Salt Geometry

**Two-sphere model:** Top and base salt balls connected by a convex hull.

**Top salt sphere:**
- 3 concentric rings of 36 jittered points each, plus 1 tip point (109 total)
- Rings at decreasing depths with decreasing radii (funnel shape)
- Center positioned randomly within ±40% of model center

**Base salt sphere:**
- Same ring structure but smaller (half the radius of top's shallow level)
- Deeper by 2.3-12.5x radius
- Laterally offset (30-170% of top center) to create tilt

**Salt radius:** Triangular distribution: X_dim/6 to X_dim/4 (mode at X_dim/5)

### 5.3 Convex Hull Segmentation

1. Generate all (X,Y,Z) voxel coordinates as N×3 array
2. Compute Delaunay triangulation of the 218 boundary points
3. Test each voxel against hull: `find_simplex(point) >= 0`
4. Result: binary volume (1=salt, 0=no salt)

### 5.4 Depth Map Drag

For each horizon passing through salt:
1. Increment `relative_salt_depth` counter
2. Shift horizon downward: `depth -= 2 * relative_salt_depth` (2 samples per salt layer)
3. Smooth with Gaussian filter (sigma=3) to remove artifacts
4. Remove negative thicknesses from layer inversion
5. Mute (set to NaN) depth picks inside salt body

### 5.5 Seismic Properties

Salt voxels receive fixed, depth-independent properties:
- Density: 2.17 g/cc (halite)
- Vp: 4500 m/s
- Vs: 2250 m/s

---

## Process 6: Closure (Trap) Identification

### 6.1 Overview

Closures are structural or stratigraphic traps that could hold hydrocarbons. Identification uses a priority-queue flood-fill algorithm on each horizon surface, followed by 3D voxelization and classification.

### 6.2 Flood-Fill Algorithm

**Algorithm: `_flood_fill(horizon, max_column_height)`**

For each sand horizon:

1. **Mark fault intersections** — Set values < 1.0 as empty; dilate with 3×3 kernel
2. **Create boundary walls** — Set 3-pixel border to 0; add `max_column_height` at dilated fault gaps
3. **Set sentinel** — `map[0,0] = 1.0e5` as flood-fill collection point
4. **Invert and flood** — Call `fill_to_spill()` which inverts the surface and runs `flood_fill_heap()`:
   - Initialize min-heap from edge pixels
   - Pop minimum, propagate to 4-connected neighbors
   - For each neighbor: `output = max(h_current, input[neighbor])`
   - This fills depressions from edges inward respecting topography
5. **Apply column height limit:**
   - Label connected components (8-connectivity)
   - Remove regions < 50 pixels
   - For each component: limit closure height to `max_column_height`
   - Compute spill point depths

**Output:** 2D closure depth map where `closure_depth - top_depth = closure thickness`.

### 6.3 3D Voxelization

For each horizon with closure:
1. Compute max closure thickness (rounded integer)
2. Loop k = 0 to max_closure:
   - Compute depth slice: `horizon_slice = k + top_structure_map`
   - Find voxels where slice < closure_depth_map
   - Increment `closure_segments[i, j, k]` for those voxels
3. Result: 3D volume where values indicate closure membership

### 6.4 Closure Classification

Each closure segment is classified by intersecting with structural features:

| Type | Criterion | Description |
|------|-----------|-------------|
| **Simple (4-way)** | No faults AND no onlaps intersect | Pure structural anticline |
| **Faulted** | Fault planes intersect closure | Fault-bounded trap |
| **Onlap** | Upward onlap surfaces intersect | Stratigraphic trap |
| **False** | Downward onlap surfaces intersect | Non-sealing, not a valid trap |
| **Salt-bounded** | Salt body intersects closure | Salt-sealed trap (if salt enabled) |

### 6.5 Fluid Assignment

Each closure receives a fluid type (oil, gas, or brine) based on pre-computed migration paths. Separate binary volumes are maintained for each combination of closure type and fluid type (e.g., `faulted_closures_oil`, `simple_closures_gas`).

### 6.6 Minimum Voxel Filtering

Closures with fewer than the configured minimum voxel count are discarded:
- Simple: `min_closure_voxels_simple` (typical: 500)
- Faulted: `min_closure_voxels_faulted` (typical: 2500)
- Onlap: `min_closure_voxels_onlap` (typical: 500)
- Connected component labeling uses 50-pixel minimum threshold

---

## Process 7: Rock Physics Modeling

### 7.1 Overview

Rock physics models convert geological structure (lithology, depth, fluid type) into elastic properties (Vp, Vs, Rho) needed for seismic forward modeling.

### 7.2 Dynamic Model Loading

The `select_rpm(cfg)` function conditionally imports the RPM module matching `cfg.project`:
```
cfg.project == "example" → rockphysics.rpm_example.RPMExample
```

Each RPM must implement four methods returning `RockProperties(rho, vp, vs)` objects:
- `calc_shale_properties(z_rho, z_vp, z_vs)`
- `calc_brine_sand_properties(z_rho, z_vp, z_vs)`
- `calc_oil_sand_properties(z_rho, z_vp, z_vs)`
- `calc_gas_sand_properties(z_rho, z_vp, z_vs)`

### 7.3 Depth Trends (Example Model)

Properties are polynomial functions of depth (z in meters):

**Shale:**
```
Rho(z) = 7.7e-12*z^3 - 8.8e-8*z^2 + 4e-4*z + 1.957  (g/cc)
Vp(z)  = -0.00013*z^2 + 1.13*z + 1580                  (m/s)
Vs(z)  = -0.0001*z^2 + 0.96*z + 279                     (m/s)
```

**Brine sand:**
```
Rho(z) = -7.8e-9*z^2 + 1.2e-4*z + 2.021
Vp(z)  = -1.34e-5*z^2 + 0.49*z + 2317
Vs(z)  = -1.0785e-5*z^2 + 0.391*z + 1007
```

**Oil sand:** Lower Vp than brine (AVO effect)
**Gas sand:** Lowest Vp and Rho (bright spot indicator)

### 7.4 Property Mixing

**Per-voxel process:**

1. Look up shale properties at randomized depth: `z + delta_z_layer + delta_z_property`
2. If voxel is sand (net_to_gross > 0):
   - Look up sand properties at randomized depth
   - Determine fluid type from closure volumes (brine/oil/gas)
   - Mix shale + sand using `EndMemberMixing`:
     - **Inverse velocity** (default): Harmonic mean for Vp/Vs, arithmetic for Rho
     - **Backus moduli**: Mix Lame parameters, reconstruct velocities

3. Apply scaling factors to shales and sands separately
4. Override salt voxels with fixed properties (Rho=2.17, Vp=4500, Vs=2250)
5. Fix zero values at volume base by propagating from above

### 7.5 Depth Randomization

Two levels of randomization prevent unrealistic vertical coherence:
- **Layer shift** (same for all properties): `delta_z_layer ~ U(-layershiftsamples, +layershiftsamples)`
- **Property shift** (independent per property): `delta_z_rho, delta_z_vp, delta_z_vs` each ~ U(-RPshiftsamples, +RPshiftsamples)`
- Shallow layers (z <= 20) are NOT randomized to avoid surface artifacts

### 7.6 RockProperties Class

Stores Rho, Vp, Vs and lazily computes derived quantities:
- Acoustic impedance: AI = Rho * Vp
- Shear impedance: SI = Rho * Vs
- Vp/Vs ratio
- Poisson's ratio: PR = (Vp/Vs)^2 - 2) / (2*(Vp/Vs)^2 - 2)
- P-wave modulus: M = Rho * Vp^2
- Shear modulus: mu = Rho * Vs^2
- Bulk modulus: K = M - 4/3*mu
- Lame parameter: lambda = Rho * (Vp^2 - 2*Vs^2)

Uses `bruges.rockphysics.moduli` for modulus calculations.

---

## Process 8: Seismic Volume Generation

### 8.1 Overview

Converts elastic properties into angle-dependent seismic reflectivity using Zoeppritz equations, then applies noise, filtering, augmentation, and normalization to produce final seismic cubes.

### 8.2 Zoeppritz Reflection Coefficients

**Algorithm: `RFC.zoeppritz_reflectivity()`**

At each interface (z to z+1), for each incident angle theta:

1. Compute ray parameter: `p = sin(theta) / Vp_upper`
2. Compute converted wave angles via Snell's law:
   - `theta2 = arcsin(p * Vp_lower)` (transmitted P)
   - `phi1 = arcsin(p * Vs_upper)` (reflected S)
   - `phi2 = arcsin(p * Vs_lower)` (transmitted S)
3. Construct 4x4 Zoeppritz matrix with boundary conditions
4. Solve for PP reflection coefficient (real part)

**Implementation uses complex-typed angles** for numerical stability when ray parameter exceeds critical angle.

**Chunk-based computation via Dask `map_blocks()`** — each spatial chunk processes all interfaces and angles independently.

**Output:** `rfc_raw` of shape `(n_angles, X, Y, Z-1)`.

**Alternative approximations available (not used for generation):**
- Normal incidence: `R = (Z1 - Z0) / (Z1 + Z0)`
- Shuey 3-term: `R = R0 + G*sin^2(theta) + F*(tan^2(theta) - sin^2(theta))`

### 8.3 Noise Addition

**Algorithm: `add_weighted_noise()`**

Angle-dependent noise using Hilterman weighting:

1. Generate two 3D noise fields (exponential distribution with random ±signs)
2. Compute data standard deviation below seafloor
3. Compute noise standard deviation
4. Convert S/N from dB: `std_ratio = sqrt(10^(sn_db/10))`
5. For each angle: `noise = noise_0 * cos^2(angle) + noise_45 * sin^2(angle)`
6. Scale: `normalized_noise = noise * (data_std / noise_std) / std_ratio`
7. Add to RFC: `rfc_noise_added = rfc_raw + normalized_noise`

### 8.4 Bandpass Filtering

**Butterworth bandpass** (scipy.signal):
1. Design filter: `butter(order, [low/nyq, high/nyq], btype="bandpass")`
2. Apply zero-phase filter: `filtfilt(b, a, data, method="gust")`
3. Gustafsson method reduces edge artifacts
4. Applied via Dask `map_blocks()` for memory efficiency

### 8.5 Wavelet Convolution (Optional)

When wavelet filter specifications are available:
1. Load 3 pre-computed filter specs (.npy files) for near/mid/far angles
2. Generate random percentiles for frequency band selection
3. Create frequency-domain wavelet via percentile interpolation
4. Convert to time domain via IFFT with Hanning taper
5. Convolve each trace: `np.convolve(trace, wavelet, mode="same")`

### 8.6 Lateral Filtering

Uniform (box) filter applied in XY plane only:
```
uniform_filter(data, size=(0, n_filt, n_filt, 0))
```
Filter size from config: `lateral_filter_size` in {1, 3, 5}.

### 8.7 Cumulative Sum

```
cumsum = da.cumsum(data, axis=-1)  # Integrate reflectivity along Z
bandpass(cumsum, 2Hz, 100Hz)        # Apply Ormsby-like filter
```

This converts reflectivity series to seismic impedance contrast representation.

### 8.8 Augmentation

**Time-depth stretch (`tz_stretch`):**
1. Smooth seafloor horizon with 31x31 uniform filter
2. Random squeeze parameters: max_squeeze 5-25%, max depth between seafloor and 80%
3. Create spline fit through control points (seafloor, squeeze point, bottom)
4. Resample each trace using inverse spline

**Uniform stretch (`uniform_stretch`):**
1. Random squeeze factors (0.85-1.15) for X, Y, Z dimensions
2. Interpolate seismic using scaled coordinates
3. Roll array if squeeze > 1 (compensate for clipped samples)

### 8.9 Residual Moveout (RMO)

**Algorithm: `compute_randomRMO_1D()`**

1. Generate 1-3 random tie points with velocity fractions (±1.5%)
2. Smooth interpolation via `UnivariateSpline`
3. For each angle: `output_indices = input_indices * (1 + fraction * tan^2(angle))`
4. Apply per-trace via `np.interp(range, rmo_indices, trace)`
5. Optional mean removal to preserve absolute timing

### 8.10 Amplitude Normalization

**Histogram equalization to standard normal:**
1. Compute CDF of input amplitude distribution
2. Generate reference CDF from standard normal
3. Map input CDF to normal CDF via interpolation
4. Apply cubic spline transformation
5. Result: seismic amplitudes follow N(0,1) distribution

### 8.11 Final Output

1. Scale by `100 / mean(std per angle)` for amplitude consistency
2. Apply per-angle factors: near, mid, far multipliers from RPM scaling
3. Write unscaled and normalized angle stacks to Zarr storage

---

## Process 9: Cleanup

After all pipeline stages complete:

1. **Close storage** — `p.storage.close()` flushes Zarr store
2. **Delete temp folder** — `os.system("rm -rf " + p.temp_folder)`
3. **Restore directory** — `os.chdir(p.current_dir)`
4. **Set permissions** — `os.system("chmod -R 777 " + p.work_subfolder)`
5. **Close plots** — `plt.close("all")` to free matplotlib memory
6. **Return** — `p.work_subfolder` path

**Known issue:** Cleanup is NOT in a try/finally block. If any pipeline stage raises an unhandled exception, temporary files and storage are not cleaned up.

---

## Appendix A: Key Mathematical Models

### A.1 Zoeppritz Equations (Full)
PP reflection coefficient for incident P-wave at angle theta at interface between two elastic half-spaces. Full 4x4 matrix solution accounting for P-to-P, P-to-S, S-to-P, and S-to-S wave modes.

### A.2 Shuey Approximation (3-term)
```
R(theta) = R0 + G*sin^2(theta) + F*(tan^2(theta) - sin^2(theta))
R0 = 0.5 * (dVp/Vp + dRho/Rho)
G  = 0.5 * dVp/Vp - 2*(Vs/Vp)^2 * (dRho/Rho + 2*dVs/Vs)
F  = 0.5 * dVp/Vp
```

### A.3 Ricker Wavelet
```
w(t) = (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2)
```

### A.4 Butterworth Bandpass
```
H(s) = s^N / ((s^2 + s/Q + 1)^(N/2))
Applied via forward-backward (zero-phase) filtering.
```

### A.5 Howard-Knutson Channel Migration
```
R0 = kl * curvature
R1 = sinuosity^(-2/3) * convolution(R0, exp(-alpha*s))
x_new = x + R1 * dy/ds * dt
y_new = y - R1 * dx/ds * dt
```

### A.6 Perlin Noise
Gradient noise function evaluated on normalized grid with configurable octaves. Generates spatially correlated random fields for geological property variation.

### A.7 Markov Chain Facies
Two-state (sand/shale) with transition probabilities derived from target sand fraction and average bed thickness:
```
P(shale→sand) = sand_pct / (sand_thickness * (1 - sand_pct))
P(sand→shale) = 1 / sand_thickness
```

---

## Appendix B: Parallelization Summary

| Operation | Method | Scheduler | Granularity |
|-----------|--------|-----------|-------------|
| Age model creation | dask.delayed | threads | Per-trace |
| Fault displacement | dask.delayed | threads | Per-trace |
| Horizon re-interpolation | dask.delayed | threads | Per-trace |
| Zoeppritz computation | dask.map_blocks | default | Per-chunk |
| Noise generation | dask.random | default | Per-chunk |
| Bandpass filtering | dask.map_blocks | default | Per-chunk |
| Cumulative sum | dask.cumsum | default | Full array |
| Zarr I/O | dask.to_zarr | default | Per-chunk |
