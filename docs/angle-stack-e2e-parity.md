# End-to-end angle-stack parity against legacy Python

Test-only hygiene. This adds no new feature and changes no output.

## Why

The generic parity harness (`tests/fixtures/parity_cubes_8.json`,
`synthoseis-core::parity`) checks angle stacks against a linear formula
(`0.1 i + 0.05 j + 0.02 k`), not against legacy output. Until now nothing
compared Rust and legacy angle stacks on a full model.

## Fixture

`tests/fixtures/generate_angle_stack_e2e.py` runs the **real** legacy code in
the repository's locked environment (`uv sync --frozen`: numpy 2.4.4, scipy 1.17.1,
numba 0.65.1):

1. `main.build_model` steps up to `SeismicVolume.build_elastic_properties("inv_vel")`
   (horizons, facies, geomodels, faults, closures, `RPMExample`) on two full
   models: `config/example.json` with `cube_shape [16, 16, 500]` (+10 pad
   samples, so nk = 510), `include_salt: false`, `closure_types: ["simple"]`,
   and `min_closure_voxels_simple: 20` so a 16 × 16 cube keeps its closures.

   | seed | lateral filter | bandpass (order 4) | closures |
   |---|---|---|---|
   | 25 | 5 | 4.373–30.960 Hz | oil 358 voxels, gas 91 voxels |
   | 3 | 3 | 3.893–27.717 Hz | none (brine) |

2. `create_rfc_volumes` (the Numba kernel `datagenerator/zoeppritz_kernel.py`)
   at the legacy angles 0, 7, 15, 24 and 45°: `incident_angles` plus the
   `model_qc_volumes` 0° and 45°.
3. `postprocess_rfc_cubes(rfc_raw, "noise_free")`, the noise-free
   `model_qc_volumes` branch of `build_seismic_volumes`:
   `apply_bandlimits`, then `apply_lateral_filter`, then `apply_cumsum`.
   `write_final_cubes_to_disk` is intercepted, so the stacks are captured
   before legacy output scaling (`_scale_seismic` and the rpm factors are not
   ported).

Noise is not included. `add_weighted_noise` draws from numpy's stream, and the
Rust port is only statistically equivalent to it (see `filters-port.md`).

Rust cannot regenerate the legacy geology because its toy model draws from
its own RNG. So none of the `--toy-geometry planar --legacy-*` flag
combinations can reproduce a legacy Python model; those flags reproduce
earlier *Rust* masters. The fixture therefore stores:

- the legacy elastic cubes (Vp, Vs, rho; float32; zlib);
- FNV-1a 64 hashes and std / max |x| of every legacy output cube;
- four sampled traces per angle: (0, 0), (8, 7), (15, 15) and a gas column
  for seed 25. The sampled traces are stored for reflectivity, stack and
  cumsum.

The two `.bin` files are about 1.1 MB in total.

## Test

`rust/synthoseis-core/tests/angle_stack_legacy_e2e.rs` feeds the legacy
elastic cubes through the Rust production chain:

- `synthoseis_gpu::fuse_props_tile_cpu` with `NO_WAVELET` and the form that
  `--legacy-zoeppritz` selects;
- then `apply_filters_to_volume` with the legacy model's own corners, order
  and lateral size;
- then `cumsum_traces_f32` and the 2–100 Hz bandpass (kernel only).

Legacy reflectivity has `nk − 1` samples per trace. The Rust fuse writes `nk`
samples with a trailing 0. The parity checks therefore run the Rust filters on
the legacy `nk − 1` grid. The last test measures what the production `nk`
convention does instead.

## Results

The full-cube numbers come from an offline comparison against the complete
legacy cubes (130,304 samples per angle and model). The CI test checks the
hashes, the stats and the sampled traces.

### Reflectivity (`--legacy-zoeppritz` vs the legacy Numba kernel)

| | seed 25 | seed 3 |
|---|---|---|
| 0° | bit-identical (whole-cube hash) | bit-identical |
| samples with \|r\| > 1e-6, 7–45° | 100 % bit-identical | all bit-identical except 2 samples at 1 ulp (\|r\| = 3.8e-6 at 15°, 1.4e-6 at 45°) |
| samples with \|r\| ≤ 1e-6 | residue Δ ≤ 2.8e-16 | residue Δ ≤ 2.9e-16 |
| textbook form vs legacy, max \|Δ\| (sensitivity) | 5.1e-3 / 2.30e-2 / 5.64e-2 / 0.164 at 7 / 15 / 24 / 45° | same |

The ≤ 1e-6 samples sit at interfaces with identical properties on both sides,
where the exact reflection is 0. Legacy's Numba `fastmath` real path leaves a
round-off residue of 1e-31 to 3e-16 there. The Rust complex path leaves a
different residue, or exactly 0. That residue is why the whole-cube hashes
differ at non-zero angles.

### Noise-free stacks and cumsum (legacy `nk − 1` grid)

| | bit-identical | max \|Δ\| | peak | rel. RMS |
|---|---|---|---|---|
| stack (bandpass + lateral) | 99.4–99.7 % | 3.7e-9 | 0.061–0.086 | ≤ 4.7e-9 |
| cumsum (cumsum + 2–100 Hz) | 96.3–98.6 % | 1.5e-8 | 0.18–0.26 | ≤ 1.1e-8 |

Both differences are at the f32 rounding level. They were isolated as follows:

- **Lateral filter:** bit-exact. scipy `uniform_filter` applied to the Rust
  bandpassed cube equals the Rust stack.
- **Bandpass:** `b` and `a` are bit-identical to scipy's. `lfilter_zi` is
  not:
  - scipy 1.17.1 (the locked version) solves `(I − A) zi = B` with
    `linalg.solve`;
  - scipy 1.18.1 and the Rust port use the closed form;
  - the two `zi` differ by ≤ 1.6e-9 relative.

  With the Rust `zi` substituted into scipy's filtfilt, the output equals the
  Rust bandpass bit for bit. `filters-port.md` reported bit-exact parity
  because it used scipy 1.18.1.
- **Reflectivity residues:** these feed the IIR filter and add rounding-level
  differences.

**Tolerance:** max |Δ| ≤ 8 ulp(peak) and relative RMS ≤ 1e-7. The measured
worst case is about 1 ulp(peak) over the full cube (cumsum 1.5e-8 against a peak of 0.176). A modelling error (wrong corner, order, padding,
lateral size or Zoeppritz form) would be at least 1e-3.

### Known discrepancy: the Rust trailing reflectivity sample moves the bandpass edge

This PR **does not fix** the gap described here.

With the bandpass on, the production pipeline filters the `nk`-sample Rust
trace, whose last sample is 0. Legacy filters its `nk − 1` samples. So
`filtfilt`'s odd extension is taken about 0 instead of about the last
reflectivity value, one sample deeper. That changes the deep part of every
trace. The table compares Rust production with legacy over the first
`nk − 1` samples, full cube, all angles:

| samples above the base | seed 25 max \|Δ\| | seed 3 max \|Δ\| |
|---|---|---|
| 0–10 | 1.2–1.8e-2 | 1.4–1.6e-2 |
| 10–25 | 3.1–4.9e-3 | 3.8–4.3e-3 |
| 25–50 | 2.0–3.2e-3 | 2.5–3.0e-3 |
| 50–100 | 0.74–1.2e-3 | 0.95–1.1e-3 |
| 100–200 | 1.1–1.7e-4 | 2.6–3.0e-4 |
| ≥ 200 | ≤ 9.0e-6 | ≤ 1.4e-5 |
| whole cube, rel. RMS | 10.6–16.0 % | 5.3–6.3 % |

For scale, the stack peaks are 0.061–0.086 and the stack RMS is about 4e-3.

- **Extent:** the reach follows the low-cut (3.9–4.4 Hz here). Lower corners
  or longer traces change the depth extent, not the mechanism.
- **Which paths:** every Rust path that bandpasses is affected: chunked,
  streaming, strip, multi-process, geometry-once and classic. Paths without
  the bandpass are unaffected: a Ricker "same" convolution treats the
  trailing 0 as a no-op for the first `nk − 1` samples.
- **Legacy is also edge-affected here:** the 10 legacy pad samples absorb
  part of this in legacy, and legacy writes them out uncropped. The two
  edges are simply different.
- **Test:** `known_gap_trailing_sample_moves_bandpass_edge` pins the size
  and depth extent of the gap, so a fix or a regression fails loudly. Update
  that test and this section together.
- **Possible fixes (not done here):** bandpass only the first `nk − 1`
  samples and keep the trailing sample at 0, or drop it from the
  deliverable. Either changes output.

## Regenerate

```bash
uv sync --frozen   # the locked environment (numpy 2.4.4, scipy 1.17.1, numba 0.65.1)
uv run --frozen python tests/fixtures/generate_angle_stack_e2e.py
cd rust && cargo test -p synthoseis-core --test angle_stack_legacy_e2e -- --nocapture
```
