# End-to-end angle-stack parity against legacy Python

Added in #35 as test-only hygiene. The trailing-sample follow-up (stacked on
#35) fixes the one modelling gap it found; see
[Trailing reflectivity sample](#trailing-reflectivity-sample-bandpass-edge-fixed).

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
- then the production default `apply_filters_to_volume` with the legacy
  model's own corners, order and lateral size, on the full `nk`-sample fuse
  output;
- then `cumsum_traces_f32` and the 2–100 Hz bandpass (kernel only).

Legacy reflectivity has `nk − 1` samples per trace. The Rust fuse writes `nk`
samples with a trailing 0, and the default bandpass filters only the first
`nk − 1` (the legacy trace) and leaves the trailing sample 0. The checks
compare the first `nk − 1` output samples with legacy and assert the trailing
sample is 0. The last test pins the fix against the old whole-trace bandpass
(`--bandpass-trailing-sample`).

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

### Noise-free stacks and cumsum (first `nk − 1` samples)

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

### Trailing reflectivity sample: bandpass edge (fixed)

Up to #35 the production pipeline bandpassed the whole `nk`-sample Rust
trace, whose last sample is 0, while legacy filters its `nk − 1` samples. So
`filtfilt`'s odd extension was taken about 0 instead of about the last
reflectivity value, one sample deeper, and the deep part of every trace
changed. The fix (a scoped parity bug fix) bandpasses only the first
`nk − 1` samples when the bandpass replaces the Ricker, and writes the
trailing sample as 0; legacy has no sample there (`rfc_raw` is `nk − 1`
long, and augmentation crops to `cube_shape[2] + pad − 1`). The old output
stays available, bit for bit, as `--bandpass-trailing-sample` /
`FilterConfig::bandpass_trailing_sample`. Details in
[`filters-port.md`](filters-port.md#trailing-sample-legacy-parity-bug-fix).

Max |Rust − legacy| of the stack over the first `nk − 1` samples, full cube
(all 256 traces), all five angles, by distance from the base:

| samples above the base | seed 25 before | seed 25 after | seed 3 before | seed 3 after |
|---|---|---|---|---|
| 0–10 | 1.2–1.8e-2 | ≤ 9.3e-10 | 1.4–1.6e-2 | ≤ 1.2e-10 |
| 10–25 | 3.1–4.9e-3 | ≤ 1.2e-10 | 3.8–4.3e-3 | ≤ 2.3e-10 |
| 25–50 | 2.0–3.2e-3 | ≤ 4.1e-10 | 2.5–3.0e-3 | ≤ 2.3e-10 |
| 50–100 | 0.74–1.2e-3 | ≤ 2.3e-10 | 0.95–1.1e-3 | ≤ 4.7e-10 |
| 100–200 | 1.1–1.7e-4 | ≤ 4.7e-10 | 2.6–3.0e-4 | ≤ 1.9e-9 |
| ≥ 200 | ≤ 9.0e-6 | ≤ 1.9e-9 | ≤ 1.4e-5 | ≤ 3.7e-9 |
| whole cube, rel. RMS | 10.6–16.0 % | ≤ 2.1e-9 | 5.3–6.3 % | ≤ 4.7e-9 |
| whole cube, bit-identical | 7.2–9.9 % | 99.7 % | 4.3–5.3 % | 99.4–99.5 % |

For scale, the stack peaks are 0.061–0.086 and 8 ulp(peak) is 3–6e-8. After
the fix the base of the trace is no different from the rest: the residual is
the `lfilter_zi` / reflectivity-residue rounding described above, and the
default output equals the whole-trace filter run on the legacy `nk − 1`
grid bit for bit.

- **Which paths:** every Rust path that bandpasses with the Ricker skipped
  (chunked, streaming, strip, multi-process, geometry-once, classic) uses the
  same trace kernel. Paths without the bandpass, and `--keep-ricker`, are
  unchanged: a Ricker-convolved trace has real signal in its last sample.
- **Legacy is also edge-affected here:** its 10 pad samples absorb part of
  the filtfilt edge, and legacy writes them out uncropped. Rust now has the
  *same* edge; changing it (mirrored padding, a taper) is a later realism
  item, not part of this fix.
- **Dead trailing sample:** for parity the last sample of every bandpassed
  trace is an exact 0, so the bottom depth slice of every stack is constant
  0 (noise-free even with noise on). It carries no geology and a network
  could learn to spot it. Crop or mask it for training; the edge-handling
  realism item (mirrored padding or a taper) should remove it.
- **Ricker path:** legacy has no Ricker → bandpass chain (its dormant
  wavelet path convolves the `nk − 1` trace and applies only the lateral
  filter), so `--keep-ricker` is unchanged. The Rust Ricker + lateral path
  matches the forced legacy wavelet path to ≤ 3e-8 down to the base; see
  [`filters-port.md`](filters-port.md#trailing-sample-legacy-parity-bug-fix).
- **Test:** `trailing_sample_bandpass_edge_matches_legacy_to_the_base`
  checks, per angle and model, that every depth band down to the last sample
  is within 8 ulp(peak) of legacy on the sampled columns, that the default
  equals the whole-trace filter on the `nk − 1` grid bit for bit over the
  full cube, and that `--bandpass-trailing-sample` still shows the old gap.
  Update that test and this section together.
- **Figure (not committed):** `bandpass_trailing_sample_before_after.png`
  from the PR, drawn from the full legacy cubes of `run_legacy` and the Rust
  stacks: max |Δ| vs samples above the base, before and after, both models,
  and the worst old trace against legacy.

## Regenerate

```bash
uv sync --frozen   # the locked environment (numpy 2.4.4, scipy 1.17.1, numba 0.65.1)
uv run --frozen python tests/fixtures/generate_angle_stack_e2e.py
cd rust && cargo test -p synthoseis-core --test angle_stack_legacy_e2e -- --nocapture
```
