# Rock physics port: reflectivity root cause and plan

Status: **survey and plan only. The pipeline is unchanged** (no Rust code in
this PR yet). The root cause below is a real bug in the Rust Zoeppritz
*inputs*, so per the port rules the default behaviour is not changed until
the fix is approved.

## 1. Root cause: DC offset and reflectivity down to −16.1

On the 64×64×128, seed 7, 4-fault demo cube at 15°, the raw Rust
reflectivity (`angle_rfc.f32`) has:

| | value |
|---|---|
| range | −16.13 … 0.119 |
| mean | −0.0489 |
| median | +0.0006 |
| samples with \|r\| > 1 | 0.40 % |
| non-zero samples | 100 % |

Real PP coefficients lie in [−1, 1] and are zero inside homogeneous layers.

**Zoeppritz is not the bug.** `synthoseis-seismic::zoeppritz_pp` matches
`datagenerator/zoeppritz_kernel.py` term for term (complex arithmetic, real
part), and the `seismic_kernels.json` goldens pass. A numpy rebuild from the
labels, the `rpm_example` trends and that formula reproduces the Rust
reflectivity **bit for bit** (`tests/fixtures/rpm_reflectivity_rootcause.py`).
The problem is the elastic inputs built in `pipeline.rs::generate_tiny_cube`
and `pipeline_stream_0.rs::depth_trends` (+ `synthoseis-gpu::props_f32`).
There are three defects.

1. **Depth scale is 25× legacy.** `DEPTH_PER_SAMPLE = 100.0` m per sample.
   Legacy converts samples to metres with `digi` (4 m; `Faults.py`,
   `work_cube_depth *= digi`), and the example config spans 1250 × 4 m = 5 km.
   The `rpm_example` quadratics are only meaningful over about 0–5 km:
   - shale Vp peaks at 4.35 km and reaches 0 at 9.92 km;
   - shale Vs reaches 0 at 9.88 km.

   A 128-sample Rust cube reaches 12.7 km. The −16.1 sample is at k = 96
   (9.6 km), where a shale sits over a brine sand:

   | layer | Vp (m/s) | Vs (m/s) | ρ (g/cc) |
   |---|---|---|---|
   | shale (upper) | 447 | 279 | 4.50 |
   | brine sand (lower) | 5809 | 3785 | 2.45 |

   At 15°, `p·Vp2 = 3.36`, so the interface is post-critical, and the
   real part of the complex Zoeppritz ratio is −16.1. The legacy kernel gives
   the same value for these inputs.
2. **Depth is per sample from the cube top, not per layer below the mudline.**
   Legacy's depth cube is the layer's TVD below mudline
   (`tvdml_map = previous_depth_map − seabed`), constant inside a layer (per
   trace, fraction-weighted for partial voxels). Properties are therefore
   constant inside a layer, and reflectivity is zero except at interfaces.
   Rust evaluates the trends at `k·100 m`, so every sample inside a layer is
   an impedance step of 2–5 %. That produces a reflection of +0.01 … +0.03
   at *every* sample (the DC offset and the "red" spectrum). It turns
   negative below the shale-Vp vertex.
3. **The water column is oil sand.** Label 255 (above the seabed) falls
   through the `_ =>` arm to the oil-sand trend. Legacy `water_properties`
   uses ρ 1.028, Vp 1500, Vs 1000 (the legacy Vs is non-physical but
   harmless). This only affects the top 1–4 samples of the toy cube.

Effect of fixing each one:

| inputs | min | max | mean | \|r\| > 1 | non-zero |
|---|---|---|---|---|---|
| master (defects 1–3) | −16.1 | 0.119 | −0.0489 | 0.40 % | 100 % |
| 4 m/sample only (fix 1) | −0.091 | 0.104 | 0.00077 | 0 | 100 % |
| legacy depth model: per-layer TVDML × 4 m + water (fixes 1–3) | −0.535 | 0.519 | 0.0012 | 0 | 2.8 % |

The remaining ±0.53 is the seabed (water over shale, ρ 1.028 → 1.96),
which is also what legacy produces. Figure:
`/workspace/synthoseis-bench/out/reflectivity_rootcause_before_after.png`
(range and mean per sample, histogram, spectrum, before/after sections).

## 2. Legacy rock physics survey

| Legacy | What it does | Rust today |
|---|---|---|
| `rockphysics/rpm_example.py` `RPMExample` | quadratic/cubic depth trends (z in m below mudline) for shale, brine, oil and gas sand: Vp, Vs, ρ | `RpmExampleTrends::*` (golden-tested); gas unused |
| `rpm_tagilsk_trends.py` | alternative polynomial trend set | `tagilsk_*` kernels, partly (oil-sand polys missing) |
| `Faults.py` depth cube | depth = TVDML of the layer base × `digi`, constant per layer per trace; partial voxels fraction-weighted; faulted with the geology | per-sample `k·100 m` (defects 1–2) |
| `Seismic.build_property_models_randomised_depth` | per layer above `first_random_lyr`: random depth shifts `k_rho, k_vp, k_vs` (independent per property, ±half_range) → decorrelates Vp/Vs/ρ | none |
| `calculate_shales` | shale everywhere except water | label 0 |
| `calculate_sands` + `EndMemberMixing` | sand (brine/oil/gas by closure fluid) mixed with shale by net-to-gross: `inverse_velocity_mixing` (default: arithmetic ρ, harmonic velocity) or `backus_moduli_mixing` | label 1 = pure brine sand; no N/G, no oil/gas closures |
| fluids | "fluid substitution" is **trend selection** (brine/oil/gas trend sets per closure); there is no Gassmann step in legacy | none |
| `water_properties` | lith < 0 → 1.028 / 1500 / 1000 | oil sand (defect 3) |
| salt | ρ 2.17 and fixed velocities | none |
| `fix_zero_values_at_base` | forward-fill zero properties at the trace base | n/a |
| `clip_vs_via_poissons_ratio` / `clip_vp_via_poissons_ratio` | Poisson-ratio guards | none |

## 3. Implementation plan (this PR, held, not merged)

A `RockPhysicsConfig` (off by default; `E2eConfig` field, CLI `--rock-physics legacy`):

1. **Legacy depth model.** A per-column pass over the labels:
   - seabed = first non-water sample;
   - each label run gets `(run base − seabed)·digi`, constant over the run;
   - water gets the legacy water properties.

   It is a pure function of one label column, so every tile, halo, strip
   worker and process computes identical bits with no extra memory beyond
   one column. The fuser takes a per-column property provider instead of
   the 1-D `[Vec<f64>; 9]` trends. `props_f32` and the WGSL buffer keep the
   current trends when the config is off, so **off is bit-identical to
   master**.
2. **End-member mixing.** Net-to-gross mixing (inverse-velocity default,
   Backus optional), ported from `RockPropertyModels.EndMemberMixing` with
   golden fixtures from the real Python class.
3. **Randomised depth shifts.** Per-layer shifts from a counter-based RNG
   keyed by (seed, layer id, property), the same Philox approach as the
   noise. Deterministic and tiling-invariant; legacy parity is statistical
   (distribution of shifts and of Vp/Vs/ρ per layer).
4. **Fluids.** Oil/gas trend selection wherever closures mark a fluid (the
   closure fluid labels are already ported in `synthoseis-closures`).
   There is no Gassmann step, as in legacy.

Parity plan:
- the depth model is bit-exact against legacy on the same labels;
- mixing gets kernel goldens;
- labels are unchanged (IoU = 1);
- angle stacks use the existing parity harness against a legacy-model numpy
  rebuild, plus statistical checks where legacy is random.

Invariance tests reuse the chunk / worker / process / geometry-once matrix
of `noise_pipeline.rs`. MSRV 1.83.

## 4. Decisions needed

- **Default fix.** Should the corrected depth model become the default?
  It changes every angle-stack output. The alternative is to keep master
  behaviour behind a `legacy`/`toy` switch, as with the noise seabed fix.
- **Depth scale.** Should `DEPTH_PER_SAMPLE` become `digi` (4 m), or should
  the toy cubes be treated as sub-sampled (for example `digi × infill`)? It
  sets how deep a given `nk` reaches.
