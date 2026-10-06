//! Rock physics: elastic properties (Vp, Vs, rho) per voxel from the labels.
//!
//! Port of the legacy property chain (`Faults.build_faulted_property_geomodels`
//! depth model, `Seismic.build_property_models_randomised_depth`,
//! `EndMemberMixing`, `Closures.assign_fluid_types`). See
//! `docs/rock-physics-port.md`.
//!
//! # Default model ([`ElasticModel::Rpm`])
//! * **Depth.** One depth per layer and column, measured below the seabed at
//!   [`RockPhysicsConfig::depth_step_m`] (legacy `digi` = 4 m) per sample:
//!   `TVDML = (z_{L+1} - z_0) * digi` in float32, with `z` the horizon maps in
//!   samples (legacy per-layer TVDML: the depth of the layer base below the
//!   mudline). Bit-identical to legacy without faults.
//! * **Faults.** Labels are faulted, horizon maps are not. Each run of a label
//!   in a faulted column is shifted by the integer offset between its observed
//!   base and the unfaulted base (the "label run" approximation; legacy
//!   re-picks the faulted horizon maps instead).
//! * **Water** above the seabed: rho 1.028, Vp 1500, Vs 1000. Label 255 below
//!   the seabed (gaps between layers, the base pad) forward-fills the sample
//!   above (legacy `fix_zero_values_at_base`).
//! * **Lithology.** Sand or shale per layer from the legacy sand-fraction
//!   Markov chain ([`crate::lithology`]; even shale / odd sand with
//!   `--toy-lithology alternating`, the planar geometry or the legacy toy);
//!   sand voxels mix brine / oil / gas sand with shale by the layer's
//!   net-to-gross map (inverse velocity by default, Backus optional).
//! * **Random depth shifts** per layer (and per property), keyed by seed and
//!   layer, for layers deeper than [`RockPhysicsConfig::first_random_layer`].
//! * **Fluids** from spill-point closures on the post-fault top of each sand
//!   unit (consecutive sand layers, legacy `Closures.top_lith_indices`; the
//!   deepest unit is skipped). The hydrocarbon column runs down to the unit
//!   base, across the unit's internal horizons ([`sand_unit_fluids`],
//!   docs/closures-per-sand-unit.md). The legacy switch
//!   [`RockPhysicsConfig::closures_per_layer`] (`--closures-per-layer`)
//!   uses each sand layer's own top instead (master 8b5988f).
//!   By default closures are segmented in 3D across faults: each 3D
//!   connected compartment gets its own fluid ([`crate::closure_segments`],
//!   docs/closure-segmentation-faults.md).
//!   [`RockPhysicsConfig::closures_unsegmented`] (`--closures-unsegmented`)
//!   restores master ef2dc42.
//! * **Zoeppritz.** Textbook PP expression ([`ZoeppritzForm::Exact`]);
//!   [`RockPhysicsConfig::legacy_zoeppritz`] (`--legacy-zoeppritz`) restores
//!   the legacy `det` typo bit for bit (master 33a3a93 default output).
//!
//! Every quantity is either global and deterministic in the config (maps,
//! draws) or computed per column from that column's labels, so the output does
//! not depend on chunk shape, worker count or process count.
//!
//! # Legacy toy model ([`ElasticModel::LegacyToy`])
//! [`RockPhysicsConfig::legacy_toy_depth`] reproduces master 10f4dcd bit for
//! bit: depth `k * 100 m` from the cube top, label 0 shale, 1 brine sand, any
//! other label (including 255) oil sand, legacy Zoeppritz (the master
//! guarantee needs the `det` typo, so `legacy_toy_depth` implies
//! `legacy_zoeppritz`).

use std::collections::VecDeque;

use synthoseis_closures::flood_fill_heap_2d;
use synthoseis_rpm::{legacy_column_properties, LayerShifts, VoxelKind};
use synthoseis_seismic::splitmix64;

pub use synthoseis_rpm::{Elastic32, Fluid, MixingMethod};
pub use synthoseis_seismic::ZoeppritzForm;

use crate::pipeline::E2eConfig;
pub use crate::lithology::ToyLithology;

/// Net-to-gross of sand layers.
#[derive(Debug, Clone, PartialEq)]
pub enum NetToGross {
    /// Legacy `create_random_net_over_gross_map`: a lateral fBm map per sand
    /// layer, normalised to mean `U(avg)` and std `U(stdev)`, clipped to `avg`.
    Legacy {
        avg: [f64; 2],
        stdev: [f64; 2],
        octaves: usize,
    },
    /// The same net-to-gross for every sand voxel (`1.0` = pure sand).
    Constant(f32),
}

impl Default for NetToGross {
    fn default() -> Self {
        NetToGross::Legacy {
            avg: [0.45, 0.9],
            stdev: [0.01, 0.05],
            octaves: 9,
        }
    }
}

/// Minimum closure size (whole cells) below which a closure compartment
/// stays brine (legacy `remove_small_objects(min_closure_voxels_simple)`).
/// See docs/rock-physics-port.md ("Closure minimum").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClosureMinimum {
    /// `clamp(round(ni·nj / 180), 20, 500)` from the cube's map area
    /// (default). 180 cells² per voxel anchors the rule at legacy's
    /// 300 × 300 design cube (exactly 500 there).
    #[default]
    Scaled,
    /// A fixed minimum (`Fixed(500)` = master bad1daa8 and legacy).
    Fixed(usize),
}

impl ClosureMinimum {
    /// Master bad1daa8 / legacy `min_closure_voxels_simple`.
    pub const LEGACY: Self = ClosureMinimum::Fixed(500);
    /// Map cells per voxel of [`ClosureMinimum::Scaled`] (300·300 / 500).
    pub const SCALED_CELLS_PER_VOXEL: usize = 180;
    /// Floor of [`ClosureMinimum::Scaled`] (middle of the empty 10-50
    /// voxel band between specks and real traps).
    pub const SCALED_FLOOR: usize = 20;
    /// Cap of [`ClosureMinimum::Scaled`]: never stricter than the legacy
    /// 500. `None` would let the rule grow past 500 above 300 × 300.
    pub const SCALED_CAP: Option<usize> = Some(500);

    /// The threshold (whole cells) applied on an `ni × nj` cube. Integer
    /// arithmetic; depends only on the global map size, so every tile,
    /// worker and process sees the same value. Never below 1.
    pub fn voxels(self, ni: usize, nj: usize) -> usize {
        match self {
            ClosureMinimum::Scaled => {
                let per = Self::SCALED_CELLS_PER_VOXEL;
                let t = ((ni * nj + per / 2) / per).max(Self::SCALED_FLOOR);
                Self::SCALED_CAP.map_or(t, |cap| t.min(cap))
            }
            ClosureMinimum::Fixed(n) => n.max(1),
        }
    }

    /// Short label for summaries and the `closure_minimum` attribute.
    pub fn as_str(self) -> &'static str {
        match self {
            ClosureMinimum::Scaled => "scaled-area",
            ClosureMinimum::Fixed(_) => "fixed",
        }
    }
}

/// Rock physics settings. `Default` is the corrected legacy model.
#[derive(Debug, Clone, PartialEq)]
pub struct RockPhysicsConfig {
    /// Reproduce master 10f4dcd (toy depth `k * 100 m`, label >= 2 and 255
    /// as oil sand). CLI `--legacy-toy-depth`.
    pub legacy_toy_depth: bool,
    /// Metres per sample for depth below the seabed (legacy `digi`, 4).
    pub depth_step_m: f64,
    /// Sand/shale mixing.
    pub mixing: MixingMethod,
    /// Net-to-gross of sand layers.
    pub net_to_gross: NetToGross,
    /// Random depth shifts apply to legacy layer numbers `> first_random_layer`
    /// (legacy `first_random_lyr = 20`; legacy layer 0 is the water layer, so
    /// Rust layer `L` is legacy layer `L + 1`).
    pub first_random_layer: usize,
    /// Half range (samples) of the per-layer shift. `None` = legacy draw
    /// `int(triangular(35, 75, 125))` keyed by the seed.
    pub layer_shift_samples: Option<u32>,
    /// Half range (samples) of the per-property shift. `None` = legacy draw
    /// `int(triangular(5, 11, 20))`.
    pub property_shift_samples: Option<u32>,
    /// Oil / gas / brine selection from closures (`false` = all brine).
    pub fluids: bool,
    /// Maximum hydrocarbon column in metres (legacy `max_column_height`).
    pub max_column_m: f64,
    /// Minimum closure (compartment) size in whole cells: smaller closures
    /// stay brine. Default [`ClosureMinimum::Scaled`]
    /// (`clamp(round(ni·nj / 180), 20, 500)`), legacy
    /// `min_closure_voxels_simple` scaled from its 300 × 300 design cube.
    /// [`ClosureMinimum::LEGACY`] (`Fixed(500)`, CLI
    /// `--legacy-closure-minimum`) reproduces master bad1daa8; CLI
    /// `--min-closure-voxels N` is `Fixed(N)`. Use
    /// [`ClosureMinimum::voxels`] for the threshold actually applied.
    pub closure_minimum: ClosureMinimum,
    /// Legacy switch: store the closure contact capped at the integer unit
    /// base cell (`min(fill, cap, base)`, master bad1daa8) as the fluid
    /// contact of segmented closures. Default: capped at `base + ½`, so
    /// partial voxels fill a full-unit trap down to the true sub-cell unit
    /// base. No effect with whole voxels (bit-identical) or on the
    /// unsegmented / per-layer paths (no base clamp). CLI
    /// `--legacy-closure-contact-cap`. See [`crate::closure_segments`].
    pub legacy_closure_contact_cap: bool,
    /// Evaluate Zoeppritz with the legacy `det` typo instead of the textbook
    /// expression. CLI `--legacy-zoeppritz`. Implied by `legacy_toy_depth`.
    pub legacy_zoeppritz: bool,
    /// Per-layer lithology rule of the layered toy geometry (default: legacy
    /// sand-fraction Markov chain). CLI `--toy-lithology`. The planar
    /// geometry and `legacy_toy_depth` always alternate; see
    /// [`crate::lithology`].
    pub lithology: ToyLithology,
    /// Model sand fraction for [`ToyLithology::Markov`]. `None` = legacy
    /// per-model draw U(0.05, 0.25). CLI `--sand-layer-fraction`.
    pub sand_layer_fraction: Option<f64>,
    /// Mean sand unit thickness in layers (legacy `sand_layer_thickness`,
    /// 2). CLI `--sand-layer-thickness`.
    pub sand_layer_thickness: f64,
    /// Legacy switch: closures on every sand layer's own top (master
    /// 8b5988f), instead of on the top of each sand unit with the deepest
    /// unit skipped, as legacy `Closures` does. CLI `--closures-per-layer`.
    pub closures_per_layer: bool,
    /// Legacy switch: closures per sand unit without 3D segmentation
    /// (master ef2dc42): one contact per 2D region, and fluids per 2D
    /// region rather than per 3D compartment. CLI `--closures-unsegmented`.
    /// See [`crate::closure_segments`].
    pub closures_unsegmented: bool,
    /// Salt body (legacy `include_salt`, `true` in the shipped
    /// `config/example.json`): one convex-hull salt body per model, horizon
    /// drag against its flanks, salt properties and closure walls. Only the
    /// layered geometry has salt ([`E2eConfig::effective_salt`]). CLI
    /// `--no-salt` (`false`) reproduces master b4f4259. See [`crate::salt`].
    pub salt: bool,
    /// Legacy switch: absolute `U(150, 300)` sample top offset of the salt
    /// below horizon 1, instead of the default scaled by
    /// `min(nk / 1250, 1)`. CLI
    /// `--salt-legacy-top-offset`.
    pub salt_legacy_top_offset: bool,
    /// Legacy switch: keep fault labels inside the salt body (master
    /// 2b3850ba). By default `data/fault_labels` is `fault AND NOT salt`
    /// ([`E2eConfig::effective_fault_salt_mask`]): faults die out against
    /// salt and have no seismic expression inside it. CLI
    /// `--fault-labels-through-salt`. See `docs/salt-bodies.md`.
    pub fault_labels_through_salt: bool,
    /// Partial voxels (spec-partial-voxels; **on by default** since PR B2):
    /// exact vertical volume fractions with Backus mixing (cell mode) or
    /// sub-cell reflectivity (time mode). Opt out with
    /// `PartialVoxelConfig::whole_voxels()` (`legacy_whole_voxels = true`,
    /// = the d8b96e69 default). CLI `--partial-voxel-reflectivity` /
    /// `--legacy-whole-voxels`. See [`crate::partial_voxels`].
    pub partial_voxels: crate::partial_voxels::PartialVoxelConfig,
}

impl Default for RockPhysicsConfig {
    fn default() -> Self {
        Self {
            legacy_toy_depth: false,
            depth_step_m: 4.0,
            mixing: MixingMethod::InverseVelocity,
            net_to_gross: NetToGross::default(),
            first_random_layer: 20,
            layer_shift_samples: None,
            property_shift_samples: None,
            fluids: true,
            max_column_m: 150.0,
            closure_minimum: ClosureMinimum::default(),
            legacy_closure_contact_cap: false,
            legacy_zoeppritz: false,
            lithology: ToyLithology::default(),
            sand_layer_fraction: None,
            sand_layer_thickness: crate::lithology::SAND_LAYER_THICKNESS,
            closures_per_layer: false,
            closures_unsegmented: false,
            salt: true,
            salt_legacy_top_offset: false,
            fault_labels_through_salt: false,
            partial_voxels: crate::partial_voxels::PartialVoxelConfig::default(),
        }
    }
}

impl RockPhysicsConfig {
    /// Master 10f4dcd behaviour (`--legacy-toy-depth`): toy depth and
    /// legacy Zoeppritz.
    pub fn legacy_toy() -> Self {
        Self {
            legacy_toy_depth: true,
            legacy_zoeppritz: true,
            lithology: ToyLithology::Alternating,
            ..Self::default()
        }
    }

    /// Zoeppritz expression of this configuration: legacy when
    /// `legacy_zoeppritz` or `legacy_toy_depth` is set, else textbook.
    pub fn zoeppritz_form(&self) -> ZoeppritzForm {
        if self.legacy_zoeppritz || self.legacy_toy_depth {
            ZoeppritzForm::Legacy
        } else {
            ZoeppritzForm::Exact
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.depth_step_m.is_finite() && self.depth_step_m > 0.0) {
            return Err(format!("depth_step_m must be > 0, got {}", self.depth_step_m));
        }
        if !(self.max_column_m.is_finite() && self.max_column_m >= 0.0) {
            return Err(format!("max_column_m must be >= 0, got {}", self.max_column_m));
        }
        // Worst case of the legacy draw is its upper bound.
        crate::lithology::validate(
            self.sand_layer_fraction.unwrap_or(crate::lithology::SAND_LAYER_FRACTION[1]),
            self.sand_layer_thickness,
        )?;
        match &self.net_to_gross {
            NetToGross::Constant(v) if !(0.0..=1.0).contains(v) => {
                Err(format!("net-to-gross must be in [0, 1], got {v}"))
            }
            NetToGross::Legacy { avg, stdev, .. }
                if !(avg[0] <= avg[1] && stdev[0] <= stdev[1] && avg[0] >= 0.0 && avg[1] <= 1.0) =>
            {
                Err(format!("invalid net-to-gross ranges avg={avg:?} stdev={stdev:?}"))
            }
            _ => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Keyed deterministic draws (independent of tiling / worker / process).

const ROCK_SALT: u64 = 0x80C4_F151_C500_0001;
const STREAM_HALF_RANGE: u64 = 1;
const STREAM_SHIFT: u64 = 2;
const STREAM_NG: u64 = 3;
pub(crate) const STREAM_FLUID: u64 = 4;

#[inline]
fn key_step(h: u64, p: u64) -> u64 {
    splitmix64(h ^ p.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn key(parts: &[u64]) -> u64 {
    parts.iter().fold(ROCK_SALT, |h, &p| key_step(h, p))
}

#[inline]
fn unit_of(h: u64) -> f64 {
    (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform `[0, 1)` draw keyed by `parts`.
pub fn keyed_unit(parts: &[u64]) -> f64 {
    unit_of(key(parts))
}

fn keyed_uniform(lo: f64, hi: f64, parts: &[u64]) -> f64 {
    lo + (hi - lo) * keyed_unit(parts)
}

/// numpy `triangular(left, mode, right)` by inverse CDF of a unit draw.
pub fn triangular_inv(left: f64, mode: f64, right: f64, u: f64) -> f64 {
    let fc = (mode - left) / (right - left);
    if u <= fc {
        left + (u * (right - left) * (mode - left)).sqrt()
    } else {
        right - ((1.0 - u) * (right - left) * (right - mode)).sqrt()
    }
}

/// Legacy `int(rng.uniform(-h, h))` (truncation toward zero).
pub fn legacy_shift(half_range: u32, parts: &[u64]) -> i64 {
    let h = half_range as f64;
    keyed_uniform(-h, h, parts) as i64
}

/// Per-model shift half ranges `(layer, property)` in samples.
pub fn shift_half_ranges(seed: u64, rp: &RockPhysicsConfig) -> (u32, u32) {
    let l = rp.layer_shift_samples.unwrap_or_else(|| {
        triangular_inv(35.0, 75.0, 125.0, keyed_unit(&[seed, STREAM_HALF_RANGE, 0])) as u32
    });
    let p = rp.property_shift_samples.unwrap_or_else(|| {
        triangular_inv(5.0, 11.0, 20.0, keyed_unit(&[seed, STREAM_HALF_RANGE, 1])) as u32
    });
    (l, p)
}

/// Shifts of Rust layer `layer` (legacy layer `layer + 1`).
pub fn layer_shifts(seed: u64, rp: &RockPhysicsConfig, layer: usize) -> LayerShifts {
    if layer < rp.first_random_layer {
        return LayerShifts::default();
    }
    let (lh, ph) = shift_half_ranges(seed, rp);
    let l = layer as u64;
    let mut s = LayerShifts {
        layer: legacy_shift(lh, &[seed, STREAM_SHIFT, l, 0, 0]),
        props: [[0; 3]; 4],
    };
    for (r, row) in s.props.iter_mut().enumerate() {
        for (p, v) in row.iter_mut().enumerate() {
            *v = legacy_shift(ph, &[seed, STREAM_SHIFT, l, 1 + r as u64, p as u64]);
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Net-to-gross maps.

/// Lateral fBm value noise on `(ni, nj)`: `octaves` passes, lacunarity 1.9,
/// persistence 0.5, coordinates `i / ni`, `j / nj` (legacy `_perlin` layout;
/// value noise replaces opensimplex).
pub fn fbm_map(ni: usize, nj: usize, octaves: usize, parts: &[u64]) -> Vec<f64> {
    let mut out = vec![0.0f64; ni * nj];
    let (mut amp, mut freq) = (1.0f64, 1.0f64);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    for o in 0..octaves.max(1) {
        // Lattice value at (x, y): keyed by (parts.., octave, x, y), in [-1, 1).
        let prefix = key_step(key(parts), o as u64);
        let lattice = |x: f64, y: f64| {
            2.0 * unit_of(key_step(key_step(prefix, x as u64), y as u64)) - 1.0
        };
        for i in 0..ni {
            let x = i as f64 / ni as f64 * freq;
            let (x0, tx) = (x.floor(), smooth(x - x.floor()));
            for j in 0..nj {
                let y = j as f64 / nj as f64 * freq;
                let (y0, ty) = (y.floor(), smooth(y - y.floor()));
                let v = |dx: f64, dy: f64| lattice(x0 + dx, y0 + dy);
                let a = v(0.0, 0.0) + tx * (v(1.0, 0.0) - v(0.0, 0.0));
                let b = v(0.0, 1.0) + tx * (v(1.0, 1.0) - v(0.0, 1.0));
                out[i * nj + j] += amp * (a + ty * (b - a));
            }
        }
        amp *= 0.5;
        freq *= 1.9;
    }
    out
}

/// Net-to-gross map of Rust layer `layer` (legacy
/// `create_random_net_over_gross_map`), float32 like the legacy cube.
pub fn net_to_gross_map(seed: u64, ng: &NetToGross, ni: usize, nj: usize, layer: usize) -> Vec<f32> {
    match *ng {
        NetToGross::Constant(v) => vec![v; ni * nj],
        NetToGross::Legacy {
            avg,
            stdev,
            octaves,
        } => {
            let l = layer as u64;
            let mut m = fbm_map(ni, nj, octaves, &[seed, STREAM_NG, l, 0]);
            let mean_t = keyed_uniform(avg[0], avg[1], &[seed, STREAM_NG, l, 1]);
            let std_t = keyed_uniform(stdev[0], stdev[1], &[seed, STREAM_NG, l, 2]);
            let n = m.len().max(1) as f64;
            let mean = m.iter().sum::<f64>() / n;
            let var = m.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            let sd = var.sqrt();
            for v in &mut m {
                let z = if sd > 0.0 { (*v - mean) * (std_t / sd) } else { 0.0 };
                *v = (z + mean_t).clamp(avg[0], avg[1]);
            }
            m.into_iter().map(|v| v as f32).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Depth model.

/// Maps label ids to horizon intervals: label `n` is the `n`-th non-empty
/// interval of the unfaulted fill (what `relabel_consecutive` produces).
pub fn label_intervals(maps: &[f64], ni: usize, nj: usize, nh: usize, nk: usize) -> Vec<usize> {
    let mut present = vec![false; nh.saturating_sub(1)];
    for col in 0..ni * nj {
        for (h, p) in present.iter_mut().enumerate() {
            let z0 = (maps[col * nh + h].ceil() as isize).clamp(0, nk as isize);
            let z1 = (maps[col * nh + h + 1].floor() as isize).clamp(0, nk as isize);
            *p |= z1 > z0;
        }
    }
    present
        .iter()
        .enumerate()
        .filter_map(|(h, &p)| p.then_some(h))
        .collect()
}

/// Legacy per-layer TVDML depth for every sample of one column (metres, f32).
///
/// * `col` — the column's (possibly faulted) labels;
/// * `maps_col` — the column's unfaulted horizon depths in samples (`nh`);
/// * `intervals` — label id → interval (see [`label_intervals`]).
///
/// Samples above the seabed (the first non-255 sample) get 0 (legacy water);
/// 255 below the seabed and unknown labels take the depth of the sample above.
/// Label `L` (interval `h`) gets `(f32(z_{h+1}) - f32(z_0)) * step`, the
/// float32 legacy formula, plus `(dk - dk0) * step` where `dk` is the offset
/// between the observed end of this run of `L` and its unfaulted end
/// `floor(z_{h+1})` and `dk0` the same for the seabed (both 0 without faults).
///
/// Returns the seabed sample (`nk` when the column is all water).
pub fn layer_depth_trace(
    col: &[u8],
    maps_col: &[f64],
    intervals: &[usize],
    step: f32,
    depth: &mut [f32],
) -> usize {
    let nk = col.len();
    assert_eq!(depth.len(), nk);
    let seabed = col.iter().position(|&v| v != 255).unwrap_or(nk);
    depth[..seabed].fill(0.0);
    if seabed == nk {
        return seabed;
    }
    let z0 = maps_col[0];
    let dk0 = seabed as i64 - (z0.ceil() as i64).clamp(0, nk as i64);
    let mut k = seabed;
    while k < nk {
        let lab = col[k];
        let mut e = k + 1;
        while e < nk && col[e] == lab {
            e += 1;
        }
        let h = intervals.get(lab as usize).copied();
        match h {
            Some(h) if lab != 255 && h + 1 < maps_col.len() => {
                let zb = maps_col[h + 1];
                let expect = (zb.floor() as i64).clamp(0, nk as i64);
                let dk = e as i64 - expect;
                let mut tv = zb as f32 - z0 as f32;
                if dk != dk0 {
                    tv += (dk - dk0) as f32;
                }
                depth[k..e].fill(tv * step);
            }
            _ => {
                let prev = if k > 0 { depth[k - 1] } else { 0.0 };
                depth[k..e].fill(prev);
            }
        }
        k = e;
    }
    seabed
}

// ---------------------------------------------------------------------------
// Closures and fluids.

/// Hydrocarbon closure of one sand layer: per column, samples `k < contact`
/// of the layer hold `fluid` (`contact = -inf` outside every closure).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerFluids {
    pub contact: Vec<f32>,
    pub fluid: Vec<Fluid>,
    /// `(fluid, crest sample, contact sample, columns, voxels)` per closure kept.
    pub closures: Vec<(Fluid, f64, f64, usize, usize)>,
}

/// Spill-point closures of the top of label `lab` (priority-flood fill of the
/// post-fault top-of-layer surface), capped at `max_column` samples below the
/// crest. Each closure draws brine / oil / gas uniformly (legacy
/// `rng.integers(3)`), keyed by `(seed, layer, closure rank)`; ranks follow
/// raster order of each closure's first column.
#[allow(clippy::too_many_arguments)]
pub fn layer_fluids(
    labels: &[u8],
    shape: [usize; 3],
    lab: u8,
    layer: usize,
    seed: u64,
    max_column: f64,
    min_voxels: usize,
) -> LayerFluids {
    unit_fluids(labels, shape, &[lab], layer, seed, max_column, min_voxels)
}

/// Fluid of closure `rank` on the top of `layer` (a sand layer, or the top
/// interval of a sand unit): brine / oil / gas with probability 1/3 each
/// (legacy `rng.integers(3)`), `floor(3 u)` of the keyed unit draw
/// `u = keyed_unit(seed, STREAM_FLUID, layer, rank)` (53-bit, so the
/// rounding bias is below 1e-15).
pub fn closure_fluid(seed: u64, layer: usize, rank: u64) -> Fluid {
    let code = (keyed_unit(&[seed, STREAM_FLUID, layer as u64, rank]) * 3.0) as u32;
    Fluid::from_code(code.min(2))
}

impl LayerFluids {
    /// No closures: all brine.
    pub fn empty(n: usize) -> Self {
        Self {
            contact: vec![f32::NEG_INFINITY; n],
            fluid: vec![Fluid::Brine; n],
            closures: Vec::new(),
        }
    }
}

/// Closures per sand unit ([`crate::lithology::closure_units`]) as
/// `(label, fluids)` for every label of every unit.
/// - The unit's contact and fluid maps are shared by all of its member
///   labels.
/// - The closure list is reported once, on the shallowest member label.
/// - Labels of the deepest unit (skipped by legacy) get no closures.
pub fn sand_unit_fluids(
    labels: &[u8],
    shape: [usize; 3],
    intervals: &[usize],
    sand: &[bool],
    seed: u64,
    max_column: f64,
    min_voxels: usize,
) -> Vec<(usize, LayerFluids)> {
    sand_unit_fluids_salt(labels, shape, intervals, sand, seed, max_column, min_voxels, None)
}

/// [`sand_unit_fluids`] with legacy salt walls.
#[allow(clippy::too_many_arguments)]
pub fn sand_unit_fluids_salt(
    labels: &[u8],
    shape: [usize; 3],
    intervals: &[usize],
    sand: &[bool],
    seed: u64,
    max_column: f64,
    min_voxels: usize,
    salt: Option<&crate::salt::SaltBody>,
) -> Vec<(usize, LayerFluids)> {
    let n = shape[0] * shape[1];
    let mut out = Vec::new();
    for (top, end) in crate::lithology::closure_units(sand) {
        let mut members: Vec<(usize, usize)> = intervals
            .iter()
            .enumerate()
            .filter(|&(lab, &h)| lab < 255 && h >= top && h < end)
            .map(|(lab, &h)| (h, lab))
            .collect();
        if members.is_empty() {
            continue;
        }
        members.sort_unstable();
        let ids: Vec<u8> = members.iter().map(|&(_, lab)| lab as u8).collect();
        let f = unit_fluids_salt(labels, shape, &ids, top, seed, max_column, min_voxels, salt);
        for (k, &(_, lab)) in members.iter().enumerate() {
            let mut g = f.clone();
            if k > 0 {
                g.closures.clear();
            }
            out.push((lab, g));
        }
        debug_assert!(out.iter().all(|(_, g)| g.contact.len() == n));
    }
    out
}

/// First sample and end of the first run of labels in `unit` in `col`.
pub(crate) fn first_unit_run(col: &[u8], unit: &[bool; 256]) -> Option<(usize, usize)> {
    let a = col.iter().position(|&v| unit[v as usize])?;
    let b = col[a..].iter().position(|&v| !unit[v as usize]).map_or(col.len(), |n| a + n);
    Some((a, b))
}

/// [`layer_fluids`] for a sand unit made of the labels `members` (legacy
/// closures per lithology unit):
/// - The top surface is the first sample of any member label in each column.
/// - The base is the end of that contiguous run of member labels, so the
///   hydrocarbon column can cross internal horizons down to the unit base.
/// - Draws are keyed by `layer`, the unit's top interval.
///
/// With one member this is exactly [`layer_fluids`].
#[allow(clippy::too_many_arguments)]
pub fn unit_fluids(
    labels: &[u8],
    shape: [usize; 3],
    members: &[u8],
    layer: usize,
    seed: u64,
    max_column: f64,
    min_voxels: usize,
) -> LayerFluids {
    unit_fluids_salt(labels, shape, members, layer, seed, max_column, min_voxels, None)
}

/// [`unit_fluids`] with legacy salt walls
/// ([`crate::salt::closure_fill_input`]); `None` is exactly [`unit_fluids`].
#[allow(clippy::too_many_arguments)]
pub fn unit_fluids_salt(
    labels: &[u8],
    shape: [usize; 3],
    members: &[u8],
    layer: usize,
    seed: u64,
    max_column: f64,
    min_voxels: usize,
    salt: Option<&crate::salt::SaltBody>,
) -> LayerFluids {
    let [ni, nj, nk] = shape;
    let n = ni * nj;
    let mut unit = [false; 256];
    for &m in members {
        if m != 255 {
            unit[m as usize] = true;
        }
    }
    let mut top = vec![f64::NAN; n];
    let mut base = vec![0usize; n];
    for c in 0..n {
        if let Some((a, b)) = first_unit_run(&labels[c * nk..(c + 1) * nk], &unit) {
            top[c] = a as f64;
            base[c] = b;
        }
    }
    let mut out = LayerFluids {
        contact: vec![f32::NEG_INFINITY; n],
        fluid: vec![Fluid::Brine; n],
        closures: Vec::new(),
    };
    if ni < 3 || nj < 3 {
        return out;
    }
    let (input, excluded) = crate::salt::closure_fill_input(&top, ni, nj, max_column, salt);
    let filled = flood_fill_heap_2d(&input, [ni, nj], 1e30);
    let closed: Vec<bool> = (0..n)
        .map(|c| {
            top[c].is_finite()
                && filled[c].is_finite()
                && filled[c] > top[c]
                && !excluded.get(c).copied().unwrap_or(false)
        })
        .collect();
    let mut comp = vec![usize::MAX; n];
    let mut rank = 0u64;
    for start in 0..n {
        if !closed[start] || comp[start] != usize::MAX {
            continue;
        }
        let mut cells = Vec::new();
        let mut q = VecDeque::from([start]);
        comp[start] = start;
        while let Some(c) = q.pop_front() {
            cells.push(c);
            let (i, j) = (c / nj, c % nj);
            let nb = [
                (i > 0).then(|| c - nj),
                (i + 1 < ni).then(|| c + nj),
                (j > 0).then(|| c - 1),
                (j + 1 < nj).then(|| c + 1),
            ];
            for d in nb.into_iter().flatten() {
                if closed[d] && comp[d] == usize::MAX {
                    comp[d] = start;
                    q.push_back(d);
                }
            }
        }
        let spill = cells.iter().map(|&c| filled[c]).fold(f64::NEG_INFINITY, f64::max);
        let crest = cells.iter().map(|&c| top[c]).fold(f64::INFINITY, f64::min);
        let contact = spill.min(crest + max_column);
        let voxels: usize = cells
            .iter()
            .map(|&c| (contact.ceil().max(0.0) as usize).min(base[c]).saturating_sub(top[c] as usize))
            .sum();
        let this = rank;
        rank += 1;
        if voxels < min_voxels.max(1) {
            continue;
        }
        let fluid = closure_fluid(seed, layer, this);
        for &c in &cells {
            out.contact[c] = contact as f32;
            out.fluid[c] = fluid;
        }
        out.closures.push((fluid, crest, contact, cells.len(), voxels));
    }
    out
}

// ---------------------------------------------------------------------------
// Elastic model.

/// Per-layer data of the default model.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerModel {
    /// Horizon interval of this label.
    pub interval: usize,
    pub sand: bool,
    pub shifts: LayerShifts,
    /// Net-to-gross map `(ni, nj)` (empty for shale).
    pub ng: Vec<f32>,
    /// Closures (`None` for shale or when fluids are off).
    pub fluids: Option<LayerFluids>,
}

/// Default (corrected legacy) elastic model; see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct RpmModel {
    pub shape: [usize; 3],
    /// Unfaulted horizon maps `(ni, nj, nh)` in samples.
    pub maps: Vec<f64>,
    pub nh: usize,
    /// Label id → interval.
    pub intervals: Vec<usize>,
    /// One entry per label id.
    pub layers: Vec<LayerModel>,
    /// `layers[..].shifts`, contiguous for the column kernel.
    pub shifts: Vec<LayerShifts>,
    pub step: f32,
    pub mixing: MixingMethod,
    /// Zoeppritz expression used by every fuse path.
    pub zoeppritz: ZoeppritzForm,
    /// Salt body (legacy lithology 2; salt properties override every other
    /// voxel kind). `None` without salt.
    pub salt: Option<crate::salt::SaltBody>,
    /// Output time axis when the run converts to two-way time
    /// ([`E2eConfig::time_axis`]); `None` on the legacy depth-as-time axis.
    /// Every fuse path reads it from here, so the model, its depth and its
    /// output axis always travel together.
    pub time: Option<crate::pipeline::TimeAxis>,
    /// Partial-voxel state (`None` = whole voxels, the default); see
    /// [`crate::partial_model`].
    pub partial: Option<Box<crate::partial_model::PartialModel>>,
}

/// Elastic properties used by every fuse path.
#[derive(Debug, Clone, PartialEq)]
pub enum ElasticModel {
    /// Master 10f4dcd toy trends (`--legacy-toy-depth`): 9 trends
    /// (shale / brine / oil x vp, vs, rho) indexed by sample.
    LegacyToy(Box<[Vec<f64>; 9]>),
    /// Corrected legacy rock physics (default).
    Rpm(Box<RpmModel>),
}

/// Build the elastic model for `cfg` from its (possibly faulted) labels.
/// Deterministic in `cfg` and the labels; every worker and process that
/// rebuilds it gets the same model.
pub fn elastic_model(cfg: &E2eConfig, labels: &[u8], shape: [usize; 3]) -> ElasticModel {
    if cfg.rock_physics.legacy_toy_depth {
        return ElasticModel::LegacyToy(Box::new(crate::pipeline_stream::depth_trends(shape[2])));
    }
    let (maps, nh) = crate::pipeline_stream::toy_horizon_maps(cfg);
    let rp = &cfg.rock_physics;
    let sand = crate::lithology::interval_sand(
        cfg.effective_lithology(),
        cfg.seed,
        nh,
        rp.sand_layer_fraction,
        rp.sand_layer_thickness,
    );
    let rp = RockPhysicsConfig {
        closures_per_layer: cfg.effective_closures_per_layer(),
        ..rp.clone()
    };
    let salt = crate::salt::salt_body(cfg);
    let mut model = RpmModel::build_with_salt(cfg.seed, &rp, &maps, nh, labels, shape, &sand, salt);
    model.time = cfg.time_axis();
    if let Some(r) = cfg.effective_partial_voxels() {
        model.partial = Some(Box::new(crate::partial_model::PartialModel::build(cfg, &model, &rp, &sand, r)));
    }
    ElasticModel::Rpm(Box::new(model))
}

/// Closure census of a run: the applied minimum and the whole-cell size of
/// every closure compartment of the active closure mode (3D compartments
/// by default; 2D regions with `closures_unsegmented`; per-layer regions
/// with `closures_per_layer` / the planar geometry), kept or not.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosureCensus {
    /// Threshold applied ([`ClosureMinimum::voxels`]).
    pub minimum: usize,
    /// Mode of [`RockPhysicsConfig::closure_minimum`].
    pub rule: ClosureMinimum,
    /// Whole-cell size of every compartment with at least one voxel.
    pub sizes: Vec<usize>,
}

impl ClosureCensus {
    /// Compartments at or above the minimum (they get a fluid draw).
    pub fn kept(&self) -> usize {
        self.sizes.iter().filter(|&&v| v >= self.minimum).count()
    }

    /// Share of all closure voxels (any size) in kept compartments; 1 with
    /// no closures.
    pub fn kept_volume_fraction(&self) -> f64 {
        let total: usize = self.sizes.iter().sum();
        let kept: usize = self.sizes.iter().filter(|&&v| v >= self.minimum).sum();
        if total == 0 {
            1.0
        } else {
            kept as f64 / total as f64
        }
    }
}

/// [`ClosureCensus`] of `cfg` on its (possibly faulted) labels, with the
/// same closure stage as [`elastic_model`]. `None` without fluids or with
/// `legacy_toy_depth` (no closures).
pub fn closure_census(cfg: &E2eConfig, labels: &[u8], shape: [usize; 3]) -> Option<ClosureCensus> {
    let rp = &cfg.rock_physics;
    if rp.legacy_toy_depth || !rp.fluids {
        return None;
    }
    let [ni, nj, nk] = shape;
    let (maps, nh) = crate::pipeline_stream::toy_horizon_maps(cfg);
    let sand = crate::lithology::interval_sand(
        cfg.effective_lithology(),
        cfg.seed,
        nh,
        rp.sand_layer_fraction,
        rp.sand_layer_thickness,
    );
    let salt = crate::salt::salt_body(cfg);
    let intervals = label_intervals(&maps, ni, nj, nh, nk);
    let max_column = rp.max_column_m / rp.depth_step_m;
    let sizes: Vec<usize> = if cfg.effective_closures_per_layer() {
        intervals
            .iter()
            .enumerate()
            .filter(|&(lab, &h)| lab < 255 && sand.get(h).copied().unwrap_or(false))
            .flat_map(|(lab, &h)| {
                unit_fluids_salt(labels, shape, &[lab as u8], h, cfg.seed, max_column, 1, salt.as_ref())
                    .closures
                    .into_iter()
                    .map(|c| c.4)
            })
            .collect()
    } else if rp.closures_unsegmented {
        sand_unit_fluids_salt(labels, shape, &intervals, &sand, cfg.seed, max_column, 1, salt.as_ref())
            .into_iter()
            .flat_map(|(_, f)| f.closures.into_iter().map(|c| c.4))
            .collect()
    } else {
        crate::closure_segments::segmented_sand_unit_fluids_with(
            labels,
            shape,
            &intervals,
            &sand,
            cfg.seed,
            max_column,
            1,
            salt.as_ref(),
            rp.legacy_closure_contact_cap,
        )
        .1
        .into_iter()
        .map(|c| c.voxels)
        .collect()
    };
    Some(ClosureCensus {
        minimum: rp.closure_minimum.voxels(ni, nj),
        rule: rp.closure_minimum,
        sizes: sizes.into_iter().filter(|&v| v > 0).collect(),
    })
}

/// Root MDIO attributes of the closure minimum: `closure_min_voxels` and
/// `closure_minimum: "scaled-area"`, in [`ClosureMinimum::Scaled`] mode
/// only (with fluids on and not `legacy_toy_depth`), so `Fixed` runs
/// (`--legacy-closure-minimum`, `--min-closure-voxels`) keep master bytes.
pub fn write_closure_attrs(store: &synthoseis_io::MdioStore, cfg: &E2eConfig) -> Result<(), String> {
    let rp = &cfg.rock_physics;
    if rp.legacy_toy_depth || !rp.fluids || rp.closure_minimum != ClosureMinimum::Scaled {
        return Ok(());
    }
    let t = rp.closure_minimum.voxels(cfg.inline_count, cfg.crossline_count);
    store
        .set_root_attrs(&[
            ("closure_min_voxels", serde_json::json!(t)),
            ("closure_minimum", serde_json::json!(ClosureMinimum::Scaled.as_str())),
        ])
        .map_err(|e| e.to_string())
}

impl RpmModel {
    /// Build from horizon maps `(ni, nj, nh)`, labels and the per-interval
    /// sand flags (see [`crate::lithology::interval_sand`]; intervals beyond
    /// `sand` are shale).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        seed: u64,
        rp: &RockPhysicsConfig,
        maps: &[f64],
        nh: usize,
        labels: &[u8],
        shape: [usize; 3],
        sand: &[bool],
    ) -> Self {
        Self::build_with_salt(seed, rp, maps, nh, labels, shape, sand, None)
    }

    /// [`RpmModel::build`] with an optional salt body: salt voxels take the
    /// legacy salt properties and closures are walled off around salt gaps.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_salt(
        seed: u64,
        rp: &RockPhysicsConfig,
        maps: &[f64],
        nh: usize,
        labels: &[u8],
        shape: [usize; 3],
        sand: &[bool],
        salt: Option<crate::salt::SaltBody>,
    ) -> Self {
        let [ni, nj, nk] = shape;
        assert_eq!(maps.len(), ni * nj * nh);
        assert_eq!(labels.len(), ni * nj * nk);
        let intervals = label_intervals(maps, ni, nj, nh, nk);
        let step = rp.depth_step_m as f32;
        let max_column = rp.max_column_m / rp.depth_step_m;
        // Global map size: every tile, worker and process rebuilds the full
        // label volume, so all see the same threshold.
        let min_voxels = rp.closure_minimum.voxels(ni, nj);
        let unit_fluid_maps = if rp.closures_per_layer || !rp.fluids {
            Vec::new()
        } else if rp.closures_unsegmented {
            sand_unit_fluids_salt(
                labels,
                shape,
                &intervals,
                sand,
                seed,
                max_column,
                min_voxels,
                salt.as_ref(),
            )
        } else {
            crate::closure_segments::segmented_sand_unit_fluids_with(
                labels,
                shape,
                &intervals,
                sand,
                seed,
                max_column,
                min_voxels,
                salt.as_ref(),
                rp.legacy_closure_contact_cap,
            )
            .0
        };
        let layers = intervals
            .iter()
            .enumerate()
            .map(|(lab, &h)| {
                let sand = sand.get(h).copied().unwrap_or(false);
                LayerModel {
                    interval: h,
                    sand,
                    shifts: layer_shifts(seed, rp, h),
                    ng: if sand {
                        net_to_gross_map(seed, &rp.net_to_gross, ni, nj, h)
                    } else {
                        Vec::new()
                    },
                    fluids: (sand && rp.fluids && lab < 255).then(|| {
                        if rp.closures_per_layer {
                            unit_fluids_salt(
                                labels,
                                shape,
                                &[lab as u8],
                                h,
                                seed,
                                max_column,
                                min_voxels,
                                salt.as_ref(),
                            )
                        } else {
                            unit_fluid_maps
                                .iter()
                                .find(|(l, _)| *l == lab)
                                .map(|(_, f)| f.clone())
                                .unwrap_or_else(|| LayerFluids::empty(ni * nj))
                        }
                    }),
                }
            })
            .collect::<Vec<LayerModel>>();
        let shifts = layers.iter().map(|l| l.shifts).collect();
        Self {
            shape,
            maps: maps.to_vec(),
            nh,
            intervals,
            layers,
            shifts,
            step,
            mixing: rp.mixing,
            zoeppritz: rp.zoeppritz_form(),
            salt,
            time: None,
            partial: None,
        }
    }

    /// Depth trace (metres below the seabed) of column `(i, j)`.
    pub fn depth_trace(&self, i: usize, j: usize, col: &[u8], depth: &mut [f32]) -> usize {
        let nj = self.shape[1];
        let c = i * nj + j;
        layer_depth_trace(col, &self.maps[c * self.nh..(c + 1) * self.nh], &self.intervals, self.step, depth)
    }

    /// Elastic properties of column `(i, j)` from its labels `col`.
    /// `scratch` holds the depth trace and voxel kinds.
    #[allow(clippy::too_many_arguments)]
    pub fn column(
        &self,
        i: usize,
        j: usize,
        col: &[u8],
        scratch: &mut ColumnScratch,
        rho: &mut [f32],
        vp: &mut [f32],
        vs: &mut [f32],
    ) {
        let nk = col.len();
        let nj = self.shape[1];
        let c = i * nj + j;
        scratch.depth.resize(nk, 0.0);
        scratch.kinds.clear();
        let seabed = self.depth_trace(i, j, col, &mut scratch.depth);
        let salt = self.salt.as_ref().filter(|s| s.runs[c].1 > s.runs[c].0);
        for (k, &lab) in col.iter().enumerate() {
            let kind = if salt.is_some_and(|s| s.contains(c, k)) {
                VoxelKind::Salt
            } else if k < seabed {
                VoxelKind::Water
            } else if let Some(l) = self.layers.get(lab as usize).filter(|_| lab != 255) {
                let (ng, fluid) = if l.sand {
                    let fluid = match &l.fluids {
                        Some(f) if (k as f32) < f.contact[c] => f.fluid[c],
                        _ => Fluid::Brine,
                    };
                    (l.ng[c], fluid)
                } else {
                    (0.0, Fluid::Brine)
                };
                VoxelKind::Layer {
                    layer: lab as usize,
                    ng,
                    fluid,
                }
            } else {
                VoxelKind::Unfilled
            };
            scratch.kinds.push(kind);
        }
        legacy_column_properties(
            &scratch.depth,
            &scratch.kinds,
            &self.shifts,
            self.mixing,
            rho,
            vp,
            vs,
        );
    }
}

/// Reusable per-trace scratch for [`RpmModel::column`].
#[derive(Debug, Default, Clone)]
pub struct ColumnScratch {
    pub depth: Vec<f32>,
    pub kinds: Vec<VoxelKind>,
    /// Partial-voxel scratch ([`RpmModel::column_partial`]).
    pub partial: crate::partial_model::PartialScratch,
}

impl ElasticModel {
    /// Salt body of the model (`None` without salt).
    pub fn salt(&self) -> Option<&crate::salt::SaltBody> {
        match self {
            ElasticModel::Rpm(m) => m.salt.as_ref(),
            ElasticModel::LegacyToy(_) => None,
        }
    }

    /// Output time axis (`None` on the legacy depth-as-time axis and for
    /// the master toy model).
    pub fn time(&self) -> Option<&crate::pipeline::TimeAxis> {
        match self {
            ElasticModel::Rpm(m) => m.time.as_ref(),
            ElasticModel::LegacyToy(_) => None,
        }
    }

    /// Output samples per trace for a depth model of `nk` samples: `nt` in
    /// time mode, else `nk`.
    pub fn output_nk(&self, nk: usize) -> usize {
        self.time().map_or(nk, |t| t.nt)
    }

    /// `true` for the master toy model.
    pub fn is_legacy_toy(&self) -> bool {
        matches!(self, ElasticModel::LegacyToy(_))
    }

    /// Zoeppritz expression: always legacy for the master toy model.
    pub fn zoeppritz_form(&self) -> ZoeppritzForm {
        match self {
            ElasticModel::LegacyToy(_) => ZoeppritzForm::Legacy,
            ElasticModel::Rpm(m) => m.zoeppritz,
        }
    }

    /// Fixed bytes held by the model (trends or maps).
    pub fn model_bytes(&self) -> usize {
        match self {
            ElasticModel::LegacyToy(t) => t.iter().map(|v| v.len() * 8).sum(),
            ElasticModel::Rpm(m) => {
                m.maps.len() * 8
                    + m.layers
                        .iter()
                        .map(|l| {
                            l.ng.len() * 4 + l.fluids.as_ref().map_or(0, |f| f.contact.len() * 5)
                        })
                        .sum::<usize>()
                    + m.shifts.len() * std::mem::size_of::<LayerShifts>()
                    + m.salt.as_ref().map_or(0, |s| s.runs.len() * 8)
                    + m.partial.as_ref().map_or(0, |p| p.delta.len() * 4 + p.salt_bounds.len() * 24)
            }
        }
    }

    /// Fill `(ti, tj, nk)` Vp / Vs / rho for the tile `[i0,i1) x [j0,j1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn tile_properties(
        &self,
        labels: &[u8],
        shape: [usize; 3],
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        vp: &mut [f32],
        vs: &mut [f32],
        rho: &mut [f32],
    ) {
        let [_, nj, nk] = shape;
        let tj = j1 - j0;
        let mut scratch = ColumnScratch::default();
        let partial = match self {
            ElasticModel::Rpm(m) if m.partial.is_some() => Some(m.partial_tile(i0, i1, j0, j1)),
            _ => None,
        };
        for i in i0..i1 {
            for j in j0..j1 {
                let g = (i * nj + j) * nk;
                let o = ((i - i0) * tj + (j - j0)) * nk;
                let col = &labels[g..g + nk];
                match self {
                    ElasticModel::LegacyToy(t) => {
                        for k in 0..nk {
                            let (p, s, r) = synthoseis_gpu::props_f32(col[k], k, t);
                            vp[o + k] = p;
                            vs[o + k] = s;
                            rho[o + k] = r;
                        }
                    }
                    ElasticModel::Rpm(m) => match &partial {
                        None => m.column(
                            i,
                            j,
                            col,
                            &mut scratch,
                            &mut rho[o..o + nk],
                            &mut vp[o..o + nk],
                            &mut vs[o..o + nk],
                        ),
                        Some(t) => m.column_partial(
                            i,
                            j,
                            col,
                            t,
                            &mut scratch,
                            &mut rho[o..o + nk],
                            &mut vp[o..o + nk],
                            &mut vs[o..o + nk],
                        ),
                    },
                }
            }
        }
    }
}
