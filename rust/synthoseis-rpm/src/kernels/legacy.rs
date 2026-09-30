//! f32 port of the legacy property builder
//! (`Seismic.build_property_models_randomised_depth`, `RPMExample` trends,
//! `EndMemberMixing`, `Seismic.fix_zero_values_at_base`).
//!
//! Legacy works on float32 zarr cubes, so every trend and mixing formula runs
//! in numpy float32 arithmetic (python float coefficients are cast to f32,
//! NEP 50). These kernels reproduce that arithmetic operation by operation:
//! `z**2` is `z * z`, `z**3` is libm `powf(z, 3)` (numpy's scalar float32
//! power loop), and no fused multiply-add is used.
//!
//! On AVX-512 hosts numpy dispatches float32 `z**3` to its SIMD (SVML) power,
//! which differs from libm `powf` by up to 1 ULP. Only the shale density
//! (the one cubic trend) is affected; see `docs/rock-physics-port.md`.

/// Pore fluid of a sand voxel (legacy `oil_closures` / `gas_closures`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Fluid {
    #[default]
    Brine,
    Oil,
    Gas,
}

impl Fluid {
    /// Legacy `Closures.assign_fluid_types` code: 0 brine, 1 oil, 2 gas.
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Fluid::Oil,
            2 => Fluid::Gas,
            _ => Fluid::Brine,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Fluid::Brine => "brine",
            Fluid::Oil => "oil",
            Fluid::Gas => "gas",
        }
    }
}

/// Sand/shale end-member mixing (legacy `mixing_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum MixingMethod {
    /// `inv_vel`: arithmetic density, harmonic (slowness) velocities. Legacy default.
    #[default]
    InverseVelocity,
    /// Backus average of the Lamé moduli (harmonic lambda and mu, arithmetic rho).
    BackusModuli,
}

impl MixingMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            MixingMethod::InverseVelocity => "inverse-velocity",
            MixingMethod::BackusModuli => "backus",
        }
    }
}

/// One voxel's elastic properties (float32, legacy units: g/cc, m/s).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Elastic32 {
    pub rho: f32,
    pub vp: f32,
    pub vs: f32,
}

/// Legacy `Seismic.water_properties`: rho 1.028, Vp 1500, Vs 1000.
/// Legacy salt properties (`Seismic.py`: `rho = 2.17`, `vp = 4500`,
/// `vs = 2250` for `lith == 2`, stored as float32).
pub const SALT: Elastic32 = Elastic32 {
    rho: 2.17,
    vp: 4500.0,
    vs: 2250.0,
};

pub const WATER: Elastic32 = Elastic32 {
    rho: 1.028,
    vp: 1500.0,
    vs: 1000.0,
};

/// Python float coefficient as numpy casts it next to a float32 array.
#[inline]
fn c(x: f64) -> f32 {
    x as f32
}

/// numpy float32 `z**3`: the scalar ufunc loop calls libm `powf`.
#[inline]
fn cube(z: f32) -> f32 {
    z.powf(3.0)
}

/// `RPMExample` trends in numpy float32 arithmetic (depth in metres).
pub mod example_f32 {
    use super::{c, cube};

    pub fn shale_rho(z: f32) -> f32 {
        c(7.7e-12) * cube(z) + c(-8.8e-08) * (z * z) + c(0.0004) * z + c(1.957)
    }
    pub fn shale_vp(z: f32) -> f32 {
        c(-0.00013) * (z * z) + c(1.13) * z + 1580.0
    }
    pub fn shale_vs(z: f32) -> f32 {
        c(-0.0001) * (z * z) + c(0.96) * z + 279.0
    }
    pub fn brine_sand_rho(z: f32) -> f32 {
        c(-7.8e-09) * (z * z) + c(0.00012) * z + c(2.021)
    }
    pub fn brine_sand_vp(z: f32) -> f32 {
        c(-1.34e-05) * (z * z) + c(0.49) * z + 2317.0
    }
    pub fn brine_sand_vs(z: f32) -> f32 {
        c(-1.0785e-05) * (z * z) + c(0.391) * z + 1007.0
    }
    pub fn oil_sand_rho(z: f32) -> f32 {
        c(-9.23e-09) * (z * z) + c(0.00014) * z + c(1.916)
    }
    pub fn oil_sand_vp(z: f32) -> f32 {
        c(-8.876e-06) * (z * z) + c(0.505) * z + 1998.0
    }
    pub fn oil_sand_vs(z: f32) -> f32 {
        c(-1.126e-05) * (z * z) + c(0.391) * z + 1036.0
    }
    pub fn gas_sand_rho(z: f32) -> f32 {
        c(-1.818e-08) * (z * z) + c(0.000247) * z + c(1.612)
    }
    pub fn gas_sand_vp(z: f32) -> f32 {
        c(-3.216e-06) * (z * z) + c(0.4796) * z + 1996.0
    }
    pub fn gas_sand_vs(z: f32) -> f32 {
        c(-1.0687e-05) * (z * z) + c(0.3662) * z + 1135.0
    }
}

/// Shale properties with separate depths for rho / Vp / Vs
/// (`RPMABC.calc_shale_properties`).
pub fn shale_f32(z_rho: f32, z_vp: f32, z_vs: f32) -> Elastic32 {
    Elastic32 {
        rho: example_f32::shale_rho(z_rho),
        vp: example_f32::shale_vp(z_vp),
        vs: example_f32::shale_vs(z_vs),
    }
}

/// Sand properties for `fluid` (`calc_{brine,oil,gas}_sand_properties`).
pub fn sand_f32(fluid: Fluid, z_rho: f32, z_vp: f32, z_vs: f32) -> Elastic32 {
    use example_f32 as e;
    match fluid {
        Fluid::Brine => Elastic32 {
            rho: e::brine_sand_rho(z_rho),
            vp: e::brine_sand_vp(z_vp),
            vs: e::brine_sand_vs(z_vs),
        },
        Fluid::Oil => Elastic32 {
            rho: e::oil_sand_rho(z_rho),
            vp: e::oil_sand_vp(z_vp),
            vs: e::oil_sand_vs(z_vs),
        },
        Fluid::Gas => Elastic32 {
            rho: e::gas_sand_rho(z_rho),
            vp: e::gas_sand_vp(z_vp),
            vs: e::gas_sand_vs(z_vs),
        },
    }
}

/// `EndMemberMixing._arithmetic_mean`.
#[inline]
pub fn arithmetic_mean(a0: f32, a1: f32, w: f32) -> f32 {
    (a0 * (1.0 - w)) + (a1 * w)
}

/// `EndMemberMixing._harmonic_mean`.
#[inline]
pub fn harmonic_mean(a0: f32, a1: f32, w: f32) -> f32 {
    1.0 / ((1.0 / a0 * (1.0 - w)) + (1.0 / a1 * w))
}

/// Mix `shale` and `sand` with net-to-gross `ng` (fraction of sand).
pub fn mix_f32(shale: Elastic32, sand: Elastic32, ng: f32, method: MixingMethod) -> Elastic32 {
    match method {
        MixingMethod::InverseVelocity => Elastic32 {
            rho: arithmetic_mean(shale.rho, sand.rho, ng),
            vp: harmonic_mean(shale.vp, sand.vp, ng),
            vs: harmonic_mean(shale.vs, sand.vs, ng),
        },
        MixingMethod::BackusModuli => {
            // bruges: lam = rho * (vp**2 - 2.*vs**2.), mu = rho * vs**2
            let lam = |p: Elastic32| p.rho * (p.vp * p.vp - 2.0 * (p.vs * p.vs));
            let mu = |p: Elastic32| p.rho * (p.vs * p.vs);
            let lam_mix = harmonic_mean(lam(shale), lam(sand), ng);
            let mu_mix = harmonic_mean(mu(shale), mu(sand), ng);
            let rho = arithmetic_mean(shale.rho, sand.rho, ng);
            Elastic32 {
                rho,
                vp: ((lam_mix + 2.0 * mu_mix) / rho).sqrt(),
                vs: (mu_mix / rho).sqrt(),
            }
        }
    }
}

/// Per-layer depth shifts in samples (legacy `delta_z_layer` plus the
/// `(rho, vp, vs)` property shifts for shale, brine, oil and gas sand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerShifts {
    pub layer: i64,
    /// `[shale, brine, oil, gas][rho, vp, vs]`.
    pub props: [[i64; 3]; 4],
}

impl LayerShifts {
    fn row(&self, fluid: Option<Fluid>) -> [i64; 3] {
        let r = match fluid {
            None => 0,
            Some(Fluid::Brine) => 1,
            Some(Fluid::Oil) => 2,
            Some(Fluid::Gas) => 3,
        };
        self.props[r]
    }
}

/// Classification of one voxel for [`legacy_column_properties`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoxelKind {
    /// Legacy `lith < 0`: water properties.
    Water,
    /// Never written by the layer loop: left at 0, then forward-filled.
    Unfilled,
    /// Sediment of `layer` (index into the shift table) with net-to-gross
    /// `ng` (sand fraction; `ng > 0` mixes in `fluid` sand).
    Layer { layer: usize, ng: f32, fluid: Fluid },
    /// Legacy `lith == 2` (salt body): constant [`SALT`] properties, set
    /// before the base forward-fill (legacy `Seismic.py`, after the layer
    /// loop).
    Salt,
}

/// Legacy `np.clip(k + dz, 0, nk - 10)` followed by numpy indexing (a
/// negative clip bound wraps from the end, as numpy does for `nk < 10`).
#[inline]
pub fn shifted_index(k: usize, dz: i64, nk: usize) -> usize {
    let hi = nk as i64 - 10;
    let v = (k as i64 + dz).max(0).min(hi);
    if v < 0 {
        (v + nk as i64).max(0) as usize
    } else {
        v as usize
    }
}

/// Legacy `fix_zero_values_at_base` on one trace: every zero sample takes the
/// nearest non-zero value above it; leading zeros stay.
pub fn forward_fill_zeros(v: &mut [f32]) {
    let mut last: Option<f32> = None;
    for x in v.iter_mut() {
        if *x != 0.0 {
            last = Some(*x);
        } else if let Some(l) = last {
            *x = l;
        }
    }
}

/// Properties of one voxel (`k` in a trace whose depth cube column is
/// `depth`), before the base forward-fill.
#[inline]
pub fn voxel_properties(
    depth: &[f32],
    k: usize,
    kind: VoxelKind,
    shifts: &[LayerShifts],
    mixing: MixingMethod,
) -> Elastic32 {
    match kind {
        VoxelKind::Water => WATER,
        VoxelKind::Salt => SALT,
        VoxelKind::Unfilled => Elastic32::default(),
        VoxelKind::Layer { layer, ng, fluid } => {
            let nk = depth.len();
            let s = shifts[layer];
            let z = |dz: i64| depth[shifted_index(k, s.layer + dz, nk)];
            let d = s.row(None);
            let shale = shale_f32(z(d[0]), z(d[1]), z(d[2]));
            if ng > 0.0 {
                let d = s.row(Some(fluid));
                let sand = sand_f32(fluid, z(d[0]), z(d[1]), z(d[2]));
                mix_f32(shale, sand, ng, mixing)
            } else {
                shale
            }
        }
    }
}

/// Legacy property builder for one trace: water, shale for every sediment
/// voxel, sand of the voxel's fluid mixed in with its net-to-gross, then the
/// base forward-fill. `depth` is the legacy `faulted_depth` column (metres
/// below the mudline, float32). Salt voxels ([`VoxelKind::Salt`]) take the
/// legacy constants before the forward-fill, as legacy does. The final
/// scaling factors (all 1.0 by default, and skipping salt) are not ported.
pub fn legacy_column_properties(
    depth: &[f32],
    kinds: &[VoxelKind],
    shifts: &[LayerShifts],
    mixing: MixingMethod,
    rho: &mut [f32],
    vp: &mut [f32],
    vs: &mut [f32],
) {
    let nk = depth.len();
    assert!(kinds.len() == nk && rho.len() == nk && vp.len() == nk && vs.len() == nk);
    for k in 0..nk {
        let p = voxel_properties(depth, k, kinds[k], shifts, mixing);
        rho[k] = p.rho;
        vp[k] = p.vp;
        vs[k] = p.vs;
    }
    forward_fill_zeros(rho);
    forward_fill_zeros(vp);
    forward_fill_zeros(vs);
}
