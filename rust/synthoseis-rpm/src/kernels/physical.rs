//! Physical rock physics: Athy compaction, soft-sand dry frame, Batzle-Wang
//! fluids, Wood mixing and Gassmann fluid substitution.
//!
//! Spec: "Spec: compaction and Gassmann fluid substitution" (Notion,
//! Synthoseis hub; docs/rock-physics-physical.md). Every function here is
//! pointwise (one voxel, no neighbours), so it needs no halo and is chunk-,
//! worker- and process-invariant by construction.
//!
//! Units inside the kernel: depth m, pressure MPa (GPa for moduli),
//! temperature °C, density g/cc, velocity m/s.
//!
//! Precision: everything is evaluated in f64 and rounded once to f32 at the
//! end ([`Elastic32`]), so the f32 output differs from the f64 reference by
//! at most half an f32 ulp (≤ 6e-8 relative), inside the spec's 1e-5.
//!
//! References
//! - Athy (1930) AAPG Bull. 14(1) 1–24; Sclater & Christie (1980) JGR 85(B7)
//!   3711–3739 (φ0, c).
//! - Batzle & Wang (1992) Geophysics 57(11) 1396–1408 (fluids; brine S²
//!   coefficient −1820, the erratum to the printed −820).
//! - Dvorkin & Nur (1996) Geophysics 61(5) 1363–1370 (soft sand).
//! - Gassmann (1951); Mavko, Mukerji & Dvorkin (2009) The Rock Physics
//!   Handbook, 2nd ed. (Hertz-Mindlin, Hashin-Shtrikman, mineral moduli).

use super::legacy::{
    forward_fill_zeros, mix_f32, shifted_index, Elastic32, Fluid, LayerShifts, MixingMethod,
    VoxelKind, WATER,
};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

/// Porosity-depth law (spec §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum CompactionForm {
    /// φ = φ0 exp(−c z) (default).
    #[default]
    Depth,
    /// φ = φ0 exp(−P_eff / σ0).
    Stress,
}

impl CompactionForm {
    pub fn as_str(self) -> &'static str {
        match self {
            CompactionForm::Depth => "depth",
            CompactionForm::Stress => "stress",
        }
    }
}

/// Dry-frame model (spec §3.4, D5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum DryFrame {
    /// Dvorkin & Nur (1996) soft sand (default).
    #[default]
    SoftSand,
    /// Nur et al. (1998) critical porosity: K = K_m(1 − φ/φ_c), same for μ.
    CriticalPorosity,
}

impl DryFrame {
    pub fn as_str(self) -> &'static str {
        match self {
            DryFrame::SoftSand => "soft-sand",
            DryFrame::CriticalPorosity => "critical-porosity",
        }
    }
}

/// One lithology end-member: compaction and mineral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lithology {
    /// Surface porosity φ0.
    pub phi0: f64,
    /// Athy coefficient c (1/km).
    pub athy_c: f64,
    /// Mineral bulk / shear modulus (GPa) and density (g/cc).
    pub k_m: f64,
    pub mu_m: f64,
    pub rho_m: f64,
    /// Critical porosity φ_c.
    pub phi_c: f64,
    /// Coordination number n.
    pub coord: f64,
}

/// Quartz (Mavko et al. 2009, mineral table).
pub const QUARTZ: (f64, f64, f64) = (36.6, 45.0, 2.65);
/// Clay (Mavko et al. 2009, "clay" average used for shales).
pub const CLAY: (f64, f64, f64) = (21.0, 7.0, 2.6);

/// Voigt-Reuss-Hill bulk and shear moduli and volume-average density of a
/// clay-quartz mineral with clay volume fraction `clay`.
pub fn clay_quartz_mineral(clay: f64) -> (f64, f64, f64) {
    let q = 1.0 - clay;
    let vrh = |a: f64, b: f64| 0.5 * ((clay * a + q * b) + 1.0 / (clay / a + q / b));
    (
        vrh(CLAY.0, QUARTZ.0),
        vrh(CLAY.1, QUARTZ.1),
        clay * CLAY.2 + q * QUARTZ.2,
    )
}

/// Shale clay volume fraction: a **fitted** parameter (spec §10 D1), tuned
/// with [`SHALE_COORDINATION`] and φ_c 0.65 so shale Vp ≈ 3000 m/s at 2 km
/// and the sand/shale contrast guard (test 8) passes. It makes the "shale" a
/// silty mudstone; documented range 0.2-0.9.
pub const SHALE_CLAY_FRACTION: f64 = 0.30;
/// Shale Hertz-Mindlin coordination number: a **fitted** stiffness knob
/// (spec §10 D1), not a measured contact count. Documented range 5-14; the
/// empirical n(φ) of Mavko, Mukerji & Dvorkin (2009), *The Rock Physics
/// Handbook*, 2nd ed., Cambridge UP, doi:10.1017/CBO9780511626753, gives ~9
/// at 36 % porosity and fewer above, so 12 at φ_c 0.65 is a calibration.
/// Using n as a free fit is standard in contact models (Avseth, Mukerji &
/// Mavko 2005, doi:10.1017/CBO9780511600074).
pub const SHALE_COORDINATION: f64 = 12.0;

/// Parameters of the physical model (spec §5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysParams {
    pub sand: Lithology,
    pub shale: Lithology,
    pub compaction: CompactionForm,
    /// σ0 (MPa) of the stress form; `None` = derived per lithology as
    /// (ρ_m − ρ_w)·g/c, the same curve at hydrostatic pressure.
    pub sigma0: Option<f64>,
    pub dry_frame: DryFrame,
    /// Seabed temperature (°C) and gradient (°C/km).
    pub t_seabed: f64,
    pub t_gradient: f64,
    /// Brine NaCl weight fraction.
    pub salinity: f64,
    pub gas_gravity: f64,
    pub oil_api: f64,
    /// Gas-oil ratio (L/L).
    pub oil_gor: f64,
    pub gas_saturation: f64,
    pub oil_saturation: f64,
    /// Water density for pore pressure (g/cc) and gravity (m/s²).
    pub rho_w: f64,
    pub g: f64,
}

impl Default for PhysParams {
    fn default() -> Self {
        let (kq, mq, rq) = QUARTZ;
        let (ks, ms, rs) = clay_quartz_mineral(SHALE_CLAY_FRACTION);
        Self {
            sand: Lithology {
                phi0: 0.49,
                athy_c: 0.27,
                k_m: kq,
                mu_m: mq,
                rho_m: rq,
                phi_c: 0.40,
                coord: 9.0,
            },
            shale: Lithology {
                phi0: 0.63,
                athy_c: 0.51,
                k_m: ks,
                mu_m: ms,
                rho_m: rs,
                phi_c: 0.65,
                coord: SHALE_COORDINATION,
            },
            compaction: CompactionForm::Depth,
            sigma0: None,
            dry_frame: DryFrame::SoftSand,
            t_seabed: 4.0,
            t_gradient: 30.0,
            salinity: 0.035,
            gas_gravity: 0.6,
            oil_api: 32.0,
            oil_gor: 100.0,
            gas_saturation: 0.8,
            oil_saturation: 0.8,
            rho_w: 1.03,
            g: 9.81,
        }
    }
}

impl PhysParams {
    /// Range checks of spec §5 (exit 2 at the CLI).
    pub fn validate(&self) -> Result<(), String> {
        let chk = |name: &str, v: f64, lo: f64, hi: f64| {
            if v.is_finite() && v >= lo && v <= hi {
                Ok(())
            } else {
                Err(format!("{name} must be in [{lo}, {hi}], got {v}"))
            }
        };
        chk("--sand-phi0", self.sand.phi0, 0.25, 0.6)?;
        chk("--sand-athy-c", self.sand.athy_c, 0.05, 1.0)?;
        chk("--shale-phi0", self.shale.phi0, 0.4, 0.8)?;
        chk("--shale-athy-c", self.shale.athy_c, 0.1, 1.5)?;
        if let Some(s) = self.sigma0 {
            chk("--athy-sigma0", s, 5.0, 200.0)?;
        }
        chk("--seabed-temperature", self.t_seabed, 0.0, 25.0)?;
        chk("--temperature-gradient", self.t_gradient, 15.0, 50.0)?;
        chk("--brine-salinity", self.salinity, 0.0, 0.3)?;
        chk("--gas-gravity", self.gas_gravity, 0.55, 1.0)?;
        chk("--oil-api", self.oil_api, 10.0, 50.0)?;
        chk("--oil-gor", self.oil_gor, 0.0, 300.0)?;
        chk("--gas-saturation", self.gas_saturation, 0.0, 1.0)?;
        chk("--oil-saturation", self.oil_saturation, 0.0, 1.0)?;
        if self.shale.phi_c <= self.shale.phi0 {
            return Err("shale critical porosity must exceed --shale-phi0".into());
        }
        Ok(())
    }
}

/// Voxels whose physical result was non-finite and fell back to brine sand
/// (spec §4.3). Reported in the run summary.
pub static GUARDED_VOXELS: AtomicU64 = AtomicU64::new(0);

/// Lowest sediment-layer Vp (f32 bits; positive floats order as u32) this
/// process produced, for the run-end depth-staircase warning. `u32::MAX`
/// until a layer voxel is evaluated.
pub static VP_MIN_OBSERVED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

#[inline]
fn note_vp(vp: f32) {
    let b = vp.to_bits();
    if vp > 0.0 && b < VP_MIN_OBSERVED.load(Ordering::Relaxed) {
        VP_MIN_OBSERVED.fetch_min(b, Ordering::Relaxed);
    }
}

/// Forget the recorded minimum. Every pipeline run entry calls this first,
/// so a process that runs several models (tests, library users) reports
/// each run's own minimum, not the lowest of all runs so far.
pub fn reset_observed_vp_min() {
    VP_MIN_OBSERVED.store(u32::MAX, Ordering::Relaxed);
}

/// Zero [`GUARDED_VOXELS`] (the NaN-guard count of the run-end line).
pub fn reset_guarded_voxels() {
    GUARDED_VOXELS.store(0, Ordering::Relaxed);
}

/// Reset every per-run record of this process ([`VP_MIN_OBSERVED`] and
/// [`GUARDED_VOXELS`]). Every pipeline run entry calls this first, so a
/// process that runs several models reports each run on its own.
pub fn reset_run_records() {
    reset_observed_vp_min();
    reset_guarded_voxels();
}

/// [`VP_MIN_OBSERVED`] in m/s, if any layer voxel was evaluated since the
/// last [`reset_observed_vp_min`].
pub fn observed_vp_min() -> Option<f32> {
    let b = VP_MIN_OBSERVED.load(Ordering::Relaxed);
    (b != u32::MAX).then(|| f32::from_bits(b))
}

/// Athy porosity φ0·exp(−c z) with z in metres, c in 1/km (unclamped).
#[inline]
pub fn athy_porosity(phi0: f64, c_per_km: f64, z_m: f64) -> f64 {
    phi0 * (-c_per_km * z_m * 1e-3).exp()
}

/// Effective (differential) pressure in MPa for Athy-compacting sediment at
/// burial z (m) below the seabed with hydrostatic pore pressure (§3.1):
/// g (ρ_m − ρ_w) [z − φ0 (1 − e^{−cz}) / c] · 1e-6.
#[inline]
pub fn effective_pressure(
    z_m: f64,
    phi0: f64,
    c_per_km: f64,
    rho_m: f64,
    rho_w: f64,
    g: f64,
) -> f64 {
    let c = c_per_km * 1e-3;
    let integ = if c > 0.0 {
        z_m - phi0 * (1.0 - (-c * z_m).exp()) / c
    } else {
        z_m * (1.0 - phi0)
    };
    g * (rho_m - rho_w) * 1000.0 * integ * 1e-6
}

/// Batzle-Wang pure water velocity coefficients w[i][j] (T^i P^j), Table 1.
const BW_W: [[f64; 4]; 5] = [
    [1402.85, 1.524, 3.437e-3, -1.197e-5],
    [4.871, -0.0111, 1.739e-4, -1.628e-6],
    [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
    [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
    [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13],
];

/// Pure water density (g/cc) and velocity (m/s), Batzle & Wang (1992).
pub fn bw_water(t: f64, p: f64) -> (f64, f64) {
    let rho = 1.0
        + 1e-6
            * (-80.0 * t - 3.3 * t * t + 0.00175 * t * t * t + 489.0 * p - 2.0 * t * p
                + 0.016 * t * t * p
                - 1.3e-5 * t * t * t * p
                - 0.333 * p * p
                - 0.002 * t * p * p);
    let mut v = 0.0;
    let mut ti = 1.0;
    for row in BW_W.iter() {
        let mut pj = 1.0;
        for &w in row.iter() {
            v += w * ti * pj;
            pj *= p;
        }
        ti *= t;
    }
    (rho, v)
}

/// Brine density (g/cc), velocity (m/s) and bulk modulus (GPa).
pub fn bw_brine(t: f64, p: f64, s: f64) -> (f64, f64, f64) {
    let (rw, vw) = bw_water(t, p);
    let rho = rw
        + s * (0.668
            + 0.44 * s
            + 1e-6
                * (300.0 * p - 2400.0 * p * s
                    + t * (80.0 + 3.0 * t - 3300.0 * s - 13.0 * p + 47.0 * p * s)));
    let v = vw
        + s * (1170.0 - 9.6 * t + 0.055 * t * t - 8.5e-5 * t * t * t + 2.6 * p
            - 0.0029 * t * p
            - 0.0476 * p * p)
        + s.powf(1.5) * (780.0 - 10.0 * p + 0.16 * p * p)
        - 1820.0 * s * s;
    (rho, v, rho * v * v * 1e-6)
}

/// Gas density (g/cc) and bulk modulus (GPa), Batzle & Wang (1992).
pub fn bw_gas(t: f64, p: f64, gravity: f64) -> (f64, f64) {
    let ta = t + 273.15;
    let ppr = p / (4.892 - 0.4048 * gravity);
    let tpr = ta / (94.72 + 170.75 * gravity);
    let a = 0.03 + 0.00527 * (3.5 - tpr).powi(3);
    let b = 0.642 * tpr - 0.007 * tpr.powi(4) - 0.52;
    let d = 0.45 + 8.0 * (0.56 - 1.0 / tpr).powi(2);
    let e = 0.109 * (3.85 - tpr).powi(2) * (-d * ppr.powf(1.2) / tpr).exp();
    let z = a * ppr + b + e;
    let dz = a - e * d * 1.2 * ppr.powf(0.2) / tpr;
    let rho = 28.8 * gravity * p / (z * 8.31441 * ta);
    let g0 =
        0.85 + 5.6 / (ppr + 2.0) + 27.1 / (ppr + 3.5).powi(2) - 8.7 * (-0.65 * (ppr + 1.0)).exp();
    let k = p * g0 / (1.0 - ppr / z * dz) * 1e-3;
    (rho, k)
}

/// Live-oil density (g/cc) and bulk modulus (GPa), Batzle & Wang (1992)
/// eqs. 20-24 (API, GOR in L/L, gas gravity): velocity from the eq. 22-23
/// pseudo-density, density ρ_G (eq. 24) at saturation.
pub fn bw_oil(t: f64, p: f64, api: f64, gor: f64, gravity: f64) -> (f64, f64) {
    let rho0 = 141.5 / (api + 131.5);
    let b0 = 0.972 + 0.00038 * (2.4 * gor * (gravity / rho0).sqrt() + t + 17.8).powf(1.175);
    let rho_p = rho0 / b0 / (1.0 + 0.001 * gor);
    // B&W 1992 eq. 24: ρ_G is the live-oil density at saturation, and B0
    // already carries the thermal expansion at T. Eqs. 18-19 adjust a
    // density *from standard conditions* (eq. 19's divisor is ~1 at 15.6 °C);
    // applying them to ρ_G again counted temperature twice (~4 % low at
    // 64 °C). The reservoir is taken to be at saturation (GOR is the
    // in-situ dissolved gas), so ρ = ρ_G, as in rockphypy `BW.rho_K_go`.
    let rho = (rho0 + 0.0012 * gravity * gor) / b0;
    let v = 2096.0 * (rho_p / (2.6 - rho_p)).sqrt() - 3.7 * t
        + 4.64 * p
        + 0.0115 * (4.12 * (1.08 / rho_p - 1.0).sqrt() - 1.0) * t * p;
    (rho, rho * v * v * 1e-6)
}

/// Wood (Reuss) mix of hydrocarbon at saturation `s_hc` with brine:
/// returns (ρ_fl, K_fl).
#[inline]
pub fn wood(s_hc: f64, rho_hc: f64, k_hc: f64, rho_b: f64, k_b: f64) -> (f64, f64) {
    let k = 1.0 / (s_hc / k_hc + (1.0 - s_hc) / k_b);
    (s_hc * rho_hc + (1.0 - s_hc) * rho_b, k)
}

/// Poisson's ratio of a mineral.
#[inline]
fn poisson(k: f64, mu: f64) -> f64 {
    (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
}

/// Hertz-Mindlin pack moduli (GPa) at critical porosity, effective pressure
/// `p_gpa` (GPa), coordination number `n`.
pub fn hertz_mindlin(k_m: f64, mu_m: f64, phi_c: f64, n: f64, p_gpa: f64) -> (f64, f64) {
    let nu = poisson(k_m, mu_m);
    let a = n * n * (1.0 - phi_c).powi(2) * mu_m * mu_m * p_gpa;
    let k = (a / (18.0 * PI * PI * (1.0 - nu).powi(2))).cbrt();
    let mu = (5.0 - 4.0 * nu) / (5.0 * (2.0 - nu))
        * (3.0 * a / (2.0 * PI * PI * (1.0 - nu).powi(2))).cbrt();
    (k, mu)
}

/// Soft-sand dry moduli (Dvorkin & Nur 1996, modified lower HS bound).
pub fn soft_sand_dry(k_m: f64, mu_m: f64, phi: f64, phi_c: f64, n: f64, p_gpa: f64) -> (f64, f64) {
    let (khm, mhm) = hertz_mindlin(k_m, mu_m, phi_c, n, p_gpa);
    let a = phi / phi_c;
    let k =
        1.0 / (a / (khm + 4.0 / 3.0 * mhm) + (1.0 - a) / (k_m + 4.0 / 3.0 * mhm)) - 4.0 / 3.0 * mhm;
    let z = mhm / 6.0 * (9.0 * khm + 8.0 * mhm) / (khm + 2.0 * mhm);
    let mu = 1.0 / (a / (mhm + z) + (1.0 - a) / (mu_m + z)) - z;
    (k, mu)
}

/// Gassmann saturated bulk modulus (GPa).
#[inline]
pub fn gassmann(k_dry: f64, k_m: f64, k_fl: f64, phi: f64) -> f64 {
    let den = (phi / k_fl + (1.0 - phi) / k_m - k_dry / (k_m * k_m)).max(1e-9);
    k_dry + (1.0 - k_dry / k_m).powi(2) / den
}

/// Inverse Gassmann: dry bulk modulus from a saturated one.
#[inline]
pub fn gassmann_inverse(k_sat: f64, k_m: f64, k_fl: f64, phi: f64) -> f64 {
    let a = k_sat * (phi * k_m / k_fl + 1.0 - phi) - k_m;
    let b = phi * k_m / k_fl + k_sat / k_m - 1.0 - phi;
    a / b
}

/// (ρ, Vp, Vs) from moduli (GPa) and density (g/cc).
#[inline]
pub fn elastic_from_moduli(k: f64, mu: f64, rho: f64) -> (f64, f64, f64) {
    (
        rho,
        ((k + 4.0 / 3.0 * mu) / rho).sqrt() * 1000.0,
        (mu / rho).sqrt() * 1000.0,
    )
}

/// Pore fluid kind of an end-member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoreFluid {
    Brine,
    Oil,
    Gas,
}

/// Full chain for one end-member at burial `z` (m) under water depth `w`
/// (m), f64. Returns (ρ, Vp, Vs) and porosity.
pub fn end_member_f64(
    lith: &Lithology,
    fluid: PoreFluid,
    z: f64,
    w: f64,
    p: &PhysParams,
) -> ((f64, f64, f64), f64) {
    let z = z.max(0.0);
    let w = w.max(0.0);
    let peff = effective_pressure(z, lith.phi0, lith.athy_c, lith.rho_m, p.rho_w, p.g).max(0.01);
    let phi_raw = match p.compaction {
        CompactionForm::Depth => athy_porosity(lith.phi0, lith.athy_c, z),
        CompactionForm::Stress => {
            let s0 = p.sigma0.unwrap_or_else(|| {
                (lith.rho_m - p.rho_w) * 1000.0 * p.g / (lith.athy_c * 1e-3) * 1e-6
            });
            lith.phi0 * (-peff / s0).exp()
        }
    };
    let phi = phi_raw.clamp(0.01, lith.phi_c - 0.001);
    let t = (p.t_seabed + p.t_gradient * z * 1e-3).clamp(0.0, 200.0);
    let pp = (p.rho_w * 1000.0 * p.g * (w + z) * 1e-6).clamp(0.1, 100.0);
    let s = p.salinity.clamp(0.0, 0.3);
    let (rb, _, kb) = bw_brine(t, pp, s);
    let (rho_fl, k_fl) = match fluid {
        PoreFluid::Brine => (rb, kb),
        PoreFluid::Gas => {
            let (rg, kg) = bw_gas(t, pp, p.gas_gravity);
            wood(p.gas_saturation, rg, kg, rb, kb)
        }
        PoreFluid::Oil => {
            let (ro, ko) = bw_oil(t, pp, p.oil_api, p.oil_gor, p.gas_gravity);
            wood(p.oil_saturation, ro, ko, rb, kb)
        }
    };
    let k_fl = k_fl.max(1e-4);
    let (kd, md) = match p.dry_frame {
        DryFrame::SoftSand => soft_sand_dry(
            lith.k_m,
            lith.mu_m,
            phi,
            lith.phi_c,
            lith.coord,
            peff * 1e-3,
        ),
        DryFrame::CriticalPorosity => {
            let f = 1.0 - phi / lith.phi_c;
            (lith.k_m * f, lith.mu_m * f)
        }
    };
    let kd = kd.clamp(0.0, lith.k_m);
    let md = md.clamp(0.0, lith.mu_m);
    let ks = gassmann(kd, lith.k_m, k_fl, phi);
    let rho = (1.0 - phi) * lith.rho_m + phi * rho_fl;
    (elastic_from_moduli(ks, md, rho), phi)
}

fn to32(e: (f64, f64, f64)) -> Elastic32 {
    Elastic32 {
        rho: e.0 as f32,
        vp: e.1 as f32,
        vs: e.2 as f32,
    }
}

fn finite(e: Elastic32) -> bool {
    e.rho.is_finite()
        && e.vp.is_finite()
        && e.vs.is_finite()
        && e.rho > 0.0
        && e.vp > 0.0
        && e.vs >= 0.0
}

/// One end-member in f32 with the NaN guard of spec §4.3: a non-finite
/// result falls back to brine sand at the same z and is counted.
pub fn end_member(lith: &Lithology, fluid: PoreFluid, z: f64, w: f64, p: &PhysParams) -> Elastic32 {
    let e = to32(end_member_f64(lith, fluid, z, w, p).0);
    debug_assert!(
        finite(e),
        "physical rock physics produced {e:?} at z={z} w={w}"
    );
    if finite(e) {
        e
    } else {
        GUARDED_VOXELS.fetch_add(1, Ordering::Relaxed);
        to32(end_member_f64(&p.sand, PoreFluid::Brine, z, w, p).0)
    }
}

/// [`end_member`] without touching the guard counter: (value, guarded).
fn end_member_checked(lith: &Lithology, fluid: PoreFluid, z: f64, w: f64, p: &PhysParams) -> (Elastic32, bool) {
    let e = to32(end_member_f64(lith, fluid, z, w, p).0);
    if finite(e) {
        (e, false)
    } else {
        (to32(end_member_f64(&p.sand, PoreFluid::Brine, z, w, p).0), true)
    }
}

/// Slots of the per-thread end-member memo (power of two).
const CACHE_SLOTS: usize = 4096;

/// Per-thread end-member memo: the params it was filled with, then a
/// direct-mapped table of (key, value, guarded). The key packs the kind
/// (0 shale, 1..=3 brine / oil / gas sand) with the exact z and w bits.
struct EndMemberCache {
    params: Option<PhysParams>,
    slots: Vec<(u64, u8, Elastic32, bool)>,
}

std::thread_local! {
    /// Depth is constant down a layer run (spec §2: one burial depth per
    /// layer) and every pipeline pass revisits the same columns, so a
    /// column has only a few dozen distinct (kind, z, w) inputs. The memo
    /// runs the chain once per input instead of once per voxel and pass.
    /// A hit is keyed on the exact input bits and params, so it returns the
    /// bit-identical value; the NaN guard is still counted per voxel.
    static END_MEMBER_CACHE: std::cell::RefCell<EndMemberCache> =
        const { std::cell::RefCell::new(EndMemberCache { params: None, slots: Vec::new() }) };
}

fn cached_end_member(kind: u8, lith: &Lithology, fluid: PoreFluid, z: f32, w: f32, p: &PhysParams) -> Elastic32 {
    let key = ((z.to_bits() as u64) << 32) | w.to_bits() as u64;
    let (e, guarded) = END_MEMBER_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.params.as_ref() != Some(p) {
            c.params = Some(*p);
            c.slots.clear();
            c.slots.resize(CACHE_SLOTS, (0, u8::MAX, Elastic32::default(), false));
        }
        let h = (key ^ ((kind as u64) << 61)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let slot = &mut c.slots[(h >> 52) as usize & (CACHE_SLOTS - 1)];
        if slot.0 == key && slot.1 == kind {
            (slot.2, slot.3)
        } else {
            let (e, g) = end_member_checked(lith, fluid, z as f64, w as f64, p);
            *slot = (key, kind, e, g);
            (e, g)
        }
    });
    if guarded {
        GUARDED_VOXELS.fetch_add(1, Ordering::Relaxed);
    }
    e
}

/// Physical shale end-member at burial z (m), water depth w (m).
pub fn physical_shale(z: f32, w: f32, p: &PhysParams) -> Elastic32 {
    cached_end_member(0, &p.shale, PoreFluid::Brine, z, w, p)
}

/// Physical sand end-member of `fluid`.
pub fn physical_sand(fluid: Fluid, z: f32, w: f32, p: &PhysParams) -> Elastic32 {
    match fluid {
        Fluid::Brine => cached_end_member(1, &p.sand, PoreFluid::Brine, z, w, p),
        Fluid::Oil => cached_end_member(2, &p.sand, PoreFluid::Oil, z, w, p),
        Fluid::Gas => cached_end_member(3, &p.sand, PoreFluid::Gas, z, w, p),
    }
}

/// Physical counterpart of [`super::legacy::voxel_properties`] (spec §2):
/// one depth per layer, `depth[k + layer shift + shale ρ shift]` (D4: the
/// shale rho shift is the single jitter for the whole layer), drives both
/// end-members; sand and shale are mixed by net-to-gross with the legacy
/// [`mix_f32`]. `w` is the column's water depth (m).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn physical_voxel_properties(
    depth: &[f32],
    k: usize,
    kind: VoxelKind,
    shifts: &[LayerShifts],
    mixing: MixingMethod,
    salt: Elastic32,
    w: f32,
    p: &PhysParams,
) -> Elastic32 {
    match kind {
        VoxelKind::Water => WATER,
        VoxelKind::Salt => salt,
        VoxelKind::Unfilled => Elastic32::default(),
        VoxelKind::Layer { layer, ng, fluid } => {
            let nk = depth.len();
            let s = shifts[layer];
            let z = depth[shifted_index(k, s.layer + s.props[0][0], nk)];
            let shale = physical_shale(z, w, p);
            let e = if ng > 0.0 {
                mix_f32(shale, physical_sand(fluid, z, w, p), ng, mixing)
            } else {
                shale
            };
            note_vp(e.vp);
            e
        }
    }
}

/// Physical counterpart of [`super::legacy::legacy_column_properties`].
#[allow(clippy::too_many_arguments)]
pub fn physical_column_properties(
    depth: &[f32],
    kinds: &[VoxelKind],
    shifts: &[LayerShifts],
    mixing: MixingMethod,
    salt: Elastic32,
    w: f32,
    p: &PhysParams,
    rho: &mut [f32],
    vp: &mut [f32],
    vs: &mut [f32],
) {
    let nk = depth.len();
    assert!(kinds.len() == nk && rho.len() == nk && vp.len() == nk && vs.len() == nk);
    for k in 0..nk {
        let e = physical_voxel_properties(depth, k, kinds[k], shifts, mixing, salt, w, p);
        rho[k] = e.rho;
        vp[k] = e.vp;
        vs[k] = e.vs;
    }
    forward_fill_zeros(rho);
    forward_fill_zeros(vp);
    forward_fill_zeros(vs);
}

/// Rock model of a run: legacy trends (master 77131867) or the physical
/// chain. `RpmModel` dispatches every property call on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RockModel {
    /// `--legacy-rock-physics`: depth-trend polynomials.
    Legacy,
    /// Athy + soft sand + Gassmann + Batzle-Wang (default).
    Physical(PhysParams),
}

impl Default for RockModel {
    fn default() -> Self {
        RockModel::Physical(PhysParams::default())
    }
}

