//! Optional fault drag (roadmap 7c, spec-fault-drag).
//!
//! One extra vertical term in each fault's stage of the heave gather
//! ([`super::kinematics`]). For fault `n` at its stage position `x`:
//!
//! ```text
//! ρ     = |f − 1| / |∇_m f|,   ∇_m f = (f_x/bin, f_y/bin, f_z/dz)   (metres)
//! D_loc = w_tip · profile[k]                 (no hanging-wall gate, samples)
//! g(q)  = cos²(π q / 2) for q < 1, else 0
//! W     = min(ω · D_max,m, W_cap)            (D_max,m = T·dz / sin δ_c)
//! w     = +s β_h D_loc g(ρ/W)   (f ≤ 1, hanging wall)
//!         −s β_f D_loc g(ρ/W)   (f > 1, footwall)
//! a     = clip(z − d − w, 0, nk − 1 + P)
//! ```
//!
//! `s = +1` is reverse drag. The heave stays driven by `d` only. The
//! source may lie up to `P = ceil(β_f · T_max) + 1` samples below the cube
//! (footwall uplift), so consumers that read the unfaulted fill take it on
//! `nk + P` samples ([`FaultModel::drag_pad`]).

use super::apply::FaultModel;
use super::kinematics::FaultKinematics;

/// Grasemann et al. (2005) critical marker–fault angle (degrees).
pub const DRAG_THETA_C_DEG: f64 = 35.0;

/// Monotonicity budget: `B = β_max · T_max · (π/2) · dz / W_min < 0.5`.
pub const DRAG_B_MAX: f64 = 0.5;

/// Drag sense (spec §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragSense {
    /// Grasemann angle rule, fixed per fault from θ at its peak-throw point.
    #[default]
    Auto,
    /// `s = +1`.
    Reverse,
    /// `s = −1`.
    Normal,
}

/// Taper of the drag zone. Only [`DragTaper::Cos2`] is used by the
/// pipeline; the others are mutation-test hooks (spec §4, M4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragTaper {
    #[default]
    Cos2,
    /// Mutant M4a: `exp(−3q)`, no cutoff.
    #[doc(hidden)]
    ExpMutant,
    /// Mutant M4b: box (1 for q < 1).
    #[doc(hidden)]
    BoxMutant,
}

/// User-facing drag settings (CLI `--fault-drag*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragParams {
    /// β_h, hanging-wall amplitude (fraction of local throw, 0–0.5).
    pub beta_h: f64,
    /// β_f, footwall amplitude (0–0.5).
    pub beta_f: f64,
    /// ω, zone width in units of D_max (1–10).
    pub width: f64,
    /// W_cap in metres (50–800).
    pub max_width_m: f64,
    pub sense: DragSense,
    /// Test hook (mutants only).
    #[doc(hidden)]
    pub taper: DragTaper,
    /// Test hook (mutant M6): drop the bottom pad.
    #[doc(hidden)]
    pub no_pad: bool,
}

impl Default for DragParams {
    fn default() -> Self {
        Self {
            beta_h: 0.30,
            beta_f: 0.15,
            width: 3.0,
            max_width_m: 400.0,
            sense: DragSense::Auto,
            taper: DragTaper::Cos2,
            no_pad: false,
        }
    }
}

impl DragParams {
    /// Range checks of spec §2 (the CLI's exit-2 errors).
    pub fn validate(&self) -> Result<(), String> {
        let r = |v: f64, lo: f64, hi: f64, n: &str| {
            if (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("{n} {v} outside {lo}..={hi}"))
            }
        };
        r(self.beta_h, 0.0, 0.5, "--fault-drag-hw")?;
        r(self.beta_f, 0.0, 0.5, "--fault-drag-fw")?;
        r(self.width, 1.0, 10.0, "--fault-drag-width")?;
        r(self.max_width_m, 50.0, 800.0, "--fault-drag-max-width-m")?;
        if self.sense == DragSense::Normal && self.beta_h + self.beta_f > 0.9 + 1e-12 {
            return Err(format!(
                "--fault-drag-sense normal needs --fault-drag-hw + --fault-drag-fw <= 0.9 (got {})",
                self.beta_h + self.beta_f
            ));
        }
        Ok(())
    }
}

/// Per-fault resolved drag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragFault {
    /// Zone width W (metres).
    pub w_m: f64,
    /// Sense `s` (±1).
    pub s: f64,
    /// θ at the peak-throw point (degrees).
    pub theta_deg: f64,
}

/// Resolved drag of a model (global, once).
#[derive(Debug, Clone, PartialEq)]
pub struct DragModel {
    pub params: DragParams,
    /// Effective amplitudes (after the B clamp and the normal-sense cap).
    pub beta_h: f64,
    pub beta_f: f64,
    pub faults: Vec<DragFault>,
    /// Bottom pad P (samples).
    pub pad: usize,
    /// B with the effective β.
    pub b: f64,
    /// B with the requested β.
    pub b_requested: f64,
    pub bin_m: f64,
    pub dz_m: f64,
    /// Warnings for the run metadata (β clamped, …).
    pub warnings: Vec<String>,
}

impl DragModel {
    /// Taper `g(q)`.
    #[inline]
    pub fn taper(&self, q: f64) -> f64 {
        match self.params.taper {
            DragTaper::Cos2 if q < 1.0 => (std::f64::consts::FRAC_PI_2 * q).cos().powi(2),
            DragTaper::Cos2 => 0.0,
            DragTaper::ExpMutant => (-3.0 * q).exp(),
            DragTaper::BoxMutant if q < 1.0 => 1.0,
            DragTaper::BoxMutant => 0.0,
        }
    }

    /// Drag `w` (samples, + = rock moves down) of fault `n` at a stage
    /// position with surface value `v`, voxel gradient `g` and local throw
    /// `d_loc` (no hanging-wall gate).
    #[inline]
    pub fn displacement(&self, n: usize, v: f64, g: [f64; 3], d_loc: f64) -> f64 {
        let df = &self.faults[n];
        let gm = [g[0] / self.bin_m, g[1] / self.bin_m, g[2] / self.dz_m];
        let gn = (gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]).sqrt();
        if !(gn > 0.0) {
            return 0.0;
        }
        let rho = (v - 1.0).abs() / gn;
        let t = self.taper(rho / df.w_m);
        if t == 0.0 {
            return 0.0;
        }
        if v <= 1.0 {
            df.s * self.beta_h * d_loc * t
        } else {
            -df.s * self.beta_f * d_loc * t
        }
    }

    /// One-line summary for the run metadata.
    pub fn summary(&self) -> String {
        let senses: Vec<&str> = self
            .faults
            .iter()
            .map(|f| if f.s > 0.0 { "reverse" } else { "normal" })
            .collect();
        let widths: Vec<String> = self.faults.iter().map(|f| format!("{:.0}", f.w_m)).collect();
        format!(
            "fault drag: beta_hw={:.3} beta_fw={:.3} width={} cap={} m, W_m=[{}], sense=[{}], pad={} samples, B={:.3}{}",
            self.beta_h,
            self.beta_f,
            self.params.width,
            self.params.max_width_m,
            widths.join(","),
            senses.join(","),
            self.pad,
            self.b,
            if self.warnings.is_empty() {
                String::new()
            } else {
                format!(" (warning: {})", self.warnings.join("; "))
            }
        )
    }
}

/// Physical dip (degrees, from the voxel gradient) of a surface patch.
fn physical_dip_deg(g: [f64; 3], bin_m: f64, dz_m: f64) -> f64 {
    let gp = [g[0] / bin_m, g[1] / bin_m, g[2] / dz_m];
    let nh = gp[0].hypot(gp[1]);
    nh.atan2(gp[2].abs()).to_degrees()
}

impl FaultModel {
    /// The same model with fault drag (spec-fault-drag). Needs heave
    /// kinematics (bin, depth step, dip floor); a legacy model is returned
    /// unchanged. Call after [`Self::with_tips`].
    pub fn with_drag(self, p: &DragParams) -> Self {
        let FaultKinematics::Heave(hp) = self.kinematics() else {
            return self;
        };
        let (bin_m, dz_m) = (hp.bin_m, hp.dz_m);
        let mut warnings = Vec::new();
        let mut faults = Vec::new();
        let mut t_max = 0.0f64;
        for f in self.faults() {
            let t = f.profile.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            t_max = t_max.max(t);
            let c = [f.center[0] as f64, f.center[1] as f64, f.center[2] as f64];
            let g = f.geometry.gradient_at(c[0], c[1], c[2]);
            // θ: angle between the (flat) horizon normal ẑ and the fault
            // normal = the physical dip, with the heave's dip floor.
            let theta = physical_dip_deg(g, bin_m, dz_m).max(hp.min_dip_deg);
            let d_max_m = match &f.tip {
                Some(tl) => tl.d_max,
                None => t * dz_m / theta.to_radians().sin(),
            };
            let w_m = (p.width * d_max_m).min(p.max_width_m);
            let s = match p.sense {
                DragSense::Reverse => 1.0,
                DragSense::Normal => -1.0,
                DragSense::Auto if theta >= DRAG_THETA_C_DEG => 1.0,
                DragSense::Auto => -1.0,
            };
            faults.push(DragFault {
                w_m,
                s,
                theta_deg: theta,
            });
        }
        let (mut bh, mut bf) = (p.beta_h, p.beta_f);
        if faults.iter().any(|f| f.s < 0.0) && bh + bf > 0.9 {
            let k = 0.9 / (bh + bf);
            bh *= k;
            bf *= k;
            warnings.push(format!("normal-sense beta_hw + beta_fw capped at 0.9 (x{k:.3})"));
        }
        let w_min = faults.iter().map(|f| f.w_m).fold(f64::INFINITY, f64::min);
        let bterm = |b: f64| {
            if faults.is_empty() || !(w_min > 0.0) {
                0.0
            } else {
                b * t_max * std::f64::consts::FRAC_PI_2 * dz_m / w_min
            }
        };
        let b_requested = bterm(bh.max(bf));
        if b_requested >= DRAG_B_MAX {
            // Clamp β to just under the bound.
            let lim = 0.999 * DRAG_B_MAX / bterm(1.0);
            warnings.push(format!(
                "monotonicity bound B={b_requested:.3} >= {DRAG_B_MAX}: beta clamped to {lim:.4}"
            ));
            bh = bh.min(lim);
            bf = bf.min(lim);
        }
        let b = bterm(bh.max(bf));
        let pad = if p.no_pad || faults.is_empty() {
            0
        } else {
            (bf * t_max).ceil() as usize + 1
        };
        let m = DragModel {
            params: *p,
            beta_h: bh,
            beta_f: bf,
            faults,
            pad,
            b,
            b_requested,
            bin_m,
            dz_m,
            warnings,
        };
        self.set_drag(Some(std::sync::Arc::new(m)))
    }
}
