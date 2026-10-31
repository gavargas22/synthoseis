//! Tip-bounded throw and displacement–length scaling (spec-fault-kinematics
//! 7b).
//!
//! Every heave fault gets an elliptical tip line on the plane tangent to
//! its surface at the centre `c` (metres, `s` along strike, `t` along dip):
//!
//! ```text
//! r²   = (s / R_s)² + (t / R_t)²,   R_s = L / 2,   R_t = R_s / A
//! D(r) = D_max · cos²(π r / 2)  for r < 1, else 0
//! L    = D_max / γ              (D_max = γ L, n = 1, Schlische et al. 1996)
//! D_max = T · dz / sin δ_c      (δ_c = physical dip at c, dip-floored)
//! ```
//!
//! The legacy Gaussian lateral weight of the throw is replaced by the taper
//! `cos²(π r / 2)`: the throw at a point is `taper · z_shift(k)`, so the
//! vertical profile and the seabed roll-down still apply and the throw at
//! the centre is unchanged. γ is drawn per fault from its own sub-stream
//! (`seed ^ 0xFA17_0000_0000_0002`), so the legacy draws are untouched.

use super::apply::FaultModel;
use super::params::FaultRng;

/// Sub-stream salt of the 7b draws (spec §3).
pub const TIP_SEED_SALT: u64 = 0xFA17_0000_0000_0002;

/// Throw taper across the tip ellipse. Only [`TipTaper::Cos2`] is used by
/// the pipeline; the others are mutants for the spec §9 tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TipTaper {
    /// `cos²(π r / 2)` for `r < 1`, else 0 (C¹, finite tip gradient).
    #[default]
    Cos2,
    /// Mutant (row 6): `√(1 − r²)`, infinite gradient at the tip.
    #[doc(hidden)]
    SqrtMutant,
    /// Mutant (row 7): `cos²(π r / 2)` without the `r < 1` cutoff.
    #[doc(hidden)]
    UncutMutant,
}

/// 7b settings (spec §5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TipParams {
    /// γ = D_max / L is drawn log-uniform in `[gamma_min, gamma_max]`.
    pub gamma_min: f64,
    pub gamma_max: f64,
    /// Aspect ratio A = R_s / R_t (fixed, spec Q4).
    pub aspect: f64,
    /// D–L exponent n in `D_max = γ Lⁿ` (1; tests use 2 as the mutant).
    pub dl_exponent: f64,
    pub taper: TipTaper,
}

impl Default for TipParams {
    fn default() -> Self {
        Self {
            gamma_min: 0.01,
            gamma_max: 0.1,
            aspect: 2.0,
            dl_exponent: 1.0,
            taper: TipTaper::Cos2,
        }
    }
}

/// Elliptical tip line of one fault.
#[derive(Debug, Clone, PartialEq)]
pub struct TipLine {
    /// Centre `c` (voxels).
    pub c: [f64; 3],
    /// Physical size of a voxel step `[bin, bin, dz]` (metres).
    pub scale: [f64; 3],
    /// Unit strike and in-plane dip directions (physical space).
    pub s_hat: [f64; 3],
    pub t_hat: [f64; 3],
    /// Semi-axes (metres).
    pub r_s: f64,
    pub r_t: f64,
    /// γ, D_max (metres of slip) and L (metres).
    pub gamma: f64,
    pub d_max: f64,
    pub length: f64,
    /// Physical dip at the centre (degrees, after the floor).
    pub dip_c_deg: f64,
    pub taper: TipTaper,
}

impl TipLine {
    /// Build the tip line of a fault with throw `throw` (samples) whose
    /// surface gradient at the centre `c` is `g` (voxel units).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: [f64; 3],
        g: [f64; 3],
        throw: f64,
        gamma: f64,
        bin_m: f64,
        dz_m: f64,
        min_dip_deg: f64,
        tp: &TipParams,
    ) -> Self {
        let scale = [bin_m, bin_m, dz_m];
        // Physical normal: ∇f in metres⁻¹.
        let gp = [g[0] / bin_m, g[1] / bin_m, g[2] / dz_m];
        let gn = (gp[0] * gp[0] + gp[1] * gp[1] + gp[2] * gp[2]).sqrt();
        let n = if gn > 0.0 {
            [gp[0] / gn, gp[1] / gn, gp[2] / gn]
        } else {
            [0.0, 0.0, 1.0]
        };
        let nh = n[0].hypot(n[1]);
        // Strike = ẑ × n̂ (horizontal); x axis on a horizontal patch.
        let s_hat = if nh > 1e-12 {
            [-n[1] / nh, n[0] / nh, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let t_hat = [
            n[1] * s_hat[2] - n[2] * s_hat[1],
            n[2] * s_hat[0] - n[0] * s_hat[2],
            n[0] * s_hat[1] - n[1] * s_hat[0],
        ];
        let dip = nh.atan2(n[2].abs()).to_degrees().max(min_dip_deg);
        let d_max = throw * dz_m / dip.to_radians().sin();
        let length = (d_max / gamma).powf(1.0 / tp.dl_exponent);
        let r_s = length / 2.0;
        Self {
            c,
            scale,
            s_hat,
            t_hat,
            r_s,
            r_t: r_s / tp.aspect,
            gamma,
            d_max,
            length,
            dip_c_deg: dip,
            taper: tp.taper,
        }
    }

    /// Normalised tip radius `r` at voxel position `(x, y, z)`.
    #[inline]
    pub fn radius(&self, x: f64, y: f64, z: f64) -> f64 {
        let p = [
            (x - self.c[0]) * self.scale[0],
            (y - self.c[1]) * self.scale[1],
            (z - self.c[2]) * self.scale[2],
        ];
        let s = p[0] * self.s_hat[0] + p[1] * self.s_hat[1] + p[2] * self.s_hat[2];
        let t = p[0] * self.t_hat[0] + p[1] * self.t_hat[1] + p[2] * self.t_hat[2];
        ((s / self.r_s).powi(2) + (t / self.r_t).powi(2)).sqrt()
    }

    /// Throw weight `D / D_max` at `(x, y, z)`: `cos²(π r / 2)` inside the
    /// tip line, exactly 0 on and outside it.
    #[inline]
    pub fn factor(&self, x: f64, y: f64, z: f64) -> f64 {
        let r = self.radius(x, y, z);
        let half_pi = std::f64::consts::FRAC_PI_2;
        match self.taper {
            TipTaper::Cos2 if r < 1.0 => (half_pi * r).cos().powi(2),
            TipTaper::Cos2 => 0.0,
            TipTaper::SqrtMutant if r < 1.0 => (1.0 - r * r).sqrt(),
            TipTaper::SqrtMutant => 0.0,
            TipTaper::UncutMutant => (half_pi * r).cos().powi(2),
        }
    }
}

/// γ of every fault parameter index `n` (log-uniform, own sub-stream).
pub fn draw_gamma(seed: u64, n: usize, tp: &TipParams) -> f64 {
    let mut rng = FaultRng::new(seed ^ TIP_SEED_SALT).fork(n as u64 + 1);
    let (a, b) = (tp.gamma_min.ln(), tp.gamma_max.ln());
    (a + (b - a) * rng.next_f64()).exp()
}

impl FaultModel {
    /// The same model with 7b tip lines on every fault (spec §4): γ from
    /// `seed`'s tip sub-stream, indexed by the fault's parameter index.
    /// Needs heave kinematics for the bin, depth step and dip floor; a
    /// legacy model is returned unchanged.
    pub fn with_tips(mut self, seed: u64, tp: &TipParams) -> Self {
        let super::kinematics::FaultKinematics::Heave(hp) = self.kinematics() else {
            return self;
        };
        let skipped: Vec<usize> = self.skipped().iter().map(|s| s.0).collect();
        let mut n = 0usize;
        for f in self.faults_mut_tips() {
            while skipped.contains(&n) {
                n += 1;
            }
            let c = [f.center[0] as f64, f.center[1] as f64, f.center[2] as f64];
            let g = f.geometry.gradient_at(c[0], c[1], c[2]);
            let gamma = draw_gamma(seed, n, tp);
            f.tip = Some(TipLine::new(
                c,
                g,
                f.params.throw,
                gamma,
                hp.bin_m,
                hp.dz_m,
                hp.min_dip_deg,
                tp,
            ));
            n += 1;
        }
        self
    }

    /// Whether any fault has a 7b tip line.
    pub fn has_tips(&self) -> bool {
        self.faults().iter().any(|f| f.tip.is_some())
    }
}
