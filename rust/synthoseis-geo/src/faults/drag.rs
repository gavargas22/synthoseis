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

/// Per-fault drag-slope target (spec §3, corrected Nov 1 2026):
/// `B_f = β · T_f · (π/2) · cos θ_f · dz / W_f ≤ 0.16`, β clamped per fault.
pub const DRAG_B_TARGET: f64 = 0.16;

/// Numerical slope limiter (spec §3 step 2). NOT physics: a per-column
/// guard on the read slope `Δlookup / Δk` over the ±[`LIMITER_HALF`]
/// window (clipped at the column ends). Wherever the drag-off read slope
/// of a window is ≥ [`LIMITER_MIN_SLOPE`] and the drag-on window holds no
/// fault-plane label, the drag-on read slope is kept ≥ the same value
/// (drag never pushes a window past 2× stretch). Windows already past 2×
/// with drag off are left as they are.
pub const LIMITER_MIN_SLOPE: f64 = 0.5;
/// Half window of the limiter (samples).
pub const LIMITER_HALF: usize = 4;

/// Per-column activation record of the numerical limiter (samples raised
/// per column; `u32::MAX` = column not gathered yet). Stores, never adds,
/// so re-gathering a column (cache misses, halos) cannot double count.
pub struct LimiterLog {
    cols: Vec<std::sync::atomic::AtomicU32>,
}

impl LimiterLog {
    fn new(n: usize) -> Self {
        Self {
            cols: (0..n).map(|_| std::sync::atomic::AtomicU32::new(u32::MAX)).collect(),
        }
    }

    pub(crate) fn store(&self, col: usize, raised: u32) {
        if let Some(c) = self.cols.get(col) {
            c.store(raised, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub(crate) fn get(&self, col: usize) -> Option<u32> {
        let v = self.cols.get(col)?.load(std::sync::atomic::Ordering::Relaxed);
        (v != u32::MAX).then_some(v)
    }
}

impl std::fmt::Debug for LimiterLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LimiterLog({} columns)", self.cols.len())
    }
}

impl PartialEq for LimiterLog {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// [`LimiterStats`] plus whether every column has been recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LimiterStatsPartial {
    pub samples: u64,
    pub columns: u64,
    pub complete: bool,
}

/// Activation count of the numerical limiter over a whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LimiterStats {
    /// Lookup samples raised.
    pub samples: u64,
    /// Columns with at least one raised sample.
    pub columns: u64,
}

/// The limiter on one column (in place on `on`, the drag-on source depth
/// of every sample, already f32-rounded). `ids` are the drag-on labels,
/// `off(k)` the drag-off source depth. Returns the samples raised.
pub(crate) fn limit_column(
    on: &mut [f64],
    ids: &[u8],
    top_src: f64,
    mut off: impl FnMut() -> Vec<f64>,
) -> u32 {
    let nk = on.len();
    if nk < 2 {
        return 0;
    }
    let win = |k: usize| (k.saturating_sub(LIMITER_HALF), (k + LIMITER_HALF).min(nk - 1));
    let short = |lk: &[f64], lo: usize, hi: usize| {
        ((lk[hi] as f32 - lk[lo] as f32) as f64) < LIMITER_MIN_SLOPE * (hi - lo) as f64
    };
    let free = |lo: usize, hi: usize| ids[lo..=hi].iter().all(|&d| d == 0);
    // Cheap pre-check: no unlabelled drag-on window below the slope.
    if !(0..nk).any(|k| {
        let (lo, hi) = win(k);
        free(lo, hi) && short(on, lo, hi)
    }) {
        return 0;
    }
    let off = off();
    let mut raised = 0u32;
    // Windows sorted by their lower end... processed by upper end, so the
    // lower end of every window is final when the window is enforced.
    let mut ws: Vec<(usize, usize)> = (0..nk).map(win).collect();
    ws.sort_by_key(|&(lo, hi)| (hi, lo));
    ws.dedup();
    let mut last = usize::MAX;
    for (lo, hi) in ws {
        if !free(lo, hi) || !short(on, lo, hi) || short(&off, lo, hi) {
            continue;
        }
        let lo32 = on[lo] as f32;
        let mut v = (on[lo] + LIMITER_MIN_SLOPE * (hi - lo) as f64) as f32;
        while ((v - lo32) as f64) < LIMITER_MIN_SLOPE * (hi - lo) as f64 {
            v = f32::from_bits(v.to_bits() + 1);
        }
        on[hi] = (v as f64).min(top_src) as f32 as f64;
        if hi != last {
            raised += 1;
            last = hi;
        }
    }
    raised
}

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
    /// Test hook (mutant M7): drop the numerical slope limiter.
    #[doc(hidden)]
    pub no_limiter: bool,
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
            no_limiter: false,
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
    /// Peak throw T_f (samples).
    pub t: f64,
    /// Effective β_h, β_f of this fault (after the B_f clamp).
    pub beta_h: f64,
    pub beta_f: f64,
    /// B_f with the effective β (max of the two).
    pub b: f64,
}

/// One per-fault β clamp (spec §3 step 1), recorded in the run metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragClamp {
    pub fault: usize,
    /// T_f (samples), W_f (metres), θ_f (degrees).
    pub t: f64,
    pub w_m: f64,
    pub theta_deg: f64,
    /// (β_h, β_f) before and after.
    pub before: (f64, f64),
    pub after: (f64, f64),
}

/// Resolved drag of a model (global, once).
#[derive(Debug, Clone, PartialEq)]
pub struct DragModel {
    pub params: DragParams,
    /// Amplitudes after the normal-sense cap (the per-fault B_f clamp is
    /// in [`DragFault::beta_h`] / [`DragFault::beta_f`]).
    pub beta_h: f64,
    pub beta_f: f64,
    pub faults: Vec<DragFault>,
    /// Bottom pad P (samples).
    pub pad: usize,
    /// max_f B_f with the effective β.
    pub b: f64,
    /// max_f B_f with the requested β.
    pub b_requested: f64,
    /// Every per-fault β clamp.
    pub clamps: Vec<DragClamp>,
    pub bin_m: f64,
    pub dz_m: f64,
    /// Warnings for the run metadata (β clamped, …).
    pub warnings: Vec<String>,
    /// Activations of the numerical slope limiter (not physics).
    pub limiter: std::sync::Arc<LimiterLog>,
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
            df.s * df.beta_h * d_loc * t
        } else {
            -df.s * df.beta_f * d_loc * t
        }
    }

    /// Limiter activations recorded so far, and whether every column was
    /// recorded (see [`FaultModel::limiter_stats`] for a complete count).
    pub fn limiter_stats(&self) -> LimiterStatsPartial {
        let mut st = LimiterStatsPartial { complete: true, ..Default::default() };
        for c in 0..self.limiter.cols.len() {
            match self.limiter.get(c) {
                Some(r) => {
                    st.samples += r as u64;
                    st.columns += (r > 0) as u64;
                }
                None => st.complete = false,
            }
        }
        st
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
            "fault drag: beta_hw={:.3} beta_fw={:.3} width={} cap={} m, W_m=[{}], sense=[{}], pad={} samples, B={:.3}{}{}",
            self.beta_h,
            self.beta_f,
            self.params.width,
            self.params.max_width_m,
            widths.join(","),
            senses.join(","),
            self.pad,
            self.b,
            if self.params.no_limiter {
                String::new()
            } else {
                let st = self.limiter_stats();
                format!(
                    ", numerical slope limiter (not physics, min read slope {LIMITER_MIN_SLOPE} over +-{LIMITER_HALF} samples): {} samples raised in {} columns{}",
                    st.samples,
                    st.columns,
                    if st.complete { "" } else { " (columns gathered so far)" }
                )
            },
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
                t,
                beta_h: 0.0,
                beta_f: 0.0,
                b: 0.0,
            });
        }
        let (mut bh, mut bf) = (p.beta_h, p.beta_f);
        if faults.iter().any(|f| f.s < 0.0) && bh + bf > 0.9 {
            let k = 0.9 / (bh + bf);
            bh *= k;
            bf *= k;
            warnings.push(format!("normal-sense beta_hw + beta_fw capped at 0.9 (x{k:.3})"));
        }
        // Spec §3 step 1: per-fault clamp B_f = β T_f (π/2) cos θ_f dz / W_f ≤ 0.16.
        let mut clamps = Vec::new();
        let (mut b, mut b_requested) = (0.0f64, 0.0f64);
        for (n, f) in faults.iter_mut().enumerate() {
            let unit = if f.w_m > 0.0 {
                f.t * std::f64::consts::FRAC_PI_2 * f.theta_deg.to_radians().cos() * dz_m / f.w_m
            } else {
                0.0
            };
            let lim = if unit > 0.0 { DRAG_B_TARGET / unit } else { f64::INFINITY };
            f.beta_h = bh.min(lim);
            f.beta_f = bf.min(lim);
            f.b = f.beta_h.max(f.beta_f) * unit;
            b = b.max(f.b);
            b_requested = b_requested.max(bh.max(bf) * unit);
            if f.beta_h < bh || f.beta_f < bf {
                let c = DragClamp {
                    fault: n,
                    t: f.t,
                    w_m: f.w_m,
                    theta_deg: f.theta_deg,
                    before: (bh, bf),
                    after: (f.beta_h, f.beta_f),
                };
                warnings.push(format!(
                    "fault {n}: B_f={:.3} > {DRAG_B_TARGET} (T={:.2} samples, W={:.0} m, theta={:.1} deg): beta_hw {:.4}->{:.4}, beta_fw {:.4}->{:.4}",
                    bh.max(bf) * unit, c.t, c.w_m, c.theta_deg, bh, f.beta_h, bf, f.beta_f
                ));
                clamps.push(c);
            }
        }
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
            clamps,
            bin_m,
            dz_m,
            warnings,
            limiter: std::sync::Arc::new(LimiterLog::new(self.shape()[0] * self.shape()[1])),
        };
        self.set_drag(Some(std::sync::Arc::new(m)))
    }
}

impl FaultModel {
    /// Limiter activations of the whole run (columns not gathered yet are
    /// gathered now). `None` without drag.
    pub fn limiter_stats(&self) -> Option<LimiterStats> {
        let dm = self.drag()?;
        let [ni, nj, nk] = self.shape();
        let mut st = LimiterStats::default();
        let mut src = Vec::new();
        let mut id = Vec::new();
        for c in 0..ni * nj {
            let r = match dm.limiter.get(c) {
                Some(r) => r,
                None => {
                    src.clear();
                    id.clear();
                    self.gather_column(c / nj, c % nj, nk, &mut src, &mut id)
                }
            };
            st.samples += r as u64;
            st.columns += (r > 0) as u64;
        }
        Some(st)
    }

    /// Gather column `(i, j)` (appended to `src`, `id`), with the numerical
    /// slope limiter when drag is on. Returns the samples it raised.
    pub(crate) fn gather_column(
        &self,
        i: usize,
        j: usize,
        nk: usize,
        src: &mut Vec<[f64; 3]>,
        id: &mut Vec<u8>,
    ) -> u32 {
        let o = src.len();
        for k in 0..nk {
            let g = self.gather(i, j, k);
            src.push(g.src);
            id.push(g.id);
        }
        let Some(dm) = self.drag() else { return 0 };
        if dm.params.no_limiter {
            return 0;
        }
        let FaultKinematics::Heave(p) = self.kinematics() else {
            return 0;
        };
        let mut on: Vec<f64> = src[o..].iter().map(|s| s[2]).collect();
        let top_src = (nk.max(1) - 1) as f64 + dm.pad as f64;
        let raised = limit_column(&mut on, &id[o..], top_src, || {
            (0..nk).map(|k| self.gather_plain(&p, i, j, k, false).src[2]).collect()
        });
        if raised > 0 {
            for (s, z) in src[o..].iter_mut().zip(&on) {
                s[2] = *z;
            }
        }
        dm.limiter.store(i * self.shape()[1] + j, raised);
        raised
    }
}
