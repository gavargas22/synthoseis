//! Analytic folds of the layered toy geometry (spec "analytic folds",
//! roadmap item 5).
//!
//! A shift field `F(x, y)` (metres, positive down) is the sum of
//! cylindrical (sinusoidal, optionally plunging and periclinal) folds and
//! elliptical Gaussian domes / basins. It is applied once, to the f64
//! horizon stack at the end of `toy_geometry::layered_stack`, before
//! rounding, so salt, faults, closures, partial voxels and burial depth all
//! see folded horizons. The existing dome of the layered geometry stays as
//! it is ("fold 0").
//!
//! * **Similar** style (default, vertical shear): `z_h = z_h⁰ + τ_h F_h / Δz`,
//!   with growth taper `τ_h = (burial_h / burial_base)^q`, `τ_seabed = 0`,
//!   and an axial-surface lean that shifts the profile by `d_h cot δ`.
//!   Exactly reversible ([`unfold`]).
//! * **Parallel** style (flexural slip): the fold field is taken as a
//!   reference surface at the base and each horizon sits at its fixed true
//!   (normal) distance `t_h` above it; the foot point is found by a
//!   fixed-count Newton iteration. Bounded by `t_h κ_max ≤ 0.8`.
//!
//! Draws use their own stream ([`STREAM_FOLD`]), so every existing geometry
//! and rock-physics draw is unchanged. Everything is a pure function of the
//! full `(ni, nj, nk)` grid and the seed, so the result does not depend on
//! tiling or process count. References: Fossen (2016) *Structural Geology*
//! 2nd ed., doi:10.1017/9781107415096; Wu et al. (2019) Geophysics 84(3)
//! IM35, doi:10.1190/geo2018-0646.1; Wu et al. (2020) Geophysics 85(4)
//! WA27, doi:10.1190/geo2019-0375.1.

use std::f64::consts::{PI, TAU};

use crate::rock_physics::keyed_unit;

/// Keyed draw stream of the folds (independent of `STREAM_GEOM` 0x6E0 and
/// the rock-physics streams 1-4).
pub const STREAM_FOLD: u64 = 0xF01D;
/// Minimum cylindrical wavelength in columns (spec §4.3).
pub const MIN_WAVELENGTH_COLS: f64 = 8.0;
/// Minimum dome radius in columns (spec §5).
pub const MIN_DOME_RADIUS_COLS: f64 = 4.0;
/// Parallel style: `max_h t_h κ_max` limit (spec §4.3; fixed, not a flag).
pub const PARALLEL_CURVATURE_LIMIT: f64 = 0.8;
/// Newton iterations of the parallel-style foot-point solve (fixed count,
/// so the result is bit-reproducible).
pub const PARALLEL_NEWTON_ITERS: usize = 8;
/// Fixed bisection steps of the shared amplitude factor (§4.3 dip limit).
pub const LIMIT_BISECTION_ITERS: usize = 48;

/// Fold style (spec §3.4 / §3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FoldStyle {
    /// Vertical shear (default).
    #[default]
    Similar,
    /// Flexural slip: true thickness preserved.
    Parallel,
}

impl FoldStyle {
    pub fn as_str(&self) -> &'static str {
        match self {
            FoldStyle::Similar => "similar",
            FoldStyle::Parallel => "parallel",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "similar" => Ok(FoldStyle::Similar),
            "parallel" => Ok(FoldStyle::Parallel),
            o => Err(format!(
                "--fold-style must be similar or parallel, got {o:?}"
            )),
        }
    }
}

/// Fold parameters of one run (spec §5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoldConfig {
    pub style: FoldStyle,
    /// Cylindrical folds per cube (0-4).
    pub cylindrical: usize,
    /// Domes / basins per cube (0-6).
    pub domes: usize,
    /// Bin size Δx = Δy in metres (5-50, D4).
    pub bin_m: f64,
    /// Depth step Δz in metres per sample of the geometry.
    pub dz_m: f64,
    /// Growth exponent q (0-3, D3).
    pub growth: f64,
    /// Maximum dip of the added folds in samples per column (0.1-2, D7;
    /// default 2), checked against the measured slope on the grid.
    pub max_dip: f64,
}

impl Default for FoldConfig {
    fn default() -> Self {
        Self {
            style: FoldStyle::Similar,
            cylindrical: 1,
            domes: 1,
            bin_m: 12.5,
            dz_m: crate::toy_geometry::GEOMETRY_DIGI_M,
            growth: 1.0,
            max_dip: 2.0,
        }
    }
}

impl FoldConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.cylindrical > 4 {
            return Err(format!(
                "--folds-cylindrical must be in 0-4, got {}",
                self.cylindrical
            ));
        }
        if self.domes > 6 {
            return Err(format!("--folds-domes must be in 0-6, got {}", self.domes));
        }
        if !(5.0..=50.0).contains(&self.bin_m) {
            return Err(format!("--bin-m must be in 5-50, got {}", self.bin_m));
        }
        if !(0.0..=3.0).contains(&self.growth) {
            return Err(format!("--fold-growth must be in 0-3, got {}", self.growth));
        }
        if !(0.1..=2.0).contains(&self.max_dip) {
            return Err(format!(
                "--fold-max-dip must be in 0.1-2, got {}",
                self.max_dip
            ));
        }
        if !(self.dz_m.is_finite() && self.dz_m > 0.0) {
            return Err(format!("fold dz must be > 0, got {}", self.dz_m));
        }
        Ok(())
    }
}

/// One drawn fold, lengths in metres, angles in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fold {
    Cylindrical {
        /// Reference point (cube centre), metres.
        x0: [f64; 2],
        theta: f64,
        wavelength: f64,
        amp: f64,
        phase: f64,
        /// tan(plunge), signed.
        tan_plunge: f64,
        /// Periclinal length; `f64::INFINITY` = pure cylinder.
        pericline: f64,
        /// cot(axial-surface dip), signed (lean sense).
        cot_dip: f64,
    },
    Dome {
        x0: [f64; 2],
        theta: f64,
        /// −1 dome (crest up), +1 basin.
        sign: f64,
        amp: f64,
        ra: f64,
        rb: f64,
        cot_dip: f64,
    },
}

impl Fold {
    /// Lean direction (unit vector, azimuth θ + 90°).
    fn lean_dir(theta: f64) -> [f64; 2] {
        [-theta.sin(), theta.cos()]
    }

    /// `(F, ∂F/∂x, ∂F/∂y)` at `(x, y)` for a horizon of mean burial `d`
    /// metres (axial-surface lean).
    pub fn eval(&self, x: f64, y: f64, d: f64) -> (f64, f64, f64) {
        match *self {
            Fold::Cylindrical {
                x0,
                theta,
                wavelength,
                amp,
                phase,
                tan_plunge,
                pericline,
                cot_dip,
            } => {
                let (c, s) = (theta.cos(), theta.sin());
                let (dx, dy) = (x - x0[0], y - x0[1]);
                let sc = dx * c + dy * s - d * cot_dip;
                let a = -dx * s + dy * c;
                let k = TAU / wavelength;
                let (sn, cs) = (k * sc + phase).sin_cos();
                let (e, de) = if pericline.is_finite() {
                    let e = (-a * a / (2.0 * pericline * pericline)).exp();
                    (e, -a / (pericline * pericline) * e)
                } else {
                    (1.0, 0.0)
                };
                let f = amp * sn * e + a * tan_plunge;
                let df_ds = amp * k * cs * e;
                let df_da = amp * sn * de + tan_plunge;
                // s = (dx c + dy s), a = (−dx s + dy c).
                (f, df_ds * c - df_da * s, df_ds * s + df_da * c)
            }
            Fold::Dome {
                x0,
                theta,
                sign,
                amp,
                ra,
                rb,
                cot_dip,
            } => {
                let l = Self::lean_dir(theta);
                let (cx, cy) = (x0[0] + d * cot_dip * l[0], x0[1] + d * cot_dip * l[1]);
                let (c, s) = (theta.cos(), theta.sin());
                let (dx, dy) = (x - cx, y - cy);
                let u = dx * c + dy * s;
                let v = -dx * s + dy * c;
                let g = (-(u * u / (2.0 * ra * ra) + v * v / (2.0 * rb * rb))).exp();
                let f = sign * amp * g;
                let df_du = -f * u / (ra * ra);
                let df_dv = -f * v / (rb * rb);
                (f, df_du * c - df_dv * s, df_du * s + df_dv * c)
            }
        }
    }

    /// Gradient `(∂F/∂x, ∂F/∂y)` split into the amplitude-proportional part
    /// and the plunge part (independent of amplitude).
    fn grad_split(&self, x: f64, y: f64, d: f64) -> ([f64; 2], [f64; 2]) {
        let (_, gx, gy) = self.eval(x, y, d);
        match *self {
            Fold::Cylindrical {
                theta, tan_plunge, ..
            } => {
                let (c, s) = (theta.cos(), theta.sin());
                let p = [-s * tan_plunge, c * tan_plunge];
                ([gx - p[0], gy - p[1]], p)
            }
            Fold::Dome { .. } => ([gx, gy], [0.0, 0.0]),
        }
    }

    /// Bound of the amplitude-proportional slope (m/m) per metre of
    /// amplitude, and the amplitude-independent slope (plunge). Kept for
    /// the old summed-bound diagnostic ([`FoldSet::dip_bound`]).
    fn slope_terms(&self) -> (f64, f64) {
        match *self {
            Fold::Cylindrical {
                wavelength,
                pericline,
                tan_plunge,
                ..
            } => {
                let k = TAU / wavelength;
                let pe = if pericline.is_finite() {
                    (-0.5f64).exp() / pericline
                } else {
                    0.0
                };
                ((k * k + pe * pe).sqrt(), tan_plunge.abs())
            }
            Fold::Dome { rb, .. } => ((-0.5f64).exp() / rb, 0.0),
        }
    }

    /// Bound of the curvature (1/m) per metre of amplitude.
    fn curvature_term(&self) -> f64 {
        match *self {
            Fold::Cylindrical {
                wavelength,
                pericline,
                ..
            } => {
                let k = TAU / wavelength;
                let p = if pericline.is_finite() {
                    1.0 / pericline
                } else {
                    0.0
                };
                (k + p) * (k + p)
            }
            Fold::Dome { rb, .. } => 1.0 / (rb * rb),
        }
    }

    pub fn amp(&self) -> f64 {
        match *self {
            Fold::Cylindrical { amp, .. } | Fold::Dome { amp, .. } => amp,
        }
    }

    fn scale_amp(&mut self, f: f64) {
        match self {
            Fold::Cylindrical { amp, .. } | Fold::Dome { amp, .. } => *amp *= f,
        }
    }
}

/// The drawn folds of one cube.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldSet {
    pub folds: Vec<Fold>,
    /// Folds whose amplitude was scaled down to a §4.3 limit.
    pub scaled: usize,
    /// The one factor all amplitudes were scaled by (1 = unscaled).
    pub factor: f64,
    /// Measured largest |slope| of the added folds on the grid, samples per
    /// column, before and after the limit (see [`FoldSet::measured_dip`]).
    pub dip_before: f64,
    pub dip_after: f64,
    pub cfg: FoldConfig,
}

fn u(seed: u64, k: u64, p: u64) -> f64 {
    keyed_unit(&[seed, STREAM_FOLD, k, p])
}

fn uni(lo: f64, hi: f64, seed: u64, k: u64, p: u64) -> f64 {
    lo + (hi - lo) * u(seed, k, p)
}

impl FoldSet {
    /// Draw the folds of a cube (spec §5) and apply the §4.3 limits.
    /// `tp` is the [`taper`] of the unfolded stack (burial and growth taper
    /// per horizon, deepest first); `tp[0].0` is the folded column (base to
    /// seabed, metres) used by the parallel-style curvature limit.
    pub fn draw(seed: u64, shape: [usize; 3], cfg: &FoldConfig, tp: &[(f64, f64)]) -> Self {
        let [ni, nj, nk] = shape;
        let bin = cfg.bin_m;
        let nmax = ni.max(nj).max(1) as f64;
        let nmin = ni.min(nj).max(1) as f64;
        let depth_m = nk as f64 * cfg.dz_m;
        let centre = [
            0.5 * (ni as f64 - 1.0).max(0.0) * bin,
            0.5 * (nj as f64 - 1.0).max(0.0) * bin,
        ];
        let lean = |k: u64| {
            let dip = uni(70.0, 90.0, seed, k, 7).to_radians();
            let sense = if u(seed, k, 8) < 0.5 { -1.0 } else { 1.0 };
            sense * dip.cos() / dip.sin()
        };
        let mut folds = Vec::new();
        for f in 0..cfg.cylindrical as u64 {
            let k = f;
            let wl = (uni(0.5, 1.5, seed, k, 0) * nmax).max(MIN_WAVELENGTH_COLS);
            let per = if u(seed, k, 5) < 0.5 {
                f64::INFINITY
            } else {
                (uni(0.5, 1.5, seed, k, 6) * nmax).max(0.5 * wl) * bin
            };
            let plunge = uni(0.0, 3.0, seed, k, 4).to_radians();
            let sense = if u(seed, k, 9) < 0.5 { -1.0 } else { 1.0 };
            folds.push(Fold::Cylindrical {
                x0: centre,
                theta: uni(0.0, PI, seed, k, 1),
                wavelength: wl * bin,
                amp: uni(0.02, 0.06, seed, k, 2) * depth_m,
                phase: uni(0.0, TAU, seed, k, 3),
                tan_plunge: sense * plunge.tan(),
                pericline: per,
                cot_dip: lean(k),
            });
        }
        for f in 0..cfg.domes as u64 {
            let k = 100 + f;
            let ra = (uni(0.10, 0.25, seed, k, 0) * nmin).max(MIN_DOME_RADIUS_COLS);
            let rb = (ra / uni(1.0, 3.0, seed, k, 1)).max(MIN_DOME_RADIUS_COLS);
            folds.push(Fold::Dome {
                x0: [
                    uni(0.15, 0.85, seed, k, 2) * ni as f64 * bin,
                    uni(0.15, 0.85, seed, k, 3) * nj as f64 * bin,
                ],
                theta: uni(0.0, PI, seed, k, 4),
                sign: if u(seed, k, 5) < 0.6 { -1.0 } else { 1.0 },
                amp: uni(0.03, 0.08, seed, k, 6) * depth_m,
                ra: ra * bin,
                rb: rb * bin,
                cot_dip: lean(k),
            });
        }
        let mut set = FoldSet {
            folds,
            scaled: 0,
            factor: 1.0,
            dip_before: 0.0,
            dip_after: 0.0,
            cfg: *cfg,
        };
        set.apply_limits(shape, tp);
        set
    }

    /// Old summed per-fold bound of the dip (samples per column, τ = 1):
    /// Σ peak slopes. Conservative (the peaks rarely coincide); kept only as
    /// a diagnostic, the limit uses [`FoldSet::measured_dip`].
    pub fn dip_bound(&self) -> f64 {
        let (mut a, mut c) = (0.0, 0.0);
        for f in &self.folds {
            let (g, k) = f.slope_terms();
            a += f.amp() * g;
            c += k;
        }
        (a + c) * self.cfg.bin_m / self.cfg.dz_m
    }

    /// Worst-case curvature (1/m) of the whole field.
    pub fn curvature_bound(&self) -> f64 {
        self.folds
            .iter()
            .map(|f| f.amp() * f.curvature_term())
            .sum()
    }

    /// Slope samples of the added folds on the grid: per point the
    /// amplitude part `A` and plunge part `P` of `τ ∇F` (the slope at
    /// amplitude factor `f` is `|f A + P|`). Similar: every folded horizon
    /// at its burial (lean) and taper; parallel: the reference surface
    /// (`d = 0`, τ = 1), which the offset horizons follow.
    fn slope_samples(&self, shape: [usize; 3], tp: &[(f64, f64)]) -> Vec<[f64; 4]> {
        let [ni, nj, _] = shape;
        let bin = self.cfg.bin_m;
        let levels: Vec<(f64, f64)> = match self.cfg.style {
            FoldStyle::Similar => tp
                .iter()
                .take(tp.len().saturating_sub(1))
                .filter(|(_, t)| *t > 0.0)
                .copied()
                .collect(),
            FoldStyle::Parallel => vec![(0.0, 1.0)],
        };
        let mut out = Vec::with_capacity(levels.len() * ni * nj);
        for &(d, t) in &levels {
            for i in 0..ni {
                for j in 0..nj {
                    let (x, y) = (i as f64 * bin, j as f64 * bin);
                    let mut v = [0.0; 4];
                    for f in &self.folds {
                        let (a, p) = f.grad_split(x, y, d);
                        v[0] += t * a[0];
                        v[1] += t * a[1];
                        v[2] += t * p[0];
                        v[3] += t * p[1];
                    }
                    out.push(v);
                }
            }
        }
        out
    }

    fn max_slope(samples: &[[f64; 4]], f: f64) -> f64 {
        samples
            .iter()
            .map(|v| (f * v[0] + v[2]).hypot(f * v[1] + v[3]))
            .fold(0.0, f64::max)
    }

    /// Measured largest |slope| of the added folds (fold 0, the dome and
    /// tilt of the unfolded stack, is not included), samples per column,
    /// from the analytic gradient on the grid.
    pub fn measured_dip(&self, shape: [usize; 3], tp: &[(f64, f64)]) -> f64 {
        Self::max_slope(&self.slope_samples(shape, tp), 1.0) * self.cfg.bin_m / self.cfg.dz_m
    }

    /// §4.3 (review fix): scale all amplitudes by one factor so the measured
    /// largest |slope| of the summed fields on the grid is at most
    /// `max_dip` samples per column, and (parallel) the curvature bound
    /// `column_m · κ ≤ 0.8`. Plunge is not proportional to amplitude, so
    /// the factor comes from a fixed-count bisection
    /// ([`LIMIT_BISECTION_ITERS`]) of the convex `max |f A + P|`, rounded
    /// down. Scaling keeps every draw.
    ///
    /// The parallel bound uses the full column `tp[0]` (base to seabed):
    /// correct, and conservative for the shallower horizons.
    fn apply_limits(&mut self, shape: [usize; 3], tp: &[(f64, f64)]) {
        let column_m = tp.first().map_or(0.0, |v| v.0);
        let to_samples = self.cfg.bin_m / self.cfg.dz_m;
        let budget = self.cfg.max_dip / to_samples;
        let samples = self.slope_samples(shape, tp);
        let g1 = Self::max_slope(&samples, 1.0);
        self.dip_before = g1 * to_samples;
        let mut factor = 1.0f64;
        if g1 > budget {
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            if Self::max_slope(&samples, 0.0) > budget {
                hi = 0.0;
            }
            for _ in 0..LIMIT_BISECTION_ITERS {
                let mid = 0.5 * (lo + hi);
                if Self::max_slope(&samples, mid) <= budget {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            factor = lo;
        }
        if self.cfg.style == FoldStyle::Parallel && column_m > 0.0 {
            let kappa = self.curvature_bound();
            if kappa * column_m > PARALLEL_CURVATURE_LIMIT {
                factor = factor.min(PARALLEL_CURVATURE_LIMIT / (kappa * column_m));
            }
        }
        if factor < 1.0 {
            for f in &mut self.folds {
                if f.amp() > 0.0 {
                    f.scale_amp(factor);
                    self.scaled += 1;
                }
            }
            self.factor = factor;
        }
        self.dip_after = Self::max_slope(&samples, self.factor) * to_samples;
    }

    /// `(F, ∂F/∂x, ∂F/∂y)` of the total field, metres.
    pub fn field(&self, x: f64, y: f64, d: f64) -> (f64, f64, f64) {
        let (mut f, mut fx, mut fy) = (0.0, 0.0, 0.0);
        for k in &self.folds {
            let (a, b, c) = k.eval(x, y, d);
            f += a;
            fx += b;
            fy += c;
        }
        (f, fx, fy)
    }

    pub fn summary(&self) -> String {
        format!(
            "folds: {} cylindrical + {} dome/basin, style {}{}",
            self.cfg.cylindrical,
            self.cfg.domes,
            self.cfg.style.as_str(),
            if self.scaled > 0 {
                format!(" (all folds scaled by {:.3} to the dip/curvature limit)", self.factor)
            } else {
                String::new()
            }
        )
    }
}

/// Per-horizon `(mean burial d_h in metres, growth taper τ_h)` of an
/// unfolded stack, deepest first (stack index 0 = base, last = seabed).
pub fn taper(stack: &[Vec<f64>], cfg: &FoldConfig) -> Vec<(f64, f64)> {
    let nh = stack.len();
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let zbar: Vec<f64> = stack.iter().map(mean).collect();
    let top = zbar[nh - 1];
    let col = zbar[0] - top;
    (0..nh)
        .map(|s| {
            let d = (zbar[s] - top) * cfg.dz_m;
            let t = if s == nh - 1 || col <= 0.0 {
                0.0
            } else if cfg.growth == 0.0 {
                1.0
            } else {
                ((zbar[s] - top) / col).max(0.0).powf(cfg.growth)
            };
            (d, t)
        })
        .collect()
}

/// Shift (samples) of each stack horizon at every column, deepest first.
pub fn shifts(set: &FoldSet, stack: &[Vec<f64>], ni: usize, nj: usize) -> Vec<Vec<f64>> {
    let cfg = &set.cfg;
    let nh = stack.len();
    let tp = taper(stack, cfg);
    let bin = cfg.bin_m;
    let mut out = vec![vec![0.0f64; ni * nj]; nh];
    match cfg.style {
        FoldStyle::Similar => {
            for s in 0..nh - 1 {
                let (d, t) = tp[s];
                if t == 0.0 {
                    continue;
                }
                for i in 0..ni {
                    for j in 0..nj {
                        let (f, _, _) = set.field(i as f64 * bin, j as f64 * bin, d);
                        out[s][i * nj + j] = t * f / cfg.dz_m;
                    }
                }
            }
        }
        FoldStyle::Parallel => {
            // Reference surface S = F at the base (no lean, τ_base = 1).
            // Horizon at true distance t above S; offset G = S(x') − t n_z(x') + t.
            for s in 0..nh - 1 {
                let th = tp[s].0;
                let tdist = tp[0].0 - th;
                for i in 0..ni {
                    for j in 0..nj {
                        let (x, y) = (i as f64 * bin, j as f64 * bin);
                        out[s][i * nj + j] = parallel_offset(set, x, y, tdist) / cfg.dz_m;
                    }
                }
            }
        }
    }
    out
}

fn normal(set: &FoldSet, x: f64, y: f64) -> (f64, f64, f64) {
    let (_, fx, fy) = set.field(x, y, 0.0);
    let n = (1.0 + fx * fx + fy * fy).sqrt();
    (-fx / n, -fy / n, 1.0 / n)
}

/// Foot point `(x', y')` on the reference surface `S` whose point at true
/// distance `t` above `S` lands on column `(x, y)`. With depth positive
/// down and `n = (−S_x, −S_y, 1)/N` the downward unit normal ([`normal`]),
/// that point is `(x', y', S) − t n`, so the solve is
/// `(x', y') − t n_xy(x', y') = (x, y)`: fixed-count Newton
/// ([`PARALLEL_NEWTON_ITERS`]) with a central-difference Jacobian of the
/// analytic normal (the spec asks for the analytic Hessian; the central
/// difference, step 1e-3 bin, is used instead). Contracts while `t κ < 1` (spec §3.5, §4.3).
pub fn foot_point(set: &FoldSet, x: f64, y: f64, t: f64) -> (f64, f64) {
    let (mut px, mut py) = (x, y);
    let h = 1e-3 * set.cfg.bin_m;
    for _ in 0..PARALLEL_NEWTON_ITERS {
        let (nx, ny, _) = normal(set, px, py);
        let rx = px - t * nx - x;
        let ry = py - t * ny - y;
        let (nxp, nyp, _) = normal(set, px + h, py);
        let (nxm, nym, _) = normal(set, px - h, py);
        let (nxq, nyq, _) = normal(set, px, py + h);
        let (nxr, nyr, _) = normal(set, px, py - h);
        let j11 = 1.0 - t * (nxp - nxm) / (2.0 * h);
        let j12 = -t * (nxq - nxr) / (2.0 * h);
        let j21 = -t * (nyp - nym) / (2.0 * h);
        let j22 = 1.0 - t * (nyq - nyr) / (2.0 * h);
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-12 {
            break;
        }
        px -= (j22 * rx - j12 * ry) / det;
        py -= (-j21 * rx + j11 * ry) / det;
    }
    (px, py)
}

/// Parallel-style depth offset (metres) at column `(x, y)` of a horizon at
/// true distance `t` above the folded reference: `S(x') − t n_z(x') + t`,
/// which is 0 where the field is flat.
pub fn parallel_offset(set: &FoldSet, x: f64, y: f64, t: f64) -> f64 {
    let (px, py) = foot_point(set, x, y, t);
    let (s, _, _) = set.field(px, py, 0.0);
    let (_, _, nz) = normal(set, px, py);
    s - t * nz + t
}

/// Apply the folds to the f64 stack (deepest first) in place, then clamp
/// the base below the cube (`z_base ≥ nk + 1`) and every horizon at or
/// below the seabed. Returns the drawn fold set.
pub fn apply_folds(
    stack: &mut [Vec<f64>],
    seed: u64,
    shape: [usize; 3],
    cfg: &FoldConfig,
) -> FoldSet {
    let [ni, nj, nk] = shape;
    let nh = stack.len();
    let tp = taper(stack, cfg);
    let set = FoldSet::draw(seed, shape, cfg, &tp);
    if nh < 2 || set.folds.is_empty() {
        return set;
    }
    let sh = shifts(&set, stack, ni, nj);
    for s in 0..nh - 1 {
        for (v, d) in stack[s].iter_mut().zip(&sh[s]) {
            *v += *d;
        }
    }
    for v in &mut stack[0] {
        *v = v.max(nk as f64 + 1.0);
    }
    let (lo, hi) = stack.split_at_mut(nh - 1);
    for layer in lo.iter_mut() {
        for (v, sb) in layer.iter_mut().zip(&hi[0]) {
            *v = v.max(*sb);
        }
    }
    set
}

/// Inverse of [`apply_folds`] for tests: subtract the same shifts (the
/// clamps are not undone, so the base and seabed-clamped samples are not
/// restored). `unfolded` is the unfolded stack the taper was taken from.
pub fn unfold(
    folded: &mut [Vec<f64>],
    unfolded_ref: &[Vec<f64>],
    set: &FoldSet,
    ni: usize,
    nj: usize,
) {
    let sh = shifts(set, unfolded_ref, ni, nj);
    for (layer, d) in folded.iter_mut().zip(&sh) {
        for (v, dv) in layer.iter_mut().zip(d) {
            *v -= *dv;
        }
    }
}
