//! Fault kinematics (spec-fault-kinematics, PR 7a): heave.
//!
//! Legacy faults move the hanging wall straight down (a per-column `k`
//! lookup). With [`FaultKinematics::Heave`] the hanging wall slides down-dip
//! along the fault surface instead, so it also moves sideways:
//!
//! ```text
//! n̂   = ∇f / |∇f|                    (legacy ellipsoid, voxel units)
//! ŝ_d = normalize(ẑ − (ẑ·n̂) n̂)       (down-dip, ẑ = +k)
//! u   = d_legacy / ŝ_d,z · ŝ_d       (vertical part = legacy throw)
//! |u_h| = d / max(tan δ_v, tan δ_min,v)   bins
//! ```
//!
//! with `tan δ_min,v = tan δ_min · bin / dz` (the dip floor). Output voxels
//! gather their source youngest fault first: `x_{n−1} = x_n − u_n(x_n)`.
//! The vertical part reproduces the legacy `k` composition (fault `n`'s
//! `adj_n` sampled at integer `k` and interpolated linearly in `k`); the
//! lateral position is continuous. The fault label is the closed-form band
//! `|f_n − 1| / |∇f_n| ≤ 1.0` voxel ([`LABEL_BAND_VOXELS`]) AND throw > 1 sample at the stage
//! position, last writer (youngest) wins.

use super::apply::FaultModel;
use super::model::ResolvedFault;

/// Half width of the analytic fault-label band (voxels, spec §6.4): the
/// measured optimum against the legacy sobel label with heave forced to 0
/// (seeds 1–40 at 64×64×96: Jaccard mean 0.858 / min 0.735 with 2 faults;
/// 0.75 gives 0.83 / 0.69, 1.25 gives 0.71 / 0.59).
pub const LABEL_BAND_VOXELS: f64 = 1.0;

/// Default physical dip floor (degrees, spec §5 `--fault-min-dip`).
pub const DEFAULT_MIN_DIP_DEG: f64 = 30.0;

/// Heave settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeaveParams {
    /// Physical dip floor δ_min in degrees (15–60).
    pub min_dip_deg: f64,
    /// Depth step per sample (metres).
    pub dz_m: f64,
    /// Lateral bin size (metres).
    pub bin_m: f64,
    /// Multiplier on the heave (1 = physical). Tests use 0 (throw only)
    /// and −1 (sign mutant); the CLI never changes it.
    pub heave_scale: f64,
}

impl Default for HeaveParams {
    fn default() -> Self {
        Self {
            min_dip_deg: DEFAULT_MIN_DIP_DEG,
            dz_m: 4.0,
            bin_m: 12.5,
            heave_scale: 1.0,
        }
    }
}

impl HeaveParams {
    /// Voxel-space tangent of the dip floor, `tan δ_min · bin / dz`.
    pub fn tan_min_v(&self) -> f64 {
        self.min_dip_deg.to_radians().tan() * self.bin_m / self.dz_m
    }

    /// Halo in bins for a maximum throw `t_max` (samples):
    /// `ceil(t_max / tan δ_min,v) + 2` (spec §11; 19 at the defaults).
    pub fn halo(&self, t_max: f64) -> usize {
        (t_max / self.tan_min_v()).ceil() as usize + 2
    }
}

/// How faults displace the rock.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FaultKinematics {
    /// Vertical-only legacy lookup (master ce905d9e, bit for bit).
    #[default]
    Legacy,
    /// Down-dip slip with heave (7a).
    Heave(HeaveParams),
}

/// Displacement of fault `f` at `(x, y, kk)` (integer sample `kk` of the
/// fault's frame): `(a, [hx, hy], d)` with `a = clip(kk − d)` the source
/// sample and `h` the heave in bins (0 outside the hanging wall).
#[inline]
pub fn stage_slip(
    f: &ResolvedFault,
    p: &HeaveParams,
    x: f64,
    y: f64,
    kk: usize,
    lat: f64,
    top: f64,
) -> (f64, [f64; 2], f64) {
    stage_slip_pre(f, p, x, y, kk, lat, top, None)
}

/// [`stage_slip`] with the surface value and gradient at `(x, y, kk)`
/// already known (`pre`, from the label-band check at the same point).
#[inline]
#[allow(clippy::too_many_arguments)]
fn stage_slip_pre(
    f: &ResolvedFault,
    p: &HeaveParams,
    x: f64,
    y: f64,
    kk: usize,
    lat: f64,
    top: f64,
    pre: Option<(f64, [f64; 3])>,
) -> (f64, [f64; 2], f64) {
    let z = kk as f64;
    let v = match pre {
        Some((v, _)) => v,
        None => f.geometry.value_at(x, y, z),
    };
    let w = match &f.tip {
        Some(t) => t.factor(x, y, z),
        None => lat,
    };
    let d = if v <= 1.0 { w * f.profile[kk] } else { 0.0 };
    let a = (z - d).clamp(0.0, top);
    if d == 0.0 || p.heave_scale == 0.0 {
        return (a, [0.0, 0.0], d);
    }
    let g = match pre {
        Some((_, g)) => g,
        None => f.geometry.gradient_at(x, y, z),
    };
    (a, heave_from_gradient(p, g, d), d)
}

/// Heave `u_h` (bins) for throw `d` at `(x, y, z)`: down-dip, magnitude
/// `d / max(tan δ_v, tan δ_min,v)`; zero on a horizontal or vertical
/// surface patch (no down-dip azimuth / no heave).
#[inline]
pub fn heave_vector(
    f: &ResolvedFault,
    p: &HeaveParams,
    x: f64,
    y: f64,
    z: f64,
    d: f64,
) -> [f64; 2] {
    heave_from_gradient(p, f.geometry.gradient_at(x, y, z), d)
}

/// [`heave_vector`] from the surface gradient `g` at the point.
#[inline]
fn heave_from_gradient(p: &HeaveParams, g: [f64; 3], d: f64) -> [f64; 2] {
    let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if !(gn > 0.0) {
        return [0.0, 0.0];
    }
    let (nx, ny, nz) = (g[0] / gn, g[1] / gn, g[2] / gn);
    let nh = (nx * nx + ny * ny).sqrt();
    if nh < 1e-12 || nz == 0.0 {
        return [0.0, 0.0];
    }
    // ŝ_d,h ∝ −n_z n̂_h (unit azimuth −sign(n_z) n̂_h/|n̂_h|).
    let tan_v = nh / nz.abs();
    let mag = p.heave_scale * d / tan_v.max(p.tan_min_v());
    let s = -nz.signum() / nh;
    [mag * s * nx, mag * s * ny]
}

/// Linear interpolation of the profile at a continuous sample.
#[inline]
fn profile_at(f: &ResolvedFault, z: f64, top: f64) -> f64 {
    let zc = z.clamp(0.0, top);
    let k0 = zc.floor() as usize;
    let k1 = (k0 + 1).min(f.profile.len() - 1);
    let w = zc - k0 as f64;
    if w == 0.0 {
        f.profile[k0]
    } else {
        f.profile[k0] + w * (f.profile[k1] - f.profile[k0])
    }
}

/// Source position and fault id of one output voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gathered {
    /// Source `(i, j, k)` (continuous, voxel units, clamped to the cube).
    pub src: [f64; 3],
    /// 1-based youngest fault whose label band holds at its stage, else 0.
    pub id: u8,
}

impl FaultModel {
    /// Gather the source of output voxel `(i, j, k)` youngest fault first
    /// (spec §6.1). Under [`FaultKinematics::Legacy`] the lateral position
    /// never moves (heave 0) and the label band is still reported.
    pub fn gather(&self, i: usize, j: usize, k: usize) -> Gathered {
        let p = match self.kinematics() {
            FaultKinematics::Heave(p) => p,
            FaultKinematics::Legacy => HeaveParams {
                heave_scale: 0.0,
                ..HeaveParams::default()
            },
        };
        self.gather_with(&p, i, j, k, false)
    }

    /// [`Self::gather`] with explicit parameters. `oldest_first` is a
    /// mutation hook for the order test (never used by the pipeline).
    pub fn gather_with(
        &self,
        p: &HeaveParams,
        i: usize,
        j: usize,
        k: usize,
        oldest_first: bool,
    ) -> Gathered {
        let [ni, nj, nk] = self.shape();
        let top = (nk.max(1) - 1) as f64;
        let (imax, jmax) = ((ni.max(1) - 1) as f64, (nj.max(1) - 1) as f64);
        let (mut x, mut y, mut z) = (i as f64, j as f64, k as f64);
        let mut id = 0u8;
        let nf = self.faults().len();
        for step in 0..nf {
            let n = if oldest_first { step } else { nf - 1 - step };
            let f = &self.faults()[n];
            // 7b: the tip taper (evaluated per sample) replaces the
            // Gaussian lateral weight.
            let lat = if f.tip.is_some() {
                0.0
            } else {
                f.lateral_at(x, y)
            };
            let wz = match &f.tip {
                Some(t) => t.factor(x, y, z),
                None => lat,
            };
            let mut pre = None;
            if id == 0 && wz * profile_at(f, z, top) > 1.0 {
                let g = f.geometry.gradient_at(x, y, z);
                let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let v = f.geometry.value_at(x, y, z);
                if gn > 0.0 && (v - 1.0).abs() / gn <= LABEL_BAND_VOXELS {
                    id = (n + 1).min(254) as u8;
                }
                pre = Some((v, g));
            }
            let zc = z.clamp(0.0, top);
            let k0 = zc.floor() as usize;
            let w = zc - k0 as f64;
            // The band check evaluated the surface at (x, y, z); reuse it
            // when that is the stage's sample k0 (same operations, same bits).
            let pre = pre.filter(|_| w == 0.0 && z == k0 as f64);
            let (a0, h0, _) = stage_slip_pre(f, p, x, y, k0, lat, top, pre);
            let (a, h) = if w == 0.0 {
                (a0, h0)
            } else {
                let (a1, h1, _) = stage_slip(f, p, x, y, (k0 + 1).min(nk - 1), lat, top);
                (
                    a0 + w * (a1 - a0),
                    [h0[0] + w * (h1[0] - h0[0]), h0[1] + w * (h1[1] - h0[1])],
                )
            };
            // Legacy stores the composed lookup as f32 at every stage.
            z = a as f32 as f64;
            x = (x - h[0]).clamp(0.0, imax);
            y = (y - h[1]).clamp(0.0, jmax);
        }
        Gathered { src: [x, y, z], id }
    }
}

/// Heave at peak throw (spec-fault-kinematics §2, §6.4): for every surface
/// voxel of a fault (`|f − 1| / |∇f| ≤ LABEL_BAND_VOXELS`), the heave
/// `T / max(tan δ_v, tan δ_min,v)` in bins that the fault's peak throw `T`
/// would give at that voxel's dip. `T` is the largest throw (samples) the
/// kernel applies on the fault's surface voxels; faults whose surface never
/// moves more than one sample are left out.
pub fn heave_at_peak_throw(m: &FaultModel, p: &HeaveParams) -> Vec<f64> {
    let [ni, nj, nk] = m.shape();
    let top = (nk - 1) as f64;
    let mut out = Vec::new();
    for f in m.faults() {
        let (mut t, mut tans) = (0.0f64, Vec::new());
        for i in 0..ni {
            for j in 0..nj {
                let lat = f.lateral(i, j);
                for k in 0..nk {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let g = f.geometry.gradient_at(x, y, z);
                    let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                    if !(gn > 0.0)
                        || (f.geometry.value_at(x, y, z) - 1.0).abs() / gn
                            > super::LABEL_BAND_VOXELS
                    {
                        continue;
                    }
                    let (_, _, d) = stage_slip(f, p, x, y, k, lat, top);
                    t = t.max(d);
                    if g[2] != 0.0 {
                        tans.push(g[0].hypot(g[1]) / g[2].abs());
                    }
                }
            }
        }
        if t > 1.0 {
            out.extend(tans.into_iter().map(|tv| t / tv.max(p.tan_min_v())));
        }
    }
    out
}
