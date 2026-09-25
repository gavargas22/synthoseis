//! Fault geometry (rotated ellipsoid), lateral/vertical throw profiles, and
//! the one-off global resolution step (fault centre + seabed taper).
//!
//! Every per-voxel quantity is an analytic function of the global fault
//! parameters and the voxel position, so tiles can be evaluated independently
//! (see [`super::apply`]).

use super::params::{FaultParams, FaultRng, THROW_LUT_MAX, THROW_LUT_MIN};
use super::segments::{segment_block, SEG_ONE};

/// Rotated-ellipsoid geometry of one fault.
///
/// Port of `Faults.rotate_3d_ellipsoid` + `Faults.apply_3d_rotation`: voxel
/// coordinates are rotated about the *array origin* around the horizontal
/// strike vector by `theta`, then tested against the axis-aligned ellipsoid
/// `(x-x0)²/a + (y-y0)²/b + (z-z0)²/c`. Voxels with value `<= 1` are inside
/// (hanging wall, displaced).
#[derive(Debug, Clone, PartialEq)]
pub struct FaultGeometry {
    rot: [[f64; 3]; 3],
    origin: [f64; 3],
    axes: [f64; 3],
}

impl FaultGeometry {
    pub fn new(shape: [usize; 3], p: &FaultParams) -> Self {
        let (n0, n1, n2) = (shape[0] as f64, shape[1] as f64, shape[2] as f64);
        let dxc = p.x0 - n0 / 2.0;
        let dyc = p.y0 - n1 / 2.0;
        let theta = (p.tilt_pct * (dxc * dxc + dyc * dyc).sqrt()).atan2(n2);
        let dip_angle = dyc.atan2(dxc);
        let pi = std::f64::consts::PI;
        let axis = [(pi - dip_angle).sin(), (pi - dip_angle).cos(), 0.0];
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let k = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
        // Rodrigues == expm(cross(eye(3), k*theta)) for a unit axis.
        let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
        let mut k2 = [[0.0; 3]; 3];
        for (r, row) in k2.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|m| kx[r][m] * kx[m][c]).sum();
            }
        }
        let (s, c1) = (theta.sin(), 1.0 - theta.cos());
        let mut rot = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let id = if r == c { 1.0 } else { 0.0 };
                rot[r][c] = id + s * kx[r][c] + c1 * k2[r][c];
            }
        }
        Self {
            rot,
            origin: [p.x0, p.y0, p.z0],
            axes: [p.a, p.b, p.c],
        }
    }

    /// Ellipsoid function at voxel `(i, j, k)` (`<= 1` inside).
    #[inline]
    pub fn value(&self, i: usize, j: usize, k: usize) -> f64 {
        let (x, y, z) = (i as f64, j as f64, k as f64);
        let r = &self.rot;
        let xr = r[0][0] * x + r[0][1] * y + r[0][2] * z;
        let yr = r[1][0] * x + r[1][1] * y + r[1][2] * z;
        let zr = r[2][0] * x + r[2][1] * y + r[2][2] * z;
        (xr - self.origin[0]).powi(2) / self.axes[0]
            + (yr - self.origin[1]).powi(2) / self.axes[1]
            + (zr - self.origin[2]).powi(2) / self.axes[2]
    }

    #[inline]
    pub fn inside(&self, i: usize, j: usize, k: usize) -> bool {
        self.value(i, j, k) <= 1.0
    }
}

/// Seabed (water-bottom) depth in samples, used to taper the vertical throw
/// profile and to restrict fault-centre candidates to sub-seabed voxels.
#[derive(Debug, Clone, PartialEq)]
pub enum Seabed {
    Flat(f64),
    /// Row-major `(ni, nj)` map.
    Map(Vec<f64>),
}

impl Seabed {
    #[inline]
    pub fn at(&self, i: usize, j: usize, nj: usize) -> f64 {
        match self {
            Seabed::Flat(v) => *v,
            Seabed::Map(m) => m[i * nj + j],
        }
    }
}

/// Why a fault was not inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultSkip {
    /// No fault-surface voxel below the seabed (Python: "Ellipsoid larger than
    /// cube no fault inserted").
    NoSurfaceBelowSeabed,
    /// `get_middle_z` could not converge on a middle point.
    NoMiddlePoint,
}

/// One fault with every global reduction already performed.
#[derive(Debug, Clone)]
pub struct ResolvedFault {
    pub params: FaultParams,
    pub geometry: FaultGeometry,
    pub center: [usize; 3],
    /// Vertical throw profile `z_shift(k)` (already includes `throw`).
    pub profile: Vec<f64>,
    /// Samples the profile was rolled down to keep the seabed unfaulted.
    pub seabed_roll: usize,
    /// Python would add hockey-stick drag (deferred in this port).
    pub hockey_stick_deferred: bool,
    lateral: LateralGaussian,
}

impl ResolvedFault {
    /// Normalised lateral displacement `xy_dis(i, j)` (peak ≈ 1).
    #[inline]
    pub fn lateral(&self, i: usize, j: usize) -> f64 {
        self.lateral.eval(i, j, self.center)
    }

    /// `displacement_classification(i, j, k) = xy_dis(i, j) * z_shift(k)`.
    #[inline]
    pub fn displacement_classification(&self, i: usize, j: usize, k: usize) -> f64 {
        self.lateral(i, j) * self.profile[k]
    }
}

/// Analytic stand-in for Python's rotated/rolled 2-D multivariate normal.
///
/// Python evaluates `multivariate_normal([0,0], diag(vx, vy)).pdf` on a
/// `(2*Ny, 2*Nx)` `linspace` grid (`N = int(2.5 n)`), normalises by the grid
/// max, rotates it with `scipy.ndimage.rotate` (cubic spline, `reshape=False`)
/// by `alpha`, and rolls the rotated argmax onto the fault centre. For these
/// wide gaussians the spline is exact to ~1e-6, so we evaluate the rotated
/// gaussian analytically at the same sample positions, including the
/// half-pixel offset of the rotated argmax.
#[derive(Debug, Clone)]
struct LateralGaussian {
    cos_a: f64,
    sin_a: f64,
    /// Rotated-argmax offset from the big-grid centre `(row, col)`.
    off: [f64; 2],
    dx: f64,
    dy: f64,
    vx: f64,
    vy: f64,
    /// Grid max of the unrotated gaussian (normaliser).
    norm: f64,
}

impl LateralGaussian {
    fn new(
        shape: [usize; 3],
        center: [usize; 3],
        throw: f64,
        coef: f64,
        offset: Option<[f64; 2]>,
    ) -> Self {
        let (n0, n1) = (shape[0] as f64, shape[1] as f64);
        let nx = (n0 + 1.5 * n0).trunc();
        let ny = (n1 + 1.5 * n1).trunc();
        // np.linspace(-N, N, 2N) spacing.
        let dx = 2.0 * nx / (2.0 * nx - 1.0);
        let dy = 2.0 * ny / (2.0 * ny - 1.0);
        let vx = throw_variance(throw);
        let vy = vx * coef;
        // Grid max of the unrotated pdf sits at (±dx/2, ±dy/2).
        let norm = (-0.5 * ((dx / 2.0).powi(2) / vx + (dy / 2.0).powi(2) / vy)).exp();
        let alpha = (center[1] as f64 - n1 / 2.0).atan2(center[0] as f64 - n0 / 2.0);
        let deg = alpha.to_degrees();
        let rad = deg.to_radians();
        let (sin_a, cos_a) = rad.sin_cos();
        let mut lg = Self {
            cos_a,
            sin_a,
            off: [0.0, 0.0],
            dx,
            dy,
            vx,
            vy,
            norm,
        };
        // Rotated argmax: best of the four pixels around the half-integer
        // grid centre, first in row-major order on ties (np.where(...)[0][0]).
        let mut best = f64::NEG_INFINITY;
        for dr in [-0.5, 0.5] {
            for dc in [-0.5, 0.5] {
                let v = lg.rotated(dr, dc);
                if v > best {
                    best = v;
                    lg.off = [dr, dc];
                }
            }
        }
        if let Some(o) = offset {
            lg.off = o;
        }
        lg
    }

    /// Rotated field at `(ur, uc)` pixels from the big-grid centre.
    #[inline]
    fn rotated(&self, ur: f64, uc: f64) -> f64 {
        // scipy.ndimage.rotate: in = R @ (out - c) + c, R = [[c, s], [-s, c]]
        // acting on (row, col).
        let r_in = self.cos_a * ur + self.sin_a * uc;
        let c_in = -self.sin_a * ur + self.cos_a * uc;
        let x = c_in * self.dx;
        let y = r_in * self.dy;
        (-0.5 * (x * x / self.vx + y * y / self.vy)).exp() / self.norm
    }

    #[inline]
    fn eval(&self, i: usize, j: usize, center: [usize; 3]) -> f64 {
        let ur = self.off[0] + i as f64 - center[0] as f64;
        let uc = self.off[1] + j as f64 - center[1] as f64;
        self.rotated(ur, uc)
    }
}

/// Throw → lateral variance lookup from `xyz_dis`:
/// `16000 / L(34) * L(int(throw))`, `L(t) = (0.0013 t)^1.3258`.
///
/// Python raises for `int(throw)` outside `5..=34`; we clamp.
pub fn throw_variance(throw: f64) -> f64 {
    let fl = |t: i64| (0.0013 * t as f64).powf(1.3258);
    let t = (throw.trunc() as i64).clamp(THROW_LUT_MIN, THROW_LUT_MAX);
    let scale = 16000.0 / fl(THROW_LUT_MAX);
    scale * fl(t)
}

/// `scipy.signal.windows.general_gaussian(M, p, sig)` value at index `n`.
#[inline]
fn general_gaussian(n: usize, m: usize, p: f64, sig: f64) -> f64 {
    let x = n as f64 - (m as f64 - 1.0) / 2.0;
    (-0.5 * (x / sig).abs().powf(2.0 * p)).exp()
}

/// Vertical throw profile `z_shift(range(nk))` from `xyz_dis`, including the
/// padded/rolled general gaussian and the seabed taper loop.
///
/// Returns `(profile, seabed_roll_samples)`.
pub fn vertical_profile(
    nk: usize,
    throw: f64,
    sigma: f64,
    p: f64,
    center_k: usize,
    wb_max: f64,
) -> (Vec<f64>, usize) {
    let roll_int = (10.0 * sigma).trunc().max(0.0) as usize;
    let m = nk + 2 * roll_int;
    let sig = sigma as f32 as f64; // np.float32(sig)
    let g = |n: usize| throw * general_gaussian(n, m, p, sig);
    let argmax = if m % 2 == 1 { (m - 1) / 2 } else { m / 2 - 1 };
    let mi = m as i64;
    let mut shift = (center_k + argmax + roll_int) as i64;
    let rolled = |t: i64, shift: i64| g((t - shift).rem_euclid(mi) as usize);
    let probe = (roll_int as f64 + wb_max).trunc() as i64;
    let mut count = 0usize;
    while rolled(probe, shift) > 1.0 {
        shift += 5;
        count += 5;
        if count as f64 > nk as f64 - wb_max {
            break;
        }
    }
    let profile = (0..nk)
        .map(|k| rolled((roll_int + k) as i64, shift))
        .collect();
    (profile, count)
}

/// Port of `get_middle_z`: from sub-seabed fault-surface voxels (C-order),
/// narrow to the middle z, then x, then y with a growing window.
///
/// Returns the final candidate subset (`xyz_xyz`), or `None` if the window
/// outgrew `shape[0]` (Python: "could not find a suitable point").
pub fn middle_candidates(cands: &[[u32; 3]], ni: usize) -> Option<Vec<[u32; 3]>> {
    if cands.is_empty() {
        return None;
    }
    let mid = |v: &[[u32; 3]], ax: usize| -> i64 {
        let lo = v.iter().map(|p| p[ax]).min().unwrap() as i64;
        let hi = v.iter().map(|p| p[ax]).max().unwrap() as i64;
        (lo + hi) / 2
    };
    let mut thr: i64 = 5;
    loop {
        if thr >= ni as i64 {
            return None;
        }
        let zm = mid(cands, 2);
        let xyz_z: Vec<[u32; 3]> = cands
            .iter()
            .copied()
            .filter(|p| (p[2] as i64 - zm).abs() < thr)
            .collect();
        if !xyz_z.is_empty() {
            let xm = mid(&xyz_z, 0);
            let xyz_xz: Vec<[u32; 3]> = xyz_z
                .into_iter()
                .filter(|p| (p[0] as i64 - xm).abs() < thr)
                .collect();
            if !xyz_xz.is_empty() {
                let ym = mid(&xyz_xz, 1);
                let xyz_xyz: Vec<[u32; 3]> = xyz_xz
                    .into_iter()
                    .filter(|p| (p[1] as i64 - ym).abs() < thr)
                    .collect();
                if !xyz_xyz.is_empty() {
                    return Some(xyz_xyz);
                }
            }
        }
        thr += 5;
    }
}

/// Global fault-surface survey for one fault (streamed tile by tile).
///
/// Collects sub-seabed surface voxels (`fault_segments == 1 && k >= wb`) in
/// C order, plus `max(wb)` over columns holding any surface voxel.
pub fn survey_surface(
    shape: [usize; 3],
    geom: &FaultGeometry,
    seabed: &Seabed,
    tile: [usize; 2],
) -> (Vec<[u32; 3]>, Option<f64>) {
    let [ni, nj, nk] = shape;
    let (ti, tj) = (tile[0].max(1), tile[1].max(1));
    let mut cands = Vec::new();
    let mut wb_max: Option<f64> = None;
    let mut seg = Vec::new();
    let mut i0 = 0;
    while i0 < ni {
        let i1 = (i0 + ti).min(ni);
        let mut j0 = 0;
        while j0 < nj {
            let j1 = (j0 + tj).min(nj);
            segment_block(geom, shape, i0, i1, j0, j1, &mut seg);
            let tjw = j1 - j0;
            for i in i0..i1 {
                for j in j0..j1 {
                    let wb = seabed.at(i, j, nj);
                    let base = ((i - i0) * tjw + (j - j0)) * nk;
                    let mut any = false;
                    for k in 0..nk {
                        if seg[base + k] == SEG_ONE {
                            any = true;
                            if wb - k as f64 <= 0.0 {
                                cands.push([i as u32, j as u32, k as u32]);
                            }
                        }
                    }
                    if any {
                        wb_max = Some(wb_max.map_or(wb, |m: f64| m.max(wb)));
                    }
                }
            }
            j0 = j1;
        }
        i0 = i1;
    }
    cands.sort_unstable();
    (cands, wb_max)
}

/// Resolve one fault: centre (explicit or seeded `get_middle_z`), lateral
/// gaussian and vertical profile. `Err` means Python would skip the fault.
pub fn resolve_fault(
    shape: [usize; 3],
    params: &FaultParams,
    seabed: &Seabed,
    rng: &mut FaultRng,
    survey_tile: [usize; 2],
) -> Result<ResolvedFault, FaultSkip> {
    let geometry = FaultGeometry::new(shape, params);
    let (cands, wb_max) = survey_surface(shape, &geometry, seabed, survey_tile);
    if cands.is_empty() {
        return Err(FaultSkip::NoSurfaceBelowSeabed);
    }
    let center = match params.center {
        Some(c) => c,
        None => {
            let sub = middle_candidates(&cands, shape[0]).ok_or(FaultSkip::NoMiddlePoint)?;
            let p = sub[rng.below(sub.len())];
            [p[0] as usize, p[1] as usize, p[2] as usize]
        }
    };
    let wb_max = wb_max.unwrap_or(0.0);
    let (profile, seabed_roll) = vertical_profile(
        shape[2],
        params.throw,
        params.sigma,
        params.p,
        center[2],
        wb_max,
    );
    let lateral = LateralGaussian::new(
        shape,
        center,
        params.throw,
        params.coef,
        params.lateral_offset,
    );
    Ok(ResolvedFault {
        params: params.clone(),
        geometry,
        center,
        profile,
        seabed_roll,
        hockey_stick_deferred: params.is_hockey_stick(),
        lateral,
    })
}
