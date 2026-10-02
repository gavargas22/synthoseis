//! Salt bodies: a port of legacy `datagenerator/Salt.py` (`SaltModel`).
//!
//! Legacy (`include_salt: true` in the shipped `config/example.json`) adds one
//! salt body to every model, after faulting:
//! 1. `compute_salt_body_segmentation`:
//!    * radius `R = triangular(ni/6, ni/5, ni/4)` columns;
//!    * top = 99th percentile of horizon 1 + `U(150, 300)` samples;
//!    * `insertSalt3D`: a "bag of points" (a wide cap of three jittered
//!      circles plus a tip, and a narrow stem of three circles plus a tip
//!      near the base of the padded cube). Salt is every voxel inside the
//!      convex hull of the 218 points (`scipy.spatial.Delaunay.find_simplex`).
//! 2. `update_depth_maps_with_salt_segments_drag`: every horizon whose depth
//!    falls inside salt in some column is shifted up by `2 * r` samples
//!    there (`r` counts the horizons touching salt so far), every horizon is
//!    smoothed with `gaussian_filter(sigma = 3)`, and negative thicknesses are
//!    removed from the base upwards. The horizons drag up against the flanks.
//! 3. Lithology 2 inside salt; rock physics sets `rho = 2.17`, `vp = 4500`,
//!    `vs = 2250` there (before the base forward-fill).
//! 4. Closures: salt makes the horizon gaps that `Closures._flood_fill`
//!    walls off, so traps seal against the salt flank.
//!
//! This module holds the pure functions ([`salt_points`], [`hull_runs`],
//! [`drag_horizon_maps`], [`gaussian_filter_sigma3`], [`numpy_percentile`],
//! [`closure_fill_input`]); each reproduces the legacy code bit for bit on
//! the same inputs and unit draws (see `tests/salt.rs`). [`salt_body`] draws
//! the body for an [`E2eConfig`] from keyed unit draws, so every worker and
//! process rebuilds the same salt, whatever the tiling. Memory: one `[k0, k1)`
//! run per column (8 bytes per column).
//!
//! Deviations from legacy (see `docs/salt-bodies.md`):
//! * The top offset `U(150, 300)` samples is scaled by `min(nk / 1250, 1)`
//!   (legacy `example.json` has 1250 samples), so the salt sits inside small
//!   cubes and is never deeper than legacy in large ones;
//!   [`crate::rock_physics::RockPhysicsConfig::salt_legacy_top_offset`] keeps
//!   the absolute offset.
//! * The drag is applied to the unfaulted toy horizons before faulting (the
//!   Rust faults displace labels, there are no faulted maps), and the
//!   percentile uses the unfaulted horizon 1. Legacy smooths the faulted
//!   maps, which also blurs every fault offset in the horizons.
//! * The salt body itself is not faulted (as in legacy).

use crate::pipeline::E2eConfig;
use crate::rock_physics::keyed_unit;

/// Legacy `pad_samples`: the salt grid is `(ni, nj, nk + pad)`.
pub const SALT_PAD: usize = 10;
/// Number of samples of the legacy example cube (top offset scale).
pub const LEGACY_SAMPLES: f64 = 1250.0;
/// Legacy salt properties (float32 in the output).
pub const SALT_RHO: f32 = 2.17;
pub const SALT_VP: f32 = 4500.0;
pub const SALT_VS: f32 = 2250.0;

/// Keyed draw stream of the salt geometry.
const STREAM_SALT: u64 = 0x5A17;

/// One salt body on the grid `(ni, nj, nk + pad)`: per column the half-open
/// sample run `[k0, k1)` inside the convex hull (empty when `k0 == k1`).
#[derive(Debug, Clone, PartialEq)]
pub struct SaltBody {
    /// `(ni, nj, nk + pad)`.
    pub grid: [usize; 3],
    pub runs: Vec<(u32, u32)>,
    /// Radius (columns), legacy `salt_radius`.
    pub radius: f64,
    /// Top depth (samples), legacy `shallowest_salt`.
    pub top: f64,
    /// Hull points `(i, j, k)`, legacy `SaltModel.points`.
    pub points: Vec<[f64; 3]>,
}

impl SaltBody {
    /// `true` when sample `k` of column `c = i * nj + j` is salt.
    #[inline]
    pub fn contains(&self, c: usize, k: usize) -> bool {
        let (a, b) = self.runs[c];
        (a as usize) <= k && k < b as usize
    }

    /// Salt voxels in the first `nk` samples.
    pub fn voxels(&self, nk: usize) -> usize {
        self.runs
            .iter()
            .map(|&(a, b)| (b as usize).min(nk).saturating_sub(a as usize))
            .sum()
    }

    /// Columns with salt in the first `nk` samples.
    pub fn columns(&self, nk: usize) -> usize {
        self.runs
            .iter()
            .filter(|&&(a, b)| b > a && (a as usize) < nk)
            .count()
    }

    /// Salt mask (0/1) of the first `nk` samples of column `c`.
    pub fn column_mask(&self, c: usize, out: &mut [u8]) {
        out.fill(0);
        let (a, b) = self.runs[c];
        let b = (b as usize).min(out.len());
        let a = (a as usize).min(b);
        out[a..b].fill(1);
    }
}

/// numpy `Generator.uniform(lo, hi)` from the unit draw `u`.
#[inline]
pub fn numpy_uniform(lo: f64, hi: f64, u: f64) -> f64 {
    lo + (hi - lo) * u
}

/// numpy `Generator.triangular(left, mode, right)` from the unit draw `u`
/// (same operation order as numpy's `random_triangular`).
pub fn numpy_triangular(left: f64, mode: f64, right: f64, u: f64) -> f64 {
    let base = right - left;
    let leftbase = mode - left;
    let ratio = leftbase / base;
    let leftprod = leftbase * base;
    let rightprod = (right - mode) * base;
    if u <= ratio {
        left + (u * leftprod).sqrt()
    } else {
        right - ((1.0 - u) * rightprod).sqrt()
    }
}

/// numpy `percentile(values, q)` with the default linear method.
pub fn numpy_percentile(values: &[f64], q_percent: f64) -> f64 {
    assert!(!values.is_empty());
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let q = q_percent / 100.0;
    // numpy 'linear': virtual index `(n - 1) * q`.
    let virt = (n - 1) as f64 * q;
    let prev = virt.floor();
    let gamma = virt - prev;
    let pi = (prev.max(0.0) as usize).min(n - 1);
    let ni = ((prev + 1.0).max(0.0) as usize).min(n - 1);
    let (a, b) = (v[pi], v[ni]);
    let d = b - a;
    if gamma >= 0.5 {
        b - d * (1.0 - gamma)
    } else {
        a + d * gamma
    }
}

/// Legacy `create_circular_pointcloud`: 36 points at 10 degree steps with
/// `x, y` jitter `+-0.1 r` and `z` jitter `+-0.05 r` (draws x, y, z per point).
fn circle(
    cx: f64,
    cy: f64,
    cz: f64,
    r: f64,
    draw: &mut dyn FnMut() -> f64,
    out: &mut Vec<[f64; 3]>,
) {
    for n in 0..36 {
        let a = (n as f64) * 10.0;
        let jit = 0.1 * r;
        let x =
            cx + r * (a * std::f64::consts::PI / 180.0).cos() + numpy_uniform(-jit, jit, draw());
        let y =
            cy + r * (a * std::f64::consts::PI / 180.0).sin() + numpy_uniform(-jit, jit, draw());
        let z = cz + numpy_uniform(-jit / 2.0, jit / 2.0, draw());
        out.push([x, y, z]);
    }
}

/// Legacy `salt_circle_points`: deep, mid and shallow circles and a tip.
fn circle_points(
    c: [f64; 4],
    r: [f64; 3],
    cx: f64,
    cy: f64,
    draw: &mut dyn FnMut() -> f64,
    out: &mut Vec<[f64; 3]>,
) {
    circle(cx, cy, c[0], r[0], draw, out);
    circle(cx, cy, c[1], r[1], draw, out);
    circle(cx, cy, c[2], r[2], draw, out);
    let jit = 0.1 * r[2];
    let x = cx + numpy_uniform(-jit, jit, draw());
    let y = cy + numpy_uniform(-jit, jit, draw());
    let z = c[3] + numpy_uniform(-jit / 2.0, jit / 2.0, draw());
    out.push([x, y, z]);
}

/// Legacy `insertSalt3D` point cloud (218 points) for `top`, `radius` on the
/// grid `(ni, nj, nk + pad)`, in legacy draw order.
///
/// Legacy bug (kept, reported): the `y` centre range uses `cube_shape[0]`
/// (`ni`) instead of `cube_shape[1]`, so non-square cubes put the salt off
/// centre in `j` (or outside the cube).
pub fn salt_points(
    top: f64,
    radius: f64,
    grid: [usize; 3],
    draw: &mut dyn FnMut() -> f64,
) -> Vec<[f64; 3]> {
    let [ni, nj, nkp] = grid;
    let (s0, s1, s2) = (ni as f64, nj as f64, nkp as f64);
    let r = radius;
    let c1_deep = (top + r) * 1.1;
    let c1_mid = c1_deep - r * numpy_uniform(0.35, 0.45, draw());
    let c1_shallow = c1_deep - r * numpy_uniform(0.87, 0.93, draw());
    let c1_tip = c1_deep - r * numpy_uniform(0.98, 1.02, draw());
    let r1_deep = r * numpy_uniform(0.9, 1.1, draw());
    let r1_mid = r1_deep * numpy_uniform(0.37, 0.43, draw());
    let r1_shallow = r1_deep * numpy_uniform(0.18, 0.25, draw());
    let center_x = s0 / 2.0 + numpy_uniform(-s0 * 0.4, s0 * 0.4, draw());
    let center_y = s1 / 2.0 + numpy_uniform(-s0 * 0.4, s0 * 0.4, draw());
    let mut pts = Vec::with_capacity(218);
    circle_points(
        [c1_deep, c1_mid, c1_shallow, c1_tip],
        [r1_deep, r1_mid, r1_shallow],
        center_x,
        center_y,
        draw,
        &mut pts,
    );
    let a = c1_deep + r * numpy_uniform(2.3, 12.5, draw());
    let b = s2 - 2.0 * (r * numpy_uniform(-1.3, 3.5, draw()));
    // Python `max(a, b)` / `min(x, c)` keep the first argument on ties.
    let m = if b > a { b } else { a };
    let cap = s2 + 1.0 * r;
    let c2_deep = if cap < m { cap } else { m };
    let c2_mid = c2_deep + r * numpy_uniform(0.35, 0.45, draw()) / 2.0;
    let c2_shallow = c2_deep + r * numpy_uniform(0.87, 0.93, draw()) / 2.0;
    let c2_tip = c2_deep + r * numpy_uniform(0.98, 1.02, draw()) / 2.0;
    let r2_deep = r1_shallow / 2.0 * numpy_uniform(0.9, 1.1, draw());
    let r2_mid = r2_deep * numpy_uniform(0.37, 0.43, draw());
    let r2_shallow = r2_deep * numpy_uniform(0.18, 0.25, draw());
    let x_center = center_x * numpy_uniform(0.3, 1.7, draw());
    let y_center = center_y * numpy_uniform(0.3, 1.7, draw());
    circle_points(
        [c2_deep, c2_mid, c2_shallow, c2_tip],
        [r2_deep, r2_mid, r2_shallow],
        x_center,
        y_center,
        draw,
        &mut pts,
    );
    pts
}

/// Legacy `compute_salt_body_segmentation`: radius, top and point cloud from
/// horizon 1 (`h1`, `(ni, nj)`) and the draws. `top_scale` multiplies the
/// `U(150, 300)` top offset (1 = legacy).
pub fn salt_geometry(
    h1: &[f64],
    grid: [usize; 3],
    top_scale: f64,
    draw: &mut dyn FnMut() -> f64,
) -> (f64, f64, Vec<[f64; 3]>) {
    let ni = grid[0] as f64;
    let radius = numpy_triangular(ni / 6.0, ni / 5.0, ni / 4.0, draw());
    let vals: Vec<f64> = h1
        .iter()
        .copied()
        .filter(|v| v.abs() < 1.0e10 && !v.is_nan())
        .collect();
    let mut top = numpy_percentile(&vals, 99.0);
    top += top_scale * numpy_uniform(150.0, 300.0, draw());
    let pts = salt_points(top, radius, grid, draw);
    (radius, top, pts)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Outward facet planes `(unit normal, offset)` of the convex hull of `pts`
/// (incremental hull; `None` for degenerate clouds).
pub fn hull_planes(pts: &[[f64; 3]]) -> Option<Vec<([f64; 3], f64)>> {
    let n = pts.len();
    if n < 4 {
        return None;
    }
    let scale = pts
        .iter()
        .flat_map(|p| p.iter())
        .fold(1.0f64, |m, v| m.max(v.abs()));
    let eps = 1e-12 * scale * scale * scale;
    let dist_eps = 1e-12 * scale;
    // Initial tetrahedron from extreme points.
    let i0 = (0..n).min_by(|&a, &b| pts[a][0].partial_cmp(&pts[b][0]).unwrap())?;
    let d2 = |a: usize, b: usize| {
        let d = sub(pts[a], pts[b]);
        dot(d, d)
    };
    let i1 = (0..n).max_by(|&a, &b| d2(a, i0).partial_cmp(&d2(b, i0)).unwrap())?;
    let line = sub(pts[i1], pts[i0]);
    let ld = |a: usize| {
        let c = cross(line, sub(pts[a], pts[i0]));
        dot(c, c)
    };
    let i2 = (0..n).max_by(|&a, &b| ld(a).partial_cmp(&ld(b)).unwrap())?;
    let nrm = cross(line, sub(pts[i2], pts[i0]));
    let pd = |a: usize| dot(nrm, sub(pts[a], pts[i0])).abs();
    let i3 = (0..n).max_by(|&a, &b| pd(a).partial_cmp(&pd(b)).unwrap())?;
    if pd(i3) <= eps || ld(i2) <= eps * scale.recip().max(1e-300) {
        return None;
    }
    let orient = |f: [usize; 3], p: [f64; 3]| {
        let nn = cross(sub(pts[f[1]], pts[f[0]]), sub(pts[f[2]], pts[f[0]]));
        dot(nn, sub(p, pts[f[0]]))
    };
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let inner = {
        let s = [i0, i1, i2, i3].iter().fold([0.0; 3], |acc, &i| {
            [acc[0] + pts[i][0], acc[1] + pts[i][1], acc[2] + pts[i][2]]
        });
        [s[0] / 4.0, s[1] / 4.0, s[2] / 4.0]
    };
    for f in [[i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3]] {
        if orient(f, inner) > 0.0 {
            faces.push([f[0], f[2], f[1]]);
        } else {
            faces.push(f);
        }
    }
    for p in 0..n {
        if [i0, i1, i2, i3].contains(&p) {
            continue;
        }
        // Signed distance (not the raw triple product), so tiny facets
        // near the stem are tested on the same scale as the cap.
        let visible: Vec<bool> = faces
            .iter()
            .map(|&f| {
                let nn = cross(sub(pts[f[1]], pts[f[0]]), sub(pts[f[2]], pts[f[0]]));
                orient(f, pts[p]) > dist_eps * dot(nn, nn).sqrt()
            })
            .collect();
        if !visible.iter().any(|&v| v) {
            continue;
        }
        let mut edges = std::collections::HashSet::new();
        for (f, &v) in faces.iter().zip(&visible) {
            if v {
                edges.insert((f[0], f[1]));
                edges.insert((f[1], f[2]));
                edges.insert((f[2], f[0]));
            }
        }
        let mut next: Vec<[usize; 3]> = Vec::with_capacity(faces.len() + 8);
        let mut horizon = Vec::new();
        for (f, &v) in faces.iter().zip(&visible) {
            if v {
                for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    if !edges.contains(&(e.1, e.0)) {
                        horizon.push(e);
                    }
                }
            } else {
                next.push(*f);
            }
        }
        horizon.sort_unstable();
        for (a, b) in horizon {
            next.push([a, b, p]);
        }
        faces = next;
    }
    Some(
        faces
            .iter()
            .map(|f| {
                let nn = cross(sub(pts[f[1]], pts[f[0]]), sub(pts[f[2]], pts[f[0]]));
                let len = dot(nn, nn).sqrt();
                let u = [nn[0] / len, nn[1] / len, nn[2] / len];
                (u, dot(u, pts[f[0]]))
            })
            .collect(),
    )
}

/// Distance tolerance (samples) of the hull test, standing in for scipy's
/// `find_simplex` tolerance (`100 * eps` in barycentric coordinates).
const HULL_TOL: f64 = 1e-9;

/// Voxels `(i, j, k)` of `grid` inside the convex hull of `pts` (legacy
/// `util.is_it_in_hull`), as one `[k0, k1)` run per column.
pub fn hull_runs(pts: &[[f64; 3]], grid: [usize; 3]) -> Vec<(u32, u32)> {
    let [ni, nj, nkp] = grid;
    let mut runs = vec![(0u32, 0u32); ni * nj];
    let Some(planes) = hull_planes(pts) else {
        return runs;
    };
    for i in 0..ni {
        for j in 0..nj {
            let (x, y) = (i as f64, j as f64);
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
            let mut empty = false;
            for &(u, d) in &planes {
                let rhs = d - u[0] * x - u[1] * y;
                if u[2] > 0.0 {
                    hi = hi.min((rhs + HULL_TOL) / u[2]);
                } else if u[2] < 0.0 {
                    lo = lo.max((rhs + HULL_TOL) / u[2]);
                } else if rhs < -HULL_TOL {
                    empty = true;
                    break;
                }
            }
            if empty || lo > hi {
                continue;
            }
            let k0 = lo.ceil().max(0.0);
            let k1 = (hi.floor() + 1.0).min(nkp as f64);
            if k1 > k0 {
                runs[i * nj + j] = (k0 as u32, k1 as u32);
            }
        }
    }
    runs
}

/// Half-kernel of scipy `gaussian_filter(sigma = 3)` (truncate 4, radius
/// 12): `w[0]` is the centre. IEEE bits from scipy 1.18
/// `_gaussian_kernel1d(3, 0, 12)` (numpy's `exp` and pairwise sum are
/// replayed exactly by storing the weights).
const GAUSS3: [u64; 13] = [
    0x3fc105a329f98197,
    0x3fc01a25f86eb137,
    0x3fbb42a57d56c0be,
    0x3fb4a614d1afd337,
    0x3fabfde9c12bec92,
    0x3fa0fa58939b528f,
    0x3f926defcaeb0202,
    0x3f81e6bccad344ba,
    0x3f6f1e9915139406,
    0x3f58345966f69518,
    0x3f40d8a5ad43c165,
    0x3f24fbe39149e277,
    0x3f0763a210dfb306,
];

#[inline]
fn reflect(n: usize, i: isize) -> usize {
    let p = 2 * n as isize;
    let m = i.rem_euclid(p) as usize;
    if m < n {
        m
    } else {
        2 * n - 1 - m
    }
}

/// scipy `correlate1d` (symmetric branch) with the sigma-3 kernel, mode
/// `reflect`, over `line` into `out`.
fn smooth_line(line: &[f64], out: &mut [f64]) {
    let n = line.len();
    let w: [f64; 13] = GAUSS3.map(f64::from_bits);
    for l in 0..n {
        let mut t = line[l] * w[0];
        for j in (1..=12isize).rev() {
            t += (line[reflect(n, l as isize - j)] + line[reflect(n, l as isize + j)])
                * w[j as usize];
        }
        out[l] = t;
    }
}

/// scipy `ndimage.gaussian_filter(map, 3)` of an `(ni, nj)` float64 map
/// (axis 0 then axis 1), bit-exact.
pub fn gaussian_filter_sigma3(map: &[f64], ni: usize, nj: usize) -> Vec<f64> {
    assert_eq!(map.len(), ni * nj);
    let mut a = map.to_vec();
    let mut line = vec![0.0; ni.max(nj)];
    let mut out = vec![0.0; ni.max(nj)];
    for j in 0..nj {
        for i in 0..ni {
            line[i] = a[i * nj + j];
        }
        smooth_line(&line[..ni], &mut out[..ni]);
        for i in 0..ni {
            a[i * nj + j] = out[i];
        }
    }
    for i in 0..ni {
        line[..nj].copy_from_slice(&a[i * nj..(i + 1) * nj]);
        smooth_line(&line[..nj], &mut out[..nj]);
        a[i * nj..(i + 1) * nj].copy_from_slice(&out[..nj]);
    }
    a
}

/// Legacy `update_depth_maps_with_salt_segments_drag` (dragged maps; the gap
/// maps are not built, see [`closure_fill_input`]) followed by
/// `push_down_remove_negative_thickness`. `maps` is `(ni, nj, nh)` in
/// samples; horizon depths index the salt grid after `astype(int)` and a
/// clip to `[0, nk + pad - 1]` (legacy `faulted_depth.shape[2] - 1`).
///
/// Legacy quirks kept: the Gaussian smoothing is applied to every horizon,
/// including those that never touch salt; the shift counter `r` is
/// cumulative over horizons; the push-down never fixes horizons 0 / 1.
pub fn drag_horizon_maps(maps: &[f64], shape: [usize; 3], salt: &SaltBody) -> Vec<f64> {
    let [ni, nj, nh] = shape;
    let n = ni * nj;
    assert_eq!(maps.len(), n * nh);
    assert_eq!(salt.runs.len(), n);
    let hi = salt.grid[2] as i64 - 1;
    let mut out = vec![0.0f64; n * nh];
    let mut rel: i64 = 0;
    let mut m = vec![0.0f64; n];
    let mut lab = vec![false; n];
    for h in 0..nh {
        for c in 0..n {
            m[c] = maps[c * nh + h];
            let k = (m[c] as i64).clamp(0, hi) as usize;
            lab[c] = salt.contains(c, k);
        }
        if lab.iter().any(|&v| v) {
            rel += 1;
        }
        for c in 0..n {
            if lab[c] {
                m[c] -= (2 * rel) as f64;
            }
        }
        let s = gaussian_filter_sigma3(&m, ni, nj);
        for c in 0..n {
            out[c * nh + h] = s[c];
        }
    }
    synthoseis_geo::enforce_nonnegative_thicknesses(&mut out, shape);
    out
}

/// Scale of the legacy `U(150, 300)` sample top offset for a cube of `nk`
/// samples: `min(nk / 1250, 1)` by default, so small cubes keep the salt
/// inside and cubes over 1250 samples never put it deeper than legacy; `1`
/// with `legacy_top_offset` (`--salt-legacy-top-offset`).
pub fn top_offset_scale(nk: usize, legacy_top_offset: bool) -> f64 {
    if legacy_top_offset {
        1.0
    } else {
        (nk as f64 / LEGACY_SAMPLES).min(1.0)
    }
}

/// Keyed draws of the salt geometry of `seed` (one unit per legacy draw).
pub fn keyed_draws(seed: u64) -> impl FnMut() -> f64 {
    let mut n = 0u64;
    move || {
        let u = keyed_unit(&[seed, STREAM_SALT, n]);
        n += 1;
        u
    }
}

/// Salt body of `cfg` from its unfaulted, undragged toy horizons `maps`
/// (`(ni, nj, nh)`); `None` when salt is off (see
/// [`E2eConfig::effective_salt`]).
pub fn salt_body_from_maps(cfg: &E2eConfig, maps: &[f64], nh: usize) -> Option<SaltBody> {
    if !cfg.effective_salt() {
        return None;
    }
    let [ni, nj, nk] = cfg.shape();
    let grid = [ni, nj, nk + SALT_PAD];
    let hsel = 1.min(nh - 1);
    let h1: Vec<f64> = (0..ni * nj).map(|c| maps[c * nh + hsel]).collect();
    let scale = top_offset_scale(nk, cfg.rock_physics.salt_legacy_top_offset);
    let mut draw = keyed_draws(cfg.seed);
    let (radius, top, points) = salt_geometry(&h1, grid, scale, &mut draw);
    let runs = hull_runs(&points, grid);
    Some(SaltBody {
        grid,
        runs,
        radius,
        top,
        points,
    })
}

/// Salt body of `cfg` (`None` when salt is off). A pure function of `cfg`.
pub fn salt_body(cfg: &E2eConfig) -> Option<SaltBody> {
    if !cfg.effective_salt() {
        return None;
    }
    let (maps, nh) = crate::toy_geometry::layered_horizon_maps(cfg.seed, cfg.shape());
    salt_body_from_maps(cfg, &maps, nh)
}

/// Fill input and excluded cells for the closure flood fill of a unit top
/// `top` (`(ni, nj)` samples, NaN where the unit is absent), with legacy
/// `_flood_fill` walls around salt gaps:
/// * a gap is a column whose top sample is salt (legacy
///   `faulted_depth_maps_gaps` is NaN there, which becomes `-1` in the
///   closure map);
/// * gap cells get `-1 + max_column` and their 8-neighbours `top +
///   max_column` (legacy `grey_dilation(emptypicks) == 2`), so spill paths
///   through the salt are blocked and traps seal against the flank;
/// * gap and ring cells are never closed (legacy sets the fill to 0 there).
///
/// Cells on the array border are left unwalled (legacy zeroes a 3-cell
/// border before walling). Without salt the input is `top` unchanged and
/// nothing is excluded.
pub fn closure_fill_input(
    top: &[f64],
    ni: usize,
    nj: usize,
    max_column: f64,
    salt: Option<&SaltBody>,
) -> (Vec<f64>, Vec<bool>) {
    let mut input = top.to_vec();
    let Some(salt) = salt else {
        return (input, Vec::new());
    };
    let n = ni * nj;
    let gap: Vec<bool> = (0..n)
        .map(|c| top[c].is_finite() && top[c] >= 0.0 && salt.contains(c, top[c] as usize))
        .collect();
    let mut excluded = vec![false; n];
    if !gap.iter().any(|&g| g) {
        return (input, excluded);
    }
    for i in 0..ni {
        for j in 0..nj {
            let c = i * nj + j;
            if !top[c].is_finite() {
                continue;
            }
            let border = i == 0 || j == 0 || i + 1 == ni || j + 1 == nj;
            if gap[c] {
                excluded[c] = true;
                if !border {
                    input[c] = -1.0 + max_column;
                }
                continue;
            }
            let near = (i.saturating_sub(1)..(i + 2).min(ni))
                .any(|a| (j.saturating_sub(1)..(j + 2).min(nj)).any(|b| gap[a * nj + b]));
            if near {
                excluded[c] = true;
                if !border {
                    input[c] = top[c] + max_column;
                }
            }
        }
    }
    (input, excluded)
}

/// Salt mask chunk `[i0, i1) x [j0, j1) x [k0, k1)` in MDIO chunk order
/// (`(di, dj, dk)`, samples fastest) into `out`.
#[allow(clippy::too_many_arguments)]
pub fn salt_chunk(
    body: &SaltBody,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
    out: &mut Vec<u8>,
) {
    let nj = body.grid[1];
    out.clear();
    for i in i0..i1 {
        for j in j0..j1 {
            let c = i * nj + j;
            out.extend((k0..k1).map(|k| body.contains(c, k) as u8));
        }
    }
}

/// Binary salt volume `(ni, nj, nk)` of `cfg` (`None` without salt).
pub fn generate_salt_labels(cfg: &E2eConfig) -> Option<Vec<u8>> {
    let body = salt_body(cfg)?;
    let [ni, nj, nk] = cfg.shape();
    let mut out = Vec::with_capacity(ni * nj * nk);
    let mut col = Vec::new();
    for i in 0..ni {
        for j in 0..nj {
            salt_chunk(&body, i, i + 1, j, j + 1, 0, nk, &mut col);
            out.extend_from_slice(&col);
        }
    }
    Some(out)
}

/// Check `data/salt_labels` in `store` against the salt body of `cfg`,
/// chunk by chunk (memory: one chunk). `Ok` without salt.
pub fn verify_salt_labels(store: &synthoseis_io::MdioStore, cfg: &E2eConfig) -> Result<(), String> {
    let Some(body) = salt_body(cfg) else {
        return Ok(());
    };
    if cfg.time_enabled() {
        // Output-domain read-back (#34 review rule (c)): the time-domain
        // salt labels, point-sampled through each column's T.
        let want = crate::time_mode::generate_salt_labels_output(cfg).unwrap_or_default();
        let got = store.read_salt_labels_u8().map_err(|e| e.to_string())?;
        return if got == want {
            Ok(())
        } else {
            Err("time-domain salt_labels differ from the resampled salt body".into())
        };
    }
    let shape = store.shape();
    let chunks = store.config().chunks_or_shape();
    let n = |d: usize| shape[d].div_ceil(chunks[d]);
    let mut want = Vec::new();
    for ci in 0..n(0) {
        for cj in 0..n(1) {
            for ck in 0..n(2) {
                let lo = [ci * chunks[0], cj * chunks[1], ck * chunks[2]];
                let hi = [0, 1, 2].map(|d| (lo[d] + chunks[d]).min(shape[d]));
                salt_chunk(&body, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], &mut want);
                let got = store
                    .read_salt_labels_chunk([ci, cj, ck])
                    .map_err(|e| e.to_string())?;
                if got != want {
                    return Err(format!("salt_labels chunk {:?}", [ci, cj, ck]));
                }
            }
        }
    }
    Ok(())
}
