//! Zoeppritz PP reflectivity — port of `tests._zoeppritz_reference` /
//! `datagenerator.zoeppritz_kernel`.
//!
//! Complex arithmetic is required for post-critical angles (where
//! `sin(theta) * v / vp1 > 1`). Only the real part of the final PP ratio is
//! returned.
//!
//! Two forms of the explicit Aki & Richards (1980) PP solution:
//! * [`ZoeppritzForm::Exact`] (default of the rock-physics pipeline): the
//!   textbook expression (Aki & Richards eq. 5.40, bruges
//!   `reflection.zoeppritz_rpp`), whose last numerator term is
//!   `a + d cos(theta1)/vp1 cos(phi2)/vs2`.
//! * [`ZoeppritzForm::Legacy`]: legacy `datagenerator/zoeppritz_kernel.py` and
//!   `tests/_zoeppritz_reference.py`, which write `det` (the denominator)
//!   instead of `d` in that term. Identical at normal incidence, wrong at any
//!   non-zero angle. Kept bit for bit for `--legacy-zoeppritz` /
//!   `--legacy-toy-depth`.

use num_complex::Complex64;

/// Which explicit Zoeppritz PP expression to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZoeppritzForm {
    /// Textbook Aki & Richards / bruges (`a + d * ...`).
    #[default]
    Exact,
    /// Legacy kernel with the `det` typo (`a + det * ...`).
    Legacy,
}

/// Scalar PP reflectivity for one interface / incident angle (degrees), in
/// the **legacy** form ([`ZoeppritzForm::Legacy`]).
///
/// Matches `tests._zoeppritz_reference.zoeppritz_vectorised` for scalars.
pub fn zoeppritz_pp(
    vp1: f64,
    vs1: f64,
    rho1: f64,
    vp2: f64,
    vs2: f64,
    rho2: f64,
    angle_deg: f64,
) -> f32 {
    zoeppritz_pp_form(vp1, vs1, rho1, vp2, vs2, rho2, angle_deg, ZoeppritzForm::Legacy)
}

/// Scalar PP reflectivity in the textbook form ([`ZoeppritzForm::Exact`]).
pub fn zoeppritz_pp_exact(
    vp1: f64,
    vs1: f64,
    rho1: f64,
    vp2: f64,
    vs2: f64,
    rho2: f64,
    angle_deg: f64,
) -> f32 {
    zoeppritz_pp_form(vp1, vs1, rho1, vp2, vs2, rho2, angle_deg, ZoeppritzForm::Exact)
}

/// Complex PP reflection coefficient (both forms share every term but one).
#[allow(clippy::too_many_arguments)]
pub fn zoeppritz_pp_complex(
    vp1: f64,
    vs1: f64,
    rho1: f64,
    vp2: f64,
    vs2: f64,
    rho2: f64,
    angle_deg: f64,
    form: ZoeppritzForm,
) -> Complex64 {
    let theta = Complex64::new(angle_deg.to_radians(), 0.0);
    let p = theta.sin() / vp1;
    let theta2 = (p * vp2).asin();
    let phi1 = (p * vs1).asin();
    let phi2 = (p * vs2).asin();

    let sin_phi1_sq = phi1.sin().powi(2);
    let sin_phi2_sq = phi2.sin().powi(2);
    let cos_theta = theta.cos();
    let cos_theta2 = theta2.cos();
    let cos_phi1 = phi1.cos();
    let cos_phi2 = phi2.cos();

    let a = rho2 * (1.0 - 2.0 * sin_phi2_sq) - rho1 * (1.0 - 2.0 * sin_phi1_sq);
    let b = rho2 * (1.0 - 2.0 * sin_phi2_sq) + 2.0 * rho1 * sin_phi1_sq;
    let c = rho1 * (1.0 - 2.0 * sin_phi1_sq) + 2.0 * rho2 * sin_phi2_sq;
    let d = 2.0 * (rho2 * vs2 * vs2 - rho1 * vs1 * vs1);

    let e = b * cos_theta / vp1 + c * cos_theta2 / vp2;
    let f = b * cos_phi1 / vs1 + c * cos_phi2 / vs2;
    let g = a - d * cos_theta / vp1 * cos_phi2 / vs2;
    let h = a - d * cos_theta2 / vp2 * cos_phi1 / vs1;

    let det = e * f + g * h * p * p;
    let last = match form {
        ZoeppritzForm::Exact => a + d * cos_theta / vp1 * cos_phi2 / vs2,
        ZoeppritzForm::Legacy => a + det * cos_theta / vp1 * cos_phi2 / vs2,
    };
    (f * (b * cos_theta / vp1 - c * cos_theta2 / vp2) - h * p * p * last) / det
}

/// Scalar PP reflectivity (real part) in the given form.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn zoeppritz_pp_form(
    vp1: f64,
    vs1: f64,
    rho1: f64,
    vp2: f64,
    vs2: f64,
    rho2: f64,
    angle_deg: f64,
    form: ZoeppritzForm,
) -> f32 {
    zoeppritz_pp_complex(vp1, vs1, rho1, vp2, vs2, rho2, angle_deg, form).re as f32
}

/// Fill per-angle PP reflectivity volumes from layered `vp`/`vs`/`rho` cubes.
///
/// Shapes: props `(il, xl, z)` row-major; output `(n_ang, il, xl, z-1)`.
/// `angles_deg` are incident angles in degrees. Legacy form (see
/// [`compute_rfc_volumes_form`]).
pub fn compute_rfc_volumes(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    shape: [usize; 3],
    angles_deg: &[f64],
) -> Vec<f32> {
    compute_rfc_volumes_form(vp, vs, rho, shape, angles_deg, ZoeppritzForm::Legacy)
}

/// [`compute_rfc_volumes`] with an explicit [`ZoeppritzForm`].
pub fn compute_rfc_volumes_form(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    shape: [usize; 3],
    angles_deg: &[f64],
    form: ZoeppritzForm,
) -> Vec<f32> {
    let [il, xl, z] = shape;
    assert!(z >= 2, "need at least 2 samples along z");
    assert_eq!(vp.len(), il * xl * z);
    assert_eq!(vs.len(), il * xl * z);
    assert_eq!(rho.len(), il * xl * z);
    let zm1 = z - 1;
    let n_ang = angles_deg.len();
    let mut out = vec![0.0f32; n_ang * il * xl * zm1];

    for i in 0..il {
        for j in 0..xl {
            for k in 0..zm1 {
                let i0 = (i * xl + j) * z + k;
                let i1 = i0 + 1;
                let vp1 = vp[i0] as f64;
                let vs1 = vs[i0] as f64;
                let rho1 = rho[i0] as f64;
                let vp2 = vp[i1] as f64;
                let vs2 = vs[i1] as f64;
                let rho2 = rho[i1] as f64;
                for (a, &ang) in angles_deg.iter().enumerate() {
                    let o = ((a * il + i) * xl + j) * zm1 + k;
                    out[o] = zoeppritz_pp_form(vp1, vs1, rho1, vp2, vs2, rho2, ang, form);
                }
            }
        }
    }
    out
}
