//! Validation of the Zoeppritz PP fix (`det` -> `d`) against independent
//! references:
//!
//! 1. a native 4x4 complex solve of the full Zoeppritz linear system (no
//!    explicit formula), for random interfaces and 0-60 degrees, pre- and
//!    post-critical;
//! 2. `tests/fixtures/zoeppritz_reference.json` from
//!    `generate_zoeppritz_reference.py`: numpy matrix solve, bruges
//!    `zoeppritz_rpp`, bruges `akirichards`, and the legacy expression;
//! 3. Aki-Richards at small contrasts: the textbook form converges to it at
//!    second order in the contrast; the typo itself is O(contrast^2).
//!
//! [`ZoeppritzForm::Legacy`] stays bit-identical to the legacy kernel (the
//! existing `seismic_kernels.json` goldens in `tests_kernels.rs`).

use num_complex::Complex64;
use serde::Deserialize;
use synthoseis_seismic::{
    splitmix64, zoeppritz_pp, zoeppritz_pp_complex, zoeppritz_pp_exact, zoeppritz_pp_form,
    ZoeppritzForm,
};

type Props = [f64; 6];

/// PP coefficient from the Aki & Richards (1980) 4x4 system, solved by
/// Gaussian elimination with partial pivoting in complex arithmetic.
fn matrix_rpp(p: Props, angle_deg: f64) -> Complex64 {
    let [vp1, vs1, rho1, vp2, vs2, rho2] = p;
    let t1 = Complex64::new(angle_deg.to_radians(), 0.0);
    let ray = t1.sin() / vp1;
    let t2 = (ray * vp2).asin();
    let f1 = (ray * vs1).asin();
    let f2 = (ray * vs2).asin();
    let two = Complex64::new(2.0, 0.0);
    let mut m = [
        [-t1.sin(), -f1.cos(), t2.sin(), f2.cos(), t1.sin()],
        [t1.cos(), -f1.sin(), t2.cos(), -f2.sin(), t1.cos()],
        [
            (two * t1).sin(),
            (two * f1).cos() * (vp1 / vs1),
            (two * t2).sin() * (rho2 * vs2 * vs2 * vp1 / (rho1 * vs1 * vs1 * vp2)),
            (two * f2).cos() * (rho2 * vs2 * vp1 / (rho1 * vs1 * vs1)),
            (two * t1).sin(),
        ],
        [
            -(two * f1).cos(),
            (two * f1).sin() * (vs1 / vp1),
            (two * f2).cos() * (rho2 * vp2 / (rho1 * vp1)),
            -(two * f2).sin() * (rho2 * vs2 / (rho1 * vp1)),
            (two * f1).cos(),
        ],
    ];
    for col in 0..4 {
        let piv = (col..4).max_by(|&a, &b| m[a][col].norm().total_cmp(&m[b][col].norm())).unwrap();
        m.swap(col, piv);
        for r in col + 1..4 {
            let f = m[r][col] / m[col][col];
            for c in col..5 {
                let v = m[col][c];
                m[r][c] -= f * v;
            }
        }
    }
    let mut x = [Complex64::new(0.0, 0.0); 4];
    for r in (0..4).rev() {
        let mut s = m[r][4];
        for c in r + 1..4 {
            s -= m[r][c] * x[c];
        }
        x[r] = s / m[r][r];
    }
    x[0]
}

/// Aki & Richards linearised PP (average angle, as bruges `akirichards`).
fn aki_richards(p: Props, angle_deg: f64) -> f64 {
    let [vp1, vs1, rho1, vp2, vs2, rho2] = p;
    let t1 = angle_deg.to_radians();
    let t2 = (vp2 / vp1 * t1.sin()).asin();
    let th = 0.5 * (t1 + t2);
    let (dvp, dvs, drho) = (vp2 - vp1, vs2 - vs1, rho2 - rho1);
    let (vp, vs, rho) = (0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2));
    let w = 0.5 * drho / rho;
    let x = 2.0 * (vs / vp1).powi(2) * drho / rho;
    let y = 0.5 * dvp / vp;
    let z = 4.0 * (vs / vp1).powi(2) * dvs / vs;
    w - x * t1.sin().powi(2) + y / th.cos().powi(2) - z * t1.sin().powi(2)
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = splitmix64(self.0);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
}

fn random_interface(r: &mut Rng) -> Props {
    let vp1 = r.range(1500.0, 5000.0);
    let vs1 = vp1 / r.range(1.5, 2.8);
    let rho1 = r.range(1.0, 2.8);
    let vp2 = vp1 * r.range(0.6, 1.6);
    let vs2 = vp2 / r.range(1.5, 2.8);
    let rho2 = rho1 * r.range(0.7, 1.4);
    [vp1, vs1, rho1, vp2, vs2, rho2]
}

fn rpp(p: Props, a: f64, form: ZoeppritzForm) -> Complex64 {
    zoeppritz_pp_complex(p[0], p[1], p[2], p[3], p[4], p[5], a, form)
}

#[test]
fn exact_form_matches_full_matrix_solution() {
    let mut r = Rng(0x2026_0928);
    let (mut max_exact, mut max_legacy, mut n, mut post) = (0.0f64, 0.0f64, 0usize, 0usize);
    for _ in 0..2000 {
        let p = random_interface(&mut r);
        for a in (0..=60).map(|a| a as f64) {
            let m = matrix_rpp(p, a);
            let e = rpp(p, a, ZoeppritzForm::Exact);
            let l = rpp(p, a, ZoeppritzForm::Legacy);
            max_exact = max_exact.max((e - m).norm());
            max_legacy = max_legacy.max((l.re - m.re).abs());
            n += 1;
            post += usize::from(a.to_radians().sin() * p[3].max(p[1]).max(p[4]) / p[0] > 1.0);
            if a == 0.0 {
                assert_eq!(e, l, "forms agree at normal incidence");
            }
        }
    }
    eprintln!(
        "{n} interface/angle pairs ({post} post-critical): max |exact - matrix| = {max_exact:.2e}, \
         max |legacy - matrix| (real) = {max_legacy:.3}"
    );
    assert!(max_exact < 1e-9, "exact vs matrix {max_exact}");
    assert!(max_legacy > 0.05, "the legacy typo must be visible: {max_legacy}");
}

#[derive(Deserialize)]
struct RefCase {
    props: Props,
    angle_deg: f64,
    matrix_re: f64,
    bruges: f64,
    aki_richards: f64,
    legacy: f64,
}

#[derive(Deserialize)]
struct RefFile {
    bruges_version: String,
    cases: Vec<RefCase>,
}

#[test]
fn exact_form_matches_numpy_matrix_and_bruges() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/zoeppritz_reference.json");
    let fx: RefFile = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (mut d_matrix, mut d_bruges, mut d_legacy, mut d_leg_vs_matrix) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut ar = 0.0f64;
    for c in &fx.cases {
        let p = c.props;
        let e = zoeppritz_pp_exact(p[0], p[1], p[2], p[3], p[4], p[5], c.angle_deg) as f64;
        let l = zoeppritz_pp(p[0], p[1], p[2], p[3], p[4], p[5], c.angle_deg) as f64;
        d_matrix = d_matrix.max((e - c.matrix_re).abs());
        d_bruges = d_bruges.max((e - c.bruges).abs());
        d_legacy = d_legacy.max((l - c.legacy).abs());
        d_leg_vs_matrix = d_leg_vs_matrix.max((l - c.matrix_re).abs());
        if c.angle_deg <= 30.0 {
            ar = ar.max((aki_richards(p, c.angle_deg) - c.aki_richards).abs());
        }
    }
    eprintln!(
        "{} fixture cases (bruges {}): max |exact - numpy matrix| {d_matrix:.2e}, |exact - bruges| \
         {d_bruges:.2e}, |legacy - legacy ref| {d_legacy:.2e}, |legacy - matrix| {d_leg_vs_matrix:.3}; \
         Rust Aki-Richards vs bruges {ar:.2e}",
        fx.cases.len(),
        fx.bruges_version
    );
    // f32 output: a few ulp of |r| <= ~1.
    assert!(d_matrix < 2e-7 && d_bruges < 2e-7);
    assert!(d_legacy < 2e-7);
    assert!(d_leg_vs_matrix > 0.1);
    assert!(ar < 1e-12, "the Rust Aki-Richards helper must match bruges");
}

/// Small contrasts: the textbook form converges to Aki-Richards at second
/// order in the contrast (as it must). The legacy typo replaces
/// `d = 2 (rho2 vs2^2 - rho1 vs1^2)` (~1e7 in SI-ish units) with the
/// denominator `det` (~1e-6), which in effect drops the second-order term
/// `h p^2 d cos(theta1) cos(phi2) / (vp1 vs2)`: it also agrees with
/// Aki-Richards to first order and differs from the exact solution by
/// O(eps^2), so the linearisation cannot tell the two apart; the full matrix
/// solution above does. Large contrasts (seabed, gas sands) are where the
/// typo matters.
#[test]
fn exact_form_converges_to_aki_richards_at_second_order() {
    let base = [3000.0, 1500.0, 2.3];
    let dirs = [[1.0, 1.0, 1.0], [1.0, -0.5, 0.3], [-0.7, 0.4, 1.0], [0.2, 1.0, -0.6]];
    let mut prev: Option<(f64, f64)> = None;
    for eps in [0.08, 0.04, 0.02, 0.01, 0.005] {
        let (mut ee, mut el, mut dx) = (0.0f64, 0.0f64, 0.0f64);
        for d in dirs {
            let p = [
                base[0],
                base[1],
                base[2],
                base[0] * (1.0 + eps * d[0]),
                base[1] * (1.0 + eps * d[1]),
                base[2] * (1.0 + eps * d[2]),
            ];
            for a in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
                let r = aki_richards(p, a);
                let e = rpp(p, a, ZoeppritzForm::Exact).re;
                let l = rpp(p, a, ZoeppritzForm::Legacy).re;
                ee = ee.max((e - r).abs());
                el = el.max((l - r).abs());
                dx = dx.max((e - l).abs());
            }
        }
        eprintln!(
            "eps {eps:<6} max |exact - AR| {ee:.3e} ({:.3} eps^2)  |legacy - AR| {el:.3e}  \
             |exact - legacy| {dx:.3e} ({:.3} eps^2)",
            ee / eps / eps,
            dx / eps / eps
        );
        if let Some((pe, pd)) = prev {
            assert!(pe / ee > 3.3, "exact vs AR must be second order: ratio {}", pe / ee);
            assert!(pd / dx > 3.3, "typo size must be second order: ratio {}", pd / dx);
        }
        prev = Some((ee, dx));
        assert!(ee < 0.3 * eps * eps);
        assert!(dx > 0.0);
    }
}

#[test]
fn legacy_form_is_the_default_scalar_and_bit_identical() {
    let mut r = Rng(7);
    for _ in 0..500 {
        let p = random_interface(&mut r);
        for a in [0.0, 7.5, 15.0, 30.0, 45.0] {
            let old = zoeppritz_pp(p[0], p[1], p[2], p[3], p[4], p[5], a);
            let form = zoeppritz_pp_form(p[0], p[1], p[2], p[3], p[4], p[5], a, ZoeppritzForm::Legacy);
            assert_eq!(old.to_bits(), form.to_bits());
            assert_eq!(rpp(p, a, ZoeppritzForm::Legacy).re as f32, old);
        }
    }
    assert_eq!(ZoeppritzForm::default(), ZoeppritzForm::Exact);
}
