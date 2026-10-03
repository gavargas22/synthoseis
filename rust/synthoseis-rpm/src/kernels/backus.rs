//! Backus (1962) long-wave mixing of a cell made of flat sub-layers
//! (partial-voxels spec §1.3, §1.4).
//!
//! For volume fractions `f_u` (Σ f_u = 1) of end-members `(ρ_u, Vp_u, Vs_u)`:
//!
//! ```text
//! ρ̄ = Σ f_u ρ_u                      (arithmetic density)
//! 1/M̄ = Σ f_u / M_u,  M_u = ρ_u Vp_u²   (harmonic P-wave modulus, C33)
//! 1/μ̄ = Σ f_u / μ_u,  μ_u = ρ_u Vs_u²   (harmonic shear modulus, C44)
//! Vp = √(M̄/ρ̄),  Vs = √(μ̄/ρ̄)
//! ```
//!
//! These are the vertical moduli of the Backus VTI medium, exact for a
//! vertically travelling wave across horizontal layering; the anisotropic
//! C11/C13/C66 are not kept (the pipeline is isotropic). This is *not*
//! [`super::MixingMethod::BackusModuli`] (the legacy NTG rule), which
//! averages λ = ρ(Vp² − 2Vs²) instead of M and is ill-conditioned across
//! the seabed (water λ ≈ 0.25 GPa against M = 2.25 GPa). That rule is left
//! untouched.
//!
//! Units cancel (the code works in g/cc and m/s). Arithmetic is f64 in the
//! given order, cast to f32 at the end. A part with μ_u = 0 (a fluid with
//! Vs = 0) gives μ̄ = 0 (Wood/Reuss limit). Fractions are normalised by
//! their sum, so slightly unnormalised input is tolerated.

use super::Elastic32;

/// Backus mix of `parts` = `(fraction, end-member)`. Parts with a
/// non-positive fraction are ignored. A single part, or parts whose
/// end-members are all bit-identical, return that end-member unchanged
/// (pure cells stay bit-exact). Panics if no part has a positive fraction.
pub fn backus_mix(parts: &[(f64, Elastic32)]) -> Elastic32 {
    let mut live = parts.iter().filter(|(f, _)| *f > 0.0);
    let first = live
        .next()
        .expect("backus_mix: no part with a positive fraction")
        .1;
    if live.all(|(_, p)| same_bits(*p, first)) {
        return first;
    }
    let (mut fsum, mut rho, mut inv_m, mut inv_mu, mut fluid) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, false);
    for &(f, p) in parts.iter().filter(|(f, _)| *f > 0.0) {
        let (r, vp, vs) = (p.rho as f64, p.vp as f64, p.vs as f64);
        fsum += f;
        rho += f * r;
        inv_m += f / (r * vp * vp);
        let mu = r * vs * vs;
        if mu > 0.0 {
            inv_mu += f / mu;
        } else {
            fluid = true;
        }
    }
    let rho = rho / fsum;
    let m = fsum / inv_m;
    let mu = if fluid { 0.0 } else { fsum / inv_mu };
    Elastic32 {
        rho: rho as f32,
        vp: (m / rho).sqrt() as f32,
        vs: (mu / rho).sqrt() as f32,
    }
}

/// Vertical slowness sum `Σ f_u / Vp_u` (s/m) of `parts` = `(fraction,
/// Vp)`: the ray (high-frequency) one-way delay per metre of the cell. The
/// two-way time across a cell of `dz` m is `2·dz·slowness_sum` (spec §1.4);
/// it is not `2·dz / Vp_backus`, which is the zero-frequency delay.
pub fn slowness_sum(parts: &[(f64, f32)]) -> f64 {
    parts
        .iter()
        .filter(|(f, _)| *f > 0.0)
        .map(|&(f, vp)| f / vp as f64)
        .sum()
}

/// Voigt (arithmetic moduli, iso-strain) upper bound of the same mix, for
/// bounds checks: `M_V = Σ f M`, `μ_V = Σ f μ`, `ρ̄ = Σ f ρ`.
pub fn voigt_mix(parts: &[(f64, Elastic32)]) -> Elastic32 {
    let (mut fsum, mut rho, mut m, mut mu) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &(f, p) in parts.iter().filter(|(f, _)| *f > 0.0) {
        let (r, vp, vs) = (p.rho as f64, p.vp as f64, p.vs as f64);
        fsum += f;
        rho += f * r;
        m += f * r * vp * vp;
        mu += f * r * vs * vs;
    }
    let rho = rho / fsum;
    Elastic32 {
        rho: rho as f32,
        vp: (m / fsum / rho).sqrt() as f32,
        vs: (mu / fsum / rho).sqrt() as f32,
    }
}

fn same_bits(a: Elastic32, b: Elastic32) -> bool {
    a.rho.to_bits() == b.rho.to_bits()
        && a.vp.to_bits() == b.vp.to_bits()
        && a.vs.to_bits() == b.vs.to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHALE: Elastic32 = Elastic32 {
        rho: 2.277,
        vp: 2580.0,
        vs: 1139.0,
    };
    const GAS: Elastic32 = Elastic32 {
        rho: 1.841,
        vp: 2472.38,
        vs: 1490.51,
    };
    const SALT: Elastic32 = Elastic32 {
        rho: 2.16,
        vp: 4500.0,
        vs: 2600.0,
    };
    const WATER: Elastic32 = Elastic32 {
        rho: 1.0,
        vp: 1500.0,
        vs: 1000.0,
    };

    fn m(p: Elastic32) -> f64 {
        p.rho as f64 * (p.vp as f64).powi(2)
    }
    fn mu(p: Elastic32) -> f64 {
        p.rho as f64 * (p.vs as f64).powi(2)
    }

    #[test]
    fn pure_and_equal_parts_return_the_end_member_bits() {
        for p in [SHALE, GAS, SALT, WATER] {
            assert_eq!(backus_mix(&[(1.0, p)]), p);
            assert_eq!(backus_mix(&[(0.3, p), (0.7, p)]), p);
            assert_eq!(backus_mix(&[(0.0, SALT), (1.0, p)]), p);
        }
    }

    /// Two layers: ρ arithmetic, M and μ harmonic, against hand values.
    #[test]
    fn two_layer_closed_form() {
        for (a, b) in [(SHALE, GAS), (SHALE, SALT), (WATER, SHALE)] {
            for f in [0.5, 0.137, 0.9] {
                let e = backus_mix(&[(f, a), (1.0 - f, b)]);
                let rho = f * a.rho as f64 + (1.0 - f) * b.rho as f64;
                let mm = 1.0 / (f / m(a) + (1.0 - f) / m(b));
                let mmu = 1.0 / (f / mu(a) + (1.0 - f) / mu(b));
                assert!((e.rho as f64 - rho).abs() <= 1e-6 * rho);
                let (vp, vs) = ((mm / rho).sqrt(), (mmu / rho).sqrt());
                assert!(
                    (e.vp as f64 - vp).abs() <= 1e-6 * vp,
                    "{f}: {} vs {vp}",
                    e.vp
                );
                assert!((e.vs as f64 - vs).abs() <= 1e-6 * vs);
            }
        }
        // Hand value, shale/salt at f = ½: ρ = 2.2185; M_sh = 15.1567e6,
        // M_salt = 43.74e6 (g/cc·m²/s²); M̄ = 2/(1/M_sh + 1/M_salt) = 22.5124e6.
        let e = backus_mix(&[(0.5, SHALE), (0.5, SALT)]);
        assert!((e.rho - 2.2185).abs() < 1e-5);
        assert!(
            (e.vp as f64 - (22.5124e6f64 / 2.2185).sqrt()).abs() < 0.5,
            "{}",
            e.vp
        );
    }

    #[test]
    fn permutation_invariant_and_between_the_bounds() {
        let parts = [(0.2, SHALE), (0.5, SALT), (0.3, GAS)];
        let e = backus_mix(&parts);
        for perm in [[0, 2, 1], [1, 0, 2], [2, 1, 0]] {
            let q: Vec<_> = perm.iter().map(|&i| parts[i]).collect();
            let f = backus_mix(&q);
            assert!(
                (f.vp - e.vp).abs() <= 1e-3
                    && (f.vs - e.vs).abs() <= 1e-3
                    && (f.rho - e.rho).abs() <= 1e-6
            );
        }
        let v = voigt_mix(&parts);
        assert!(e.vp <= v.vp && e.vs <= v.vs);
        // Harmonic (lower) bound on M is Backus itself; check M̄ ≥ min M_u.
        let mbar = e.rho as f64 * (e.vp as f64).powi(2);
        assert!(mbar >= parts.iter().map(|p| m(p.1)).fold(f64::INFINITY, f64::min) * (1.0 - 1e-6));
    }

    #[test]
    fn fluid_part_gives_zero_shear_and_slowness_is_additive() {
        let fluid = Elastic32 { vs: 0.0, ..WATER };
        let e = backus_mix(&[(0.5, fluid), (0.5, SHALE)]);
        assert_eq!(e.vs, 0.0);
        assert!(e.vp > 1500.0 && e.vp < 2580.0);
        let s = slowness_sum(&[(0.25, 1500.0), (0.75, 2580.0)]);
        assert!((s - (0.25 / 1500.0 + 0.75 / 2580.0)).abs() < 1e-18);
    }
}
