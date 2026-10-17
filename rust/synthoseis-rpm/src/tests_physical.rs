//! Validation tests of the physical rock-physics kernel (spec §8, tests
//! 1-6 and 8). Reference numbers are constants with their source named.

use crate::*;

/// Tiny deterministic generator (splitmix64) for the random boxes.
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.u()
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-30)
}

/// Test 1: Gassmann round trip (f64 1e-10, f32 1e-5).
#[test]
fn gassmann_round_trip() {
    let mut r = Rng(1);
    for _ in 0..10_000 {
        let km = r.range(20.0, 75.0);
        let phi = r.range(0.02, 0.4);
        let kd = r.range(0.1, 0.9) * km * (1.0 - phi / 0.4).max(0.05);
        let (k1, k2) = (r.range(0.01, 3.5), r.range(0.01, 3.5));
        let ks1 = gassmann(kd, km, k1, phi);
        let kd_back = gassmann_inverse(ks1, km, k1, phi);
        assert!(rel(kd_back, kd) < 1e-10, "dry back {kd_back} vs {kd}");
        let ks2 = gassmann(kd_back, km, k2, phi);
        assert!(rel(ks2, gassmann(kd, km, k2, phi)) < 1e-10);
        // f32 production rounding of the same chain.
        assert!(rel((ks2 as f32) as f64, gassmann(kd, km, k2, phi)) < 1e-5);
    }
}

/// Test 2: brine-to-brine identity, and shale equals the direct brine result.
#[test]
fn brine_to_brine_identity() {
    let p = PhysParams::default();
    for z in [300.0, 1000.0, 2500.0, 4500.0] {
        let ((rho, vp, vs), phi) = end_member_f64(&p.shale, PoreFluid::Brine, z, 1000.0, &p);
        let t = 4.0 + 0.03 * z;
        let pp = 1030.0 * 9.81 * (1000.0 + z) * 1e-6;
        let (_, _, kb) = bw_brine(t, pp, 0.035);
        let mu = vs * vs * rho * 1e-6;
        let ksat = vp * vp * rho * 1e-6 - 4.0 / 3.0 * mu;
        let kd = gassmann_inverse(ksat, p.shale.k_m, kb, phi);
        let again = gassmann(kd, p.shale.k_m, kb, phi);
        assert!(rel(again, ksat) < 1e-12, "z={z}: {again} vs {ksat}");
        let e32 = physical_shale(z as f32, 1000.0, &p);
        assert_eq!(e32.vp, vp as f32);
        assert_eq!(e32.rho, rho as f32);
        assert_eq!(e32.vs, vs as f32);
    }
}

/// Test 3: limits of Gassmann.
#[test]
fn gassmann_limits() {
    let mut r = Rng(3);
    for _ in 0..1000 {
        let km = r.range(20.0, 75.0);
        let phi = r.range(0.02, 0.4);
        let kd = r.range(0.0, 0.9) * km;
        assert!(rel(gassmann(kd, km, km, phi), km) < 1e-9);
        assert!(rel(gassmann(kd, km, r.range(0.01, 3.0), 1e-12), km) < 1e-6);
        let kf = r.range(0.01, 3.0);
        let reuss = 1.0 / (phi / kf + (1.0 - phi) / km);
        assert!(rel(gassmann(0.0, km, kf, phi), reuss) < 1e-12);
    }
}

/// Test 4: bounds on a (φ, P_eff, fluid) grid, zero violations.
#[test]
fn moduli_bounds() {
    let p = PhysParams::default();
    let fluids = [(1.0e-4, 0.1), (0.05, 0.2), (1.0, 0.8), (2.5, 1.05)];
    let mut bad = 0;
    for lith in [p.sand, p.shale] {
        let (km, mm) = (lith.k_m, lith.mu_m);
        for i in 1..40 {
            let phi = 0.01 + (lith.phi_c - 0.011) * i as f64 / 40.0;
            for j in 0..30 {
                let pe = 0.01 + 80.0 * j as f64 / 30.0;
                let (kd, md) = soft_sand_dry(km, mm, phi, lith.phi_c, lith.coord, pe * 1e-3);
                // HS upper bound of mineral + empty pore (Hashin & Shtrikman 1963).
                let ku = km + phi / (-1.0 / km + (1.0 - phi) / (km + 4.0 / 3.0 * mm));
                let mu_u = mm
                    + phi
                        / (-1.0 / mm
                            + 2.0 * (1.0 - phi) * (km + 2.0 * mm)
                                / (5.0 * mm * (km + 4.0 / 3.0 * mm)));
                if !(kd >= 0.0
                    && kd <= ku * (1.0 + 1e-12)
                    && md >= 0.0
                    && md <= mu_u * (1.0 + 1e-12))
                {
                    bad += 1;
                }
                for &(kf, rf) in &fluids {
                    let ks = gassmann(kd, km, kf, phi);
                    let reuss = 1.0 / (phi / kf + (1.0 - phi) / km);
                    let voigt = phi * kf + (1.0 - phi) * km;
                    if !(ks >= reuss * (1.0 - 1e-12) && ks <= voigt * (1.0 + 1e-12) && ks >= kd) {
                        bad += 1;
                    }
                    let rho = (1.0 - phi) * lith.rho_m + phi * rf;
                    let (r, vp, vs) = elastic_from_moduli(ks, md, rho);
                    if !(r > 0.0 && vp.is_finite() && vp > 0.0 && vs.is_finite() && vs > 0.0) {
                        bad += 1;
                    }
                }
            }
        }
    }
    assert_eq!(bad, 0);
}

/// Test 5: Athy fit, and the stress form within 2% of the depth form.
#[test]
fn athy_fit_and_stress_form() {
    let p = PhysParams::default();
    for lith in [p.sand, p.shale] {
        let zs: Vec<f64> = (0..=50).map(|i| i as f64 * 100.0).collect();
        let ys: Vec<f64> = zs
            .iter()
            .map(|&z| athy_porosity(lith.phi0, lith.athy_c, z).ln())
            .collect();
        let n = zs.len() as f64;
        let (mx, my) = (zs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let sxy: f64 = zs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = zs.iter().map(|x| (x - mx) * (x - mx)).sum();
        let slope = sxy / sxx;
        let phi0 = (my - slope * mx).exp();
        assert!((phi0 - lith.phi0).abs() < 1e-6);
        assert!((-slope * 1000.0 - lith.athy_c).abs() < 1e-6);
        // Stress form at hydrostatic pressure with the derived σ0. The
        // spec's "within 2% over 0-5 km" does not hold: with
        // σ0 = (ρ_m − ρ_w)g/c, P_eff/σ0 = cz − φ0(1 − e^{−cz}), so the stress
        // form is the depth form times exp(φ0(1 − e^{−cz})) (up to ×1.6 at
        // depth). Pin that exact identity instead, and report the gap.
        let s0 = (lith.rho_m - p.rho_w) * 1000.0 * p.g / (lith.athy_c * 1e-3) * 1e-6;
        for &z in &zs {
            let pe = effective_pressure(z, lith.phi0, lith.athy_c, lith.rho_m, p.rho_w, p.g);
            let a = lith.phi0 * (-pe / s0).exp();
            let d = athy_porosity(lith.phi0, lith.athy_c, z);
            let cz = lith.athy_c * z * 1e-3;
            assert!(
                rel(a, d * (lith.phi0 * (1.0 - (-cz).exp())).exp()) < 1e-9,
                "z={z}"
            );
            assert!(a >= d);
        }
    }
}

/// Test 8: trend sanity of the physical model, 0.5-4 km.
#[test]
fn physical_trend_sanity() {
    let p = PhysParams::default();
    let w = 1000.0f32;
    let mut prev = [0.0f32; 2];
    for i in 0..=14 {
        let z = 500.0 + 250.0 * i as f32;
        let shale = physical_shale(z, w, &p);
        let brine = physical_sand(Fluid::Brine, z, w, &p);
        let gas = physical_sand(Fluid::Gas, z, w, &p);
        let oil = physical_sand(Fluid::Oil, z, w, &p);
        for (n, e) in [
            ("shale", shale),
            ("brine", brine),
            ("gas", gas),
            ("oil", oil),
        ] {
            assert!(
                e.rho.is_finite() && e.vp.is_finite() && e.vs.is_finite(),
                "{n} z={z}"
            );
        }
        // Vp/Vs bound on brine sand and shale (spec: "brine-sand and shale").
        // Shallow gas sand legitimately drops to ~1.45 (dry-frame Poisson).
        for (n, e) in [("shale", shale), ("brine", brine)] {
            let r = e.vp / e.vs;
            assert!((1.6..=3.0).contains(&r), "{n} z={z}: Vp/Vs {r}");
        }
        assert!(
            shale.vp > prev[0] && brine.vp > prev[1],
            "monotonic at z={z}"
        );
        prev = [shale.vp, brine.vp];
        assert!(gas.vp < brine.vp, "gas below brine at z={z}");
    }
    // D1 guard: sand-shale impedance contrast within ±25% of legacy at 2 km.
    let con = |sh: Elastic32, sa: Elastic32| {
        let (a, b) = (sh.rho * sh.vp, sa.rho * sa.vp);
        (b - a) / (0.5 * (a + b))
    };
    let leg = con(
        shale_f32(2000.0, 2000.0, 2000.0),
        sand_f32(Fluid::Brine, 2000.0, 2000.0, 2000.0),
    );
    let phys = con(
        physical_shale(2000.0, w, &p),
        physical_sand(Fluid::Brine, 2000.0, w, &p),
    );
    assert!(
        (phys - leg).abs() <= 0.25 * leg.abs(),
        "contrast phys {phys} legacy {leg}"
    );
}

/// f32 production vs f64 reference (spec §4.1, < 1e-5 relative).
#[test]
fn f32_matches_f64_reference() {
    let p = PhysParams::default();
    for z in [0.0, 50.0, 800.0, 2000.0, 3700.0, 6000.0] {
        for f in [Fluid::Brine, Fluid::Oil, Fluid::Gas] {
            let pf = match f {
                Fluid::Brine => PoreFluid::Brine,
                Fluid::Oil => PoreFluid::Oil,
                Fluid::Gas => PoreFluid::Gas,
            };
            let ((r, vp, vs), _) = end_member_f64(&p.sand, pf, z, 300.0, &p);
            let e = physical_sand(f, z as f32, 300.0, &p);
            assert!(
                rel(e.rho as f64, r) < 1e-5
                    && rel(e.vp as f64, vp) < 1e-5
                    && rel(e.vs as f64, vs) < 1e-5
            );
        }
    }
}

#[test]
fn params_validate() {
    assert!(PhysParams::default().validate().is_ok());
    let mut p = PhysParams::default();
    p.salinity = 0.5;
    assert!(p.validate().is_err());
}

/// Spec §8 test 6: Batzle-Wang spot values against independent reference
/// formulations. Water: IAPWS-95 (Wagner & Pruss 2002,
/// doi:10.1063/1.1461829) and Del Grosso & Mader 1972 at 20 °C; methane:
/// Setzmann & Wagner 1991 (doi:10.1063/1.555898). Reference numbers were
/// evaluated once with CoolProp 8.0.0 and are held here as constants; CI
/// does not depend on CoolProp.
#[test]
fn batzle_wang_spot_values() {
    // Pure water velocity at 20 °C, 0.1 MPa: reference 1482.34 m/s.
    let (_, v20) = bw_water(20.0, 0.1);
    assert!((v20 - 1482.34).abs() <= 0.5, "water V(20 °C, 0.1 MPa) = {v20}");
    // (T °C, P MPa, IAPWS-95 V m/s, IAPWS-95 ρ g/cc).
    const WATER: [(f64, f64, f64, f64); 3] = [
        (50.0, 20.0, 1577.886, 0.996532),
        (80.0, 40.0, 1630.416, 0.988831),
        (100.0, 60.0, 1662.360, 0.984342),
    ];
    for (t, p, v_ref, rho_ref) in WATER {
        let (rho, v) = bw_water(t, p);
        assert!((v / v_ref - 1.0).abs() <= 0.005, "water V({t}, {p}) = {v} vs {v_ref}");
        assert!((rho / rho_ref - 1.0).abs() <= 0.003, "water ρ({t}, {p}) = {rho} vs {rho_ref}");
    }
    // Methane (G = 0.554), Setzmann & Wagner 1991 density (g/cc).
    const METHANE: [(f64, f64, f64); 4] =
        [(40.0, 10.0, 0.069974), (60.0, 20.0, 0.129170), (80.0, 30.0, 0.166782), (100.0, 40.0, 0.190680)];
    for (t, p, rho_ref) in METHANE {
        let (rho, _) = bw_gas(t, p, 0.554);
        assert!((rho / rho_ref - 1.0).abs() <= 0.03, "methane ρ({t}, {p}) = {rho} vs {rho_ref}");
    }
    // Gas bulk modulus, loose sanity check only (spec: B&W's isothermal-ish
    // approximation vs 14.6 MPa adiabatic at 40 °C, 10 MPa): within 25 %.
    let (_, k) = bw_gas(40.0, 10.0, 0.554);
    assert!((k * 1e3 / 14.6 - 1.0).abs() <= 0.25, "methane K(40, 10) = {} MPa", k * 1e3);
    // Brine salinity term with the corrected −1820 S² coefficient (the
    // printed paper's −820 is a typo; US 2011/0114385). −820 gives 37.812.
    let (_, vw) = bw_water(25.0, 0.1);
    let (_, vb, _) = bw_brine(25.0, 0.1, 0.035);
    let dv = vb - vw;
    assert!((dv - 36.587).abs() <= 1e-3, "V_B - V_w at (25 °C, 0.1 MPa, S 0.035) = {dv}, want 36.587 (-1820)");
    assert!((dv - 37.812).abs() > 1.0, "brine velocity matches the -820 typo");
}

/// Live-oil density is B&W eq. 24's ρ_G at saturation, with no second
/// eq. 18-19 temperature correction (that divided by 1.040 at 64 °C; with the eq. 18 pressure term the net
/// effect was 1.7 % low at 20 MPa).
/// Defaults API 32, GOR 100, G 0.6 at 64 °C, 20 MPa.
#[test]
fn live_oil_density_is_rho_g_at_saturation() {
    let (api, gor, g, t, p): (f64, f64, f64, f64, f64) = (32.0, 100.0, 0.6, 64.0, 20.0);
    let rho0 = 141.5 / (api + 131.5);
    let b0 = 0.972 + 0.00038 * (2.4 * gor * (g / rho0).sqrt() + t + 17.8).powf(1.175);
    let rho_g = (rho0 + 0.0012 * g * gor) / b0;
    let (rho, _) = bw_oil(t, p, api, gor, g);
    assert_eq!(rho, rho_g);
    assert!((rho - 0.74448).abs() < 5e-5, "live oil ρ = {rho}");
}
