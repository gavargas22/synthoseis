//! 1-D depth trend polynomials from `rockphysics/rpm_example.py` and
//! `rockphysics/rpm_tagilsk_trends.py`.

/// Evaluate a polynomial with coefficients highest-degree first (numpy.polyval).
pub fn polyval(coeffs: &[f64], z: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * z + c)
}

fn polyval_slice(coeffs: &[f64], z: &[f64]) -> Vec<f64> {
    z.iter().map(|&zi| polyval(coeffs, zi)).collect()
}

/// RPMExample depth trends (`rockphysics.rpm_example.RPMExample`).
pub struct RpmExampleTrends;

impl RpmExampleTrends {
    pub fn shale_rho(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| 7.7e-12 * zi.powi(3) + -8.8e-08 * zi.powi(2) + 0.0004 * zi + 1.957)
            .collect()
    }
    pub fn shale_vp(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -0.00013 * zi.powi(2) + 1.13 * zi + 1580.0)
            .collect()
    }
    pub fn shale_vs(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -0.0001 * zi.powi(2) + 0.96 * zi + 279.0)
            .collect()
    }
    pub fn brine_sand_rho(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -7.8e-09 * zi.powi(2) + 0.00012 * zi + 2.021)
            .collect()
    }
    pub fn brine_sand_vp(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -1.34e-05 * zi.powi(2) + 0.49 * zi + 2317.0)
            .collect()
    }
    pub fn brine_sand_vs(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -1.0785e-05 * zi.powi(2) + 0.391 * zi + 1007.0)
            .collect()
    }
    pub fn oil_sand_rho(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -9.23e-09 * zi.powi(2) + 0.00014 * zi + 1.916)
            .collect()
    }
    pub fn oil_sand_vp(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -8.876e-06 * zi.powi(2) + 0.505 * zi + 1998.0)
            .collect()
    }
    pub fn oil_sand_vs(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -1.126e-05 * zi.powi(2) + 0.391 * zi + 1036.0)
            .collect()
    }
    pub fn gas_sand_rho(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -1.818e-08 * zi.powi(2) + 0.000247 * zi + 1.612)
            .collect()
    }
    pub fn gas_sand_vp(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -3.216e-06 * zi.powi(2) + 0.4796 * zi + 1996.0)
            .collect()
    }
    pub fn gas_sand_vs(z: &[f64]) -> Vec<f64> {
        z.iter()
            .map(|&zi| -1.0687e-05 * zi.powi(2) + 0.3662 * zi + 1135.0)
            .collect()
    }
}

/// Tagilsk shale / brine / gas sand trends (`rpm_tagilsk_trends.RPMTagilsk`).
pub struct TagilskTrends;

pub fn tagilsk_shale_rho(z: &[f64]) -> Vec<f64> {
    const P: [f64; 4] = [-3.27905787e-12, 1.86750139e-08, 9.64773845e-05, 2.09709627e+00];
    polyval_slice(&P, z)
}
pub fn tagilsk_shale_vp(z: &[f64]) -> Vec<f64> {
    const P: [f64; 3] = [-2.86940692e-04, 2.02356702e+00, 5.13645163e+02];
    polyval_slice(&P, z)
}
pub fn tagilsk_shale_vs(z: &[f64]) -> Vec<f64> {
    const P: [f64; 3] = [-1.51184658e-04, 1.11423506e+00, 2.17341849e+02];
    polyval_slice(&P, z)
}
pub fn tagilsk_brine_sand_rho(z: &[f64]) -> Vec<f64> {
    const P: [f64; 3] = [1.62260122e-08, 4.19501863e-05, 2.10717208e+00];
    polyval_slice(&P, z)
}
pub fn tagilsk_brine_sand_vp(z: &[f64]) -> Vec<f64> {
    const P: [f64; 3] = [-1.72905613e-04, 1.39902406e+00, 1.15717554e+03];
    polyval_slice(&P, z)
}
pub fn tagilsk_brine_sand_vs(z: &[f64]) -> Vec<f64> {
    const P: [f64; 3] = [-4.86767619e-05, 6.08845432e-01, 7.40710471e+02];
    polyval_slice(&P, z)
}
pub fn tagilsk_gas_sand_vp(z: &[f64]) -> Vec<f64> {
    const A: f64 = 2.03767992e+07;
    const B: f64 = 3.90733465e-08;
    const C: f64 = -2.03752253e+07;
    z.iter().map(|&zi| A * (B * zi).exp() + C).collect()
}
pub fn tagilsk_gas_sand_vs(z: &[f64]) -> Vec<f64> {
    tagilsk_gas_sand_vp(z)
        .into_iter()
        .map(|v| v / std::f64::consts::SQRT_2)
        .collect()
}

impl TagilskTrends {
    pub fn shale_rho(z: &[f64]) -> Vec<f64> {
        tagilsk_shale_rho(z)
    }
    pub fn shale_vp(z: &[f64]) -> Vec<f64> {
        tagilsk_shale_vp(z)
    }
    pub fn shale_vs(z: &[f64]) -> Vec<f64> {
        tagilsk_shale_vs(z)
    }
    pub fn brine_sand_rho(z: &[f64]) -> Vec<f64> {
        tagilsk_brine_sand_rho(z)
    }
    pub fn brine_sand_vp(z: &[f64]) -> Vec<f64> {
        tagilsk_brine_sand_vp(z)
    }
    pub fn brine_sand_vs(z: &[f64]) -> Vec<f64> {
        tagilsk_brine_sand_vs(z)
    }
    pub fn gas_sand_vp(z: &[f64]) -> Vec<f64> {
        tagilsk_gas_sand_vp(z)
    }
    pub fn gas_sand_vs(z: &[f64]) -> Vec<f64> {
        tagilsk_gas_sand_vs(z)
    }
}
