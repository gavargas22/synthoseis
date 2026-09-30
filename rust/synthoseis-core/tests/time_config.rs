//! `TimeConfig` (depth-to-time PR A): library only, disabled by default,
//! validation per the spec §2 / §3.3. No pipeline path reads it yet; the
//! existing golden-hash tests prove the output is unchanged.

use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::{FilterConfig, TimeConfig};
use synthoseis_seismic::TwtKernel;

#[test]
fn defaults_are_disabled_and_keep_the_cube_shape() {
    let tc = TimeConfig::default();
    assert!(!tc.enabled);
    assert_eq!((tc.dt_ms, tc.samples, tc.kernel), (4.0, None, TwtKernel::Sinc));
    let cfg = E2eConfig::default();
    let dz = cfg.rock_physics.depth_step_m;
    assert_eq!(dz, 4.0);
    // nt₀ = round(nz · (2·dz / 2000 m/s) / dt) = nz at the defaults.
    for nz in [16, 64, 256, 510] {
        assert_eq!(tc.output_samples(nz, dz), nz);
    }
    let dt2 = TimeConfig { dt_ms: 2.0, ..TimeConfig::default() };
    assert_eq!(dt2.output_samples(256, dz), 512);
    let long = TimeConfig { samples: Some(256 + 37), ..TimeConfig::default() };
    assert_eq!(long.output_samples(256, dz), 293);
    assert_eq!(TwtKernel::parse("linear"), Some(TwtKernel::Linear));
    assert_eq!(TwtKernel::parse("sinc").unwrap().as_str(), "sinc");
    assert_eq!(TwtKernel::parse("nearest"), None);
}

#[test]
fn validation_matrix() {
    let dz = 4.0;
    let off = FilterConfig::default();
    let bp = FilterConfig::legacy(4.0, 30.0, 3);
    let ok = |tc: &TimeConfig, f: &FilterConfig| tc.validate(256, dz, f);
    // Ricker 40 Hz (f_hi 100 Hz): dt ≤ 4.0 ms, the default sits at the limit.
    assert_eq!(ok(&TimeConfig::default(), &off), Ok(256));
    assert_eq!(TimeConfig::signal_max_hz(&off), 100.0);
    assert!(ok(&TimeConfig { dt_ms: 4.5, ..Default::default() }, &off).unwrap_err().contains("too coarse"));
    // Bandpass up to 30 Hz: dt up to 8 ms passes (limit 13.3 ms, range cap 8).
    assert_eq!(TimeConfig::signal_max_hz(&bp), 30.0);
    assert_eq!(ok(&TimeConfig { dt_ms: 8.0, ..Default::default() }, &bp), Ok(128));
    assert!(ok(&TimeConfig { dt_ms: 8.5, ..Default::default() }, &bp).unwrap_err().contains("0.5-8.0"));
    assert!(ok(&TimeConfig { dt_ms: 0.4, ..Default::default() }, &off).is_err());
    // 16 ≤ nt ≤ 8·nz.
    assert!(ok(&TimeConfig { samples: Some(15), ..Default::default() }, &off).is_err());
    assert_eq!(ok(&TimeConfig { samples: Some(16), ..Default::default() }, &off), Ok(16));
    assert_eq!(ok(&TimeConfig { samples: Some(2048), ..Default::default() }, &off), Ok(2048));
    assert!(ok(&TimeConfig { samples: Some(2049), ..Default::default() }, &off).is_err());
    // Depth staircase: dz ≤ 1580 · dt / 1.2 (5.27 m at 4 ms) is only a warning.
    assert!(TimeConfig::default().staircase_warning(4.0, 1580.0).is_none());
    assert!(TimeConfig::default().staircase_warning(6.0, 1580.0).is_some());
    assert!(TimeConfig { dt_ms: 2.0, ..Default::default() }.staircase_warning(4.0, 1580.0).is_some());
}
