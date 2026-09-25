//! Parity of the seismic filter port against the legacy Python generator.
//!
//! Fixture: `tests/fixtures/seismic_filters.json`, written by
//! `tests/fixtures/generate_seismic_filters.py` from the real
//! `datagenerator/Seismic.py` methods (`derive_butterworth_bandpass`,
//! `apply_bandlimits`, `apply_lateral_filter`, `apply_cumsum`).

use serde::Deserialize;
use synthoseis_seismic::{
    butterworth_bandpass, cumsum_traces_f32, lateral_source_range, lateral_uniform_tile,
    lateral_uniform_volume, legacy_digitisation_ms, reflect_index, FilterError, IirFilter,
};

#[derive(Deserialize)]
struct Fixture {
    designs: Vec<Design>,
    volume: Volume,
    bandpass: Vec<Bandpass>,
    cumsum_order4: Vec<f64>,
    chain_4p5_27p3_o4_lat3: Vec<f64>,
    lateral: Vec<Lateral>,
}

#[derive(Deserialize)]
struct Design {
    low: f64,
    high: f64,
    digi: f64,
    order: usize,
    b: Vec<f64>,
    a: Vec<f64>,
    zi: Vec<f64>,
}

#[derive(Deserialize)]
struct Volume {
    shape: [usize; 3],
    input: Vec<f64>,
}

#[derive(Deserialize)]
struct Bandpass {
    low: f64,
    high: f64,
    order: usize,
    output: Vec<f64>,
}

#[derive(Deserialize)]
struct Lateral {
    shape: [usize; 3],
    size: usize,
    input: Vec<f64>,
    output: Vec<f64>,
}

fn fixture() -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/seismic_filters.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).expect("parse seismic_filters.json")
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0, f64::max)
}

/// (max abs diff, max |expected|, exact-match fraction).
fn compare_f32(got: &[f32], want: &[f32]) -> (f32, f32, f64) {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut peak = 0.0f32;
    let mut exact = 0usize;
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        peak = peak.max(w.abs());
        exact += (g.to_bits() == w.to_bits()) as usize;
    }
    (max_abs, peak, exact as f64 / got.len() as f64)
}

fn design(low: f64, high: f64, digi: f64, order: usize) -> IirFilter {
    butterworth_bandpass(low, high, legacy_digitisation_ms(digi), order).expect("design")
}

#[test]
fn butterworth_design_matches_scipy() {
    let fix = fixture();
    // b, a agree with scipy to a few ulps (bit-identical for most designs; the
    // rest differ in the last bits of numpy's SIMD complex products). zi is
    // noisier in relative terms: y_inf = sum(b) / sum(a) divides rounding noise
    // (sum(b) is 0 in exact arithmetic) by the small sum(a), so it inherits
    // last-ulp differences in a; it only seeds the filter state, and the
    // filtered traces below are bit-identical.
    let (mut worst_ba, mut worst_zi) = (0.0f64, 0.0f64);
    for d in &fix.designs {
        let f = design(d.low, d.high, d.digi, d.order);
        assert_eq!(f.b.len(), 2 * d.order + 1);
        let (rb, ra, rz) = (
            max_rel(&f.b, &d.b),
            max_rel(&f.a, &d.a),
            max_rel(&f.zi, &d.zi),
        );
        println!(
            "butter({}, [{}, {}] Hz, dt={} ms): rel diff b={rb:.2e} a={ra:.2e} zi={rz:.2e}",
            d.order, d.low, d.high, d.digi
        );
        worst_ba = worst_ba.max(rb).max(ra);
        worst_zi = worst_zi.max(rz);
    }
    assert!(worst_ba < 1e-14, "b/a diverge from scipy: {worst_ba:e}");
    assert!(worst_zi < 1e-10, "zi diverges from scipy: {worst_zi:e}");
}

#[test]
fn bandpass_matches_legacy_apply_bandlimits() {
    let fix = fixture();
    let nk = fix.volume.shape[2];
    let input = f32s(&fix.volume.input);
    for case in &fix.bandpass {
        let f = design(case.low, case.high, 4.0, case.order);
        let mut got = input.clone();
        f.filtfilt_traces_f32(&mut got, nk).unwrap();
        let want = f32s(&case.output);
        let (max_abs, peak, exact) = compare_f32(&got, &want);
        println!(
            "bandpass {}-{} Hz order {}: max abs {max_abs:.3e} (peak {peak:.3}), bit-exact {:.2}%",
            case.low,
            case.high,
            case.order,
            exact * 100.0
        );
        assert!(
            max_abs <= 2e-7 * peak.max(1.0),
            "bandpass diverges: {max_abs}"
        );
        assert!(exact > 0.95, "bit-exact fraction {exact}");
    }
}

#[test]
fn cumsum_stage_matches_legacy_apply_cumsum() {
    let fix = fixture();
    let nk = fix.volume.shape[2];
    let mut got = f32s(&fix.volume.input);
    cumsum_traces_f32(&mut got, nk);
    design(2.0, 100.0, 4.0, 4)
        .filtfilt_traces_f32(&mut got, nk)
        .unwrap();
    let want = f32s(&fix.cumsum_order4);
    let (max_abs, peak, exact) = compare_f32(&got, &want);
    println!(
        "cumsum+2-100 Hz: max abs {max_abs:.3e} (peak {peak:.3}), bit-exact {:.2}%",
        exact * 100.0
    );
    assert!(max_abs <= 2e-7 * peak.max(1.0));
}

#[test]
fn lateral_filter_matches_legacy_uniform_filter() {
    let fix = fixture();
    for case in &fix.lateral {
        let got = lateral_uniform_volume(&f32s(&case.input), case.shape, case.size);
        let want = f32s(&case.output);
        let (max_abs, peak, exact) = compare_f32(&got, &want);
        println!(
            "lateral size {} on {:?}: max abs {max_abs:.3e} (peak {peak:.3}), bit-exact {:.2}%",
            case.size,
            case.shape,
            exact * 100.0
        );
        assert!(
            max_abs <= 1.2e-7 * peak.max(1.0),
            "lateral diverges: {max_abs}"
        );
    }
}

#[test]
fn bandpass_then_lateral_matches_legacy_chain() {
    let fix = fixture();
    let shape = fix.volume.shape;
    let mut v = f32s(&fix.volume.input);
    design(4.5, 27.3, 4.0, 4)
        .filtfilt_traces_f32(&mut v, shape[2])
        .unwrap();
    let got = lateral_uniform_volume(&v, shape, 3);
    let want = f32s(&fix.chain_4p5_27p3_o4_lat3);
    let (max_abs, peak, exact) = compare_f32(&got, &want);
    println!(
        "bandpass 4.5-27.3 + lateral 3: max abs {max_abs:.3e} (peak {peak:.3}), bit-exact {:.2}%",
        exact * 100.0
    );
    assert!(max_abs <= 2e-7 * peak.max(1.0));
}

#[test]
fn lateral_tiles_are_bit_identical_to_whole_volume() {
    let shape = [11usize, 9, 7];
    let vol: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|n| ((n as f32) * 0.618).sin() * (1.0 + (n % 13) as f32))
        .collect();
    for size in [2usize, 3, 4, 5, 7] {
        let reference = lateral_uniform_volume(&vol, shape, size);
        for (ci, cj) in [(1usize, 1usize), (2, 3), (4, 4), (5, 2), (11, 9)] {
            let mut i0 = 0;
            while i0 < shape[0] {
                let i1 = (i0 + ci).min(shape[0]);
                let mut j0 = 0;
                while j0 < shape[1] {
                    let j1 = (j0 + cj).min(shape[1]);
                    let (si0, si1) = lateral_source_range(i0, i1, shape[0], size);
                    let (sj0, sj1) = lateral_source_range(j0, j1, shape[1], size);
                    let mut src = Vec::new();
                    for i in si0..si1 {
                        for j in sj0..sj1 {
                            let b = (i * shape[1] + j) * shape[2];
                            src.extend_from_slice(&vol[b..b + shape[2]]);
                        }
                    }
                    let mut out = vec![0.0f32; (i1 - i0) * (j1 - j0) * shape[2]];
                    lateral_uniform_tile(
                        &src,
                        (si0, si1),
                        (sj0, sj1),
                        shape,
                        (i0, i1),
                        (j0, j1),
                        size,
                        &mut out,
                    );
                    for i in i0..i1 {
                        for j in j0..j1 {
                            let r = (i * shape[1] + j) * shape[2];
                            let t = ((i - i0) * (j1 - j0) + (j - j0)) * shape[2];
                            assert_eq!(
                                &out[t..t + shape[2]],
                                &reference[r..r + shape[2]],
                                "size {size} tile {ci}x{cj} at ({i},{j})"
                            );
                        }
                    }
                    j0 = j1;
                }
                i0 = i1;
            }
        }
    }
}

#[test]
fn reflect_index_matches_scipy_reflect_mode() {
    // d c b a | a b c d | d c b a (period 2n), including multiple wraps.
    let n = 4;
    let got: Vec<usize> = (-9..13).map(|i| reflect_index(i, n)).collect();
    let want = [
        0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3,
    ];
    assert_eq!(got, want);
    assert!((-5..5).all(|i| reflect_index(i, 1) == 0));
}

#[test]
fn errors_for_short_traces_and_bad_corners() {
    let f = design(3.0, 20.0, 4.0, 4);
    assert_eq!(f.padlen(), 27);
    let mut short = vec![0.0f32; 27];
    let mut scratch = Vec::new();
    assert_eq!(
        f.filtfilt_f32(&mut short, &mut scratch),
        Err(FilterError::TraceTooShort {
            len: 27,
            padlen: 27
        })
    );
    assert!(matches!(
        butterworth_bandpass(30.0, 20.0, 4.0, 4),
        Err(FilterError::BadCorners { .. })
    ));
    assert!(matches!(
        butterworth_bandpass(3.0, 130.0, 4.0, 4),
        Err(FilterError::BadCorners { .. })
    ));
    assert_eq!(
        butterworth_bandpass(3.0, 20.0, 4.0, 0),
        Err(FilterError::ZeroOrder)
    );
}

#[test]
fn bandpass_passes_in_band_and_rejects_out_of_band() {
    let f = design(5.0, 30.0, 4.0, 4);
    let nk = 1000;
    let rms = |hz: f64| {
        let mut t: Vec<f32> = (0..nk)
            .map(|k| (2.0 * std::f64::consts::PI * hz * k as f64 * 0.004).sin() as f32)
            .collect();
        f.filtfilt_traces_f32(&mut t, nk).unwrap();
        let mid = &t[200..800];
        (mid.iter().map(|v| (v * v) as f64).sum::<f64>() / mid.len() as f64).sqrt()
    };
    let pass = rms(15.0);
    assert!(
        (pass - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01,
        "passband {pass}"
    );
    assert!(rms(1.0) < 0.01 && rms(80.0) < 0.01);
}
