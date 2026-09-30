//! Depth-to-time conversion core: the analytic tests PR A owns (spec
//! "depth-to-time conversion", §5.1, §5.2, §5.3 synthetic part, §5.6), plus
//! kernel properties. Nothing here touches a pipeline path.

use synthoseis_seismic::{
    convolve_same_1d, default_twt_samples, depth_staircase_ok, insert_spikes, output_nyquist_ok,
    point_sample_labels, reflectivity_time_column, ricker, twt_column, zoeppritz_pp_form,
    KaiserSinc, TwtKernel, TwtScratch, ZoeppritzForm, SINC_HALF_WIDTH,
};

const DZ: f64 = 4.0;
const DT: f64 = 4.0;

/// Deterministic xorshift64* in [0, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A layered column with slowly varying properties (every interface
/// reflects), `nz` cells at velocity `v` scaled per cell.
fn column(nz: usize, v: impl Fn(usize) -> f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let vp: Vec<f32> = (0..nz).map(&v).collect();
    let vs: Vec<f32> = (0..nz).map(|k| 0.5 * v(k) + (k % 7) as f32).collect();
    let rho: Vec<f32> = (0..nz).map(|k| 2.0 + 0.01 * ((k * 13) % 17) as f32).collect();
    (vp, vs, rho)
}

// ---------------------------------------------------------------- §5.1 ----

/// V = 2000 m/s, dz = 4, dt = 4: `T_k = 4k` exactly, the label resample is
/// the identity and the inserted reflectivity is the depth reflectivity
/// shifted down one sample, bit for bit (both kernels).
#[test]
fn constant_2000_is_the_legacy_axis_shifted_one_sample() {
    let nz = 256;
    let mut t = vec![0.0; nz + 1];
    twt_column(&vec![2000.0f32; nz], DZ, &mut t);
    for (k, &tk) in t.iter().enumerate() {
        assert_eq!(tk, 4.0 * k as f64, "T_{k}");
    }
    assert_eq!(default_twt_samples(nz, DZ, DT), nz);

    let labels: Vec<u8> = (0..nz).map(|k| (k * 37 % 251) as u8).collect();
    let mut lt = vec![0u8; nz];
    point_sample_labels(&labels, &t, DT, &mut lt);
    assert_eq!(lt, labels);

    // Properties vary but Vp stays 2000 (Vs and rho carry the contrasts).
    let (_, vs, rho) = column(nz, |_| 2000.0);
    let vp = vec![2000.0f32; nz];
    for kernel in [TwtKernel::Sinc, TwtKernel::Linear] {
        for angle in [0.0, 15.0, 30.0] {
            let mut x = vec![0.0f64; nz];
            let mut s = TwtScratch::default();
            reflectivity_time_column(&vp, &vs, &rho, DZ, angle, ZoeppritzForm::Exact, DT, kernel, &mut s, &mut x);
            assert_eq!(x[0], 0.0);
            for k in 0..nz - 1 {
                let r = zoeppritz_pp_form(
                    vp[k] as f64, vs[k] as f64, rho[k] as f64,
                    vp[k + 1] as f64, vs[k + 1] as f64, rho[k + 1] as f64,
                    angle, ZoeppritzForm::Exact,
                );
                assert!(r != 0.0 || angle == 0.0 || k % 7 == 6);
                assert_eq!(x[k + 1].to_bits(), (r as f64).to_bits(), "{kernel:?} {angle} k {k}");
            }
        }
    }
}

/// V = 3000 m/s: `T_k = 8k/3` ms to ≤ 1e-9 ms.
#[test]
fn constant_3000_times() {
    let nz = 1000;
    let mut t = vec![0.0; nz + 1];
    twt_column(&vec![3000.0f32; nz], DZ, &mut t);
    let worst = (0..=nz).map(|k| (t[k] - 8.0 * k as f64 / 3.0).abs()).fold(0.0, f64::max);
    println!("3000 m/s: max |T_k - 8k/3| = {worst:.2e} ms over {} cells", nz);
    assert!(worst <= 1e-9, "{worst}");
}

/// V = 4500 m/s with nt = nz = 256: the column is short from T = 455 ms;
/// below it there is no reflectivity (half-space) and every label
/// forward-fills the last cell.
#[test]
fn constant_4500_short_column_half_space() {
    let nz = 256;
    let vp = vec![4500.0f32; nz];
    let mut t = vec![0.0; nz + 1];
    twt_column(&vp, DZ, &mut t);
    let t_nz = t[nz];
    assert!((t_nz - 455.111).abs() < 1e-3, "T_nz = {t_nz}");
    let nt = nz;
    assert!(t_nz < (nt - 1) as f64 * DT, "short column");

    let labels: Vec<u8> = (0..nz).map(|k| (k % 200) as u8).collect();
    let mut lt = vec![0u8; nt];
    point_sample_labels(&labels, &t, DT, &mut lt);
    let first_short = (t_nz / DT).ceil() as usize;
    assert_eq!(first_short, 114);
    assert!(lt[first_short..].iter().all(|&l| l == labels[nz - 1]));
    let set: std::collections::BTreeSet<u8> = labels.iter().copied().collect();
    assert!(lt.iter().all(|l| set.contains(l)), "no invented class");

    // Reflectivity: nothing below the last interface's band-limited tail.
    let (_, vs, rho) = column(nz, |_| 4500.0);
    let mut x = vec![0.0f64; nt];
    let mut s = TwtScratch::default();
    reflectivity_time_column(&vp, &vs, &rho, DZ, 15.0, ZoeppritzForm::Exact, DT, TwtKernel::Sinc, &mut s, &mut x);
    let last_iface = t[nz - 1] / DT;
    let tail_end = last_iface.floor() as usize + SINC_HALF_WIDTH + 1;
    assert!(x[..tail_end].iter().any(|&v| v != 0.0));
    assert!(x[tail_end..].iter().all(|&v| v == 0.0), "half-space adds no reflectivity");
}

// ---------------------------------------------------------------- §5.2 ----

/// Linear gradient V(z) = V0 + kz: `t(z) = (2/k) ln(1 + kz/V0)`
/// (Sheriff & Geldart 1995). V0 = 1600 m/s, k = 0.6 1/s, dz = 4 m,
/// nz = 250; cell velocity at the cell midpoint, stored as f32.
#[test]
fn linear_gradient_closed_form() {
    let (v0, g, nz) = (1600.0f64, 0.6f64, 250usize);
    let vp: Vec<f32> = (0..nz).map(|k| (v0 + g * (k as f64 + 0.5) * DZ) as f32).collect();
    let mut t = vec![0.0; nz + 1];
    twt_column(&vp, DZ, &mut t);
    let exact = |z: f64| 2000.0 / g * (1.0 + g * z / v0).ln();
    let worst = (0..=nz).map(|k| (t[k] - exact(k as f64 * DZ)).abs()).fold(0.0, f64::max);
    println!(
        "linear gradient: T(1000 m) = {:.4} ms (closed form {:.4}), max |T_k - t(k dz)| = {worst:.2e} ms",
        t[nz],
        exact(1000.0)
    );
    assert!((exact(1000.0) - 1061.512).abs() < 1e-3);
    assert!((t[nz] - 1061.512).abs() < 1e-3);
    assert!(worst <= 1e-3, "{worst}");
}

// ---------------------------------------------------------------- §5.3 ----

/// 16× band-limited upsampling by zero-padding the spectrum (FFT
/// interpolation), done with a direct DFT in test code: forward DFT of the
/// N-sample trace, the Nyquist bin split between ±N/2, zero padding to 16·N,
/// inverse DFT. Independent of the conversion's windowed-sinc kernel, so the
/// pick does not grade the kernel with itself. Returns 16·N samples at
/// dt/16 (sample s ↔ position s/16).
fn fft_upsample_16(y: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    const UP: usize = 16;
    let n = y.len();
    assert!(n % 2 == 0);
    // Forward DFT, bins 0..=N/2 (real input: the rest are conjugates).
    let spec: Vec<(f64, f64)> = (0..=n / 2)
        .map(|k| {
            y.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, &v)| {
                let w = -2.0 * PI * (k * t) as f64 / n as f64;
                (re + v * w.cos(), im + v * w.sin())
            })
        })
        .collect();
    let m = UP * n;
    (0..m)
        .map(|s| {
            let u = s as f64 / UP as f64; // position in input samples
            let mut acc = spec[0].0;
            for (k, &(re, im)) in spec.iter().enumerate().skip(1) {
                let w = 2.0 * PI * k as f64 * u / n as f64;
                // Bins k and N−k (conjugate pair): 2·Re(Y_k e^{iw}); the
                // Nyquist bin is split in half between ±N/2.
                let scale = if k == n / 2 { 1.0 } else { 2.0 };
                acc += scale * (re * w.cos() - im * w.sin());
            }
            acc / n as f64
        })
        .collect()
}

/// Sub-sample peak pick: the positive maximum near `guess` (samples) on the
/// 16× FFT-upsampled trace, then a parabola through the three best fine
/// samples. Also returns the plain three-point parabolic pick on the coarse
/// samples.
fn pick_peak(y: &[f64], guess: f64) -> (f64, f64) {
    let lo = (guess - 6.0).max(1.0) as usize;
    let hi = ((guess + 6.0) as usize).min(y.len() - 2);
    let n = (lo..=hi).max_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap()).unwrap();
    let (a, b, c) = (y[n - 1], y[n], y[n + 1]);
    let parabolic = n as f64 + 0.5 * (a - c) / (a - 2.0 * b + c);
    let fine = fft_upsample_16(y);
    // Sanity: FFT interpolation reproduces the input at integer positions.
    for (k, &v) in y.iter().enumerate() {
        assert!((fine[16 * k] - v).abs() < 1e-12, "upsample at {k}");
    }
    let (s_lo, s_hi) = (16 * (n - 1), 16 * (n + 1));
    let s = (s_lo + 1..s_hi).max_by(|&p, &q| fine[p].partial_cmp(&fine[q]).unwrap()).unwrap();
    let (a, b, c) = (fine[s - 1], fine[s], fine[s + 1]);
    let refined = (s as f64 + 0.5 * (a - c) / (a - 2.0 * b + c)) / 16.0;
    (refined, parabolic)
}

/// Salt pull-up on synthetic columns through the time-mode column chain
/// (insertion + 40 Hz Ricker at dt): sediment 2500 m/s with a flat base
/// reflector at 800 m (step to 3000 m/s); column B has salt (4500 m/s,
/// rho 2.17) from 400 to 600 m. Predicted t_A = 640.00 ms, pull-up
/// Δt = 2h(1/V_sed − 1/V_salt) = 71.11 ms, t_B = 568.89 ms.
#[test]
fn salt_pull_up_synthetic_columns() {
    let nz = 300; // 1200 m
    let nt = 256;
    let build = |salt: bool| {
        let mut vp = vec![2500.0f32; nz];
        let mut vs = vec![1100.0f32; nz];
        let mut rho = vec![2.25f32; nz];
        for k in 200..nz {
            (vp[k], vs[k], rho[k]) = (3000.0, 1400.0, 2.35);
        }
        if salt {
            for k in 100..150 {
                (vp[k], vs[k], rho[k]) = (4500.0, 2600.0, 2.17);
            }
        }
        (vp, vs, rho)
    };
    let wav = ricker(40.0, DT, 1);
    let v_sed = 2500.0f64;
    let h = 200.0f64;
    let dt_pred = 2000.0 * h * (1.0 / v_sed - 1.0 / 4500.0);
    let pred = [640.0, 640.0 - dt_pred];
    assert!((dt_pred - 71.111).abs() < 1e-3);
    // Gate on the 16x FFT pick (Strata's independent picks: FFT 568.884 ms,
    // pull-up 71.116 ms; analytic-Ricker least-squares fit 568.889 ms).
    const GATE_MS: f64 = 0.02;
    let mut picks = [0.0f64; 2];
    let mut linear_picks = [0.0f64; 2];
    for (c, salt) in [false, true].into_iter().enumerate() {
        let (vp, vs, rho) = build(salt);
        for kernel in [TwtKernel::Sinc, TwtKernel::Linear] {
            let mut x = vec![0.0f64; nt];
            let mut s = TwtScratch::default();
            reflectivity_time_column(&vp, &vs, &rho, DZ, 0.0, ZoeppritzForm::Exact, DT, kernel, &mut s, &mut x);
            assert!((s.t[200] - pred[c]).abs() < 1e-9, "interface time T_200 (column {c}): {}", s.t[200]);
            let y = convolve_same_1d(&x, &wav);
            let (refined, parabolic) = pick_peak(&y, pred[c] / DT);
            let (tr, tp) = (refined * DT, parabolic * DT);
            println!(
                "column {} ({kernel:?}): base reflection pick: 16x FFT {tr:.3} ms, 3-point parabolic {tp:.3} ms, predicted {:.3} ms",
                if salt { "B, salt" } else { "A" },
                pred[c]
            );
            match kernel {
                TwtKernel::Sinc => {
                    assert!((tr - pred[c]).abs() <= GATE_MS, "column {c}: {tr} vs {}", pred[c]);
                    picks[c] = tr;
                }
                TwtKernel::Linear => linear_picks[c] = tr,
            }
        }
    }
    let dt_meas = picks[0] - picks[1];
    let dt_lin = linear_picks[0] - linear_picks[1];
    println!(
        "pull-up: sinc {dt_meas:.3} ms, linear {dt_lin:.3} ms, predicted {dt_pred:.3} ms (gate +/-{GATE_MS} ms)"
    );
    assert!((dt_meas - dt_pred).abs() <= GATE_MS, "{dt_meas}");
    // The gate is tight enough to tell the kernels apart: the 2-tap linear
    // split misplaces the sub-salt reflection (column A sits on a sample, so
    // only column B moves).
    let lin_err = linear_picks[1] - pred[1];
    println!("linear kernel: column B pick error {lin_err:.3} ms, pull-up error {:.3} ms", dt_lin - dt_pred);
    assert!(lin_err.abs() > GATE_MS, "linear should fail the gate: {lin_err}");
    assert!((dt_lin - dt_pred).abs() > GATE_MS, "linear pull-up should fail the gate");
}

// ---------------------------------------------------------------- §5.6 ----

fn ricker_c(f: f64, t_ms: f64) -> f64 {
    let a = (std::f64::consts::PI * f * t_ms / 1000.0).powi(2);
    (1.0 - 2.0 * a) * (-a).exp()
}

/// Thin-bed wedge: two opposite-sign spikes, time thickness 0–20 ms, random
/// sub-sample offsets (400 cases), inserted then convolved with the sampled
/// 40 Hz Ricker, against the analytically sampled continuous synthetic
/// (Widess 1973; Kallweit & Wood 1982). Metric: max |error| over the trace
/// divided by the peak of a SINGLE wavelet (the analytic Ricker peak, 1.0),
/// not by the two-spike trace's own peak (that one shrinks towards 0 as the
/// spikes merge, which inflates relative errors). `sinc` ≤ 0.5 %; `linear`
/// reported.
#[test]
fn thin_bed_wedge_tuning() {
    let (f, nt, half) = (40.0, 256usize, 40i64);
    let wav: Vec<f64> = (-half..=half).map(|n| ricker_c(f, n as f64 * DT)).collect();
    let mut rng = Rng(0x5eed_d27);
    let mut worst = [0.0f64; 2];
    let mut mean = [0.0f64; 2];
    let mut tune_err = [0.0f64; 2];
    let cases = 400;
    for case in 0..cases {
        let sep = 20.0 * case as f64 / (cases - 1) as f64;
        let t1 = 500.0 + DT * rng.next();
        let t2 = t1 + sep;
        let (a1, a2) = (1.0f32, -1.0f32);
        let reference: Vec<f64> = (0..nt)
            .map(|n| a1 as f64 * ricker_c(f, n as f64 * DT - t1) + a2 as f64 * ricker_c(f, n as f64 * DT - t2))
            .collect();
        let ref_peak = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, kernel) in [TwtKernel::Sinc, TwtKernel::Linear].into_iter().enumerate() {
            let mut x = vec![0.0f64; nt];
            insert_spikes(&[a1, a2], &[t1, t2], DT, kernel, &mut x);
            let y = convolve_same_1d(&x, &wav);
            let e = y.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            worst[i] = worst[i].max(e);
            mean[i] += e / cases as f64;
            let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            tune_err[i] = tune_err[i].max((peak - ref_peak).abs());
        }
    }
    println!(
        "wedge (400 cases, 0-20 ms): sinc max {:.3} % mean {:.3} % tuning-peak {:.3} % | linear max {:.1} % mean {:.1} % tuning-peak {:.1} %",
        100.0 * worst[0], 100.0 * mean[0], 100.0 * tune_err[0],
        100.0 * worst[1], 100.0 * mean[1], 100.0 * tune_err[1]
    );
    assert!(worst[0] <= 0.005, "sinc wedge error {}", worst[0]);
    assert!(worst[1] > worst[0]);
}

// -------------------------------------------------------- kernel checks ----

/// The kernel interpolates (h(0) = 1, h(m) = 0), is a pure function of
/// (M, β, P), sums to ~1 at every phase (DC gain), and spans 66 KB.
#[test]
fn kaiser_sinc_table_properties() {
    let k = KaiserSinc::standard();
    assert_eq!((k.half_width(), k.phases()), (8, 512));
    assert_eq!(k.table_bytes(), 513 * 16 * 8);
    let again = KaiserSinc::new(8, 6.5, 512);
    let mut a = [0.0; 16];
    let mut b = [0.0; 16];
    let mut worst_dc = 0.0f64;
    for i in 0..1000 {
        let f = i as f64 / 1000.0;
        k.taps(f, &mut a);
        again.taps(f, &mut b);
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        worst_dc = worst_dc.max((a.iter().sum::<f64>() - 1.0).abs());
    }
    k.taps(0.0, &mut a);
    assert!(a.iter().enumerate().all(|(j, &w)| w == if j == 7 { 1.0 } else { 0.0 }));
    println!("sinc DC gain deviation over phases: {worst_dc:.2e}");
    assert!(worst_dc < 5e-4, "{worst_dc}");
    assert_eq!(k.value(0.0), 1.0);
    assert_eq!(k.value(3.0), 0.0);
    assert_eq!(k.value(8.5), 0.0);
}

/// Long columns: interfaces later than t_{nt-1} + M·dt are skipped; those in
/// the last M samples still contribute their tails. Labels are truncated.
#[test]
fn long_column_truncation() {
    let nt = 64;
    // A spike just past the end contributes; one beyond the tail does not.
    let mut x = vec![0.0f64; nt];
    insert_spikes(&[1.0], &[(nt as f64 + 2.3) * DT], DT, TwtKernel::Sinc, &mut x);
    assert!(x[nt - 6..].iter().any(|&v| v != 0.0));
    let mut y = vec![0.0f64; nt];
    insert_spikes(&[1.0], &[(nt as f64 - 1.0 + 8.0 + 0.4) * DT], DT, TwtKernel::Sinc, &mut y);
    assert!(y.iter().all(|&v| v == 0.0));
    // Labels of a long column are truncated (no clamping needed).
    let nz = 100;
    let mut t = vec![0.0; nz + 1];
    twt_column(&vec![1500.0f32; nz], DZ, &mut t);
    let labels: Vec<u8> = (0..nz as u8).collect();
    let mut lt = vec![0u8; nt];
    point_sample_labels(&labels, &t, DT, &mut lt);
    assert_eq!(*lt.last().unwrap(), ((nt - 1) as f64 * DT / t[1]).floor() as u8);
}

/// Validation helpers: output Nyquist (f_hi ≤ 0.4/dt) and the depth
/// staircase (dz ≤ Vp_min·dt/1.2).
#[test]
fn nyquist_and_staircase_constraints() {
    assert!(output_nyquist_ok(100.0, 4.0)); // Ricker 40 Hz at 4 ms: at the limit
    assert!(!output_nyquist_ok(100.0, 4.5));
    assert!(output_nyquist_ok(31.0, 12.9));
    assert!(depth_staircase_ok(4.0, 1580.0, 4.0)); // limit 5.27 m
    assert!(!depth_staircase_ok(5.3, 1580.0, 4.0));
}

/// Frequency response of the tabulated kernel as the conversion uses it
/// (continuous Fourier transform of the effective kernel: table rows plus
/// linear interpolation between phases), against the untabulated closed
/// form. Gates: ≥ −0.2 dB at 0.8 f_N and ≤ −30 dB at 1.2 f_N (spec §3.3:
/// −0.1 dB and −35.6 dB).
#[test]
fn kaiser_sinc_frequency_response() {
    let k = KaiserSinc::standard();
    let m = SINC_HALF_WIDTH as f64;
    // Effective kernel h_tab(x): x = off − f with off = j − (M − 1), f ∈ [0, 1).
    let h_tab = |x: f64| -> f64 {
        if x <= -m || x >= m {
            return 0.0;
        }
        let f = (-x).rem_euclid(1.0);
        let off = x + f; // integer offset
        let j = (off + m - 1.0).round() as usize;
        let mut taps = [0.0f64; 2 * SINC_HALF_WIDTH];
        k.taps(f, &mut taps);
        taps[j]
    };
    // |H(f)| = |∫ h(x) cos(2π f x) dx| (h is even up to tabulation), f in
    // cycles/sample; f_N = 0.5. Midpoint rule at 1/2048 sample.
    let step = 1.0 / 2048.0;
    let resp = |h: &dyn Fn(f64) -> f64, f: f64| -> f64 {
        let n = (2.0 * m / step) as usize;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..n {
            let x = -m + (i as f64 + 0.5) * step;
            let v = h(x);
            let w = 2.0 * std::f64::consts::PI * f * x;
            re += v * w.cos();
            im += v * w.sin();
        }
        (re * re + im * im).sqrt() * step
    };
    let db = |a: f64| 20.0 * a.log10();
    let closed = |x: f64| k.value(x);
    let dc_tab = resp(&h_tab, 0.0);
    let mut rows = Vec::new();
    for frac in [0.0, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0] {
        let f = 0.5 * frac;
        rows.push((frac, db(resp(&h_tab, f) / dc_tab), db(resp(&closed, f) / resp(&closed, 0.0))));
    }
    for (frac, t, c) in &rows {
        println!("sinc response at {frac:.1} f_N: tabulated {t:7.2} dB, closed form {c:7.2} dB");
    }
    let at = |fr: f64| rows.iter().find(|r| r.0 == fr).unwrap().1;
    assert!(at(0.8) >= -0.2, "0.8 f_N: {} dB", at(0.8));
    assert!(at(1.2) <= -30.0, "1.2 f_N: {} dB", at(1.2));
    assert!((dc_tab - 1.0).abs() < 5e-4, "DC {dc_tab}");
}
