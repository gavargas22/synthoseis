//! Filter-edge kernels (filter-edge spec §4, §5, §7.2): the bandpass edge
//! pad from the forward-backward impulse response, the padded
//! forward-backward filter, and the edge-pad noise counter domain.
use synthoseis_seismic::{
    butterworth_bandpass, hilterman_noise_weights, legacy_digitisation_ms, pad_counter,
    window_counter, FilterError, IirFilter, WeightedNoise, EDGE_PAD_CAP,
};

fn bp(lo: f64, hi: f64, order: usize) -> IirFilter {
    butterworth_bandpass(lo, hi, legacy_digitisation_ms(4.0), order).unwrap()
}

/// §7.2: `edge_pad()` at dt 4 ms is 369 ± 2 (4–30 Hz), 421 ± 2 (3–35 Hz)
/// and 366 ± 2 (6–20 Hz) for order 4 (Strata's probe: the response falls
/// below 1e-6 of its peak after 366–421 samples).
#[test]
fn edge_pad_matches_the_probe() {
    for (lo, hi, want) in [(4.0, 30.0, 369usize), (3.0, 35.0, 421), (6.0, 20.0, 366)] {
        let p = bp(lo, hi, 4).edge_pad().unwrap();
        println!("edge_pad {lo}-{hi} Hz order 4 = {p} (probe {want} ± 2)");
        assert!(p.abs_diff(want) <= 2, "{lo}-{hi}: {p} vs {want}");
    }
}

/// §7.2: the pad does not shrink when the tolerance tightens.
#[test]
fn edge_pad_is_monotone_in_the_tolerance() {
    for f in [
        bp(4.0, 30.0, 4),
        bp(3.0, 35.0, 4),
        bp(6.0, 20.0, 2),
        bp(8.0, 60.0, 6),
    ] {
        let pads: Vec<usize> = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            .iter()
            .map(|&t| f.edge_pad_tol(t).unwrap())
            .collect();
        assert!(pads.windows(2).all(|w| w[0] <= w[1]), "{pads:?}");
        assert!(pads[0] > 0);
    }
}

/// §7.2: an unstable design (a pole outside the unit circle) never decays,
/// so the cap is hit and `edge_pad` errors; so does a pole on the circle.
#[test]
fn edge_pad_errors_on_an_unstable_design() {
    for pole in [1.01, 1.0] {
        let f = IirFilter {
            b: vec![1.0, 0.0],
            a: vec![1.0, -pole],
            zi: vec![0.0],
        };
        assert_eq!(
            f.edge_pad(),
            Err(FilterError::EdgePadCap { cap: EDGE_PAD_CAP }),
            "pole {pole}"
        );
    }
    // A slowly decaying stable pole needs more than the cap too.
    let slow = IirFilter {
        b: vec![1.0, 0.0],
        a: vec![1.0, -0.9999],
        zi: vec![0.0],
    };
    assert!(slow.edge_pad().is_err());
    let fast = IirFilter {
        b: vec![1.0, 0.0],
        a: vec![1.0, -0.5],
        zi: vec![0.0],
    };
    assert!(fast.edge_pad().unwrap() < 64);
}

/// A reflectivity-like trace: seabed spike, then sparse spikes to the end.
fn spiky(n: usize, first: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v = vec![0.0f32; n];
    for (k, x) in v.iter_mut().enumerate().skip(first) {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        *x = if k == first { 0.3 } else { (0.08 * u) as f32 };
    }
    v
}

/// §4.6: with zero reflectivity above time 0 (no top pad, zero start
/// state) and the continuation below, the padded forward-backward pass
/// equals a far longer zero-padded run (the "truth") to the f32 floor, and
/// its window does not depend on any pad past `edge_pad`.
#[test]
fn padded_filtfilt_matches_a_long_reference() {
    for f in [bp(4.0, 30.0, 4), bp(3.0, 35.0, 4), bp(6.0, 20.0, 4)] {
        let pb = f.edge_pad().unwrap();
        let nt = 128;
        // The model continues 60 samples below the window, then a
        // half-space (zero reflectivity), as in time mode.
        let mut long = spiky(nt + 2048, 12, 7);
        long[nt + 60..].iter_mut().for_each(|v| *v = 0.0);
        let mut scratch = Vec::new();
        let run = |pb: usize, scratch: &mut Vec<f64>| {
            let mut buf = long[..nt + pb].to_vec();
            f.filtfilt_padded_f32(&mut buf, 0, nt, scratch).unwrap();
            buf[..nt].to_vec()
        };
        let got = run(pb, &mut scratch);
        let truth = run(2048, &mut scratch);
        // 2000 zeros above time 0 (zi·0 = zero state: the same pass).
        let mut zeros_above = vec![0.0f32; 2000];
        zeros_above.extend_from_slice(&long);
        f.filtfilt_padded_f32(&mut zeros_above, 2000, nt, &mut scratch)
            .unwrap();
        let peak = truth.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&truth)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "pad {pb}: max |padded - long reference| = {:.2e} of peak",
            err / peak
        );
        assert!(err <= 1e-6 * peak, "pad {pb}: max err {err} vs peak {peak}");
        assert_eq!(
            &zeros_above[2000..2000 + nt],
            &truth[..],
            "zero top pad == no top pad"
        );
    }
}

/// §4.6: the window is written back, the pads are left alone, and a bad
/// window is rejected.
#[test]
fn padded_filtfilt_writes_only_the_window() {
    let f = bp(4.0, 30.0, 4);
    let mut buf = spiky(10 + 64 + 400, 3, 1);
    let before = buf.clone();
    f.filtfilt_padded_f32(&mut buf, 10, 64, &mut Vec::new())
        .unwrap();
    assert_eq!(&buf[..10], &before[..10]);
    assert_eq!(&buf[74..], &before[74..]);
    assert_ne!(&buf[10..74], &before[10..74]);
    assert!(f
        .filtfilt_padded_f32(&mut buf, 400, 100, &mut Vec::new())
        .is_err());
}

/// §9 risk 3: edge-pad counters (word 3 = 1) never meet window counters
/// (word 3 = 0) over a full cube, pad counters are distinct per (column,
/// pad index), and the pad noise has the window's scale.
#[test]
fn pad_noise_keys_never_collide_with_window_keys() {
    let (ni, nj, nt, pads) = (16usize, 12usize, 128usize, 2 * 377usize);
    let mut window = std::collections::HashSet::new();
    for g in 0..(ni * nj * nt) as u64 {
        assert!(window.insert(window_counter(g)));
    }
    let mut pad = std::collections::HashSet::new();
    for col in 0..(ni * nj) as u64 {
        for p in 0..pads as u32 {
            let c = pad_counter(col, p);
            assert!(!window.contains(&c));
            assert!(pad.insert(c));
        }
    }
    // Large global indices: the high word of g lands in word 1, never 3.
    let big = u64::MAX - 5;
    assert_eq!(window_counter(big)[3], 0);
    assert_eq!(pad_counter(big, 3)[3], 1);
    // Same distribution: the pad samples' std matches the window's.
    let (w0, w45) = hilterman_noise_weights(15.0);
    let n = WeightedNoise::new(7, w0, w45, 0.1, 12.5);
    let std = |v: &[f32]| {
        let m = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let win: Vec<f32> = (0..200_000u64).map(|g| n.sample(g)).collect();
    let padv: Vec<f32> = (0..200_000u64)
        .map(|i| n.sample_pad(i / 800, (i % 800) as u32))
        .collect();
    let (a, b) = (std(&win), std(&padv));
    assert!((a / b - 1.0).abs() < 0.02, "window std {a} vs pad std {b}");
    assert_ne!(n.sample(5).to_bits(), n.sample_pad(5, 0).to_bits());
}
