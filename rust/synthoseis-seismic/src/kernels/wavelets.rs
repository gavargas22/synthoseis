//! Wavelet + convolution helpers from `datagenerator/wavelets.py` and
//! `datagenerator/Seismic.apply_wavelet`.

/// Hanning taper at ends of `arr`; center `pct_flat` fraction stays 1.0.
///
/// Port of `datagenerator.wavelets.hanflat`.
pub fn hanflat(arr: &[f64], pct_flat: f64) -> Vec<f64> {
    let n = arr.len();
    if n == 0 {
        return Vec::new();
    }
    let low = ((n as f64) * (1.0 - pct_flat) / 2.0).round() as usize;
    let hi = n - low;
    // Length of the Hanning window covering both tapered ends.
    let han_len = n - (hi - low);
    let han = hanning(han_len);
    let mut out = vec![1.0f64; n];
    out[..low].copy_from_slice(&han[..low]);
    let right = &han[han_len - (n - hi)..];
    out[hi..].copy_from_slice(right);
    out
}

fn hanning(n: usize) -> Vec<f64> {
    // Match `numpy.hanning(n)`: 0.5 - 0.5*cos(2*pi i / (n-1)) for n > 1.
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / denom).cos())
        .collect()
}

/// Ricker wavelet samples (after optional self-convolutions + hanflat 0.50).
///
/// Port of `datagenerator.wavelets.ricker`. Returns only the amplitude series
/// (time axis discarded after self-convolve, matching production use).
pub fn ricker(f_hz: f64, dt_ms: f64, convolutions: usize) -> Vec<f64> {
    assert!(f_hz > 0.0 && dt_ms > 0.0);
    let convs = convolutions.max(1);
    let mut lenhalf = 1250.0 / f_hz;
    let halfsmp = (lenhalf / dt_ms) as i64 + 1;
    lenhalf = halfsmp as f64 * dt_ms;
    // t in seconds for the analytic Ricker, then s built from that.
    let mut t_sec: Vec<f64> = Vec::new();
    let mut t = -lenhalf;
    while t <= lenhalf + dt_ms * 0.5 {
        // np.arange(-lenhalf, lenhalf + dt, dt) then / 1000
        t_sec.push(t / 1000.0);
        t += dt_ms;
    }
    // Guard floating endpoint (match np.arange inclusivity approximately).
    // Rebuild exactly like numpy arange:
    t_sec.clear();
    let mut v = -lenhalf;
    let end = lenhalf + dt_ms;
    while v < end - 1e-12 {
        t_sec.push(v / 1000.0);
        v += dt_ms;
    }

    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    let mut s: Vec<f64> = t_sec
        .iter()
        .map(|&tt| {
            let ft2 = f_hz * f_hz * tt * tt;
            (1.0 - 2.0 * pi2 * ft2) * (-pi2 * ft2).exp()
        })
        .collect();

    for _ in 0..(convs - 1) {
        s = convolve_full(&s, &s);
    }
    let w = hanflat(&s, 0.50);
    s.iter().zip(w.iter()).map(|(a, b)| a * b).collect()
}

fn convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let mut out = vec![0.0f64; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// 1-D convolution, `mode="same"` (output length == `signal.len()`).
pub fn convolve_same_1d(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    if kernel.is_empty() {
        return signal.to_vec();
    }
    let full = convolve_full(signal, kernel);
    let start = (kernel.len() - 1) / 2;
    full[start..start + signal.len()].to_vec()
}

/// Apply a 1-D wavelet along the last axis of a flattened `(il, xl, z)` cube.
///
/// Port of `datagenerator.Seismic.apply_wavelet` (per-trace same-mode convolve).
pub fn apply_wavelet_traces(
    cube: &[f32],
    shape: [usize; 3],
    wavelet: &[f64],
) -> Vec<f32> {
    let [il, xl, z] = shape;
    assert_eq!(cube.len(), il * xl * z);
    let mut out = vec![0.0f32; cube.len()];
    let mut trace = vec![0.0f64; z];
    for t in 0..(il * xl) {
        let base = t * z;
        for k in 0..z {
            trace[k] = cube[base + k] as f64;
        }
        let conv = convolve_same_1d(&trace, wavelet);
        for k in 0..z {
            out[base + k] = conv[k] as f32;
        }
    }
    out
}
