//! Post-convolution seismic filters from `datagenerator/Seismic.py`.
//!
//! - **Bandpass**: `derive_butterworth_bandpass` + `apply_butterworth_bandpass`,
//!   i.e. `scipy.signal.butter(order, [low, high], "bandpass", output="ba")`
//!   followed by `scipy.signal.filtfilt(b, a, x, method="pad")` along each
//!   trace (odd extension of `3 * max(len(a), len(b))` samples, steady-state
//!   initial conditions from `lfilter_zi`, forward + backward direct-form-II
//!   transposed passes).
//! - **Lateral filter**: `apply_lateral_filter`, i.e.
//!   `scipy.ndimage.uniform_filter(data, size=(0, n, n, 0))`: an `n`-point box
//!   mean along inline, then along crossline, each pass rounded to `f32`, with
//!   scipy's `reflect` boundary (`d c b a | a b c d | d c b a`).
//!
//! The arithmetic follows scipy/numpy operation by operation (f32 odd
//! extension, f64 filtering, f32 result); filtered traces are bit-identical to
//! the legacy output in every measured case (see `docs/filters-port.md`). The
//! box mean is evaluated as a direct windowed sum (fixed order per output
//! sample) instead of scipy's running sum, which makes it exactly
//! tile-invariant; the two can differ only by f64 rounding before the f32 cast.

use std::f64::consts::PI;

/// Complex number with numpy/glibc operation semantics (no FMA).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    /// numpy `square` for complex: `(r*r - i*i, r*i + i*r)`.
    fn square(self) -> Self {
        Self::new(
            self.re * self.re - self.im * self.im,
            self.re * self.im + self.im * self.re,
        )
    }

    /// numpy complex division (Smith's algorithm, `loops.c.src`).
    fn div(self, o: Self) -> Self {
        let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
        if br.abs() >= bi.abs() {
            if br == 0.0 && bi == 0.0 {
                return Self::new(ar / br.abs(), ai / br.abs());
            }
            let rat = bi / br;
            let scl = 1.0 / (br + bi * rat);
            Self::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
        } else {
            let rat = br / bi;
            let scl = 1.0 / (bi + br * rat);
            Self::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
        }
    }

    /// glibc `csqrt` for finite, moderately sized arguments.
    fn sqrt(self) -> Self {
        let (re, im) = (self.re, self.im);
        if im == 0.0 {
            return if re < 0.0 {
                Self::new(0.0, (-re).sqrt().copysign(im))
            } else {
                Self::new(re.sqrt().abs(), 0.0f64.copysign(im))
            };
        }
        if re == 0.0 {
            let r = (0.5 * im.abs()).sqrt();
            return Self::new(r, r.copysign(im));
        }
        let d = re.hypot(im);
        let (r, s) = if re > 0.0 {
            let r = (0.5 * (d + re)).sqrt();
            (r, 0.5 * (im / r))
        } else {
            let s = (0.5 * (d - re)).sqrt();
            ((0.5 * (im / s)).abs(), s)
        };
        Self::new(r, s.copysign(im))
    }
}

/// numpy pairwise summation (`np.sum` of a contiguous 1-D float64 array).
fn np_sum(a: &[f64]) -> f64 {
    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = 0.0;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0f64; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for (j, rj) in r.iter_mut().enumerate() {
                    *rj += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    0.0 + pairwise(a)
}

/// Errors from filter design / application.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterError {
    /// Order must be >= 1.
    ZeroOrder,
    /// Normalised corner frequencies must satisfy `0 < low < high < 1`.
    BadCorners { low: f64, high: f64 },
    /// `filtfilt` needs traces longer than its pad length.
    TraceTooShort { len: usize, padlen: usize },
    /// `lfilter_zi` precondition (`sum(a) != 0`).
    Unstable,
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroOrder => write!(f, "Butterworth order must be >= 1"),
            Self::BadCorners { low, high } => write!(
                f,
                "bandpass corners must satisfy 0 < low < high < Nyquist (normalised {low}, {high})"
            ),
            Self::TraceTooShort { len, padlen } => write!(
                f,
                "filtfilt needs traces longer than padlen={padlen} samples (got {len})"
            ),
            Self::Unstable => write!(f, "filter not stable: sum(a) == 0"),
        }
    }
}

impl std::error::Error for FilterError {}

/// Legacy `digitisation` argument of `derive_butterworth_bandpass`:
/// `apply_bandlimits` computes `dt = digi / 1000.0` and passes `dt * 1000.0`.
pub fn legacy_digitisation_ms(digi_ms: f64) -> f64 {
    (digi_ms / 1000.0) * 1000.0
}

/// IIR filter in transfer-function (`ba`) form plus its `lfilter_zi` state.
#[derive(Debug, Clone, PartialEq)]
pub struct IirFilter {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
    /// Steady-state step-response initial conditions (`scipy.signal.lfilter_zi`).
    pub zi: Vec<f64>,
}

/// `derive_butterworth_bandpass(lowcut, highcut, digitisation, order)`:
/// digital Butterworth bandpass in `ba` form, designed exactly like
/// `scipy.signal.butter` (`buttap` → `lp2bp_zpk` → `bilinear_zpk` → `zpk2tf`).
///
/// `digitisation_ms` is the sample interval in ms (see
/// [`legacy_digitisation_ms`]). The result has `2 * order + 1` coefficients.
pub fn butterworth_bandpass(
    lowcut_hz: f64,
    highcut_hz: f64,
    digitisation_ms: f64,
    order: usize,
) -> Result<IirFilter, FilterError> {
    if order == 0 {
        return Err(FilterError::ZeroOrder);
    }
    let fs = 1.0 / (digitisation_ms / 1000.0);
    let nyq = 0.5 * fs;
    let low = lowcut_hz / nyq;
    let high = highcut_hz / nyq;
    if !(low > 0.0 && low < high && high < 1.0) {
        return Err(FilterError::BadCorners { low, high });
    }
    // iirfilter: pre-warp with fs = 2.
    let fsd = 2.0;
    let warped = [
        2.0 * fsd * (PI * low / fsd).tan(),
        2.0 * fsd * (PI * high / fsd).tan(),
    ];
    let bw = warped[1] - warped[0];
    let wo = (warped[0] * warped[1]).sqrt();

    // buttap: p = -exp(1j*pi*m/(2N)), m = -N+1, -N+3, ..., N-1.
    let n = order;
    let two_n = Cx::new((2 * n) as f64, 0.0);
    let mut p = Vec::with_capacity(n);
    for idx in 0..n {
        let m = -(n as f64) + 1.0 + 2.0 * idx as f64;
        let jpi = Cx::new(0.0, PI);
        let arg = jpi.mul(Cx::new(m, 0.0)).div(two_n);
        let e = arg.re.exp();
        let (s, c) = if arg.im.abs() > f64::MIN_POSITIVE {
            (arg.im.sin(), arg.im.cos())
        } else {
            (arg.im, 1.0)
        };
        p.push(Cx::new(-(e * c), -(e * s)));
    }

    // lp2bp_zpk (no analog zeros; degree = N).
    let bwc = Cx::new(bw, 0.0);
    let half = Cx::new(2.0, 0.0);
    let wo2 = Cx::new(wo * wo, 0.0);
    let p_lp: Vec<Cx> = p.iter().map(|&pk| pk.mul(bwc).div(half)).collect();
    let roots: Vec<Cx> = p_lp.iter().map(|&q| q.square().sub(wo2).sqrt()).collect();
    let mut p_bp: Vec<Cx> = p_lp.iter().zip(&roots).map(|(&q, &r)| q.add(r)).collect();
    p_bp.extend(p_lp.iter().zip(&roots).map(|(&q, &r)| q.sub(r)));
    let k_bp = 1.0 * bw.powf(n as f64);

    // bilinear_zpk with fs = 2 (fs2 = 4): N zeros at z = 1, N at z = -1.
    let fs2 = Cx::new(4.0, 0.0);
    let p_z: Vec<Cx> = p_bp.iter().map(|&q| fs2.add(q).div(fs2.sub(q))).collect();
    let mut num = Cx::new(1.0, 0.0);
    for _ in 0..n {
        num = num.mul(fs2.sub(Cx::new(0.0, 0.0)));
    }
    let mut den = Cx::new(1.0, 0.0);
    for &q in &p_bp {
        den = den.mul(fs2.sub(q));
    }
    let k_z = k_bp * num.div(den).re;

    // zpk2tf: b = k * poly(z); poly((x-1)^N (x+1)^N) has exact integer coefficients.
    let mut zpoly = vec![1.0f64];
    for root in std::iter::repeat_n(1.0f64, n).chain(std::iter::repeat_n(-1.0f64, n)) {
        let mut next = vec![0.0f64; zpoly.len() + 1];
        for (i, &c) in zpoly.iter().enumerate() {
            next[i] += c;
            next[i + 1] += -root * c;
        }
        zpoly = next;
    }
    let b: Vec<f64> = zpoly.iter().map(|&c| k_z * c).collect();

    // a = poly(p_z): repeated convolution with [1, -r] (BLAS zdotu order).
    let mut apoly = vec![Cx::new(1.0, 0.0)];
    for &r in &p_z {
        let nr = Cx::new(-r.re, -r.im);
        let len = apoly.len();
        let mut next = Vec::with_capacity(len + 1);
        next.push(apoly[0]);
        for i in 1..len {
            let (x0, x1) = (apoly[i - 1], apoly[i]);
            let d0 = x0.re * nr.re + x1.re;
            let d1 = x0.im * nr.im + x1.im * 0.0;
            let d2 = x0.re * nr.im + x1.re * 0.0;
            let d3 = x0.im * nr.re + x1.im;
            next.push(Cx::new(d0 - d1, d2 + d3));
        }
        let last = apoly[len - 1];
        next.push(last.mul(nr));
        apoly = next;
    }
    let a: Vec<f64> = apoly.iter().map(|c| c.re).collect();
    let zi = lfilter_zi(&b, &a)?;
    Ok(IirFilter { b, a, zi })
}

/// `scipy.signal.lfilter_zi` (closed form used by scipy >= 1.17).
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, FilterError> {
    let (mut b, mut a) = (b.to_vec(), a.to_vec());
    if a[0] != 1.0 {
        let a0 = a[0];
        b.iter_mut().for_each(|v| *v /= a0);
        a.iter_mut().for_each(|v| *v /= a0);
    }
    let sum_a = np_sum(&a);
    if sum_a == 0.0 {
        return Err(FilterError::Unstable);
    }
    let y_inf = np_sum(&b) / sum_a;
    let n = a.len().max(b.len());
    b.resize(n, 0.0);
    a.resize(n, 0.0);
    let d: Vec<f64> = (0..n).map(|k| b[k] - y_inf * a[k]).collect();
    let mut zi = vec![0.0f64; n - 1];
    let mut acc = d[n - 1];
    for k in (0..n - 1).rev() {
        zi[k] = acc;
        acc += d[k];
    }
    // zi[k] = d[n-1] + d[n-2] + ... + d[k+1], accumulated from the end.
    Ok(zi)
}

/// Direct-form-II transposed IIR pass (scipy `_linear_filter`), with state `z`
/// (`len(b) - 1` delays; `b`, `a` equal length, `a[0] == 1`).
fn lfilter_in_place(b: &[f64], a: &[f64], z: &mut [f64], x: &mut [f64]) {
    let l = b.len();
    if l == 1 {
        for v in x.iter_mut() {
            *v *= b[0];
        }
        return;
    }
    for v in x.iter_mut() {
        let xn = *v;
        let yn = z[0] + b[0] * xn;
        for n in 0..l - 2 {
            z[n] = z[n + 1] + xn * b[n + 1] - yn * a[n + 1];
        }
        z[l - 2] = xn * b[l - 1] - yn * a[l - 1];
        *v = yn;
    }
}

impl IirFilter {
    /// `filtfilt(..., padtype="odd", padlen=None)` edge length.
    pub fn padlen(&self) -> usize {
        3 * self.a.len().max(self.b.len())
    }

    fn normalised(&self) -> (Vec<f64>, Vec<f64>) {
        let a0 = self.a[0];
        let n = self.a.len().max(self.b.len());
        let mut b: Vec<f64> = self.b.iter().map(|v| v / a0).collect();
        let mut a: Vec<f64> = self.a.iter().map(|v| v / a0).collect();
        b.resize(n, 0.0);
        a.resize(n, 0.0);
        (b, a)
    }

    /// Zero-phase filter one `f32` trace like legacy
    /// `apply_butterworth_bandpass` (float32 input → float64 filtfilt →
    /// float32 store). `scratch` is reused across traces.
    pub fn filtfilt_f32(
        &self,
        trace: &mut [f32],
        scratch: &mut Vec<f64>,
    ) -> Result<(), FilterError> {
        let nk = trace.len();
        let edge = self.padlen();
        if nk <= edge {
            return Err(FilterError::TraceTooShort {
                len: nk,
                padlen: edge,
            });
        }
        let (b, a) = self.normalised();
        scratch.clear();
        scratch.reserve(nk + 2 * edge);
        // odd_ext in float32: 2*x[0] - x[edge:0:-1], x, 2*x[-1] - x[-2:-(edge+2):-1].
        let x0 = trace[0];
        let xl = trace[nk - 1];
        for m in 0..edge {
            scratch.push((2.0f32 * x0 - trace[edge - m]) as f64);
        }
        scratch.extend(trace.iter().map(|&v| v as f64));
        for m in 0..edge {
            scratch.push((2.0f32 * xl - trace[nk - 2 - m]) as f64);
        }
        let mut z: Vec<f64> = self.zi.iter().map(|&v| v * scratch[0]).collect();
        lfilter_in_place(&b, &a, &mut z, scratch);
        scratch.reverse();
        let y0 = scratch[0];
        z.iter_mut().zip(&self.zi).for_each(|(zk, &v)| *zk = v * y0);
        lfilter_in_place(&b, &a, &mut z, scratch);
        scratch.reverse();
        for (dst, &src) in trace.iter_mut().zip(&scratch[edge..edge + nk]) {
            *dst = src as f32;
        }
        Ok(())
    }

    /// Filter every trace of a C-order `(.., nk)` volume in place.
    pub fn filtfilt_traces_f32(&self, volume: &mut [f32], nk: usize) -> Result<(), FilterError> {
        let mut scratch = Vec::new();
        for trace in volume.chunks_exact_mut(nk) {
            self.filtfilt_f32(trace, &mut scratch)?;
        }
        Ok(())
    }
}

/// Cumulative sum along each trace in float32 (numpy `cumsum(axis=-1)` of a
/// float32 array), the first step of legacy `apply_cumsum`.
pub fn cumsum_traces_f32(volume: &mut [f32], nk: usize) {
    for trace in volume.chunks_exact_mut(nk) {
        let mut acc = 0.0f32;
        for (k, v) in trace.iter_mut().enumerate() {
            acc = if k == 0 { *v } else { acc + *v };
            *v = acc;
        }
    }
}

/// scipy.ndimage `reflect` boundary: `d c b a | a b c d | d c b a`.
pub fn reflect_index(i: isize, n: usize) -> usize {
    debug_assert!(n > 0);
    let n = n as isize;
    let m = i.rem_euclid(2 * n);
    (if m >= n { 2 * n - 1 - m } else { m }) as usize
}

/// Box-filter window `[i - before, i + after]` for scipy `uniform_filter1d`
/// (`size1 = size / 2`, `size2 = size - size1 - 1`, origin 0).
pub fn uniform_window(size: usize) -> (usize, usize) {
    let before = size / 2;
    (before, size - before - 1)
}

/// Half-open source range along one axis that the lateral filter reads to
/// produce outputs `[o0, o1)` on an axis of length `n` (the lateral halo,
/// including reflected boundary samples).
pub fn lateral_source_range(o0: usize, o1: usize, n: usize, size: usize) -> (usize, usize) {
    if size <= 1 || o0 >= o1 {
        return (o0, o1);
    }
    let (before, after) = uniform_window(size);
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for o in o0..o1 {
        for d in -(before as isize)..=(after as isize) {
            let s = reflect_index(o as isize + d, n);
            lo = lo.min(s);
            hi = hi.max(s + 1);
        }
    }
    (lo, hi)
}

/// Legacy lateral filter (`uniform_filter(size=(0, n, n, 0))`) evaluated for
/// output columns `[i0, i1) x [j0, j1)` of an `(ni, nj, nk)` volume.
///
/// `src` holds the source columns `[si0, si1) x [sj0, sj1)` (C order,
/// `nk` samples each), which must cover [`lateral_source_range`] on both axes.
/// Inline pass first, then crossline, each rounded to `f32`, as scipy does.
/// The result is identical for any tiling of the output.
#[allow(clippy::too_many_arguments)]
pub fn lateral_uniform_tile(
    src: &[f32],
    src_i: (usize, usize),
    src_j: (usize, usize),
    shape: [usize; 3],
    out_i: (usize, usize),
    out_j: (usize, usize),
    size: usize,
    out: &mut [f32],
) {
    let [ni, nj, nk] = shape;
    let (si0, si1) = src_i;
    let (sj0, sj1) = src_j;
    let (i0, i1) = out_i;
    let (j0, j1) = out_j;
    let stj = sj1 - sj0;
    let (ti, tj) = (i1 - i0, j1 - j0);
    assert_eq!(src.len(), (si1 - si0) * stj * nk, "source tile size");
    assert_eq!(out.len(), ti * tj * nk, "output tile size");
    if size <= 1 {
        for i in i0..i1 {
            for j in j0..j1 {
                let s = ((i - si0) * stj + (j - sj0)) * nk;
                let d = ((i - i0) * tj + (j - j0)) * nk;
                out[d..d + nk].copy_from_slice(&src[s..s + nk]);
            }
        }
        return;
    }
    let (before, after) = uniform_window(size);
    let taps = |c: usize, n: usize, lo: usize, hi: usize| -> Vec<usize> {
        (-(before as isize)..=(after as isize))
            .map(|d| {
                let s = reflect_index(c as isize + d, n);
                assert!(s >= lo && s < hi, "lateral halo does not cover index {s}");
                s
            })
            .collect()
    };
    let inv = size as f64;
    let mut acc = vec![0.0f64; nk];
    // Pass 1 (inline): rows [i0, i1), source columns [sj0, sj1).
    let mut mid = vec![0.0f32; ti * stj * nk];
    for i in i0..i1 {
        let ii = taps(i, ni, si0, si1);
        for j in sj0..sj1 {
            acc.iter_mut().for_each(|v| *v = 0.0);
            for &s in &ii {
                let base = ((s - si0) * stj + (j - sj0)) * nk;
                for (a, &v) in acc.iter_mut().zip(&src[base..base + nk]) {
                    *a += v as f64;
                }
            }
            let d = ((i - i0) * stj + (j - sj0)) * nk;
            for (m, &a) in mid[d..d + nk].iter_mut().zip(&acc) {
                *m = (a / inv) as f32;
            }
        }
    }
    // Pass 2 (crossline).
    for j in j0..j1 {
        let jj = taps(j, nj, sj0, sj1);
        for i in i0..i1 {
            acc.iter_mut().for_each(|v| *v = 0.0);
            for &s in &jj {
                let base = ((i - i0) * stj + (s - sj0)) * nk;
                for (a, &v) in acc.iter_mut().zip(&mid[base..base + nk]) {
                    *a += v as f64;
                }
            }
            let d = ((i - i0) * tj + (j - j0)) * nk;
            for (o, &a) in out[d..d + nk].iter_mut().zip(&acc) {
                *o = (a / inv) as f32;
            }
        }
    }
}

/// Whole-volume lateral filter (reference / small cubes).
pub fn lateral_uniform_volume(volume: &[f32], shape: [usize; 3], size: usize) -> Vec<f32> {
    let [ni, nj, _] = shape;
    let mut out = vec![0.0f32; volume.len()];
    lateral_uniform_tile(
        volume,
        (0, ni),
        (0, nj),
        shape,
        (0, ni),
        (0, nj),
        size,
        &mut out,
    );
    out
}
