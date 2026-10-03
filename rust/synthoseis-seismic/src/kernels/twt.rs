//! Depth-to-time conversion kernels (spec "depth-to-time conversion", §1–§3;
//! PR A: the conversion core, not yet wired into any pipeline path).
//!
//! * [`twt_column`]: per-column two-way vertical time from the final Vp,
//!   `T_0 = 0`, `T_{k+1} = T_k + 2·dz / Vp_k`, summed in f64 in order (ms).
//! * [`KaiserSinc`]: the band-limited interpolation kernel (Kaiser-windowed
//!   sinc, half-width M = 8, β = 6.5, cut off at the output Nyquist,
//!   interpolating), tabulated at P = 512 phases with linear interpolation
//!   between phases (Smith & Gossett 1984; Kaiser & Schafer 1980).
//! * [`insert_spikes`]: band-limited insertion of depth-interface
//!   reflectivity at its exact sub-sample time (`sinc`, or the 2-tap
//!   [`linear_split`] option).
//! * [`point_sample_labels`]: categorical resampling; every output sample
//!   takes the label of the depth cell whose time interval contains it.
//! * [`reflectivity_time_column`]: steps 2–4 of the spec's per-column chain
//!   (time, Zoeppritz in depth, insertion), for one column and one angle.
//!
//! Everything here is a pure function of one column and global,
//! config-deterministic constants (class A: no lateral or vertical halo).

use std::sync::OnceLock;

use super::zoeppritz::{zoeppritz_pp_form, ZoeppritzForm};

/// The velocity the legacy axis implies (m/s): one sample is both `dz` = 4 m
/// and `dt` = 4 ms, i.e. a constant-velocity conversion at 2 × 4 m / 4 ms.
pub const TWT_REFERENCE_VELOCITY_M_S: f64 = 2000.0;

/// Kernel half-width M (output samples on each side).
pub const SINC_HALF_WIDTH: usize = 8;
/// Kaiser window β.
pub const SINC_KAISER_BETA: f64 = 6.5;
/// Number of tabulated fractional phases P.
pub const SINC_PHASES: usize = 512;

/// Resampling kernel for [`insert_spikes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TwtKernel {
    /// Kaiser-windowed sinc (M = 8, β = 6.5, P = 512). Default.
    #[default]
    Sinc,
    /// 2-tap linear spike splitting: fast, documented option; aliases above
    /// about 0.4·f_N (spec §3.3), so only for the bandpass-only chain.
    Linear,
}

impl TwtKernel {
    /// CLI / attribute name (`sinc` or `linear`).
    pub fn as_str(self) -> &'static str {
        match self {
            TwtKernel::Sinc => "sinc",
            TwtKernel::Linear => "linear",
        }
    }

    /// Parse the CLI / attribute name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sinc" => Some(TwtKernel::Sinc),
            "linear" => Some(TwtKernel::Linear),
            _ => None,
        }
    }
}

/// Two-way vertical time (ms) at the top of every depth cell and at the base.
///
/// `vp` is one column's final P velocity (m/s, one value per depth cell of
/// size `dz` m); `out` receives `vp.len() + 1` times, `out[0] = 0` (datum at
/// the cube top) and `out[k + 1] = out[k] + 2·dz / vp[k]`, in f64 and in
/// order. The increment is evaluated as `2000·dz / vp` (ms), so a uniform
/// 2000 m/s column with dz = 4 m gives exactly `4k` ms.
pub fn twt_column(vp: &[f32], dz: f64, out: &mut [f64]) {
    assert_eq!(out.len(), vp.len() + 1, "twt_column: out must have nz + 1 entries");
    let num = 2000.0 * dz;
    let mut t = 0.0f64;
    out[0] = 0.0;
    for (k, &v) in vp.iter().enumerate() {
        t += num / v as f64;
        out[k + 1] = t;
    }
}

/// Default output length `nt₀ = round(nz · (2·dz / 2000 m/s) / dt)`
/// (= `nz` at dz = 4 m, dt = 4 ms): the legacy-axis length, a pure function
/// of the config.
pub fn default_twt_samples(nz: usize, dz: f64, dt_ms: f64) -> usize {
    let t_ms = nz as f64 * 2000.0 * dz / TWT_REFERENCE_VELOCITY_M_S;
    (t_ms / dt_ms).round() as usize
}

/// Zeroth-order modified Bessel function I0 (power series, f64).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
    loop {
        term *= q / (k * k);
        sum += term;
        if term < sum * 1e-17 {
            return sum;
        }
        k += 1.0;
    }
}

/// Tabulated Kaiser-windowed sinc interpolation kernel.
///
/// `h(x) = sinc(x) · I0(β·sqrt(1 − (x/M)²)) / I0(β)` for |x| < M, 0
/// otherwise; `sinc(x) = sin(πx)/(πx)` in output samples (cut-off at the
/// output Nyquist). `h(0) = 1` and `h(m) = 0` at every other integer, stored
/// exactly, so the kernel interpolates.
///
/// Row `p` (0 ≤ p ≤ P) holds the 2M taps for fractional position `f = p/P`:
/// tap `j` weights output offset `m = j − (M − 1)` from `floor(T/dt)`, i.e.
/// `h(m − f)`. The table is a pure function of (M, β, P).
#[derive(Debug, Clone)]
pub struct KaiserSinc {
    half_width: usize,
    beta: f64,
    phases: usize,
    table: Vec<f64>,
}

impl KaiserSinc {
    /// Build the table for half-width `m`, window `beta` and `p` phases.
    pub fn new(m: usize, beta: f64, p: usize) -> Self {
        assert!(m >= 1 && p >= 1);
        let taps = 2 * m;
        let i0b = bessel_i0(beta);
        let mut table = vec![0.0f64; (p + 1) * taps];
        for row in 0..=p {
            let f = row as f64 / p as f64;
            for j in 0..taps {
                let off = j as f64 - (m as f64 - 1.0);
                let x = off - f;
                table[row * taps + j] = Self::eval(x, m as f64, beta, i0b);
            }
        }
        // Exact interpolation property at integer positions.
        for (row, shift) in [(0usize, 0usize), (p, 1usize)] {
            for j in 0..taps {
                table[row * taps + j] = if j == m - 1 + shift { 1.0 } else { 0.0 };
            }
        }
        KaiserSinc { half_width: m, beta, phases: p, table }
    }

    fn eval(x: f64, m: f64, beta: f64, i0b: f64) -> f64 {
        if x.abs() >= m {
            return 0.0;
        }
        let sinc = if x == 0.0 {
            1.0
        } else {
            let px = std::f64::consts::PI * x;
            px.sin() / px
        };
        let r = x / m;
        sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b
    }

    /// Continuous kernel value `h(x)` (x in output samples), untabulated.
    pub fn value(&self, x: f64) -> f64 {
        if x == x.round() {
            return if x == 0.0 { 1.0 } else { 0.0 };
        }
        Self::eval(x, self.half_width as f64, self.beta, bessel_i0(self.beta))
    }

    /// Half-width M.
    pub fn half_width(&self) -> usize {
        self.half_width
    }

    /// Number of phases P.
    pub fn phases(&self) -> usize {
        self.phases
    }

    /// Kaiser β.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Table size in bytes ((P + 1) · 2M · 8).
    pub fn table_bytes(&self) -> usize {
        self.table.len() * 8
    }

    /// The 2M interpolated taps for fractional position `f` in [0, 1).
    #[inline]
    pub fn taps(&self, f: f64, out: &mut [f64]) {
        let taps = 2 * self.half_width;
        debug_assert_eq!(out.len(), taps);
        let u = f * self.phases as f64;
        let q = (u.floor() as usize).min(self.phases - 1);
        let g = u - q as f64;
        let a = &self.table[q * taps..(q + 1) * taps];
        let b = &self.table[(q + 1) * taps..(q + 2) * taps];
        for j in 0..taps {
            out[j] = (1.0 - g) * a[j] + g * b[j];
        }
    }

    /// The spec kernel (M = 8, β = 6.5, P = 512), built once per process.
    pub fn standard() -> &'static KaiserSinc {
        static K: OnceLock<KaiserSinc> = OnceLock::new();
        K.get_or_init(|| KaiserSinc::new(SINC_HALF_WIDTH, SINC_KAISER_BETA, SINC_PHASES))
    }
}

/// Add `r · h(n − u)` into `x` for the 2-tap linear kernel (`h` = triangle).
#[inline]
pub fn linear_split(r: f64, u: f64, x: &mut [f64]) {
    let n0 = u.floor();
    let f = u - n0;
    let n0 = n0 as i64;
    let nt = x.len() as i64;
    if (0..nt).contains(&n0) {
        x[n0 as usize] += r * (1.0 - f);
    }
    if f != 0.0 && (0..nt).contains(&(n0 + 1)) {
        x[(n0 + 1) as usize] += r * f;
    }
}

/// Band-limited insertion of interface reflectivity into a time trace.
///
/// `x[n] += r[k] · h(n − t_iface[k] / dt_ms)` for every interface `k`, in
/// order, with `h` the chosen [`TwtKernel`]. `x` is the output time trace
/// (`t_n = n·dt`, n = 0 … nt−1); taps that fall outside it are dropped.
/// Interfaces later than `t_{nt−1} + M·dt` are skipped (their taps would all
/// fall outside); interfaces in the last M samples still contribute their
/// band-limited tails (spec §2, long columns). An exactly integer position is
/// inserted as a delta (`x[n] += r`), so a uniform 2000 m/s column is
/// bit-exact.
pub fn insert_spikes(r: &[f32], t_iface: &[f64], dt_ms: f64, kernel: TwtKernel, x: &mut [f64]) {
    assert_eq!(r.len(), t_iface.len());
    let nt = x.len() as i64;
    if nt == 0 {
        return;
    }
    let ks = KaiserSinc::standard();
    let m = ks.half_width() as i64;
    let mut taps = [0.0f64; 2 * SINC_HALF_WIDTH];
    for (&rk, &t) in r.iter().zip(t_iface) {
        if rk == 0.0 {
            continue;
        }
        let u = t / dt_ms;
        if u > (nt - 1 + m) as f64 {
            // Times are non-decreasing along a column.
            break;
        }
        let rk = rk as f64;
        let n0f = u.floor();
        let f = u - n0f;
        let n0 = n0f as i64;
        if f == 0.0 {
            if (0..nt).contains(&n0) {
                x[n0 as usize] += rk;
            }
            continue;
        }
        match kernel {
            TwtKernel::Linear => linear_split(rk, u, x),
            TwtKernel::Sinc => {
                ks.taps(f, &mut taps);
                let first = n0 - (m - 1);
                for (j, &w) in taps.iter().enumerate() {
                    let n = first + j as i64;
                    if (0..nt).contains(&n) {
                        x[n as usize] += rk * w;
                    }
                }
            }
        }
    }
}

/// Categorical resampling by point sampling (spec §3.4).
///
/// `t_cells[k]` is the time at the top of depth cell k (`T_k`, at least
/// `labels_z.len()` entries, non-decreasing; the `nz + 1` output of
/// [`twt_column`] works). Output sample n (`t_n = n·dt`) takes
/// `labels_z[k(n)]` with `k(n) = max{k : T_k ≤ t_n}`, clamped to `nz − 1`
/// (short columns forward-fill the last cell). One two-pointer merge,
/// O(nz + nt). Never invents a class; identity at uniform 2000 m/s.
pub fn point_sample_labels<L: Copy>(labels_z: &[L], t_cells: &[f64], dt_ms: f64, out: &mut [L]) {
    let nz = labels_z.len();
    assert!(nz >= 1 && t_cells.len() >= nz);
    let mut k = 0usize;
    for (n, o) in out.iter_mut().enumerate() {
        let t = n as f64 * dt_ms;
        while k + 1 < nz && t_cells[k + 1] <= t {
            k += 1;
        }
        *o = labels_z[k];
    }
}

/// Per-column scratch for [`reflectivity_time_column`].
#[derive(Debug, Default, Clone)]
pub struct TwtScratch {
    /// `T_0 … T_nz` (ms).
    pub t: Vec<f64>,
    /// Depth reflectivity `r_0 … r_{nz−2}`.
    pub r: Vec<f32>,
}

/// Time-domain raw reflectivity of one column at one angle (spec §3.1,
/// steps 2–4): cumulative time from `vp`, Zoeppritz on the depth interfaces
/// (`r_k` between cells k and k+1, the same kernel and form as the depth
/// fuse), then insertion at `T_{k+1}`. `x` (the time trace, `nt` samples) is
/// overwritten. Below `T_nz` the model is a half-space: no reflectivity.
#[allow(clippy::too_many_arguments)]
pub fn reflectivity_time_column(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    dz: f64,
    angle_deg: f64,
    form: ZoeppritzForm,
    dt_ms: f64,
    kernel: TwtKernel,
    scratch: &mut TwtScratch,
    x: &mut [f64],
) {
    let nz = vp.len();
    assert!(nz >= 1 && vs.len() == nz && rho.len() == nz);
    scratch.t.resize(nz + 1, 0.0);
    twt_column(vp, dz, &mut scratch.t);
    let t = std::mem::take(&mut scratch.t);
    reflectivity_time_column_with_twt(vp, vs, rho, &t, angle_deg, form, dt_ms, kernel, scratch, x);
    scratch.t = t;
}

/// [`reflectivity_time_column`] with the column's times `t` (`T_0 … T_nz`,
/// ms) given instead of computed from `vp` (`scratch.t` is not used). The
/// pipeline's constant-velocity test hook builds `t` from a fixed velocity
/// while Zoeppritz still sees the voxel properties.
#[allow(clippy::too_many_arguments)]
pub fn reflectivity_time_column_with_twt(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    t: &[f64],
    angle_deg: f64,
    form: ZoeppritzForm,
    dt_ms: f64,
    kernel: TwtKernel,
    scratch: &mut TwtScratch,
    x: &mut [f64],
) {
    let nz = vp.len();
    assert!(nz >= 1 && vs.len() == nz && rho.len() == nz && t.len() == nz + 1);
    scratch.r.clear();
    for k in 0..nz.saturating_sub(1) {
        scratch.r.push(zoeppritz_pp_form(
            vp[k] as f64,
            vs[k] as f64,
            rho[k] as f64,
            vp[k + 1] as f64,
            vs[k + 1] as f64,
            rho[k + 1] as f64,
            angle_deg,
            form,
        ));
    }
    x.iter_mut().for_each(|v| *v = 0.0);
    let n_if = scratch.r.len();
    insert_spikes(&scratch.r, &t[1..1 + n_if], dt_ms, kernel, x);
}

/// Output-Nyquist check (spec §3.3, constraint 1): the highest signal
/// frequency must sit in the kernel's flat passband, `f_hi ≤ 0.4 / dt`.
pub fn output_nyquist_ok(f_hi_hz: f64, dt_ms: f64) -> bool {
    f_hi_hz <= 0.4 / (dt_ms / 1000.0) + 1e-9
}

/// Depth-staircase check (spec §3.3, constraint 2): the per-cell spike comb
/// at `Vp/(2·dz)` must fall in the kernel stopband, `dz ≤ Vp_min · dt / 1.2`.
pub fn depth_staircase_ok(dz: f64, vp_min_sediment: f64, dt_ms: f64) -> bool {
    dz <= vp_min_sediment * (dt_ms / 1000.0) / 1.2
}
