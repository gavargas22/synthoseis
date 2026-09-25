//! SNR / Hilterman noise-weight helpers from `Seismic.add_weighted_noise`
//! and `tests/test_seismic_noise.py`.

/// `std_ratio = sqrt(10 ** (sn_db / 10))` used to scale noise vs signal std.
pub fn snr_std_ratio(sn_db: f64) -> f64 {
    (10.0_f64).powf(sn_db / 10.0).sqrt()
}

/// Hilterman near/far weights for an incident angle in **degrees**.
///
/// Returns `(cos²(θ), sin²(θ))` with θ in radians — see
/// `tests/test_seismic_noise.py`.
pub fn hilterman_noise_weights(angle_deg: f64) -> (f64, f64) {
    let th = angle_deg.to_radians();
    let c = th.cos();
    let s = th.sin();
    (c * c, s * s)
}

/// Legacy `Seismic.add_weighted_noise` weights: `math.cos(ang) ** 2` /
/// `math.sin(ang) ** 2` with the angle in **degrees** passed straight to the
/// radian trig functions (the bug `tests/test_seismic_noise.py` documents).
/// Only used when replicating the legacy noise mix exactly.
pub fn legacy_degree_noise_weights(angle_deg: f64) -> (f64, f64) {
    let c = angle_deg.cos();
    let s = angle_deg.sin();
    (c * c, s * s)
}

/// Population std of `w0 * n0 + w45 * n45` for independent unit-scale
/// Laplace `n0`, `n45` (variance 2 each): `sqrt(2 (w0² + w45²))`.
pub fn weighted_laplace_std(w0: f64, w45: f64) -> f64 {
    (2.0 * (w0 * w0 + w45 * w45)).sqrt()
}

/// Philox4x32-10 counter-based RNG block (Salmon et al. 2011, Random123).
///
/// A pure function of `(counter, key)`: every voxel draws from its own
/// counter, so the stream is independent of evaluation order, tiling and
/// worker split.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let mut c = counter;
    let mut k = key;
    for round in 0..10 {
        if round > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        let p0 = M0 * c[0] as u64;
        let p1 = M1 * c[2] as u64;
        c = [
            ((p1 >> 32) as u32) ^ c[1] ^ k[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ c[3] ^ k[1],
            p0 as u32,
        ];
    }
    c
}

/// SplitMix64 finaliser (used to spread a user seed over the Philox key).
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Domain salt so noise keys never collide with other seeded streams.
const NOISE_KEY_SALT: u64 = 0x6E6F_6973_655F_7631; // "noise_v1"

/// Philox key for noise seed `seed`.
pub fn noise_key(seed: u64) -> [u32; 2] {
    let z = splitmix64(seed ^ NOISE_KEY_SALT);
    [z as u32, (z >> 32) as u32]
}

/// Unit-scale Laplace variate from 64 random bits: `-ln(u)` with
/// `u = (m + 1) / 2^53 ∈ (0, 1]` from the top 53 bits, sign from bit 0.
/// Legacy: `exponential(1/100) * (±1 from binomial(1, 0.5))` (scale 0.01;
/// the scale cancels in the std normalisation).
#[inline]
fn laplace_from_bits(x: u64) -> f64 {
    let u = ((x >> 11) + 1) as f64 * (1.0 / 9_007_199_254_740_992.0);
    let e = -u.ln();
    if x & 1 == 0 {
        -e
    } else {
        e
    }
}

/// The two independent unit Laplace draws `(n0, n45)` for global voxel
/// index `g` (one Philox block per voxel; counter = `g`).
#[inline]
pub fn laplace_pair(key: [u32; 2], g: u64) -> (f64, f64) {
    let r = philox4x32_10([g as u32, (g >> 32) as u32, 0, 0], key);
    let a = (r[0] as u64) | ((r[1] as u64) << 32);
    let b = (r[2] as u64) | ((r[3] as u64) << 32);
    (laplace_from_bits(a), laplace_from_bits(b))
}

/// Deterministic, tiling-invariant replacement for legacy
/// `add_weighted_noise` at one incidence angle.
///
/// `noise(g) = f32((w0 n0(g) + w45 n45(g)) * scale)` with
/// `scale = data_std / (weighted_laplace_std(w0, w45) * std_ratio)`, where
/// legacy divides by the sample std of the weighted cube instead (identical
/// in expectation; relative difference `O(1/sqrt(N))`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedNoise {
    pub key: [u32; 2],
    pub w0: f64,
    pub w45: f64,
    pub scale: f64,
}

impl WeightedNoise {
    /// Noise for `seed`, weights `(w0, w45)`, signal std `data_std` and
    /// signal-to-noise ratio `sn_db` (dB).
    pub fn new(seed: u64, w0: f64, w45: f64, data_std: f64, sn_db: f64) -> Self {
        let denom = weighted_laplace_std(w0, w45) * snr_std_ratio(sn_db);
        Self {
            key: noise_key(seed),
            w0,
            w45,
            scale: if denom > 0.0 { data_std / denom } else { 0.0 },
        }
    }

    /// Noise sample for global voxel index `g = (i * nj + j) * nk + k`.
    #[inline]
    pub fn sample(&self, g: u64) -> f32 {
        let (n0, n45) = laplace_pair(self.key, g);
        ((self.w0 * n0 + self.w45 * n45) * self.scale) as f32
    }

    /// Add noise (in f32, like legacy `noise + rfc_raw`) to a tile covering
    /// inlines `[i0, i1)` and crosslines `[j0, j1)` of a `(ni, nj, nk)` volume.
    pub fn add_to_tile(
        &self,
        tile: &mut [f32],
        (i0, i1): (usize, usize),
        (j0, j1): (usize, usize),
        shape: [usize; 3],
    ) {
        let [_ni, nj, nk] = shape;
        let tj = j1 - j0;
        assert_eq!(tile.len(), (i1 - i0) * tj * nk);
        for (di, i) in (i0..i1).enumerate() {
            for (dj, j) in (j0..j1).enumerate() {
                let g0 = ((i * nj + j) * nk) as u64;
                let t = &mut tile[(di * tj + dj) * nk..][..nk];
                for (k, v) in t.iter_mut().enumerate() {
                    *v += self.sample(g0 + k as u64);
                }
            }
        }
    }
}

/// Streaming mean / variance (Welford, merged with Chan et al.) in f64.
///
/// Feeding the same values in the same order always gives the same bits;
/// callers reduce in a fixed global order so the result does not depend on
/// chunking or worker count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
}

impl RunningStats {
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let d = x - self.mean;
        self.mean += d / self.count as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Chan et al. pairwise merge (`self` first, then `other`).
    pub fn merge(&mut self, other: &RunningStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let n = self.count + other.count;
        let d = other.mean - self.mean;
        let nf = n as f64;
        self.mean += d * other.count as f64 / nf;
        self.m2 += other.m2 + d * d * (self.count as f64) * (other.count as f64) / nf;
        self.count = n;
    }

    /// Population std (`numpy.std`, `ddof = 0`).
    pub fn std(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.m2 / self.count as f64).sqrt()
        }
    }
}

/// Legacy normalisation mask threshold: samples `k >= wb / (digi + 15) * digi`
/// with `wb = seabed_samples * digi` (legacy depth maps are in `digi` units).
/// Replicates the legacy expression verbatim; its name (`wb_plus_15samples`)
/// suggests `wb / digi + 15` was intended.
pub fn legacy_noise_mask_threshold(seabed_samples: f64, digi: f64) -> f64 {
    seabed_samples * digi / (digi + 15.0) * digi
}

/// Normalisation mask threshold in samples: the seabed itself (`k >=
/// seabed_samples`, only sub-seabed reflectivity) by default, or the exact
/// legacy expression ([`legacy_noise_mask_threshold`], `~0.84 x` the seabed
/// sample for digi = 4, i.e. partly in the water column) when `legacy`.
pub fn noise_mask_threshold(seabed_samples: f64, digi: f64, legacy: bool) -> f64 {
    if legacy {
        legacy_noise_mask_threshold(seabed_samples, digi)
    } else {
        seabed_samples
    }
}
