//! Fault parameters and deterministic sampling.
//!
//! Port of `Faults._fault_params_random` (random mode) and the per-fault
//! random draws inside `Faults.xyz_dis` (`sigma`, `p`, `coef`). Units follow
//! what `Faults.build_faults` actually consumes: the legacy draw is expressed
//! in *infilled* units (`infill_factor = 10`) and divided down to model
//! samples before use (`c / infill²`, `z0 / infill`, `throw / infill`).
//! [`FaultParams`] always stores the divided (model-sample) values.

/// Legacy `infill_factor` used by the Python random-mode draw.
pub const LEGACY_INFILL_FACTOR: f64 = 10.0;

/// Throw (samples) at/above which Python builds a "hockey stick" drag zone
/// (`throw >= 0.85 * high_fault_throw`, `high_fault_throw = 35`).
pub const HOCKEY_STICK_MIN_THROW: f64 = 0.85 * 35.0;

/// Valid integer throw range of the legacy throw→length lookup
/// (`np.arange(5, 35)` in `xyz_dis`).
pub const THROW_LUT_MIN: i64 = 5;
/// Upper bound (inclusive) of the legacy throw→length lookup.
pub const THROW_LUT_MAX: i64 = 34;

/// Explicit parameters for one fault, in model-sample units.
///
/// `a`, `b`, `c` are **squared** semi-axes of the fault ellipsoid (the Python
/// code stores squares and prints `sqrt`). `center` is the voxel on the fault
/// surface where the maximum displacement is hung (`z_idx` in Python). When it
/// is `None`, [`crate::faults::FaultModel::resolve`] picks it with a port of
/// `get_fault_centre` / `get_middle_z` and a seeded choice.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    /// Maximum throw in samples.
    pub throw: f64,
    /// Tilt fraction controlling the ellipsoid rotation (`tilt_pct`).
    pub tilt_pct: f64,
    /// Width of the vertical general-gaussian throw profile (`sigma`).
    pub sigma: f64,
    /// Shape exponent of the vertical general gaussian (`p`).
    pub p: f64,
    /// Lateral anisotropy of the displacement gaussian (`coef`).
    pub coef: f64,
    /// Max-displacement voxel `[i, j, k]`; `None` = resolve automatically.
    pub center: Option<[usize; 3]>,
    /// Override of the rotated lateral-gaussian argmax offset `(row, col)`
    /// (each `±0.5`) relative to the big-grid centre. `None` = analytic
    /// choice. Only needed to replay Python exactly when the four centre
    /// pixels tie (rotation by a multiple of 90°), where scipy's spline
    /// round-off picks the winner.
    pub lateral_offset: Option<[f64; 2]>,
}

impl FaultParams {
    /// Build from the legacy (infilled-unit) dictionary values that
    /// `Faults.fault_parameters()` returns, dividing exactly like
    /// `Faults.build_faults`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_legacy(
        a: f64,
        b: f64,
        c: f64,
        x0: f64,
        y0: f64,
        z0: f64,
        throw: f64,
        tilt_pct: f64,
        infill_factor: f64,
    ) -> Self {
        Self {
            a,
            b,
            c: c / (infill_factor * infill_factor),
            x0,
            y0,
            z0: z0 / infill_factor,
            throw: throw / infill_factor,
            tilt_pct,
            sigma: 150.0,
            p: 2.0,
            coef: 1.4,
            center: None,
            lateral_offset: None,
        }
    }

    /// Set the `xyz_dis` draws explicitly (builder style).
    pub fn with_profile(mut self, sigma: f64, p: f64, coef: f64) -> Self {
        self.sigma = sigma;
        self.p = p;
        self.coef = coef;
        self
    }

    /// Set the max-displacement voxel explicitly (builder style).
    pub fn with_center(mut self, center: Option<[usize; 3]>) -> Self {
        self.center = center;
        self
    }

    /// Force the rotated-argmax half-pixel offset (parity replay).
    pub fn with_lateral_offset(mut self, offset: Option<[f64; 2]>) -> Self {
        self.lateral_offset = offset;
        self
    }

    /// Whether Python would add hockey-stick drag for this throw.
    pub fn is_hockey_stick(&self) -> bool {
        self.throw >= HOCKEY_STICK_MIN_THROW
    }
}

/// Random-mode sampling configuration (`Faults._fault_params_random`).
#[derive(Debug, Clone, PartialEq)]
pub struct RandomFaultConfig {
    /// Number of faults to draw.
    pub count: usize,
    /// Minimum throw in samples (Python: `low_fault_throw / infill = 5`).
    pub throw_min: f64,
    /// Maximum throw in samples (Python: `high_fault_throw / infill = 35`).
    ///
    /// Default is `29.0` so no hockey-stick drag is requested (deferred in the
    /// Rust port; see `docs/faults-port.md`).
    pub throw_max: f64,
}

impl Default for RandomFaultConfig {
    fn default() -> Self {
        Self {
            count: 0,
            throw_min: 5.0,
            throw_max: 29.0,
        }
    }
}

/// Small, dependency-free, *stable* PRNG (SplitMix64).
///
/// Chosen over `rand::StdRng` so the fault stream cannot change under a
/// dependency bump: outputs are part of the determinism contract.
#[derive(Debug, Clone)]
pub struct FaultRng {
    state: u64,
}

impl FaultRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xFA17_5EED_0DDB_A11Du64,
        }
    }

    /// Independent sub-stream (e.g. per fault index).
    pub fn fork(&self, stream: u64) -> Self {
        let mut s = Self {
            state: self.state ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        };
        s.next_u64();
        s
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `numpy.random.Generator.uniform(low, high)` semantics.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Uniform integer in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        // Lemire-style rejection-free enough for our n (≤ 2^32): use 128-bit mul.
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// Port of `Faults._fault_params_random` + the `xyz_dis` draws.
///
/// `shape` is the model cube `[ni, nj, nk]` (Python `cfg.cube_shape`).
/// Returned params have `center = None` (resolved later).
pub fn sample_random_faults(
    shape: [usize; 3],
    cfg: &RandomFaultConfig,
    seed: u64,
) -> Vec<FaultParams> {
    let [ni, nj, nk] = shape;
    let infill = LEGACY_INFILL_FACTOR;
    let x0_min = (ni as f64 / 4.0).trunc();
    let x0_max = (ni as f64 / 2.0).trunc();
    let y0_min = (nj as f64 / 4.0).trunc();
    let y0_max = (nj as f64 / 2.0).trunc();
    let nkf = nk as f64;
    let root = FaultRng::new(seed);
    (0..cfg.count)
        .map(|n| {
            let mut rng = root.fork(n as u64 + 1);
            let a = rng.uniform(100.0, 600.0).powi(2);
            let b = rng.uniform(100.0, 600.0).powi(2);
            let x0 = rng.uniform(x0_min - a.sqrt(), a.sqrt() + x0_max);
            let y0 = rng.uniform(y0_min - b.sqrt(), b.sqrt() + y0_max);
            let z0 = rng.uniform(-nkf * 6.0, -nkf * 2.0);
            let c0 = nkf * infill * 4.0 - z0;
            let c1 = c0 + nkf * infill / 4.0;
            let c = rng.uniform(c0, c1).powi(2);
            let tilt = rng.uniform(0.1, 0.75);
            let throw_raw = rng.uniform(cfg.throw_min * infill, cfg.throw_max * infill);
            let base = FaultParams::from_legacy(a, b, c, x0, y0, z0, throw_raw, tilt, infill);
            let throw = base.throw;
            let sigma = rng.uniform(10.0 * throw - 50.0, 300.0);
            let p = rng.uniform(1.5, 5.0);
            let coef = rng.uniform(1.3, 1.5);
            base.with_profile(sigma, p, coef)
        })
        .collect()
}
