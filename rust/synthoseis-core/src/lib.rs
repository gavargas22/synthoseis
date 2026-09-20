//! Core stubs: config, RNG, job partition, single-worker runner, parity harness placeholder.
//!
//! Algorithm ports (geo / seismic / RPM) live in sibling crates and are not implemented here.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Runtime configuration for a generation job (skeleton fields only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub seed: u64,
    /// Worker count. Skeleton locks to a single local worker (`1`).
    pub workers: usize,
    pub inline_count: usize,
    pub crossline_count: usize,
    pub samples: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            workers: 1,
            inline_count: 2,
            crossline_count: 2,
            samples: 4,
        }
    }
}

/// Deterministic RNG helper seeded from [`RunConfig::seed`].
pub struct SeededRng {
    inner: StdRng,
}

impl SeededRng {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.inner.gen()
    }
}

/// Job partition stub. Cloud / multi-worker sharding comes later.
#[derive(Debug, Clone)]
pub struct JobPartition {
    pub worker_id: usize,
    pub worker_count: usize,
    pub job_ids: Vec<u64>,
}

impl JobPartition {
    /// Single local worker owns the full job list.
    pub fn single_worker(config: &RunConfig) -> Self {
        let n = (config.inline_count * config.crossline_count) as u64;
        Self {
            worker_id: 0,
            worker_count: 1.max(config.workers),
            job_ids: (0..n.max(1)).collect(),
        }
    }
}

/// Placeholder result from a single-worker run (no algorithms yet).
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub seed: u64,
    pub workers: usize,
    pub job_count: usize,
    pub status: &'static str,
}

/// One local worker path — no cloud yet.
#[derive(Debug, Clone)]
pub struct SingleWorkerRunner {
    pub config: RunConfig,
    pub partition: JobPartition,
}

impl SingleWorkerRunner {
    pub fn new(config: RunConfig, partition: JobPartition) -> Self {
        Self { config, partition }
    }

    pub fn run_placeholder(&self) -> RunSummary {
        let _rng = SeededRng::from_seed(self.config.seed);
        RunSummary {
            seed: self.config.seed,
            workers: self.partition.worker_count,
            job_count: self.partition.job_ids.len(),
            status: "ok-placeholder",
        }
    }
}

/// Fixed-seed golden comparison **placeholder**.
///
/// Python baseline vs Rust MAE/IoU will be wired in a later PR. This stub only
/// checks that a fixed seed produces a stable deterministic float sequence so
/// CI has a parity-harness hook without claiming numeric product parity.
pub mod parity {
    use super::SeededRng;

    pub const GOLDEN_SEED: u64 = 0x5EED_CAFE;

    /// Placeholder golden values for the first three draws of [`GOLDEN_SEED`].
    /// Recomputed once and locked for the skeleton; replace with real volume metrics later.
    pub fn golden_stub_draws() -> [f64; 3] {
        let mut rng = SeededRng::from_seed(GOLDEN_SEED);
        [rng.next_f64(), rng.next_f64(), rng.next_f64()]
    }

    /// Compare stub draws within absolute tolerance. Not MAE/IoU vs Python.
    pub fn compare_stub(actual: &[f64; 3], expected: &[f64; 3], atol: f64) -> bool {
        actual
            .iter()
            .zip(expected.iter())
            .all(|(a, e)| (a - e).abs() <= atol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_worker_partition() {
        let cfg = RunConfig::default();
        let part = JobPartition::single_worker(&cfg);
        assert_eq!(part.worker_count, 1);
        assert_eq!(part.worker_id, 0);
        assert!(!part.job_ids.is_empty());
    }

    #[test]
    fn parity_harness_stub_is_deterministic() {
        let a = parity::golden_stub_draws();
        let b = parity::golden_stub_draws();
        assert!(parity::compare_stub(&a, &b, 0.0));
        // Document: Python baseline MAE/IoU comparison is deferred.
        assert!(
            parity::compare_stub(&a, &parity::golden_stub_draws(), 1e-12),
            "fixed-seed stub must be bit-stable across calls"
        );
    }

    #[test]
    fn runner_placeholder() {
        let cfg = RunConfig {
            seed: 7,
            workers: 1,
            ..RunConfig::default()
        };
        let part = JobPartition::single_worker(&cfg);
        let summary = SingleWorkerRunner::new(cfg, part).run_placeholder();
        assert_eq!(summary.status, "ok-placeholder");
        assert_eq!(summary.workers, 1);
    }
}
