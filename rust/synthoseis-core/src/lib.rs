//! Core: config, RNG, job partition, multi-worker runner, **parity harness**,
//! and **e2e tiny-cube pipeline** (MDIO → geo → closures → RPM → seismic → MDIO).
//!
//! Algorithm ports (geo / seismic / RPM) live in sibling crates.
//! Parity compares **label volumes** and **angle-stack volumes** — not bit-identical
//! full seismic.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Runtime configuration for a generation job (skeleton fields only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub seed: u64,
    /// Local / planned worker count (`1` = single-worker path).
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

/// Summary from a single-worker run (placeholder or e2e).
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub seed: u64,
    pub workers: usize,
    pub job_count: usize,
    pub status: &'static str,
}

/// One local worker path — used by [`MultiWorkerRunner`] fan-out and CLI.
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

    /// Run the tiny-cube e2e pipeline (CPU, full cube).
    ///
    /// When `store` is `Some`, writes labels + angle stack to MDIO and checks
    /// round-trip parity.
    pub fn run_e2e(
        &self,
        store: Option<std::path::PathBuf>,
    ) -> Result<pipeline::E2eReport, String> {
        // Tiny-cube floor: bump sub-8³ defaults (CLI smoke uses 8³).
        let cfg = pipeline::E2eConfig {
            seed: self.config.seed,
            inline_count: self.config.inline_count.max(pipeline::TINY_DIM),
            crossline_count: self.config.crossline_count.max(pipeline::TINY_DIM),
            samples: self.config.samples.max(pipeline::TINY_DIM),
            store_path: store,
        };
        pipeline::run_e2e(&cfg)
    }
}

pub mod parity;
pub mod partition;
pub mod pipeline;

pub use partition::{
    partition_jobs, JobPartition, JobPartitionPlan, MultiRunSummary, MultiWorkerRunner,
};

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

    #[test]
    fn parity_metrics_identity() {
        let labels = [0u8, 1, 1, 2, 255];
        let angles = [0.0f32, 1.0, -0.5];
        assert!((parity::macro_label_iou(&labels, &labels, 255) - 1.0).abs() < 1e-12);
        assert!((parity::label_agreement(&labels, &labels) - 1.0).abs() < 1e-12);
        assert!(parity::mean_absolute_error(&angles, &angles) == 0.0);
        assert!(parity::max_abs_diff(&angles, &angles) == 0.0);
    }

    #[test]
    fn runner_e2e_tiny_self_parity() {
        let cfg = RunConfig {
            seed: 42,
            workers: 1,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
        };
        let part = JobPartition::single_worker(&cfg);
        let report = SingleWorkerRunner::new(cfg, part)
            .run_e2e(None)
            .expect("e2e");
        assert_eq!(report.status, "ok-e2e");
        assert!(report.parity.passes_defaults());
    }

    #[test]
    fn multi_e2e_still_full_cube() {
        let cfg = RunConfig {
            seed: 42,
            workers: 4,
            inline_count: 8,
            crossline_count: 8,
            samples: 8,
        };
        let report = MultiWorkerRunner::new(cfg)
            .run_e2e(None)
            .expect("e2e");
        assert_eq!(report.status, "ok-e2e");
        assert_eq!(report.volumes.shape, [8, 8, 8]);
        assert!(report.parity.passes_defaults());
    }

    #[test]
    fn parity_harness_fixture_near_parity() {
        let (lab_ref, lab_pert, ang_ref, ang_pert) =
            parity::load_parity_cubes_8().expect("fixture");
        assert_eq!(lab_ref.len(), 8 * 8 * 8);
        assert_eq!(ang_ref.len(), 8 * 8 * 8);

        // Self-parity must be exact.
        let self_report = parity::compare_volumes(&lab_ref, &lab_ref, &ang_ref, &ang_ref);
        assert!(self_report.passes_defaults());
        assert!((self_report.label_iou - 1.0).abs() < 1e-12);
        assert!(self_report.angle_mae == 0.0);

        // Perturbed fixture stays within documented near-parity tolerances.
        let report = parity::compare_volumes(&lab_ref, &lab_pert, &ang_ref, &ang_pert);
        assert!(
            report.passes_defaults(),
            "near-parity failed: {report:?} (tolerances iou>={}, agr>={}, mae<={}, maxabs<={})",
            parity::LABEL_IOU_MIN,
            parity::LABEL_AGREEMENT_MIN,
            parity::ANGLE_MAE_MAX,
            parity::ANGLE_MAX_ABS_MAX
        );
        // Ensure metrics are real comparisons, not stubs.
        assert!(report.label_agreement < 1.0);
        assert!(report.angle_mae > 0.0);
    }
}
