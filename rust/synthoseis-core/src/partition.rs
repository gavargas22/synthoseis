//! Multi-worker job partition API and local runner.
//!
//! # Sharding policy
//! Contiguous chunks: worker `w` gets `[w*n/W, (w+1)*n/W)`.
//! Union covers all jobs with no overlap. When `jobs < workers`, empty
//! worker slots are **kept** (not dropped) for stable cloud handoff.
//!
//! [`JobPartitionPlan`] is a serde JSON cloud-handoff artifact.

use serde::{Deserialize, Serialize};
use crate::{RunConfig, RunSummary, SeededRng, SingleWorkerRunner};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobPartition {
    pub worker_id: usize,
    pub worker_count: usize,
    pub job_ids: Vec<u64>,
}

impl JobPartition {
    pub fn single_worker(config: &RunConfig) -> Self {
        let n = (config.inline_count * config.crossline_count) as u64;
        Self {
            worker_id: 0,
            worker_count: 1.max(config.workers),
            job_ids: (0..n.max(1)).collect(),
        }
    }
}

/// Shard jobs across `config.workers` contiguous chunks (keeps empty slots).
pub fn partition_jobs(config: &RunConfig) -> Vec<JobPartition> {
    let worker_count = 1.max(config.workers);
    let n = ((config.inline_count * config.crossline_count) as u64).max(1);
    (0..worker_count)
        .map(|worker_id| {
            let start = (worker_id as u64 * n) / worker_count as u64;
            let end = ((worker_id as u64 + 1) * n) / worker_count as u64;
            JobPartition {
                worker_id,
                worker_count,
                job_ids: (start..end).collect(),
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobPartitionPlan {
    pub worker_count: usize,
    pub partitions: Vec<JobPartition>,
}

impl JobPartitionPlan {
    pub fn from_config(config: &RunConfig) -> Self {
        let partitions = partition_jobs(config);
        Self { worker_count: partitions.len(), partitions }
    }
    pub fn for_worker(&self, worker_id: usize) -> Option<&JobPartition> {
        self.partitions.get(worker_id)
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[derive(Debug, Clone)]
pub struct MultiRunSummary {
    pub seed: u64,
    pub workers: usize,
    pub job_count: usize,
    pub per_worker: Vec<RunSummary>,
    pub status: &'static str,
}

/// Local multi-worker runner (`std::thread::scope`; no rayon / no cloud).
/// E2e always runs the full cube once (strip-stitch follow-up).
#[derive(Debug, Clone)]
pub struct MultiWorkerRunner {
    pub config: RunConfig,
    pub plan: JobPartitionPlan,
}

impl MultiWorkerRunner {
    pub fn new(config: RunConfig) -> Self {
        let plan = JobPartitionPlan::from_config(&config);
        Self { config, plan }
    }
    pub fn from_plan(config: RunConfig, plan: JobPartitionPlan) -> Self {
        Self { config, plan }
    }
    pub fn run_placeholder(&self) -> MultiRunSummary {
        let workers = self.plan.worker_count;
        let per_worker = if workers <= 1 {
            let part = self.plan.for_worker(0).cloned()
                .unwrap_or_else(|| JobPartition::single_worker(&self.config));
            vec![SingleWorkerRunner::new(self.config.clone(), part).run_placeholder()]
        } else {
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for part in &self.plan.partitions {
                    let cfg = self.config.clone();
                    let part = part.clone();
                    handles.push(scope.spawn(move || {
                        let _rng = SeededRng::from_seed(cfg.seed.wrapping_add(part.worker_id as u64));
                        SingleWorkerRunner::new(cfg, part).run_placeholder()
                    }));
                }
                handles.into_iter().map(|h| h.join().expect("worker")).collect()
            })
        };
        let job_count = per_worker.iter().map(|s| s.job_count).sum();
        MultiRunSummary {
            seed: self.config.seed,
            workers,
            job_count,
            per_worker,
            status: "ok-multi-placeholder",
        }
    }
    pub fn run_e2e(&self, store: Option<std::path::PathBuf>) -> Result<crate::pipeline::E2eReport, String> {
        let part = JobPartition::single_worker(&self.config);
        SingleWorkerRunner::new(self.config.clone(), part).run_e2e(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn assert_cover_no_overlap(parts: &[JobPartition], expected_jobs: u64) {
        let mut seen = BTreeSet::new();
        for p in parts {
            for &id in &p.job_ids {
                assert!(seen.insert(id), "overlap on job_id {id}");
            }
        }
        assert_eq!(seen.len() as u64, expected_jobs);
        if expected_jobs > 0 {
            assert_eq!(*seen.iter().next().unwrap(), 0);
            assert_eq!(*seen.iter().next_back().unwrap(), expected_jobs - 1);
        }
    }

    #[test]
    fn single_worker_partition() {
        let part = JobPartition::single_worker(&RunConfig::default());
        assert_eq!(part.worker_count, 1);
        assert!(!part.job_ids.is_empty());
    }

    #[test]
    fn partition_workers_one_matches_single() {
        let cfg = RunConfig { workers: 1, inline_count: 8, crossline_count: 8, ..RunConfig::default() };
        let single = JobPartition::single_worker(&cfg);
        let parts = partition_jobs(&cfg);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].job_ids, single.job_ids);
    }

    #[test]
    fn partition_four_workers_on_8x8() {
        let cfg = RunConfig { workers: 4, inline_count: 8, crossline_count: 8, samples: 8, seed: 42 };
        let parts = partition_jobs(&cfg);
        assert_eq!(parts.len(), 4);
        assert_cover_no_overlap(&parts, 64);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.job_ids.len(), 16);
            assert_eq!(p.job_ids[0], (i * 16) as u64);
        }
    }

    #[test]
    fn partition_keeps_empty_workers_when_jobs_lt_workers() {
        let cfg = RunConfig { workers: 5, inline_count: 2, crossline_count: 1, ..RunConfig::default() };
        let parts = partition_jobs(&cfg);
        assert_eq!(parts.len(), 5);
        assert_cover_no_overlap(&parts, 2);
        assert_eq!(parts.iter().filter(|p| p.job_ids.is_empty()).count(), 3);
    }

    #[test]
    fn plan_round_trip_json() {
        let cfg = RunConfig { workers: 4, inline_count: 8, crossline_count: 8, ..RunConfig::default() };
        let plan = JobPartitionPlan::from_config(&cfg);
        let back = JobPartitionPlan::from_json(&plan.to_json().unwrap()).unwrap();
        assert_eq!(plan, back);
        assert!(plan.for_worker(99).is_none());
    }

    #[test]
    fn multi_runner_workers_one_matches_single() {
        let cfg = RunConfig { seed: 7, workers: 1, inline_count: 4, crossline_count: 4, samples: 4 };
        let single = SingleWorkerRunner::new(cfg.clone(), JobPartition::single_worker(&cfg)).run_placeholder();
        let multi = MultiWorkerRunner::new(cfg).run_placeholder();
        assert_eq!(multi.job_count, single.job_count);
        assert_eq!(multi.status, "ok-multi-placeholder");
    }

    #[test]
    fn multi_runner_four_workers_aggregates_jobs() {
        let cfg = RunConfig { seed: 3, workers: 4, inline_count: 8, crossline_count: 8, samples: 8 };
        let summary = MultiWorkerRunner::new(cfg).run_placeholder();
        assert_eq!(summary.workers, 4);
        assert_eq!(summary.job_count, 64);
    }
}
