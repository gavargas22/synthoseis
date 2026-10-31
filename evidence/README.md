# Fault heave evidence (7a)

Seeds 1, 7, 13 at 128x128x256 with `--faults 4`, legacy = `--legacy-fault-kinematics`.
Per seed: `seedN_inline.png`, `seedN_depthslice.png` (legacy vs heave),
`seedN_horizon.png` (top of a mid layer; the right panel shows the heave gap
as bands along each fault), `seedN_faultlabels.png` (label overlay), and
`seedN-stats.txt` (runtime, per-fault throw/heave percentiles over each
fault's label band, heave at peak throw, share of voxels changed,
fault-label Jaccard vs legacy). A fault whose surface never moves more than
one sample prints `n=0`. Heave at peak throw over seeds 1-20 (spec §2,
§6.4) is p50 4.9, p95 11.3, max 14.5 bins; single seeds differ (1/7/13 give
p50 5.4/3.4/6.2). Runtimes were re-measured on Oct 30 2026 after the
per-tile gather cache: seed 1 is a median of 3 with master, seeds 7 and 13
are single runs.
Made with `synthoseis-core/examples/heave_evidence.rs` and `plot_heave_evidence.py`.
