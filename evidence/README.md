# Fault heave evidence (7a)

Seeds 1, 7, 13 at 128x128x256 with `--faults 4`, legacy = `--legacy-fault-kinematics`.
Per seed: `seedN_inline.png`, `seedN_depthslice.png` (legacy vs heave),
`seedN_horizon.png` (top of a mid layer; the right panel shows the heave gap
as bands along each fault), `seedN_faultlabels.png` (label overlay), and
`seedN-stats.txt` (runtime, per-fault throw/heave percentiles over each
fault's label band, share of voxels changed, fault-label Jaccard vs legacy).
Made with `synthoseis-core/examples/heave_evidence.rs` and `plot_heave_evidence.py`.
