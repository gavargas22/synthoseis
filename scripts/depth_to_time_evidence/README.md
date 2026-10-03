# Depth-to-time evidence (PR B)

Figures and pull-up numbers for `docs/depth-to-time.md`. They need numpy and
matplotlib.

```bash
cd rust
for s in 1 2 3 7 30; do
  cargo run --release -p synthoseis-core --example d2t_evidence -- "$D2T_EVIDENCE_DIR" $s 64 64 256
  cargo run --release -p synthoseis-core --example d2t_evidence -- "$D2T_EVIDENCE_DIR" $s 64 64 256 --no-salt
done
cargo run --release -p synthoseis-core --example d2t_evidence -- "$D2T_EVIDENCE_DIR" 7 64 64 256 --no-salt --faults 3
# Fault-label salt mask (#38) in time mode: masked default and unmasked.
cargo run --release -p synthoseis-core --example d2t_evidence -- "$D2T_EVIDENCE_DIR" 7 64 64 256 --faults 3
cargo run --release -p synthoseis-core --example d2t_evidence -- "$D2T_EVIDENCE_DIR" 7 64 64 256 --faults 3 --fault-labels-through-salt
SYNTHOSEIS_D2T_EVIDENCE_DIR="$D2T_EVIDENCE_DIR" cargo test --release -p synthoseis-core \
  --test angle_stack_legacy_e2e legacy_fixture_time_mode_uniform_2000
cd ../scripts/depth_to_time_evidence
D2T_EVIDENCE_DIR=... D2T_EVIDENCE_OUT=... python d2t_b_pullup_subsample.py  # table
D2T_EVIDENCE_DIR=... D2T_EVIDENCE_OUT=... python d2t_b_plots.py            # PNGs
```

| Script | What it produces |
|---|---|
| `d2t_b_pullup_subsample.py` | The measured pull-up. Each column is picked on the time-mode reflectivity, in the salt run and in the `--no-salt` run, with a 16× FFT sub-sample pick (the one from #37's salt test). Pull-up is `t_nosalt − t_salt`. The comparison is against `T_nosalt(z_L) − T_salt(z_L)` from Vp. The docstring describes how the horizon is chosen and the amplitude QC. |
| `d2t_b_pullup.py` | The same horizons picked on whole 4 ms label samples, plus the legacy-axis values. |
| `d2t_b_plots.py` | All the figures. `seed7_fault_salt_mask_time.png` also checks `masked == through AND NOT salt` voxel by voxel in depth, on the legacy axis and in time. |

`rust/synthoseis-core/tests/depth_to_time_pullup.rs` gates the same
measurement in CI on a 32×32×128 cube.
