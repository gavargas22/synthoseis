"""Before/after figure for the fault vertical-reach fix.

    python plot_faults_reach.py LEGACY_DIR FIT_DIR OUT.png

Both dirs come from `cargo run --release -p synthoseis-core --example
faults_demo -- DIR SEED FAULTS NI NJ NK {legacy|fit}`. Each row shows an
inline and a crossline section of the faulted layer-cake age with the fault
mask (red) and the seabed (cyan); the top row is `legacy_reach = true`, the
bottom row the default `FitColumn` mode.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    shape = tuple(meta["shape"])
    age = np.fromfile(d / "age.f32", dtype="<f4").reshape(shape)
    mask = np.fromfile(d / "fault_mask.u8", dtype="u1").reshape(shape)
    seabed = np.fromfile(d / "seabed.f64", dtype="<f8").reshape(shape[:2])
    return meta, age, mask, seabed


def section(ax, age2d, mask2d, sb, title):
    ax.imshow(age2d.T, cmap="gray", aspect="auto", interpolation="nearest")
    m = np.ma.masked_where(mask2d.T == 0, mask2d.T)
    ax.imshow(m, cmap="autumn", alpha=0.9, aspect="auto", interpolation="nearest")
    ax.plot(np.arange(len(sb)), sb, color="cyan", lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("k (samples)")


def main():
    runs = [load(sys.argv[1]), load(sys.argv[2])]
    out = sys.argv[3]
    # Pick the sections with the most legacy fault voxels.
    mask0 = runs[0][2]
    i = int(mask0.sum(axis=(1, 2)).argmax())
    j = int(mask0.sum(axis=(0, 2)).argmax())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, (meta, age, mask, sb) in enumerate(runs):
        tag = "legacy reach" if meta["legacy_reach"] else "FitColumn (default)"
        info = f"{tag}: {meta['fault_voxels']:,} fault voxels, {meta['above_seabed']:,} above seabed"
        section(axes[row, 0], age[i], mask[i], sb[i], f"inline i={i} | {info}")
        section(axes[row, 1], age[:, j], mask[:, j], sb[:, j], f"crossline j={j}")
    meta = runs[0][0]
    fig.suptitle(
        f"Fault vertical reach, seed {meta['seed']}, shape {tuple(meta['shape'])}, "
        f"{meta['inserted']} faults (red = fault mask, cyan = seabed)"
    )
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
