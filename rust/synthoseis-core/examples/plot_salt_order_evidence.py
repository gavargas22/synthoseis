#!/usr/bin/env python3
"""Before/after figures for spec-salt-fault-order (dumps of
salt_order_evidence.rs): inline and crossline sections through the salt
centre of master (drag before faulting), post-fault drag and faults off.

    python plot_salt_order_evidence.py DUMP_DIR OUT_PREFIX
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
ni, nj, nk = map(int, open(f"{d}/shape.txt").read().split())
ld = lambda n: np.fromfile(f"{d}/{n}", dtype=np.uint8).reshape(ni, nj, nk)
salt = ld("salt.u8")
faults = ld("faults.u8")
cubes = [("master 481805cc (drag, then fault)", ld("master_labels.u8")),
         ("post-fault drag (this PR)", ld("post_labels.u8")),
         ("faults off", ld("off_labels.u8"))]
cols = salt.sum(axis=2)
ci, cj = np.unravel_index(np.argmax(cols), cols.shape)
for axis, name, idx in ((0, "inline", ci), (1, "crossline", cj)):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for a, (t, lab) in zip(ax, cubes):
        sec = (lab[idx] if axis == 0 else lab[:, idx]).T.astype(float)
        sec[sec == 255] = np.nan
        a.imshow(sec, cmap="tab20", aspect="auto", interpolation="nearest")
        s = (salt[idx] if axis == 0 else salt[:, idx]).T
        a.imshow(np.ma.masked_where(s == 0, s), cmap="Greys", vmin=0, vmax=1, alpha=0.85,
                 aspect="auto", interpolation="nearest")
        if "off" not in t:
            f = (faults[idx] if axis == 0 else faults[:, idx]).T
            a.contour(f > 0, levels=[0.5], colors="k", linewidths=0.6)
        a.set_title(t)
        a.set_xlabel("crossline" if axis == 0 else "inline")
    ax[0].set_ylabel("sample")
    fig.suptitle(f"{name} {idx}: layer labels, salt grey, faults black")
    fig.tight_layout()
    fig.savefig(f"{out}_{name}.png", dpi=110)
    plt.close(fig)
    print(f"{out}_{name}.png")
