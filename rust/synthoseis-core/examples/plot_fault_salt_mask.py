"""Inline slice of fault labels before / after the salt mask (see fault_salt_dump.rs)."""
import sys

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
d = sys.argv[1]
sh = tuple(int(x) for x in open(f"{d}/shape.txt").read().split())
fm = np.fromfile(f"{d}/fault_masked.u8", np.uint8).reshape(sh)
ft = np.fromfile(f"{d}/fault_through.u8", np.uint8).reshape(sh)
salt = np.fromfile(f"{d}/salt.u8", np.uint8).reshape(sh)
lab = np.fromfile(f"{d}/labels.u8", np.uint8).reshape(sh)
stk = np.fromfile(f"{d}/stack.f32", np.float32).reshape(sh)
removed = ft & salt
X = np.arange(sh[1]) + 0.5
Y = np.arange(sh[2]) + 0.5
i = int(np.argmax(removed.sum(axis=(1, 2))))
print("inline", i, "removed on inline", removed[i].sum(), "total removed", removed.sum(),
      "through", ft.sum(), "masked", fm.sum(), "salt", salt.sum(), "fault&salt after", (fm & salt).sum())
fig, ax = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
ext = [0, sh[1], sh[2], 0]
lv = lab[i].T.astype(float); lv[lv == 255] = np.nan
v = np.percentile(np.abs(stk[i]), 99)
for a, f, title in [(ax[0], ft[i], f"Before (master 2b3850ba, --fault-labels-through-salt)\nfault voxels on inline: {ft[i].sum()}"),
                    (ax[1], fm[i], f"After (default: fault AND NOT salt)\nfault voxels on inline: {fm[i].sum()}")]:
    a.imshow(lv, cmap="Greys", extent=ext, aspect="auto", alpha=0.5)
    a.imshow(np.ma.masked_where(salt[i].T == 0, salt[i].T), cmap=ListedColormap(["#f2c14e"]), extent=ext, aspect="auto", alpha=0.6)
    a.imshow(np.ma.masked_where(f.T == 0, f.T), cmap=ListedColormap(["#d62728"]), extent=ext, aspect="auto")
    a.contour(X, Y, salt[i].T, levels=[0.5], colors="k", linewidths=1, )
    a.set_title(title, fontsize=10); a.set_xlabel("crossline")
ax[0].set_ylabel("sample (depth)")
r = removed[i].T
ax[2].imshow(stk[i].T, cmap="gray", vmin=-v, vmax=v, extent=ext, aspect="auto")
ax[2].contour(X, Y, salt[i].T, levels=[0.5], colors="#f2c14e", linewidths=1.2, )
ax[2].imshow(np.ma.masked_where(r == 0, r), cmap=ListedColormap(["#00bfff"]), extent=ext, aspect="auto")
ax[2].set_title(f"15° angle stack (unchanged), salt outline,\nremoved fault labels in blue ({removed[i].sum()} on inline, {removed.sum()} in cube)", fontsize=10)
ax[2].set_xlabel("crossline")
fig.suptitle(f"Salt mask on fault labels — demo cube 64×64×256, seed 7, --faults 3 (2 inserted), inline {i} (red = fault label, yellow = salt)")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
