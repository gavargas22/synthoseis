"""Fault heave evidence figures (spec-fault-kinematics 7a §12) from the raw
dump of `heave_evidence.rs`:

    python plot_heave_evidence.py DUMP_DIR OUT_PREFIX [LABEL_A LABEL_B]

(7b: labels `7a 7b`; also writes OUT_PREFIX_faultmap.png, the map-view
fault-label footprint, which shows the tip lines.)

Writes OUT_PREFIX_inline.png, _depthslice.png, _horizon.png,
_faultlabels.png and prints the share of voxels changed and the fault-label
Jaccard against legacy.
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
NAMES = dict(zip(("legacy", "heave"), sys.argv[3:5] or ("legacy", "heave")))
ni, nj, nk = map(int, open(f"{d}/shape.txt").read().split())
ld = lambda n, t: np.fromfile(f"{d}/{n}", dtype=t).reshape(ni, nj, nk)
st = {k: ld(f"{k}_stack.f32", "<f4") for k in ("legacy", "heave")}
lab = {k: ld(f"{k}_labels.u8", "u1") for k in ("legacy", "heave")}
flt = {k: ld(f"{k}_faults.u8", "u1") for k in ("legacy", "heave")}
i0, k0 = ni // 2, nk // 2
vmax = np.percentile(np.abs(st["legacy"]), 99)

def pair(fn, title, name, **kw):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for a, k in zip(ax, ("legacy", "heave")):
        a.imshow(fn(k), cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto", **kw)
        a.set_title(f"{NAMES[k]}: {title}")
    fig.savefig(f"{out}_{name}.png", dpi=110)
    plt.close(fig)

pair(lambda k: st[k][i0].T, f"inline {i0}", "inline")
pair(lambda k: st[k][:, :, k0].T, f"depth slice k={k0}", "depthslice", origin="lower")

# Horizon map: first sample at or below the top of a mid layer.
labs = lab["legacy"]
valid = labs[labs < 255]
L = int(np.median(valid))
def hmap(k):
    m = lab[k] >= L
    m &= lab[k] < 255
    z = np.where(m.any(axis=2), m.argmax(axis=2), np.nan).astype(float)
    return z
hl, hh = hmap("legacy"), hmap("heave")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
lo, hi = np.nanpercentile(hl, 1), np.nanpercentile(hl, 99)
for a, z, t in zip(ax[:2], (hl, hh), ("legacy", "heave")):
    im = a.imshow(z.T, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    # Faults in map view at that horizon (any fault label within 2 samples).
    a.set_title(f"{NAMES[t]}: top of layer {L} (samples)")
    fig.colorbar(im, ax=a)
    g = np.hypot(*np.gradient(np.nan_to_num(z, nan=lo)))
    a.contour((g > 3).T, levels=[0.5], colors="r", linewidths=0.6)
dz = hh - hl
im = ax[2].imshow(dz.T, origin="lower", cmap="RdBu_r", vmin=-15, vmax=15)
ax[2].set_title(f"{NAMES['heave']} − {NAMES['legacy']} (samples)")
fig.colorbar(im, ax=ax[2])
fig.savefig(f"{out}_horizon.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for a, k in zip(ax, ("legacy", "heave")):
    a.imshow(st[k][i0].T, cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto")
    f = np.ma.masked_equal(flt[k][i0].T, 0)
    a.imshow(f, cmap="tab10", alpha=0.7, aspect="auto", interpolation="nearest", vmin=1, vmax=10)
    a.set_title(f"{NAMES[k]}: fault labels, inline {i0}")
fig.savefig(f"{out}_faultlabels.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for a, k in zip(ax, ("legacy", "heave")):
    im = a.imshow((flt[k] > 0).sum(axis=2).T, origin="lower", cmap="magma")
    a.set_title(f"{NAMES[k]}: fault-label voxels per column (map view)")
    fig.colorbar(im, ax=a)
fig.savefig(f"{out}_faultmap.png", dpi=110)
plt.close(fig)

changed = np.mean(st["legacy"] != st["heave"])
a, b = flt["legacy"] > 0, flt["heave"] > 0
jac = (a & b).sum() / max((a | b).sum(), 1)
lchg = np.mean(lab["legacy"] != lab["heave"])
print(f"stack voxels changed {changed:.4f}; layer labels changed {lchg:.4f}; fault-label Jaccard vs {NAMES['legacy']} {jac:.4f}; fault-label voxels {a.sum()} -> {b.sum()}")
