"""Before/after figure for 3D closure segmentation across faults.

    cargo run --release -p synthoseis-core --example closure_segments_demo -- /tmp/csf 4 8 64 64 256 0.3 1
    cargo run --release -p synthoseis-core --example closure_segments_demo -- /tmp/csd 7 4 64 64 256
    cargo run --release -p synthoseis-core --example closure_segments_demo -- stats /tmp/css 2000
    python rust/synthoseis-core/examples/plot_closure_segments_demo.py /tmp/csf /tmp/csd /tmp/css OUT.png

Writes OUT.png (and prints the statistics with scipy p-values, 5 % level):
- Rows 1-2: unsegmented (`--closures-unsegmented`, master ef2dc42) and
  3D-segmented (default) on the faulted figure cube. Each row shows the
  fluid facies and the closure ids (2D region per unit vs 3D compartment)
  on the inline with the most changed voxels, and a map of closure ids.
- Row 3: the 15 deg stack change on that inline, closure / compartment
  counts and HC voxels for the figure and demo cubes, and the fluid split
  of the compartments of 2000 faulted sandy models.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import stats  # noqa: E402

fig_dir, demo_dir, stats_dir, out = (Path(a) for a in sys.argv[1:5])
meta = json.loads((fig_dir / "meta.json").read_text())
demo = json.loads((demo_dir / "meta.json").read_text())
st = json.loads((stats_dir / "stats.json").read_text())
ni, nj, nk = meta["shape"]


def vol(name, dt):
    return np.fromfile(fig_dir / name, dtype=dt).reshape(ni, nj, nk)


modes = ("unseg", "seg")
fac = {k: vol(f"{k}_facies.u8", np.uint8) for k in modes}
comp = {k: vol(f"{k}_comp.i32", np.int32) for k in modes}
s15 = {k: vol(f"{k}_stack15.f32", np.float32) for k in modes}
changed = (fac["unseg"] != fac["seg"]).sum(axis=(1, 2))
il = int(np.argmax(changed)) if changed.max() > 0 else int(np.argmax((comp["seg"] >= 0).sum(axis=(1, 2))))
inside = np.where((comp["seg"] >= 0).any(axis=(0, 1)))[0]
k0, k1 = max(0, inside.min() - 15), min(nk, inside.max() + 15)

fcmap = ListedColormap(["white", "#bdbdbd", "#6baed6", "#31a354", "#e6550d"])
titles = {"unseg": "unsegmented (--closures-unsegmented = ef2dc42)", "seg": "3D-segmented across faults (default)"}
fig, ax = plt.subplots(3, 3, figsize=(17, 13))
rng = np.random.default_rng(0)
perm = rng.permutation(4096)
for r, k in enumerate(modes):
    ax[r, 0].imshow(fac[k][il, :, k0:k1].T, cmap=fcmap, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    ax[r, 0].set_title(f"{titles[k]}\ninline {il}: shale grey, brine blue, oil green, gas orange", fontsize=9)
    c = comp[k][il, :, k0:k1].T.astype(float)
    c = np.where(c >= 0, perm[np.clip(c, 0, 4095).astype(int)] % 20, np.nan)
    ax[r, 1].imshow(np.where(fac[k][il, :, k0:k1].T > 0, 0.3, np.nan), cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax[r, 1].imshow(c, cmap="tab20", vmin=0, vmax=19, aspect="auto", interpolation="nearest")
    ax[r, 1].set_title(f"closure ids on inline {il} ({'2D region per unit' if k == 'unseg' else '3D compartment'})", fontsize=9)
    m = np.where((comp[k] >= 0).any(axis=2), perm[np.clip(np.where(comp[k] >= 0, comp[k], 4096).min(axis=2), 0, 4095)] % 20, -1).astype(float)
    m[m < 0] = np.nan
    ax[r, 2].imshow(m, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    ax[r, 2].axhline(il, color="k", lw=0.8, ls="--")
    ax[r, 2].set_title("map: shallowest closure id per column (inline down, xline across)", fontsize=9)
    for a in ax[r, :2]:
        a.set_xlabel("crossline")
        a.set_ylabel(f"sample (from {k0})")
d = (s15["seg"] - s15["unseg"])[il, :, k0:k1].T
v = np.abs(s15["unseg"]).max() * 0.5
ax[2, 0].imshow(d, cmap="seismic", vmin=-v, vmax=v, aspect="auto")
ax[2, 0].set_title(f"15 deg stack: segmented - unsegmented, inline {il}\n(rel RMS whole cube {meta['stack_change']['15']['rel_rms']:.3f})", fontsize=9)

labels = ["closures\n(ef2dc42)", "compartments\n(segmented)", "joined units", "split-off"]
x = np.arange(len(labels))
for off, mm, name in ((-0.2, meta, "figure cube"), (0.2, demo, "demo cube")):
    vals = [sum(mm["unseg"]["closures"]), mm["seg"]["compartments"], mm["seg"]["multi_unit"], mm["seg"]["split_off"]]
    ax[2, 1].bar(x + off, vals, width=0.4, label=f"{name} (HC voxels {mm['unseg']['hc_voxels']} -> {mm['seg']['hc_voxels']})")
ax[2, 1].set_xticks(x)
ax[2, 1].set_xticklabels(labels, fontsize=8)
ax[2, 1].legend(fontsize=8)
ax[2, 1].set_title(
    f"figure: seed {meta['seed']}, {meta['faults']} faults, fraction {meta['fraction']}, thickness {meta['thickness']};\n"
    f"demo: seed {demo['seed']}, {demo['faults']} faults, Markov default (unchanged)",
    fontsize=9,
)
allc = np.array(st["model"]["all"], float)
chi = stats.chisquare(allc)
ax[2, 2].bar(["brine", "oil", "gas"], allc, color=["#6baed6", "#31a354", "#e6550d"])
ax[2, 2].axhline(allc.sum() / 3, color="k", ls="--")
ax[2, 2].set_title(
    f"fluids of {int(allc.sum())} kept compartments, {st['models']} faulted sandy models\n"
    f"chi2 {chi.statistic:.2f} (df 2, 5% critical 5.991), p {chi.pvalue:.3f}",
    fontsize=9,
)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}; inline {il}, changed voxels {int(changed[il])}")

split = np.array(st["direct_split"], float)
c1 = stats.chisquare(split)
tab = np.array(st["split_vs_primary"], float)
c2 = stats.chi2_contingency(tab, correction=False)
mu = np.array(st["model"]["multi_unit"], float)
c3 = stats.chisquare(mu)
print(f"direct split-off draws n={int(split.sum())}: chi2 {c1.statistic:.3f} (df 2, crit 5.991) p {c1.pvalue:.3f}")
print(f"split-off vs primary 3x3: chi2 {c2.statistic:.3f} (df 4, crit 9.488) p {c2.pvalue:.3f}")
print(f"model compartments n={int(allc.sum())} {allc.astype(int).tolist()}: chi2 {chi.statistic:.3f} p {chi.pvalue:.3f}")
print(f"model multi-unit compartments n={int(mu.sum())} {mu.astype(int).tolist()}: chi2 {c3.statistic:.3f} p {c3.pvalue:.3f}")
