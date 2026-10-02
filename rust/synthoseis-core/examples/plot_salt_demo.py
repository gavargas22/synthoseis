"""Figures for salt bodies (default) vs --no-salt (master b4f4259).

    cargo run --release -p synthoseis-core --example salt_demo -- /tmp/saltd 7 4 64 64 256
    cargo run --release -p synthoseis-core --example salt_demo -- stats /tmp/salts 16000
    python rust/synthoseis-core/examples/plot_salt_demo.py /tmp/saltd /tmp/salts OUT_DIR

Writes (and prints the KS table with scipy p-values, 5 % level):
- OUT_DIR/salt_bodies.png: the demo cube (64 x 64 x 256, seed 7, 4 faults)
  on the inline and crossline through the salt. Row 1: facies without salt
  (master b4f4259), with salt, and the layer labels with salt (horizons
  dragged up against the flank). Row 2: 0 / 15 / 30 deg angle stacks with
  salt. Row 3: salt thickness map, crossline facies with salt, 15 deg stack
  without salt.
- OUT_DIR/salt_validation.png: legacy (numpy) vs Rust (keyed) salt-shape
  distributions (4000 legacy vs 16000 Rust bodies) with the two-sample KS
  statistic, 5 % critical and p.
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

demo_dir, stats_dir, out_dir = (Path(a) for a in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)
meta = json.loads((demo_dir / "meta.json").read_text())
ni, nj, nk = meta["shape"]


def vol(name, dt):
    return np.fromfile(demo_dir / name, dtype=dt).reshape(ni, nj, nk)


fac = {k: vol(f"{k}_facies.u8", np.uint8) for k in ("nosalt", "salt")}
lab = vol("salt_labels.u8", np.uint8)
mask = vol("salt_mask.u8", np.uint8)
st = {a: vol(f"salt_stack{a}.f32", np.float32) for a in (0, 15, 30)}
st15_off = vol("nosalt_stack15.f32", np.float32)
il = int(np.argmax(mask.sum(axis=(1, 2))))
xl = int(np.argmax(mask.sum(axis=(0, 2))))

fcmap = ListedColormap(["white", "#bdbdbd", "#6baed6", "#31a354", "#e6550d", "#7b3294"])
fig, ax = plt.subplots(3, 3, figsize=(17, 14))
kw = dict(aspect="auto", interpolation="nearest")


def outline(a, m):
    if m.any():
        a.contour(m.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)


ax[0, 0].imshow(fac["nosalt"][il].T, cmap=fcmap, vmin=0, vmax=5, **kw)
ax[0, 0].set_title(f"--no-salt (master b4f4259), inline {il}\nshale grey, brine blue, oil green, gas orange", fontsize=9)
ax[0, 1].imshow(fac["salt"][il].T, cmap=fcmap, vmin=0, vmax=5, **kw)
ax[0, 1].set_title(f"salt (default), inline {il}: salt purple", fontsize=9)
lab_il = np.where(lab[il] == 255, np.nan, lab[il] % 20).T
ax[0, 2].imshow(lab_il, cmap="tab20", vmin=0, vmax=19, **kw)
outline(ax[0, 2], mask[il])
ax[0, 2].set_title("layer labels with salt (outline): horizons dragged up\nagainst the flank, smoothed (legacy drag)", fontsize=9)
for c, a in enumerate((0, 15, 30)):
    s = st[a][il].T
    v = np.percentile(np.abs(s), 99)
    ax[1, c].imshow(s, cmap="gray", vmin=-v, vmax=v, **kw)
    outline(ax[1, c], mask[il])
    ax[1, c].set_title(f"{a} deg angle stack with salt, inline {il}", fontsize=9)
thick = mask.sum(axis=2).astype(float)
thick[thick == 0] = np.nan
im = ax[2, 0].imshow(thick.T, origin="lower", cmap="viridis", **kw)
ax[2, 0].set_title(
    f"salt thickness (samples): top {meta['salt']['salt_top']:.1f}, radius {meta['salt']['salt_radius']:.1f} columns,\n"
    f"{meta['salt']['salt_voxels']} voxels in {meta['salt']['salt_columns']} columns",
    fontsize=9,
)
ax[2, 0].axhline(xl, color="w", lw=0.6)
ax[2, 0].axvline(il, color="w", lw=0.6)
ax[2, 0].set_xlabel("inline")
ax[2, 0].set_ylabel("crossline")
plt.colorbar(im, ax=ax[2, 0], fraction=0.046)
ax[2, 1].imshow(fac["salt"][:, xl].T, cmap=fcmap, vmin=0, vmax=5, **kw)
ax[2, 1].set_title(f"salt (default), crossline {xl}", fontsize=9)
s = st15_off[il].T
v = np.percentile(np.abs(s), 99)
ax[2, 2].imshow(s, cmap="gray", vmin=-v, vmax=v, **kw)
ns, sa = meta["nosalt"], meta["salt"]
ch = meta["stack_change"]
ax[2, 2].set_title(
    f"15 deg stack --no-salt, inline {il}\nclosures b/o/g {ns['closures']} -> {sa['closures']}, "
    f"HC voxels {ns['hc_voxels']} -> {sa['hc_voxels']}\nstack rel RMS change 0/15/30: "
    f"{ch['0']['rel_rms']:.3f}/{ch['15']['rel_rms']:.3f}/{ch['30']['rel_rms']:.3f}",
    fontsize=9,
)
for a in ax.flat[:9]:
    if a is not ax[2, 0]:
        a.set_xlabel("crossline" if a is not ax[2, 1] else "inline")
        a.set_ylabel("sample")
fig.suptitle(f"Salt bodies: demo cube {ni}x{nj}x{nk}, seed {meta['seed']}, {meta['faults']} faults", fontsize=12)
fig.tight_layout()
fig.savefig(out_dir / "salt_bodies.png", dpi=110)
print("wrote", out_dir / "salt_bodies.png")

fx = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/salt_reference.json").read_text())["population"]
rust = json.loads((stats_dir / "population.json").read_text())
names = ["radius", "top", "tip", "cx", "cy", "r1", "base", "bx", "by", "r2"]
desc = {
    "radius": "radius R (columns)", "top": "top (samples)", "tip": "crest tip z", "cx": "cap centre i",
    "cy": "cap centre j", "r1": "cap radius", "base": "stem base z", "bx": "stem centre i",
    "by": "stem centre j", "r2": "stem radius",
}
n, m = len(fx["radius"]), len(rust["radius"])
crit = 1.358 * np.sqrt((n + m) / (n * m))
fig, ax = plt.subplots(2, 5, figsize=(20, 7.5))
print(f"| statistic | KS D | 5% critical | p |  (legacy n={n}, Rust m={m})")
for a, k in zip(ax.flat, names):
    t = stats.ks_2samp(fx[k], rust[k])
    print(f"| {k} | {t.statistic:.4f} | {crit:.4f} | {t.pvalue:.3f} |")
    bins = np.histogram_bin_edges(np.concatenate([fx[k], rust[k]]), 40)
    a.hist(fx[k], bins=bins, density=True, alpha=0.55, label="legacy numpy")
    a.hist(rust[k], bins=bins, density=True, histtype="step", color="k", lw=1.2, label="Rust keyed")
    a.set_title(f"{desc[k]}\nKS D={t.statistic:.4f} (5% crit {crit:.4f}), p={t.pvalue:.3f}", fontsize=9)
ax[0, 0].legend(fontsize=8)
fig.suptitle(
    "Salt shape draws on the legacy example grid (64x64x1250 + pad 10, horizon 1 at 20): legacy SaltModel vs Rust",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(out_dir / "salt_validation.png", dpi=110)
print("wrote", out_dir / "salt_validation.png")
