"""Figures for the layered toy geometry (reads `layered_geometry_demo` output).

    cargo run --release -p synthoseis-core --example layered_geometry_demo -- /tmp/lg
    python rust/synthoseis-core/examples/plot_layered_geometry_demo.py /tmp/lg OUT_DIR

Writes
* `layered_geometry_sections.png`: inline and crossline through the dome
  crest (labels, facies/fluid, Vp, 15 deg reflectivity), a closure map, the
  top-of-dome horizon map and the planar master geometry for comparison;
* `layered_geometry_stacks.png`: 0/15/30 deg angle stacks, layered vs planar,
  the per-layer random depth shifts and what the shifts and the closure
  fluids each change.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / "meta.json").read_text())
ni, nj, nk = meta["shape"]
nh = meta["nh"]


def vol(name, dt):
    return np.fromfile(src / name, dtype=dt).reshape(ni, nj, nk)


labels = vol("labels.u8", np.uint8).astype(float)
labels[labels == 255] = np.nan
plabels = vol("planar_labels.u8", np.uint8).astype(float)
plabels[plabels == 255] = np.nan
facies = vol("facies.u8", np.uint8)
vp = vol("vp.f32", np.float32)
r15 = vol("rfc15.f32", np.float32)
pr15 = vol("planar_rfc15.f32", np.float32)
ns15 = vol("noshift_rfc15.f32", np.float32)
st = {a: vol(f"stack{a}.f32", np.float32) for a in (0, 15, 30)}
pst15 = vol("planar_stack15.f32", np.float32)
nf15 = vol("nofluid_stack15.f32", np.float32)
maps = np.fromfile(src / "maps.f64", dtype=np.float64).reshape(ni, nj, nh)

ci, cj = (int(round(x)) for x in meta["params"]["dome_center"])
ci, cj = min(max(ci, 0), ni - 1), min(max(cj, 0), nj - 1)
fcmap = ListedColormap(["#9ecae1", "#8c8c8c", "#fdd49e", "#31a354", "#e6550d"])
layers = meta["layers"]
n_cl = {k: sum(c["fluid"] == k for l in layers for c in l["closures"]) for k in ("Brine", "Oil", "Gas")}
title = (
    f"layered toy geometry (default), seed {meta['seed']}, {ni}x{nj}x{nk}, {meta['faults']} faults requested: "
    f"{nh} horizons / {len(layers)} layers, closures brine {n_cl['Brine']} oil {n_cl['Oil']} gas {n_cl['Gas']}"
)


def sect(ax, v, **kw):
    im = ax.imshow(v.T, aspect="auto", interpolation="nearest", **kw)
    ax.set_ylabel("sample")
    return im


fig, axs = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle(title, fontsize=12)
for row, (name, cut) in enumerate([(f"inline {ci}", np.s_[ci, :, :]), (f"crossline {cj}", np.s_[:, cj, :])]):
    a = axs[row]
    im = sect(a[0], labels[cut], cmap="tab20")
    a[0].set_title(f"{name}: labels (255 = water)")
    fig.colorbar(im, ax=a[0])
    im = sect(a[1], facies[cut], cmap=fcmap, vmin=-0.5, vmax=4.5)
    a[1].set_title(f"{name}: water / shale / brine / oil / gas sand")
    cb = fig.colorbar(im, ax=a[1], ticks=range(5))
    cb.ax.set_yticklabels(["water", "shale", "brine", "oil", "gas"])
    im = sect(a[2], vp[cut], cmap="viridis")
    a[2].set_title(f"{name}: Vp (m/s)")
    fig.colorbar(im, ax=a[2])
    lim = np.percentile(np.abs(r15), 99.5)
    im = sect(a[3], r15[cut], cmap="seismic", vmin=-lim, vmax=lim)
    a[3].set_title(f"{name}: raw reflectivity 15 deg (textbook Zoeppritz)")
    fig.colorbar(im, ax=a[3])
    for ax in a:
        ax.set_xlabel("crossline" if row == 0 else "inline")

hc = np.isin(facies, (3, 4))
gas = (facies == 4).any(axis=2)
oil = (facies == 3).any(axis=2)
cmap_code = np.where(gas & oil, 3, np.where(gas, 2, np.where(oil, 1, 0)))
im = axs[2, 0].imshow(cmap_code.T, origin="lower", cmap=ListedColormap(["#f0f0f0", "#31a354", "#e6550d", "#756bb1"]), vmin=-0.5, vmax=3.5)
cb = fig.colorbar(im, ax=axs[2, 0], ticks=range(4))
cb.ax.set_yticklabels(["none", "oil", "gas", "oil+gas"])
axs[2, 0].set_title(f"closure map: hydrocarbon columns ({hc.sum()} voxels)")
axs[2, 0].set_xlabel("inline")
axs[2, 0].set_ylabel("crossline")
hmid = nh // 2
im = axs[2, 1].imshow(maps[:, :, hmid].T, origin="lower", cmap="viridis_r")
axs[2, 1].contour(maps[:, :, hmid].T, levels=12, colors="k", linewidths=0.5)
axs[2, 1].plot([ci], [cj], "r+", ms=12)
axs[2, 1].set_title(f"horizon {hmid} depth (samples): dome + tilt + fbm")
axs[2, 1].set_xlabel("inline")
fig.colorbar(im, ax=axs[2, 1])
im = sect(axs[2, 2], plabels[ci], cmap="tab20")
axs[2, 2].set_title(f"--toy-geometry planar labels, inline {ci}")
fig.colorbar(im, ax=axs[2, 2])
im = sect(axs[2, 3], pr15[ci], cmap="seismic", vmin=-lim, vmax=lim)
axs[2, 3].set_title(f"planar rfc15 (non-zero {100 * meta['planar_rfc15_nonzero']:.2f}% vs {100 * meta['rfc15_nonzero']:.1f}%)")
fig.colorbar(im, ax=axs[2, 3])
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out / "layered_geometry_sections.png", dpi=90)
plt.close(fig)

fig, axs = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle(title, fontsize=12)
slim = np.percentile(np.abs(st[15]), 99.5)
for c, a in enumerate((0, 15, 30)):
    im = sect(axs[0, c], st[a][ci], cmap="gray_r", vmin=-slim, vmax=slim)
    axs[0, c].set_title(f"angle stack {a} deg, inline {ci}")
    fig.colorbar(im, ax=axs[0, c])
    im = sect(axs[1, c], st[a][:, cj], cmap="gray_r", vmin=-slim, vmax=slim)
    axs[1, c].set_title(f"angle stack {a} deg, crossline {cj}")
    fig.colorbar(im, ax=axs[1, c])
plim = np.percentile(np.abs(pst15), 99.5)
im = sect(axs[0, 3], pst15[ci], cmap="gray_r", vmin=-plim, vmax=plim)
axs[0, 3].set_title(f"planar geometry: stack 15 deg, inline {ci}")
fig.colorbar(im, ax=axs[0, 3])
# AVO: amplitude vs angle at the strongest hydrocarbon interface of the section.
k_hc = np.argwhere(hc[ci])
if len(k_hc):
    j0, k0 = k_hc[len(k_hc) // 2]
    trace = {a: st[a][ci, j0] for a in (0, 15, 30)}
    for a in (0, 15, 30):
        axs[1, 3].plot(trace[a], np.arange(nk), label=f"{a} deg")
    axs[1, 3].axhline(k0, color="r", lw=1.5)
    axs[1, 3].set_xlim(-slim, slim)
    axs[1, 3].invert_yaxis()
    axs[1, 3].legend()
    axs[1, 3].set_title(f"traces il {ci} xl {j0} (red: hydrocarbon sand)")
iv = [l["interval"] for l in layers]
sh = [l["shift"] for l in layers]
axs[2, 0].bar(iv, sh, color=["#e6550d" if s else "#bdbdbd" for s in sh])
axs[2, 0].axvline(19.5, color="k", ls="--", lw=0.8)
axs[2, 0].set_title("per-layer depth shift (samples), layers >= 20")
axs[2, 0].set_xlabel("horizon interval")
dlim = np.percentile(np.abs(r15 - ns15), 99.5) or 1
im = sect(axs[2, 1], (r15 - ns15)[ci], cmap="seismic", vmin=-dlim, vmax=dlim)
e = meta["shift_effect"]
axs[2, 1].set_title(f"rfc15: default - no shifts (rel RMS {100 * e['rfc15_rel_rms']:.0f}%)")
fig.colorbar(im, ax=axs[2, 1])
dlim = np.percentile(np.abs(st[15] - nf15), 99.5) or 1
im = sect(axs[2, 2], (st[15] - nf15)[ci], cmap="seismic", vmin=-dlim, vmax=dlim)
e = meta["fluid_effect"]
axs[2, 2].set_title(f"stack15: default - brine only (rel RMS {100 * e['stack15_rel_rms']:.0f}%)")
fig.colorbar(im, ax=axs[2, 2])
im = sect(axs[2, 3], np.abs(r15[:, cj]), cmap="magma")
axs[2, 3].set_title(f"|rfc15|, crossline {cj}")
fig.colorbar(im, ax=axs[2, 3])
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out / "layered_geometry_stacks.png", dpi=90)
plt.close(fig)
print("wrote", out / "layered_geometry_sections.png", out / "layered_geometry_stacks.png")
