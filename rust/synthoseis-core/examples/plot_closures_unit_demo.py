"""Before/after figure for closures per sand unit (reads `closures_unit_demo` output).

    cargo run --release -p synthoseis-core --example closures_unit_demo -- /tmp/cu
    python rust/synthoseis-core/examples/plot_closures_unit_demo.py /tmp/cu OUT_DIR

Writes `closures_per_unit.png`:
- Rows 1-2: per sand layer (`--closures-per-layer`, master 8b5988f) and per
  sand unit (default). Each row shows facies/fluid on two inlines and the
  15 deg stack.
- Row 3: the per-interval lithology with the closure units, closure counts
  per fluid, and the Rust vs legacy closure-unit population
  (`tests/fixtures/closure_units_reference.json`).
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
repo = Path(__file__).resolve().parents[3]
fix = json.loads((repo / "tests" / "fixtures" / "closure_units_reference.json").read_text())


def vol(name, dt):
    return np.fromfile(src / name, dtype=dt).reshape(ni, nj, nk)


modes = ("layer", "unit")
fac = {k: vol(f"{k}_facies.u8", np.uint8) for k in modes}
s15 = {k: vol(f"{k}_stack15.f32", np.float32) for k in modes}
hc_l = (fac["layer"] >= 3).sum(axis=(1, 2))
hc_u = (fac["unit"] >= 3).sum(axis=(1, 2))
ci = int(np.argmax(hc_l))
cu = int(np.argmax(hc_u)) if hc_u.max() > 0 else ci
fcmap = ListedColormap(["#9ecae1", "#8c8c8c", "#fdd49e", "#31a354", "#e6550d"])
slim = np.percentile(np.abs(s15["layer"]), 99.5)
names = {
    "layer": "per sand layer (--closures-per-layer, master 8b5988f)",
    "unit": "per sand unit (default, legacy Closures)",
}

fig, axs = plt.subplots(3, 4, figsize=(21, 13))
fig.suptitle(
    f"closures per sand unit vs per layer: layered geometry, Markov lithology f={meta['sand_fraction']:.3f} "
    f"T={meta['thickness']}, seed {meta['seed']}, {ni}x{nj}x{nk}, {meta['faults']} faults",
    fontsize=13,
)
for r, k in enumerate(modes):
    m = meta[k]
    br, oil, gas = m["closures"]
    a = axs[r]
    for c, il in enumerate((ci, cu)):
        im = a[c].imshow(fac[k][il].T, aspect="auto", cmap=fcmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
        a[c].set_title(
            f"{names[k]}, inline {il}\nclosures brine {br} / oil {oil} / gas {gas}, HC voxels {m['hc_voxels']}",
            fontsize=9,
        )
        cb = fig.colorbar(im, ax=a[c], ticks=range(5))
        cb.ax.set_yticklabels(["water", "shale", "brine", "oil", "gas"])
    im = a[2].imshow(s15[k][ci].T, aspect="auto", cmap="gray_r", vmin=-slim, vmax=slim)
    a[2].set_title(f"angle stack 15 deg, inline {ci}", fontsize=9)
    fig.colorbar(im, ax=a[2])
    for ax in a[:3]:
        ax.set_xlabel("crossline")
        ax.set_ylabel("sample")
d = s15["unit"] - s15["layer"]
im = axs[0, 3].imshow(d[ci].T, aspect="auto", cmap="seismic", vmin=-slim, vmax=slim)
rel = np.sqrt((d**2).mean() / (s15["layer"] ** 2).mean())
axs[0, 3].set_title(f"stack 15: unit - layer, inline {ci} (cube rel RMS {100 * rel:.1f}%)", fontsize=9)
fig.colorbar(im, ax=axs[0, 3])
axs[0, 3].set_xlabel("crossline")

# Closure list: interval of the closure's top vs voxels.
ax = axs[1, 3]
col = ["#fdd49e", "#31a354", "#e6550d"]
for k, mk, off in (("layer", "o", -0.15), ("unit", "s", 0.15)):
    for h, f, v in meta[k]["list"]:
        ax.scatter(h + off, v, marker=mk, color=col[f], edgecolor="k", s=50)
ax.scatter([], [], marker="o", color="w", edgecolor="k", label="per layer")
ax.scatter([], [], marker="s", color="w", edgecolor="k", label="per unit")
ax.set_xlabel("interval of the closure's top (layer / unit top)")
ax.set_ylabel("closure voxels")
ax.set_title("closures (colour = brine / oil / gas)", fontsize=9)
ax.legend(fontsize=8)

ax = axs[2, 0]
sand = meta["sand"]
for h, s in enumerate(sand):
    ax.add_patch(plt.Rectangle((0, h), 0.8, 1, color="#fdd49e" if s else "#8c8c8c"))
for a_, b_ in meta["units"]:
    ax.add_patch(plt.Rectangle((1, a_), 0.8, b_ - a_, facecolor="none", edgecolor="C3", lw=2))
    ax.plot([1, 1.8], [a_, a_], color="C3", lw=3)
ax.set_xlim(-0.2, 2)
ax.set_ylim(len(sand), 0)
ax.set_xticks([0.4, 1.4])
ax.set_xticklabels(["sand (tan)", "closure units"])
ax.set_ylabel("interval (0 = below seabed)")
ax.set_title("lithology and closure units (thick line = closure top;\ndeepest unit skipped as in legacy)", fontsize=9)

ax = axs[2, 1]
x = np.arange(3)
ax.bar(x - 0.2, meta["layer"]["closures"], 0.4, label="per layer")
ax.bar(x + 0.2, meta["unit"]["closures"], 0.4, label="per unit")
ax.set_xticks(x)
ax.set_xticklabels(["brine", "oil", "gas"])
ax.set_title("closure count by fluid (this cube)", fontsize=9)
ax.legend(fontsize=8)

pop = fix["population"]
ax = axs[2, 2]
bins = np.arange(-0.5, 12.5)
ax.hist(pop["n_units"], bins=bins, alpha=0.6, label=f"legacy ({len(pop['n_units'])} models)")
if (src / "rust_population.json").exists():
    rp = json.loads((src / "rust_population.json").read_text())
    ax.hist(rp["n_units"], bins=bins, histtype="step", lw=2, label="Rust keyed draws")
ax.set_xlabel("closure units per model (40 layers, f ~ U(0.05, 0.25), T = 2)")
ax.set_title("closure-unit count: Rust vs legacy", fontsize=9)
ax.legend(fontsize=8)
ax = axs[2, 3]
bins = np.arange(0.5, 10.5)
ax.hist(pop["unit_thickness"], bins=bins, alpha=0.6, label="legacy")
if (src / "rust_population.json").exists():
    ax.hist(rp["unit_thickness"], bins=bins, histtype="step", lw=2, label="Rust")
ax.set_xlabel("closure unit thickness (layers)")
ax.set_yscale("log")
ax.set_title("sand-unit thickness: Rust vs legacy", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out / "closures_per_unit.png", dpi=85)
print("wrote", out / "closures_per_unit.png")
