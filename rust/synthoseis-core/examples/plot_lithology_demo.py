"""Figure for the toy lithology (reads `lithology_demo` output).

    cargo run --release -p synthoseis-core --example lithology_demo -- /tmp/lith
    python rust/synthoseis-core/examples/plot_lithology_demo.py /tmp/lith OUT_DIR

Writes `toy_lithology.png`. Rows 1-2 are the previous alternating layers
(`--toy-lithology alternating`) and the legacy sand-fraction Markov chain
(default), each with facies/fluid, 15 deg reflectivity and 15/30 deg stacks
on the crest inline. Row 3 has the per-layer lithology logs, the stack
difference, and the Rust vs legacy population of per-model sand proportions
(`tests/fixtures/lithology_reference.json`).
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
fix = json.loads((repo / "tests" / "fixtures" / "lithology_reference.json").read_text())


def vol(name, dt):
    return np.fromfile(src / name, dtype=dt).reshape(ni, nj, nk)


fac = {k: vol(f"{k}_facies.u8", np.uint8) for k in ("alt", "markov")}
rfc = {k: vol(f"{k}_rfc15.f32", np.float32) for k in ("alt", "markov")}
s15 = {k: vol(f"{k}_stack15.f32", np.float32) for k in ("alt", "markov")}
s30 = {k: vol(f"{k}_stack30.f32", np.float32) for k in ("alt", "markov")}
hc = (fac["alt"] >= 3).sum(axis=(1, 2))
ci = int(np.argmax(hc))
fcmap = ListedColormap(["#9ecae1", "#8c8c8c", "#fdd49e", "#31a354", "#e6550d"])
rlim = np.percentile(np.abs(rfc["alt"]), 99.5)
slim = np.percentile(np.abs(s15["alt"]), 99.5)
names = {
    "alt": "alternating (previous, --toy-lithology alternating)",
    "markov": f"legacy sand-fraction Markov (default), f={meta['sand_fraction']:.3f}, T=2",
}

fig, axs = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle(
    f"toy lithology on the layered geometry, seed {meta['seed']}, {ni}x{nj}x{nk}, inline {ci}", fontsize=13
)
for r, k in enumerate(("alt", "markov")):
    m = meta[k]
    nsand = sum(m["sand"])
    br, oil, gas = m["closures"]
    a = axs[r]
    im = a[0].imshow(fac[k][ci].T, aspect="auto", cmap=fcmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    a[0].set_title(f"{names[k]}\n{nsand}/{len(m['sand'])} sand layers, sand {100 * m['sand_voxel_fraction']:.0f}% of sediment", fontsize=9)
    cb = fig.colorbar(im, ax=a[0], ticks=range(5))
    cb.ax.set_yticklabels(["water", "shale", "brine", "oil", "gas"])
    im = a[1].imshow(rfc[k][ci].T, aspect="auto", cmap="seismic", vmin=-rlim, vmax=rlim, interpolation="nearest")
    a[1].set_title(f"raw reflectivity 15 deg; closures brine {br} oil {oil} gas {gas}", fontsize=9)
    fig.colorbar(im, ax=a[1])
    im = a[2].imshow(s15[k][ci].T, aspect="auto", cmap="gray_r", vmin=-slim, vmax=slim)
    a[2].set_title("angle stack 15 deg", fontsize=9)
    fig.colorbar(im, ax=a[2])
    im = a[3].imshow(s30[k][ci].T, aspect="auto", cmap="gray_r", vmin=-slim, vmax=slim)
    a[3].set_title("angle stack 30 deg", fontsize=9)
    fig.colorbar(im, ax=a[3])
    for ax in a:
        ax.set_xlabel("crossline")
        ax.set_ylabel("sample")

ax = axs[2, 0]
for x, k in enumerate(("alt", "markov")):
    sand = np.array(meta[k]["sand"])
    for h, s in enumerate(sand):
        ax.add_patch(plt.Rectangle((x, h), 0.8, 1, color="#fdd49e" if s else "#8c8c8c"))
ax.set_xlim(-0.2, 2)
ax.set_ylim(len(meta["alt"]["sand"]), 0)
ax.set_xticks([0.4, 1.4])
ax.set_xticklabels(["alternating", "markov"])
ax.set_ylabel("interval (0 = below seabed)")
ax.set_title("per-layer lithology (sand = tan)", fontsize=9)
d = s15["markov"] - s15["alt"]
im = axs[2, 1].imshow(d[ci].T, aspect="auto", cmap="seismic", vmin=-slim, vmax=slim)
rel = np.sqrt((d**2).mean() / (s15["alt"] ** 2).mean())
axs[2, 1].set_title(f"stack 15: markov - alternating (rel RMS {100 * rel:.0f}%)", fontsize=9)
fig.colorbar(im, ax=axs[2, 1])
pop = fix["population"]
L = pop["layers"]
bins = np.arange(-0.5, L + 1.5) / L
axs[2, 2].hist(np.array(pop["sand_count"]) / L, bins=bins, alpha=0.6, label=f"legacy Facies ({pop['models']} models)")
axs[2, 2].hist(np.array(meta["population_sand_count"]) / L, bins=bins, histtype="step", lw=2, label="Rust keyed draws")
axs[2, 2].set_xlim(0, 0.6)
axs[2, 2].set_xlabel(f"per-model sand proportion ({L} layers, f ~ U(0.05, 0.25), T = 2)")
axs[2, 2].legend(fontsize=8)
axs[2, 2].set_title("sand proportion: Rust vs legacy", fontsize=9)
x = np.arange(1, pop["max_run_bin"] + 1)
axs[2, 3].semilogy(x, pop["sand_run_hist"], "o-", label="legacy sand runs")
axs[2, 3].semilogy(x, pop["shale_run_hist"], "s-", label="legacy shale runs")
axs[2, 3].set_xlabel(f"run length (layers; last bin >= {pop['max_run_bin']})")
axs[2, 3].set_title("legacy run-length histograms (Rust matched by chi2 in tests)", fontsize=9)
axs[2, 3].legend(fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out / "toy_lithology.png", dpi=90)
print("wrote", out / "toy_lithology.png")
