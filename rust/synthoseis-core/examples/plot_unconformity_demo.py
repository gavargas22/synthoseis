"""Before / after figures of the bevel-and-ponded-fill unconformity from
`unconformity_demo` output.

python plot_unconformity_demo.py DEMO_OUT FIGURE_DIR
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
summary = json.load(open(os.path.join(src, "summary.json")))


def load(name, tag, kind):
    (p,) = glob.glob(os.path.join(src, f"{name}_{tag}_{kind}_*.f32" if tag else f"{name}_{kind}_*.f32"))
    a, b = map(int, p.rsplit("_", 1)[1][:-4].split("x"))
    return np.fromfile(p, dtype="<f4").reshape(a, b)


tags = [("before", "#56 head (--legacy-unconformity)"), ("after", "unconformity (default)")]
for name in ["seed4", "seed7"]:
    s = summary[f"{name}_after"]["s"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for a, (tag, title) in zip(ax, tags):
        h, lab = load(name, tag, "horizons"), load(name, tag, "labels")
        a.imshow(lab.T % 7, cmap="Pastel2", aspect="auto", interpolation="nearest", extent=[-0.5, lab.shape[0] - 0.5, lab.shape[1] - 0.5, -0.5])
        for k in range(h.shape[1]):
            a.plot(np.arange(h.shape[0]), h[:, k], "r-" if k == s else "k-", lw=1.4 if k == s else 0.5)
        a.set_ylim(lab.shape[1], 0)
        a.set_title(title)
        a.set_xlabel("crossline (column)")
    ax[0].set_ylabel("depth (samples, 4 m)")
    fig.suptitle(f"{name}: inline section i = ni/2, horizons over labels (red: the unconformity, horizon {s} from the seabed)")
    fig.tight_layout(); fig.savefig(os.path.join(dst, f"unconformity_{name}_horizons_inline.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    st = [load(name, t, "stack") for t, _ in tags]
    v = np.percentile(np.abs(np.concatenate([x.ravel() for x in st])), 99)
    for a, x, (tag, title) in zip(ax, st, tags):
        a.imshow(x.T, cmap="gray_r", vmin=-v, vmax=v, aspect="auto", interpolation="bilinear")
        a.set_title(title); a.set_xlabel("crossline (column)")
    ax[0].set_ylabel("two-way time sample (4 ms)")
    fig.suptitle(f"{name}: 15 degree angle stack, inline section i = ni/2")
    fig.tight_layout(); fig.savefig(os.path.join(dst, f"unconformity_{name}_stack_inline.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ms = [load(name, t, "map_surface") for t, _ in tags]
    lo, hi = min(m.min() for m in ms), max(m.max() for m in ms)
    for a, m, (tag, title) in zip(ax[:2], ms, tags):
        im = a.imshow(m.T, cmap="viridis_r", vmin=lo, vmax=hi, origin="upper")
        a.set_title(f"horizon {s} depth: {tag}", fontsize=10); a.set_xlabel("inline"); a.set_ylabel("crossline")
        fig.colorbar(im, ax=a, label="samples")
    d = ms[1] - ms[0]
    im = ax[2].imshow(d.T, cmap="RdBu", vmin=-abs(d).max(), vmax=abs(d).max())
    ax[2].set_title("horizon change (after - before)"); fig.colorbar(im, ax=ax[2], label="samples")
    e = load(name, None, "eroded")
    im = ax[3].imshow(e.T, cmap="magma_r")
    ax[3].set_title("eroded thickness"); fig.colorbar(im, ax=ax[3], label="samples")
    fig.suptitle(f"{name}: map view of the unconformity horizon")
    fig.tight_layout(); fig.savefig(os.path.join(dst, f"unconformity_{name}_map.png"), dpi=110); plt.close(fig)
