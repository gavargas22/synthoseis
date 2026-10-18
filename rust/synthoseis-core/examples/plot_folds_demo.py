"""Before / after figures of the analytic folds from `folds_demo` output.

python plot_folds_demo.py DEMO_OUT FIGURE_DIR
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)


def load(name, tag, kind):
    (p,) = glob.glob(os.path.join(src, f"{name}_{tag}_{kind}_*.f32"))
    a, b = map(int, p.rsplit("_", 1)[1][:-4].split("x"))
    return np.fromfile(p, dtype="<f4").reshape(a, b)


tags = [("legacy", "7e80f2af (--legacy-fold-geometry)"), ("similar", "folded default (similar)"), ("parallel", "--fold-style parallel")]
for name in ["seed4", "seed7"]:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for a, (tag, title) in zip(ax, tags):
        h = load(name, tag, "horizons")
        lab = load(name, tag, "labels")
        a.imshow(lab.T % 7, cmap="Pastel2", aspect="auto", interpolation="nearest", extent=[-0.5, lab.shape[0] - 0.5, lab.shape[1] - 0.5, -0.5])
        for k in range(h.shape[1]):
            a.plot(np.arange(h.shape[0]), h[:, k], "k-", lw=0.6)
        a.set_ylim(lab.shape[1], 0)
        a.set_title(title)
        a.set_xlabel("crossline (column)")
    ax[0].set_ylabel("depth (samples, 4 m)")
    fig.suptitle(f"{name}: inline section i = ni/2, continuous horizons over labels")
    fig.tight_layout()
    fig.savefig(os.path.join(dst, f"folds_{name}_horizons_inline.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    st = [load(name, t, "stack") for t, _ in tags]
    v = np.percentile(np.abs(np.concatenate([s.ravel() for s in st])), 99)
    for a, s, (tag, title) in zip(ax, st, tags):
        a.imshow(s.T, cmap="gray_r", vmin=-v, vmax=v, aspect="auto", interpolation="bilinear")
        a.set_title(title)
        a.set_xlabel("crossline (column)")
    ax[0].set_ylabel("two-way time sample (4 ms)")
    fig.suptitle(f"{name}: 15° angle stack, inline section i = ni/2")
    fig.tight_layout()
    fig.savefig(os.path.join(dst, f"folds_{name}_stack_inline.png"), dpi=110)
    plt.close(fig)
print("ok")
