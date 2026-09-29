"""Before/after figures for the rock-physics port (reads `rock_physics_demo` output).

    cargo run --release -p synthoseis-core --example rock_physics_demo -- /tmp/rpd 7 4 64 64 128
    python rust/synthoseis-core/examples/plot_rock_physics_demo.py /tmp/rpd OUT_DIR

Writes `rock_physics_reflectivity.png` (range, histogram, spectrum),
`rock_physics_angle_stacks.png` (inline sections 0/15/30 deg) and
`rock_physics_trends.png` (Vp / Vs / rho vs depth against the legacy
`RPMExample` trends). "Before" is master 10f4dcd (`--legacy-toy-depth`),
"after" the corrected default model.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / "meta.json").read_text())
shape = tuple(meta["shape"])
step = meta["depth_step_m"]


def load(name, dtype=np.float32):
    return np.fromfile(src / name, dtype).reshape(shape)


labels = load("labels.u8", np.uint8)
depth = load("depth.f32")
models = {"before: master toy": "toy", "after: corrected (inverse velocity)": "rpm",
          "after: corrected (Backus)": "backus"}
colors = {"toy": "tab:red", "rpm": "tab:blue", "backus": "tab:green"}

# 1. Reflectivity range / histogram / spectrum (raw 15 deg reflectivity).
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for i, (title, m) in enumerate(models.items()):
    r = load(f"{m}_rfc15.f32")
    s = meta[f"{m}_rfc15"]
    ax[0].bar(i, s["max"] - s["min"], bottom=s["min"], color=colors[m], alpha=0.7)
    ax[0].text(i, s["max"], f"[{s['min']:.3g}, {s['max']:.3g}]\n|r|>1: {s['abs_gt_1_pct']:.2f}%",
               ha="center", va="bottom", fontsize=8)
    nz = r[r != 0]
    ax[1].hist(np.clip(nz, -1.5, 1.5), bins=120, range=(-1.5, 1.5), histtype="step",
               color=colors[m], label=f"{title} ({s['nonzero_pct']:.1f}% non-zero)", log=True)
    stack = load(f"{m}_stack15.f32").reshape(-1, shape[2])
    spec = np.abs(np.fft.rfft(stack, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(shape[2], d=0.004)
    ax[2].semilogy(f, spec / spec.max(), color=colors[m], label=title)
ax[0].axhspan(-1, 1, color="0.9", zorder=-1, label="|r| <= 1 (physical)")
ax[0].set_xticks(range(3), [t.split(":")[0] + "\n" + t.split(": ")[1] for t in models], fontsize=8)
ax[0].set_ylabel("15 deg reflectivity range")
ax[0].set_yscale("symlog", linthresh=1)
ax[0].legend(fontsize=8)
ax[1].set_xlabel("non-zero 15 deg reflectivity (clipped to +-1.5)")
ax[1].set_ylabel("count")
ax[1].legend(fontsize=7)
ax[2].set_xlabel("frequency (Hz, 4 ms sampling)")
ax[2].set_ylabel("mean |FFT| of 15 deg stack (normalised)")
ax[2].legend(fontsize=7)
fig.suptitle(f"Reflectivity before/after, seed {meta['seed']}, {meta['faults']} faults, shape {shape}")
fig.tight_layout()
fig.savefig(out / "rock_physics_reflectivity.png", dpi=110)

# 2. Angle-stack inline sections.
il = shape[0] // 2
fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for row, m in enumerate(("toy", "rpm")):
    for col, ang in enumerate((0, 15, 30)):
        sec = load(f"{m}_stack{ang}.f32")[il].T
        v = np.percentile(np.abs(sec), 99) or 1.0
        ax[row, col].imshow(sec, cmap="seismic", vmin=-v, vmax=v, aspect="auto")
        ax[row, col].set_title(f"{'before (toy)' if m == 'toy' else 'after (corrected)'}: {ang} deg, clip {v:.3g}",
                               fontsize=9)
for a in ax[:, 0]:
    a.set_ylabel("sample")
for a in ax[1]:
    a.set_xlabel("crossline")
fig.suptitle(f"Angle stacks, inline {il}")
fig.tight_layout()
fig.savefig(out / "rock_physics_angle_stacks.png", dpi=110)

# 3. Vp / Vs / rho vs depth against the legacy RPMExample trends.
z = np.linspace(0, max(float(depth.max()), 600.0), 200)
legacy = {
    "shale": ((7.7e-12 * z**3 - 8.8e-08 * z**2 + 0.0004 * z + 1.957),
              (-0.00013 * z**2 + 1.13 * z + 1580), (-0.0001 * z**2 + 0.96 * z + 279)),
    "brine sand": ((-7.8e-09 * z**2 + 0.00012 * z + 2.021),
                   (-1.34e-05 * z**2 + 0.49 * z + 2317), (-1.0785e-05 * z**2 + 0.391 * z + 1007)),
    "oil sand": ((-9.23e-09 * z**2 + 0.00014 * z + 1.916),
                 (-8.876e-06 * z**2 + 0.505 * z + 1998), (-1.126e-05 * z**2 + 0.391 * z + 1036)),
    "gas sand": ((-1.818e-08 * z**2 + 0.000247 * z + 1.612),
                 (-3.216e-06 * z**2 + 0.4796 * z + 1996), (-1.0687e-05 * z**2 + 0.3662 * z + 1135)),
}
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
k_depth = np.broadcast_to(np.arange(shape[2]) * 100.0, shape)  # master toy: k * 100 m
for col, (prop, idx) in enumerate((("rho", 0), ("vp", 1), ("vs", 2))):
    for row, (m, d) in enumerate((("toy", k_depth), ("rpm", depth))):
        a = ax[row, col]
        p = load(f"{m}_{prop}.f32")
        sel = np.zeros(shape, bool)
        sel[::4, ::4, :] = True
        for lab in np.unique(labels):
            mask = sel & (labels == lab)
            if m == "rpm":
                mask &= depth > 0
            a.scatter(p[mask], d[mask], s=3, alpha=0.4, label=f"label {lab}")
        if m == "rpm":
            w = sel & (depth == 0)
            a.scatter(p[w], d[w], s=12, marker="x", color="k", label="water (1.028 / 1500 / 1000)")
        for name, curves in legacy.items():
            a.plot(curves[idx], z, lw=1.5, label=f"legacy {name}")
        a.invert_yaxis()
        a.set_xlabel(prop)
        a.set_ylabel("master toy depth k*100 m" if m == "toy" else "depth below seabed (m)")
        a.set_title(("before: master toy" if m == "toy" else "after: corrected default") + f" - {prop}",
                    fontsize=9)
        if m == "toy":
            a.set_ylim(min(float(k_depth.max()), 13000), 0)
ax[1, 2].legend(fontsize=7, loc="lower left")
ax[0, 2].legend(fontsize=7, loc="lower left")
fig.suptitle("Elastic properties vs depth against legacy RPMExample trends (every 4th trace)")
fig.tight_layout()
fig.savefig(out / "rock_physics_trends.png", dpi=110)
print("wrote", *(str(out / n) for n in ("rock_physics_reflectivity.png",
      "rock_physics_angle_stacks.png", "rock_physics_trends.png")))
