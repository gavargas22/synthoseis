"""Figure for the Zoeppritz `det` -> `d` fix (reads `zoeppritz_demo` output).

    cargo run --release -p synthoseis-core --example zoeppritz_demo -- /tmp/zd
    python rust/synthoseis-core/examples/plot_zoeppritz_demo.py /tmp/zd OUT_DIR

Writes `zoeppritz_fix.png`:
(a) AVO curves of the demo cube's interfaces and a class-III gas sand:
    textbook form (default), legacy typo, and the full 4x4 matrix solution;
(b) |form - matrix| against angle over `tests/fixtures/zoeppritz_reference.json`;
(c-e) 30 degree inline sections, textbook vs legacy and their difference.
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
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tests" / "fixtures"))
from generate_zoeppritz_reference import legacy_rpp, matrix_rpp  # noqa: E402


def textbook(vp1, vs1, rho1, vp2, vs2, rho2, ang):
    t = np.radians(ang) + 0j
    p = np.sin(t) / vp1
    t2, f1, f2 = np.arcsin(p * vp2), np.arcsin(p * vs1), np.arcsin(p * vs2)
    a = rho2 * (1 - 2 * np.sin(f2) ** 2) - rho1 * (1 - 2 * np.sin(f1) ** 2)
    b = rho2 * (1 - 2 * np.sin(f2) ** 2) + 2 * rho1 * np.sin(f1) ** 2
    c = rho1 * (1 - 2 * np.sin(f1) ** 2) + 2 * rho2 * np.sin(f2) ** 2
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    e = b * np.cos(t) / vp1 + c * np.cos(t2) / vp2
    f = b * np.cos(f1) / vs1 + c * np.cos(f2) / vs2
    g = a - d * np.cos(t) / vp1 * np.cos(f2) / vs2
    h = a - d * np.cos(t2) / vp2 * np.cos(f1) / vs1
    det = e * f + g * h * p * p
    return ((f * (b * np.cos(t) / vp1 - c * np.cos(t2) / vp2) - h * p * p * (a + d * np.cos(t) / vp1 * np.cos(f2) / vs2)) / det).real


fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
ang = np.arange(51)

ax = fig.add_subplot(gs[0, 0])
names = {0: "seabed (water / shale)", 1: "shale / sand"}
curves = [(names.get(n, f"k={it['k']}"), it["props"], np.array(it["exact"]), np.array(it["legacy"]))
          for n, it in enumerate(meta["interfaces"])]
gas = [2800.0, 1400.0, 2.35, 2600.0, 1500.0, 2.1]
curves.append(("class-III gas sand (textbook example)", gas,
               np.array([textbook(*gas, a) for a in ang]), np.array([legacy_rpp(*gas, a).real for a in ang])))
for (name, props, ex, lg), col in zip(curves, ["tab:blue", "tab:orange", "tab:green"]):
    mat = np.array([matrix_rpp(*props, a).real for a in ang])
    ax.plot(ang, ex, color=col, lw=2, label=f"{name}: textbook (default)")
    ax.plot(ang, lg, color=col, lw=1.5, ls="--", label="legacy det typo")
    ax.plot(ang[::5], mat[::5], "o", color=col, mfc="none", ms=7, label="4x4 matrix solve")
ax.set_xlabel("incidence angle (deg)")
ax.set_ylabel("Rpp")
ax.set_title("(a) AVO: textbook vs legacy vs full matrix")
ax.legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(0.0, 0.62))
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ref = json.loads((repo / "tests/fixtures/zoeppritz_reference.json").read_text())["cases"]
a_ = np.array([c["angle_deg"] for c in ref])
m_ = np.array([c["matrix_re"] for c in ref])
b_ = np.array([c["bruges"] for c in ref])
l_ = np.array([c["legacy"] for c in ref])
t_ = np.array([textbook(*c["props"], c["angle_deg"]) for c in ref])
jit = (np.random.default_rng(0).random(len(a_)) - 0.5) * 2
ax.semilogy(a_ + jit, np.abs(l_ - m_) + 1e-18, ".", color="tab:red", ms=4, label="legacy typo - matrix")
ax.semilogy(a_ + jit, np.abs(t_ - m_) + 1e-18, ".", color="tab:blue", ms=4, label="textbook - matrix")
ax.semilogy(a_ + jit, np.abs(b_ - m_) + 1e-18, "x", color="tab:gray", ms=3, label="bruges - matrix")
ax.set_ylim(1e-18, 1)
ax.set_xlabel("incidence angle (deg)")
ax.set_ylabel("|Rpp - matrix solution|")
ax.set_title(f"(b) {len(ref)} reference cases (45 interfaces)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(gs[0, 2])
ch = meta["change"]
angs = [c["angle"] for c in ch]
nz = [c for c in ch if c["angle"] > 0]  # identical at normal incidence
ax.plot([c["angle"] for c in nz], [c["rfc_max_abs"] for c in nz], "o-", label="max |textbook - legacy| (raw rfc; 0 at 0 deg)")
ax.plot([c["angle"] for c in nz], [c["rfc_rel_rms"] for c in nz], "s-", label="relative RMS change (raw rfc)")
ax.plot(angs, [max(c["gpu_cpu_rfc_exact"], 1e-9) for c in ch], "^-", label="GPU vs CPU max |d| (textbook)")
ax.set_yscale("log")
ax.set_xlabel("incidence angle (deg)")
ax.set_title(f"(c) demo cube {shape}, seed {meta['seed']}, {meta['faults']} faults")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

il = shape[0] // 2
e30 = np.fromfile(src / "exact_stack30.f32", np.float32).reshape(shape)[il]
l30 = np.fromfile(src / "legacy_stack30.f32", np.float32).reshape(shape)[il]
v = np.percentile(np.abs(e30), 99.5)
for n, (img, title, scale) in enumerate([
    (e30, "(d) 30 deg stack, textbook (default)", 1),
    (l30, "(e) 30 deg stack, legacy typo (--legacy-zoeppritz)", 1),
    (e30 - l30, "(f) difference x10", 10),
]):
    ax = fig.add_subplot(gs[1, n])
    ax.imshow((img * scale).T, cmap="seismic", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("crossline")
    ax.set_ylabel("sample")
fig.suptitle("Zoeppritz PP: textbook `a + d cos/vp1 cos/vs2` (default) vs legacy `a + det ...`", fontsize=13)
fig.tight_layout()
fig.savefig(out / "zoeppritz_fix.png", dpi=110)
print(out / "zoeppritz_fix.png")
