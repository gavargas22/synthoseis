"""PR B evidence figures -> $D2T_EVIDENCE_OUT (default /workspace/synthoseis-bench/out/d2t-b/)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from d2t_b_pullup import load, pullup, first_index, D, OUT
import d2t_b_pullup_subsample as sub
BEFORE = "before: --legacy-depth-as-time\n(= master 0eb937b5, #35 stack-parity baseline)"
AFTER = "after: time output (default)"

def label_img(lab):
    img = np.ma.masked_equal(lab.astype(float), 255)
    return np.ma.masked_array(img.data % 20, img.mask)

def extent(n, step, nk):
    return [0, n, nk * step, 0]

def sections(seed):
    s = load(f"seed{seed}")
    row, pu = pullup(seed)
    i = row["thickest"]["i"]
    m = s["meta"]; nz = m["depth_shape"][2]; nt = m["nt"]; dt = m["dt_ms"]; digi = m["digi_legacy"]
    nj = m["depth_shape"][1]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    vmax = np.percentile(np.abs(s["time_stack"][i]), 99)
    for c, (mode, step, n, title) in enumerate((("legacy", digi, nz, BEFORE), ("time", dt, nt, AFTER))):
        ext = extent(nj, step, n)
        ax[0, c].imshow(s[f"{mode}_stack"][i].T, cmap="gray_r", vmin=-vmax, vmax=vmax, aspect="auto", extent=ext)
        ax[1, c].imshow(label_img(s[f"{mode}_labels"][i]).T, cmap="tab20", vmin=0, vmax=19, aspect="auto",
                        extent=ext, interpolation="nearest")
        for r in range(2):
            if f"{mode}_salt" in s:
                ax[r, c].contour(np.arange(nj) + 0.5, (np.arange(n) + 0.5) * step, s[f"{mode}_salt"][i].T,
                                 levels=[0.5], colors="cyan" if r == 0 else "k", linewidths=1.2)
            ax[r, c].set_xlabel("crossline")
            ax[r, c].set_ylabel("TWT (ms)" if mode == "time" else "sample x 4 ms (depth / 2000 m/s)")
        ax[0, c].set_title(f"seed {seed} inline {i} stack, {title}")
        ax[1, c].set_title(f"labels (layer id mod 20), salt outline")
        # Sub-salt horizon L picks and the velocity-predicted time.
        k = first_index(s[f"{mode}_labels"], pu["L"])[i]
        jj = np.nonzero(k >= 0)[0]
        ax[0, c].plot(jj + 0.5, k[jj] * step, "y.", ms=3, label=f"top of layer {pu['L']} (picked on output labels)")
        if mode == "time":
            zs = pu["zs"][i]
            pred = np.array([s["twt"][i, j, zs[j]] for j in range(nj)])
            ax[0, c].plot(np.arange(nj) + 0.5, pred, "r-", lw=0.8, label="T from Vp at that horizon (predicted)")
        ax[0, c].legend(loc="lower left", fontsize=8)
    t = row["thickest"]
    fig.suptitle(f"seed {seed}: thickest salt column (j={t['j']}, {t['salt_m']:.0f} m): pull-up predicted "
                 f"{t['pred']:.0f} ms; 4 ms label-sample picks: time {t['time']:.0f} ms, legacy {t['legacy']:.0f} ms")
    p = f"{OUT}/seed{seed}_stack_labels_depth_vs_time.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p

SHOWCASE = (7, 30)
CAVEAT = (2, 3)

def pullup_fig(seeds):
    """Sub-sample (16x FFT) pull-up picks vs the velocity prediction.

    Seeds 7 and 30 are the showcase. Seeds 2 and 3 are drawn faded: under
    their thick salt the only sub-salt reflectors are very weak (Strata:
    |rc| 0.009 / 0.006, picks -3.1 / -2.2 ms at clean columns), so their
    measured columns are those with a stronger reflector, not the thickest
    salt.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)
    rows = []
    for seed in seeds:
        row, d = sub.measure(seed); rows.append(row)
        show = seed in SHOWCASE
        cav = seed in CAVEAT
        alpha = 1.0 if show else (0.25 if cav else 0.6)
        tag = " SHOWCASE" if show else (" caveat: weak sub-salt rc" if cav else "")
        h, = ax[0].plot(d["pred"], d["meas"], ".", ms=4 if show else 2.5, alpha=alpha,
                        label=f"seed {seed} (L{d['L']}, {row['columns']} cols, |rc| {row['event_rc_median']:.3f}){tag}")
        ax[0].plot(d["pred"], d["legacy"], "x", ms=3, alpha=0.3 * alpha, color=h.get_color())
        r = d["meas"] - d["pred"]
        ax[1].hist(r, bins=np.arange(-3.5, 3.55, 0.1), histtype="step", color=h.get_color(),
                   lw=1.8 if show else 0.8, alpha=max(alpha, 0.4), ls="-" if not cav else "--",
                   label=f"seed {seed}: mean {r.mean():+.2f}, std {r.std():.2f}, max {np.abs(r).max():.2f} ms{tag}")
    lim = max(r["pred_max"] for r in rows) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
    ax[0].set_xlabel("predicted pull-up from Vp (ms)")
    ax[0].set_ylabel("measured pull-up (ms)")
    ax[0].set_title("salt pull-up per column: dots = time output, 16x FFT sub-sample picks\n"
                    "x = legacy axis (before), 4 ms label picks", fontsize=10)
    ax[0].legend(fontsize=7, markerscale=3, loc="upper left")
    ax[1].set_xlabel("measured - predicted (ms)")
    ax[1].set_title("residual, sub-sample picks on the time-mode reflectivity", fontsize=10)
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].text(0.99, 0.02, "Seeds 2, 3 (dashed): under the thickest salt the only sub-salt\n"
               "reflectors are very weak (|rc| 0.009 / 0.006); Strata's picks there\n"
               "are -3.1 / -2.2 ms (interference). Measured columns here are the\n"
               "ones with a stronger reflector, not the thickest salt.",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=7,
               bbox=dict(fc="lightyellow", ec="0.6"))
    p = f"{OUT}/pullup_measured_vs_predicted.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p, rows

def faults_fig(tag="seed7_nosalt_faults3"):
    s = load(tag); m = s["meta"]; nz = m["depth_shape"][2]; nt = m["nt"]; dt = m["dt_ms"]; nj = m["depth_shape"][1]
    df = np.fromfile(f"{D}/{tag}_depth_faults.u8", np.uint8).reshape(s["labels"].shape)
    lf = np.fromfile(f"{D}/{tag}_legacy_faults.u8", np.uint8).reshape(s["legacy_labels"].shape)
    tf = np.fromfile(f"{D}/{tag}_time_faults.u8", np.uint8).reshape(s["time_labels"].shape)
    assert (lf == df).all()
    i = int(np.argmax(tf.sum(axis=(1, 2))))
    # Independent check: resample the depth fault cube with the TWT cube.
    t = s["twt"].astype(np.float64); tn = np.arange(nt) * dt
    k = np.stack([np.clip(np.searchsorted(t[a, b], tn, side="right") - 1, 0, nz - 1)
                  for a in range(t.shape[0]) for b in range(t.shape[1])]).reshape(t.shape[0], t.shape[1], nt)
    ref = np.take_along_axis(df, k, axis=2)
    mism = int((ref != tf).sum())
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    vmax = np.percentile(np.abs(s["time_stack"][i]), 99)
    red = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.8)])
    for c, (mode, step, n, f, title) in enumerate((("legacy", 4.0, nz, lf, BEFORE), ("time", dt, nt, tf, AFTER))):
        ext = extent(nj, step, n)
        ax[0, c].imshow(s[f"{mode}_stack"][i].T, cmap="gray_r", vmin=-vmax, vmax=vmax, aspect="auto", extent=ext)
        ax[0, c].imshow(f[i].T, cmap=red, vmin=0, vmax=1, aspect="auto", extent=ext, interpolation="nearest")
        ax[1, c].imshow(label_img(s[f"{mode}_labels"][i]).T, cmap="tab20", vmin=0, vmax=19, aspect="auto",
                        extent=ext, interpolation="nearest")
        ax[1, c].contour(np.arange(nj) + 0.5, (np.arange(n) + 0.5) * step, f[i].T, levels=[0.5], colors="k", linewidths=0.8)
        ax[0, c].set_title(f"stack + fault_labels (red), {title}")
        ax[1, c].set_title("labels + fault_labels outline")
        for r in range(2):
            ax[r, c].set_xlabel("crossline"); ax[r, c].set_ylabel("TWT (ms)" if mode == "time" else "sample x 4 ms")
    fig.suptitle(f"seed 7, 3 faults, --no-salt, inline {i}: fault voxels depth {int(df.sum())} -> time {int(tf.sum())}; "
                 f"\ntime fault_labels vs an independent numpy resample of the depth fault_labels with T(Vp): {mism} mismatching voxels")
    p = f"{OUT}/seed7_faults_labels_depth_vs_time.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p, dict(depth_voxels=int(df.sum()), time_voxels=int(tf.sum()), mismatches=mism, inline=i)

def uniform2000_fig():
    js = json.load(open("/workspace/synthoseis-d2t/tests/fixtures/angle_stack_e2e.json"))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for r, case in enumerate(js["cases"]):
        seed = case["seed"]; nk1 = case["shape"][2] - 1; cols = case["legacy"]["sample_columns"]
        a = np.fromfile(f"{D}/uniform2000_seed{seed}_angle15.f32", np.float32)
        a = a.reshape(len(cols), 2, nk1)
        t = np.arange(nk1) * 4.0
        for ci, (i, j) in enumerate(cols):
            ax[r, 0].plot(a[ci, 0] + 0.15 * ci, t, "k-", lw=1.4)
            ax[r, 0].plot(a[ci, 1] + 0.15 * ci, t, "r--", lw=0.8)
            d = np.abs(a[ci, 1].astype(np.float64) - a[ci, 0])
            ax[r, 1].semilogx(np.maximum(d, 1e-12), t, lw=0.8, label=f"column ({i},{j})")
        ax[r, 0].invert_yaxis(); ax[r, 1].invert_yaxis()
        ax[r, 0].set_title(f"seed {seed}, 15 deg stack at uniform 2000 m/s: black = #35 legacy Python baseline, "
                           f"red = time output (read one sample down)", fontsize=9)
        ax[r, 0].set_ylabel("ms"); ax[r, 1].set_xlabel("|time - #35 baseline|")
        peak = case["legacy"]["stack"]["stats"][2]["max_abs"]
        ax[r, 1].axvline(8 * np.spacing(np.float32(peak)), color="gray", ls=":", label="8 ulp of peak (#35 tol)")
        ax[r, 1].set_title(f"seed {seed}: difference (top-edge columns: seabed reflection within padlen = 27 samples)", fontsize=9)
        ax[r, 1].legend(fontsize=8)
    p = f"{OUT}/uniform2000_time_vs_pr35_baseline.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p

if __name__ == "__main__":
    paths = [sections(s) for s in (1, 2, 3, 30)]
    p, rows = pullup_fig((7, 30, 1, 2, 3)); paths.append(p)
    p, finfo = faults_fig(); paths.append(p)
    paths.append(uniform2000_fig())
    json.dump(dict(pullup=rows, faults=finfo, pngs=paths), open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(finfo)); print("\n".join(paths))
