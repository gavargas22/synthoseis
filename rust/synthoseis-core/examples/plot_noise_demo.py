"""Rust vs legacy seismic noise: slices, spectra, histograms and chain parity.

    python plot_noise_demo.py DEMO_DIR OUT_DIR

DEMO_DIR is the output of ``cargo run --release -p synthoseis-core --example
noise_demo -- DEMO_DIR``. The script runs the *real* legacy
``add_weighted_noise`` (via ``tests/fixtures/generate_seismic_noise.py``) on
the same Rust raw reflectivity, and writes

* ``OUT_DIR/noise_slices.png``: inline slices of the Rust and legacy noise,
  the noisy reflectivity, and the noisy bandpassed stack (Rust pipeline vs
  legacy ``apply_bandlimits`` + ``apply_lateral_filter`` fed the Rust
  reflectivity + Rust noise);
* ``OUT_DIR/noise_spectra.png``: amplitude spectra, amplitude histograms and
  inter-angle correlation, Rust vs legacy, for the default (radian) and the
  legacy (degree) angle weights;
* ``DEMO_DIR/noise_parity.json``: the numbers.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ANGLES = (5, 15, 25)


def load(name):
    path = REPO / "tests" / "fixtures" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def spectrum(vol, dt):
    tr = vol.reshape(-1, vol.shape[-1]).astype(np.float64)
    amp = np.abs(np.fft.rfft(tr, axis=1)).mean(axis=0) / (tr.std() * np.sqrt(tr.shape[1]))
    return np.fft.rfftfreq(tr.shape[1], dt), amp


def kurt(x):
    x = x.astype(np.float64).ravel()
    return float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3)


def main():
    demo, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    gn = load("generate_seismic_noise")
    gf = load("generate_seismic_filters")
    meta = json.loads((demo / "meta.json").read_text())
    ni, nj, nk = meta["shape"]
    S = meta["noise_seeds"]
    sn_db = meta["snr_db"]
    dt = meta["digi_ms"] / 1000.0
    _, rfc3, seabed = gn.load_demo(demo)  # (3, ni, nj, nk - 1)

    rust = {
        mode: np.stack(
            [
                np.fromfile(demo / f"noise_{mode}_{a}.f32", "<f4").reshape(S, ni, nj, nk)[..., : nk - 1]
                for a in ANGLES
            ],
            axis=1,
        )  # (S, 3, ni, nj, nk - 1)
        for mode in ("radians", "legacy")
    }
    legacy = {
        mode: np.stack(
            [gn.legacy_add_noise(rfc3, seabed, sn_db, s, radians=(mode == "radians"))[0] for s in range(1, S + 1)]
        )
        for mode in ("radians", "legacy")
    }

    report = {"meta": meta, "legacy_data_std": gn.legacy_data_std(rfc3, seabed), "modes": {}}
    for mode in ("radians", "legacy"):
        r, l = rust[mode], legacy[mode]
        rows = {}
        for x, a in enumerate(ANGLES):
            rows[str(a)] = {
                "rust_mean": float(r[:, x].mean()),
                "legacy_mean": float(l[:, x].mean()),
                "rust_std": float(r[:, x].std()),
                "legacy_std": float(l[:, x].std()),
                "rust_kurtosis": float(np.mean([kurt(r[s, x]) for s in range(S)])),
                "legacy_kurtosis": float(np.mean([kurt(l[s, x]) for s in range(S)])),
            }
        rows["corr_5_25"] = {
            "rust": float(np.mean([np.corrcoef(r[s, 0].ravel(), r[s, 2].ravel())[0, 1] for s in range(S)])),
            "legacy": float(np.mean([np.corrcoef(l[s, 0].ravel(), l[s, 2].ravel())[0, 1] for s in range(S)])),
        }
        report["modes"][mode] = rows

    # Chain parity: Rust (rfc + noise -> bandpass -> lateral) vs legacy
    # postprocess fed the Rust rfc + Rust noise (same f32 sum).
    rfc15 = np.fromfile(demo / "rfc_15.f32", "<f4").reshape(ni, nj, nk)
    n15 = np.fromfile(demo / "noise_radians_15.f32", "<f4").reshape(S, ni, nj, nk)[0]
    noisy = (rfc15 + n15).astype(np.float32)
    chain = np.fromfile(demo / "chain_noise_bp.f32", "<f4").reshape(ni, nj, nk)
    chain_alt = np.fromfile(demo / "chain_noise_bp_alt.f32", "<f4").reshape(ni, nj, nk)
    leg_chain = gf.legacy_filter(noisy, meta["low"], meta["high"], meta["order"], meta["lateral"])
    quiet = gf.legacy_filter(rfc15, meta["low"], meta["high"], meta["order"], meta["lateral"])
    d = chain.astype(np.float64) - leg_chain.astype(np.float64)
    report["chain_parity"] = {
        "max_abs_err": float(np.abs(d).max()),
        "peak": float(np.abs(leg_chain).max()),
        "bit_exact_fraction": float((chain.view(np.uint32) == leg_chain.view(np.uint32)).mean()),
        "samples": int(chain.size),
        "tiles_16x16_vs_5x7_bit_identical": bool((chain.view(np.uint32) == chain_alt.view(np.uint32)).all()),
        "snr_after_bandpass_db": float(10 * np.log10((quiet**2).mean() / ((leg_chain - quiet) ** 2).mean())),
    }
    (demo / "noise_parity.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "meta"}, indent=1))

    # ---- slices ----------------------------------------------------------
    i = ni // 2
    rl = gn.legacy_add_noise(rfc3, seabed, sn_db, 1, radians=True)[0][1]
    panels = [
        ("Rust noise (seed 1, 15°)", n15[i, :, : nk - 1]),
        ("legacy add_weighted_noise (seed 1, 15°)", rl[i]),
        ("raw reflectivity 15°", rfc15[i]),
        ("reflectivity + Rust noise", noisy[i]),
        ("Rust: + bandpass + lateral", chain[i]),
        (f"Rust − legacy chain (max {report['chain_parity']['max_abs_err']:.1e})", d[i]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, img) in zip(axes.ravel(), panels):
        v = np.percentile(np.abs(img), 99.5) or 1e-12
        ax.imshow(img.T, aspect="auto", cmap="seismic", vmin=-v, vmax=v)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("crossline")
        ax.set_ylabel("sample")
    fig.suptitle(
        f"Seismic noise, inline {i}: S/N {sn_db} dB, data_std={meta['data_std']:.3e} "
        f"(legacy {report['legacy_data_std']:.3e}); chain bit-exact "
        f"{100 * report['chain_parity']['bit_exact_fraction']:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(out / "noise_slices.png", dpi=110)

    # ---- spectra / histograms -------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, mode in enumerate(("radians", "legacy")):
        r, l = rust[mode], legacy[mode]
        ax = axes[row, 0]
        for x, a in enumerate(ANGLES):
            f, ra = spectrum(r[:, x], dt)
            _, la = spectrum(l[:, x], dt)
            ax.plot(f, ra, lw=1.2, label=f"Rust {a}°")
            ax.plot(f, la, "--", lw=1, label=f"legacy {a}°")
        ax.set_ylim(0.8, 0.95)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("mean |FFT| / (std·√n)")
        ax.set_title(f"{mode} weights: amplitude spectrum (white)")
        ax.legend(fontsize=7, ncol=2)
        ax = axes[row, 1]
        bins = np.linspace(-6, 6, 121)
        for x, a in enumerate(ANGLES):
            sr = r[:, x].std()
            hr, _ = np.histogram(r[:, x].ravel() / sr, bins, density=True)
            hl, _ = np.histogram(l[:, x].ravel() / l[:, x].std(), bins, density=True)
            c = 0.5 * (bins[1:] + bins[:-1])
            ax.semilogy(c, hr, lw=1.2, label=f"Rust {a}° (κ={report['modes'][mode][str(a)]['rust_kurtosis']:.2f})")
            ax.semilogy(c, hl, "--", lw=1, label=f"legacy {a}° (κ={report['modes'][mode][str(a)]['legacy_kurtosis']:.2f})")
        ax.set_xlabel("amplitude / std")
        ax.set_title(f"{mode} weights: amplitude pdf ({S} seeds each)")
        ax.legend(fontsize=7)
        ax = axes[row, 2]
        rows = report["modes"][mode]
        xs = np.arange(len(ANGLES))
        ax.bar(xs - 0.2, [rows[str(a)]["rust_std"] for a in ANGLES], 0.4, label="Rust std")
        ax.bar(xs + 0.2, [rows[str(a)]["legacy_std"] for a in ANGLES], 0.4, label="legacy std")
        ax.set_xticks(xs, [f"{a}°" for a in ANGLES])
        ax.set_ylim(0, 1.3 * max(rows[str(a)]["legacy_std"] for a in ANGLES))
        ax.set_title(
            f"std (target data_std/std_ratio); corr(5°,25°) Rust {rows['corr_5_25']['rust']:.3f} "
            f"legacy {rows['corr_5_25']['legacy']:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
    fig.suptitle(f"Rust counter-based noise vs legacy add_weighted_noise ({S} seeds, S/N {sn_db} dB)")
    fig.tight_layout()
    fig.savefig(out / "noise_spectra.png", dpi=110)
    print("wrote", out / "noise_slices.png", out / "noise_spectra.png")


if __name__ == "__main__":
    main()
