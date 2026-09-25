"""Legacy parity check + plots for `cargo run --example filters_demo`.

    python plot_filters_demo.py DEMO_DIR OUT_PREFIX

Filters the *unfiltered* Rust stack (`angle_raw.f32`) with the real legacy
code (`SeismicVolume.apply_bandlimits` + `apply_lateral_filter`, via
`tests/fixtures/generate_seismic_filters.py`), compares it with the Rust
filtered stack (`angle_filtered.f32`) and writes

* OUT_PREFIX_before_after.png: inline / crossline / time slices before and
  after filtering, plus the Rust - legacy difference,
* OUT_PREFIX_spectra.png: mean trace amplitude spectra (raw, Rust, legacy)
  and the ideal filtfilt response |H(f)|^2,
* DEMO_DIR/parity.json: max abs / relative error, bit-exact fraction, ULPs.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import freqz  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def load_generator():
    path = REPO / "tests" / "fixtures" / "generate_seismic_filters.py"
    spec = importlib.util.spec_from_file_location("generate_seismic_filters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ulps(a, b):
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**31)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**31)) - ib, ib)
    return np.abs(ia - ib)


def spectrum(vol, dt):
    tr = vol.reshape(-1, vol.shape[-1]).astype(np.float64)
    amp = np.abs(np.fft.rfft(tr, axis=1)).mean(axis=0)
    return np.fft.rfftfreq(vol.shape[-1], dt), amp


def main():
    d = Path(sys.argv[1])
    prefix = sys.argv[2]
    meta = json.loads((d / "meta.json").read_text())
    shape = tuple(meta["shape"])
    raw = np.fromfile(d / "angle_raw.f32", "<f4").reshape(shape)
    rust = np.fromfile(d / "angle_filtered.f32", "<f4").reshape(shape)
    alt = np.fromfile(d / "angle_filtered_alt.f32", "<f4").reshape(shape)
    gen = load_generator()
    low, high, lat, order = meta["low"], meta["high"], meta["lateral"], meta["order"]
    legacy = gen.legacy_filter(raw, low, high, order, lat, meta["digi_ms"])

    diff = rust.astype(np.float64) - legacy.astype(np.float64)
    u = ulps(rust, legacy)
    dt = meta["digi_ms"] / 1000.0
    f, a_raw = spectrum(raw, dt)
    _, a_rust = spectrum(rust, dt)
    _, a_leg = spectrum(legacy, dt)
    band = (f >= low) & (f <= high)
    b, a = gen.derive_butterworth_bandpass(low, high, meta["digi_ms"], order=order)
    _, h = freqz(b, a, worN=f, fs=1.0 / dt)
    parity = {
        "shape": list(shape),
        "filters": {"low": low, "high": high, "order": order, "lateral": lat},
        "max_abs_err": float(np.abs(diff).max()),
        "max_abs_legacy": float(np.abs(legacy).max()),
        "max_rel_err": float(np.abs(diff).max() / np.abs(legacy).max()),
        "rms_rel_err": float(np.sqrt((diff**2).mean() / (legacy.astype(np.float64) ** 2).mean())),
        "bit_exact_fraction": float((rust.view(np.uint32) == legacy.view(np.uint32)).mean()),
        "max_ulps": int(u.max()),
        "spectrum_max_rel_diff_in_band": float(
            (np.abs(a_rust - a_leg)[band] / a_leg[band]).max()),
        "tiles_16x16_vs_5x7_bit_identical": bool(np.array_equal(rust.view(np.uint32),
                                                                alt.view(np.uint32))),
        "rms_raw": float(np.sqrt((raw.astype(np.float64) ** 2).mean())),
        "rms_filtered": float(np.sqrt((rust.astype(np.float64) ** 2).mean())),
    }
    (d / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2))

    ni, nj, nk = shape
    i0, j0 = ni // 2, nj // 2
    k0 = int(np.abs(raw).mean(axis=(0, 1)).argmax())  # strongest reflector
    vmax = np.percentile(np.abs(raw), 99)
    vmax_f = np.percentile(np.abs(rust), 99)
    t = np.arange(nk) * meta["digi_ms"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    rows = [
        (f"inline {i0}", lambda v: v[i0].T, "crossline", (0, nj, t[-1], 0)),
        (f"crossline {j0}", lambda v: v[:, j0].T, "inline", (0, ni, t[-1], 0)),
        (f"time slice {t[k0]:.0f} ms", lambda v: v[:, :, k0], "crossline", (0, nj, ni, 0)),
    ]
    dmax = max(np.abs(diff).max(), 1e-30)
    for r, (name, sl, xl, ext) in enumerate(rows):
        panels = [
            (sl(raw), "unfiltered", vmax, "gray_r"),
            (sl(rust), f"Rust filtered ({low:g}-{high:g} Hz, lateral {lat})", vmax_f, "gray_r"),
            (sl(legacy), "legacy Python filtered", vmax_f, "gray_r"),
            (sl(diff), f"Rust - legacy (max {dmax:.1e})", dmax, "RdBu_r"),
        ]
        for c, (img, title, vm, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto", extent=ext,
                           interpolation="nearest")
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xlabel(xl)
            ax.set_ylabel("inline" if r == 2 else "time (ms)")
            if c == 3:
                fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Post-convolution filters on a faulted {ni}x{nj}x{nk} cube (15 deg stack, "
        f"{meta['faults']} faults requested): bit-exact fraction "
        f"{parity['bit_exact_fraction']:.4f}, max |Rust-legacy| {parity['max_abs_err']:.2e} "
        f"(rel {parity['max_rel_err']:.1e})")
    fig.tight_layout()
    fig.savefig(f"{prefix}_before_after.png", dpi=110)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(f, a_raw / a_raw.max(), label="unfiltered", color="0.5")
    ax1.plot(f, a_leg / a_raw.max(), label="legacy filtered", lw=3, alpha=0.5)
    ax1.plot(f, a_rust / a_raw.max(), "--", label="Rust filtered", color="k")
    ax1.plot(f, np.abs(h) ** 2, ":", label="|H(f)|^2 (filtfilt ideal)", color="C3")
    for x in (low, high):
        ax1.axvline(x, color="C3", lw=0.5)
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("mean amplitude (normalised)")
    ax1.set_title("Mean trace amplitude spectrum")
    ax1.legend()
    rel = np.abs(a_rust - a_leg) / np.maximum(a_leg, 1e-30)
    ax2.semilogy(f, np.maximum(rel, 1e-18))
    ax2.axvspan(low, high, color="C3", alpha=0.1, label="passband")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("|Rust - legacy| / legacy")
    ax2.set_title("Relative spectrum difference (floor 1e-18 = identical)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_spectra.png", dpi=110)
    plt.close(fig)
    print(f"{prefix}_before_after.png", f"{prefix}_spectra.png")


if __name__ == "__main__":
    main()
