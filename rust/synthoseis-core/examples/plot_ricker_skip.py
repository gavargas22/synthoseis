"""Ricker-skip parity + before/after spectra for `cargo run --example filters_demo`.

    python plot_ricker_skip.py DEMO_DIR OUT_PNG

Legacy `postprocess_rfc_cubes` bandpasses the raw reflectivity (no wavelet):
the Butterworth filter *is* the wavelet. The Rust pipeline now skips the
Ricker convolution when the bandpass is on (`FilterConfig::keep_ricker` /
`--keep-ricker` restores Ricker + bandpass). This script

* feeds the Rust raw reflectivity (`angle_rfc.f32`, the pre-wavelet angle
  stack) to the real legacy `apply_bandlimits` + `apply_lateral_filter`
  (via `tests/fixtures/generate_seismic_filters.py`) and compares with the
  Rust filtered stack (`angle_filtered.f32`, Ricker skipped);
* checks the `keep_ricker` stack against legacy filters applied to the Rust
  Ricker stack (`angle_raw.f32`), i.e. the #25 behaviour is unchanged;
* plots mean trace amplitude spectra before/after (OUT_PNG) and writes
  DEMO_DIR/ricker_skip_parity.json.
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


def compare(rust, legacy):
    diff = rust.astype(np.float64) - legacy.astype(np.float64)
    return {
        "max_abs_err": float(np.abs(diff).max()),
        "max_abs_legacy": float(np.abs(legacy).max()),
        "max_rel_err": float(np.abs(diff).max() / np.abs(legacy).max()),
        "bit_exact_fraction": float((rust.view(np.uint32) == legacy.view(np.uint32)).mean()),
        "max_ulps": int(ulps(rust, legacy).max()),
    }


def ricker(freq, digi_ms, length_s=0.2):
    # Shape only (for the plot); the Rust kernel is synthoseis_seismic::ricker.
    dt = digi_ms / 1000.0
    t = np.arange(-length_s / 2, length_s / 2 + dt / 2, dt)
    y = (1 - 2 * (np.pi * freq * t) ** 2) * np.exp(-((np.pi * freq * t) ** 2))
    return y


def main():
    d = Path(sys.argv[1])
    out_png = sys.argv[2]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["ricker_skipped"], "demo must run with the default (Ricker skipped)"
    shape = tuple(meta["shape"])
    load = lambda n: np.fromfile(d / n, "<f4").reshape(shape)  # noqa: E731
    rfc, ricker_stack = load("angle_rfc.f32"), load("angle_raw.f32")
    new, alt, old = load("angle_filtered.f32"), load("angle_filtered_alt.f32"), load(
        "angle_filtered_keep.f32")
    gen = load_generator()
    low, high, lat, order, digi = (meta["low"], meta["high"], meta["lateral"], meta["order"],
                                   meta["digi_ms"])
    legacy_new = gen.legacy_filter(rfc, low, high, order, lat, digi)
    legacy_old = gen.legacy_filter(ricker_stack, low, high, order, lat, digi)

    dt = digi / 1000.0
    f, a_rfc = spectrum(rfc, dt)
    _, a_ricker = spectrum(ricker_stack, dt)
    _, a_old = spectrum(old, dt)
    _, a_new = spectrum(new, dt)
    _, a_leg = spectrum(legacy_new, dt)
    b, a = gen.derive_butterworth_bandpass(low, high, digi, order=order)
    _, h = freqz(b, a, worN=f, fs=1.0 / dt)
    h2 = np.abs(h) ** 2
    band = (f >= low) & (f <= high)
    peak = lambda amp: float(f[np.argmax(amp)])  # noqa: E731
    parity = {
        "shape": list(shape),
        "filters": {"low": low, "high": high, "order": order, "lateral": lat},
        "ricker_skip_vs_legacy_on_rust_rfc": compare(new, legacy_new),
        "keep_ricker_vs_legacy_on_rust_ricker_stack": compare(old, legacy_old),
        "spectrum_max_rel_diff_in_band_skip_vs_legacy": float(
            (np.abs(a_new - a_leg)[band] / a_leg[band]).max()),
        "tiles_16x16_vs_5x7_bit_identical": bool(
            np.array_equal(new.view(np.uint32), alt.view(np.uint32))),
        "peak_frequency_hz": {"rfc": peak(a_rfc), "ricker_stack": peak(a_ricker),
                              "ricker_plus_bandpass_old": peak(a_old),
                              "rfc_plus_bandpass_new": peak(a_new)},
        "rms": {k: float(np.sqrt((v.astype(np.float64) ** 2).mean()))
                for k, v in {"rfc": rfc, "ricker_stack": ricker_stack,
                             "ricker_plus_bandpass_old": old,
                             "rfc_plus_bandpass_new": new}.items()},
    }
    (d / "ricker_skip_parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2))

    w = ricker(meta.get("ricker_hz", 40.0), digi)
    _, hw = freqz(w, [1.0], worN=f, fs=1.0 / dt)
    hw = np.abs(hw) / np.abs(hw).max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ax = axes[0]
    ax.plot(f, a_rfc / a_rfc.max(), color="0.6", label="raw reflectivity (no wavelet)")
    ax.plot(f, a_ricker / a_ricker.max(), color="C1", label="Ricker 40 Hz stack (filters off)")
    ax.plot(f, hw, ":", color="C1", label="|W(f)| Ricker")
    ax.plot(f, h2, ":", color="C3", label="|H(f)|^2 filtfilt")
    for x in (low, high):
        ax.axvline(x, color="C3", lw=0.5)
    ax.set_title("Inputs (each normalised to its max)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean amplitude (normalised)")
    ax.legend(fontsize=8)

    ax = axes[1]
    norm = a_new.max()
    ax.plot(f, a_old / a_old.max(), color="C1", lw=2,
            label=f"before (#25): Ricker + bandpass, peak {peak(a_old):.1f} Hz")
    ax.plot(f, a_leg / norm, color="C0", lw=4, alpha=0.4,
            label="legacy postprocess on Rust reflectivity")
    ax.plot(f, a_new / norm, "--", color="k",
            label=f"after: reflectivity + bandpass (Rust), peak {peak(a_new):.1f} Hz")
    ax.plot(f, h2, ":", color="C3", label="|H(f)|^2")
    for x in (low, high):
        ax.axvline(x, color="C3", lw=0.5)
    ax.set_xlim(0, min(3 * high, f[-1]))
    ax.set_title(f"Filtered spectra ({low:g}-{high:g} Hz order {order}, lateral {lat})")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=8)

    ax = axes[2]
    rel = np.abs(a_new - a_leg) / np.maximum(a_leg, 1e-30)
    ax.semilogy(f, np.maximum(rel, 1e-18), label="|Rust - legacy| / legacy (skip)")
    ratio = (a_old / a_old.max()) / np.maximum(a_new / a_new.max(), 1e-30)
    ax.semilogy(f, np.maximum(ratio, 1e-18), color="C1",
                label="before / after (normalised) = extra Ricker shaping")
    ax.axvspan(low, high, color="C3", alpha=0.1, label="passband")
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("Parity (floor 1e-18 = identical) and the Ricker's effect")
    ax.legend(fontsize=8)
    p = parity["ricker_skip_vs_legacy_on_rust_rfc"]
    fig.suptitle(
        f"Ricker skip on a faulted {shape[0]}x{shape[1]}x{shape[2]} cube (15 deg): "
        f"Rust vs legacy postprocess on the same reflectivity: bit-exact fraction "
        f"{p['bit_exact_fraction']:.4f}, max |diff| {p['max_abs_err']:.1e}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
