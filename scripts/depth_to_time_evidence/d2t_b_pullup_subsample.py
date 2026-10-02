"""Sub-sample salt pull-up: 16x FFT-upsampled event picks (the #37 salt-test pick).

Horizon and columns: the sub-salt horizon L (top below the salt base, both
runs' events inside the trace) with the most salt-footprint columns whose
event *dominates* on the raw reflectivity trace of both runs: the larger
|x| of samples n_lab-1, n_lab is >= 2x every other |x| within +-6 samples
(amplitude QC only; no timing information is used). Those columns are
measured. Interference from neighbouring interfaces is what this excludes:
a weak contrast next to a strong one is picked on the strong one.
Pooled check: every sub-salt horizon x column passing the same QC.

Per column, in both the salt and the --no-salt run:
  1. Locate: n_lab = first output sample of layer L in the *time label cube*
     (the reflection of the interface lies in (n_lab - 1, n_lab]).
  2. Polarity: sign of the largest |x| among samples n_lab-1 .. n_lab.
  3. Pick (#37 `pick_peak`): coarse extremum of pol*x in n_lab-2 .. n_lab+1,
     then the extremum of the 16x FFT-upsampled trace (spectrum zero padding,
     Nyquist bin split) in (16(n-1), 16(n+1)), refined by a parabola through
     the three best fine samples.
  t_pick = refined * dt. Measured pull-up = t_nosalt - t_salt.
Traces: the production time-mode fuse without a wavelet (raw TWT
reflectivity, primary) and the deliverable stack (Ricker 40 Hz, cross-check).
Predicted = T_nosalt(z_L) - T_salt(z_L) from Vp (d2t_b_pullup.py).
"""
import json
import numpy as np
from d2t_b_pullup import load, pullup, first_index, D, OUT

UP = 16

def fft_upsample_16(y):
    """Rows of y (..., N) -> (..., 16N); identical to #37's direct-DFT version."""
    n = y.shape[-1]
    assert n % 2 == 0
    Y = np.fft.rfft(y.astype(np.float64), axis=-1)
    m = UP * n
    Z = np.zeros(y.shape[:-1] + (m // 2 + 1,), complex)
    Z[..., : n // 2 + 1] = Y
    Z[..., n // 2] *= 0.5  # Nyquist bin split between +-N/2
    return np.fft.irfft(Z, m, axis=-1) * UP

def pick(traces, n_lab):
    """traces (C, N), n_lab (C,) -> (t in samples, polarity)."""
    C, N = traces.shape
    fine = fft_upsample_16(traces)
    assert np.allclose(fine[:, ::UP], traces, atol=1e-6 * np.abs(traces).max())
    out = np.full(C, np.nan); pol = np.zeros(C)
    for c in range(C):
        y, nl = traces[c].astype(np.float64), int(n_lab[c])
        if nl < 3 or nl > N - 3:
            continue
        w = y[nl - 1 : nl + 1]
        p = np.sign(w[np.argmax(np.abs(w))]) or 1.0
        n = nl - 2 + int(np.argmax(p * y[nl - 2 : nl + 2]))
        f = p * fine[c]
        s = UP * (n - 1) + 1 + int(np.argmax(f[UP * (n - 1) + 1 : UP * (n + 1)]))
        a, b, cc = f[s - 1], f[s], f[s + 1]
        out[c] = (s + 0.5 * (a - cc) / (a - 2 * b + cc)) / UP
        pol[c] = p
    return out, pol

def stats(x):
    return dict(mean=float(x.mean()), std=float(x.std()), max_abs=float(np.abs(x).max()))

DOM_MIN, DOM_HALF = 2.0, 6

def dominance(tr, n_lab):
    out = np.zeros(len(n_lab))
    for c, n in enumerate(n_lab):
        n = int(n)
        if n < DOM_HALF + 2 or n > tr.shape[1] - DOM_HALF - 2:
            continue
        y = np.abs(tr[c])
        nb = np.r_[y[n - 1 - DOM_HALF : n - 1], y[n + 1 : n + 1 + DOM_HALF]].max()
        out[c] = y[n - 1 : n + 1].max() / max(nb, 1e-30)
    return out

def runs(seed):
    s, n = load(f"seed{seed}"), load(f"seed{seed}_nosalt")
    nt = s["meta"]["nt"]
    for r, tag in ((s, f"seed{seed}"), (n, f"seed{seed}_nosalt")):
        r["time_rfc"] = np.fromfile(f"{D}/{tag}_time_rfc.f32", np.float32).reshape(-1, nt)
    return s, n

def horizon_columns(s, n, L, foot, base):
    nj = s["labels"].shape[1]; nt = s["meta"]["nt"]
    ii, jj = np.nonzero(foot)
    zs, zn = first_index(s["labels"], L)[ii, jj], first_index(n["labels"], L)[ii, jj]
    ts, tn = first_index(s["time_labels"], L)[ii, jj], first_index(n["time_labels"], L)[ii, jj]
    ok = (zs > base[ii, jj]) & (zs >= 0) & (zn >= 0) & (ts >= 0) & (tn >= 0)
    idx = ii * nj + jj
    d = np.minimum(dominance(s["time_rfc"][idx], ts), dominance(n["time_rfc"][idx], tn))
    return dict(ii=ii, jj=jj, idx=idx, zs=zs, zn=zn, ts=ts, tn=tn, ok=ok & (d >= DOM_MIN), d=d)

def measure_horizon(s, n, L, h):
    dt = s["meta"]["dt_ms"]
    ii, jj, idx, ok = h["ii"][h["ok"]], h["jj"][h["ok"]], h["idx"][h["ok"]], h["ok"]
    pred = n["twt"][ii, jj, h["zn"][ok]] - s["twt"][ii, jj, h["zs"][ok]]
    legacy = (first_index(n["legacy_labels"], L)[ii, jj] - first_index(s["legacy_labels"], L)[ii, jj]) * s["meta"]["digi_legacy"]
    out = dict(pred=pred, legacy=legacy)
    for kind in ("rfc", "stack"):
        t = {}
        for name, r, tl, z in (("salt", s, h["ts"], h["zs"]), ("nosalt", n, h["tn"], h["zn"])):
            tr = r["time_rfc"] if kind == "rfc" else r["time_stack"].reshape(-1, r["meta"]["nt"])
            t[name], t[name + "_pol"] = pick(tr[idx], tl[ok])
            nn = np.clip(np.nan_to_num(np.round(t[name])).astype(int), 0, tr.shape[1] - 1)
            t[name + "_amp"] = np.abs(tr[idx, nn])
            t[name + "_abs"] = t[name] * dt - r["twt"][ii, jj, z[ok]]
        meas = (t["nosalt"] - t["salt"]) * dt
        good = np.isfinite(meas) & (t["salt_pol"] == t["nosalt_pol"])
        out[kind] = dict(meas=meas, good=good, abs_salt=t["salt_abs"], abs_nosalt=t["nosalt_abs"],
                         amp=np.minimum(t["salt_amp"], t["nosalt_amp"]))
    return out

def measure(seed):
    s, n = runs(seed)
    salt = s["salt"] == 1; foot = salt.any(axis=2); nz = salt.shape[2]
    base = np.where(foot, nz - 1 - salt[:, :, ::-1].argmax(axis=2), -1)
    nl = int(s["labels"][s["labels"] != 255].max())
    hs = {L: horizon_columns(s, n, L, foot, base) for L in range(1, nl + 1)}
    L = max(hs, key=lambda k: hs[k]["ok"].sum())
    m = measure_horizon(s, n, L, hs[L])
    g = m["rfc"]["good"]; pred = m["pred"][g]; meas = m["rfc"]["meas"][g]
    res = dict(seed=seed, horizon=int(L), salt_columns=int(foot.sum()), columns=int(g.sum()),
               pred_med=float(np.median(pred)), pred_max=float(pred.max()),
               meas_med=float(np.median(meas)), meas_max=float(meas.max()),
               resid=stats(meas - pred),
               abs_pick_err=stats(np.r_[m["rfc"]["abs_salt"][g], m["rfc"]["abs_nosalt"][g]]))
    res["event_rc_median"] = float(np.median(m["rfc"]["amp"][g]))
    k = int(np.argmax(pred)); res["at_max_pred"] = dict(pred=float(pred[k]), meas=float(meas[k]))
    gs = m["stack"]["good"]
    res["stack_check"] = dict(columns=int(gs.sum()), resid=stats(m["stack"]["meas"][gs] - m["pred"][gs]))
    # Pooled: every sub-salt horizon x column passing the QC.
    pool = []
    for LL, h in hs.items():
        if h["ok"].sum() == 0:
            continue
        mm = measure_horizon(s, n, LL, h)
        gg = mm["rfc"]["good"]
        pool.append(mm["rfc"]["meas"][gg] - mm["pred"][gg])
    pool = np.concatenate(pool)
    res["pooled"] = dict(picks=int(pool.size), resid=stats(pool),
                         within_1ms=float((np.abs(pool) <= 1).mean()))
    leg = m["legacy"][g]
    res["legacy_med"] = float(np.median(leg)); res["legacy_max"] = float(leg.max())
    return res, dict(pred=pred, meas=meas, legacy=leg, L=L)

if __name__ == "__main__":
    rows = [measure(s)[0] for s in (1, 2, 3, 7, 30)]
    json.dump(rows, open(f"{OUT}/pullup_subsample.json", "w"), indent=1)
    for r in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
