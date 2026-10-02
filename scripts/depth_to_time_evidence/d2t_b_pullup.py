"""Salt pull-up on whole label samples (4 ms) vs predicted from the velocities.

Superseded for the measured values by d2t_b_pullup_subsample.py (16x FFT
sub-sample picks); kept for the label picks and the legacy-axis column.

For a sub-salt horizon L (top of layer id L: the layer whose top lies below the salt base in
the most salt-footprint columns; only those columns are used), per salt-footprint column:
  predicted = T_nosalt(z_L nosalt) - T_salt(z_L salt)   (cell-boundary TWT of
              each run's own horizon, from its Vp; includes the salt drag)
  measured  = t(first output sample of layer L, --no-salt run)
            - t(first output sample of layer L, salt run)
on the time deliverable and on the legacy (depth-as-time) deliverable.
"""
import json, os
import numpy as np

# Cubes from `cargo run --release -p synthoseis-core --example d2t_evidence`.
D = os.environ.get("D2T_EVIDENCE_DIR", "/workspace/.d2t-work/evidence")
OUT = os.environ.get("D2T_EVIDENCE_OUT", "/workspace/synthoseis-bench/out/d2t-b")

def load(tag):
    m = json.load(open(f"{D}/{tag}.json"))
    ni, nj, nz = m["depth_shape"]
    nt = m["time_shape"][2]
    r = dict(meta=m)
    r["labels"] = np.fromfile(f"{D}/{tag}_depth_labels.u8", np.uint8).reshape(ni, nj, nz)
    r["twt"] = np.fromfile(f"{D}/{tag}_twt.f32", np.float32).reshape(ni, nj, nz + 1)
    for mode, n in (("legacy", nz), ("time", nt)):
        r[f"{mode}_labels"] = np.fromfile(f"{D}/{tag}_{mode}_labels.u8", np.uint8).reshape(ni, nj, n)
        r[f"{mode}_stack"] = np.fromfile(f"{D}/{tag}_{mode}_stack.f32", np.float32).reshape(ni, nj, n)
    try:
        r["salt"] = np.fromfile(f"{D}/{tag}_depth_salt.u8", np.uint8).reshape(ni, nj, nz)
        for mode, n in (("legacy", nz), ("time", nt)):
            r[f"{mode}_salt"] = np.fromfile(f"{D}/{tag}_{mode}_salt.u8", np.uint8).reshape(ni, nj, n)
    except FileNotFoundError:
        pass
    return r

def first_index(cube, L):
    """First sample with label == L per column (-1 if none)."""
    hit = cube == L
    k = hit.argmax(axis=2)
    k[~hit.any(axis=2)] = -1
    return k

def pullup(seed):
    s, n = load(f"seed{seed}"), load(f"seed{seed}_nosalt")
    dt = s["meta"]["dt_ms"]; digi = s["meta"]["digi_legacy"]
    salt = s["salt"] == 1
    foot = salt.any(axis=2)
    nz = salt.shape[2]
    base = np.where(foot, nz - 1 - salt[:, :, ::-1].argmax(axis=2), -1)
    nl = int(s["labels"][s["labels"] != 255].max())
    best = None
    for L in range(1, nl + 1):
        zs, zn = first_index(s["labels"], L), first_index(n["labels"], L)
        ok = foot & (zs > base) & (zs >= 0) & (zn >= 0)
        if best is None or ok.sum() > best[0]:
            best = (ok.sum(), L, zs, zn, ok)
    _, L, zs, zn, use = best
    full = foot
    foot = use
    ii, jj = np.nonzero(foot)
    pred = n["twt"][ii, jj, zn[ii, jj]] - s["twt"][ii, jj, zs[ii, jj]]
    ts, tn = first_index(s["time_labels"], L), first_index(n["time_labels"], L)
    ls_, ln = first_index(s["legacy_labels"], L), first_index(n["legacy_labels"], L)
    valid = (ts[ii, jj] >= 0) & (tn[ii, jj] >= 0)   # horizon inside both traces
    meas = (tn[ii, jj] - ts[ii, jj]) * dt
    leg = (ln[ii, jj] - ls_[ii, jj]) * digi
    thick = salt.sum(axis=2)[ii, jj] * s["meta"]["dz"]
    v = valid
    row = dict(seed=seed, horizon=L, salt_columns=int(full.sum()), columns=int(foot.sum()), in_trace=int(v.sum()),
               max_salt_m=float(thick.max()),
               pred_med=float(np.median(pred[v])), pred_max=float(pred[v].max()),
               time_med=float(np.median(meas[v])), time_max=float(meas[v].max()),
               legacy_med=float(np.median(leg[v])), legacy_max=float(leg[v].max()),
               err_max=float(np.abs(meas[v] - pred[v]).max()),
               err_mean=float((meas[v] - pred[v]).mean()))
    k = np.argmax(np.where(v, thick, -1))
    row["thickest"] = dict(i=int(ii[k]), j=int(jj[k]), salt_m=float(thick[k]), pred=float(pred[k]),
                           time=float(meas[k]), legacy=float(leg[k]))
    return row, dict(ii=ii, jj=jj, pred=pred, meas=meas, leg=leg, valid=v, L=L, zs=zs, ts=ts, ls=ls_)

if __name__ == "__main__":
    rows = [pullup(s)[0] for s in (1, 2, 3, 7, 30)]
    json.dump(rows, open(f"{OUT}/pullup_labels.json", "w"), indent=1)
    for r in rows:
        print(json.dumps(r))
