#!/usr/bin/env python3
"""Regenerate seismic_kernels.json + rpm_trends.json goldens.

Mirrors:
  tests._zoeppritz_reference.zoeppritz_vectorised
  datagenerator.wavelets.ricker / hanflat
  datagenerator.Seismic SNR / Hilterman weights
  rockphysics.rpm_example / rpm_tagilsk_trends (subset)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def zoeppritz_vectorised(vp1, vs1, rho1, vp2, vs2, rho2, angle_deg):
    theta = np.deg2rad(np.float64(angle_deg)) + 0j
    p = np.sin(theta) / vp1
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)
    phi2 = np.arcsin(p * vs2)
    sin_phi1_sq = np.sin(phi1) ** 2
    sin_phi2_sq = np.sin(phi2) ** 2
    cos_theta = np.cos(theta)
    cos_theta2 = np.cos(theta2)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)
    a = rho2 * (1 - 2 * sin_phi2_sq) - rho1 * (1 - 2 * sin_phi1_sq)
    b = rho2 * (1 - 2 * sin_phi2_sq) + 2 * rho1 * sin_phi1_sq
    c = rho1 * (1 - 2 * sin_phi1_sq) + 2 * rho2 * sin_phi2_sq
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    e = b * cos_theta / vp1 + c * cos_theta2 / vp2
    f = b * cos_phi1 / vs1 + c * cos_phi2 / vs2
    g = a - d * cos_theta / vp1 * cos_phi2 / vs2
    h = a - d * cos_theta2 / vp2 * cos_phi1 / vs1
    det = e * f + g * h * p**2
    zoep_pp = (
        f * (b * cos_theta / vp1 - c * cos_theta2 / vp2)
        - h * p**2 * (a + det * cos_theta / vp1 * cos_phi2 / vs2)
    ) / det
    return np.real(zoep_pp).astype(np.float32, copy=True)


def hanflat(inarray, pctflat):
    numsamples = len(inarray)
    lowflatindex = int(round(numsamples * (1.0 - pctflat) / 2.0))
    hiflatindex = numsamples - lowflatindex
    hanwgt = np.hanning(len(inarray) - (hiflatindex - lowflatindex))
    outarray = np.ones(len(inarray), dtype=float)
    outarray[:lowflatindex] = hanwgt[:lowflatindex]
    outarray[hiflatindex:] = hanwgt[numsamples - hiflatindex :]
    return outarray


def ricker(f, dt, convolutions=2):
    lenhalf = 1250.0 / f
    halfsmp = int(lenhalf / dt) + 1
    lenhalf = halfsmp * dt
    t = np.arange(-lenhalf, lenhalf + dt, dt) / 1000.0
    s = (1 - 2 * np.pi**2 * f**2 * t**2) * np.exp(-(np.pi**2) * f**2 * t**2)
    for _ in range(convolutions - 1):
        s = np.convolve(s, s, mode="full")
    return s * hanflat(s, 0.50)


def snr_std_ratio(sn_db):
    return float(np.sqrt(10 ** (sn_db / 10.0)))


def hilterman(angle_deg):
    ang = np.deg2rad(angle_deg)
    return float(np.cos(ang) ** 2), float(np.sin(ang) ** 2)


def round_nested(obj, nd=10):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [round_nested(x, nd) for x in obj]
    if isinstance(obj, dict):
        return {k: round_nested(v, nd) for k, v in obj.items()}
    return obj


def build_seismic():
    ni, nj, nk = 4, 4, 5
    vp = np.zeros((ni, nj, nk), np.float32)
    vs = np.zeros((ni, nj, nk), np.float32)
    rho = np.zeros((ni, nj, nk), np.float32)
    for k in range(nk):
        if k < 2:
            vp[:, :, k] = 2200.0 + 10.0 * k
            vs[:, :, k] = 900.0 + 5.0 * k
            rho[:, :, k] = 2.15 + 0.01 * k
        else:
            vp[:, :, k] = 2800.0 + 15.0 * (k - 2)
            vs[:, :, k] = 1400.0 + 8.0 * (k - 2)
            rho[:, :, k] = 2.35 + 0.02 * (k - 2)

    angles = [0.0, 15.0, 30.0]
    rfc = np.zeros((len(angles), ni, nj, nk - 1), np.float32)
    for a_i, ang in enumerate(angles):
        rfc[a_i] = zoeppritz_vectorised(
            vp[:, :, :-1],
            vs[:, :, :-1],
            rho[:, :, :-1],
            vp[:, :, 1:],
            vs[:, :, 1:],
            rho[:, :, 1:],
            ang,
        )

    scalar_cases = [
        {
            "vp1": 2200.0,
            "vs1": 900.0,
            "rho1": 2.15,
            "vp2": 2800.0,
            "vs2": 1400.0,
            "rho2": 2.35,
            "angle_deg": ang,
        }
        for ang in (0.0, 15.0, 30.0)
    ]
    scalar_cases.append(
        {
            "vp1": 2000.0,
            "vs1": 800.0,
            "rho1": 2.0,
            "vp2": 3500.0,
            "vs2": 1800.0,
            "rho2": 2.5,
            "angle_deg": 35.0,
        }
    )
    for c in scalar_cases:
        r = float(
            zoeppritz_vectorised(
                np.array(c["vp1"]),
                np.array(c["vs1"]),
                np.array(c["rho1"]),
                np.array(c["vp2"]),
                np.array(c["vs2"]),
                np.array(c["rho2"]),
                c["angle_deg"],
            )
        )
        c["rpp"] = round(r, 10)

    s1 = ricker(30.0, 4.0, convolutions=1)
    s2 = ricker(25.0, 4.0, convolutions=2)
    hf_out = hanflat(np.ones(8), 0.50)
    trace = np.array([0, 0, 1.0, -0.5, 0.25, 0, 0, 0], dtype=float)
    wav = np.array([0.25, 0.5, 0.25], dtype=float)
    conv = np.convolve(trace, wav, mode="same")
    sn_dbs = [0.0, 10.0, 20.0, 30.0]

    return {
        "meta": {
            "shape_props": [ni, nj, nk],
            "shape_rfc": [len(angles), ni, nj, nk - 1],
            "angles_deg": angles,
            "ported_python": [
                "tests._zoeppritz_reference.zoeppritz_vectorised",
                "datagenerator.zoeppritz_kernel.compute_rfc_volumes",
                "datagenerator.wavelets.ricker",
                "datagenerator.wavelets.hanflat",
                "datagenerator.Seismic.apply_wavelet (1D same-mode convolve)",
                "datagenerator.Seismic.add_weighted_noise (snr_std_ratio + Hilterman weights)",
            ],
        },
        "scalar_zoeppritz": scalar_cases,
        "props": {
            "vp": [round(float(x), 6) for x in vp.reshape(-1)],
            "vs": [round(float(x), 6) for x in vs.reshape(-1)],
            "rho": [round(float(x), 6) for x in rho.reshape(-1)],
        },
        "rfc": [round(float(x), 10) for x in rfc.reshape(-1)],
        "ricker_30hz_dt4_c1": [round(float(x), 10) for x in s1],
        "ricker_25hz_dt4_c2_len": int(len(s2)),
        "ricker_25hz_dt4_c2_peak_idx": int(np.argmax(s2)),
        "ricker_25hz_dt4_c2_peak": round(float(np.max(s2)), 10),
        "hanflat_ones8_pct050": [round(float(x), 10) for x in hf_out],
        "convolve_same": {
            "trace": trace.tolist(),
            "wavelet": wav.tolist(),
            "out": [round(float(x), 10) for x in conv],
        },
        "snr": {
            "sn_db": sn_dbs,
            "std_ratio": [round(snr_std_ratio(x), 10) for x in sn_dbs],
        },
        "hilterman_weights": [
            {
                "angle_deg": a,
                "near": round(hilterman(a)[0], 10),
                "far": round(hilterman(a)[1], 10),
            }
            for a in [0, 10, 18, 22.5, 26, 45]
        ],
    }


def build_rpm():
    z = np.array([0.0, 500.0, 1000.0, 2000.0, 3000.0])
    example = {
        "z": z.tolist(),
        "shale_rho": (7.7e-12 * z**3 + -8.8e-08 * z**2 + 0.0004 * z + 1.957).tolist(),
        "shale_vp": (-0.00013 * z**2 + 1.13 * z + 1580).tolist(),
        "shale_vs": (-0.0001 * z**2 + 0.96 * z + 279).tolist(),
        "brine_sand_rho": (-7.8e-09 * z**2 + 0.00012 * z + 2.021).tolist(),
        "brine_sand_vp": (-1.34e-05 * z**2 + 0.49 * z + 2317).tolist(),
        "brine_sand_vs": (-1.0785e-05 * z**2 + 0.391 * z + 1007).tolist(),
        "oil_sand_rho": (-9.23e-09 * z**2 + 0.00014 * z + 1.916).tolist(),
        "oil_sand_vp": (-8.876e-06 * z**2 + 0.505 * z + 1998).tolist(),
        "oil_sand_vs": (-1.126e-05 * z**2 + 0.391 * z + 1036).tolist(),
        "gas_sand_rho": (-1.818e-08 * z**2 + 0.000247 * z + 1.612).tolist(),
        "gas_sand_vp": (-3.216e-06 * z**2 + 0.4796 * z + 1996).tolist(),
        "gas_sand_vs": (-1.0687e-05 * z**2 + 0.3662 * z + 1135).tolist(),
    }
    tag = {"z": z.tolist()}
    tag["shale_rho"] = np.polyval(
        [-3.27905787e-12, 1.86750139e-08, 9.64773845e-05, 2.09709627e00], z
    ).tolist()
    tag["shale_vp"] = np.polyval(
        [-2.86940692e-04, 2.02356702e00, 5.13645163e02], z
    ).tolist()
    tag["shale_vs"] = np.polyval(
        [-1.51184658e-04, 1.11423506e00, 2.17341849e02], z
    ).tolist()
    tag["brine_sand_rho"] = np.polyval(
        [1.62260122e-08, 4.19501863e-05, 2.10717208e00], z
    ).tolist()
    tag["brine_sand_vp"] = np.polyval(
        [-1.72905613e-04, 1.39902406e00, 1.15717554e03], z
    ).tolist()
    tag["brine_sand_vs"] = np.polyval(
        [-4.86767619e-05, 6.08845432e-01, 7.40710471e02], z
    ).tolist()
    a, b, c = 2.03767992e07, 3.90733465e-08, -2.03752253e07
    gas_vp = a * np.exp(b * z) + c
    tag["gas_sand_vp"] = gas_vp.tolist()
    tag["gas_sand_vs"] = (gas_vp / np.sqrt(2)).tolist()
    return round_nested(
        {
            "meta": {
                "ported_python": [
                    "rockphysics.rpm_example.RPMExample.calc_*",
                    "rockphysics.rpm_tagilsk_trends.RPMTagilsk.calc_* (subset)",
                ]
            },
            "example": example,
            "tagilsk": tag,
        }
    )


def main():
    dest_s = ROOT / "tests" / "fixtures" / "seismic_kernels.json"
    dest_r = ROOT / "tests" / "fixtures" / "rpm_trends.json"
    dest_s.write_text(json.dumps(build_seismic(), separators=(",", ":")) + "\n")
    dest_r.write_text(json.dumps(build_rpm(), separators=(",", ":")) + "\n")
    print(f"wrote {dest_s} ({dest_s.stat().st_size} bytes)")
    print(f"wrote {dest_r} ({dest_r.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
