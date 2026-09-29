"""Independent references for the Zoeppritz PP fix (`det` -> `d`).

Writes `zoeppritz_reference.json`: for a grid of interfaces and incidence
angles, the PP coefficient from

* ``matrix``: the full 4x4 Zoeppritz linear system (Aki & Richards 1980,
  eq. 5.39 in matrix form) solved with numpy, complex; no explicit formula;
* ``bruges``: ``bruges.reflection.zoeppritz_rpp`` (textbook explicit form);
* ``aki_richards``: ``bruges.reflection.akirichards`` (linearised);
* ``legacy``: the legacy ``zoeppritz_kernel.py`` expression with ``det``.

Run: ``python tests/fixtures/generate_zoeppritz_reference.py`` (needs numpy
and bruges 0.5).
"""

import json
from pathlib import Path

import bruges
import numpy as np


def matrix_rpp(vp1, vs1, rho1, vp2, vs2, rho2, ang):
    t1 = np.radians(ang) + 0j
    p = np.sin(t1) / vp1
    t2, f1, f2 = np.arcsin(p * vp2), np.arcsin(p * vs1), np.arcsin(p * vs2)
    s, c = np.sin, np.cos
    m = np.array(
        [
            [-s(t1), -c(f1), s(t2), c(f2)],
            [c(t1), -s(f1), c(t2), -s(f2)],
            [
                s(2 * t1),
                vp1 / vs1 * c(2 * f1),
                rho2 * vs2**2 * vp1 / (rho1 * vs1**2 * vp2) * s(2 * t2),
                rho2 * vs2 * vp1 / (rho1 * vs1**2) * c(2 * f2),
            ],
            [
                -c(2 * f1),
                vs1 / vp1 * s(2 * f1),
                rho2 * vp2 / (rho1 * vp1) * c(2 * f2),
                -rho2 * vs2 / (rho1 * vp1) * s(2 * f2),
            ],
        ]
    )
    rhs = np.array([s(t1), c(t1), s(2 * t1), c(2 * f1)])
    return np.linalg.solve(m, rhs)[0]


def legacy_rpp(vp1, vs1, rho1, vp2, vs2, rho2, ang):
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
    return (f * (b * np.cos(t) / vp1 - c * np.cos(t2) / vp2) - h * p * p * (a + det * np.cos(t) / vp1 * np.cos(f2) / vs2)) / det


def main():
    rng = np.random.default_rng(20260928)
    interfaces = [
        (3000.0, 1500.0, 2.3, 3200.0, 1600.0, 2.2),
        (1500.0, 1000.0, 1.028, 2400.0, 1000.0, 2.2),  # water (legacy Vs 1000) / sediment
        (2500.0, 1200.0, 2.2, 3500.0, 2000.0, 2.4),
        (3500.0, 2000.0, 2.4, 2500.0, 1200.0, 2.2),
        (2800.0, 1400.0, 2.35, 2600.0, 1500.0, 2.1),  # shale over gas sand (class III-like)
    ]
    for _ in range(40):
        vp1 = rng.uniform(1800, 4500)
        vs1 = vp1 / rng.uniform(1.6, 2.6)
        rho1 = rng.uniform(1.9, 2.7)
        vp2 = vp1 * rng.uniform(0.7, 1.4)
        vs2 = vp2 / rng.uniform(1.5, 2.6)
        rho2 = rho1 * rng.uniform(0.85, 1.15)
        interfaces.append((vp1, vs1, rho1, vp2, vs2, rho2))
    angles = [float(a) for a in range(0, 61, 5)]
    cases = []
    for itf in interfaces:
        for ang in angles:
            m = matrix_rpp(*itf, ang)
            br = complex(bruges.reflection.zoeppritz_rpp(*itf, ang))
            ar = complex(bruges.reflection.akirichards(*itf, ang))
            lg = legacy_rpp(*itf, ang)
            cases.append(
                {
                    "props": list(itf),
                    "angle_deg": ang,
                    "matrix_re": m.real,
                    "matrix_im": m.imag,
                    "bruges": br.real,
                    "aki_richards": ar.real,
                    "legacy": lg.real,
                }
            )
    out = Path(__file__).with_name("zoeppritz_reference.json")
    out.write_text(json.dumps({"bruges_version": bruges.__version__, "cases": cases}, indent=0))
    mm = max(abs(c["matrix_re"] - c["bruges"]) for c in cases)
    ml = max(abs(c["matrix_re"] - c["legacy"]) for c in cases)
    print(f"{len(cases)} cases; max |matrix - bruges| = {mm:.3e}; max |matrix - legacy| = {ml:.3e}")


if __name__ == "__main__":
    main()
