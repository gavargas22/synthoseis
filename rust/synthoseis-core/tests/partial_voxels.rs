//! Partial-voxel kernels (PR A; the pipeline wiring of PR B1 is tested in
//! `partial_voxels_pipeline.rs`):
//! validation of the fraction kernel, the Backus mixer and the sub-cell
//! splitter against the partial-voxels spec §5.1, §5.2, §5.3, §5.5 (centre
//! rule, label guard) §5.6 (chunk-edge invariance of the fractions) and
//! §5.7.
//!
//! Panels use the production end-members at 1 km (shale 2580/1139/2.277,
//! gas sand 2472.38/1490.51/1.841, brine sand from the production trend,
//! salt 4500/2250/2.17 and water 1500/1000/1.028 (the production constants
//! `synthoseis_rpm::{SALT, WATER}`); the seabed is also run over mudline
//! shale 1580/279/1.957), a 40 Hz Ricker, dt = 1 ms and dz = 4 m. Three models of the same geometry are compared with the
//! analytic convolutional trace:
//! * whole: today's whole voxels (centre rule);
//! * S: sub-cell interfaces at exact ray times (`subcell_reflectivity`);
//! * C: Backus voxels with the cell-to-cell reflectivity, T through the
//!   slowness sum (spec §1.4).
use synthoseis_core::partial_voxels::{
    cell_model, column_parts, column_parts_range, unit_thickness, CellModel, ColumnGeometry,
    ColumnParts, Part, PartKind, PartialVoxelStats, Unit, EPS_FRAC,
};
use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::rock_physics::label_intervals;
use synthoseis_core::salt::salt_body;
use synthoseis_core::ToyGeometry;
use synthoseis_rpm::{
    backus_mix, sand_f32, shale_f32, slowness_sum, Elastic32, Fluid, SALT, WATER,
};
use synthoseis_seismic::{
    reflectivity_time_column, reflectivity_time_column_with_twt, subcell_column,
    subcell_reflectivity, SubLayer, SubcellColumn, TwtKernel, TwtScratch, ZoeppritzForm,
};

const DZ: f64 = 4.0;
const DT: f64 = 1.0;
const F_PEAK: f64 = 40.0;
const NZ: usize = 80;
const NT: usize = 260;

fn shale() -> Elastic32 {
    shale_f32(1000.0, 1000.0, 1000.0)
}
fn brine() -> Elastic32 {
    sand_f32(Fluid::Brine, 1000.0, 1000.0, 1000.0)
}
const GAS: Elastic32 = Elastic32 {
    rho: 1.841,
    vp: 2472.38,
    vs: 1490.51,
};
/// Mudline shale (TVDML 0), the real contrast under the seabed.
fn mudline_shale() -> Elastic32 {
    shale_f32(0.0, 0.0, 0.0)
}

fn ricker(t_ms: f64) -> f64 {
    let a = (std::f64::consts::PI * F_PEAK * t_ms / 1000.0).powi(2);
    (1.0 - 2.0 * a) * (-a).exp()
}

/// Sampled reflectivity `x` convolved with the analytic Ricker.
fn convolve(x: &[f64]) -> Vec<f64> {
    let half = 60;
    (0..x.len())
        .map(|n| {
            let lo = n.saturating_sub(half);
            let hi = (n + half).min(x.len() - 1);
            (lo..=hi)
                .map(|m| x[m] * ricker((n as f64 - m as f64) * DT))
                .sum()
        })
        .collect()
}

/// Analytic convolutional trace of spikes `(t_ms, r)`.
fn analytic(spikes: &[(f64, f64)]) -> Vec<f64> {
    (0..NT)
        .map(|n| {
            spikes
                .iter()
                .map(|&(t, r)| r * ricker(n as f64 * DT - t))
                .sum()
        })
        .collect()
}

fn rpp(a: Elastic32, b: Elastic32) -> f64 {
    synthoseis_seismic::zoeppritz_pp_form(
        a.vp as f64,
        a.vs as f64,
        a.rho as f64,
        b.vp as f64,
        b.vs as f64,
        b.rho as f64,
        0.0,
        ZoeppritzForm::Exact,
    ) as f64
}

/// Band-limited (sinc) interpolant of `x` at fractional sample `s`.
fn interp(x: &[f64], s: f64) -> f64 {
    x.iter()
        .enumerate()
        .map(|(n, &v)| {
            let u = s - n as f64;
            if u == 0.0 {
                v
            } else {
                let p = std::f64::consts::PI * u;
                v * p.sin() / p
            }
        })
        .sum()
}

/// Peak pick (spec §5.1): the extremum of polarity `sign` on the 16×
/// upsampled trace, refined with a parabola. Returns `(t_ms, amplitude)`.
fn pick(x: &[f64], sign: f64) -> (f64, f64) {
    let n0 = (0..x.len())
        .max_by(|&a, &b| (sign * x[a]).total_cmp(&(sign * x[b])))
        .unwrap();
    let up: Vec<(f64, f64)> = (-32..=32)
        .map(|m| {
            let s = n0 as f64 + m as f64 / 16.0;
            (s, sign * interp(x, s))
        })
        .collect();
    let m = (1..up.len() - 1)
        .max_by(|&a, &b| up[a].1.total_cmp(&up[b].1))
        .unwrap();
    let (y0, y1, y2) = (up[m - 1].1, up[m].1, up[m + 1].1);
    let d = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2);
    let s = up[m].0 + d / 16.0;
    (s * DT, interp(x, s))
}

/// Per-cell sub-layers of one column's parts with end-members `props`.
fn sublayers(parts: &ColumnParts, props: &dyn Fn(PartKind) -> Elastic32) -> Vec<Vec<SubLayer>> {
    (0..parts.len())
        .map(|k| {
            parts
                .cell(k)
                .iter()
                .map(|p| {
                    let e = props(p.kind);
                    SubLayer {
                        frac: p.frac,
                        vp: e.vp,
                        vs: e.vs,
                        rho: e.rho,
                    }
                })
                .collect()
        })
        .collect()
}

/// Unit containing the centre `k + ½` (centre rule, ties to the upper part).
fn centre_kind(cell: &[Part]) -> PartKind {
    let mut acc = 0.0;
    for p in cell {
        acc += p.frac;
        if acc >= 0.5 {
            return p.kind;
        }
    }
    cell.last().unwrap().kind
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Whole,
    Sub,
    Cell,
    /// C with T from the Backus voxel Vp (reported only, see the test).
    CellBackusT,
}

/// Trace of one column in `mode`.
fn model_trace(parts: &ColumnParts, props: &dyn Fn(PartKind) -> Elastic32, mode: Mode) -> Vec<f64> {
    let mut x = vec![0.0; NT];
    match mode {
        Mode::Sub => {
            let layers = sublayers(parts, props);
            let mut col = SubcellColumn::default();
            subcell_column(layers.iter().map(|c| c.as_slice()), DZ, &mut col);
            subcell_reflectivity(
                &col,
                0.0,
                ZoeppritzForm::Exact,
                DT,
                TwtKernel::Sinc,
                &mut Vec::new(),
                &mut x,
            );
        }
        Mode::Whole | Mode::Cell | Mode::CellBackusT => {
            let e: Vec<Elastic32> = (0..parts.len())
                .map(|k| {
                    let c = parts.cell(k);
                    if mode == Mode::Whole {
                        props(centre_kind(c))
                    } else {
                        let mix: Vec<(f64, Elastic32)> =
                            c.iter().map(|p| (p.frac, props(p.kind))).collect();
                        backus_mix(&mix)
                    }
                })
                .collect();
            let vp: Vec<f32> = e.iter().map(|p| p.vp).collect();
            let vs: Vec<f32> = e.iter().map(|p| p.vs).collect();
            let rho: Vec<f32> = e.iter().map(|p| p.rho).collect();
            if mode != Mode::Cell {
                reflectivity_time_column(
                    &vp,
                    &vs,
                    &rho,
                    DZ,
                    0.0,
                    ZoeppritzForm::Exact,
                    DT,
                    TwtKernel::Sinc,
                    &mut TwtScratch::default(),
                    &mut x,
                );
            } else {
                // Time mode: T through the slowness sum (spec §1.4), with the
                // Backus voxels for the reflectivity.
                let mut t = vec![0.0f64];
                for k in 0..parts.len() {
                    let sl: Vec<(f64, f32)> = parts
                        .cell(k)
                        .iter()
                        .map(|p| (p.frac, props(p.kind).vp))
                        .collect();
                    t.push(t[k] + 2000.0 * DZ * slowness_sum(&sl));
                }
                reflectivity_time_column_with_twt(
                    &vp,
                    &vs,
                    &rho,
                    &t,
                    0.0,
                    ZoeppritzForm::Exact,
                    DT,
                    TwtKernel::Sinc,
                    &mut TwtScratch::default(),
                    &mut x,
                );
            }
        }
    }
    convolve(&x)
}

struct PanelStats {
    rms_ms: f64,
    max_ms: f64,
    max_amp_err: f64,
    max_resid: f64,
    energy: f64,
}

/// Dipping interface panel (spec §5.1): `z_b(i) = 30 + 0.137 i`, i = 0…63.
fn dipping_panel(pair: &str, mode: Mode) -> PanelStats {
    let (upper, lower) = match pair {
        "shale/gas" => (shale(), GAS),
        "shale/brine" => (shale(), brine()),
        "shale/salt" => (shale(), SALT),
        "water/shale" => (WATER, shale()),
        "water/mudline" => (WATER, mudline_shale()),
        _ => unreachable!(),
    };
    let r = rpp(upper, lower);
    let (mut se, mut max_ms, mut max_amp, mut max_res, mut e_num, mut e_den) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0, 0.0);
    for i in 0..64 {
        let zb = 30.0 + 0.137 * i as f64;
        let mut parts = ColumnParts::default();
        // The boundary goes through the real kernel as the boundary kind of
        // the pair: horizon (sediments), seabed (water) or salt top.
        let (z, salt): (Vec<f64>, Option<(f64, f64)>) = match pair {
            "water/shale" | "water/mudline" => (vec![zb, 1e9], None),
            "shale/salt" => (vec![0.0, 1e9], Some((zb - 0.5, 1e6))),
            _ => (vec![0.0, zb, 1e9], None),
        };
        column_parts(
            &ColumnGeometry {
                horizons: &z,
                salt,
                contacts: &[],
            },
            NZ,
            &mut parts,
        );
        let props = |k: PartKind| match k {
            PartKind::Water => WATER,
            PartKind::Salt => SALT,
            PartKind::Interval { h: 0, .. } if !pair.starts_with("water/") => upper,
            _ => lower,
        };
        let trace = model_trace(&parts, &props, mode);
        let t_true = 2000.0 * zb * DZ / upper.vp as f64;
        let truth = analytic(&[(t_true, r)]);
        let (tp, amp) = pick(&trace, r.signum());
        let dt = tp - t_true;
        se += dt * dt;
        max_ms = max_ms.max(dt.abs());
        max_amp = max_amp.max((amp - r).abs() / r.abs());
        let res = trace
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        max_res = max_res.max(res / r.abs());
        e_num += trace
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        e_den += truth.iter().map(|b| b * b).sum::<f64>();
    }
    PanelStats {
        rms_ms: (se / 64.0).sqrt(),
        max_ms,
        max_amp_err: max_amp,
        max_resid: max_res,
        energy: e_num / e_den,
    }
}

/// Spec §5.1 / §5.2: the dipping-interface timing panel, whole vs S vs C.
///
/// C is gated with the time-mode T of spec §1.4 (slowness sum through the
/// mixed cell). The variant with T from the Backus voxel Vp (`2dz/Vp̄`,
/// what the spec's probe column appears to use: it matches the probe in
/// sediments) is printed for the record; at the seabed it exceeds the
/// 0.45 ms gate, which is why the gate applies to the slowness T.
#[test]
fn dipping_interface_staircase() {
    for pair in ["shale/gas", "shale/brine", "shale/salt", "water/shale", "water/mudline"] {
        let w = dipping_panel(pair, Mode::Whole);
        let s = dipping_panel(pair, Mode::Sub);
        let c = dipping_panel(pair, Mode::Cell);
        let cb = dipping_panel(pair, Mode::CellBackusT);
        println!(
            "PV dipping {pair}: C with T from Backus Vp (reported): rms {:.3} max {:.3} resid {:.1}%",
            cb.rms_ms, cb.max_ms, 100.0 * cb.max_resid
        );
        println!(
            "PV dipping {pair}: whole rms {:.3} max {:.3} resid {:.1}% | S rms {:.4} max {:.4} amp {:.3}% resid {:.3}% E {:.2e} ratio {:.0}x | C rms {:.3} max {:.3} resid {:.1}% ratio {:.1}x",
            w.rms_ms, w.max_ms, 100.0 * w.max_resid,
            s.rms_ms, s.max_ms, 100.0 * s.max_amp_err, 100.0 * s.max_resid, s.energy, w.rms_ms / s.rms_ms,
            c.rms_ms, c.max_ms, 100.0 * c.max_resid, w.rms_ms / c.rms_ms,
        );
        // Whole voxels: the error is uniform over ±½ cell.
        let half = 0.5 * 2000.0 * DZ
            / if pair.starts_with("water/") {
                1500.0
            } else {
                2580.0
            };
        assert!(
            (w.rms_ms - half / 3f64.sqrt()).abs() < 0.05 * half,
            "{pair} whole rms {}",
            w.rms_ms
        );
        // S (spec §5.1, §5.2).
        assert!(s.max_ms <= 0.02, "{pair} S max {}", s.max_ms);
        assert!(s.max_amp_err <= 0.005, "{pair} S amp {}", s.max_amp_err);
        assert!(s.max_resid <= 0.005, "{pair} S resid {}", s.max_resid);
        assert!(s.energy <= 1e-4, "{pair} S E {}", s.energy);
        assert!(w.rms_ms / s.rms_ms >= 10.0, "{pair} S ratio");
        // C (Backus).
        if pair.starts_with("water/") {
            // Seabed gate (spec §5.1, slowness timing). Strata measured
            // 0.260/0.381 ms over mudline shale, 0.060/0.114 over 1 km shale.
            assert!(
                c.rms_ms <= 0.45 && c.max_resid <= 0.35,
                "{pair} C rms {} resid {}",
                c.rms_ms,
                c.max_resid
            );
        } else {
            assert!(
                c.rms_ms <= 0.10 && c.max_resid <= 0.15,
                "{pair} C rms {} resid {}",
                c.rms_ms,
                c.max_resid
            );
            assert!(w.rms_ms / c.rms_ms >= 8.0, "{pair} C ratio");
        }
    }
}

/// Spec §5.3: thin bed (wedge), h = 0.08 … 12 m, swept over several
/// fractional top positions (Strata's 6 random tops: gas C ≤ 17.6 %, salt
/// C ≤ 8.0 %; gate 20 %).
#[test]
fn thin_bed_wedge() {
    let sh = shale();
    for (name, thin) in [("gas", GAS), ("salt", SALT)] {
        let r_top = rpp(sh, thin);
        let r_bot = rpp(thin, sh);
        let hs: Vec<f64> = (0..60)
            .map(|n| 0.08 * (12.0f64 / 0.08).powf(n as f64 / 59.0))
            .collect();
        const TOPS: [f64; 7] = [30.0, 30.13, 30.37, 30.5, 30.62, 30.81, 30.97];
        let mut stats = [(0.0f64, 0.0f64); 3];
        for (&h, zt) in hs.iter().flat_map(|h| TOPS.iter().map(move |&t| (h, t))) {
            let zb = zt + h / DZ;
            let t1 = 2000.0 * zt * DZ / sh.vp as f64;
            let t2 = t1 + 2000.0 * h / thin.vp as f64;
            let exact = analytic(&[(t1, r_top), (t2, r_bot)]);
            let mut parts = ColumnParts::default();
            column_parts(
                &ColumnGeometry {
                    horizons: &[0.0, zt, zb, 1e9],
                    salt: None,
                    contacts: &[],
                },
                NZ,
                &mut parts,
            );
            let props = |k: PartKind| {
                if k == (PartKind::Interval { h: 1, hc: false }) {
                    thin
                } else {
                    sh
                }
            };
            for (m, mode) in [Mode::Whole, Mode::Sub, Mode::Cell].into_iter().enumerate() {
                let tr = model_trace(&parts, &props, mode);
                let e = tr
                    .iter()
                    .zip(&exact)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
                    / r_top.abs();
                stats[m].0 = stats[m].0.max(e);
                stats[m].1 += e / (hs.len() * TOPS.len()) as f64;
            }
        }
        println!(
            "PV wedge {name}: whole max {:.1}% mean {:.1}% | S max {:.3}% | C max {:.1}% mean {:.1}%",
            100.0 * stats[0].0, 100.0 * stats[0].1, 100.0 * stats[1].0, 100.0 * stats[2].0, 100.0 * stats[2].1
        );
        assert!(stats[1].0 <= 0.005, "{name} S max {}", stats[1].0);
        assert!(
            stats[2].0 <= 0.20 && stats[2].1 <= 0.10,
            "{name} C {:?}",
            stats[2]
        );
    }
}

fn layered_cfg(seed: u64, shape: [usize; 3], salt: bool) -> E2eConfig {
    let mut cfg = E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        geometry: ToyGeometry::Layered,
        ..E2eConfig::default()
    };
    cfg.rock_physics.salt = salt;
    cfg
}

fn snap(x: f64) -> f64 {
    if (x - x.round()).abs() < EPS_FRAC {
        x.round()
    } else {
        x
    }
}

/// `|[a, b) ∩ [lo, hi)|`.
fn overlap(a: f64, b: f64, lo: f64, hi: f64) -> f64 {
    (b.min(hi) - a.max(lo)).max(0.0)
}

/// Spec §5.7: Σ_u f_u = 1 per cell (1e-12), Σ_k f_u = thickness (1e-9),
/// salt mass vs salt label count < 1 per column, no sliver parts; with a
/// fluid contact in every interval so HC/brine splits are exercised.
#[test]
fn fractions_conserve_mass_on_real_geometry() {
    let shape = [32usize, 28, 128];
    let nzf = shape[2] as f64;
    let mut multi = 0;
    for seed in [7u64, 1, 2, 3, 30] {
        for salt in [false, true] {
            let cfg = layered_cfg(seed, shape, salt);
            let (maps, nh) =
                synthoseis_core::pipeline_stream::toy_horizon_maps_continuous(&cfg).unwrap();
            let body = salt_body(&cfg);
            let bounds = body.as_ref().map(|b| b.hull_bounds());
            let mut parts = ColumnParts::default();
            let mut salt_seen = false;
            for c in 0..shape[0] * shape[1] {
                let z = &maps[c * nh..(c + 1) * nh];
                let contacts: Vec<f32> = (0..nh - 1)
                    .map(|h| (0.3 * z[h] + 0.7 * z[h + 1]) as f32)
                    .collect();
                let sb = bounds.as_ref().and_then(|b| b[c]);
                column_parts(
                    &ColumnGeometry {
                        horizons: z,
                        salt: sb,
                        contacts: &contacts,
                    },
                    shape[2],
                    &mut parts,
                );
                for k in 0..shape[2] {
                    let cell = parts.cell(k);
                    let s: f64 = cell.iter().map(|p| p.frac).sum();
                    assert!((s - 1.0).abs() <= 1e-12, "seed {seed} c {c} k {k} sum {s}");
                    assert!(cell.iter().all(|p| p.frac >= EPS_FRAC));
                    multi += (cell.len() >= 3) as usize;
                }
                // Reference thicknesses from the (snapped) boundaries.
                let (slo, shi) = sb.map_or((0.0, 0.0), |(lo, hi)| (snap(lo + 0.5), snap(hi + 0.5)));
                let sed = |a: f64, b: f64| {
                    overlap(a, b, 0.0, nzf) - overlap(a, b, slo.max(0.0), shi.min(nzf))
                };
                let zs: Vec<f64> = z.iter().map(|&v| snap(v)).collect();
                let water = sed(f64::NEG_INFINITY, zs[0]);
                assert!((unit_thickness(&parts, Unit::Water) - water).abs() <= 1e-9);
                for h in 0..nh - 1 {
                    let a = zs[h].max(zs[0]);
                    let b = zs[h + 1].max(a);
                    let t = sed(a, b);
                    let got = unit_thickness(&parts, Unit::Interval(h));
                    assert!(
                        (got - t).abs() <= 1e-9,
                        "seed {seed} c {c} h {h}: {got} vs {t}"
                    );
                }
                if let Some(b) = &body {
                    let mass = unit_thickness(&parts, Unit::Salt);
                    let (k0, k1) = b.runs[c];
                    let count = (k1 as usize).min(shape[2]).saturating_sub(k0 as usize) as f64;
                    assert!(
                        (mass - count).abs() < 1.0,
                        "seed {seed} c {c}: salt {mass} vs {count}"
                    );
                    salt_seen |= count > 0.0;
                }
            }
            assert_eq!(salt_seen, salt && body.is_some());
        }
    }
    assert!(multi > 0, "no cell with three or more parts exercised");
}

/// Spec §1.6 / §5.5: the centre rule on the continuous geometry reproduces
/// today's labels (unfaulted) and salt labels, and the label-support guard
/// never fires unfaulted.
#[test]
fn centre_rule_matches_labels_unfaulted() {
    let shape = [32usize, 28, 128];
    for seed in [7u64, 1, 2, 3, 30] {
        for salt in [false, true] {
            let cfg = layered_cfg(seed, shape, salt);
            let (labels, _) = synthoseis_core::generate_labels(&cfg);
            let (cont, nh) =
                synthoseis_core::pipeline_stream::toy_horizon_maps_continuous(&cfg).unwrap();
            let rounded: Vec<f64> = cont.iter().map(|v| v.round()).collect();
            let intervals = label_intervals(&rounded, shape[0], shape[1], nh, shape[2]);
            let body = salt_body(&cfg);
            let bounds = body.as_ref().map(|b| b.hull_bounds());
            let (mut sed_parts, mut all_parts) = (ColumnParts::default(), ColumnParts::default());
            let mut stats = PartialVoxelStats::default();
            let mut mismatches = 0;
            for c in 0..shape[0] * shape[1] {
                let z = &cont[c * nh..(c + 1) * nh];
                column_parts(
                    &ColumnGeometry {
                        horizons: z,
                        salt: None,
                        contacts: &[],
                    },
                    shape[2],
                    &mut sed_parts,
                );
                let sb = bounds.as_ref().and_then(|b| b[c]);
                column_parts(
                    &ColumnGeometry {
                        horizons: z,
                        salt: sb,
                        contacts: &[],
                    },
                    shape[2],
                    &mut all_parts,
                );
                for k in 0..shape[2] {
                    let lab = labels[c * shape[2] + k];
                    let want = if lab == 255 {
                        None
                    } else {
                        Some(intervals[lab as usize])
                    };
                    let got = match centre_kind(sed_parts.cell(k)) {
                        PartKind::Interval { h, .. } => Some(h),
                        _ => None,
                    };
                    mismatches += (want != got) as usize;
                    let in_salt = body.as_ref().is_some_and(|b| b.contains(c, k));
                    mismatches +=
                        (in_salt != (centre_kind(all_parts.cell(k)) == PartKind::Salt)) as usize;
                    let unit = if in_salt {
                        Unit::Salt
                    } else {
                        match want {
                            Some(h) => Unit::Interval(h),
                            None if (k as f64) < z[0] => Unit::Water,
                            None => Unit::Below,
                        }
                    };
                    let _ = cell_model(all_parts.cell(k), unit, &mut stats);
                }
            }
            assert_eq!(mismatches, 0, "seed {seed} salt {salt}");
            assert_eq!(stats.label_guard, 0, "seed {seed} salt {salt}: {stats:?}");
            assert!(stats.mixed > 0);
            println!("PV centre rule seed {seed} salt {salt}: 0 mismatches, {stats:?}");
        }
    }
}

/// Spec §5.6: fractions are per column (class A, no halo): any column order
/// or tiling, and any k-split, gives the same bits.
#[test]
fn fractions_are_chunk_edge_invariant() {
    let shape = [20usize, 24, 96];
    let cfg = layered_cfg(30, shape, true);
    let (maps, nh) = synthoseis_core::pipeline_stream::toy_horizon_maps_continuous(&cfg).unwrap();
    let bounds = salt_body(&cfg).map(|b| b.hull_bounds());
    let geom = |c: usize| ColumnGeometry {
        horizons: &maps[c * nh..(c + 1) * nh],
        salt: bounds.as_ref().and_then(|b| b[c]),
        contacts: &[],
    };
    let bits = |p: &ColumnParts, k: usize| -> Vec<(PartKind, u64)> {
        p.cell(k)
            .iter()
            .map(|q| (q.kind, q.frac.to_bits()))
            .collect()
    };
    // Reference: whole columns in natural order.
    let n = shape[0] * shape[1];
    let mut reference = Vec::with_capacity(n);
    let mut p = ColumnParts::default();
    for c in 0..n {
        column_parts(&geom(c), shape[2], &mut p);
        reference.push((0..shape[2]).map(|k| bits(&p, k)).collect::<Vec<_>>());
    }
    for chunk in [[1usize, 1, 96], [5, 7, 96], [3, 20, 16], [8, 5, 16]] {
        for i0 in (0..shape[0]).step_by(chunk[0]).rev() {
            for j0 in (0..shape[1]).step_by(chunk[1]) {
                for k0 in (0..shape[2]).step_by(chunk[2]) {
                    let k1 = (k0 + chunk[2]).min(shape[2]);
                    for i in i0..(i0 + chunk[0]).min(shape[0]) {
                        for j in j0..(j0 + chunk[1]).min(shape[1]) {
                            let c = i * shape[1] + j;
                            column_parts_range(&geom(c), k0, k1, &mut p);
                            for (k, want) in reference[c].iter().enumerate().take(k1).skip(k0) {
                                assert_eq!(bits(&p, k - k0), *want, "chunk {chunk:?} c {c} k {k}");
                            }
                        }
                    }
                }
            }
        }
    }
    // A deeper cube does not change the cells above.
    for c in (0..n).step_by(37) {
        column_parts(&geom(c), shape[2] + 40, &mut p);
        for (k, want) in reference[c].iter().enumerate() {
            assert_eq!(bits(&p, k), *want);
        }
    }
}

/// Mixed cells use only the parts; pure cells short-circuit (whole path).
#[test]
fn cell_model_short_circuits_pure_cells() {
    let mut p = ColumnParts::default();
    column_parts(
        &ColumnGeometry {
            horizons: &[2.0, 5.5, 50.0],
            salt: None,
            contacts: &[],
        },
        10,
        &mut p,
    );
    let mut s = PartialVoxelStats::default();
    assert_eq!(
        cell_model(p.cell(2), Unit::Interval(0), &mut s),
        CellModel::Whole
    );
    assert!(matches!(
        cell_model(p.cell(5), Unit::Interval(1), &mut s),
        CellModel::Mixed(_)
    ));
    assert_eq!((s.cells, s.mixed, s.label_guard), (2, 1, 0));
}
