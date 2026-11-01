//! Diagnostics for the fault-drag misses (T13, T4, T7). Ignored; prints only.
mod salt_order_metrics;
use salt_order_metrics as m;
use salt_order_metrics::Mode;
use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::{DragParams, DragSense};

fn with_drag(c: &E2eConfig, sense: Option<DragSense>) -> E2eConfig {
    let mut c = c.clone();
    c.faults.drag = sense.map(|sense| DragParams { sense, ..DragParams::default() });
    c.with_geometry_cache()
}

/// Spec zone quantities of fault `f` at output voxel (i, j, k), from the
/// ellipsoid geometry: (v, rho_m, d_loc).
fn zone(f: &synthoseis_geo::faults::ResolvedFault, bin: f64, dz: f64, i: f64, j: f64, k: usize) -> (f64, f64, f64) {
    let (x, y, z) = (i, j, k as f64);
    let v = f.geometry.value_at(x, y, z);
    let g = f.geometry.gradient_at(x, y, z);
    let gn = ((g[0] / bin).powi(2) + (g[1] / bin).powi(2) + (g[2] / dz).powi(2)).sqrt();
    let rho = if gn > 0.0 { (v - 1.0).abs() / gn } else { f64::INFINITY };
    let wt = match &f.tip { Some(t) => t.factor(x, y, z), None => f.lateral_at(x, y) };
    (v, rho, wt * f.profile[k])
}

#[test]
#[ignore = "diagnostic"]
fn t13_diag() {
    for seed in [13u64, 17] {
        let c = m::cfg(seed, [96, 96, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let cd = with_drag(&c, Some(DragSense::Auto));
        let on = m::run(&cd);
        let fm = synthoseis_core::fault_model(&cd).unwrap();
        let dm = fm.drag().unwrap();
        let [ni, nj, nk] = off.shape;
        let wmax = dm.faults.iter().map(|f| f.w_m).fold(0.0, f64::max);
        let planes: Vec<usize> = (0..ni * nj * nk).filter(|&v| off.fault_id[v] != 0).collect();
        let pos = |v: usize| (((v / nk) / nj) as f64 * dm.bin_m, ((v / nk) % nj) as f64 * dm.bin_m, (v % nk) as f64 * dm.dz_m);
        let labn: Vec<usize> = (0..fm.faults().len()).map(|n| (0..ni * nj * nk).filter(|&v| off.fault_id[v] == (n + 1) as u8).count()).collect();
        for (n, f) in fm.faults().iter().enumerate() {
            let dmax = f.profile.iter().cloned().fold(0.0, f64::max);
            eprintln!("T13D seed {seed} fault {n}: W {:.0} m, s {}, plane voxels {}, profile max {dmax:.2}, tip {}", dm.faults[n].w_m, dm.faults[n].s, labn[n], f.tip.is_some());
        }
        let (lk_off, lk_on) = (&off.src.as_ref().unwrap().1, &on.src.as_ref().unwrap().1);
        let (mut n, mut lkd, mut inz, mut inz_unlab, mut dl1) = (0, 0, 0, 0, 0);
        let mut by_fault = vec![0usize; fm.faults().len()];
        let mut qs = Vec::new();
        for v in 0..ni * nj * nk {
            if off.labels[v] == on.labels[v] { continue; }
            let p = pos(v);
            if !planes.iter().all(|&w| { let f = pos(w); (f.0 - p.0).powi(2) + (f.1 - p.1).powi(2) + (f.2 - p.2).powi(2) > (1.5 * wmax).powi(2) }) {
                continue;
            }
            n += 1;
            lkd += (lk_off[v] != lk_on[v]) as usize;
            let (i, j, k) = ((v / nk) / nj, (v / nk) % nj, v % nk);
            let mut best = (f64::INFINITY, 0usize, 0.0);
            for (fi, f) in fm.faults().iter().enumerate() {
                let (_, rho, d) = zone(f, dm.bin_m, dm.dz_m, i as f64, j as f64, k);
                let q = rho / dm.faults[fi].w_m;
                if d != 0.0 && q < best.0 { best = (q, fi, d); }
            }
            if best.0 < 1.0 {
                inz += 1;
                by_fault[best.1] += 1;
                inz_unlab += (labn[best.1] == 0) as usize;
                dl1 += (best.2.abs() <= 1.0) as usize;
                qs.push(best.0);
            }
        }
        qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = qs.get(qs.len() / 2).copied().unwrap_or(f64::NAN);
        eprintln!("T13D seed {seed}: differing far voxels {n}; lookup changed {lkd}; inside a spec zone (q<1, D_loc!=0) {inz} by fault {by_fault:?}; of those fault has no plane voxels {inz_unlab}, |D_loc|<=1 sample {dl1}; median q {med:.3}");
    }
}

fn in_zone(fm: &synthoseis_geo::faults::FaultModel, dm: &synthoseis_geo::faults::DragModel, i: usize, j: usize, k: usize) -> usize {
    fm.faults().iter().enumerate().filter(|(n, f)| {
        let (_, rho, d) = zone(f, dm.bin_m, dm.dz_m, i as f64, j as f64, k);
        d != 0.0 && rho < dm.faults[*n].w_m
    }).count()
}

#[test]
#[ignore = "diagnostic"]
fn t7_diag() {
    for seed in [2u64, 3, 4, 7, 10, 13, 17, 18] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let cd = with_drag(&c, Some(DragSense::Auto));
        let on = m::run(&cd);
        let fm = synthoseis_core::fault_model(&cd).unwrap();
        let dm = fm.drag().unwrap();
        let [ni, nj, nk] = off.shape;
        // [run][inside][kind]: kind 0 = flat or reversed span, 1 = span>0 ratio>2
        let mut cnt = [[[0usize; 2]; 2]; 2];
        let mut n = [[0usize; 2]; 2];
        let (mut out_diff, mut out_flag_change, mut top0) = (0, 0, [0usize; 2]);
        for (ri, r) in [&off, &on].iter().enumerate() {
            let lk = &r.src.as_ref().unwrap().1;
            for q in 0..ni * nj {
                for k in 0..nk {
                    let (lo, hi) = (k.saturating_sub(4), (k + 4).min(nk - 1));
                    if (lo..=hi).any(|x| r.fault_id[q * nk + x] != 0) { continue; }
                    let inside = (lo..=hi).any(|x| in_zone(&fm, dm, q / nj, q % nj, x) > 0) as usize;
                    n[ri][inside] += 1;
                    let span = (lk[q * nk + hi] - lk[q * nk + lo]) as f64;
                    if span <= 0.0 { cnt[ri][inside][0] += 1; if lk[q * nk + hi] <= 0.0 { top0[ri] += 1 } }
                    else if (hi - lo) as f64 / span > 2.0 { cnt[ri][inside][1] += 1; }
                }
            }
        }
        let (lo_, lo1) = (&off.src.as_ref().unwrap().1, &on.src.as_ref().unwrap().1);
        for v in 0..ni * nj * nk {
            let (q, k) = (v / nk, v % nk);
            if in_zone(&fm, dm, q / nj, q % nj, k) == 0 && lo_[v] != lo1[v] { out_diff += 1; }
            if in_zone(&fm, dm, q / nj, q % nj, k) == 0 && (off.fault_id[v] != 0) != (on.fault_id[v] != 0) { out_flag_change += 1; }
        }
        eprintln!("T7D seed {seed}: windows off in/out {:?} on in/out {:?}", n[0], n[1]);
        eprintln!("T7D seed {seed}: off  inside flat {} ratio>2 {} | outside flat {} ratio>2 {} | flat at lookup 0 (top clamp) {}", cnt[0][1][0], cnt[0][1][1], cnt[0][0][0], cnt[0][0][1], top0[0]);
        eprintln!("T7D seed {seed}: on   inside flat {} ratio>2 {} | outside flat {} ratio>2 {} | flat at lookup 0 (top clamp) {}", cnt[1][1][0], cnt[1][1][1], cnt[1][0][0], cnt[1][0][1], top0[1]);
        eprintln!("T7D seed {seed}: voxels outside every zone with a different lookup {out_diff}; with a different fault-plane flag {out_flag_change}");
    }
}

fn hard_inv(col: &[u8]) -> bool {
    let v: Vec<u8> = col.iter().copied().filter(|&x| x != 255).collect();
    let mut run = 1;
    for w in 1..v.len() {
        if v[w] == v[w - 1] { run += 1; continue; }
        if v[w] < v[w - 1] && run >= 2 { return true; }
        run = 1;
    }
    false
}

#[test]
#[ignore = "diagnostic"]
fn t4_diag() {
    for (seed, sense) in [(7u64, DragSense::Reverse), (14, DragSense::Reverse), (18, DragSense::Auto), (7, DragSense::Auto)] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let cd = with_drag(&c, Some(sense));
        let on = m::run(&cd);
        let fm = synthoseis_core::fault_model(&cd).unwrap();
        let dm = fm.drag().unwrap();
        let [ni, nj, nk] = off.shape;
        let (lo, l1) = (&off.src.as_ref().unwrap().1, &on.src.as_ref().unwrap().1);
        let (mut cols, mut grad_gt, mut fold_cols, mut fold_offplane) = (0, 0, 0, 0);
        let mut nzones = std::collections::BTreeMap::new();
        let mut gmax_all = 0f64;
        let mut thick = Vec::new();
        for q in 0..ni * nj {
            let co = &off.labels[q * nk..(q + 1) * nk];
            let cn = &on.labels[q * nk..(q + 1) * nk];
            if hard_inv(cn) == false || hard_inv(co) { continue; }
            cols += 1;
            let z = (0..nk).map(|k| in_zone(&fm, dm, q / nj, q % nj, k)).max().unwrap();
            *nzones.entry(z).or_insert(0) += 1;
            // combined drag shift s(k) = lookup_on - lookup_off; gradient -ds/dk
            let mut g = 0f64;
            let mut fold = false;
            let mut fold_np = false;
            for k in 0..nk - 1 {
                let v = q * nk + k;
                let plane = off.fault_id[v] != 0 || off.fault_id[v + 1] != 0 || on.fault_id[v] != 0 || on.fault_id[v + 1] != 0;
                if plane { continue; }
                let s0 = (l1[v] - lo[v]) as f64;
                let s1 = (l1[v + 1] - lo[v + 1]) as f64;
                g = g.max((s1 - s0).abs());
                if l1[v + 1] < l1[v] { fold = true; if lo[v + 1] >= lo[v] { fold_np = true } }
            }
            gmax_all = gmax_all.max(g);
            grad_gt += (g > 0.5) as usize;
            fold_cols += fold as usize;
            fold_offplane += fold_np as usize;
            // thinnest layer (run length) in the drag-off column
            let mut runs = vec![];
            let mut r = 1;
            for k in 1..nk { if co[k] == co[k - 1] { r += 1 } else { runs.push(r); r = 1 } }
            runs.push(r);
            thick.push(*runs.iter().min().unwrap());
        }
        thick.sort();
        // where is the new inversion: distance (samples) to a plane voxel, and
        // the lookup jump across it, drag off vs on
        let (mut at_plane, mut far_plane) = (0, 0);
        let mut jumps = Vec::new();
        for q in 0..ni * nj {
            let co = &off.labels[q * nk..(q + 1) * nk];
            let cn = &on.labels[q * nk..(q + 1) * nk];
            if !hard_inv(cn) || hard_inv(co) { continue; }
            let mut run = 1;
            for k in 1..nk {
                if cn[k] == cn[k - 1] { run += 1; continue; }
                if cn[k] < cn[k - 1] && run >= 2 && cn[k] != 255 && cn[k - 1] != 255 {
                    let near = (k.saturating_sub(3)..(k + 3).min(nk)).any(|x| on.fault_id[q * nk + x] != 0 || off.fault_id[q * nk + x] != 0);
                    if near { at_plane += 1 } else { far_plane += 1 }
                    let (a, b) = (k.saturating_sub(3), (k + 2).min(nk - 1));
                    jumps.push(((lo[q * nk + b] - lo[q * nk + a]) * 10.0).round() / 10.0);
                    jumps.push(((l1[q * nk + b] - l1[q * nk + a]) * 10.0).round() / 10.0);
                }
                run = 1;
            }
        }
        // was the column already inverted by a single cell (tolerated) with drag off,
        // and does the gather column (i, j) move across the step?
        let (mut was_soft, mut ij_moves, mut src_inv) = (0, 0, 0);
        let ijo = &on.src.as_ref().unwrap().0;
        let und = synthoseis_core::generate_labels(&{ let mut c0 = c.clone(); c0.faults = Default::default(); c0 }).0;
        for q in 0..ni * nj {
            let co = &off.labels[q * nk..(q + 1) * nk];
            let cn = &on.labels[q * nk..(q + 1) * nk];
            if !hard_inv(cn) || hard_inv(co) { continue; }
            let mut soft = false;
            for k in 1..nk { if co[k] < co[k - 1] && co[k] != 255 { soft = true } }
            was_soft += soft as usize;
            let mut run = 1;
            for k in 1..nk {
                if cn[k] == cn[k - 1] { run += 1; continue; }
                if cn[k] < cn[k - 1] && run >= 2 && cn[k] != 255 {
                    let (a, b) = (q * nk + k - 1, q * nk + k);
                    let mv = ijo[a].0.round() != ijo[b].0.round() || ijo[a].1.round() != ijo[b].1.round();
                    ij_moves += mv as usize;
                    // undragged labels at the two sources (nearest)
                    let at = |v: usize| { let (i, j) = (ijo[v].0.round() as usize, ijo[v].1.round() as usize); und[(i * nj + j) * nk + (l1[v].round() as usize).min(nk - 1)] };
                    src_inv += (at(b) < at(a)) as usize;
                }
                run = 1;
            }
        }
        eprintln!("T4D seed {seed} {sense:?}: new hard columns that had a 1-cell flip with drag off {was_soft}; inversion steps whose gather column (i,j) changes across the step {ij_moves}; steps where the unfaulted labels at the two sources are themselves inverted {src_inv}");
        let pmin = fm.faults().iter().map(|f| f.profile.iter().cloned().fold(f64::INFINITY, f64::min)).collect::<Vec<_>>();
        let th: Vec<_> = dm.faults.iter().map(|f| (f.s, (f.theta_deg * 10.0).round() / 10.0)).collect();
        eprintln!("T4D seed {seed} {sense:?}: new inversion steps within 3 samples of a fault plane {at_plane}, away {far_plane}; lookup span over the 5-sample step window (off, on pairs) {:?}; fault profile min {pmin:?} (>=0: normal throw); drag (s, theta at peak) {th:?}", &jumps[..jumps.len().min(24)]);
        eprintln!("T4D seed {seed} {sense:?}: new inverted cols {cols}; zones overlapping the column {nzones:?}; combined drag-shift |grad| max {gmax_all:.3}, cols >0.5: {grad_gt}; cols where the drag-on lookup decreases off the planes {fold_cols} (lookup monotone with drag off there: {fold_offplane}); thinnest drag-off layer per col (samples) {thick:?}; beta_h {} beta_f {} B {:.3}", dm.beta_h, dm.beta_f, dm.b);
    }
}

/// Per-stage trace of the drag gather (re-derived here from the public
/// pieces, checked against the pipeline lookup): per fault n,
/// (z_in, v, rho_m, d, w, a_out).
fn trace(fm: &synthoseis_geo::faults::FaultModel, dm: &synthoseis_geo::faults::DragModel, i: usize, j: usize, k: usize) -> (f64, Vec<(usize, f64, f64, f64, f64, f64, f64)>) {
    use synthoseis_geo::faults::{heave_vector, FaultKinematics, HeaveParams};
    let p = match fm.kinematics() { FaultKinematics::Heave(p) => p, _ => HeaveParams { heave_scale: 0.0, ..HeaveParams::default() } };
    let [ni, nj, nk] = fm.shape();
    let top = (nk - 1) as f64;
    let top_src = top + dm.pad as f64;
    let (mut x, mut y, mut z) = (i as f64, j as f64, k as f64);
    let mut out = vec![];
    let nf = fm.faults().len();
    for n in (0..nf).rev() {
        let f = &fm.faults()[n];
        let lat = if f.tip.is_some() { 0.0 } else { f.lateral_at(x, y) };
        let st = |kk: usize| {
            let zz = kk as f64;
            let v = f.geometry.value_at(x, y, zz);
            let wt = match &f.tip { Some(t) => t.factor(x, y, zz), None => lat };
            let dl = wt * f.profile[kk];
            let d = if v <= 1.0 { dl } else { 0.0 };
            if dl == 0.0 { return ((zz - d).clamp(0.0, top_src), [0.0, 0.0], v, d, 0.0); }
            let g = f.geometry.gradient_at(x, y, zz);
            let w = dm.displacement(n, v, g, dm.amplitude(n, f, x, y, zz, lat));
            let a = (zz - d - w).clamp(0.0, top_src);
            let h = if d == 0.0 { [0.0, 0.0] } else { heave_vector(f, &p, x, y, zz, d) };
            (a, h, v, d, w)
        };
        let zc = z.clamp(0.0, top);
        let excess = z - zc;
        let k0 = zc.floor() as usize;
        let fr = zc - k0 as f64;
        let (a0, h0, v0, d0, w0) = st(k0);
        let (mut a, h, d, w) = if fr == 0.0 { (a0, h0, d0, w0) } else {
            let (a1, h1, _, d1, w1) = st((k0 + 1).min(nk - 1));
            (a0 + fr * (a1 - a0), [h0[0] + fr * (h1[0] - h0[0]), h0[1] + fr * (h1[1] - h0[1])], d0 + fr * (d1 - d0), w0 + fr * (w1 - w0))
        };
        if excess > 0.0 { a = (a + excess).clamp(0.0, top_src) }
        let (_, rho, _) = zone(f, dm.bin_m, dm.dz_m, x, y, k0);
        out.push((n, z, v0, rho, d, w, a));
        z = a as f32 as f64;
        x = (x - h[0]).clamp(0.0, (ni - 1) as f64);
        y = (y - h[1]).clamp(0.0, (nj - 1) as f64);
    }
    (z, out)
}

#[test]
#[ignore = "diagnostic"]
fn t4_seed18_trace() {
    for (seed, sense) in [(18u64, DragSense::Auto), (7, DragSense::Reverse), (14, DragSense::Reverse)] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let on = m::run(&with_drag(&c, Some(sense)));
        let fm = synthoseis_core::fault_model(&with_drag(&c, Some(sense))).unwrap();
        let dm = fm.drag().unwrap();
        let [ni, nj, nk] = off.shape;
        let (lo, l1) = (&off.src.as_ref().unwrap().1, &on.src.as_ref().unwrap().1);
        let mut mism = 0;
        for q in 0..ni * nj { for k in 0..nk { if (trace(&fm, dm, q / nj, q % nj, k).0 as f32 - l1[q * nk + k]).abs() > 1e-3 { mism += 1 } } }
        eprintln!("TR seed {seed} {sense:?}: trace vs pipeline lookup mismatches {mism}; s {:?} W_m {:?}", dm.faults.iter().map(|f| f.s).collect::<Vec<_>>(), dm.faults.iter().map(|f| f.w_m.round()).collect::<Vec<_>>());
        let mut shown = 0;
        let mut cls = std::collections::BTreeMap::new();
        for q in 0..ni * nj {
            let co = &off.labels[q * nk..(q + 1) * nk];
            let cn = &on.labels[q * nk..(q + 1) * nk];
            if !hard_inv(cn) || hard_inv(co) { continue; }
            let mut run = 1;
            for k in 1..nk {
                if cn[k] == cn[k - 1] { run += 1; continue; }
                if cn[k] < cn[k - 1] && run >= 2 && cn[k] != 255 && cn[k - 1] != 255 {
                    let (a, b) = (k.saturating_sub(3), (k + 2).min(nk - 1));
                    let near = (k.saturating_sub(3)..(k + 3).min(nk)).any(|x| on.fault_id[q * nk + x] != 0 || off.fault_id[q * nk + x] != 0);
                    let son = l1[q * nk + b] - l1[q * nk + a];
                    // decreasing steps of the drag-on lookup in the window, and which fault's (d, w) changed most there
                    for kk in a..b {
                        let (z0, t0) = trace(&fm, dm, q / nj, q % nj, kk);
                        let (z1, t1) = trace(&fm, dm, q / nj, q % nj, kk + 1);
                        if z1 >= z0 { continue; }
                        // per fault: change in d and w, and which side (v<=1)
                        let mut key = String::new();
                        for (s0, s1) in t0.iter().zip(t1.iter()) {
                            if (s0.4 - s1.4).abs() > 0.5 || (s0.5 - s1.5).abs() > 0.5 || (s0.2 <= 1.0) != (s1.2 <= 1.0) {
                                key += &format!("f{}:{}{}{} ", s0.0, if s0.2 <= 1.0 { 'H' } else { 'F' }, if s1.2 <= 1.0 { 'H' } else { 'F' }, if (s0.4 - s1.4).abs() > 0.5 { "d" } else { "" });
                            }
                        }
                        *cls.entry(key.clone()).or_insert(0) += 1;
                        if shown < 6 {
                            shown += 1;
                            eprintln!("TR seed {seed}: col ({},{}) k {kk}->{} near {near} span off {:.1} on {son:.1}; lookup off {:.2}->{:.2} on {z0:.2}->{z1:.2}", q / nj, q % nj, kk + 1, lo[q * nk + b] - lo[q * nk + a], lo[q * nk + kk], lo[q * nk + kk + 1]);
                            for (s0, s1) in t0.iter().zip(t1.iter()) {
                                eprintln!("   fault {}: z_in {:.2}->{:.2} v {:.3}->{:.3} rho_m {:.1}->{:.1} d {:.2}->{:.2} w {:.2}->{:.2} a {:.2}->{:.2}", s0.0, s0.1, s1.1, s0.2, s1.2, s0.3, s1.3, s0.4, s1.4, s0.5, s1.5, s0.6, s1.6);
                            }
                        }
                    }
                }
                run = 1;
            }
        }
        eprintln!("TR seed {seed} {sense:?}: decreasing lookup steps by cause {cls:?}");
    }
}

#[test]
#[ignore = "diagnostic"]
fn t6_closure_diag() {
    use synthoseis_core::{closure_census, generate_labels};
    let (mut tk, mut tt) = ([0usize; 2], [0usize; 2]);
    for seed in 1u64..=20 {
        let mut row = vec![];
        for drag in [false, true] {
            let mut c = E2eConfig { seed, inline_count: 32, crossline_count: 32, samples: 128, faults: synthoseis_core::pipeline::FaultConfig::with_count(3), ..E2eConfig::default() };
            c.faults.legacy_profile = true;
            if drag { c.faults.drag = Some(DragParams::default()); }
            let (labels, sh) = generate_labels(&c);
            let k = closure_census(&c, &labels, sh).unwrap();
            let kept: usize = k.sizes.iter().filter(|&&v| v >= k.minimum).sum();
            let total: usize = k.sizes.iter().sum();
            let small = k.sizes.iter().filter(|&&v| v < k.minimum).count();
            let mut sz = k.sizes.clone(); sz.sort_unstable_by(|a, b| b.cmp(a));
            tk[drag as usize] += kept; tt[drag as usize] += total;
            row.push(format!("kept {kept} / total {total} ({:.2} %), closures {} (< min {}: {small}), largest {:?}", 100.0 * kept as f64 / total.max(1) as f64, k.sizes.len(), k.minimum, &sz[..sz.len().min(4)]));
        }
        eprintln!("T6D seed {seed}: off {} | on {}", row[0], row[1]);
    }
    eprintln!("T6D totals: off {} / {} on {} / {}", tk[0], tt[0], tk[1], tt[1]);
}

#[test]
#[ignore = "diagnostic"]
fn clamp_diag() {
    for seed in [3u64, 4, 7, 10, 14, 18] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let fm = synthoseis_core::fault_model(&with_drag(&c, Some(DragSense::Auto))).unwrap();
        let dm = fm.drag().unwrap();
        let f: Vec<_> = dm.faults.iter().map(|f| (f.t, f.w_m.round(), (f.theta_deg * 10.0).round() / 10.0, (f.b * 1000.0).round() / 1000.0)).collect();
        eprintln!("CL seed {seed}: (T, W, theta, B_f) {f:?}; clamps {}", dm.clamps.len());
    }
}

/// Stage positions (x, y, z_in) and drag w of every fault, youngest first
/// (same stages as `trace`).
fn stages(fm: &synthoseis_geo::faults::FaultModel, dm: &synthoseis_geo::faults::DragModel, i: usize, j: usize, k: usize) -> Vec<(usize, [f64; 3], f64)> {
    use synthoseis_geo::faults::{heave_vector, FaultKinematics, HeaveParams};
    let p = match fm.kinematics() { FaultKinematics::Heave(p) => p, _ => HeaveParams { heave_scale: 0.0, ..HeaveParams::default() } };
    let [ni, nj, nk] = fm.shape();
    let top = (nk - 1) as f64;
    let top_src = top + dm.pad as f64;
    let (mut x, mut y, mut z) = (i as f64, j as f64, k as f64);
    let mut out = vec![];
    for n in (0..fm.faults().len()).rev() {
        let f = &fm.faults()[n];
        let lat = if f.tip.is_some() { 0.0 } else { f.lateral_at(x, y) };
        let st = |kk: usize| {
            let zz = kk as f64;
            let v = f.geometry.value_at(x, y, zz);
            let wt = match &f.tip { Some(t) => t.factor(x, y, zz), None => lat };
            let dl = wt * f.profile[kk];
            let d = if v <= 1.0 { dl } else { 0.0 };
            if dl == 0.0 { return ((zz - d).clamp(0.0, top_src), [0.0, 0.0], 0.0); }
            let g = f.geometry.gradient_at(x, y, zz);
            let w = dm.displacement(n, v, g, dm.amplitude(n, f, x, y, zz, lat));
            let h = if d == 0.0 { [0.0, 0.0] } else { heave_vector(f, &p, x, y, zz, d) };
            ((zz - d - w).clamp(0.0, top_src), h, w)
        };
        let zc = z.clamp(0.0, top);
        let excess = z - zc;
        let k0 = zc.floor() as usize;
        let fr = zc - k0 as f64;
        let (a0, h0, w0) = st(k0);
        let (mut a, h, w) = if fr == 0.0 { (a0, h0, w0) } else {
            let (a1, h1, w1) = st((k0 + 1).min(nk - 1));
            (a0 + fr * (a1 - a0), [h0[0] + fr * (h1[0] - h0[0]), h0[1] + fr * (h1[1] - h0[1])], w0 + fr * (w1 - w0))
        };
        if excess > 0.0 { a = (a + excess).clamp(0.0, top_src) }
        out.push((n, [x, y, z], w));
        z = a as f32 as f64;
        x = (x - h[0]).clamp(0.0, (ni - 1) as f64);
        y = (y - h[1]).clamp(0.0, (nj - 1) as f64);
    }
    out
}

/// Inline taper/throw of fault n at a continuous position: (g(q), g'(q)/W·∂ρ/∂z, D, ∂D/∂z, side sβ).
fn parts(f: &synthoseis_geo::faults::ResolvedFault, df: &synthoseis_geo::faults::DragFault, bin: f64, dz: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64, f64, f64) {
    let nk = f.profile.len();
    let rho = |z: f64| {
        let v = f.geometry.value_at(x, y, z);
        let g = f.geometry.gradient_at(x, y, z);
        let gn = ((g[0] / bin).powi(2) + (g[1] / bin).powi(2) + (g[2] / dz).powi(2)).sqrt();
        ((v - 1.0).abs() / gn, v)
    };
    let tap = |q: f64| if q < 1.0 { (std::f64::consts::FRAC_PI_2 * q).cos().powi(2) } else { 0.0 };
    let dcont = |z: f64| {
        let zc = z.clamp(0.0, (nk - 1) as f64);
        let k0 = zc.floor() as usize;
        let k1 = (k0 + 1).min(nk - 1);
        let fr = zc - k0 as f64;
        let pr = f.profile[k0] + fr * (f.profile[k1] - f.profile[k0]);
        let wt = match &f.tip { Some(t) => t.factor(x, y, z), None => f.lateral_at(x, y) };
        wt * pr
    };
    let h = 0.25;
    let (r0, v) = rho(z);
    let q = r0 / df.w_m;
    let dg = (tap(rho(z + h).0 / df.w_m) - tap(rho(z - h).0 / df.w_m)) / (2.0 * h);
    let dd = (dcont(z + h) - dcont(z - h)) / (2.0 * h);
    let sb = if v <= 1.0 { df.s * df.beta_h } else { -df.s * df.beta_f };
    (tap(q), dg, dcont(z), dd, sb)
}

#[test]
#[ignore = "diagnostic"]
fn slope_breakdown() {
    for seed in [3u64, 7, 4, 10, 18] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let cd = with_drag(&c, Some(DragSense::Auto));
        let on = m::run(&cd);
        let fm = synthoseis_core::fault_model(&cd).unwrap();
        let dm = fm.drag().unwrap();
        let [ni, nj, nk] = off.shape;
        let (lo_, l1) = (&off.src.as_ref().unwrap().1, &on.src.as_ref().unwrap().1);
        // global per-voxel max |term1| per fault vs B_f
        let nf = fm.faults().len();
        let mut t1max = vec![0.0f64; nf];
        let mut t2max = vec![0.0f64; nf];
        let mut win = vec![];
        for q in 0..ni * nj {
            for k in 0..nk {
                let (i, j) = (q / nj, q % nj);
                for (n, p, _) in stages(&fm, dm, i, j, k) {
                    let (g, dg, d, dd, sb) = parts(&fm.faults()[n], &dm.faults[n], dm.bin_m, dm.dz_m, p[0], p[1], p[2]);
                    t1max[n] = t1max[n].max((sb * d * dg).abs());
                    t2max[n] = t2max[n].max((sb * g * dd).abs());
                }
                let (lo, hi) = (k.saturating_sub(4), (k + 4).min(nk - 1));
                if (lo..=hi).any(|x| on.fault_id[q * nk + x] != 0) { continue; }
                let len = (hi - lo) as f64;
                let son = (l1[q * nk + hi] - l1[q * nk + lo]) as f64 / len;
                let soff = (lo_[q * nk + hi] - lo_[q * nk + lo]) as f64 / len;
                if son < 0.5 && soff >= 0.5 { win.push((son, soff, q, lo, hi)); }
            }
        }
        eprintln!("SB seed {seed}: B_f {:?}", dm.faults.iter().map(|f| (f.b * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        eprintln!("SB seed {seed}: max per-voxel |term1| {:?}  |term2| {:?}", t1max.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(), t2max.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        win.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // aggregate over all new windows: window-mean of each term summed over faults
        let mut agg = [0.0f64; 4];
        for (wi, &(son, soff, q, lo, hi)) in win.iter().enumerate() {
            let (i, j) = (q / nj, q % nj);
            let len = (hi - lo) as f64;
            let mut per = vec![[0.0f64; 4]; nf]; // dw, t1, t2, t3
            for k in lo..hi {
                let s0 = stages(&fm, dm, i, j, k);
                let s1 = stages(&fm, dm, i, j, k + 1);
                for (a, b) in s0.iter().zip(&s1) {
                    let n = a.0;
                    let (g, dg, d, dd, sb) = parts(&fm.faults()[n], &dm.faults[n], dm.bin_m, dm.dz_m, a.1[0], a.1[1], a.1[2]);
                    let dw = b.2 - a.2;
                    let (t1, t2) = (sb * d * dg, sb * g * dd);
                    per[n][0] += dw / len;
                    per[n][1] += t1 / len;
                    per[n][2] += t2 / len;
                    per[n][3] += (dw - t1 - t2) / len;
                }
            }
            for p in &per { for t in 0..4 { agg[t] += p[t].abs() } }
            if wi < 4 {
                eprintln!("SB seed {seed}: window col ({i},{j}) k {lo}..{hi}: slope off {soff:.3} on {son:.3}");
                for (n, p) in per.iter().enumerate() {
                    eprintln!("   fault {n}: dw/dk {:+.3} = taper {:+.3} + dD/dz {:+.3} + chain {:+.3}", p[0], p[1], p[2], p[3]);
                }
            }
        }
        let nw = win.len().max(1) as f64;
        eprintln!("SB seed {seed}: {} new windows; mean |dw/dk| {:.3}, |taper| {:.3}, |dD/dz| {:.3}, |chain| {:.3}", win.len(), agg[0] / nw, agg[1] / nw, agg[2] / nw, agg[3] / nw);
    }
}

/// Per-stage chain (youngest first) with drag on (`dm = Some`) or off:
/// (n, input [x, y, z], d, w, heave [hx, hy]) plus the final (x, y, z).
/// Inline replica of the heave gather; the final z is checked against the
/// production lookup by the caller.
#[allow(clippy::type_complexity)]
fn chain(fm: &synthoseis_geo::faults::FaultModel, dm: Option<&synthoseis_geo::faults::DragModel>, i: usize, j: usize, k: usize) -> (Vec<(usize, [f64; 3], f64, f64, [f64; 2])>, [f64; 3]) {
    use synthoseis_geo::faults::{heave_vector, FaultKinematics, HeaveParams};
    let p = match fm.kinematics() { FaultKinematics::Heave(p) => p, _ => HeaveParams { heave_scale: 0.0, ..HeaveParams::default() } };
    let [ni, nj, nk] = fm.shape();
    let top = (nk - 1) as f64;
    let top_src = top + dm.map_or(0, |d| d.pad) as f64;
    let (mut x, mut y, mut z) = (i as f64, j as f64, k as f64);
    let mut out = vec![];
    for n in (0..fm.faults().len()).rev() {
        let f = &fm.faults()[n];
        let lat = if f.tip.is_some() { 0.0 } else { f.lateral_at(x, y) };
        let st = |kk: usize| {
            let zz = kk as f64;
            let v = f.geometry.value_at(x, y, zz);
            let wt = match &f.tip { Some(t) => t.factor(x, y, zz), None => lat };
            let dl = wt * f.profile[kk];
            let d = if v <= 1.0 { dl } else { 0.0 };
            if dl == 0.0 { return ((zz - d).clamp(0.0, top_src), [0.0, 0.0], 0.0, d); }
            let w = match dm { Some(dm) => dm.displacement(n, v, f.geometry.gradient_at(x, y, zz), dm.amplitude(n, f, x, y, zz, lat)), None => 0.0 };
            let h = if d == 0.0 { [0.0, 0.0] } else { heave_vector(f, &p, x, y, zz, d) };
            ((zz - d - w).clamp(0.0, top_src), h, w, d)
        };
        let zc = z.clamp(0.0, top);
        let excess = z - zc;
        let k0 = zc.floor() as usize;
        let fr = zc - k0 as f64;
        let (a0, h0, w0, d0) = st(k0);
        let (mut a, h, w, d) = if fr == 0.0 { (a0, h0, w0, d0) } else {
            let (a1, h1, w1, d1) = st((k0 + 1).min(nk - 1));
            (a0 + fr * (a1 - a0), [h0[0] + fr * (h1[0] - h0[0]), h0[1] + fr * (h1[1] - h0[1])], w0 + fr * (w1 - w0), d0 + fr * (d1 - d0))
        };
        if excess > 0.0 { a = (a + excess).clamp(0.0, top_src) }
        out.push((n, [x, y, z], d, w, h));
        z = a as f32 as f64;
        x = (x - h[0]).clamp(0.0, (ni - 1) as f64);
        y = (y - h[1]).clamp(0.0, (nj - 1) as f64);
    }
    (out, [x, y, z])
}

/// Independent throw D_n(x, y, z) of fault n at a continuous position
/// (zero on the footwall side), straight from the resolved geometry, profile
/// and tip; used for the analytic partials.
fn throw_at(f: &synthoseis_geo::faults::ResolvedFault, x: f64, y: f64, z: f64) -> f64 {
    let nk = f.profile.len();
    let zc = z.clamp(0.0, (nk - 1) as f64);
    let k0 = zc.floor() as usize;
    let k1 = (k0 + 1).min(nk - 1);
    let fr = zc - k0 as f64;
    let side = |kk: usize| if f.geometry.value_at(x, y, kk as f64) <= 1.0 { 1.0 } else { 0.0 };
    let dk = |kk: usize| {
        let wt = match &f.tip { Some(t) => t.factor(x, y, kk as f64), None => f.lateral_at(x, y) };
        side(kk) * wt * f.profile[kk]
    };
    dk(k0) + fr * (dk(k1) - dk(k0))
}

/// Gap term (Strata, 2:51 PM): the drag shift moves where the older fault's
/// throw is read sideways. Over the worst windows, per fault stage n:
///   Δslope = −Σ_n [ (ΔD_n^on − ΔD_n^off) + ΔW_n ] / len   (exact telescoping)
/// and ΔD_n^on − ΔD_n^off is split with independent central partials of D_n
/// into vertical (∂D/∂z·Δz_in), lateral (∂D/∂x·Δx_in + ∂D/∂y·Δy_in, the
/// sideways-read term) and a residual.
#[test]
#[ignore = "diagnostic"]
fn gap_term() {
    for no_gf in [true, false] {
    eprintln!("GAP ===== B {} =====", if no_gf { "step 1 (taper only, 8965fe51 clamp)" } else { "with G_f" });
    for (seed, ci, cj, lo, hi) in [(7u64, 12usize, 19usize, 44usize, 52usize), (3, 24, 15, 54, 62)] {
        let c = m::cfg(seed, [32, 32, 128], 4, Mode::PostFault);
        let off = m::run(&c);
        let mut cd = c.clone();
        cd.faults.drag = Some(DragParams { sense: DragSense::Auto, no_gf, ..DragParams::default() });
        let cd = cd.with_geometry_cache();
        let on = m::run(&cd);
        let fm = synthoseis_core::fault_model(&cd).unwrap();
        let fm0 = synthoseis_core::fault_model(&c).unwrap();
        let dm = fm.drag().unwrap();
        let [_, nj, nk] = off.shape;
        let q = ci * nj + cj;
        let len = (hi - lo) as f64;
        let (lz_on, lz_off) = (&on.src.as_ref().unwrap().1, &off.src.as_ref().unwrap().1);
        let s_on = (lz_on[q * nk + hi] - lz_on[q * nk + lo]) as f64 / len;
        let s_off = (lz_off[q * nk + hi] - lz_off[q * nk + lo]) as f64 / len;
        // replica check against production lookups
        let mut rep = 0.0f64;
        for k in lo..=hi {
            rep = rep.max((chain(&fm, Some(dm), ci, cj, k).1[2] - lz_on[q * nk + k] as f64).abs());
            rep = rep.max((chain(&fm0, None, ci, cj, k).1[2] - lz_off[q * nk + k] as f64).abs());
        }
        eprintln!("GAP seed {seed} col ({ci},{cj}) k {lo}..{hi}: slope off {s_off:.3} on {s_on:.3} drop {:.3}; replica max |dz| vs production {rep:.2e}", s_off - s_on);
        let nf = fm.faults().len();
        // per fault: [ΔW, ΔD_on−ΔD_off, vertical, lateral, residual] (slope units)
        let mut t = vec![[0.0f64; 5]; nf];
        let h = 0.05;
        for k in lo..hi {
            let (a_on, _) = chain(&fm, Some(dm), ci, cj, k);
            let (b_on, _) = chain(&fm, Some(dm), ci, cj, k + 1);
            let (a_off, _) = chain(&fm0, None, ci, cj, k);
            let (b_off, _) = chain(&fm0, None, ci, cj, k + 1);
            let mut lat_k = vec![];
            for s in 0..nf {
                let n = a_on[s].0;
                let f = &fm.faults()[n];
                let dw = b_on[s].3 - a_on[s].3;
                let dd = (b_on[s].2 - a_on[s].2) - (b_off[s].2 - a_off[s].2);
                let mid: [f64; 3] = std::array::from_fn(|c| 0.5 * (a_on[s].1[c] + b_on[s].1[c]));
                let pd = |c: usize| {
                    let mut p1 = mid; let mut p0 = mid; p1[c] += h; p0[c] -= h;
                    (throw_at(f, p1[0], p1[1], p1[2]) - throw_at(f, p0[0], p0[1], p0[2])) / (2.0 * h)
                };
                let din: [f64; 3] = std::array::from_fn(|c| (b_on[s].1[c] - a_on[s].1[c]) - (b_off[s].1[c] - a_off[s].1[c]));
                let vert = pd(2) * din[2];
                let lat = pd(0) * din[0] + pd(1) * din[1];
                lat_k.push((n, lat));
                for (o, v) in t[n].iter_mut().zip([dw, dd, vert, lat, dd - vert - lat]) { *o += v / len; }
            }
            eprintln!("   k {k}->{}: lateral-read term per fault {:?}", k + 1, lat_k.iter().map(|(n, v)| (*n, (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
            for s in 0..nf {
                let f = &fm.faults()[a_on[s].0];
                let side = |q: [f64; 3]| f.geometry.value_at(q[0], q[1], q[2]);
                eprintln!("      fault {}: z_in on {:.3} off {:.3} | v on {:.3} off {:.3} | D on {:.3} off {:.3} | w on {:.3}", a_on[s].0, a_on[s].1[2], a_off[s].1[2], side(a_on[s].1), side(a_off[s].1), a_on[s].2, a_off[s].2, a_on[s].3);
            }
        }
        let mut tot = 0.0;
        for (n, v) in t.iter().enumerate() {
            if v.iter().all(|x| *x == 0.0) { continue; }
            eprintln!("   fault {n}: dW {:+.3} | dD_on-dD_off {:+.3} = vertical {:+.3} + lateral(sideways read) {:+.3} + residual {:+.3}", v[0], v[1], v[2], v[3], v[4]);
            tot += v[0] + v[1];
        }
        eprintln!("GAP seed {seed}: predicted drop sum_n(dW + dD_on-dD_off) {tot:.3} vs observed {:.3}", s_off - s_on);
    }
}
}
