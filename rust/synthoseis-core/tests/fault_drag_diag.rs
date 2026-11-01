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
    for seed in [2u64, 3, 7] {
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
        eprintln!("T4D seed {seed} {sense:?}: new inverted cols {cols}; zones overlapping the column {nzones:?}; combined drag-shift |grad| max {gmax_all:.3}, cols >0.5: {grad_gt}; cols where the drag-on lookup decreases off the planes {fold_cols} (lookup monotone with drag off there: {fold_offplane}); thinnest drag-off layer per col (samples) {thick:?}; beta_h {} beta_f {} B {:.3}", dm.beta_h, dm.beta_f, dm.b);
    }
}
