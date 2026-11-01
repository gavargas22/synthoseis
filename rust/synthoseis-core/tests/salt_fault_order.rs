//! Post-fault salt drag (spec-salt-fault-order, option B): the bars of
//! spec §5 on 128×128×256 cubes with 4 faults. Measured values for seeds
//! 1–20 are in `docs/salt-bodies.md`. The 20-seed bars are `#[ignore]`d
//! for run time only; the old-reference numbers (stair vs faults-off and
//! master, Jaccard vs master) are printed, not asserted.
mod salt_order_metrics;
use m::Mode;
use salt_order_metrics as m;
use synthoseis_core::time_mode::{resample_columns, tile_twt};

const SHAPE: [usize; 3] = [128, 128, 256];
const FAR_SHAPE: [usize; 3] = [256, 256, 128];

fn body(seed: u64) -> Option<synthoseis_core::salt::SaltBody> {
    m::salt(&m::cfg(seed, SHAPE, 4, Mode::PostFault))
}

/// Spec test 1 (replaces `salt_flank_fragments_stay_bounded`): no non-salt
/// voxel samples a source inside the salt body with a label that differs
/// from the undragged one. Master (drag before faulting) is the legacy
/// reference: 52 236 voxels (1.30 % of non-salt) on seed 1, pinned ≥ 0.5 %.
#[test]
fn hidden_drag_exposure_is_zero() {
    let mut tested = 0;
    for seed in [1u64, 7, 13] {
        let Some(s) = body(seed) else { continue };
        let und = m::undragged(seed, SHAPE);
        let post = m::exposed(&m::run(&m::cfg(seed, SHAPE, 4, Mode::PostFault)), &s, &und);
        assert_eq!(post, 0, "seed {seed}: {post} exposed voxels");
        if seed == 1 {
            let master = m::exposed(&m::run(&m::cfg(seed, SHAPE, 4, Mode::Master)), &s, &und);
            let nonsalt = SHAPE.iter().product::<usize>() - s.voxels(SHAPE[2]);
            assert!(
                master as f64 >= 0.005 * nonsalt as f64,
                "master reference {master} voxels < 0.5 %"
            );
        }
        tested += 1;
    }
    assert!(tested >= 2, "{tested} seeds with salt");
}

/// Spec test 1 on seeds 1–20 (measured: 0 on every seed; master sum
/// 414 581 voxels, mean 0.52 %). Ignored for run time only.
#[test]
#[ignore = "slow: 60 cube runs; passes (0 exposed on seeds 1-20)"]
fn hidden_drag_exposure_is_zero_seeds_1_20() {
    let mut frac = 0.0;
    let mut n = 0;
    for seed in 1u64..=20 {
        let Some(s) = body(seed) else { continue };
        let und = m::undragged(seed, SHAPE);
        let post = m::exposed(&m::run(&m::cfg(seed, SHAPE, 4, Mode::PostFault)), &s, &und);
        assert_eq!(post, 0, "seed {seed}: {post} exposed voxels");
        let master = m::exposed(&m::run(&m::cfg(seed, SHAPE, 4, Mode::Master)), &s, &und);
        frac += master as f64 / (SHAPE.iter().product::<usize>() - s.voxels(SHAPE[2])) as f64;
        n += 1;
    }
    assert!(
        frac / n as f64 >= 0.005,
        "master mean {:.4} %",
        100.0 * frac / n as f64
    );
}

/// Spec test 2 against the additive baseline (Strata ruling on #61 at
/// e35b2b76): stair index ≤ 1.1× B summed over seeds 1–20 and ≤ 1.2× B per
/// seed, `B = faults-off + faulted undragged − flat`. Master must fail the
/// summed bar (measured 1.242), so the bar still sees the artifact. Post vs
/// faults-off and master are printed (informational).
#[test]
#[ignore = "slow: 60 cube runs; passes at c080c695 (sum 0.961, seeds 0.855 (2) to 1.018 (13), seed 15 0.989)"]
fn stair_index_vs_additive_baseline() {
    let (mut sp, mut sm, mut sb) = (0.0, 0.0, 0.0);
    let mut fails = Vec::new();
    for seed in 1u64..=20 {
        let Some(s) = body(seed) else { continue };
        let (p, mm, b, o) = m::stair_with_baseline(seed, SHAPE, &s);
        if seed == 1 {
            eprintln!();
        }
        eprintln!(
            "seed {seed}: post/B {:.3} master/B {:.3} post/off {:.3}",
            p / b,
            mm / b,
            p / o
        );
        if p > 1.2 * b {
            fails.push((seed, p / b));
        }
        (sp, sm, sb) = (sp + p, sm + mm, sb + b);
    }
    eprintln!("sum: post/B {:.3} master/B {:.3}", sp / sb, sm / sb);
    assert!(sp <= 1.1 * sb, "sum: post/B {:.3}", sp / sb);
    assert!(fails.is_empty(), "per seed (seed, post/B): {fails:?}");
    assert!(sm > 1.1 * sb, "master passes the bar (vacuous): {:.3}", sm / sb);
}

/// The per-seed baseline bar (≤ 1.2× B) on seeds 1–3, in CI.
#[test]
fn stair_index_seed_checks() {
    for seed in 1u64..=3 {
        let s = body(seed).expect("salt");
        let (p, _, b, _) = m::stair_with_baseline(seed, SHAPE, &s);
        assert!(p <= 1.2 * b, "seed {seed}: post {p} vs B {b}");
    }
}

/// Spec test M1c (#61 review): drag-warp stretch in the band (footprint
/// columns included). Voxels stretched > 5× are ≤ 1 % of the band, and
/// those stretched > 2× exceed the same warp on the unfaulted labels by at
/// most 0.1 % (measured ≤ 0.009 % and ≤ +0.02 %, seeds 1–20).
fn stretch_ok(seed: u64) -> Result<(), String> {
    let Some(s) = body(seed) else { return Ok(()) };
    let (p2, p5) = m::stretch_fractions(&m::run(&m::cfg(seed, SHAPE, 4, Mode::PostFault)), &s);
    let (o2, _) = m::stretch_fractions(&m::unfaulted_warp_run(seed, SHAPE), &s);
    if p5 > 0.01 || p2 > o2 + 0.001 {
        return Err(format!(
            "seed {seed}: >5x {p5:.5}, >2x {p2:.5} vs unfaulted {o2:.5}"
        ));
    }
    Ok(())
}

#[test]
fn stretch_seed_14() {
    stretch_ok(14).unwrap(); // the largest >2x fraction of seeds 1-20
}

#[test]
#[ignore = "slow: 40 cube runs; passes (seeds 1-20)"]
fn stretch_meets_spec_bars_seeds_1_20() {
    let e: Vec<String> = (1u64..=20).filter_map(|s| stretch_ok(s).err()).collect();
    assert!(e.is_empty(), "{e:?}");
}

/// Spec test 3 (synthetic): a planar 60° fault of throw T between two
/// columns; both columns are dragged by D = −3 on horizon 1. The dragged
/// boundary keeps the offset T ± 1 across the fault.
#[test]
fn drag_follows_faulted_horizons() {
    let (nk, nh, t) = (64usize, 2usize, 7usize);
    // Columns 0..3 footwall (boundary 20), 3..6 hanging wall (20 + T): the
    // 60° fault's offset between adjacent columns of a vertical section.
    let shape = [1usize, 6, nk];
    let mut labels = vec![0u8; 6 * nk];
    for c in 0..6 {
        let top = if c < 3 { 20 } else { 20 + t };
        for k in top..nk {
            labels[c * nk + k] = 1;
        }
    }
    let mut table = [u8::MAX; 256];
    table[0] = 0;
    table[1] = 1;
    let d: Vec<f64> = (0..6).flat_map(|_| [0.0, -3.0]).collect();
    let maps: Vec<f64> = (0..6).flat_map(|_| [0.0, 20.0]).collect();
    synthoseis_core::salt::drag_warp_labels(&mut labels, shape, &d, &maps, nh, &table, None);
    let top = |c: usize| (0..nk).find(|&k| labels[c * nk + k] == 1).unwrap() as i64;
    for c in 0..6 {
        let want = if c < 3 { 17 } else { 17 + t as i64 };
        assert!(
            (top(c) - want).abs() <= 1,
            "column {c}: top {} want {want}",
            top(c)
        );
    }
    let off = top(3) - top(2);
    assert!((off - t as i64).abs() <= 1, "offset {off} vs throw {t}");
}

/// Spec test 3 on a real cube: every output label-run top inside the drag
/// band is the source run top (of the faulted labels) lifted by its own
/// offset at its depth, |k − a − D(a)| ≤ 1 (`a` the faulted run top), or below
/// it where the monotone clamp binds. Kills knots taken from the unfaulted
/// fill.
#[test]
fn drag_knots_are_faulted_run_tops() {
    let (tops, bad) = knot_misses(1, 0);
    assert!(tops > 1000, "{tops} run tops in the band");
    assert_eq!(bad, 0, "{bad} of {tops} run tops off their knot");
}

/// The knot check restricted to run tops the faults moved: the faulted run
/// top lies ≥ 3 samples from the same label's top in the unfaulted fill
/// (seed 7). Knots taken from the unfaulted labels (mutant M3) put exactly
/// these tops at the unfaulted top + D, off their knot.
#[test]
fn drag_knots_follow_fault_displaced_tops() {
    let (tops, bad) = knot_misses(7, 3);
    assert!(tops > 200, "{tops} displaced run tops in the band");
    assert_eq!(bad, 0, "{bad} of {tops} displaced run tops off their knot");
}

/// `(run tops checked, run tops off their knot)` of `seed`; with
/// `min_throw > 0` only tops whose faulted source top is at least that far
/// from the label's top in the unfaulted fill.
fn knot_misses(seed: u64, min_throw: usize) -> (usize, usize) {
    let c = m::cfg(seed, SHAPE, 4, Mode::PostFault);
    let r = m::run(&c);
    let und = (min_throw > 0).then(|| m::undragged(seed, SHAPE));
    let d = synthoseis_core::pipeline_stream::salt_drag_offsets(&c).expect("drag");
    let table = synthoseis_core::pipeline_stream::salt_drag_label_table(&c).expect("table");
    let maps = m::maps(&c);
    let [ni, nj, nk] = SHAPE;
    let nh = d.len() / (ni * nj);
    let (mut tops, mut bad) = (0usize, 0usize);
    for q in 0..ni * nj {
        let dc = &d[q * nh..(q + 1) * nh];
        if !synthoseis_core::salt::drag_column_active(dc) {
            continue;
        }
        let mc = &maps[q * nh..(q + 1) * nh];
        let dz = |z: f64| synthoseis_core::salt::depth_offset(mc, dc, z);
        let mut prev_b = 0.0f64;
        for k in 1..nk {
            let v = q * nk + k;
            let (l, lp) = (r.labels[v], r.labels[v - 1]);
            // Skip run tops whose source is clamped to the last cell (the
            // run starts below the cube).
            if l == lp || l == 255 || table[l as usize] == 255 || r.warp[v] as usize >= nk - 1 {
                continue;
            }
            // The faulted run top `a` lies in (w⁻¹(k − 1), w⁻¹(k)]: where the
            // warp stretches, output cells skip source cells.
            let (lo, hi) = (r.warp[v - 1] as f64 + 1.0, r.warp[v] as f64);
            if let Some(u) = &und {
                let col = &u[q * nk..(q + 1) * nk];
                let moved = col
                    .iter()
                    .position(|&x| x == l)
                    .is_none_or(|t| (t as f64 - hi).abs() >= min_throw as f64);
                if !moved {
                    continue;
                }
            }
            // The source top with its knot nearest to k (D varies with
            // depth, so solve a + D(a) = k over the bracket).
            let src = (0..=((hi - lo).max(0.0) * 4.0) as usize)
                .map(|t| lo + t as f64 * 0.25)
                .min_by(|a, b| {
                    let e = |x: f64| (k as f64 - x - dz(x)).abs();
                    e(*a).total_cmp(&e(*b))
                })
                .unwrap_or(lo);
            let want = src + dz(src);
            // The monotone clamp only pushes a knot down (a pinched-out
            // interval between the two runs, or a lower b before it).
            let adjacent = lp != 255
                && table[lp as usize] != 255
                && table[l as usize] == table[lp as usize] + 1;
            let clamped = !adjacent || want < prev_b - 1e-9 || lo + dz(lo) < prev_b - 1e-9;
            if clamped && (k as f64) < want - 1.0 - 1e-9 {
                bad += 1; // the clamp never lifts above the knot
            }
            prev_b = prev_b.max(want);
            tops += 1;
            if !clamped && (k as f64 - want).abs() > 1.0 + 1e-9 {
                bad += 1;
                if bad <= 8 {
                    let col = &r.labels[q * nk..(q + 1) * nk];
                    let w = &r.warp[q * nk..(q + 1) * nk];
                    eprintln!("q {q} k {k} l {l} h {} src {src} want {want} prev_b {prev_b} labels {:?} warp {:?}", table[l as usize], &col[k.saturating_sub(4)..(k + 4).min(nk)], &w[k.saturating_sub(4)..(k + 4).min(nk)]);
                }
            }
        }
    }
    (tops, bad)
}

/// No column leaves the drag with inverted horizon order (#61 delta, Strata
/// ruling at e35b2b76), seeds 1–20, both directions. For every run top `a`
/// of the faulted labels before the warp, its expected output position is
/// the push-up clamp of the targets, `b_r = max(0, min(a_r + D(a_r),
/// b_{r+1}))` (computed here, bottom-up over the column's run tops), and
/// its output position (the first cell whose source is at or below `a`)
/// lies within 1 sample of `b` either way: not pinned below (the old
/// pin-down clamp; seeds 1, 4, 6 fail at 81d486a2) and not lifted above.
/// The 1 sample is the half-sample rounding of `floor(w⁻¹(k + ½))`, the
/// same for every seed. Tops a knot in the pad below the cube can push
/// (target ≥ nk + min D) are checked only for "not pinned below". Also: no band column is non-monotone
/// (salt and unset excluded) over a source column that is monotone (salt
/// included): the warp is monotone, so only the pad can do that (seed 15 had
/// 2 at e35b2b76: the pad fell back to the bottom label in interval gaps).
/// Columns whose source is non-monotone only under the salt mask (fault
/// repeats the drag lifts into view) are printed. Second M3 killer with
/// `drag_knots_are_faulted_run_tops`.
#[test]
#[ignore = "slow: 20 seeds; passes (0 off by more than 1 sample, 0 drag-made non-monotone columns)"]
fn drag_keeps_horizon_order_seeds_1_20() {
    let mut bad = Vec::new();
    for seed in 1u64..=20 {
        let Some(o) = horizon_order(seed) else { continue };
        eprintln!(
            "seed {seed}: tops {} (pad-reach skipped {}) below {} above {} | columns using the \
             tolerance {} | drag-made non-monotone {} | non-monotone from under the salt mask {}",
            o.tops, o.pad_skip, o.below, o.above, o.tol_cols, o.nonmono, o.unmasked
        );
        if o.below + o.above + o.nonmono > 0 {
            bad.push((seed, o.below, o.above, o.nonmono));
        }
    }
    assert!(bad.is_empty(), "(seed, below, above, non-monotone): {bad:?}");
}

/// The order check on seeds 1, 4, 6 and 15, in CI.
#[test]
fn drag_keeps_horizon_order() {
    for seed in [1u64, 4, 6, 15] {
        let o = horizon_order(seed).expect("salt");
        assert!(o.tops > 1000, "seed {seed}: {} tops", o.tops);
        assert_eq!(
            (o.below, o.above, o.nonmono),
            (0, 0, 0),
            "seed {seed}: (below, above, drag-made non-monotone)"
        );
    }
}

struct Order {
    tops: usize,
    below: usize,
    above: usize,
    tol_cols: usize,
    nonmono: usize,
    /// Tops skipped: within reach of a knot in the pad below the cube.
    pad_skip: usize,
    /// Columns whose output order is non-monotone where the source is
    /// non-monotone only under the salt mask (fault repeats lifted into
    /// view; informational).
    unmasked: usize,
}

fn horizon_order(seed: u64) -> Option<Order> {
    let s = body(seed)?;
    let c = m::cfg(seed, SHAPE, 4, Mode::PostFault);
    let r = m::run(&c);
    let src = synthoseis_core::pipeline_stream::faulted_labels_undragged(&c);
    let d = synthoseis_core::pipeline_stream::salt_drag_offsets(&c).expect("drag");
    let maps = m::maps(&c);
    let [ni, nj, nk] = SHAPE;
    let nh = d.len() / (ni * nj);
    let fp = m::footprint(&s, SHAPE);
    let dist = m::distance(&fp, ni, nj, m::BAND);
    let mut o = Order {
        tops: 0,
        below: 0,
        above: 0,
        tol_cols: 0,
        nonmono: 0,
        pad_skip: 0,
        unmasked: 0,
    };
    let mono = |col: &[u8], q: usize, mask: bool| {
        let v: Vec<u8> = (0..nk)
            .filter(|&k| col[k] != 255 && !(mask && s.contains(q, k)))
            .map(|k| col[k])
            .collect();
        v.windows(2).all(|w| w[1] >= w[0])
    };
    for q in 0..ni * nj {
        let (col, w) = (&src[q * nk..(q + 1) * nk], &r.warp[q * nk..(q + 1) * nk]);
        // The warp is monotone, so output order is a subsequence of the
        // source's: a non-monotone output column over a source that is
        // monotone (salt cells included) can only come from the pad.
        if dist[q] <= m::BAND && !mono(&r.labels[q * nk..(q + 1) * nk], q, true) {
            if mono(col, q, false) {
                o.nonmono += 1;
            } else if mono(col, q, true) {
                o.unmasked += 1;
            }
        }
        let dc = &d[q * nh..(q + 1) * nh];
        if !synthoseis_core::salt::drag_column_active(dc) {
            continue;
        }
        let mc = &maps[q * nh..(q + 1) * nh];
        let tops: Vec<usize> = (1..nk).filter(|&a| col[a] != col[a - 1]).collect();
        let mut b: Vec<f64> = tops
            .iter()
            .map(|&a| a as f64 + synthoseis_core::salt::depth_offset(mc, dc, a as f64))
            .collect();
        for t in (0..b.len().saturating_sub(1)).rev() {
            b[t] = b[t].min(b[t + 1]);
        }
        // A knot in the pad (source at or below nk) has target ≥ nk + min D;
        // tops whose clamp could reach it depend on the pad fill.
        let reach = nk as f64 + dc.iter().copied().fold(0.0f64, f64::min);
        let mut tol = false;
        for (t, &a) in tops.iter().enumerate() {
            let Some(y) = w.iter().position(|&x| x as usize >= a) else { continue };
            let want = b[t].max(0.0);
            let e = y as f64 - want;
            o.below += usize::from(e > 1.0 + 1e-9);
            // The pad only pushes up: such tops are checked one way.
            if b[t] >= reach - 1e-9 {
                o.pad_skip += 1;
                continue;
            }
            o.tops += 1;
            o.above += usize::from(e < -1.0 - 1e-9);
            tol |= e.abs() > 0.5 + 1e-9;
        }
        o.tol_cols += usize::from(tol);
    }
    Some(o)
}

/// Spec tests 4 and 5 on seed 1: the warp only lifts (`k − w⁻¹(k) ≤ 0`),
/// mean lift in the first 3 columns off the flank ≥ 1 sample (measured
/// 4.21), monotone in every column, no unset voxel (no interior water
/// beyond master's, which has the same 0 on seed 1 and 2 seabed speckles
/// on seed 10, at the same voxels (125,111,35) and (126,114,35) under post-fault
/// drag; faults-off has none, so they come from faulting).
#[test]
fn drag_sense_and_warp_monotone() {
    let s = body(1).expect("seed 1 salt");
    let r = m::run(&m::cfg(1, SHAPE, 4, Mode::PostFault));
    let (worst, lift, nonmono, unset) = m::warp_checks(&r, &s);
    let (.., unset_master) = m::warp_checks(&m::run(&m::cfg(1, SHAPE, 4, Mode::Master)), &s);
    assert!(worst <= 0, "output below its source by {worst}");
    assert!(lift >= 1.0, "mean lift {lift}");
    assert_eq!(nonmono, 0, "{nonmono} non-monotone columns");
    assert!(
        unset <= unset_master,
        "{unset} unset voxels (master {unset_master})"
    );
    // Sources may lie in the pad below the cube (at most ⌈max −D⌉ + 1).
    let d =
        synthoseis_core::pipeline_stream::salt_drag_offsets(&m::cfg(1, SHAPE, 4, Mode::PostFault))
            .expect("drag");
    let pad = (-d.iter().copied().fold(0.0f64, f64::min)).ceil() as usize + 1;
    assert!(r.warp.iter().all(|&w| (w as usize) < SHAPE[2] + pad));
}

/// Spec tests 6 and 7: the salt body and the fault ids are those of master
/// (481805cc order). Kills salt gathered through the faults.
#[test]
fn salt_and_fault_labels_unchanged() {
    for seed in [1u64, 7] {
        let cp = m::cfg(seed, SHAPE, 4, Mode::PostFault);
        let cm = m::cfg(seed, SHAPE, 4, Mode::Master);
        let (sp, sm) = (m::salt(&cp).expect("salt"), m::salt(&cm).expect("salt"));
        let n = SHAPE[0] * SHAPE[1];
        for q in 0..n {
            for k in 0..SHAPE[2] {
                assert_eq!(
                    sp.contains(q, k),
                    sm.contains(q, k),
                    "seed {seed} salt at {q},{k}"
                );
            }
        }
        assert!(
            m::run(&cp).fault_id == m::run(&cm).fault_id,
            "seed {seed} fault ids differ"
        );
    }
}

/// Spec test 8: layer labels equal master's at output columns farther than
/// 12 + heave halo (Chebyshev) from any lifted cell (D ≠ 0). On 256×256
/// columns: at 128×128 the halo (39 with 4 faults) plus 12 covers every
/// column of seeds 1 and 13.
#[test]
fn far_field_labels_unchanged() {
    for seed in [1u64, 13] {
        let cp = m::cfg(seed, FAR_SHAPE, 4, Mode::PostFault);
        let d = synthoseis_core::pipeline_stream::salt_drag_offsets(&cp).expect("drag");
        let halo = synthoseis_core::fault_model(&cp).map_or(0, |f| f.heave_halo());
        let [ni, nj, nk] = FAR_SHAPE;
        let nh = d.len() / (ni * nj);
        let lifted: Vec<bool> = (0..ni * nj)
            .map(|q| d[q * nh..(q + 1) * nh].iter().any(|&x| x != 0.0))
            .collect();
        let dist = m::distance(&lifted, ni, nj, 12 + halo + 1);
        let (rp, rm) = (
            m::run(&cp),
            m::run(&m::cfg(seed, FAR_SHAPE, 4, Mode::Master)),
        );
        let mut far = 0;
        for q in 0..ni * nj {
            if dist[q] > 12 + halo {
                far += 1;
                assert!(
                    rp.labels[q * nk..(q + 1) * nk] == rm.labels[q * nk..(q + 1) * nk],
                    "seed {seed} column {q}"
                );
            }
        }
        eprintln!("seed {seed}: {far} far-field columns (halo {halo})");
        assert!(far > 0, "seed {seed}: no far-field column");
    }
}

/// Jaccard vs master over near-salt non-salt voxels outside master's
/// exposed region (informational since the Strata ruling at e35b2b76:
/// printed by `drag_band_jaccard_vs_analytic`; seed 1 0.8959, mean 0.9644).
fn jaccard_out(seed: u64) -> Option<f64> {
    let s = body(seed)?;
    let rm = m::run(&m::cfg(seed, SHAPE, 4, Mode::Master));
    let skip = m::exposed_mask(&rm, &s, &m::undragged(seed, SHAPE));
    Some(m::jaccard_outside(
        &m::run(&m::cfg(seed, SHAPE, 4, Mode::PostFault)),
        &rm,
        &s,
        &skip,
    ))
}

/// Label parity in the drag band (Strata ruling on #61 at e35b2b76):
/// Jaccard vs an independent post-fault drag (`m::analytic_drag`, per output
/// column at present depth, not `drag_warp_labels`) ≥ 0.97 per seed and
/// ≥ 0.98 mean over seeds 1–20 (measured at c080c695: min 0.9825 seed 16, mean 0.9953).
/// Jaccard vs master outside its exposed region is printed (informational:
/// master drags in pre-fault coordinates).
#[test]
#[ignore = "slow: 40 cube runs; passes at c080c695 (min 0.9825 seed 16, mean 0.9953)"]
fn drag_band_jaccard_vs_analytic() {
    let mut js = Vec::new();
    for seed in 1u64..=20 {
        let Some(s) = body(seed) else { continue };
        let j = m::jaccard_vs_analytic(seed, SHAPE, &s);
        if seed == 1 {
            eprintln!();
        }
        eprintln!("seed {seed}: J analytic {j:.4} J master-out {:.4}", jaccard_out(seed).unwrap());
        js.push((seed, j));
    }
    let mean = js.iter().map(|x| x.1).sum::<f64>() / js.len() as f64;
    let low: Vec<_> = js.iter().filter(|x| x.1 < 0.97).collect();
    eprintln!("mean {mean:.4}");
    assert!(low.is_empty() && mean >= 0.98, "mean {mean:.4}, below 0.97: {low:?}");
}

/// The per-seed analytic Jaccard bar on seed 2, in CI.
#[test]
fn drag_band_jaccard_analytic_seed_check() {
    let s = body(2).expect("seed 2 salt");
    let j = m::jaccard_vs_analytic(2, SHAPE, &s);
    assert!(j >= 0.97, "seed 2: {j}");
}

fn output_masks(seed: u64, mode: Mode, time: bool) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut c = m::cfg(seed, [64, 64, 128], 4, mode);
    c.time = if time {
        synthoseis_core::TimeConfig::default()
    } else {
        synthoseis_core::TimeConfig::legacy()
    };
    let c = c.with_geometry_cache();
    (
        synthoseis_core::generate_salt_labels_output(&c),
        synthoseis_core::generate_fault_labels_output(&c),
    )
}

/// Spec tests 6 and 7 on the written masks, depth axis
/// (`--legacy-depth-as-time`): `salt_labels` and `fault_labels` equal
/// master's.
#[test]
fn output_masks_unchanged_depth() {
    for seed in [1u64, 7] {
        let (sp, fp) = output_masks(seed, Mode::PostFault, false);
        let (sm, fm) = output_masks(seed, Mode::Master, false);
        assert!(sp.is_some() && fp.is_some(), "seed {seed}: masks");
        assert!(sp == sm, "seed {seed}: salt_labels differ");
        assert!(fp == fm, "seed {seed}: fault_labels differ");
    }
}

/// Spec tests 6 and 7 in time mode (Strata ruling on #61, her test
/// `zz_strata_timemask.rs`): depth-domain salt and fault labels equal
/// master's byte for byte, and the time-mode masks are exactly those depth
/// labels mapped through the NEW time-depth function.
#[test]
fn time_masks_are_depth_masks_through_new_twt() {
    for seed in [1u64, 7, 13] {
        let mut c = m::cfg(seed, [64, 64, 128], 4, Mode::PostFault);
        c.time = synthoseis_core::TimeConfig::default();
        let c = c.with_geometry_cache();
        let mut cm = m::cfg(seed, [64, 64, 128], 4, Mode::Master);
        cm.time = synthoseis_core::TimeConfig::default();
        let cm = cm.with_geometry_cache();
        let shape = c.shape();
        let (labels, _) = synthoseis_core::pipeline_stream::generate_labels(&c);
        let model = synthoseis_core::rock_physics::elastic_model(&c, &labels, shape);
        let axis = model.time().unwrap().clone();
        let t = tile_twt(&model, &labels, shape, 0, shape[0], 0, shape[1], &axis);
        // depth masks of the PR and of master (time-independent in depth)
        let sd = synthoseis_core::salt::generate_salt_labels(&c).unwrap();
        let fd = synthoseis_core::pipeline_stream::generate_fault_labels(&c).unwrap();
        assert_eq!(
            sd,
            synthoseis_core::salt::generate_salt_labels(&cm).unwrap(),
            "seed {seed} salt depth"
        );
        assert_eq!(
            fd,
            synthoseis_core::pipeline_stream::generate_fault_labels(&cm).unwrap(),
            "seed {seed} fault depth"
        );
        let [f, s, _] = synthoseis_core::time_mode::generate_output_masks(&c);
        let rs = |d: &[u8]| resample_columns(d, shape[2], &t, axis.nt, axis.dt_ms);
        assert!(s.unwrap() == rs(&sd), "seed {seed}: salt time != depth∘T");
        assert!(f.unwrap() == rs(&fd), "seed {seed}: fault time != depth∘T");
        // The mapping is non-trivial: master's TWT differs.
        let (lm, _) = synthoseis_core::pipeline_stream::generate_labels(&cm);
        let mm = synthoseis_core::rock_physics::elastic_model(&cm, &lm, shape);
        let tm = tile_twt(&mm, &lm, shape, 0, shape[0], 0, shape[1], &axis);
        let dmax = t
            .iter()
            .zip(&tm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "seed {seed}: max |ΔTWT| vs master {dmax:.2} ms (dt {} ms)",
            axis.dt_ms
        );
        assert!(
            dmax > 0.0,
            "seed {seed}: new and master TWT agree (vacuous)"
        );
    }
}
