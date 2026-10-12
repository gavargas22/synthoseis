//! 3D closure segmentation across faults (the default for closures per sand
//! unit) vs the legacy generator, and the `closures_unsegmented` switch
//! (`--closures-unsegmented`, master ef2dc42).
//!
//! Fixture: `tests/fixtures/closure_segments_reference.json` from
//! `tests/fixtures/generate_closure_segments_reference.py`, which runs the
//! real legacy `Closures._flood_fill` (+ the top / base clamp of
//! `create_closure_labels_from_depth_maps`) and `Closures.segment_closures`.
//! * Bit-exact: the per-column closure depth of faulted domes (fault cliffs,
//!   max-column cap, base clamp where a fault offsets the unit by more than
//!   its thickness).
//! * Bit-exact: 3D components (18-connectivity) and the min-voxel filter of
//!   voxel run sets.
//! * Statistical at the 5 % level (chi-squared): uniform brine / oil / gas
//!   for split-off compartments, independent of the primary draw.
use serde::Deserialize;
use synthoseis_core::closure_segments::{
    segment_runs, segmented_sand_unit_fluids, split_compartment_fluid, unit_closure_runs,
};
use synthoseis_core::lithology::interval_sand;
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, FilterConfig, RockPhysicsConfig};
use synthoseis_core::rock_physics::{closure_fluid, elastic_model, ElasticModel};
use synthoseis_core::{generate_chunked, generate_labels, ToyGeometry};

#[derive(Deserialize)]
struct FillCase {
    shape: [usize; 2],
    max_column: f64,
    t: Vec<usize>,
    b: Vec<usize>,
    cd2: Vec<i64>,
}

#[derive(Deserialize)]
struct SegmentCase {
    shape: [usize; 3],
    min_voxels: usize,
    runs: Vec<[usize; 3]>,
    label: Vec<usize>,
}

#[derive(Deserialize)]
struct Fixture {
    fill: Vec<FillCase>,
    segment: Vec<SegmentCase>,
}

fn fixture() -> Fixture {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/closure_segments_reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Bit-exact: the closure depth of every column equals legacy
/// `min(max(_flood_fill(t), t), b)` on faulted domes; the closure voxels
/// are `[t, ceil(cd))`. Legacy labels `int(cd) - t` voxels, which is the
/// same count except where a fractional max column (37.5 samples = 150 m
/// at 4 m) caps the closure (a pre-existing one-voxel rounding difference,
/// kept from ef2dc42 and counted here).
#[test]
fn closure_depth_matches_legacy_flood_fill() {
    let fx = fixture();
    assert!(fx.fill.len() >= 20);
    let (mut closed, mut capped_frac, mut base_clamped) = (0, 0, 0);
    for (n, c) in fx.fill.iter().enumerate() {
        let [ni, nj] = c.shape;
        let nk = 128;
        let mut labels = vec![0u8; ni * nj * nk];
        for col in 0..ni * nj {
            let l = &mut labels[col * nk..(col + 1) * nk];
            l[c.t[col]..c.b[col]].fill(1);
            l[c.b[col]..].fill(2);
        }
        let runs = unit_closure_runs(&labels, [ni, nj, nk], &[1], 1, c.max_column);
        let mut got = vec![None; ni * nj];
        for r in &runs {
            assert_eq!(r.k0, c.t[r.col], "case {n} col {}", r.col);
            assert!(got[r.col].is_none());
            got[r.col] = Some(*r);
        }
        for col in 0..ni * nj {
            let cd = c.cd2[col] as f64 / 2.0;
            let t = c.t[col] as f64;
            match got[col] {
                None => assert_eq!(cd, t, "case {n} col {col}: legacy closed, Rust not"),
                Some(r) => {
                    closed += 1;
                    assert_eq!(
                        (2.0 * r.contact).to_bits(),
                        (c.cd2[col] as f64).to_bits(),
                        "case {n} col {col}"
                    );
                    assert_eq!(r.k1, cd.ceil() as usize, "case {n} col {col}");
                    let legacy_count = cd as usize - c.t[col];
                    if cd.fract() != 0.0 {
                        assert_eq!(r.k1 - r.k0, legacy_count + 1);
                        capped_frac += 1;
                    } else {
                        assert_eq!(r.k1 - r.k0, legacy_count, "case {n} col {col}");
                    }
                    base_clamped += (r.k1 == c.b[col]) as usize;
                }
            }
        }
    }
    eprintln!("fill: {closed} closed columns bit-exact, {base_clamped} clamped to the base, {capped_frac} fractional caps");
    assert!(closed > 5000 && base_clamped > 1000 && capped_frac > 0);
}

/// Bit-exact: components and kept / removed runs equal legacy
/// `segment_closures` (up to label numbering).
#[test]
fn segmentation_matches_legacy_segment_closures() {
    let fx = fixture();
    let (mut runs_total, mut comps_total, mut removed) = (0, 0, 0);
    for (n, c) in fx.segment.iter().enumerate() {
        let [ni, nj, _] = c.shape;
        let runs: Vec<(usize, usize, usize)> = c.runs.iter().map(|r| (r[0], r[1], r[2])).collect();
        let comp = segment_runs(&runs, ni, nj);
        let nc = comp.iter().max().map_or(0, |m| m + 1);
        let mut voxels = vec![0usize; nc];
        for (r, &k) in comp.iter().enumerate() {
            voxels[k] += runs[r].2 - runs[r].1;
        }
        let mut to_legacy = vec![None; nc];
        let mut from_legacy = std::collections::HashMap::new();
        for (r, &k) in comp.iter().enumerate() {
            let kept = voxels[k] >= c.min_voxels.max(1);
            let l = c.label[r];
            assert_eq!(kept, l != 0, "case {n} run {r}: kept");
            if !kept {
                removed += 1;
                continue;
            }
            assert_eq!(
                *to_legacy[k].get_or_insert(l),
                l,
                "case {n} run {r}: split vs legacy"
            );
            assert_eq!(
                *from_legacy.entry(l).or_insert(k),
                k,
                "case {n} run {r}: merged vs legacy"
            );
        }
        runs_total += runs.len();
        comps_total += from_legacy.len();
    }
    eprintln!("segment: {runs_total} runs, {comps_total} kept components, {removed} removed runs");
    assert!(runs_total > 10_000 && comps_total > 200 && removed > 1000);
}

/// Labels of one sand unit (label 1, `thick` samples) on a dome cut by a
/// fault at `i = 20` with `throw` samples of throw; shale above (0) and
/// below (2).
fn faulted_dome(throw: usize, thick: usize) -> (Vec<u8>, [usize; 3]) {
    let (ni, nj, nk) = (40, 36, 96);
    let mut labels = vec![0u8; ni * nj * nk];
    for i in 0..ni {
        for j in 0..nj {
            let r2 = (i as f64 - 19.5).powi(2) + (j as f64 - 17.5).powi(2);
            let t = ((20.0 + 0.1 * r2).floor() as usize).min(60) + if i >= 20 { throw } else { 0 };
            let col = i * nj + j;
            let l = &mut labels[col * nk..(col + 1) * nk];
            l[t..t + thick].fill(1);
            l[t + thick..].fill(2);
        }
    }
    (labels, [ni, nj, nk])
}

/// A dome on one sand unit cut by a fault with more throw than the unit is
/// thick: one 2D closure region, two 3D compartments (legacy
/// `segment_closures` splits it the same way), each with its own draw.
/// With less throw than thickness it stays one compartment.
#[test]
fn fault_offset_splits_a_closure() {
    let intervals = [0usize, 1, 2];
    let sand = [false, true, false, false];
    let max_column = 1e9;
    let (labels, shape) = faulted_dome(8, 4);
    let nj = shape[1];
    let runs = unit_closure_runs(&labels, shape, &[1], 1, max_column);
    assert!(runs.iter().all(|r| r.rank == 0), "one 2D region");
    assert!(runs.iter().any(|r| r.col / nj >= 20) && runs.iter().any(|r| r.col / nj < 20));
    for seed in 0..64 {
        let (fluids, comps) =
            segmented_sand_unit_fluids(&labels, shape, &intervals, &sand, seed, max_column, 1);
        assert_eq!(comps.len(), 2, "two compartments");
        assert!(comps
            .iter()
            .all(|c| c.kept && c.pieces == 1 && c.units == 1));
        assert_eq!(comps.iter().filter(|c| c.primary).count(), 1);
        let (_, f) = fluids.iter().find(|(lab, _)| *lab == 1).unwrap();
        assert_eq!(f.closures.len(), 2);
        let primary = comps.iter().find(|c| c.primary).unwrap();
        let split = comps.iter().find(|c| !c.primary).unwrap();
        assert_eq!(primary.fluid, closure_fluid(seed, 1, 0));
        let first_split = runs.iter().find(|r| r.col / nj >= 20).unwrap().col;
        assert_eq!(
            split.fluid,
            split_compartment_fluid(seed, 1, 0, first_split)
        );
        // Each fault block keeps its own fluid.
        for r in &runs {
            let want = if r.col / nj >= 20 {
                split.fluid
            } else {
                primary.fluid
            };
            assert_eq!(f.fluid[r.col], want);
        }
    }
    let (labels, shape) = faulted_dome(3, 4);
    let (fluids, comps) =
        segmented_sand_unit_fluids(&labels, shape, &intervals, &sand, 5, max_column, 1);
    assert_eq!(comps.len(), 1);
    assert!(comps[0].primary && comps[0].columns > 100);
    assert_eq!(
        fluids
            .iter()
            .find(|(lab, _)| *lab == 1)
            .unwrap()
            .1
            .closures
            .len(),
        1
    );
}

fn chi2_uniform(counts: &[f64]) -> f64 {
    let n: f64 = counts.iter().sum();
    let e = n / counts.len() as f64;
    counts.iter().map(|o| (o - e).powi(2) / e).sum()
}

/// Statistical, 5 % level: split-off compartment draws are uniform over
/// brine / oil / gas (chi-squared, df 2, critical 5.991) and independent
/// of the primary draw of the same closure (3x3 contingency, df 4,
/// critical 9.488).
#[test]
fn split_compartment_fluid_uniform_and_independent() {
    let mut counts = [0f64; 3];
    let mut table = [[0f64; 3]; 3];
    for seed in 0..10u64 {
        for layer in 0..10usize {
            for rank in 0..5u64 {
                for col in (0..4000usize).step_by(67) {
                    let s = split_compartment_fluid(seed, layer, rank, col) as usize;
                    let p = closure_fluid(seed, layer, rank) as usize;
                    counts[s] += 1.0;
                    table[p][s] += 1.0;
                }
            }
        }
    }
    let n: f64 = counts.iter().sum();
    let x2 = chi2_uniform(&counts);
    let (rows, cols): (Vec<f64>, Vec<f64>) = (
        (0..3).map(|i| table[i].iter().sum()).collect(),
        (0..3).map(|j| (0..3).map(|i| table[i][j]).sum()).collect(),
    );
    let mut x2i = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let e = rows[i] * cols[j] / n;
            x2i += (table[i][j] - e).powi(2) / e;
        }
    }
    eprintln!("split fluids: n {n}, uniform chi2 {x2:.3} (5 % critical 5.991), vs primary chi2 {x2i:.3} (5 % critical 9.488)");
    assert!(x2 < 5.991, "uniform chi2 {x2}");
    assert!(x2i < 9.488, "independence chi2 {x2i}");
}

fn faulted_sandy(seed: u64, unsegmented: bool) -> E2eConfig {
    E2eConfig {
        time: Default::default(),
        seed,
        inline_count: 24,
        crossline_count: 20,
        samples: 128,
        store_path: None,
        chunk_shape: Some([8, 5, 128]),
        faults: FaultConfig::with_count(4),
        filters: FilterConfig::default(),
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: Some(0.5),
            sand_layer_thickness: 1.0,
            closures_unsegmented: unsegmented,
            // Pinned master b4f4259 scenario (tests/salt.rs covers salt).
            salt: false,
            ..RockPhysicsConfig::default()
        },
        geometry: ToyGeometry::Layered,
    }
}

/// On a faulted sandy cube the fault juxtaposes closed sands of two units
/// into one compartment; the output differs from `closures_unsegmented`,
/// which is the ef2dc42 model. Without faults both are identical.
#[test]
fn faults_merge_juxtaposed_closures_and_switch_restores_ef2dc42() {
    let c = faulted_sandy(7, false);
    let (labels, shape) = generate_labels(&c);
    let ElasticModel::Rpm(m) = elastic_model(&c, &labels, shape) else {
        panic!()
    };
    let sand = interval_sand(c.effective_lithology(), c.seed, m.nh, Some(0.5), 1.0);
    let rp = &c.rock_physics;
    let (_, comps) = segmented_sand_unit_fluids(
        &labels,
        shape,
        &m.intervals,
        &sand,
        c.seed,
        rp.max_column_m / rp.depth_step_m,
        rp.closure_minimum.voxels(shape[0], shape[1]),
    );
    assert!(
        comps.iter().any(|k| k.kept && k.units > 1),
        "juxtaposed units joined: {comps:?}"
    );
    let seg = generate_chunked(&c).0.angle_stack;
    let unseg = generate_chunked(&faulted_sandy(7, true)).0.angle_stack;
    assert_ne!(seg, unseg);
    // Without faults segmented == ef2dc42 bit for bit on the default
    // (partial voxels), with the scaled and with the legacy closure minimum.
    // The brine sliver master bad1daa8 shows here is caused by the #33
    // contact clamp: segmentation stores `min(fill, crest + max_column,
    // base)`, capped at the unit's integer base *cell*, while ef2dc42's
    // unsegmented contact has no base clamp. Partial voxels only expose it:
    // they resolve the sand between the integer base and the true unit base
    // (a sliver under half a cell), and that sliver keeps the brine
    // end-member; whole voxels give that cell another label. The
    // closure clamp (`ClosureRun::contact`) is unchanged; closure-minimum
    // spec §4 adds a separate fluid-contact cap, `ClosureRun::fluid_contact`
    // = `min(fill, cap, base + ½ cell)`, stored in the fluid maps, so the
    // sliver fills as in ef2dc42. Per §6.5a this check therefore no longer
    // runs through a whole-voxel pin. `legacy_closure_contact_cap` +
    // `ClosureMinimum::LEGACY` (master bad1daa8) still differ in 256 / 2,000
    // / 75 / 0 samples for seeds 7 / 1 / 2 / 3.
    use synthoseis_core::ClosureMinimum;
    for (seed, master_differs) in [(7u64, 256usize), (1, 2000), (2, 75), (3, 0)] {
        for minimum in [ClosureMinimum::Scaled, ClosureMinimum::LEGACY] {
            let flat = |u, cap| {
                let mut c = E2eConfig {
                    faults: FaultConfig::with_count(0),
                    ..faulted_sandy(seed, u)
                };
                c.rock_physics.closure_minimum = minimum;
                c.rock_physics.legacy_closure_contact_cap = cap && !u;
                // The master bad1daa8 counts predate the physical filter
                // edges (filter-edge spec §6 item 4).
                c.time.legacy_filter_edges = true;
                c
            };
            let (a, b) = (
                generate_chunked(&flat(false, false)).0.angle_stack,
                generate_chunked(&flat(true, false)).0.angle_stack,
            );
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "seed {seed} {minimum:?}, no faults: segmented == ef2dc42"
            );
            let l = generate_chunked(&flat(false, true)).0.angle_stack;
            let n = l
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!("seed {seed} {minimum:?}: legacy contact cap differs from unsegmented in {n}/{} samples", b.len());
            if minimum == ClosureMinimum::LEGACY {
                assert_eq!(
                    n, master_differs,
                    "seed {seed}: master bad1daa8 vs unsegmented"
                );
            }
        }
    }
}
