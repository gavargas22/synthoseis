//! 3D closure segmentation across faults: the default for closures per sand
//! unit. `RockPhysicsConfig::closures_unsegmented` (`--closures-unsegmented`)
//! restores master ef2dc42.
//!
//! This ports how legacy `datagenerator/Closures.py` turns closure maps into
//! closures (`create_closure_labels_from_depth_maps` → `_flood_fill` →
//! `segment_closures` → `assign_fluid_types`):
//!
//! 1. **Closure map per column.** Each sand unit's post-fault top surface is
//!    flood-filled, and the closure depth of a column is
//!    `min(fill, crest + max_column, base)`. This is legacy `_flood_fill`
//!    (per-region max-column cap) plus the `min(max(cd, top), base)` clamp.
//!    - The fill level is constant over a connected closed region. A fault
//!      block that forms its own pit is its own region, with its own crest
//!      and spill point, both here and in ef2dc42. The voxels of every
//!      region are the same as in ef2dc42; only the stored contact is now
//!      clamped to the unit base.
//!    - Legacy walls off fault gaps (NaN cells of the faulted depth maps),
//!      but its gap-inserting `Faults.partial_faulting` is never called, so
//!      the walls never engage there. They are not ported.
//! 2. **3D segmentation.** The closure voxels of all units are split into
//!    3D connected components with 18-connectivity (legacy
//!    `measure.label(connectivity=2)`, sand voxels only).
//!    - A trap offset by a fault so that its voxels no longer touch becomes
//!      separate compartments.
//!    - Sand juxtaposed across a fault joins the closures on both sides into
//!      one compartment.
//!    - Compartments below `min_closure_voxels` stay brine, as with legacy
//!      `remove_small_objects(closure_min_voxels)`.
//! 3. **Fluids.** One brine / oil / gas draw per compartment, as with
//!    legacy `rng.integers(3)` per component.
//!    - The draw is keyed by the compartment's primary closure: the smallest
//!      `(unit top, closure rank)` whose first voxel column lies in the
//!      compartment. With that key, [`closure_fluid`] reproduces the
//!      ef2dc42 draw for an unsplit closure.
//!    - A split-off compartment is keyed by `(unit top, rank, 1 + first
//!      column)`.
//!
//! Everything is computed from the full label volume, which every path
//! already rebuilds, so the result does not depend on tiling, workers or
//! processes. Memory is `O(closure columns)` for the runs plus one set of
//! unit maps at a time.
//!
//! Not ported: see docs/closure-segmentation-faults.md (the legacy `3x3x1`
//! opening, `grow_to_fault2`, and fault-gap walls).

use std::collections::VecDeque;

use synthoseis_closures::flood_fill_heap_2d;

use crate::rock_physics::{
    closure_fluid, first_unit_run, keyed_unit, Fluid, LayerFluids, STREAM_FLUID,
};

/// Closure voxels `[k0, k1)` of one column in one 2D closure region of a
/// sand unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosureRun {
    /// Unit top interval (the fluid key layer).
    pub layer: usize,
    /// Rank of the 2D closure region in the unit (raster order of its
    /// first column, counting every closed region as ef2dc42 does).
    pub rank: u64,
    pub col: usize,
    pub k0: usize,
    pub k1: usize,
    /// Per-column contact (`min(fill, crest + max_column, base)`).
    pub contact: f64,
}

/// 2D stage for one sand unit: closure voxel runs, in raster order of
/// their columns within each region, with regions in rank order.
pub fn unit_closure_runs(
    labels: &[u8],
    shape: [usize; 3],
    members: &[u8],
    layer: usize,
    max_column: f64,
) -> Vec<ClosureRun> {
    let [ni, nj, nk] = shape;
    let n = ni * nj;
    let mut unit = [false; 256];
    for &m in members {
        if m != 255 {
            unit[m as usize] = true;
        }
    }
    let mut top = vec![f64::NAN; n];
    let mut base = vec![0usize; n];
    for c in 0..n {
        if let Some((a, b)) = first_unit_run(&labels[c * nk..(c + 1) * nk], &unit) {
            top[c] = a as f64;
            base[c] = b;
        }
    }
    let mut runs = Vec::new();
    if ni < 3 || nj < 3 {
        return runs;
    }
    let filled = flood_fill_heap_2d(&top, [ni, nj], 1e30);
    let closed: Vec<bool> = (0..n)
        .map(|c| top[c].is_finite() && filled[c].is_finite() && filled[c] > top[c])
        .collect();
    let mut seen = vec![false; n];
    let mut rank = 0u64;
    for start in 0..n {
        if !closed[start] || seen[start] {
            continue;
        }
        let mut cells = Vec::new();
        let mut q = VecDeque::from([start]);
        seen[start] = true;
        while let Some(c) = q.pop_front() {
            cells.push(c);
            let (i, j) = (c / nj, c % nj);
            let nb = [
                (i > 0).then(|| c - nj),
                (i + 1 < ni).then(|| c + nj),
                (j > 0).then(|| c - 1),
                (j + 1 < nj).then(|| c + 1),
            ];
            for d in nb.into_iter().flatten() {
                if closed[d] && !seen[d] {
                    seen[d] = true;
                    q.push_back(d);
                }
            }
        }
        cells.sort_unstable();
        let crest = cells.iter().map(|&c| top[c]).fold(f64::INFINITY, f64::min);
        let cap = crest + max_column;
        for &c in &cells {
            let contact = filled[c].min(cap).min(base[c] as f64);
            let k0 = top[c] as usize;
            let k1 = (contact.ceil().max(0.0) as usize).min(base[c]);
            if k1 > k0 {
                runs.push(ClosureRun {
                    layer,
                    rank,
                    col: c,
                    k0,
                    k1,
                    contact,
                });
            }
        }
        rank += 1;
    }
    runs
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi] = lo;
    }
}

/// 3D connected components (18-connectivity, legacy
/// `measure.label(connectivity=2)`) of the voxel runs `(col, k0, k1)` on an
/// `(ni, nj)` grid. Runs in one column must not overlap. Returns a
/// component id per run; ids are numbered in order of first appearance.
pub fn segment_runs(runs: &[(usize, usize, usize)], ni: usize, nj: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by_key(|&r| (runs[r].0, runs[r].1));
    // Per column: range of `order`.
    let mut start = vec![usize::MAX; ni * nj];
    let mut end = vec![0usize; ni * nj];
    for (p, &r) in order.iter().enumerate() {
        let c = runs[r].0;
        if start[c] == usize::MAX {
            start[c] = p;
        }
        end[c] = p + 1;
    }
    let mut parent: Vec<usize> = (0..runs.len()).collect();
    // Face / edge neighbours (|dk| <= 1 allowed): same column and the two
    // forward in-plane orthogonal neighbours. In-plane diagonals need dk = 0.
    for (p, &r) in order.iter().enumerate() {
        let (c, a0, a1) = runs[r];
        let (i, j) = (c / nj, c % nj);
        for &q in &order[p + 1..end[c]] {
            let (_, b0, b1) = runs[q];
            if a0 <= b1 && b0 <= a1 {
                union(&mut parent, r, q);
            }
        }
        let nbs = [
            (i + 1 < ni).then(|| (c + nj, false)),
            (j + 1 < nj).then(|| (c + 1, false)),
            (i + 1 < ni && j + 1 < nj).then(|| (c + nj + 1, true)),
            (i + 1 < ni && j > 0).then(|| (c + nj - 1, true)),
        ];
        for (d, diag) in nbs.into_iter().flatten() {
            if start[d] == usize::MAX {
                continue;
            }
            for &q in &order[start[d]..end[d]] {
                let (_, b0, b1) = runs[q];
                let touch = if diag {
                    a0 < b1 && b0 < a1
                } else {
                    a0 <= b1 && b0 <= a1
                };
                if touch {
                    union(&mut parent, r, q);
                }
            }
        }
    }
    let mut id_of_root = vec![usize::MAX; runs.len()];
    let mut next = 0;
    (0..runs.len())
        .map(|r| {
            let root = find(&mut parent, r);
            if id_of_root[root] == usize::MAX {
                id_of_root[root] = next;
                next += 1;
            }
            id_of_root[root]
        })
        .collect()
}

/// A 3D closure compartment.
#[derive(Debug, Clone, PartialEq)]
pub struct Compartment {
    pub fluid: Fluid,
    pub voxels: usize,
    pub columns: usize,
    /// Number of `(unit, 2D region)` pieces it joins.
    pub pieces: usize,
    /// Number of sand units it spans (> 1: juxtaposition across a fault).
    pub units: usize,
    /// At least `min_closure_voxels` (smaller compartments stay brine).
    pub kept: bool,
    /// Keyed like an unsplit ef2dc42 closure (`false`: split-off piece).
    pub primary: bool,
}

/// Fluid of a split-off compartment, keyed by its first run.
pub fn split_compartment_fluid(seed: u64, layer: usize, rank: u64, col: usize) -> Fluid {
    let u = keyed_unit(&[seed, STREAM_FLUID, layer as u64, rank, 1 + col as u64]);
    Fluid::from_code(((u * 3.0) as u32).min(2))
}

/// Segmented closures per sand unit (default). The output has the same shape
/// as [`crate::rock_physics::sand_unit_fluids`]: `(label, fluids)` for every
/// member label of every closure unit. Also returns the compartments.
pub fn segmented_sand_unit_fluids(
    labels: &[u8],
    shape: [usize; 3],
    intervals: &[usize],
    sand: &[bool],
    seed: u64,
    max_column: f64,
    min_voxels: usize,
) -> (Vec<(usize, LayerFluids)>, Vec<Compartment>) {
    let [ni, nj, _] = shape;
    let n = ni * nj;
    // Units with member labels, shallowest first.
    let mut units: Vec<Vec<usize>> = Vec::new();
    let mut runs: Vec<ClosureRun> = Vec::new();
    let mut run_unit: Vec<usize> = Vec::new();
    for (top, end) in crate::lithology::closure_units(sand) {
        let mut members: Vec<(usize, usize)> = intervals
            .iter()
            .enumerate()
            .filter(|&(lab, &h)| lab < 255 && h >= top && h < end)
            .map(|(lab, &h)| (h, lab))
            .collect();
        if members.is_empty() {
            continue;
        }
        members.sort_unstable();
        let ids: Vec<u8> = members.iter().map(|&(_, lab)| lab as u8).collect();
        let u = units.len();
        for r in unit_closure_runs(labels, shape, &ids, top, max_column) {
            runs.push(r);
            run_unit.push(u);
        }
        units.push(members.iter().map(|&(_, lab)| lab).collect());
    }
    let comp = segment_runs(
        &runs.iter().map(|r| (r.col, r.k0, r.k1)).collect::<Vec<_>>(),
        ni,
        nj,
    );
    let nc = comp.iter().copied().max().map_or(0, |m| m + 1);
    // Primary run of every region: its first run (runs are in cell order
    // within a region).
    let mut key: Vec<Option<(usize, u64)>> = vec![None; nc];
    let mut first: Vec<Option<(usize, u64, usize)>> = vec![None; nc];
    let mut voxels = vec![0usize; nc];
    let mut columns = vec![0usize; nc];
    let mut piece_set: Vec<Vec<(usize, u64)>> = vec![Vec::new(); nc];
    let mut unit_set: Vec<Vec<usize>> = vec![Vec::new(); nc];
    let mut prev_region: Option<(usize, u64)> = None;
    for (r, run) in runs.iter().enumerate() {
        let c = comp[r];
        let region = (run.layer, run.rank);
        voxels[c] += run.k1 - run.k0;
        columns[c] += 1;
        if !piece_set[c].contains(&region) {
            piece_set[c].push(region);
        }
        if !unit_set[c].contains(&run_unit[r]) {
            unit_set[c].push(run_unit[r]);
        }
        if prev_region != Some(region) {
            // First run of this region.
            key[c] = Some(key[c].map_or(region, |k| k.min(region)));
            prev_region = Some(region);
        }
        let f = (run.layer, run.rank, run.col);
        first[c] = Some(first[c].map_or(f, |g| g.min(f)));
    }
    let compartments: Vec<Compartment> = (0..nc)
        .map(|c| {
            let fluid = match key[c] {
                Some((layer, rank)) => closure_fluid(seed, layer, rank),
                None => {
                    let (layer, rank, col) = first[c].unwrap();
                    split_compartment_fluid(seed, layer, rank, col)
                }
            };
            Compartment {
                fluid,
                voxels: voxels[c],
                columns: columns[c],
                pieces: piece_set[c].len(),
                units: unit_set[c].len(),
                kept: voxels[c] >= min_voxels.max(1),
                primary: key[c].is_some(),
            }
        })
        .collect();
    let mut per_unit: Vec<LayerFluids> = units.iter().map(|_| LayerFluids::empty(n)).collect();
    // Closure pieces per unit: (rank, compartment) -> (fluid, crest, contact, columns, voxels).
    let mut pieces: Vec<Vec<((u64, usize), (Fluid, f64, f64, usize, usize))>> =
        vec![Vec::new(); units.len()];
    for (r, run) in runs.iter().enumerate() {
        let c = comp[r];
        if !compartments[c].kept {
            continue;
        }
        let u = run_unit[r];
        let f = compartments[c].fluid;
        per_unit[u].contact[run.col] = run.contact as f32;
        per_unit[u].fluid[run.col] = f;
        let k = (run.rank, c);
        let list = &mut pieces[u];
        let entry = match list.iter().position(|(key, _)| *key == k) {
            Some(p) => &mut list[p].1,
            None => {
                list.push((k, (f, f64::INFINITY, f64::NEG_INFINITY, 0, 0)));
                &mut list.last_mut().unwrap().1
            }
        };
        entry.1 = entry.1.min(run.k0 as f64);
        entry.2 = entry.2.max(run.contact);
        entry.3 += 1;
        entry.4 += run.k1 - run.k0;
    }
    let mut out = Vec::new();
    for (u, members) in units.iter().enumerate() {
        per_unit[u].closures = pieces[u].iter().map(|(_, p)| *p).collect();
        for (k, &lab) in members.iter().enumerate() {
            let mut g = per_unit[u].clone();
            if k > 0 {
                g.closures.clear();
            }
            out.push((lab, g));
        }
    }
    (out, compartments)
}
