//! Partial voxels: exact 1D vertical overlap fractions per depth cell
//! (partial-voxels spec §1.1–§1.2, §1.6, §3.1–§3.2, §3.7).
//!
//! PR A ships the kernel only: nothing here is wired into the pipeline, and
//! no output changes. Fractions are recomputed per column from the
//! continuous (pre-rounding) geometry, so nothing new is stored per voxel.
//!
//! # Coordinates
//!
//! Depth index ζ in samples; cell `k` is `[k, k+1)` (the depth-to-time
//! convention, centre `k + ½`). Boundaries:
//! * horizon / seabed `h`: the unrounded map value `z_h`
//!   ([`crate::toy_geometry::layered_horizon_maps_continuous`],
//!   [`crate::pipeline_stream::toy_horizon_maps_continuous`]);
//! * salt: `[lo + ½, hi + ½]` from the continuous hull bounds
//!   ([`crate::salt::hull_bounds`]; the hull test samples integer k, the
//!   centre of cell k);
//! * fluid contact of interval h: `contact` (HC where ζ < contact), as in
//!   `RpmModel::column`'s `(k as f32) < contact`.
//!
//! Rounding these positions reproduces today's whole-voxel labels (centre
//! rule, §1.6), so labels are never recomputed.
//!
//! # Breakpoint merge (§3.2)
//!
//! All boundaries of a column are sorted once and merged with the cell edges
//! by a moving pointer (O(nz + nh log nh)). Each sub-interval is classified
//! at its midpoint with the priority salt > water (ζ < z_0) > interval h
//! (the deepest h with z_h ≤ ζ, HC if ζ < contact_h) > below the deepest
//! horizon, and equal neighbours merge. Fractions are summed in depth order
//! in f64 relative to the cell top.
//!
//! # `EPS_FRAC`
//!
//! A boundary within [`EPS_FRAC`] of a cell edge snaps to it, and a boundary
//! within `EPS_FRAC` of the previous kept breakpoint is dropped, so no part
//! has `f < EPS_FRAC`, fractions of a cell sum to 1 and integer boundaries
//! give pure cells.

/// Parts with `f < EPS_FRAC` are not formed (spec §2): boundaries within
/// `EPS_FRAC` cells of a cell edge or of the previous breakpoint are merged.
pub const EPS_FRAC: f64 = 1e-6;

/// The unit a part of a cell belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartKind {
    /// Above the seabed `z_0`.
    Water,
    /// Inside the salt hull.
    Salt,
    /// Horizon interval `h` (`z_h ≤ ζ < z_{h+1}`); `hc` when above the
    /// interval's fluid contact.
    Interval { h: usize, hc: bool },
    /// Below the deepest horizon (whole-voxel `Unfilled`).
    Below,
}

impl PartKind {
    /// The labelled unit of this kind (the fluid sub-kind is not a unit).
    pub fn unit(self) -> Unit {
        match self {
            PartKind::Water => Unit::Water,
            PartKind::Salt => Unit::Salt,
            PartKind::Interval { h, .. } => Unit::Interval(h),
            PartKind::Below => Unit::Below,
        }
    }
}

/// A labelled unit: what the whole-voxel (centre-rule) label of a cell says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    Water,
    Salt,
    Interval(usize),
    Below,
}

/// One pure part of a cell: `frac` of the cell (`EPS_FRAC ≤ frac ≤ 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Part {
    pub kind: PartKind,
    pub frac: f64,
}

/// Continuous geometry of one column.
#[derive(Debug, Clone, Copy)]
pub struct ColumnGeometry<'a> {
    /// Horizon depths `z_0 … z_{nh-1}` (samples, unrounded; `z_0` is the
    /// seabed). Non-decreasing from horizon 1 on, as the maps are.
    pub horizons: &'a [f64],
    /// Continuous hull bounds `(lo, hi)` of the column ([`crate::salt::hull_bounds`]);
    /// salt occupies `[lo + ½, hi + ½]`.
    pub salt: Option<(f64, f64)>,
    /// Fluid contact per interval `h` (`nh - 1` entries, or empty for none;
    /// `-inf` where the interval holds no hydrocarbon).
    pub contacts: &'a [f32],
}

impl ColumnGeometry<'_> {
    /// Kind of the material at depth `zeta` (spec §3.2 priority). `h` is a
    /// moving pointer for increasing `zeta` (start at 0).
    fn classify(&self, zeta: f64, h: &mut usize) -> PartKind {
        if let Some((lo, hi)) = self.salt {
            if lo + 0.5 <= zeta && zeta <= hi + 0.5 {
                return PartKind::Salt;
            }
        }
        let z = self.horizons;
        if z.is_empty() || zeta < z[0] {
            return PartKind::Water;
        }
        while *h + 1 < z.len() && z[*h + 1] <= zeta {
            *h += 1;
        }
        if *h + 1 >= z.len() {
            return PartKind::Below;
        }
        let hc = self.contacts.get(*h).is_some_and(|&c| zeta < c as f64);
        PartKind::Interval { h: *h, hc }
    }
}

/// The parts of every cell of one column (see [`column_parts`]). Transient,
/// per column: nothing is stored per voxel.
#[derive(Debug, Clone, Default)]
pub struct ColumnParts {
    pub parts: Vec<Part>,
    /// `offsets[k]..offsets[k + 1]` indexes the parts of cell k (`nz + 1`).
    pub offsets: Vec<usize>,
    events: Vec<f64>,
}

impl ColumnParts {
    /// Number of cells.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Parts of cell `k`, shallow first.
    pub fn cell(&self, k: usize) -> &[Part] {
        &self.parts[self.offsets[k]..self.offsets[k + 1]]
    }

    /// `true` when cell `k` holds more than one part.
    pub fn is_mixed(&self, k: usize) -> bool {
        self.offsets[k + 1] - self.offsets[k] > 1
    }

    /// Hydrocarbon fraction of cell `k` (closure hook, spec §3.7).
    pub fn hc_fraction(&self, k: usize) -> f64 {
        hc_fraction(self.cell(k))
    }
}

/// Hydrocarbon fraction of one cell's parts (closure hook, spec §3.7).
pub fn hc_fraction(parts: &[Part]) -> f64 {
    parts
        .iter()
        .filter(|p| matches!(p.kind, PartKind::Interval { hc: true, .. }))
        .map(|p| p.frac)
        .sum()
}

/// Snap `x` to the nearest integer when within [`EPS_FRAC`].
#[inline]
fn snap(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() < EPS_FRAC {
        r
    } else {
        x
    }
}

/// Exact vertical overlap fractions of every cell `0..nz` of one column
/// (spec §3.2), written into `out`.
pub fn column_parts(geom: &ColumnGeometry, nz: usize, out: &mut ColumnParts) {
    column_parts_range(geom, 0, nz, out);
}

/// [`column_parts`] for the cells `k0..k1` only (`out.cell(k - k0)`). Each
/// cell depends on the column geometry alone, so any k-split gives the same
/// bits as the whole column (chunk-edge invariance, spec §5.6).
pub fn column_parts_range(geom: &ColumnGeometry, k0: usize, k1: usize, out: &mut ColumnParts) {
    out.parts.clear();
    out.offsets.clear();
    out.offsets.push(0);
    let top = k0 as f64;
    let bot = k1 as f64;
    let ev = &mut out.events;
    ev.clear();
    let mut push = |x: f64| {
        if x.is_finite() {
            let x = snap(x);
            if top < x && x < bot && x.fract() != 0.0 {
                ev.push(x);
            }
        }
    };
    geom.horizons.iter().for_each(|&z| push(z));
    if let Some((lo, hi)) = geom.salt {
        if lo <= hi {
            push(lo + 0.5);
            push(hi + 0.5);
        }
    }
    // Contacts only matter inside their interval; outside they would split
    // a cell into equal kinds, which the merge undoes anyway.
    for (h, &c) in geom.contacts.iter().enumerate() {
        let c = c as f64;
        let z = geom.horizons;
        if h + 1 < z.len() && z[h] < c && c < z[h + 1] {
            push(c);
        }
    }
    ev.sort_by(f64::total_cmp);
    let mut e = 0usize;
    let mut hp = 0usize;
    for k in k0..k1 {
        let kf = k as f64;
        let start = out.parts.len();
        // Breakpoints inside (k, k+1), relative to the cell top.
        let mut a = 0.0f64;
        while e < ev.len() && ev[e] < kf + 1.0 {
            let b = ev[e] - kf;
            e += 1;
            if b - a < EPS_FRAC || 1.0 - b < EPS_FRAC {
                continue;
            }
            push_part(
                &mut out.parts,
                start,
                geom.classify(kf + 0.5 * (a + b), &mut hp),
                b - a,
            );
            a = b;
        }
        push_part(
            &mut out.parts,
            start,
            geom.classify(kf + 0.5 * (a + 1.0), &mut hp),
            1.0 - a,
        );
        out.offsets.push(out.parts.len());
    }
}

/// Append a part to the current cell (from `start`), merging with an equal
/// previous kind.
#[inline]
fn push_part(parts: &mut Vec<Part>, start: usize, kind: PartKind, frac: f64) {
    if parts.len() > start {
        let last = parts.last_mut().unwrap();
        if last.kind == kind {
            last.frac += frac;
            return;
        }
    }
    parts.push(Part { kind, frac });
}

/// How a cell is modelled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellModel<'a> {
    /// The existing whole-voxel path (pure cells, and the label-support
    /// fallback): bit-identical to today.
    Whole,
    /// Mix these parts (Backus in C mode, sub-layers in S mode).
    Mixed(&'a [Part]),
}

/// Per-run partial-voxel counters (spec §3.6).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PartialVoxelStats {
    /// Cells seen.
    pub cells: u64,
    /// Cells modelled as mixed.
    pub mixed: u64,
    /// Mixed cells with three or more parts.
    pub multi: u64,
    /// Cells whose labelled unit has zero fraction (whole-voxel fallback).
    pub label_guard: u64,
}

impl PartialVoxelStats {
    pub fn add(&mut self, o: &PartialVoxelStats) {
        self.cells += o.cells;
        self.mixed += o.mixed;
        self.multi += o.multi;
        self.label_guard += o.label_guard;
    }
}

/// Decide how a cell with `parts` and whole-voxel label unit `label` is
/// modelled (spec §1.6, §3.4): a pure cell short-circuits to the
/// whole-voxel path; a mixed cell whose labelled unit has no part (f = 0)
/// falls back to the whole voxel and is counted in `stats.label_guard`.
pub fn cell_model<'a>(
    parts: &'a [Part],
    label: Unit,
    stats: &mut PartialVoxelStats,
) -> CellModel<'a> {
    stats.cells += 1;
    if parts.len() <= 1 {
        return CellModel::Whole;
    }
    if !parts.iter().any(|p| p.kind.unit() == label) {
        stats.label_guard += 1;
        return CellModel::Whole;
    }
    stats.mixed += 1;
    stats.multi += (parts.len() >= 3) as u64;
    CellModel::Mixed(parts)
}

/// Total fraction of `unit` over a column (its thickness inside `[0, nz)`).
pub fn unit_thickness(parts: &ColumnParts, unit: Unit) -> f64 {
    parts
        .parts
        .iter()
        .filter(|p| p.kind.unit() == unit)
        .map(|p| p.frac)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts_of(z: &[f64], salt: Option<(f64, f64)>, contacts: &[f32], nz: usize) -> ColumnParts {
        let mut out = ColumnParts::default();
        column_parts(
            &ColumnGeometry {
                horizons: z,
                salt,
                contacts,
            },
            nz,
            &mut out,
        );
        out
    }

    #[test]
    fn single_boundary_is_clipped_linear() {
        let z = [3.3, 6.0, 20.0];
        let p = parts_of(&z, None, &[], 10);
        assert_eq!(
            p.cell(2),
            &[Part {
                kind: PartKind::Water,
                frac: 1.0
            }]
        );
        let c3 = p.cell(3);
        assert_eq!(c3.len(), 2);
        assert_eq!(c3[0].kind, PartKind::Water);
        assert!((c3[0].frac - 0.3).abs() < 1e-12);
        assert_eq!(c3[1].kind, PartKind::Interval { h: 0, hc: false });
        // Integer horizon: pure cells on both sides.
        assert_eq!(p.cell(5).len(), 1);
        assert_eq!(
            p.cell(6),
            &[Part {
                kind: PartKind::Interval { h: 1, hc: false },
                frac: 1.0
            }]
        );
    }

    #[test]
    fn eps_snapping_gives_pure_cells() {
        let z = [3.0 + 0.4 * EPS_FRAC, 6.0 - 0.4 * EPS_FRAC, 20.0];
        let p = parts_of(&z, None, &[], 10);
        assert!((0..10).all(|k| !p.is_mixed(k)));
        assert_eq!(p.cell(3)[0].kind, PartKind::Interval { h: 0, hc: false });
        assert_eq!(p.cell(6)[0].kind, PartKind::Interval { h: 1, hc: false });
        // Two boundaries closer than EPS_FRAC: no sliver part.
        let z = [3.5, 3.5 + 0.3 * EPS_FRAC, 20.0];
        let p = parts_of(&z, None, &[], 10);
        assert!(p.parts.iter().all(|q| q.frac >= EPS_FRAC));
        assert_eq!(p.cell(3).len(), 2);
    }

    #[test]
    fn salt_contact_priority_and_hc() {
        // Salt [lo+½, hi+½] = [4.7, 7.2]; contact of interval 0 at 3.4.
        let z = [2.25, 9.0, 30.0];
        let p = parts_of(&z, Some((4.2, 6.7)), &[3.4, f32::NEG_INFINITY], 12);
        let c2 = p.cell(2);
        assert_eq!(c2.len(), 2);
        assert_eq!(c2[1].kind, PartKind::Interval { h: 0, hc: true });
        let c3 = p.cell(3);
        assert_eq!(c3.len(), 2);
        assert!((c3[0].frac - 0.4f32 as f64 - 0.0).abs() < 1e-6);
        assert_eq!(c3[1].kind, PartKind::Interval { h: 0, hc: false });
        assert!((p.hc_fraction(3) - c3[0].frac).abs() < 1e-15);
        assert_eq!(p.cell(4).last().unwrap().kind, PartKind::Salt);
        assert_eq!(
            p.cell(6),
            &[Part {
                kind: PartKind::Salt,
                frac: 1.0
            }]
        );
        let c7 = p.cell(7);
        assert_eq!(c7[0].kind, PartKind::Salt);
        assert!((c7[0].frac - 0.2).abs() < 1e-12);
        let salt = unit_thickness(&p, Unit::Salt);
        assert!((salt - 2.5).abs() < 1e-12);
    }

    #[test]
    fn guard_counts_unsupported_labels() {
        let parts = [
            Part {
                kind: PartKind::Water,
                frac: 0.4,
            },
            Part {
                kind: PartKind::Interval { h: 0, hc: false },
                frac: 0.6,
            },
        ];
        let mut s = PartialVoxelStats::default();
        assert_eq!(
            cell_model(&parts, Unit::Interval(0), &mut s),
            CellModel::Mixed(&parts)
        );
        assert_eq!(
            cell_model(&parts, Unit::Interval(1), &mut s),
            CellModel::Whole
        );
        assert_eq!(
            cell_model(&parts[..1], Unit::Water, &mut s),
            CellModel::Whole
        );
        assert_eq!(
            s,
            PartialVoxelStats {
                cells: 3,
                mixed: 1,
                multi: 0,
                label_guard: 1
            }
        );
    }

    /// `round(continuous) == production` for the layered maps and the
    /// salt-dragged toy maps (spec §3.1): the refactors cannot move labels.
    #[test]
    fn rounded_continuous_maps_equal_production() {
        use crate::pipeline::E2eConfig;
        for seed in [7u64, 1, 2, 3, 30, 11] {
            for shape in [[64usize, 64, 256], [24, 40, 96]] {
                let (prod, nh) = crate::toy_geometry::layered_horizon_maps(seed, shape);
                let (cont, nh2) = crate::toy_geometry::layered_horizon_maps_continuous(seed, shape);
                assert_eq!(nh, nh2);
                assert!(prod
                    .iter()
                    .zip(&cont)
                    .all(|(p, c)| p.to_bits() == c.round().to_bits()));
                assert!(cont.iter().any(|c| c.fract() != 0.0));
                for salt in [false, true] {
                    let mut cfg = E2eConfig {
                        seed,
                        inline_count: shape[0],
                        crossline_count: shape[1],
                        samples: shape[2],
                        geometry: crate::toy_geometry::ToyGeometry::Layered,
                        ..E2eConfig::default()
                    };
                    cfg.rock_physics.salt = salt;
                    let (prod, nh) = crate::pipeline_stream::toy_horizon_maps(&cfg);
                    let (cont, nh2) =
                        crate::pipeline_stream::toy_horizon_maps_continuous(&cfg).unwrap();
                    assert_eq!(nh, nh2);
                    let bad = prod
                        .iter()
                        .zip(&cont)
                        .filter(|(p, c)| p.to_bits() != c.round().to_bits())
                        .count();
                    assert_eq!(bad, 0, "seed {seed} shape {shape:?} salt {salt}");
                }
            }
        }
    }
}
