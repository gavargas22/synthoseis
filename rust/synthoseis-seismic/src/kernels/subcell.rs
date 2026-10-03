//! Sub-cell reflectivity for partial voxels (partial-voxels spec §1.4,
//! §1.5 mode S; PR A: kernel only, not wired into any pipeline path).
//!
//! A mixed depth cell is split into its pure sub-layers (fraction `f_j` of
//! the cell, end-member `P_j`). Every internal boundary gets its own
//! reflection at its exact ray (slowness) time:
//!
//! ```text
//! t_j     = T_k + 2·dz · Σ_{j' ≤ j} f_{j'} / Vp_{j'}      (ms, f64, in order)
//! T_{k+1} = T_k + 2·dz · Σ_j f_j / Vp_j
//! ```
//!
//! Cell-boundary interfaces (last sub-layer of k against the first of k+1)
//! sit at `T_{k+1}` as before. With one part per cell (`f = 1`) the times
//! and reflection coefficients are exactly those of
//! [`super::twt::twt_column`] and [`super::twt::reflectivity_time_column`],
//! bit for bit, so pure columns are unchanged. The interfaces go to the
//! existing [`super::twt::insert_spikes`] (windowed sinc).

use super::twt::{insert_spikes, TwtKernel};
use super::zoeppritz::{zoeppritz_pp_form, ZoeppritzForm};

/// One pure sub-layer of a depth cell: `frac` of the cell (`0 < frac ≤ 1`)
/// with isotropic properties (m/s, g/cc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubLayer {
    pub frac: f64,
    pub vp: f32,
    pub vs: f32,
    pub rho: f32,
}

/// The interfaces of one column split into sub-layers (see
/// [`subcell_column`]).
#[derive(Debug, Clone, Default)]
pub struct SubcellColumn {
    /// `T_0 … T_nz` (ms): cell-top times through the slowness sum (the same
    /// T feeds label point sampling).
    pub t_cells: Vec<f64>,
    /// Interface times (ms), non-decreasing: internal sub-layer boundaries
    /// and cell boundaries, in depth order.
    pub t_iface: Vec<f64>,
    /// Medium above / below each interface.
    pub upper: Vec<SubLayer>,
    pub lower: Vec<SubLayer>,
    /// Number of internal (sub-cell) interfaces among `t_iface`.
    pub internal: usize,
}

/// Split one column into sub-layers: `cells[k]` lists the pure parts of
/// depth cell k (shallow first, fractions summing to 1; a pure cell has one
/// part with `frac = 1`). `dz` is the cell size (m). Writes the cell times
/// and every interface with its exact two-way time into `out`.
pub fn subcell_column<'a, I>(cells: I, dz: f64, out: &mut SubcellColumn)
where
    I: IntoIterator<Item = &'a [SubLayer]>,
{
    out.t_cells.clear();
    out.t_iface.clear();
    out.upper.clear();
    out.lower.clear();
    out.internal = 0;
    let num = 2000.0 * dz;
    let mut t = 0.0f64;
    out.t_cells.push(0.0);
    let mut prev: Option<SubLayer> = None;
    for parts in cells {
        assert!(
            !parts.is_empty(),
            "subcell_column: a cell needs at least one part"
        );
        for (j, &p) in parts.iter().enumerate() {
            if let Some(u) = prev {
                // j == 0: cell boundary at T_k; otherwise an internal one.
                out.t_iface.push(t);
                out.upper.push(u);
                out.lower.push(p);
                out.internal += (j > 0) as usize;
            }
            // `num * f / v`: with f = 1 this is exactly twt_column's `num / v`.
            t += num * p.frac / p.vp as f64;
            prev = Some(p);
        }
        out.t_cells.push(t);
    }
}

/// Reflectivity of every interface of `col` at `angle_deg` (same Zoeppritz
/// kernel and form as the depth fuse) inserted into the time trace `x`
/// (overwritten; `t_n = n·dt_ms`). `r` is scratch.
#[allow(clippy::too_many_arguments)]
pub fn subcell_reflectivity(
    col: &SubcellColumn,
    angle_deg: f64,
    form: ZoeppritzForm,
    dt_ms: f64,
    kernel: TwtKernel,
    r: &mut Vec<f32>,
    x: &mut [f64],
) {
    r.clear();
    r.extend(col.upper.iter().zip(&col.lower).map(|(a, b)| {
        zoeppritz_pp_form(
            a.vp as f64,
            a.vs as f64,
            a.rho as f64,
            b.vp as f64,
            b.vs as f64,
            b.rho as f64,
            angle_deg,
            form,
        )
    }));
    x.iter_mut().for_each(|v| *v = 0.0);
    insert_spikes(r, &col.t_iface, dt_ms, kernel, x);
}

#[cfg(test)]
mod tests {
    use super::super::twt::{reflectivity_time_column, twt_column, TwtScratch};
    use super::*;

    fn layer(frac: f64, vp: f32, vs: f32, rho: f32) -> SubLayer {
        SubLayer { frac, vp, vs, rho }
    }

    /// Pure cells reproduce the whole-voxel time chain bit for bit.
    #[test]
    fn pure_column_matches_reflectivity_time_column() {
        let nz = 40;
        let vp: Vec<f32> = (0..nz)
            .map(|k| 1500.0 + 37.0 * k as f32 + if k > 20 { 900.0 } else { 0.0 })
            .collect();
        let vs: Vec<f32> = vp.iter().map(|v| v * 0.5).collect();
        let rho: Vec<f32> = (0..nz).map(|k| 1.9 + 0.01 * k as f32).collect();
        let cells: Vec<Vec<SubLayer>> = (0..nz)
            .map(|k| vec![layer(1.0, vp[k], vs[k], rho[k])])
            .collect();
        let mut col = SubcellColumn::default();
        subcell_column(cells.iter().map(|c| c.as_slice()), 4.0, &mut col);
        let mut t = vec![0.0; nz + 1];
        twt_column(&vp, 4.0, &mut t);
        assert_eq!(col.t_cells, t);
        assert_eq!(col.internal, 0);
        for (dt, kernel) in [
            (4.0, TwtKernel::Sinc),
            (1.0, TwtKernel::Sinc),
            (2.0, TwtKernel::Linear),
        ] {
            let nt = (t[nz] / dt) as usize + 20;
            let (mut a, mut b) = (vec![0.0; nt], vec![0.0; nt]);
            let mut s = TwtScratch::default();
            reflectivity_time_column(
                &vp,
                &vs,
                &rho,
                4.0,
                15.0,
                ZoeppritzForm::Exact,
                dt,
                kernel,
                &mut s,
                &mut a,
            );
            subcell_reflectivity(
                &col,
                15.0,
                ZoeppritzForm::Exact,
                dt,
                kernel,
                &mut Vec::new(),
                &mut b,
            );
            assert!(
                a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                "dt {dt}"
            );
        }
    }

    /// Internal boundaries sit at the exact slowness time.
    #[test]
    fn internal_boundary_times() {
        let sh = layer(1.0, 2580.0, 1139.0, 2.277);
        let gas = layer(1.0, 2472.38, 1490.51, 1.841);
        let mixed = [SubLayer { frac: 0.3, ..sh }, SubLayer { frac: 0.7, ..gas }];
        let cells: Vec<&[SubLayer]> = vec![
            std::slice::from_ref(&sh),
            &mixed,
            std::slice::from_ref(&gas),
        ];
        let mut col = SubcellColumn::default();
        subcell_column(cells, 4.0, &mut col);
        let t1 = 8000.0 / 2580.0;
        let t_int = t1 + 8000.0 * 0.3 / 2580.0;
        let t2 = t_int + 8000.0 * 0.7 / 2472.38f32 as f64;
        assert_eq!(col.t_iface.len(), 3);
        assert_eq!(col.internal, 1);
        assert!((col.t_iface[1] - t_int).abs() < 1e-12);
        assert!((col.t_cells[2] - t2).abs() < 1e-12);
        // Cell boundaries 0|1 (shale|shale, r = 0) and 1|2 (gas|gas, r = 0).
        let mut r = Vec::new();
        let mut x = vec![0.0; 16];
        subcell_reflectivity(
            &col,
            0.0,
            ZoeppritzForm::Exact,
            1.0,
            TwtKernel::Sinc,
            &mut r,
            &mut x,
        );
        assert_eq!((r[0], r[2]), (0.0, 0.0));
        assert!(r[1] < 0.0);
    }
}
