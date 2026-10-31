//! Tile-wise fault application: displacement lookup, fault labels, and the
//! trace/label remapping helpers.
//!
//! Port of the displacement/label bookkeeping in `Faults.build_faults` and
//! `Faults.apply_xyz_displacement`. For each *active* fault `n` (in order):
//!
//! ```text
//! d_n(v)   = inside_n(v) ? xy_n(i,j) * z_shift_n(k) : 0
//! adj_n(k) = clip(k - d_n(k), 0, nk-1)
//! F        = f32(interp(adj_n, arange(nk), F))           # _faulted_depths
//! L        = f32(interp(adj_n, arange(nk), L))           # fault_planes
//!            L ∈ (0, .25) → 0 ; L ∈ (.25, 1) → 1
//! L       += seg_n(v) > .25 && xy_n*z_shift_n > 1        # fault_segm
//! ```
//!
//! and finally `mask = L > 0.05`, `faulted(v) = interp(F(v), arange, trace)`.
//! Every step only touches one trace plus an analytic halo, so tiles are
//! independent and results do not depend on tiling or worker count.

use super::kinematics::FaultKinematics;
use super::model::{FaultSkip, ReachMode, ResolvedFault, Seabed};
use super::params::{FaultParams, FaultRng};
use super::segments::{segment_block, SEG_ZERO};

/// A set of resolved faults over one model cube.
#[derive(Debug, Clone)]
pub struct FaultModel {
    shape: [usize; 3],
    faults: Vec<ResolvedFault>,
    skipped: Vec<(usize, FaultSkip)>,
    mode: ReachMode,
    /// Seabed for the `FitColumn` mask clamp (`None` = no clamp).
    clamp: Option<Seabed>,
    /// Legacy vertical lookup or heave (spec-fault-kinematics 7a).
    kinematics: FaultKinematics,
    /// Per-tile gather results shared by every consumer of a heave run
    /// ([`Self::with_gather_cache`]); `None` = gather on demand.
    cache: Option<std::sync::Arc<GatherCache>>,
}

/// Gather results of one tile `[i0,i1) × [j0,j1) × [0,nk)` (row-major).
#[derive(Debug)]
pub struct GatherTile {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub nk: usize,
    /// Continuous source `(i, j, k)` of every voxel ([`super::kinematics::Gathered::src`]).
    pub src: Vec<[f64; 3]>,
    /// Fault id of every voxel ([`super::kinematics::Gathered::id`]).
    pub id: Vec<u8>,
}

impl GatherTile {
    #[inline]
    fn offset(&self, i: usize, j: usize) -> usize {
        debug_assert!(i >= self.i0 && i < self.i1 && j >= self.j0 && j < self.j1);
        ((i - self.i0) * (self.j1 - self.j0) + (j - self.j0)) * self.nk
    }
}

/// Lazily filled grid of [`GatherTile`]s: each tile is gathered once per
/// run, by whichever consumer asks first, and reused by the others.
#[derive(Debug)]
struct GatherCache {
    grid: [usize; 2],
    ntj: usize,
    tiles: Vec<std::sync::OnceLock<std::sync::Arc<GatherTile>>>,
}

/// Default tile used for the global survey pass (does not affect results).
pub const SURVEY_TILE: [usize; 2] = [16, 16];

impl FaultModel {
    /// Resolve every fault (global, once): centres, seabed taper, profiles.
    ///
    /// Faults Python would skip ("no fault inserted") are recorded in
    /// [`Self::skipped`]. `seed` only matters for faults without an explicit
    /// `center` (seeded port of `get_middle_z`'s `rng.choice`).
    ///
    /// Uses the default [`ReachMode::FitColumn`], which is bit-identical to
    /// [`ReachMode::Legacy`] whenever every legacy seabed taper succeeds (always
    /// the case once the sub-seabed column is >= ~582 samples).
    pub fn resolve(shape: [usize; 3], params: &[FaultParams], seabed: &Seabed, seed: u64) -> Self {
        Self::resolve_with_mode(shape, params, seabed, seed, ReachMode::default())
    }

    /// [`Self::resolve`] with an explicit [`ReachMode`]. Fault parameters,
    /// centres and lateral profiles do not depend on `mode`.
    pub fn resolve_with_mode(
        shape: [usize; 3],
        params: &[FaultParams],
        seabed: &Seabed,
        seed: u64,
        mode: ReachMode,
    ) -> Self {
        let root = FaultRng::new(seed ^ 0xCE47_3E00);
        let mut faults = Vec::new();
        let mut skipped = Vec::new();
        for (n, p) in params.iter().enumerate() {
            let mut rng = root.fork(n as u64 + 1);
            match super::model::resolve_fault_mode(shape, p, seabed, &mut rng, SURVEY_TILE, mode) {
                Ok(f) => faults.push(f),
                Err(why) => skipped.push((n, why)),
            }
        }
        let clamp = (mode == ReachMode::FitColumn).then(|| seabed.clone());
        Self {
            shape,
            faults,
            skipped,
            mode,
            clamp,
            kinematics: FaultKinematics::Legacy,
            cache: None,
        }
    }

    /// A model with no faults (identity lookup, empty mask).
    pub fn empty(shape: [usize; 3]) -> Self {
        Self {
            shape,
            faults: Vec::new(),
            skipped: Vec::new(),
            mode: ReachMode::default(),
            clamp: None,
            kinematics: FaultKinematics::Legacy,
            cache: None,
        }
    }

    /// The same model with other kinematics (no RNG draws).
    pub fn with_kinematics(mut self, kinematics: FaultKinematics) -> Self {
        self.kinematics = kinematics;
        self.cache = None;
        self
    }

    /// The same model with a shared per-tile gather cache on the `grid`
    /// tiling (clones share it). Results are identical with and without
    /// the cache; it only removes repeated gathers of a heave run.
    pub fn with_gather_cache(mut self, grid: [usize; 2]) -> Self {
        let [ni, nj, _] = self.shape;
        let grid = [grid[0].max(1), grid[1].max(1)];
        let (nti, ntj) = (ni.div_ceil(grid[0]), nj.div_ceil(grid[1]));
        self.cache = Some(std::sync::Arc::new(GatherCache {
            grid,
            ntj,
            tiles: (0..nti * ntj).map(|_| std::sync::OnceLock::new()).collect(),
        }));
        self
    }

    /// Lateral halo (bins) that holds every heave source of a tile:
    /// `ceil(Σ_f t_max,f / tan δ_min,v) + 2`, the spec §11 halo with the
    /// largest throws of all faults summed, because a voxel's gather path
    /// can cross several hanging walls (19 bins for one fault at the
    /// defaults; 0 without heave). Each stage moves at most
    /// `t_f / tan δ_min,v` bins (lateral weight ≤ 1).
    pub fn heave_halo(&self) -> usize {
        match self.kinematics {
            FaultKinematics::Heave(p) if !self.faults.is_empty() => {
                let t: f64 = self
                    .faults
                    .iter()
                    .map(|f| f.profile.iter().map(|v| v.abs()).fold(0.0f64, f64::max))
                    .sum();
                p.halo(t)
            }
            _ => 0,
        }
    }

    /// Gather every voxel of `[i0,i1) × [j0,j1)` (no cache).
    fn gather_block(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> GatherTile {
        let nk = self.shape[2];
        let n = (i1 - i0) * (j1 - j0) * nk;
        let mut src = Vec::with_capacity(n);
        let mut id = Vec::with_capacity(n);
        for i in i0..i1 {
            for j in j0..j1 {
                for k in 0..nk {
                    let g = self.gather(i, j, k);
                    src.push(g.src);
                    id.push(g.id);
                }
            }
        }
        GatherTile {
            i0,
            i1,
            j0,
            j1,
            nk,
            src,
            id,
        }
    }

    /// Gather results of the cache tile holding column `(i, j)`; `None`
    /// without a cache.
    fn cached_tile(&self, i: usize, j: usize) -> Option<std::sync::Arc<GatherTile>> {
        let c = self.cache.as_ref()?;
        let [ni, nj, _] = self.shape;
        let (ti, tj) = (i / c.grid[0], j / c.grid[1]);
        let t = c.tiles[ti * c.ntj + tj].get_or_init(|| {
            let (i0, j0) = (ti * c.grid[0], tj * c.grid[1]);
            std::sync::Arc::new(self.gather_block(
                i0,
                (i0 + c.grid[0]).min(ni),
                j0,
                (j0 + c.grid[1]).min(nj),
            ))
        });
        Some(t.clone())
    }

    /// Continuous source `(i, j)` of every sample of column `(i, j)`
    /// (from the gather cache when the model has one).
    pub fn column_sources(&self, i: usize, j: usize, out: &mut Vec<(f64, f64)>) {
        out.clear();
        match self.cached_tile(i, j) {
            Some(t) => {
                let o = t.offset(i, j);
                out.extend(t.src[o..o + t.nk].iter().map(|s| (s[0], s[1])));
            }
            None => out.extend((0..self.shape[2]).map(|k| {
                let g = self.gather(i, j, k);
                (g.src[0], g.src[1])
            })),
        }
    }

    /// Kinematics the tiles are computed with.
    pub fn kinematics(&self) -> FaultKinematics {
        self.kinematics
    }

    /// Whether the model moves rock sideways (`FaultTile::src_ij` is set).
    pub fn has_heave(&self) -> bool {
        matches!(self.kinematics, FaultKinematics::Heave(_)) && !self.faults.is_empty()
    }

    /// Reach mode the model was resolved with.
    pub fn mode(&self) -> ReachMode {
        self.mode
    }

    /// Number of faults whose sigma [`ReachMode::FitColumn`] had to shrink.
    pub fn reach_rescued(&self) -> usize {
        self.faults.iter().filter(|f| f.reach_rescued).count()
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Mutable faults (tests: swap in an analytic geometry).
    #[cfg(test)]
    pub(crate) fn faults_mut(&mut self) -> &mut Vec<ResolvedFault> {
        self.cache = None;
        &mut self.faults
    }

    /// Active (inserted) faults in application order.
    pub fn faults(&self) -> &[ResolvedFault] {
        &self.faults
    }

    /// `(input index, reason)` for faults that were not inserted.
    pub fn skipped(&self) -> &[(usize, FaultSkip)] {
        &self.skipped
    }

    pub fn is_empty(&self) -> bool {
        self.faults.is_empty()
    }

    /// Evaluate one spatial tile `[i0,i1) × [j0,j1) × [0,nk)`.
    pub fn compute_tile(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> FaultTile {
        let [ni, nj, nk] = self.shape;
        assert!(
            i0 <= i1 && i1 <= ni && j0 <= j1 && j1 <= nj,
            "tile out of range"
        );
        if self.has_heave() {
            return self.compute_tile_heave(i0, i1, j0, j1);
        }
        let (ti, tj) = (i1 - i0, j1 - j0);
        let n = ti * tj * nk;
        let mut lookup = vec![0.0f32; n];
        for col in 0..ti * tj {
            for k in 0..nk {
                lookup[col * nk + k] = k as f32;
            }
        }
        let mut level = vec![0.0f32; n];
        let mut segment_id = vec![0u8; n];
        let mut seg = Vec::new();
        let mut adj = vec![0.0f64; nk];
        let mut tmp = vec![0.0f32; nk];
        let mut tmp_id = vec![0u8; nk];
        let top = (nk.max(1) - 1) as f64;

        for (fi, f) in self.faults.iter().enumerate() {
            let fid = (fi + 1).min(254) as u8;
            segment_block(&f.geometry, self.shape, i0, i1, j0, j1, &mut seg);
            for di in 0..ti {
                let i = i0 + di;
                for dj in 0..tj {
                    let j = j0 + dj;
                    let base = (di * tj + dj) * nk;
                    let xy = f.lateral(i, j);
                    for (k, a) in adj.iter_mut().enumerate() {
                        let d = if f.geometry.inside(i, j, k) {
                            xy * f.profile[k]
                        } else {
                            0.0
                        };
                        *a = (k as f64 - d).clamp(0.0, top);
                    }
                    // _faulted_depths
                    let col = &mut lookup[base..base + nk];
                    interp_trace_into(&adj, col, &mut tmp);
                    col.copy_from_slice(&tmp);
                    // fault_planes: displace previous labels, re-binarise, add.
                    let lv = &mut level[base..base + nk];
                    interp_trace_into(&adj, lv, &mut tmp);
                    let ids = &mut segment_id[base..base + nk];
                    carry_ids(&adj, ids, &mut tmp_id);
                    for k in 0..nk {
                        let mut v = tmp[k];
                        if v > 0.0 && v < 0.25 {
                            v = 0.0;
                        } else if v > 0.25 && v < 1.0 {
                            v = 1.0;
                        }
                        let mut id = if v > 0.0 { tmp_id[k] } else { 0 };
                        let on = seg[base + k] != SEG_ZERO && xy * f.profile[k] > 1.0;
                        if on {
                            v = (v as f64 + 1.0) as f32;
                            id = fid;
                        }
                        lv[k] = v;
                        ids[k] = id;
                    }
                }
            }
        }

        let mut mask: Vec<u8> = level.iter().map(|&v| u8::from(v > 0.05)).collect();
        // FitColumn: no fault labels above the seabed. A no-op whenever every
        // taper succeeded against a flat seabed (profiles <= 1 there and
        // content only moves down).
        if let Some(sb) = &self.clamp {
            for di in 0..ti {
                for dj in 0..tj {
                    let wb = sb.at(i0 + di, j0 + dj, nj);
                    let base = (di * tj + dj) * nk;
                    for (k, m) in mask[base..base + nk].iter_mut().enumerate() {
                        if (k as f64) < wb {
                            *m = 0;
                        }
                    }
                }
            }
        }
        for (id, &m) in segment_id.iter_mut().zip(mask.iter()) {
            if m == 0 {
                *id = 0;
            }
        }
        FaultTile {
            i0,
            i1,
            j0,
            j1,
            nk,
            lookup,
            mask,
            segment_id,
            src_ij: None,
        }
    }

    /// [`Self::compute_tile`] with heave: every voxel gathers its 3-D
    /// source (see [`super::kinematics`]); `src_ij` holds the source
    /// `(i, j)` and `lookup` the source `k`.
    fn compute_tile_heave(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> FaultTile {
        let [_, nj, nk] = self.shape;
        let (ti, tj) = (i1 - i0, j1 - j0);
        let n = ti * tj * nk;
        let mut lookup = vec![0.0f32; n];
        let mut src_ij = vec![(0.0f32, 0.0f32); n];
        let mut mask = vec![0u8; n];
        let mut segment_id = vec![0u8; n];
        let own = if self.cache.is_none() {
            Some(std::sync::Arc::new(self.gather_block(i0, i1, j0, j1)))
        } else {
            None
        };
        for di in 0..ti {
            for dj in 0..tj {
                let (i, j) = (i0 + di, j0 + dj);
                let g = match &own {
                    Some(t) => t.clone(),
                    None => self.cached_tile(i, j).expect("cache"),
                };
                let o = g.offset(i, j);
                let base = (di * tj + dj) * nk;
                let wb = self.clamp.as_ref().map(|sb| sb.at(i, j, nj));
                for k in 0..nk {
                    let (s, id) = (g.src[o + k], g.id[o + k]);
                    lookup[base + k] = s[2] as f32;
                    src_ij[base + k] = (s[0] as f32, s[1] as f32);
                    let on = id != 0 && !wb.is_some_and(|w| (k as f64) < w);
                    if on {
                        mask[base + k] = 1;
                        segment_id[base + k] = id;
                    }
                }
            }
        }
        FaultTile {
            i0,
            i1,
            j0,
            j1,
            nk,
            lookup,
            mask,
            segment_id,
            src_ij: Some(src_ij),
        }
    }

    /// Run `f` on every tile of a heave model with a window of the
    /// *unfaulted* `data` holding the tile plus its [`Self::heave_halo`]
    /// (and writes the tile in place). Tiles go strip by strip in `i`; the
    /// window keeps the original rows `[i0 − H, i1 + H)` of the current
    /// strip only (the rows above it are already faulted), so no copy or
    /// read of the whole cube is made. Every source read is checked to lie
    /// inside the tile plus halo.
    fn heave_in_place<T: Copy>(
        &self,
        data: &mut [T],
        tile: [usize; 2],
        mut f: impl FnMut(&FaultTile, &SrcWindow<'_, T>, &mut [T]),
    ) {
        let [ni, nj, nk] = self.shape;
        let h = self.heave_halo();
        let row = nj * nk;
        let (ti, tj) = (tile[0].max(1), tile[1].max(1));
        // Original rows [w0, w1) of the previous window.
        let (mut w0, mut w1, mut buf): (usize, usize, Vec<T>) = (0, 0, Vec::new());
        let mut i0 = 0;
        while i0 < ni {
            let i1 = (i0 + ti).min(ni);
            let (n0, n1) = (i0.saturating_sub(h), (i1 + h).min(ni));
            // Rows < i0 come from the previous window (already faulted in
            // `data`); rows >= i0 are still original in `data`.
            let mut next = Vec::with_capacity((n1 - n0) * row);
            for r in n0..n1 {
                if r < i0 {
                    debug_assert!(r >= w0 && r < w1);
                    next.extend_from_slice(&buf[(r - w0) * row..(r - w0 + 1) * row]);
                } else {
                    next.extend_from_slice(&data[r * row..(r + 1) * row]);
                }
            }
            (w0, w1, buf) = (n0, n1, next);
            let mut j0 = 0;
            while j0 < nj {
                let j1 = (j0 + tj).min(nj);
                let t = self.compute_tile(i0, i1, j0, j1);
                let win = SrcWindow {
                    buf: &buf,
                    w0,
                    nj,
                    nk,
                    lo: [i0.saturating_sub(h), j0.saturating_sub(h)],
                    hi: [(i1 + h).min(ni), (j1 + h).min(nj)],
                };
                f(&t, &win, data);
                j0 = j1;
            }
            i0 = i1;
        }
        let _ = w1;
    }

    /// Fault the whole cube tile by tile.
    ///
    /// * `labels` (u8, row-major `(ni,nj,nk)`) are remapped in place with
    ///   nearest-sample lookup (categorical);
    /// * returns the binary fault mask and fault segment ids (1-based order of
    ///   active faults, 0 = no fault).
    pub fn apply_to_labels(&self, labels: &mut [u8], tile: [usize; 2]) -> (Vec<u8>, Vec<u8>) {
        let [ni, nj, nk] = self.shape;
        assert_eq!(labels.len(), ni * nj * nk);
        let mut mask = vec![0u8; labels.len()];
        let mut ids = vec![0u8; labels.len()];
        let mut col = vec![0u8; nk];
        if self.has_heave() {
            self.heave_in_place(labels, tile, |t, src, labels| {
                for i in t.i0..t.i1 {
                    for j in t.j0..t.j1 {
                        let g = (i * nj + j) * nk;
                        let l = t.col_offset(i, j);
                        for k in 0..nk {
                            let [si, sj, sk] = t.nearest_source(i, j, k, ni, nj);
                            labels[g + k] = src.at(si, sj, sk);
                        }
                        mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                        ids[g..g + nk].copy_from_slice(&t.segment_id[l..l + nk]);
                    }
                }
            });
            return (mask, ids);
        }
        self.for_each_tile(tile, |t| {
            for i in t.i0..t.i1 {
                for j in t.j0..t.j1 {
                    let g = (i * nj + j) * nk;
                    let l = t.col_offset(i, j);
                    col.copy_from_slice(&labels[g..g + nk]);
                    remap_nearest(&t.lookup[l..l + nk], &col, &mut labels[g..g + nk]);
                    mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                    ids[g..g + nk].copy_from_slice(&t.segment_id[l..l + nk]);
                }
            }
        });
        (mask, ids)
    }

    /// Displace a float volume (e.g. geologic age) with linear interpolation
    /// (`Faults.apply_xyz_displacement`). Returns the fault mask.
    pub fn apply_to_volume_f32(&self, vol: &mut [f32], tile: [usize; 2]) -> Vec<u8> {
        let [ni, nj, nk] = self.shape;
        assert_eq!(vol.len(), ni * nj * nk);
        let mut mask = vec![0u8; vol.len()];
        let mut col = vec![0.0f32; nk];
        let mut out = vec![0.0f32; nk];
        let mut lk = vec![0.0f64; nk];
        if self.has_heave() {
            self.heave_in_place(vol, tile, |t, src, vol| {
                for i in t.i0..t.i1 {
                    for j in t.j0..t.j1 {
                        let g = (i * nj + j) * nk;
                        let l = t.col_offset(i, j);
                        for k in 0..nk {
                            let (x, y) = t.source_ij(i, j, k);
                            vol[g + k] = trilinear_with(
                                |a, b, c| src.at(a, b, c),
                                [ni, nj, nk],
                                x,
                                y,
                                t.lookup[l + k] as f64,
                            );
                        }
                        mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                    }
                }
            });
            return mask;
        }
        self.for_each_tile(tile, |t| {
            for i in t.i0..t.i1 {
                for j in t.j0..t.j1 {
                    let g = (i * nj + j) * nk;
                    let l = t.col_offset(i, j);
                    col.copy_from_slice(&vol[g..g + nk]);
                    for (d, &s) in lk.iter_mut().zip(&t.lookup[l..l + nk]) {
                        *d = s as f64;
                    }
                    interp_trace_into(&lk, &col, &mut out);
                    vol[g..g + nk].copy_from_slice(&out);
                    mask[g..g + nk].copy_from_slice(&t.mask[l..l + nk]);
                }
            }
        });
        mask
    }

    /// Visit tiles in C order (`tile = [ti, tj]`, clamped to ≥1).
    pub fn for_each_tile(&self, tile: [usize; 2], mut f: impl FnMut(&FaultTile)) {
        let [ni, nj, _] = self.shape;
        let (ti, tj) = (tile[0].max(1), tile[1].max(1));
        let mut i0 = 0;
        while i0 < ni {
            let i1 = (i0 + ti).min(ni);
            let mut j0 = 0;
            while j0 < nj {
                let j1 = (j0 + tj).min(nj);
                let t = self.compute_tile(i0, i1, j0, j1);
                f(&t);
                j0 = j1;
            }
            i0 = i1;
        }
    }
}

/// Original rows `[w0, w0 + len)` of a cube around one tile, with the
/// tile-plus-halo bounds every read must stay in.
pub struct SrcWindow<'a, T> {
    buf: &'a [T],
    w0: usize,
    nj: usize,
    nk: usize,
    lo: [usize; 2],
    hi: [usize; 2],
}

impl<T: Copy> SrcWindow<'_, T> {
    /// Original sample `(i, j, k)`; panics outside the tile plus halo.
    #[inline]
    pub fn at(&self, i: usize, j: usize, k: usize) -> T {
        assert!(
            i >= self.lo[0] && i < self.hi[0] && j >= self.lo[1] && j < self.hi[1],
            "heave source ({i}, {j}) outside tile + halo {:?}..{:?}",
            self.lo,
            self.hi
        );
        self.buf[((i - self.w0) * self.nj + j) * self.nk + k]
    }
}

/// Fault products for one spatial tile (row-major `(i1-i0, j1-j0, nk)`).
#[derive(Debug, Clone, PartialEq)]
pub struct FaultTile {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub nk: usize,
    /// Source sample for each output sample (`displacement_vectors`).
    pub lookup: Vec<f32>,
    /// Binary fault mask (`fault_segments` deliverable).
    pub mask: Vec<u8>,
    /// 1-based active-fault id at mask voxels (last writer wins), else 0.
    pub segment_id: Vec<u8>,
    /// Source `(i, j)` of every voxel with heave; `None` = the voxel's own
    /// column (legacy vertical lookup).
    pub src_ij: Option<Vec<(f32, f32)>>,
}

impl FaultTile {
    /// Offset of the `(i, j)` trace inside the tile buffers.
    #[inline]
    pub fn col_offset(&self, i: usize, j: usize) -> usize {
        ((i - self.i0) * (self.j1 - self.j0) + (j - self.j0)) * self.nk
    }

    /// Nearest source voxel `(i, j, k)` of output voxel `(i, j, k)` (round
    /// in `i`, `j`; legacy `floor(k + ½)` in `k`), clamped to the cube.
    #[inline]
    pub fn nearest_source(&self, i: usize, j: usize, k: usize, ni: usize, nj: usize) -> [usize; 3] {
        let o = self.col_offset(i, j) + k;
        let r = |x: f64, n: usize| (x + 0.5).floor().clamp(0.0, (n.max(1) - 1) as f64) as usize;
        let sk = r(self.lookup[o] as f64, self.nk);
        match &self.src_ij {
            None => [i, j, sk],
            Some(s) => [r(s[o].0 as f64, ni), r(s[o].1 as f64, nj), sk],
        }
    }

    /// Continuous source `(i, j)` of output voxel `(i, j, k)`.
    #[inline]
    pub fn source_ij(&self, i: usize, j: usize, k: usize) -> (f64, f64) {
        match &self.src_ij {
            None => (i as f64, j as f64),
            Some(s) => {
                let v = s[self.col_offset(i, j) + k];
                (v.0 as f64, v.1 as f64)
            }
        }
    }
}

/// Trilinear sample of a row-major `(ni, nj, nk)` volume at a continuous
/// position (clamped to the cube).
pub fn trilinear(vol: &[f32], shape: [usize; 3], x: f64, y: f64, z: f64) -> f32 {
    let nj = shape[1];
    let nk = shape[2];
    trilinear_with(|i, j, k| vol[(i * nj + j) * nk + k], shape, x, y, z)
}

/// [`trilinear`] reading samples through `at(i, j, k)`.
#[inline]
pub fn trilinear_with(
    at: impl Fn(usize, usize, usize) -> f32,
    shape: [usize; 3],
    x: f64,
    y: f64,
    z: f64,
) -> f32 {
    let [ni, nj, nk] = shape;
    let ax = |v: f64, n: usize| {
        let v = v.clamp(0.0, (n.max(1) - 1) as f64);
        let a = v.floor() as usize;
        (a, (a + 1).min(n - 1), v - a as f64)
    };
    let (i0, i1, wi) = ax(x, ni);
    let (j0, j1, wj) = ax(y, nj);
    let (k0, k1, wk) = ax(z, nk);
    let at = |i: usize, j: usize, k: usize| at(i, j, k) as f64;
    let lerp = |a: f64, b: f64, w: f64| if w == 0.0 { a } else { a + w * (b - a) };
    let c = |i: usize| {
        let a = lerp(at(i, j0, k0), at(i, j0, k1), wk);
        let b = lerp(at(i, j1, k0), at(i, j1, k1), wk);
        lerp(a, b, wj)
    };
    lerp(c(i0), c(i1), wi) as f32
}

/// `np.interp(x, arange(len(fp)), fp)` for a single `x` (numpy semantics:
/// clamp outside, exact knot hit, `slope*(x-xj)+fp[j]`).
#[inline]
pub fn interp_uniform(x: f64, fp: &[f32]) -> f64 {
    let n = fp.len();
    if n == 0 {
        return f64::NAN;
    }
    let last = (n - 1) as f64;
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 {
        return fp[0] as f64;
    }
    if x >= last {
        return fp[n - 1] as f64;
    }
    let j = x.floor() as usize;
    let xj = j as f64;
    if xj == x {
        return fp[j] as f64;
    }
    let slope = fp[j + 1] as f64 - fp[j] as f64;
    slope * (x - xj) + fp[j] as f64
}

/// `apply_faulting` / `apply_xyz_displacement` for one trace, rounding to
/// f32 like the float32 working arrays in Python. Constant traces are copied.
pub fn interp_trace_into(x: &[f64], trace: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), trace.len());
    let first = trace.first().copied().unwrap_or(0.0);
    if trace.iter().all(|&v| v == first) {
        out.copy_from_slice(trace);
        return;
    }
    for (o, &xv) in out.iter_mut().zip(x) {
        *o = interp_uniform(xv, trace) as f32;
    }
}

/// Carry fault ids with the same lookup: prefer the nearer neighbour sample
/// with a non-zero id (so ids cover every voxel whose interpolated label > 0).
fn carry_ids(x: &[f64], ids: &[u8], out: &mut [u8]) {
    let n = ids.len();
    for (o, &xv) in out.iter_mut().zip(x) {
        let j = (xv.floor().max(0.0) as usize).min(n - 1);
        let j1 = (j + 1).min(n - 1);
        let frac = xv - j as f64;
        let (a, b) = if frac <= 0.5 { (j, j1) } else { (j1, j) };
        let fa = if a == j { 1.0 - frac } else { frac };
        *o = if ids[a] != 0 && fa > 0.0 {
            ids[a]
        } else if ids[b] != 0 && (1.0 - fa) > 0.0 {
            ids[b]
        } else {
            0
        };
    }
}

/// Nearest-sample remap of a categorical trace by a lookup trace.
pub fn remap_nearest(lookup: &[f32], src: &[u8], out: &mut [u8]) {
    let n = src.len();
    for (o, &x) in out.iter_mut().zip(lookup) {
        let k = ((x as f64) + 0.5).floor().clamp(0.0, (n - 1) as f64) as usize;
        *o = src[k];
    }
}

/// `np.interp(h, age_trace, arange(nk))` for a non-decreasing age trace —
/// the per-trace core of `Faults.improve_depth_maps_post_faulting`
/// (displaced horizon depth in samples).
pub fn horizon_depth_from_age(age: &[f32], h: f64) -> f64 {
    let n = age.len();
    if n == 0 {
        return f64::NAN;
    }
    if h < age[0] as f64 {
        return 0.0;
    }
    if h > age[n - 1] as f64 {
        return (n - 1) as f64;
    }
    // Largest j with age[j] <= h (numpy binary_search_with_guess on sorted xp).
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (age[mid] as f64) <= h {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let j = lo - 1;
    if j == n - 1 {
        return (n - 1) as f64;
    }
    let xj = age[j] as f64;
    if xj == h {
        return j as f64;
    }
    let slope = 1.0 / (age[j + 1] as f64 - xj);
    slope * (h - xj) + j as f64
}
