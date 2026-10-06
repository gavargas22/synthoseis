//! Salt bodies (legacy `datagenerator/Salt.py`) vs the legacy generator, and
//! the `salt` switch (`--no-salt`, master b4f4259).
//!
//! Fixture: `tests/fixtures/salt_reference.json` from
//! `tests/fixtures/generate_salt_reference.py`, which runs the real legacy
//! `SaltModel` (with scipy `Delaunay`), its horizon drag and
//! `Closures._flood_fill`.
//! * Bit-exact: the salt point cloud and the voxel mask (convex hull) from
//!   the model's own unit draws, including non-square cubes (legacy `y`
//!   centre bug).
//! * Bit-exact: horizon drag, Gaussian smoothing (scipy `reflect`) and the
//!   negative-thickness push-down.
//! * Bit-exact: closure depths with salt gaps walled off.
//! * Statistical at the 5 % level (two-sample KS): 10 shape statistics of
//!   4000 legacy salt bodies vs 4000 Rust keyed-draw bodies.
use serde::Deserialize;
use synthoseis_core::closure_segments::unit_closure_runs_salt;
use synthoseis_core::pipeline::{E2eConfig, FaultConfig, RockPhysicsConfig};
use synthoseis_core::salt::{
    drag_horizon_maps, hull_runs, keyed_draws, salt_body, salt_geometry, top_offset_scale,
    SaltBody, SALT_PAD,
};
use synthoseis_core::{generate_chunked, generate_labels, ToyGeometry};

#[derive(Deserialize)]
struct GeometryCase {
    shape: [usize; 3],
    h1: Vec<u64>,
    draws: Vec<u64>,
    points: Vec<u64>,
    runs: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
struct DragCase {
    shape: [usize; 3],
    nh: usize,
    maps: Vec<u64>,
    runs: Vec<[u32; 2]>,
    dragged: Vec<u64>,
}

#[derive(Deserialize)]
struct FillCase {
    shape: [usize; 2],
    max_column: f64,
    t: Vec<i64>,
    b: Vec<usize>,
    gap: Vec<u8>,
    cd2: Vec<i64>,
}

#[derive(Deserialize)]
struct Population {
    radius: Vec<f64>,
    top: Vec<f64>,
    tip: Vec<f64>,
    cx: Vec<f64>,
    cy: Vec<f64>,
    r1: Vec<f64>,
    base: Vec<f64>,
    bx: Vec<f64>,
    by: Vec<f64>,
    r2: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixture {
    geometry: Vec<GeometryCase>,
    drag: Vec<DragCase>,
    fill: Vec<FillCase>,
    population: Population,
}

fn fixture() -> Fixture {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/salt_reference.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn f64s(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from_bits(b)).collect()
}

fn body(grid: [usize; 3], runs: &[[u32; 2]]) -> SaltBody {
    SaltBody {
        grid,
        runs: runs.iter().map(|r| (r[0], r[1])).collect(),
        radius: 0.0,
        top: 0.0,
        points: Vec::new(),
    }
}

/// Bit-exact: legacy `compute_salt_body_segmentation` (radius, 99th
/// percentile top, 218-point cloud) replayed from the model's unit draws,
/// and the salt mask (voxels inside the convex hull) per column.
#[test]
fn salt_geometry_matches_legacy() {
    let fx = fixture();
    assert!(fx.geometry.len() >= 8);
    let (mut voxels, mut cols) = (0usize, 0usize);
    for (n, g) in fx.geometry.iter().enumerate() {
        let [ni, nj, nk] = g.shape;
        let grid = [ni, nj, nk + SALT_PAD];
        let draws = f64s(&g.draws);
        let mut it = draws.iter().copied();
        let mut draw = || it.next().expect("fixture draws exhausted");
        let (_r, _top, pts) = salt_geometry(&f64s(&g.h1), grid, 1.0, &mut draw);
        let want = f64s(&g.points);
        assert_eq!(pts.len() * 3, want.len(), "case {n}");
        for (p, w) in pts.iter().flatten().zip(&want) {
            assert_eq!(p.to_bits(), w.to_bits(), "case {n}: point");
        }
        let runs = hull_runs(&pts, grid);
        for (c, (got, w)) in runs.iter().zip(&g.runs).enumerate() {
            let w = if w[0] == w[1] { (0, 0) } else { (w[0], w[1]) };
            assert_eq!(*got, w, "case {n} column {c}");
            voxels += (w.1 - w.0) as usize;
            cols += (w.1 > w.0) as usize;
        }
    }
    eprintln!(
        "geometry: {} cases, {voxels} salt voxels in {cols} columns bit-exact",
        fx.geometry.len()
    );
    assert!(voxels > 10_000);
}

/// Bit-exact: legacy `update_depth_maps_with_salt_segments_drag` (shift,
/// scipy `gaussian_filter(sigma = 3)` of every horizon, push-down),
/// including maps smaller than the kernel radius.
#[test]
fn salt_drag_matches_legacy() {
    let fx = fixture();
    let mut moved = 0usize;
    for (n, d) in fx.drag.iter().enumerate() {
        let [ni, nj, nk] = d.shape;
        let salt = body([ni, nj, nk + SALT_PAD], &d.runs);
        let maps = f64s(&d.maps);
        let got = drag_horizon_maps(&maps, [ni, nj, d.nh], &salt);
        for (k, (g, w)) in got.iter().zip(&d.dragged).enumerate() {
            assert_eq!(
                g.to_bits(),
                *w,
                "case {n} value {k}: {g} vs {}",
                f64::from_bits(*w)
            );
        }
        moved += got
            .iter()
            .zip(&maps)
            .filter(|(a, b)| (*a - *b).abs() > 1.0)
            .count();
    }
    eprintln!(
        "drag: {} cases bit-exact, {moved} map values moved by > 1 sample",
        fx.drag.len()
    );
    assert!(moved > 100);
}

/// Bit-exact: closure depth with salt gaps (legacy `_flood_fill` walls):
/// gap and ring cells are never closed, and traps against the salt flank
/// match legacy column by column.
#[test]
fn salt_walled_closures_match_legacy() {
    let fx = fixture();
    assert!(fx.fill.len() >= 20);
    let (mut closed, mut gaps, mut differs) = (0usize, 0usize, 0usize);
    for (n, c) in fx.fill.iter().enumerate() {
        let [ni, nj] = c.shape;
        let nk = 160;
        let gap_top = 20usize;
        let mut labels = vec![0u8; ni * nj * nk];
        let mut runs = vec![[0u32; 2]; ni * nj];
        for col in 0..ni * nj {
            let t = if c.gap[col] == 1 {
                gap_top
            } else {
                c.t[col] as usize
            };
            let l = &mut labels[col * nk..(col + 1) * nk];
            l[t..c.b[col]].fill(1);
            l[c.b[col]..].fill(2);
            if c.gap[col] == 1 {
                runs[col] = [gap_top as u32 - 2, gap_top as u32 + 40];
            }
        }
        let salt = body([ni, nj, nk + SALT_PAD], &runs);
        let got_runs =
            unit_closure_runs_salt(&labels, [ni, nj, nk], &[1], 1, c.max_column, Some(&salt));
        let plain = unit_closure_runs_salt(&labels, [ni, nj, nk], &[1], 1, c.max_column, None);
        let mut got = vec![None; ni * nj];
        for r in &got_runs {
            assert!(got[r.col].is_none());
            got[r.col] = Some(*r);
        }
        let mut plain_cd = vec![None; ni * nj];
        for r in &plain {
            plain_cd[r.col] = Some(r.contact);
        }
        for col in 0..ni * nj {
            let cd = c.cd2[col] as f64 / 2.0;
            if c.gap[col] == 1 {
                gaps += 1;
                assert_eq!(c.cd2[col], 0);
                assert!(got[col].is_none(), "case {n} col {col}: gap closed");
                continue;
            }
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
                    assert_eq!(r.k0, c.t[col] as usize);
                }
            }
            if got[col].map(|r| r.contact) != plain_cd[col] {
                differs += 1;
            }
        }
    }
    eprintln!("fill: {closed} closed columns bit-exact, {gaps} gap columns open, {differs} columns changed by the walls");
    assert!(closed > 2000 && differs > 100);
}

fn ks(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
    }
    d
}

/// Rust keyed-draw population on the legacy example grid
/// (64 x 64 x 1250 + pad, flat horizon 1 at 20 samples).
pub fn rust_population(n: usize) -> Vec<[f64; 10]> {
    let grid = [64, 64, 1250 + SALT_PAD];
    let h1 = vec![20.0; 64 * 64];
    (0..n as u64)
        .map(|s| {
            let mut draw = keyed_draws(0x5A17_0000 + s);
            let (radius, top, p) = salt_geometry(&h1, grid, 1.0, &mut draw);
            let stats = |c: &[[f64; 3]]| {
                let mx = c.iter().map(|q| q[0]).sum::<f64>() / c.len() as f64;
                let my = c.iter().map(|q| q[1]).sum::<f64>() / c.len() as f64;
                let r =
                    c.iter().map(|q| (q[0] - mx).hypot(q[1] - my)).sum::<f64>() / c.len() as f64;
                let z = c.iter().map(|q| q[2]).sum::<f64>() / c.len() as f64;
                (mx, my, r, z)
            };
            let (cx, cy, r1, _) = stats(&p[0..36]);
            let (bx, by, r2, base) = stats(&p[109..145]);
            [radius, top, p[108][2], cx, cy, r1, base, bx, by, r2]
        })
        .collect()
}

/// Statistical (5 %): two-sample KS of 10 salt-shape statistics, legacy
/// numpy draws (4000 bodies) vs Rust keyed draws (16000 bodies). 5 % critical
/// `1.358 * sqrt((n + m) / (n m))`.
#[test]
fn salt_population_matches_legacy_ks() {
    let fx = fixture();
    let p = &fx.population;
    let rust = rust_population(16_000);
    let legacy: [(&str, &Vec<f64>); 10] = [
        ("radius", &p.radius),
        ("top", &p.top),
        ("tip", &p.tip),
        ("cx", &p.cx),
        ("cy", &p.cy),
        ("r1", &p.r1),
        ("base", &p.base),
        ("bx", &p.bx),
        ("by", &p.by),
        ("r2", &p.r2),
    ];
    let (n, m) = (p.radius.len() as f64, rust.len() as f64);
    let crit = 1.358 * ((n + m) / (n * m)).sqrt();
    for (k, (name, l)) in legacy.iter().enumerate() {
        let r: Vec<f64> = rust.iter().map(|v| v[k]).collect();
        let d = ks(l, &r);
        eprintln!("KS {name}: D = {d:.4}, 5% critical {crit:.4}");
        assert!(d < crit, "{name}: D = {d} >= {crit}");
    }
}

fn demo(salt: bool) -> E2eConfig {
    E2eConfig {
        seed: 7,
        inline_count: 32,
        crossline_count: 32,
        samples: 128,
        faults: FaultConfig {
            count: 2,
            ..FaultConfig::default()
        },
        rock_physics: RockPhysicsConfig {
            salt,
            ..RockPhysicsConfig::default()
        },
        geometry: ToyGeometry::Layered,
        time: synthoseis_core::TimeConfig::legacy(),
        ..E2eConfig::default()
    }
}

/// Salt voxels take the legacy salt properties (rho 2.17, vp 4500,
/// vs 2250) in the elastic model, whatever the layer label, on the default
/// config (partial voxels on since PR B2). Salt occupies
/// `ζ ∈ [lo + ½, hi + ½]` of each column (continuous hull bounds, cell `k`
/// = `[k, k+1)`):
/// * cells fully inside that interval are pure salt: exactly the legacy
///   properties;
/// * cells that straddle a salt bound are Backus mixes of salt and the
///   host sediment: never the pure salt triple, made of a salt sub-layer
///   (the salt part of the cell) and host sub-layers, with density and
///   P modulus between the host's and the salt's (the Backus bounds: rho
///   is the arithmetic and M = rho·vp² the harmonic mean of the parts), so
///   the check does not assume the host is slower than salt;
/// * every salt column has at least one straddling cell;
/// * cells that do not touch the salt never carry salt properties;
/// * the salt labels are the whole-voxel ones (centre rule).
#[test]
fn salt_voxels_take_legacy_properties() {
    let cfg = demo(true);
    assert!(
        cfg.effective_partial_voxels().is_some(),
        "partial voxels on by default"
    );
    let body = salt_body(&cfg).expect("salt on by default");
    let bounds = body.hull_bounds();
    let [ni, nj, nk] = cfg.shape();
    assert!(body.voxels(nk) > 500, "salt voxels {}", body.voxels(nk));
    let (labels, shape) = generate_labels(&cfg);
    let model = synthoseis_core::rock_physics::elastic_model(&cfg, &labels, shape);
    assert!(model.salt().is_some());
    let n = ni * nj * nk;
    let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    let synthoseis_core::rock_physics::ElasticModel::Rpm(rpm) = &model else {
        panic!("default model is the rock-physics model")
    };
    let salt = (2.17f32, 4500.0f32, 2250.0f32);
    let modulus = |rho: f32, vp: f32| rho as f64 * vp as f64 * vp as f64;
    let (mut labelled, mut inside, mut straddling, mut salt_columns) = (0, 0, 0, 0);
    let mut scratch = synthoseis_core::rock_physics::ColumnScratch::default();
    let (mut crho, mut cvp, mut cvs) = (vec![0f32; nk], vec![0f32; nk], vec![0f32; nk]);
    assert_eq!(bounds.len(), ni * nj);
    for (c, column) in bounds.iter().enumerate() {
        // Sub-layers of the column (the parts each cell is mixed from).
        let (i, j) = (c / nj, c % nj);
        let tile = rpm.partial_tile(i, i + 1, j, j + 1);
        let col = &labels[c * nk..(c + 1) * nk];
        rpm.column_partial(
            i,
            j,
            col,
            &tile,
            &mut scratch,
            &mut crho,
            &mut cvp,
            &mut cvs,
        );
        let cells: Vec<_> = scratch.partial.cells().collect();
        assert_eq!(cells.len(), nk);
        let mut column_straddling = 0;
        let mut column_salt = false;
        for k in 0..nk {
            let v = c * nk + k;
            let props = (rho[v], vp[v], vs[v]);
            if body.contains(c, k) {
                labelled += 1;
            }
            // Salt part of cell k: [k, k+1) ∩ [lo + ½, hi + ½].
            let overlap = column.map_or(0.0, |(lo, hi)| {
                ((k + 1) as f64).min(hi + 0.5) - (k as f64).max(lo + 0.5)
            });
            column_salt |= overlap > 0.0;
            if overlap >= 1.0 {
                inside += 1;
                assert!(
                    body.contains(c, k),
                    "column {c} cell {k}: inside the hull but not labelled salt"
                );
                assert_eq!(props, salt, "column {c} cell {k}: pure salt");
            } else if overlap > 0.0 {
                straddling += 1;
                column_straddling += 1;
                assert_ne!(
                    props, salt,
                    "column {c} cell {k}: straddling cell (salt part {overlap:.3}) is pure salt"
                );
                assert_eq!(
                    (crho[k], cvp[k], cvs[k]),
                    props,
                    "column {c} cell {k}: column and tile paths agree"
                );
                let parts = cells[k];
                let salt_frac: f64 = parts
                    .iter()
                    .filter(|p| (p.rho, p.vp, p.vs) == salt)
                    .map(|p| p.frac)
                    .sum();
                assert!(
                    (salt_frac - overlap).abs() < 1e-6,
                    "column {c} cell {k}: salt sub-layer {salt_frac} vs salt part {overlap}"
                );
                assert!(
                    parts
                        .iter()
                        .any(|p| p.frac > 0.0 && (p.rho, p.vp, p.vs) != salt),
                    "column {c} cell {k}: no host sub-layer"
                );
                // Backus bounds: between the host's and the salt's values.
                let live = parts.iter().filter(|p| p.frac > 0.0);
                let (r0, r1) = live.clone().fold((f32::MAX, f32::MIN), |(a, b), p| {
                    (a.min(p.rho), b.max(p.rho))
                });
                let (m0, m1) = live.fold((f64::MAX, f64::MIN), |(a, b), p| {
                    let m = modulus(p.rho, p.vp);
                    (a.min(m), b.max(m))
                });
                let (r, m) = (rho[v], modulus(rho[v], vp[v]));
                let tol = 1e-5;
                assert!(
                    r >= r0 * (1.0 - tol as f32) && r <= r1 * (1.0 + tol as f32),
                    "column {c} cell {k}: rho {r} outside the parts' [{r0}, {r1}]"
                );
                assert!(
                    m >= m0 * (1.0 - tol) && m <= m1 * (1.0 + tol),
                    "column {c} cell {k}: P modulus {m} outside the parts' [{m0}, {m1}]"
                );
            } else {
                assert!(
                    vp[v] != salt.1 || vs[v] != salt.2,
                    "column {c} cell {k}: non-salt cell with salt properties"
                );
            }
        }
        if column_salt {
            salt_columns += 1;
            assert!(
                column_straddling > 0,
                "salt column {c}: no straddling (partial-salt) cell"
            );
        }
    }
    assert_eq!(labelled, body.voxels(nk), "salt labels");
    println!(
        "salt cells: {labelled} labelled, {inside} fully inside, {straddling} straddling \
         in {salt_columns} salt columns"
    );
    assert!(inside > 500, "fully-inside salt cells {inside}");
    assert!(salt_columns > 0, "no salt columns");
}

/// Salt is on by default for the layered geometry only; `salt: false`
/// removes it, and the planar geometry / `legacy_toy_depth` never have it.
#[test]
fn salt_default_and_switches() {
    assert!(RockPhysicsConfig::default().salt);
    assert!(salt_body(&demo(true)).is_some());
    assert!(salt_body(&demo(false)).is_none());
    let planar = E2eConfig {
        geometry: ToyGeometry::Planar,
        ..demo(true)
    };
    assert!(salt_body(&planar).is_none());
    let toy = E2eConfig {
        rock_physics: RockPhysicsConfig::legacy_toy(),
        ..demo(true)
    };
    assert!(salt_body(&toy).is_none());
    // Salt changes the model: horizons drag and salt properties appear.
    let (with, _) = generate_chunked(&demo(true));
    let (without, _) = generate_chunked(&demo(false));
    assert_ne!(with.labels, without.labels);
    assert_ne!(with.angle_stack, without.angle_stack);
    // Deterministic.
    let (again, _) = generate_chunked(&demo(true));
    assert_eq!(with.labels, again.labels);
    assert_eq!(
        with.angle_stack
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        again
            .angle_stack
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

/// The legacy absolute top offset (`--salt-legacy-top-offset`) puts the
/// salt 150-300 samples below horizon 1: below a 128-sample cube.
#[test]
fn salt_legacy_top_offset_is_absolute() {
    let mut cfg = demo(true);
    let scaled = salt_body(&cfg).unwrap();
    cfg.rock_physics.salt_legacy_top_offset = true;
    let legacy = salt_body(&cfg).unwrap();
    assert_eq!(scaled.radius, legacy.radius);
    assert!(legacy.top - scaled.top > 100.0);
    assert_eq!(legacy.voxels(cfg.samples), 0);
}

/// The default top-offset scale is capped at 1 (`min(nk / 1250, 1)`): on a
/// cube over 1250 samples the default salt equals the legacy absolute offset
/// bit for bit, and is shallower than an uncapped `nk / 1250` scale would
/// put it. Below 1250 samples the scale is `nk / 1250` (unchanged goldens).
#[test]
fn salt_top_offset_scale_capped_at_one() {
    assert_eq!(top_offset_scale(128, false), 128.0 / 1250.0);
    assert_eq!(top_offset_scale(1250, false), 1.0);
    assert_eq!(top_offset_scale(1600, false), 1.0);
    assert_eq!(top_offset_scale(128, true), 1.0);
    assert_eq!(top_offset_scale(1600, true), 1.0);

    let mut cfg = demo(true);
    cfg.inline_count = 16;
    cfg.crossline_count = 16;
    cfg.samples = 1600;
    let default = salt_body(&cfg).unwrap();
    cfg.rock_physics.salt_legacy_top_offset = true;
    let legacy = salt_body(&cfg).unwrap();
    assert_eq!(default.top.to_bits(), legacy.top.to_bits());
    assert_eq!(default.runs, legacy.runs);
    assert!(default.voxels(cfg.samples) > 0);

    let (maps, nh) = synthoseis_core::toy_geometry::layered_horizon_maps(cfg.seed, cfg.shape());
    let h1: Vec<f64> = (0..16 * 16).map(|c| maps[c * nh + 1]).collect();
    let grid = [16, 16, 1600 + SALT_PAD];
    let (_, uncapped, _) = salt_geometry(&h1, grid, 1600.0 / 1250.0, &mut keyed_draws(cfg.seed));
    assert!(
        uncapped > default.top + 1.0,
        "uncapped {uncapped} vs capped {}",
        default.top
    );
}

/// The bounded-memory read-back check (`verify_salt_labels`, one chunk at a
/// time) accepts a streamed store and catches a corrupted salt chunk.
#[test]
fn salt_labels_chunked_verify_catches_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("s.mdio");
    let cfg = E2eConfig {
        chunk_shape: Some([8, 8, 32]),
        store_path: Some(p.clone()),
        ..demo(true)
    };
    synthoseis_core::run_e2e_streaming(&cfg).unwrap();
    let store = synthoseis_io::MdioStore::open(&p).unwrap();
    synthoseis_core::salt::verify_salt_labels(&store, &cfg).unwrap();
    let full = store.read_salt_labels_u8().unwrap();
    assert_eq!(
        Some(full),
        synthoseis_core::salt::generate_salt_labels(&cfg)
    );
    // Flip one voxel in a chunk that holds salt.
    let body = salt_body(&cfg).unwrap();
    let c = (0..32 * 32)
        .find(|&c| body.runs[c].1 > body.runs[c].0 && (body.runs[c].0 as usize) < 128)
        .unwrap();
    let (i, j, k) = (c / 32, c % 32, body.runs[c].0 as usize);
    let idx = [i / 8, j / 8, k / 32];
    let mut chunk = store.read_salt_labels_chunk(idx).unwrap();
    assert!(chunk.iter().any(|&v| v == 1));
    let off = ((i % 8) * 8 + j % 8) * 32 + k % 32;
    assert_eq!(chunk[off], 1);
    chunk[off] = 0;
    store.write_salt_labels_chunk(idx, &chunk).unwrap();
    let e = synthoseis_core::salt::verify_salt_labels(&store, &cfg).unwrap_err();
    assert!(e.contains(&format!("{idx:?}")), "{e}");
}

/// Fault labels are `fault AND NOT salt` by default (faults die out against
/// salt; a fault label inside the salt has no seismic expression), and
/// `fault_labels_through_salt` reproduces master 2b3850ba. Counts pinned on
/// the demo cube (64x64x256, seed 7, 4 faults) and the seed-11 invariance
/// case of `tests/rock_physics.rs`.
#[test]
fn fault_labels_exclude_salt() {
    let demo_cube = E2eConfig {
        seed: 7,
        inline_count: 64,
        crossline_count: 64,
        samples: 256,
        faults: FaultConfig::with_count(4),
        geometry: ToyGeometry::Layered,
        ..E2eConfig::default()
    };
    let seed11 = E2eConfig {
        seed: 11,
        inline_count: 24,
        crossline_count: 24,
        samples: 128,
        faults: FaultConfig::with_count(4),
        rock_physics: RockPhysicsConfig {
            sand_layer_fraction: Some(0.4),
            ..RockPhysicsConfig::default()
        },
        geometry: ToyGeometry::Layered,
        ..E2eConfig::default()
    };
    // (fault voxels through salt, masked fault voxels, salt voxels)
    for (cfg, through_n, masked_n, salt_n) in
        [(demo_cube, 33_812, 32_193, 18_535), (seed11, 4_074, 3_919, 2_851)]
    {
        assert!(cfg.effective_fault_salt_mask());
        let salt = synthoseis_core::salt::generate_salt_labels(&cfg).unwrap();
        let masked = synthoseis_core::generate_fault_labels(&cfg).unwrap();
        let mut through_cfg = cfg.clone();
        through_cfg.rock_physics.fault_labels_through_salt = true;
        assert!(!through_cfg.effective_fault_salt_mask());
        let through = synthoseis_core::generate_fault_labels(&through_cfg).unwrap();
        let count = |v: &[u8]| v.iter().map(|&x| x as usize).sum::<usize>();
        let both = |f: &[u8]| f.iter().zip(&salt).filter(|(a, b)| **a == 1 && **b == 1).count();
        assert_eq!(count(&salt), salt_n);
        assert_eq!(count(&through), through_n);
        assert_eq!(count(&masked), masked_n);
        assert_eq!(both(&masked), 0, "no voxel is both fault and salt");
        assert_eq!(both(&through), through_n - masked_n);
        assert!(through_n > masked_n);
        for v in 0..salt.len() {
            assert_eq!(masked[v], through[v] & (1 - salt[v]), "voxel {v}");
        }
        // Tile level: mask and segment_id are cleared together, so
        // `segment_id != 0 <=> mask == 1` still holds.
        let model = synthoseis_core::fault_model(&cfg).unwrap();
        let body = salt_body(&cfg).unwrap();
        let [ni, nj, nk] = cfg.shape();
        let mut t = model.compute_tile(0, ni, 0, nj);
        let before = t.mask.clone();
        synthoseis_core::salt::mask_fault_tile_salt(&mut t, &body);
        for (l, (&m, &s)) in t.mask.iter().zip(&t.segment_id).enumerate() {
            assert_eq!(m == 1, s != 0, "segment_id invariant at {l}");
        }
        for i in 0..ni {
            for j in 0..nj {
                let l = t.col_offset(i, j);
                let g = (i * nj + j) * nk;
                assert_eq!(t.mask[l..l + nk], masked[g..g + nk]);
                assert_eq!(before[l..l + nk], through[g..g + nk]);
            }
        }
    }
}
