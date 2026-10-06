//! Depth-to-time conversion wired through the pipeline (spec "depth-to-time
//! conversion" §5, PR B): tiling invariance of every path, the label round
//! trip, the dead last sample, the dt = 2 ms axis, the uniform 2000 m/s case
//! against the depth fuse, the per-tile cost gate and the legacy switch
//! against master f3720fb2.

use std::time::Instant;

use synthoseis_core::pipeline::{generate_tiny_cube, E2eConfig};
use synthoseis_core::pipeline_stream::{fuse_tile_filtered, fuse_tile_local};
use synthoseis_core::time_mode::tile_twt;
use synthoseis_core::{
    elastic_model, generate_chunked, generate_fault_labels, generate_labels,
    generate_output_labels, generate_reflectivity, run_e2e_geometry_once_seismic_many,
    run_e2e_multiprocess, run_e2e_streaming, run_e2e_streaming_overlapped, run_e2e_strip_stitched,
    time_column_summary,
    FaultConfig, FilterConfig, NoiseConfig, RockPhysicsConfig, SeismicFilters, TimeConfig,
    TwtKernel, WorkingSetStats, DEFAULT_INCIDENCE_DEG,
};

mod common;
use common::whole;

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv_f32(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The CLI `RICH` set as a library config (3 faults, 4-30 Hz bandpass,
/// 12.5 dB noise; salt on, layered Markov geometry), in time mode with
/// `nt` output samples (`None` = nt₀ = nz).
fn rich(seed: u64, shape: [usize; 3], nt: Option<usize>, lateral: usize) -> E2eConfig {
    let mut filters = FilterConfig::legacy(4.0, 30.0, lateral);
    filters.noise = NoiseConfig { snr_db: Some(12.5), ..NoiseConfig::default() };
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        faults: FaultConfig::with_count(3),
        filters,
        time: TimeConfig { samples: nt, ..TimeConfig::default() },
        ..E2eConfig::default()
    }
}

struct Store {
    angles: Vec<f32>,
    labels: Vec<u8>,
    faults: Option<Vec<u8>>,
    salt: Option<Vec<u8>>,
    shape: [usize; 3],
    digi: f64,
}

fn read_store(p: &std::path::Path) -> Store {
    let s = synthoseis_io::MdioStore::open(p).unwrap();
    let has = |a: &str| p.join("data").join(a).join(".zarray").is_file();
    Store {
        angles: s.read_volume().unwrap(),
        labels: s.read_labels_u8().unwrap(),
        faults: has("fault_labels").then(|| s.read_fault_labels_u8().unwrap()),
        salt: has("salt_labels").then(|| s.read_salt_labels_u8().unwrap()),
        shape: s.shape(),
        digi: s.config().digi,
    }
}

/// Reference outputs of `cfg`: the classic full-volume pass and the
/// output-domain label cubes.
struct Reference {
    angles: Vec<u32>,
    labels: Vec<u8>,
    faults: Option<Vec<u8>>,
    salt: Option<Vec<u8>>,
}

fn reference(cfg: &E2eConfig) -> Reference {
    let classic = generate_tiny_cube(cfg);
    assert_eq!(classic.shape, cfg.output_shape());
    let (labels, shape) = generate_labels(cfg);
    let model = elastic_model(cfg, &labels, shape);
    let out = generate_output_labels(cfg, &labels, &model);
    assert_eq!(classic.labels, out.labels, "classic labels vs output labels");
    Reference { angles: bits(&classic.angle_stack), labels: out.labels, faults: out.faults, salt: out.salt }
}

fn assert_store(name: &str, cfg: &E2eConfig, s: &Store, r: &Reference) {
    assert_eq!(s.shape, cfg.output_shape(), "{name}: shape");
    assert_eq!(s.digi, cfg.time.dt_ms, "{name}: digi");
    assert!(bits(&s.angles) == r.angles, "{name}: angle_stack");
    assert!(s.labels == r.labels, "{name}: labels");
    assert!(s.faults == r.faults, "{name}: fault_labels");
    assert!(s.salt == r.salt, "{name}: salt_labels");
}

/// Every path writes the same time-domain cubes (angle stack, labels,
/// fault_labels, salt_labels): classic, chunked and streaming and overlapped
/// streaming (every chunk shape given, including single-column and
/// k-chunked tiles), strip-stitch 2/3/4, multiprocess 1/2/3 and
/// geometry-once (its store; spec §5.4).
fn assert_tiling_invariant(cfg: &E2eConfig, chunk_shapes: &[[usize; 3]]) {
    let r = reference(cfg);
    let nt = cfg.output_samples();
    let dir = tempfile::tempdir().unwrap();
    for (n, &c) in chunk_shapes.iter().enumerate() {
        let c = [c[0], c[1], if c[2] == 0 { nt } else { c[2] }];
        let cc = E2eConfig { chunk_shape: Some(c), ..cfg.clone() };
        let (v, _) = generate_chunked(&cc);
        assert!(bits(&v.angle_stack) == r.angles, "chunked {c:?}: angle_stack");
        assert!(v.labels == r.labels, "chunked {c:?}: labels");
        let p = dir.path().join(format!("stream{n}.mdio"));
        run_e2e_streaming(&E2eConfig { store_path: Some(p.clone()), ..cc.clone() }).unwrap();
        assert_store(&format!("streaming {c:?}"), cfg, &read_store(&p), &r);
        let p = dir.path().join(format!("overlap{n}.mdio"));
        run_e2e_streaming_overlapped(&E2eConfig { store_path: Some(p.clone()), ..cc.clone() }).unwrap();
        assert_store(&format!("overlap {c:?}"), cfg, &read_store(&p), &r);
    }
    let strip_chunks = [3, 4, 16];
    for workers in [2, 3, 4] {
        let p = dir.path().join(format!("strip{workers}.mdio"));
        let sc = E2eConfig { store_path: Some(p.clone()), chunk_shape: Some(strip_chunks), ..cfg.clone() };
        run_e2e_strip_stitched(&sc, workers).unwrap();
        assert_store(&format!("strip {workers}"), cfg, &read_store(&p), &r);
    }
    for workers in [1, 2, 3] {
        let p = dir.path().join(format!("mp{workers}.mdio"));
        let mc = E2eConfig { store_path: Some(p.clone()), chunk_shape: Some([2, 5, 8]), ..cfg.clone() };
        run_e2e_multiprocess(&mc, workers).unwrap();
        assert_store(&format!("multiprocess {workers}"), cfg, &read_store(&p), &r);
    }
    let p = dir.path().join("geo.mdio");
    let gc = E2eConfig { chunk_shape: Some([4, 4, nt]), store_path: Some(p), ..cfg.clone() };
    let (g, _) = run_e2e_geometry_once_seismic_many(&gc, &[DEFAULT_INCIDENCE_DEG]).unwrap();
    assert!(bits(&g.stacks[0].volumes.angle_stack) == r.angles, "geometry-once: angle_stack");
    assert!(g.labels == r.labels, "geometry-once: labels");
    let gp = g.stacks[0].store_path.as_ref().expect("geometry-once store");
    assert_store("geometry-once store", cfg, &read_store(gp), &r);
}

const TILING_CHUNKS: [[usize; 3]; 4] = [[1, 1, 0], [5, 7, 0], [3, 20, 16], [8, 5, 16]];

/// Tiling invariance (spec §5.4) on the RICH flags, seeds 30 and 11, at
/// nt = nz and nt = nz + 37 (the long trace runs past every column's base:
/// half-space zeros). Columns are both short and long at nt = nz.
#[test]
fn time_mode_tiling_invariance_rich() {
    let shape = [16, 20, 64];
    let mut short_and_long = false;
    for (seed, lateral) in [(30u64, 3usize), (11, 1)] {
        for nt in [None, Some(shape[2] + 37)] {
            let cfg = rich(seed, shape, nt, lateral);
            let s = time_column_summary(&cfg).unwrap();
            println!("seed {seed} nt {:?}: {s:?}", cfg.output_samples());
            assert_eq!(s.columns, shape[0] * shape[1]);
            short_and_long |= s.short > 0 && s.long > 0;
            let (labels, sh) = generate_labels(&cfg);
            let model = elastic_model(&cfg, &labels, sh);
            let out = generate_output_labels(&cfg, &labels, &model);
            assert!(out.faults.as_ref().unwrap().iter().any(|&v| v == 1), "seed {seed}: faults");
            if nt.is_none() && seed == 30 {
                assert!(out.salt.as_ref().map_or(false, |m| m.iter().any(|&v| v == 1)), "seed 30: salt");
            }
            assert_tiling_invariant(&cfg, &TILING_CHUNKS);
        }
    }
    assert!(short_and_long, "some configuration must have both short and long columns");
}

/// Output-domain fault labels of `cfg` with the fault-label salt mask
/// (`masked`, the default) and without it (`through`, the
/// `--fault-labels-through-salt` switch), and the output salt labels.
struct MaskCubes {
    masked: Vec<u8>,
    through: Vec<u8>,
    salt: Vec<u8>,
}

fn through_salt(cfg: &E2eConfig) -> E2eConfig {
    let mut t = cfg.clone();
    t.rock_physics.fault_labels_through_salt = true;
    t
}

fn mask_cubes(cfg: &E2eConfig) -> MaskCubes {
    assert!(cfg.effective_fault_salt_mask(), "the mask must be on by default here");
    let (labels, sh) = generate_labels(cfg);
    let model = elastic_model(cfg, &labels, sh);
    let out = generate_output_labels(cfg, &labels, &model);
    let through = generate_output_labels(&through_salt(cfg), &labels, &model);
    assert_eq!(out.labels, through.labels, "the mask only touches fault_labels");
    assert_eq!(out.salt, through.salt, "the mask only touches fault_labels");
    MaskCubes { masked: out.faults.unwrap(), through: through.faults.unwrap(), salt: out.salt.unwrap() }
}

/// The removed voxels of one time-mode output (`masked` against `through`,
/// with the output salt labels `salt`): asserts `masked == through AND NOT
/// salt` voxel by voxel, so the removed count equals the fault ∩ salt
/// overlap of the unmasked cube, and returns that count.
fn assert_mask_removes_overlap(what: &str, masked: &[u8], through: &[u8], salt: &[u8]) -> usize {
    assert_eq!(masked.len(), through.len(), "{what}");
    assert_eq!(masked.len(), salt.len(), "{what}");
    let sum = |m: &[u8]| m.iter().map(|&v| v as usize).sum::<usize>();
    let overlap = through.iter().zip(salt).filter(|(&f, &s)| f == 1 && s == 1).count();
    let removed = sum(through) - sum(masked);
    for v in 0..masked.len() {
        let want = through[v] == 1 && salt[v] == 0;
        assert_eq!(masked[v] == 1, want, "{what}: voxel {v} is not fault AND NOT salt");
    }
    assert_eq!(removed, overlap, "{what}: removed count vs fault ∩ salt");
    removed
}

/// Faults and salt together at 24 × 24 × 128 (seed 4, RICH flags). Without
/// the mask the output cubes have 150 fault ∩ salt voxels; the mask (#38)
/// removes exactly those. 30–36 of the 150 (30, 30 and 36 for the three
/// chunk shapes here) touch a chunk face: they have a removed neighbour in
/// the next chunk across an i, j or k face. So every per-chunk writer masks
/// on both sides of chunk edges (spec §5.4, §8), and every path is
/// bit-identical with the mask on.
#[test]
fn time_mode_tiling_invariance_faults_and_salt_across_chunk_edges() {
    let cfg = rich(4, [24, 24, 128], None, 3);
    let m = mask_cubes(&cfg);
    let removed = assert_mask_removes_overlap("seed 4 output labels", &m.masked, &m.through, &m.salt);
    let [ni, nj, nt] = cfg.output_shape();
    let gone = |v: [usize; 3]| {
        let x = (v[0] * nj + v[1]) * nt + v[2];
        m.through[x] == 1 && m.salt[x] == 1
    };
    let removed_at: Vec<[usize; 3]> = (0..ni * nj * nt)
        .filter(|&x| m.through[x] == 1 && m.salt[x] == 1)
        .map(|x| [x / (nj * nt), (x / nt) % nj, x % nt])
        .collect();
    assert_eq!(removed, 150, "seed 4: fault ∩ salt voxels of the unmasked output");
    let chunks = [[5, 7, 0], [8, 5, 16], [3, 20, 32]];
    for c in chunks {
        let c = [c[0], c[1], if c[2] == 0 { nt } else { c[2] }];
        let dims = [ni, nj, nt];
        // Removed voxels with a removed neighbour in the next chunk across
        // an i, j or k face (counted once per voxel, on the lower side).
        let on_face = removed_at
            .iter()
            .filter(|v| {
                (0..3).any(|d| {
                    let mut w = **v;
                    w[d] += 1;
                    w[d] < dims[d] && w[d] % c[d] == 0 && gone(w)
                })
            })
            .count();
        let mut ids: Vec<[usize; 3]> = removed_at.iter().map(|v| [v[0] / c[0], v[1] / c[1], v[2] / c[2]]).collect();
        ids.sort();
        ids.dedup();
        println!(
            "seed 4: mask removed {removed} fault ∩ salt voxels; {on_face} touch a chunk face of {c:?}; in {} chunks",
            ids.len()
        );
        assert!(on_face > 0, "the mask must remove voxels across a chunk face of {c:?}");
        assert!(ids.len() >= 2, "removed voxels must span several chunks of {c:?}");
    }
    assert_tiling_invariant(&cfg, &chunks);
}

/// Fault-label salt mask on every time-mode output path (spec §8, #38 +
/// #39), RICH flags at 24 × 24 × 128: for the in-memory output cubes and
/// the store of the classic, streaming, overlapped streaming,
/// strip-stitched and multi-process writers, the removed count (fault
/// voxels without the mask − with it) equals the fault ∩ salt overlap of
/// the unmasked output, voxel by voxel (`fault AND NOT salt`), and
/// `--fault-labels-through-salt` turns the mask off on every path.
/// Seed 30 (the #34 salt case): its 3 faults never reach the salt, so the
/// overlap is 0 and the mask must change nothing. Seed 7 (the demo
/// seed): 79 fault ∩ salt voxels removed with whole voxels (the d8b96e69
/// default, `whole`), and a partial-voxel count with the default (PR B2: the
/// time-domain cubes are sampled through the partial traveltime). The
/// scaled closure minimum (20 voxels here) keeps more traps, which moves
/// the partial traveltime: 84, and master bad1daa8's 85 under
/// `ClosureMinimum::LEGACY` (`--legacy-closure-minimum`). The whole-voxel
/// pins are unchanged (labels and the depth-sampled cubes never depend on
/// fluids). Per PR: every case in memory and through the tiled streaming
/// writer; every writer path runs nightly
/// ([`time_mode_fault_salt_mask_removed_count_every_path`]).
#[test]
fn time_mode_fault_salt_mask_removed_count() {
    mask_removed_count(MASK_CASES, false);
}

/// [`time_mode_fault_salt_mask_removed_count`] through every writer path
/// (classic, streaming, overlap, strip 3, multiprocess 2).
#[test]
#[ignore = "nightly: fault/salt mask count through every writer path"]
fn time_mode_fault_salt_mask_removed_count_every_path() {
    mask_removed_count(MASK_CASES, true);
}

/// `(seed, removed, whole voxels, legacy minimum)`.
const MASK_CASES: &[(u64, usize, bool, bool)] = &[
    (30, 0, true, false),
    (7, 79, true, false),
    (7, 84, false, false),
    (7, 85, false, true),
];

/// The in-memory mask count of each case, then the store of the tiled
/// streaming writer (`every_path`: every writer path).
fn mask_removed_count(cases: &[(u64, usize, bool, bool)], every_path: bool) {
    for &(seed, want, whole_voxels, legacy_minimum) in cases {
        let mut cfg = rich(seed, [24, 24, 128], None, 3);
        if legacy_minimum {
            cfg.rock_physics.closure_minimum = synthoseis_core::ClosureMinimum::LEGACY;
        }
        let cfg = if whole_voxels { whole(&cfg) } else { cfg };
        let m = mask_cubes(&cfg);
        let removed = assert_mask_removes_overlap(&format!("seed {seed} in-memory"), &m.masked, &m.through, &m.salt);
        assert_eq!(removed, want, "seed {seed}: fault ∩ salt overlap of the unmasked output");
        assert!(m.salt.iter().any(|&v| v == 1) && m.masked.iter().any(|&v| v == 1), "seed {seed}");
        let dir = tempfile::tempdir().unwrap();
        let c = Some([5, 7, 32]);
        type Run = fn(&E2eConfig);
        let paths: [(&str, Run); 5] = [
            ("classic", |c| {
                synthoseis_core::pipeline::run_e2e(c).unwrap();
            }),
            ("streaming", |c| {
                run_e2e_streaming(c).unwrap();
            }),
            ("overlap", |c| {
                run_e2e_streaming_overlapped(c).unwrap();
            }),
            ("strip 3", |c| {
                run_e2e_strip_stitched(c, 3).unwrap();
            }),
            ("multiprocess 2", |c| {
                run_e2e_multiprocess(c, 2).unwrap();
            }),
        ];
        for (name, run) in paths {
            if !every_path && name != "streaming" {
                continue;
            }
            let mut got = Vec::new();
            for (tag, cc) in [("masked", cfg.clone()), ("through", through_salt(&cfg))] {
                let p = dir.path().join(format!("{}-{tag}.mdio", name.replace(' ', "")));
                run(&E2eConfig { store_path: Some(p.clone()), chunk_shape: c, ..cc });
                let s = read_store(&p);
                got.push((s.faults.expect("fault_labels"), s.salt.expect("salt_labels")));
            }
            let ((masked, salt), (through, salt2)) = (&got[0], &got[1]);
            assert!(salt == salt2 && *salt == m.salt, "seed {seed} {name}: salt_labels");
            assert!(*masked == m.masked, "seed {seed} {name}: masked fault_labels vs output labels");
            assert!(*through == m.through, "seed {seed} {name}: unmasked fault_labels vs output labels");
            let r = assert_mask_removes_overlap(&format!("seed {seed} {name}"), masked, through, salt);
            println!("seed {seed} {name}: mask removed {r} fault ∩ salt voxels");
            assert_eq!(r, removed, "seed {seed} {name}");
        }
    }
}

/// dt = 2 ms on a 2 m depth grid: tiling invariant, no staircase warning
/// (2 m ≤ 1580 · 2 ms / 1.2 = 2.63 m). dt = 2 ms on the default 4 m grid
/// fires the staircase warning (spec §3.3).
#[test]
fn dt_2ms_invariance_and_staircase_warning() {
    let mut cfg = rich(7, [12, 10, 96], None, 1);
    cfg.rock_physics.depth_step_m = 2.0;
    cfg.time.dt_ms = 2.0;
    assert_eq!(cfg.output_samples(), 96);
    assert_eq!(cfg.digi_ms(), 2.0);
    assert!(cfg.time_warning().is_none(), "{:?}", cfg.time_warning());
    assert_tiling_invariant(&cfg, &[[1, 1, 0], [5, 7, 0], [3, 10, 16]]);

    let coarse = E2eConfig {
        rock_physics: RockPhysicsConfig { depth_step_m: 4.0, ..cfg.rock_physics.clone() },
        ..cfg.clone()
    };
    let w = coarse.time_warning().expect("dz = 4 m at dt = 2 ms must warn");
    assert!(w.contains("depth step 4 m") && w.contains("2.63"), "{w}");
    assert_eq!(coarse.output_samples(), 192);
    // The legacy axis never warns.
    assert!(E2eConfig { time: TimeConfig::legacy(), ..coarse }.time_warning().is_none());
}

/// Label round trip (spec §5.5) for labels, fault labels and salt labels,
/// with the fault-label salt mask on (default) and off
/// (`--fault-labels-through-salt`):
///
/// * forward, every output sample (asserted): `n` takes the label of the
///   depth cell that contains `t_n`, `k(n) = max{k : T_k ≤ t_n}` (clamped to
///   the last cell), computed here from the column's T. All cubes use the
///   same `k(n)`, so fault ∧ salt in time equals fault ∧ salt at k(n) on
///   every sample: 0 everywhere with the mask, and the depth overlap
///   carried over sample for sample without it;
/// * seabed, exact (asserted): an output sample is 255 if and only if
///   `t_n < T_sb`, the two-way time of the seabed;
/// * no class is invented (asserted);
/// * back to depth (reported, not gated): each depth cell k inside the trace
///   reads the output sample nearest its centre time; the recovered
///   fraction, and the label runs thinner than dt that no output sample
///   lands in, are printed. Under point sampling every cell at least dt
///   thick is recovered by construction, so the fraction measures run
///   thickness against dt, not the code.
#[test]
fn label_round_trip() {
    for (seed, nt) in [(30u64, None), (11, Some(64 + 37))] {
        for through in [false, true] {
            let mut cfg = rich(seed, [16, 20, 64], nt, 1);
            cfg.rock_physics.fault_labels_through_salt = through;
            let (labels, shape) = generate_labels(&cfg);
            let [ni, nj, nz] = shape;
            let model = elastic_model(&cfg, &labels, shape);
            let out = generate_output_labels(&cfg, &labels, &model);
            let faults_z = generate_fault_labels(&cfg).unwrap();
            let salt_z = synthoseis_core::salt::generate_salt_labels(&cfg).expect("salt");
            let axis = cfg.time_axis().unwrap();
            let (nt, dt) = (axis.nt, axis.dt_ms);
            let t = tile_twt(&model, &labels, shape, 0, ni, 0, nj, &axis);
            let (fo, so) = (out.faults.as_ref().unwrap(), out.salt.as_ref().unwrap());
            let (mut cells, mut recovered, mut seabeds) = (0usize, 0usize, 0usize);
            let (mut runs, mut thin_dropped, mut both_t, mut both_z) = (0usize, 0usize, 0usize, 0usize);
            for c in 0..ni * nj {
                let tc = &t[c * (nz + 1)..(c + 1) * (nz + 1)];
                let (z, o) = (c * nz, c * nt);
                let cubes: [(&[u8], &[u8], &str); 3] = [
                    (&labels[z..z + nz], &out.labels[o..o + nt], "labels"),
                    (&faults_z[z..z + nz], &fo[o..o + nt], "fault_labels"),
                    (&salt_z[z..z + nz], &so[o..o + nt], "salt_labels"),
                ];
                // Forward: every output sample, k(n) by a two-pointer walk.
                let mut k = 0usize;
                for n in 0..nt {
                    let tn = n as f64 * dt;
                    while k + 1 < nz && tc[k + 1] <= tn {
                        k += 1;
                    }
                    for (zc, oc, what) in &cubes {
                        assert_eq!(oc[n], zc[k], "seed {seed} through {through} col {c} n {n}: {what} vs cell {k}");
                    }
                    let bt = fo[o + n] == 1 && so[o + n] == 1;
                    assert_eq!(bt, faults_z[z + k] == 1 && salt_z[z + k] == 1, "fault ∧ salt col {c} n {n}");
                    assert!(through || !bt, "seed {seed} col {c} n {n}: fault ∧ salt with the mask on");
                    both_t += bt as usize;
                }
                // Back to depth, reported.
                for k in 0..nz {
                    let centre = 0.5 * (tc[k] + tc[k + 1]);
                    let n = (centre / dt).round() as usize;
                    if n >= nt {
                        break;
                    }
                    cells += 1;
                    recovered += cubes.iter().all(|(zc, oc, _)| oc[n] == zc[k]) as usize;
                    both_z += (faults_z[z + k] == 1 && salt_z[z + k] == 1) as usize;
                }
                // Label runs (of the layer labels) no output sample lands in.
                let mut k0 = 0usize;
                while k0 < nz {
                    let mut k1 = k0 + 1;
                    while k1 < nz && labels[z + k1] == labels[z + k0] {
                        k1 += 1;
                    }
                    if tc[k0] <= (nt - 1) as f64 * dt {
                        runs += 1;
                        let hit = (0..nt).any(|n| {
                            let tn = n as f64 * dt;
                            tn >= tc[k0] && (tn < tc[k1] || k1 == nz)
                        });
                        thin_dropped += !hit as usize;
                    }
                    k0 = k1;
                }
                for (zc, oc, what) in &cubes {
                    assert!(oc.iter().all(|v| zc.contains(v)), "seed {seed} col {c}: {what}: invented class");
                }
                // Seabed: 255 exactly where t_n < T_sb.
                let ol = &out.labels[o..o + nt];
                if let Some(ksb) = labels[z..z + nz].iter().position(|&v| v != 255) {
                    let t_sb = tc[ksb];
                    for (n, &v) in ol.iter().enumerate() {
                        assert_eq!(v == 255, (n as f64 * dt) < t_sb, "seed {seed} col {c} n {n}: seabed at {t_sb} ms");
                    }
                    seabeds += 1;
                }
            }
            println!(
                "seed {seed} nt {nt} through-salt {through}: {recovered}/{cells} depth cells recovered ({:.2} %, reported); {thin_dropped}/{runs} layer runs thinner than dt dropped; fault ∧ salt: {both_t} output samples, {both_z} depth cells; seabed exact in {seabeds} columns",
                100.0 * recovered as f64 / cells as f64
            );
            assert_eq!(seabeds, ni * nj);
            if !through {
                assert_eq!(both_z, 0, "#38: fault ∧ salt = 0 in depth with the mask");
            }
        }
    }
}

/// Dead last sample (spec §3.7): with the bandpass replacing the Ricker the
/// last output sample `nt − 1` is exactly 0 on every trace, and it is the
/// only sample the zeroing touches (with the zeroing skipped through the
/// test hook, samples `0 … nt − 2` are bit-identical and the last sample
/// carries the filtered trace's value). nt = nz and nz + 37, sinc and
/// linear insertion; the production chunked path matches.
#[test]
fn dead_last_sample() {
    for nt in [None, Some(64 + 37)] {
        for kernel in [TwtKernel::Sinc, TwtKernel::Linear] {
            let mut cfg = rich(30, [12, 10, 64], nt, 3);
            cfg.faults = FaultConfig::default();
            cfg.time.kernel = kernel;
            let (labels, shape) = generate_labels(&cfg);
            let [ni, nj, _] = shape;
            let n = cfg.output_samples();
            let model = elastic_model(&cfg, &labels, shape);
            let f = SeismicFilters::resolve(&cfg, &labels, shape).unwrap().unwrap();
            let mut g = f.clone();
            g.zero_trailing_sample = false;
            let run = |f: &SeismicFilters| {
                let mut out = vec![0.0f32; ni * nj * n];
                fuse_tile_filtered(
                    &labels,
                    shape,
                    0,
                    ni,
                    0,
                    nj,
                    &model,
                    &cfg.ricker(),
                    DEFAULT_INCIDENCE_DEG,
                    Some(f),
                    &mut out,
                    &mut WorkingSetStats::default(),
                );
                out
            };
            let (on, off) = (run(&f), run(&g));
            assert!(on.chunks_exact(n).all(|t| t[n - 1].to_bits() == 0), "{kernel:?} nt {n}: last sample");
            for (a, b) in on.chunks_exact(n).zip(off.chunks_exact(n)) {
                assert_eq!(bits(&a[..n - 1]), bits(&b[..n - 1]), "{kernel:?} nt {n}: 0..nt-2");
            }
            assert!(off.chunks_exact(n).any(|t| t[n - 1] != 0.0), "{kernel:?} nt {n}: hook inert");
            let (v, _) = generate_chunked(&E2eConfig { chunk_shape: Some([5, 4, n]), ..cfg.clone() });
            assert_eq!(bits(&v.angle_stack), bits(&on), "{kernel:?} nt {n}: production path");
        }
    }
}

/// At a uniform 2000 m/s (the legacy axis' implied velocity, through the
/// constant-velocity test hook) the time-mode raw reflectivity is the
/// legacy depth-fuse reflectivity moved down one sample, bit for bit: the
/// interface between cells k and k+1 sits at T = 4 (k + 1) ms (the toy
/// model; the legacy Python fixture is in `angle_stack_legacy_e2e.rs`).
#[test]
fn uniform_2000_time_reflectivity_is_the_depth_fuse_one_sample_down() {
    for (shape, seed) in [([12, 10, 64], 30u64), ([8, 8, 128], 7)] {
        let depth = E2eConfig {
            seed,
            inline_count: shape[0],
            crossline_count: shape[1],
            samples: shape[2],
            faults: FaultConfig::with_count(2),
            time: TimeConfig::legacy(),
            ..E2eConfig::default()
        };
        let time = E2eConfig {
            time: TimeConfig { constant_twt_vp: Some(2000.0), ..TimeConfig::default() },
            ..depth.clone()
        };
        let nz = shape[2];
        for angle in [0.0, 15.0, 30.0] {
            let d = generate_reflectivity(&depth, angle);
            let t = generate_reflectivity(&time, angle);
            assert_eq!(d.len(), t.len());
            let mut nonzero = 0;
            for (dc, tc) in d.chunks_exact(nz).zip(t.chunks_exact(nz)) {
                assert_eq!(tc[0].to_bits(), 0);
                assert_eq!(bits(&tc[1..]), bits(&dc[..nz - 1]), "seed {seed} {angle}°");
                assert_eq!(dc[nz - 1], 0.0);
                nonzero += dc.iter().filter(|&&x| x != 0.0).count();
            }
            assert!(nonzero > 0);
        }
        // Labels at 2000 m/s are the depth labels (identity, spec §3.4).
        let (lz, sh) = generate_labels(&time);
        let model = elastic_model(&time, &lz, sh);
        assert_eq!(generate_output_labels(&time, &lz, &model).labels, lz);
    }
}

/// Per-tile cost gate (spec §6): the time-mode fuse tile (T, Zoeppritz in
/// depth, sinc insertion, Ricker on the time trace) costs at most 1.3× the
/// legacy depth fuse tile of the same model. Best of several repetitions.
#[test]
fn time_mode_fuse_tile_cost_within_1_3x() {
    let shape = [32, 32, 256];
    let legacy = E2eConfig {
        seed: 7,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        time: TimeConfig::legacy(),
        ..E2eConfig::default()
    };
    let time = E2eConfig { time: TimeConfig::default(), ..legacy.clone() };
    let (labels, sh) = generate_labels(&legacy);
    let time_tile = |cfg: &E2eConfig| {
        let model = elastic_model(cfg, &labels, sh);
        let w = cfg.ricker();
        let tile = 16;
        let mut out = vec![0.0f32; tile * tile * cfg.output_samples()];
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            for i0 in (0..shape[0]).step_by(tile) {
                for j0 in (0..shape[1]).step_by(tile) {
                    fuse_tile_local(
                        &labels,
                        sh,
                        i0,
                        i0 + tile,
                        j0,
                        j0 + tile,
                        &model,
                        &w,
                        DEFAULT_INCIDENCE_DEG,
                        &mut out,
                        &mut WorkingSetStats::default(),
                    );
                }
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        best
    };
    // Interleave and keep the best ratio of three rounds (CI noise).
    let mut ratio = f64::INFINITY;
    for _ in 0..3 {
        let (d, t) = (time_tile(&legacy), time_tile(&time));
        ratio = ratio.min(t / d);
        println!("fuse tile: depth {:.3} ms, time {:.3} ms, ratio {:.3}", d * 1e3, t * 1e3, t / d);
    }
    assert!(ratio <= 1.3, "time-mode fuse tile {ratio:.3}x the depth tile (gate 1.3x)");
}

/// The legacy switch (`TimeConfig::legacy()`, CLI `--legacy-depth-as-time`)
/// reproduces master f3720fb2 bit for bit: angle stack, labels and fault
/// labels of the 64 × 64 × 256 demo cube, plain, with bandpass + noise, with
/// 3 faults, and both, plus 3 faults with `--fault-labels-through-salt`.
/// Hashes from the master f3720fb2 library (same configs,
/// `generate_chunked`). #38 changed only the fault labels against
/// 0eb937b5: masked by the salt (`FAULTS`); through-salt equals 0eb937b5's
/// (`FAULTS_THROUGH`). The time-mode default differs. Also with master
/// bad1daa8's fixed 500-voxel closure minimum (`ClosureMinimum::LEGACY`,
/// closure-minimum spec §6.9): f3720fb2 predates the scaled minimum.
#[test]
fn legacy_depth_as_time_reproduces_master_f3720fb2() {
    let mut demo = whole(&E2eConfig {
        seed: 7,
        inline_count: 64,
        crossline_count: 64,
        samples: 256,
        time: TimeConfig::legacy(),
        ..E2eConfig::default()
    });
    demo.rock_physics.closure_minimum = synthoseis_core::ClosureMinimum::LEGACY;
    let mut f = FilterConfig::legacy(4.0, 30.0, 3);
    f.noise = NoiseConfig { snr_db: Some(12.5), ..NoiseConfig::default() };
    let bp = E2eConfig { filters: f, ..demo.clone() };
    let faulted = E2eConfig { faults: FaultConfig::with_count(3), ..demo.clone() };
    let faulted_bp = E2eConfig { faults: FaultConfig::with_count(3), ..bp.clone() };
    let through = through_salt(&faulted);
    const LABELS: u64 = 0x1960_f999_e466_ce51;
    const FAULTED_LABELS: u64 = 0x3737_d29c_cb8c_0876;
    const FAULTS: u64 = 0xfe82_57da_8c61_a79a;
    const FAULTS_THROUGH: u64 = 0xc6f1_d8cd_5672_af47;
    for (name, cfg, angle, labels, faults) in [
        ("demo", demo, 0x8542_0900_0696_82e2u64, LABELS, None),
        ("demo_bp_noise", bp, 0xd675_c078_55ec_b323, LABELS, None),
        ("demo_faults3", faulted, 0xd639_47b9_3028_52f7, FAULTED_LABELS, Some(FAULTS)),
        ("demo_faults3_bp_noise", faulted_bp, 0x03f7_c415_cd6b_f911, FAULTED_LABELS, Some(FAULTS)),
        ("demo_faults3_through_salt", through, 0xd639_47b9_3028_52f7, FAULTED_LABELS, Some(FAULTS_THROUGH)),
    ] {
        let (v, _) = generate_chunked(&cfg);
        assert_eq!(fnv_f32(&v.angle_stack), angle, "{name}: angle_stack");
        assert_eq!(fnv(v.labels.iter().copied()), labels, "{name}: labels");
        assert_eq!(generate_fault_labels(&cfg).map(|m| fnv(m)), faults, "{name}: fault_labels");
        let t = E2eConfig { time: TimeConfig::default(), ..cfg };
        let (w, _) = generate_chunked(&t);
        assert_ne!(fnv_f32(&w.angle_stack), angle, "{name}: time mode must differ");
    }
}
