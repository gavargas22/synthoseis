//! CLI partial-voxel switches (spec "partial voxels" PR B1/B2): on by default
//! since B2 (subcell in time mode, cell on the legacy axis), equal to the
//! explicit `--partial-voxel-reflectivity`; `--legacy-whole-voxels` opts out
//! (the d8b96e69 default, pinned). The switch and the opt-out reach
//! multi-process workers (bit-identical to one process) and strip-stitch;
//! the three root attributes and the summary line appear only when on;
//! invalid combinations exit 2.
use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], store: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthoseis"))
        .args(["run", "--e2e"])
        .args(args)
        .arg("--store")
        .arg(store)
        .output()
        .expect("spawn synthoseis")
}

fn with<'a>(base: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    base.iter().chain(extra).copied().collect()
}

/// FNV-1a over the stored angle stack and label cubes.
fn store_hash(p: &Path) -> u64 {
    let s = synthoseis_io::MdioStore::open(p).unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for x in s.read_volume().unwrap() {
        x.to_bits().to_le_bytes().into_iter().for_each(&mut eat);
    }
    s.read_labels_u8().unwrap().into_iter().for_each(&mut eat);
    h
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const PV_ATTRS: [&str; 3] = [
    "voxel_model",
    "partial_voxel_mixing",
    "partial_voxel_reflectivity",
];
const PLAIN: &[&str] = &["--chunked", "--shape", "12,10,64", "--chunk-i", "5"];
const MP: &[&str] = &[
    "--chunked",
    "--multiprocess",
    "--workers",
    "3",
    "--chunk-i",
    "2",
    "--chunk-j",
    "4",
    "--chunk-k",
    "8",
];

// Pinned hashes of `store_hash` (angle stack + labels). The new defaults
// equal master d8b96e69 run with the switch on (`--partial-voxel-reflectivity
// subcell`, and `cell` with `--legacy-depth-as-time`); the opt-out equals
// master d8b96e69's default (PR B2 evidence, whole store trees `diff -r`).
//
// Time-mode rows moved with the physical filter edges (filter-edge spec
// §6: the last 7 samples of columns that continue below the window); the
// `_1C22` values are master 1c22b653's, reproduced with
// `--legacy-filter-edges` (plus `--legacy-salt-pv-maps` for the
// partial-voxel defaults). The labels part of the hash does not move with
// the filter edges.
//
// `DEFAULT_*` moved again with the continuous salt partial-voxel maps
// (salt pv-maps spec §7: stack, and in time mode the time labels and
// `salt_labels`); master a60ec34a's values are `*_A60E` and are
// reproduced with `--legacy-salt-pv-maps`.
//
// `DEFAULT_*` and `WHOLE_*` moved again with the physical rock physics
// (gassmann spec §7); master 77131867's values are `*_7713` and are
// reproduced with `--legacy-rock-physics`.
//
// `DEFAULT_*` and `WHOLE_*` moved again with the analytic folds (folds spec
// §7); master 7e80f2af's values are `*_7E80` and are reproduced with
// `--legacy-fold-geometry`.
const DEFAULT_PLAIN: u64 = 0xa50f_4bfa_7724_ad01;
const DEFAULT_PLAIN_LEGACY_AXIS: u64 = 0xac88_b6e9_d9da_fb24;
const DEFAULT_MP: u64 = 0xc112_78b0_86f7_37bb;
const DEFAULT_PLAIN_7E80: u64 = 0xda29_ca4c_fa9b_a276;
const DEFAULT_PLAIN_LEGACY_AXIS_7E80: u64 = 0x8033_d531_528f_a878;
const DEFAULT_MP_7E80: u64 = 0xf054_6bf3_765d_4902;
const WHOLE_PLAIN_7E80: u64 = 0x3546_0607_cdc9_7b13;
const WHOLE_MP_7E80: u64 = 0xbe7e_6b4e_d66e_e92c;
const DEFAULT_PLAIN_7713: u64 = 0xd59f_0b05_6072_4ef4;
const DEFAULT_PLAIN_LEGACY_AXIS_7713: u64 = 0x526a_50f7_562b_04f3;
const DEFAULT_MP_7713: u64 = 0xcf00_54b2_6533_c33a;
const WHOLE_PLAIN_7713: u64 = 0x38d8_efe5_ef93_16a4;
const WHOLE_MP_7713: u64 = 0x91c3_c68d_c372_80f1;
const DEFAULT_PLAIN_A60E: u64 = 0xc5f1_6ffa_eed0_333e;
const DEFAULT_PLAIN_LEGACY_AXIS_A60E: u64 = 0xa2f4_7082_3b85_5563;
const DEFAULT_MP_A60E: u64 = 0x91c3_c68d_c372_80f1;
const WHOLE_PLAIN: u64 = 0x3df7_8a73_2d58_4380;
const WHOLE_MP: u64 = 0xe94f_c577_84e8_d2e5;
const DEFAULT_PLAIN_1C22: u64 = 0x82bf_e625_85ee_8c92;
const DEFAULT_MP_1C22: u64 = 0x28e5_1787_5502_bb1d;
const WHOLE_PLAIN_1C22: u64 = 0x6420_9caa_660f_d978;
const WHOLE_MP_1C22: u64 = 0x28e5_1787_5502_bb1d;

#[test]
fn default_is_partial_and_legacy_whole_voxels_opts_out() {
    let dir = tempfile::tempdir().expect("tempdir");
    let attrs = |p: &Path| {
        synthoseis_io::MdioStore::open(p)
            .unwrap()
            .root_attrs()
            .unwrap()
    };
    for (name, base, explicit, mode, pinned, whole) in [
        (
            "time",
            PLAIN.to_vec(),
            "subcell",
            "subcell",
            DEFAULT_PLAIN,
            Some(WHOLE_PLAIN),
        ),
        (
            "legacy-axis",
            with(PLAIN, &["--legacy-depth-as-time"]),
            "cell",
            "cell",
            DEFAULT_PLAIN_LEGACY_AXIS,
            None,
        ),
        (
            "mp",
            MP.to_vec(),
            "subcell",
            "subcell",
            DEFAULT_MP,
            Some(WHOLE_MP),
        ),
    ] {
        let d = dir.path().join(format!("{name}-default.mdio"));
        // Folded geometry of #56 (unconformity spec §6.3): isolated, no re-pin.
        let out = run(&with(&base, &["--legacy-unconformity"]), &d);
        assert!(out.status.success(), "{name}: {out:?}");
        let so = stdout(&out);
        assert!(
            so.contains("partial voxels: mixed=") && so.contains(&format!("mode={mode}")),
            "{name}: {so}"
        );
        assert!(so.contains("salt partial-voxel maps: continuous (clamped to the label cell)\n"), "{name}: {so}");
        assert_eq!(attrs(&d)["voxel_model"], "partial-z", "{name}");
        assert_eq!(attrs(&d)["partial_voxel_reflectivity"], mode, "{name}");
        assert_eq!(store_hash(&d), pinned, "{name}: new default {:#018x}", store_hash(&d));
        // `--legacy-fold-geometry` restores master 7e80f2af's default.
        let (d7e80, w7e80) = match name {
            "time" => (DEFAULT_PLAIN_7E80, Some(WHOLE_PLAIN_7E80)),
            "legacy-axis" => (DEFAULT_PLAIN_LEGACY_AXIS_7E80, None),
            _ => (DEFAULT_MP_7E80, Some(WHOLE_MP_7E80)),
        };
        let r = dir.path().join(format!("{name}-legacy-fold.mdio"));
        let out = run(&with(&base, &["--legacy-fold-geometry"]), &r);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(stdout(&out).contains("folds: off (--legacy-fold-geometry)\n"), "{name}");
        assert_eq!(store_hash(&r), d7e80, "{name}: --legacy-fold-geometry = 7e80f2af");
        assert_ne!(store_hash(&r), store_hash(&d), "{name}: the folds move the default");
        if let Some(w) = w7e80 {
            let r = dir.path().join(format!("{name}-legacy-fold-whole.mdio"));
            assert!(run(&with(&base, &["--legacy-fold-geometry", "--legacy-whole-voxels"]), &r).status.success());
            assert_eq!(store_hash(&r), w, "{name}: --legacy-fold-geometry --legacy-whole-voxels = 7e80f2af");
        }
        // `--legacy-rock-physics` restores master 77131867's default.
        let (d7713, w7713) = match name {
            "time" => (DEFAULT_PLAIN_7713, Some(WHOLE_PLAIN_7713)),
            "legacy-axis" => (DEFAULT_PLAIN_LEGACY_AXIS_7713, None),
            _ => (DEFAULT_MP_7713, Some(WHOLE_MP_7713)),
        };
        let r = dir.path().join(format!("{name}-legacy-rp.mdio"));
        let out = run(&with(&base, &["--legacy-rock-physics", "--legacy-fold-geometry"]), &r);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(
            stdout(&out).contains("rock physics: legacy trends (--legacy-rock-physics)\n"),
            "{name}"
        );
        assert_eq!(
            store_hash(&r),
            d7713,
            "{name}: --legacy-rock-physics = 77131867"
        );
        assert_ne!(store_hash(&r), store_hash(&d), "{name}: the default moves");
        if let Some(w) = w7713 {
            let r = dir.path().join(format!("{name}-legacy-rp-whole.mdio"));
            assert!(run(
                &with(&base, &["--legacy-rock-physics", "--legacy-fold-geometry", "--legacy-whole-voxels"]),
                &r
            )
            .status
            .success());
            assert_eq!(
                store_hash(&r),
                w,
                "{name}: --legacy-rock-physics --legacy-whole-voxels = 77131867"
            );
        }
        // `--legacy-salt-pv-maps` (+ `--legacy-rock-physics`) restores master
        // a60ec34a's default.
        let a60e = match name {
            "time" => DEFAULT_PLAIN_A60E,
            "legacy-axis" => DEFAULT_PLAIN_LEGACY_AXIS_A60E,
            _ => DEFAULT_MP_A60E,
        };
        let r = dir.path().join(format!("{name}-rounded-maps.mdio"));
        let out = run(
            &with(&base, &["--legacy-salt-pv-maps", "--legacy-rock-physics", "--legacy-fold-geometry"]),
            &r,
        );
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(stdout(&out).contains("salt partial-voxel maps: rounded (--legacy-salt-pv-maps)\n"), "{name}");
        assert_eq!(store_hash(&r), a60e, "{name}: --legacy-salt-pv-maps = a60ec34a {:#018x}", store_hash(&r));
        assert_ne!(store_hash(&r), store_hash(&d), "{name}: the default moves");
        let e = dir.path().join(format!("{name}-explicit.mdio"));
        assert!(run(
            &with(&base, &["--legacy-unconformity", "--partial-voxel-reflectivity", explicit]),
            &e
        )
        .status
        .success());
        assert_eq!(
            store_hash(&e),
            store_hash(&d),
            "{name}: default = explicit {explicit}"
        );
        let l = dir.path().join(format!("{name}-whole.mdio"));
        let out = run(&with(&base, &["--legacy-unconformity", "--legacy-whole-voxels"]), &l);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(
            !stdout(&out).contains("partial voxels:") && !stdout(&out).contains("salt partial-voxel maps"),
            "{name}: {}",
            stdout(&out)
        );
        for k in PV_ATTRS {
            assert!(attrs(&l).get(k).is_none(), "{name}: opt-out store has {k}");
        }
        // Every case differs, including the 8³ multi-process cube: its
        // salt body has 0 voxels inside the cube, and before the
        // continuous salt partial-voxel maps that snapped every horizon to
        // whole samples (default == whole voxels there).
        assert_ne!(store_hash(&l), store_hash(&d), "{name}: opt-out vs default");
        if let Some(w) = whole {
            assert_eq!(
                store_hash(&l),
                w,
                "{name}: --legacy-whole-voxels (physical filter edges) {:#018x}",
                store_hash(&l)
            );
        }
        // `--legacy-filter-edges` restores master 1c22b653 (whose
        // `--legacy-whole-voxels` is the d8b96e69 default) in time mode.
        let legacy_edges = match name {
            "time" => Some((DEFAULT_PLAIN_1C22, WHOLE_PLAIN_1C22)),
            "mp" => Some((DEFAULT_MP_1C22, WHOLE_MP_1C22)),
            _ => None,
        };
        if let Some((d_1c22, w_1c22)) = legacy_edges {
            for (extra, want) in [(&[][..], d_1c22), (&["--legacy-whole-voxels"][..], w_1c22)] {
                let p = dir.path().join(format!("{name}-1c22.mdio"));
                // 1c22b653 also had the rounded salt partial-voxel maps.
                let pv: &[&str] = if extra.is_empty() { &["--legacy-salt-pv-maps"] } else { &[] };
                let args = with(
                    &with(
                        &with(&base, extra),
                        &["--legacy-filter-edges", "--legacy-rock-physics", "--legacy-fold-geometry"],
                    ),
                    pv,
                );
                assert!(run(&args, &p).status.success(), "{name} {extra:?}");
                assert_eq!(store_hash(&p), want, "{name} {extra:?}: --legacy-filter-edges = 1c22b653");
            }
        }
    }
    // Planar is whole-voxel by construction: no attrs, no summary line.
    let p = dir.path().join("planar.mdio");
    let out = run(&with(PLAIN, &["--toy-geometry", "planar"]), &p);
    assert!(out.status.success(), "{out:?}");
    assert!(!stdout(&out).contains("partial voxels:"));
    for k in PV_ATTRS {
        assert!(attrs(&p).get(k).is_none(), "planar store has {k}");
    }
}

#[test]
fn partial_on_attrs_summary_and_modes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut hashes = Vec::new();
    for (name, args, mode) in [
        (
            "subcell",
            with(PLAIN, &["--partial-voxel-reflectivity", "subcell"]),
            "subcell",
        ),
        (
            "cell",
            with(PLAIN, &["--partial-voxel-reflectivity", "cell"]),
            "cell",
        ),
        (
            "legacy-axis-cell",
            with(
                PLAIN,
                &[
                    "--legacy-depth-as-time",
                    "--partial-voxel-reflectivity",
                    "cell",
                ],
            ),
            "cell",
        ),
        (
            "overlap",
            vec![
                "--chunked",
                "--overlap",
                "--chunk-i",
                "4",
                "--partial-voxel-reflectivity",
                "subcell",
            ],
            "subcell",
        ),
        (
            "strip",
            vec![
                "--chunked",
                "--workers",
                "3",
                "--chunk-i",
                "2",
                "--partial-voxel-reflectivity",
                "subcell",
            ],
            "subcell",
        ),
        (
            "classic",
            vec!["--partial-voxel-reflectivity", "cell"],
            "cell",
        ),
    ] {
        let p = dir.path().join(format!("{name}.mdio"));
        let out = run(&args, &p);
        assert!(out.status.success(), "{name}: {out:?}");
        let so = stdout(&out);
        assert!(
            so.contains("partial voxels: mixed=") && so.contains(&format!("mode={mode}")),
            "{name}: {so}"
        );
        let attrs = synthoseis_io::MdioStore::open(&p)
            .unwrap()
            .root_attrs()
            .unwrap();
        assert_eq!(attrs["voxel_model"], "partial-z", "{name}");
        assert_eq!(attrs["partial_voxel_mixing"], "backus", "{name}");
        assert_eq!(attrs["partial_voxel_reflectivity"], mode, "{name}");
        hashes.push(store_hash(&p));
    }
    let off = dir.path().join("off.mdio");
    assert!(run(&with(PLAIN, &["--legacy-whole-voxels"]), &off)
        .status
        .success());
    assert_ne!(hashes[0], store_hash(&off), "subcell vs off");
    assert_ne!(hashes[1], store_hash(&off), "cell vs off");
    assert_ne!(hashes[0], hashes[1], "subcell vs cell");
}

/// The switch and the opt-out reach multi-process workers: bit-identical to
/// one process with the same chunking, for the default, S, C, C on the
/// legacy axis, the legacy-axis default and `--legacy-whole-voxels`.
#[test]
fn partial_flag_reaches_multiprocess_workers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let single_base: Vec<&str> = MP
        .iter()
        .copied()
        .filter(|a| !["--multiprocess", "--workers", "3"].contains(a))
        .collect();
    for flags in [
        &[][..],
        &["--legacy-whole-voxels"][..],
        &["--legacy-depth-as-time"][..],
        &["--legacy-depth-as-time", "--legacy-whole-voxels"][..],
        &["--partial-voxel-reflectivity", "subcell"][..],
        &["--partial-voxel-reflectivity", "cell"][..],
        &[
            "--legacy-depth-as-time",
            "--partial-voxel-reflectivity",
            "cell",
        ][..],
    ] {
        let single = dir.path().join("single.mdio");
        let out = run(&with(&single_base, flags), &single);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        let mp = dir.path().join("mp.mdio");
        let out = run(&with(MP, flags), &mp);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        assert_eq!(
            store_hash(&single),
            store_hash(&mp),
            "{flags:?}: multiprocess vs single"
        );
        let attrs = synthoseis_io::MdioStore::open(&mp)
            .unwrap()
            .root_attrs()
            .unwrap();
        if flags.contains(&"--legacy-whole-voxels") {
            assert!(attrs.get("voxel_model").is_none(), "{flags:?}");
        } else {
            assert_eq!(attrs["voxel_model"], "partial-z", "{flags:?}");
        }
    }
}

#[test]
fn invalid_partial_flags_exit_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bad, msg) in [
        (
            ["--legacy-whole-voxels", "--toy-geometry", "planar"].as_slice(),
            "--legacy-whole-voxels has no effect with --toy-geometry planar",
        ),
        (
            &["--legacy-whole-voxels", "--legacy-toy-depth"],
            "--legacy-whole-voxels has no effect",
        ),
        (
            &[
                "--legacy-whole-voxels",
                "--partial-voxel-reflectivity",
                "cell",
            ],
            "--partial-voxel-reflectivity has no effect with --legacy-whole-voxels",
        ),
        (
            &[
                "--partial-voxel-reflectivity",
                "cell",
                "--toy-geometry",
                "planar",
            ],
            "requires the layered geometry",
        ),
        (
            &[
                "--partial-voxel-reflectivity",
                "subcell",
                "--legacy-depth-as-time",
            ],
            "subcell requires the time axis",
        ),
        (
            &["--partial-voxel-reflectivity", "exact"],
            "--partial-voxel-reflectivity",
        ),
    ] {
        let out = run(&with(PLAIN, bad), &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(stderr(&out).contains(msg), "{bad:?}: {}", stderr(&out));
    }
}



