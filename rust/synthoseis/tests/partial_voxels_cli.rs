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
const DEFAULT_PLAIN: u64 = 0x82bf_e625_85ee_8c92;
const DEFAULT_PLAIN_LEGACY_AXIS: u64 = 0xa2f4_7082_3b85_5563;
const DEFAULT_MP: u64 = 0x28e5_1787_5502_bb1d;
const WHOLE_PLAIN: u64 = 0x6420_9caa_660f_d978;
const WHOLE_MP: u64 = 0x28e5_1787_5502_bb1d;

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
        let out = run(&base, &d);
        assert!(out.status.success(), "{name}: {out:?}");
        let so = stdout(&out);
        assert!(
            so.contains("partial voxels: mixed=") && so.contains(&format!("mode={mode}")),
            "{name}: {so}"
        );
        assert_eq!(attrs(&d)["voxel_model"], "partial-z", "{name}");
        assert_eq!(attrs(&d)["partial_voxel_reflectivity"], mode, "{name}");
        assert_eq!(store_hash(&d), pinned, "{name}: new default");
        let e = dir.path().join(format!("{name}-explicit.mdio"));
        assert!(run(
            &with(&base, &["--partial-voxel-reflectivity", explicit]),
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
        let out = run(&with(&base, &["--legacy-whole-voxels"]), &l);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(
            !stdout(&out).contains("partial voxels:"),
            "{name}: {}",
            stdout(&out)
        );
        for k in PV_ATTRS {
            assert!(attrs(&l).get(k).is_none(), "{name}: opt-out store has {k}");
        }
        // On the tiny multi-process 8³ cube salt/partial may not move the
        // stack enough to differ; the plain / legacy-axis cases do.
        if name != "mp" {
            assert_ne!(store_hash(&l), store_hash(&d), "{name}: opt-out vs default");
        }
        if let Some(w) = whole {
            assert_eq!(
                store_hash(&l),
                w,
                "{name}: --legacy-whole-voxels = d8b96e69 default"
            );
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



