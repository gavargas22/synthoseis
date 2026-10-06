//! CLI closure minimum (closure-minimum spec §5, §6.7, §6.8):
//! `--legacy-closure-minimum`, `--min-closure-voxels N`,
//! `--legacy-closure-contact-cap`, the summary line, the root attributes,
//! forwarding to multi-process workers, and byte identity with master
//! bad1daa8 under both legacy flags.
use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], store: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthoseis"))
        .args(["run", "--e2e", "--chunked"])
        .args(args)
        .arg("--store")
        .arg(store)
        .output()
        .expect("spawn synthoseis")
}

/// FNV-1a over every file of a store (sorted relative paths and contents),
/// ignoring the `"created"` timestamp lines of `.zattrs` / `.zmetadata`.
fn store_dir_hash(root: &Path) -> u64 {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    files.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for f in files {
        let rel = f.strip_prefix(root).unwrap().to_string_lossy().into_owned();
        eat(rel.as_bytes());
        let bytes = std::fs::read(&f).unwrap();
        if rel.ends_with(".zattrs") || rel.ends_with(".zmetadata") {
            let text = String::from_utf8(bytes).unwrap();
            for line in text.lines().filter(|l| !l.contains("\"created\"")) {
                eat(line.as_bytes());
            }
        } else {
            eat(&bytes);
        }
    }
    h
}

fn volume_bits(p: &Path) -> Vec<u32> {
    synthoseis_io::MdioStore::open(p)
        .unwrap()
        .read_volume()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn zattrs(p: &Path) -> String {
    std::fs::read_to_string(p.join(".zattrs")).unwrap()
}

/// Demo cube (time mode, partial voxels, salt, 3 faults): 2 of 3
/// compartments kept at T = 20, none at 500.
const DEMO: &[&str] = &["--seed", "7", "--shape", "32,32,128", "--faults", "3"];
/// `store_dir_hash` of the `DEMO` store written by a master bad1daa8
/// binary (default flags).
const MASTER_BAD1DAA8_DEMO: u64 = 0x99e8_6ae5_7a12_b80c;
/// `store_dir_hash` of the `DEMO` store with the new defaults (scaled
/// minimum, contact at base + ½): the regression pin for the default.
const SCALED_DEFAULT_DEMO: u64 = 0x127f_83c0_2d88_af2b;

/// §6.8: both legacy flags reproduce the bad1daa8 default byte for byte
/// (every array and attribute); the default differs and is pinned.
#[test]
fn legacy_closure_flags_reproduce_master_bad1daa8() {
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join("legacy.mdio");
    let out = run(
        &[
            DEMO,
            &["--legacy-closure-minimum", "--legacy-closure-contact-cap"],
        ]
        .concat(),
        &legacy,
    );
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("closures: minimum 500 voxels (fixed 500, --legacy-closure-minimum), kept 0 of 3 compartments, contact cap at base cell (--legacy-closure-contact-cap)"),
        "{stdout}"
    );
    assert_eq!(
        store_dir_hash(&legacy),
        MASTER_BAD1DAA8_DEMO,
        "{:#018x}",
        store_dir_hash(&legacy)
    );
    assert!(!zattrs(&legacy).contains("closure_min"));
    let new = dir.path().join("default.mdio");
    let out = run(DEMO, &new);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("closures: minimum 20 voxels (scaled ni·nj/180, floor 20, cap 500), kept 2 of 3 compartments\n"),
        "{stdout}"
    );
    let a = zattrs(&new);
    assert!(
        a.contains("\"closure_min_voxels\": 20,")
            && a.contains("\"closure_minimum\": \"scaled-area\""),
        "{a}"
    );
    assert_ne!(volume_bits(&new), volume_bits(&legacy));
    assert_eq!(
        store_dir_hash(&new),
        SCALED_DEFAULT_DEMO,
        "{:#018x}",
        store_dir_hash(&new)
    );
}

/// §6.7: `--min-closure-voxels 100` equals the library `Fixed(100)`, writes
/// no closure attributes, and reports `(fixed 100)`.
#[test]
fn min_closure_voxels_equals_library_fixed() {
    let dir = tempfile::tempdir().unwrap();
    let args = [
        "--seed",
        "102",
        "--shape",
        "24,20,128",
        "--faults",
        "4",
        "--sand-layer-fraction",
        "0.4",
        "--chunk-i",
        "5",
    ];
    let p = dir.path().join("fixed100.mdio");
    let out = run(&[&args[..], &["--min-closure-voxels", "100"]].concat(), &p);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("closures: minimum 100 voxels (fixed 100), kept 4 of 9 compartments"),
        "{stdout}"
    );
    assert!(!zattrs(&p).contains("closure_min"));
    let cfg = synthoseis_core::pipeline::E2eConfig {
        seed: 102,
        inline_count: 24,
        crossline_count: 20,
        samples: 128,
        faults: synthoseis_core::FaultConfig::with_count(4),
        rock_physics: synthoseis_core::RockPhysicsConfig {
            sand_layer_fraction: Some(0.4),
            closure_minimum: synthoseis_core::ClosureMinimum::Fixed(100),
            ..Default::default()
        },
        ..Default::default()
    };
    let lib: Vec<u32> = synthoseis_core::generate_chunked(&cfg)
        .0
        .angle_stack
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert!(
        volume_bits(&p) == lib,
        "--min-closure-voxels 100 vs library Fixed(100)"
    );
    // The default (T = 20) keeps the fifth compartment: the output differs.
    let q = dir.path().join("default.mdio");
    let out = run(&args, &q);
    assert!(String::from_utf8_lossy(&out.stdout).contains("closures: minimum 20 voxels (scaled ni·nj/180, floor 20, cap 500), kept 5 of 9 compartments"));
    assert_ne!(volume_bits(&q), lib);
}

/// Closure options reach multi-process workers: the multi-process store
/// equals the single-process one for every closure flag set.
#[test]
fn closure_flags_reach_multiprocess_workers() {
    let dir = tempfile::tempdir().unwrap();
    const MP: &[&str] = &["--multiprocess", "--workers", "3"];
    const TILES: &[&str] = &["--chunk-i", "2", "--chunk-j", "4", "--chunk-k", "8"];
    for flags in [
        &[][..],
        &["--legacy-closure-minimum"][..],
        &["--min-closure-voxels", "3"][..],
        &["--legacy-closure-contact-cap"][..],
        &["--legacy-closure-minimum", "--legacy-closure-contact-cap"][..],
    ] {
        let single = dir.path().join("single.mdio");
        let out = run(&[TILES, flags].concat(), &single);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        let mp = dir.path().join("mp.mdio");
        let out = run(&[MP, TILES, flags].concat(), &mp);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        assert_eq!(
            store_dir_hash(&single),
            store_dir_hash(&mp),
            "{flags:?}: multiprocess vs single"
        );
    }
}

/// §5: invalid combinations exit 2.
#[test]
fn invalid_closure_flags_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    for (bad, msg) in [
        (
            &["--legacy-closure-minimum", "--min-closure-voxels", "50"][..],
            "mutually exclusive",
        ),
        (
            &["--min-closure-voxels", "0"][..],
            "--min-closure-voxels expects N >= 1",
        ),
        (
            &["--legacy-closure-minimum", "--no-fluids"][..],
            "have no effect with --no-fluids or --legacy-toy-depth",
        ),
        (
            &["--min-closure-voxels", "50", "--no-fluids"][..],
            "have no effect with --no-fluids",
        ),
        (
            &["--legacy-closure-contact-cap", "--no-fluids"][..],
            "have no effect with --no-fluids",
        ),
        (
            &["--legacy-closure-minimum", "--legacy-toy-depth"][..],
            "or --legacy-toy-depth",
        ),
        (
            &["--legacy-closure-contact-cap", "--legacy-whole-voxels"][..],
            "--legacy-closure-contact-cap has no effect with --legacy-whole-voxels",
        ),
        (
            &["--legacy-closure-contact-cap", "--toy-geometry", "planar"][..],
            "--legacy-closure-contact-cap has no effect",
        ),
        (
            &["--legacy-closure-contact-cap", "--closures-unsegmented"][..],
            "--closures-unsegmented or --closures-per-layer",
        ),
        (
            &["--legacy-closure-contact-cap", "--closures-per-layer"][..],
            "--closures-unsegmented or --closures-per-layer",
        ),
    ] {
        let out = run(
            &[&["--shape", "12,10,64"][..], bad].concat(),
            &dir.path().join("bad.mdio"),
        );
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(msg),
            "{bad:?}: {out:?}"
        );
    }
    // Accepted: the minimum flags on the planar and the legacy closure
    // modes (the rule applies there unchanged).
    for ok in [
        &["--legacy-closure-minimum", "--toy-geometry", "planar"][..],
        &["--min-closure-voxels", "5", "--closures-unsegmented"][..],
        &["--min-closure-voxels", "5", "--closures-per-layer"][..],
        &["--legacy-closure-minimum", "--legacy-whole-voxels"][..],
    ] {
        let out = run(
            &[&["--shape", "12,10,64"][..], ok].concat(),
            &dir.path().join("ok.mdio"),
        );
        assert!(out.status.success(), "{ok:?}: {out:?}");
    }
}
