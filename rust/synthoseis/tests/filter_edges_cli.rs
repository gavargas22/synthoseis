//! `--legacy-filter-edges` (filter-edge spec §5, §7.7, §7.10): the switch
//! restores master 1c22b653 store trees bit for bit (apart from the
//! `created` stamp) on every path, the physical default moves only the
//! angle stacks, and the summary line and store attribute say which edges
//! a store has. The exit-2 rules and the multi-process forwarding are in
//! `depth_to_time_cli.rs`.

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

/// FNV-1a over every file of a store tree (sorted relative paths and
/// contents), skipping the `created` lines of `.zattrs` / `.zmetadata`.
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

const BP: &[&str] = &[
    "--chunked",
    "--faults",
    "3",
    "--shape",
    "24,20,64",
    "--chunk-i",
    "5",
    "--chunk-j",
    "7",
];

/// The cases: name, flags, stores written (`many` writes one per angle).
fn cases() -> Vec<(&'static str, Vec<&'static str>)> {
    let with = |extra: &[&'static str]| BP.iter().chain(extra).copied().collect::<Vec<_>>();
    vec![
        (
            "demo",
            vec![
                "--chunked",
                "--seed",
                "7",
                "--shape",
                "32,32,128",
                "--faults",
                "3",
            ],
        ),
        (
            "rich",
            vec![
                "--chunked",
                "--shape",
                "12,10,64",
                "--chunk-i",
                "5",
                "--faults",
                "3",
                "--bandpass",
                "4,30",
                "--noise-snr-db",
                "12.5",
            ],
        ),
        (
            "salt",
            vec![
                "--chunked",
                "--seed",
                "4",
                "--shape",
                "24,24,128",
                "--faults",
                "4",
                "--sand-layer-fraction",
                "0.4",
            ],
        ),
        (
            "bplat",
            with(&["--bandpass", "4,30", "--lateral-filter", "3"]),
        ),
        (
            "noise",
            with(&["--noise-snr-db", "12.5", "--noise-seed", "7"]),
        ),
        ("keep", with(&["--bandpass", "4,30", "--keep-ricker"])),
        (
            "bplatnoise",
            with(&[
                "--bandpass",
                "4,30",
                "--lateral-filter",
                "3",
                "--noise-snr-db",
                "12.5",
                "--noise-seed",
                "7",
            ]),
        ),
        (
            "dt2lin",
            vec![
                "--chunked",
                "--shape",
                "24,20,64",
                "--dt-ms",
                "2",
                "--twt-kernel",
                "linear",
                "--bandpass",
                "4,30",
            ],
        ),
        (
            "whole",
            vec![
                "--chunked",
                "--seed",
                "7",
                "--shape",
                "32,32,128",
                "--faults",
                "3",
                "--legacy-whole-voxels",
            ],
        ),
        ("single", vec![]),
        ("stream", vec!["--chunked"]),
        ("overlap", vec!["--chunked", "--overlap"]),
        (
            "strips",
            vec![
                "--chunked",
                "--workers",
                "4",
                "--chunk-i",
                "2",
                "--chunk-j",
                "4",
            ],
        ),
        (
            "mp",
            vec![
                "--chunked",
                "--multiprocess",
                "--workers",
                "3",
                "--chunk-i",
                "2",
                "--chunk-j",
                "4",
            ],
        ),
        (
            "mpk",
            vec![
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
                "--twt-samples",
                "12",
            ],
        ),
        ("many", vec!["--chunked", "--angles", "0,15,30"]),
    ]
}

/// `store_dir_hash` of the stores master 1c22b653 writes for [`cases`]
/// (recorded on 1c22b653 before the change).
const MASTER_1C22B653: &[(&str, u64)] = &[
    ("bplat", 0xef49_2e8a_55ea_f323),
    ("bplatnoise", 0x8abb_d70c_853a_7b8e),
    ("demo", 0xa91e_7b31_5d63_fcba),
    ("dt2lin", 0x29f9_4e8f_ab0f_7d9b),
    ("keep", 0x3822_8de1_6fa3_3f90),
    ("many_a0", 0x330c_248f_7ecf_4a5e),
    ("many_a15", 0xe30f_78a9_6c6f_de6a),
    ("many_a30", 0x1711_8c5b_243e_675b),
    ("mp", 0xe7c9_a986_e335_d861),
    ("mpk", 0xf617_f193_f9e2_1396),
    ("noise", 0x4267_c8c9_72c5_352b),
    ("overlap", 0x0c7d_a2dc_279b_7b89),
    ("rich", 0xc3e7_1a52_9f40_6a3a),
    ("salt", 0x363b_b3fb_f870_a979),
    ("single", 0xe30f_78a9_6c6f_de6a),
    ("stream", 0xe30f_78a9_6c6f_de6a),
    ("strips", 0xe7c9_a986_e335_d861),
    ("whole", 0x2676_751d_4661_646a),
];

/// The same cases with the physical filter edges (the new default; only
/// the angle stacks and the `filter_edges` root attribute differ).
const PHYSICAL: &[(&str, u64)] = &[
    ("bplat", 0xa63f_a3f1_b4c6_5154),
    ("bplatnoise", 0x0004_e11a_3ecb_7e58),
    ("demo", 0x1dba_aaf9_2418_e1fe),
    ("dt2lin", 0x6ec9_3b54_c79d_b3f2),
    ("keep", 0x8832_ffeb_4963_544d),
    ("many_a0", 0x4b08_83cd_8e12_6880),
    ("many_a15", 0xed98_caea_fb1b_5f91),
    ("many_a30", 0xe7e4_d0cf_6da3_9f9b),
    ("mp", 0xcef0_bf30_3f56_5387),
    ("mpk", 0x5b47_aa85_4ef6_59da),
    ("noise", 0x889f_feae_b617_34dc),
    ("overlap", 0x6b7e_6534_d235_59cf),
    ("rich", 0xaffa_71a9_7909_ad44),
    ("salt", 0xbbb2_c26d_f5c3_267d),
    ("single", 0xed98_caea_fb1b_5f91),
    ("stream", 0xed98_caea_fb1b_5f91),
    ("strips", 0xcef0_bf30_3f56_5387),
    ("whole", 0x3da6_ba0a_97d5_8b13),
];

const SUMMARY_PHYSICAL: &str =
    "filter edges: physical (water above, model below; reflect sideways)\n";
const SUMMARY_LEGACY: &str = "filter edges: legacy 1c22b653 (--legacy-filter-edges)\n";

/// Run every case with `extra`, check the summary line and the
/// `filter_edges` attribute, and return `(store, hash)` for every store.
fn run_cases(dir: &Path, extra: &[&str], summary: &str, attr: Option<&str>) -> Vec<(String, u64)> {
    let mut got = Vec::new();
    for (name, args) in cases() {
        let args: Vec<&str> = args.iter().copied().chain(extra.iter().copied()).collect();
        let out = run(&args, &dir.join(format!("{name}.mdio")));
        assert!(out.status.success(), "{name}: {out:?}");
        let so = String::from_utf8_lossy(&out.stdout);
        assert!(so.contains(summary), "{name}: {so}");
        let stores: Vec<String> = if name == "many" {
            ["many_a0", "many_a15", "many_a30"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            vec![name.to_string()]
        };
        for s in stores {
            let p = dir.join(format!("{s}.mdio"));
            let attrs = synthoseis_io::MdioStore::open(&p)
                .unwrap()
                .root_attrs()
                .unwrap();
            assert_eq!(
                attrs.get("filter_edges").and_then(|v| v.as_str()),
                attr,
                "{s}"
            );
            got.push((s, store_dir_hash(&p)));
        }
    }
    got
}

fn check(got: &[(String, u64)], want: &[(&str, u64)], what: &str) {
    assert_eq!(got.len(), want.len());
    let mut bad = Vec::new();
    for (name, h) in got {
        let w = want
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name}: no pin"))
            .1;
        if *h != w {
            bad.push(format!("{name}: {h:#018x} (pinned {w:#018x})"));
        }
    }
    assert!(bad.is_empty(), "{what}:\n{}", bad.join("\n"));
}

/// §7.7: `--legacy-filter-edges` reproduces master 1c22b653 bit for bit
/// on every path (classic single, streaming, overlap, strips,
/// multi-process with and without k-chunks, geometry-once angles) and for
/// every filter chain (Ricker, bandpass, lateral, noise, keep-Ricker, dt 2
/// ms linear, whole voxels, salt).
#[test]
fn legacy_filter_edges_reproduce_1c22b653_on_every_path() {
    let dir = tempfile::tempdir().unwrap();
    let got = run_cases(dir.path(), &["--legacy-filter-edges"], SUMMARY_LEGACY, None);
    check(
        &got,
        MASTER_1C22B653,
        "--legacy-filter-edges vs master 1c22b653",
    );
}

/// The physical default on the same cases: every store moves (angle
/// stacks only; the labels are checked in
/// `synthoseis-core/tests/filter_edges.rs`), and says so in the summary
/// line and the `filter_edges = "physical"` root attribute.
#[test]
fn physical_filter_edges_default_pins() {
    let dir = tempfile::tempdir().unwrap();
    let got = run_cases(dir.path(), &[], SUMMARY_PHYSICAL, Some("physical"));
    check(&got, PHYSICAL, "physical filter edges");
    for (name, h) in &got {
        let old = MASTER_1C22B653.iter().find(|(n, _)| n == name).unwrap().1;
        assert_ne!(*h, old, "{name}: the default moves");
    }
}

/// The legacy depth axis has no time-mode filter edges: no summary line,
/// no attribute, and the switch is rejected (exit 2).
#[test]
fn legacy_axis_has_no_filter_edge_line() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("axis.mdio");
    let out = run(
        &[
            "--chunked",
            "--shape",
            "12,10,64",
            "--legacy-depth-as-time",
            "--bandpass",
            "4,30",
        ],
        &p,
    );
    assert!(out.status.success(), "{out:?}");
    let so = String::from_utf8_lossy(&out.stdout);
    assert!(!so.contains("filter edges:"), "{so}");
    let attrs = synthoseis_io::MdioStore::open(&p)
        .unwrap()
        .root_attrs()
        .unwrap();
    assert!(attrs.get("filter_edges").is_none());
    let out = run(
        &[
            "--chunked",
            "--legacy-depth-as-time",
            "--legacy-filter-edges",
        ],
        &p,
    );
    assert_eq!(out.status.code(), Some(2), "{out:?}");
}
