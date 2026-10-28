//! `--unconformities`, `--legacy-unconformity` and the unconformity flags
//! (erosion and onlap spec §5, §7.7, §7.8): the exit-2 rules, the legacy
//! switches, the summary lines, the `unconformity_labels` array and the
//! multi-process forwarding.

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

/// FNV-1a over every file of a store tree, skipping the `created` lines.
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

const BP: &[&str] = &["--chunked", "--seed", "4", "--shape", "32,28,160", "--faults", "2", "--chunk-i", "5", "--chunk-j", "7"];

fn with(extra: &[&str]) -> Vec<String> {
    BP.iter().chain(extra).map(|s| s.to_string()).collect()
}

const MP: &[&str] = &["--chunked", "--seed", "4", "--chunk-i", "2", "--chunk-j", "4"];

/// The default shape, no faults: the multi-process path takes neither.
fn go_mp(extra: &[&str], store: &Path) -> Output {
    let args: Vec<&str> = MP.iter().chain(extra).copied().collect();
    run(&args, store)
}

fn go(extra: &[&str], store: &Path) -> Output {
    let args = with(extra);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run(&refs, store)
}

fn has_labels(store: &Path) -> bool {
    store.join("data/unconformity_labels").exists()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout)
}

#[test]
fn invalid_unconformity_flags_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    for (bad, msg) in [
        (&["--legacy-unconformity", "--unconformities", "1"][..], "no effect with --legacy-unconformity"),
        (&["--legacy-unconformity", "--unconformity-level", "0.5"][..], "no effect with --legacy-unconformity"),
        (&["--legacy-unconformity", "--unconformity-shale", "both"][..], "no effect with --legacy-unconformity"),
        (&["--toy-geometry", "planar", "--legacy-unconformity"][..], "--toy-geometry planar"),
        (&["--toy-geometry", "planar", "--unconformities", "1"][..], "--toy-geometry planar"),
        (&["--legacy-toy-depth", "--legacy-unconformity"][..], "--legacy-toy-depth"),
        (&["--legacy-toy-depth", "--unconformity-fill-ratio", "3"][..], "--legacy-toy-depth"),
        (&["--legacy-fold-geometry", "--unconformities", "1"][..], "no effect with --legacy-fold-geometry"),
        (&["--legacy-fold-geometry", "--unconformity-relief-m", "10"][..], "no effect with --legacy-fold-geometry"),
        (&["--unconformities", "0", "--unconformity-position", "0.5"][..], "no effect with --unconformities 0"),
        (&["--unconformities", "3"][..], "--unconformities must be 0, 1 or 2"),
        (&["--unconformity-position", "0.05"][..], "--unconformity-position must be in [0.1, 0.9]"),
        (&["--unconformity-level", "0.95"][..], "--unconformity-level must be in [0.1, 0.9]"),
        (&["--unconformity-relief-m", "250"][..], "--unconformity-relief-m must be in [0, 200]"),
        (&["--unconformity-max-dip", "0.01"][..], "--unconformity-max-dip must be in [0.05, 2]"),
        (&["--unconformity-fill-ratio", "7"][..], "--unconformity-fill-ratio must be in [1, 6]"),
        (&["--unconformity-shale", "side"][..], "shale"),
    ] {
        let out = go(bad, &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        let err = stderr(&out);
        assert!(err.contains(msg), "{bad:?}: {err}");
    }
}

/// `--legacy-fold-geometry` implies no unconformities (D8); both legacy
/// flags together are accepted and agree with either alone.
#[test]
fn legacy_switches_and_summary_lines() {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(format!("{n}.mdio"));
    let default = go(&[], &p("default"));
    assert!(default.status.success(), "{default:?}");
    let text = stderr(&default);
    assert!(text.contains("unconformity 1: horizon"), "{text}");
    assert!(text.contains("unconformities: max dip 1 samples/column, fill ratio 2, shale seal below, exhumation burial on"), "{text}");
    assert!(has_labels(&p("default")));

    let legacy = go(&["--legacy-unconformity"], &p("legacy"));
    let zero = go(&["--unconformities", "0"], &p("zero"));
    assert!(legacy.status.success() && zero.status.success());
    assert!(stderr(&legacy).contains("unconformities: off (--legacy-unconformity)"), "{}", stderr(&legacy));
    assert!(stderr(&zero).contains("unconformities: off (--unconformities 0)"), "{}", stderr(&zero));
    assert!(!has_labels(&p("legacy")) && !has_labels(&p("zero")));
    assert_eq!(store_dir_hash(&p("legacy")), store_dir_hash(&p("zero")), "--legacy-unconformity == --unconformities 0");
    assert_ne!(store_dir_hash(&p("legacy")), store_dir_hash(&p("default")));

    let lfg = go(&["--legacy-fold-geometry"], &p("lfg"));
    let both = go(&["--legacy-fold-geometry", "--legacy-unconformity"], &p("both"));
    assert!(lfg.status.success() && both.status.success(), "{both:?}");
    assert!(!has_labels(&p("lfg")));
    assert_eq!(store_dir_hash(&p("lfg")), store_dir_hash(&p("both")), "D8");

    // The explicit default equals the implicit one.
    let explicit = go(&["--unconformities", "1", "--unconformity-shale", "below", "--unconformity-max-dip", "1", "--unconformity-fill-ratio", "2"], &p("explicit"));
    assert!(explicit.status.success());
    assert_eq!(store_dir_hash(&p("explicit")), store_dir_hash(&p("default")));
}

/// The multi-process path takes the flags (workers run real processes on
/// the default cube, which is below the D10 margins, so every setting equals
/// legacy and writes no labels); on a cube that applies the unconformity the
/// flags change the store and the labels are written. Store equality across
/// the writers on applying cubes is in `synthoseis-core/tests/unconformity.rs`
/// and the worker round trip in `main.rs`.
#[test]
fn flags_are_accepted_by_workers_and_change_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join("legacy.mdio");
    assert!(go_mp(&["--legacy-unconformity"], &legacy).status.success());
    for (tag, flags) in [
        ("default", &[][..]),
        ("two", &["--unconformities", "2", "--unconformity-position", "0.3"][..]),
        (
            "flags",
            &[
                "--unconformity-level", "0.42", "--unconformity-relief-m", "20", "--unconformity-max-dip", "0.5",
                "--unconformity-fill-ratio", "3", "--unconformity-shale", "both", "--no-exhumation-burial",
            ][..],
        ),
    ] {
        let single = dir.path().join(format!("{tag}-single.mdio"));
        let mp = dir.path().join(format!("{tag}-mp.mdio"));
        let out = go_mp(flags, &single);
        assert!(out.status.success(), "{tag}: {out:?}");
        let out = go_mp(&[&["--multiprocess", "--workers", "3"][..], flags].concat(), &mp);
        assert!(out.status.success(), "{tag}: {out:?}");
        assert_eq!(store_dir_hash(&single), store_dir_hash(&mp), "{tag}: multi-process = one process");
    }
    // An applying cube: the flags move the store, the default writes labels.
    let big = |n: &str, extra: &[&str]| {
        let p = dir.path().join(format!("{n}.mdio"));
        let out = go(extra, &p);
        assert!(out.status.success(), "{n}: {out:?}");
        p
    };
    let d = big("big-default", &[]);
    let l = big("big-level", &["--unconformity-level", "0.3"]);
    let s = big("big-burial", &["--no-exhumation-burial"]);
    assert!(has_labels(&d) && has_labels(&l) && has_labels(&s));
    assert_ne!(store_dir_hash(&d), store_dir_hash(&l));
    assert_ne!(store_dir_hash(&d), store_dir_hash(&s));
}

/// Time mode and the legacy time axis both write the labels; a cube too
/// small for the margins (D10) writes none and matches legacy.
#[test]
fn time_mode_labels_and_small_cubes() {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(format!("{n}.mdio"));
    for (n, extra) in [("time", &[][..]), ("axis", &["--legacy-depth-as-time"][..])] {
        let out = go(extra, &p(n));
        assert!(out.status.success(), "{n}: {out:?}");
        assert!(has_labels(&p(n)), "{n}");
    }
    let small = ["--chunked", "--seed", "4", "--shape", "8,8,16"];
    let (a, b) = (p("small"), p("small-legacy"));
    assert!(run(&small, &a).status.success());
    assert!(run(&[&small[..], &["--legacy-unconformity"][..]].concat(), &b).status.success());
    assert!(!has_labels(&a));
    assert_eq!(store_dir_hash(&a), store_dir_hash(&b));
}
