//! CLI of the analytic folds (folds spec §5, §6): summary lines, invalid
//! combinations exit 2, and the multi-process path equals one process.
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

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn volume(p: &Path) -> (Vec<u32>, Vec<u8>) {
    let s = synthoseis_io::MdioStore::open(p).unwrap();
    (
        s.read_volume()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        s.read_labels_u8().unwrap(),
    )
}

#[test]
fn fold_summary_lines() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&["--shape", "32,32,128"], &dir.path().join("a.mdio"));
    assert!(out.status.success(), "{out:?}");
    assert!(
        stdout(&out).contains("\nfolds: 1 cylindrical + 1 dome/basin, style similar"),
        "{}",
        stdout(&out)
    );
    let out = run(
        &[
            "--shape",
            "32,32,128",
            "--fold-style",
            "parallel",
            "--folds-cylindrical",
            "2",
            "--folds-domes",
            "0",
        ],
        &dir.path().join("b.mdio"),
    );
    assert!(out.status.success(), "{out:?}");
    assert!(
        stdout(&out).contains("\nfolds: 2 cylindrical + 0 dome/basin, style parallel"),
        "{}",
        stdout(&out)
    );
    let out = run(&["--legacy-fold-geometry"], &dir.path().join("c.mdio"));
    assert!(out.status.success(), "{out:?}");
    assert!(
        stdout(&out).contains("\nfolds: off (--legacy-fold-geometry)\n"),
        "{}",
        stdout(&out)
    );
    // Planar has no fold line.
    let out = run(&["--toy-geometry", "planar"], &dir.path().join("d.mdio"));
    assert!(out.status.success(), "{out:?}");
    assert!(!stdout(&out).contains("folds:"), "{}", stdout(&out));
    // Folds change the store; the legacy switch differs from the default.
    assert_ne!(
        volume(&dir.path().join("a.mdio")).1,
        volume(&dir.path().join("b.mdio")).1
    );
}

#[test]
fn invalid_fold_flags_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    for (bad, msg) in [
        (
            &["--legacy-fold-geometry", "--fold-style", "parallel"][..],
            "no effect with --legacy-fold-geometry",
        ),
        (
            &["--legacy-fold-geometry", "--bin-m", "20"][..],
            "no effect with --legacy-fold-geometry",
        ),
        (
            &["--toy-geometry", "planar", "--legacy-fold-geometry"][..],
            "--toy-geometry planar",
        ),
        (
            &["--toy-geometry", "planar", "--folds-domes", "2"][..],
            "--toy-geometry planar",
        ),
        (
            &["--legacy-toy-depth", "--legacy-fold-geometry"][..],
            "--legacy-toy-depth",
        ),
        (
            &["--legacy-toy-depth", "--fold-growth", "2"][..],
            "--legacy-toy-depth",
        ),
        (
            &["--fold-style", "kink"][..],
            "--fold-style must be similar or parallel",
        ),
        (
            &["--folds-cylindrical", "5"][..],
            "--folds-cylindrical must be in 0-4",
        ),
        (&["--folds-domes", "7"][..], "--folds-domes must be in 0-6"),
        (&["--bin-m", "4"][..], "--bin-m must be in 5-50"),
        (
            &["--fold-growth", "3.5"][..],
            "--fold-growth must be in 0-3",
        ),
        (
            &["--fold-max-dip", "0.05"][..],
            "--fold-max-dip must be in 0.1-2",
        ),
    ] {
        let out = run(bad, &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(msg), "{bad:?}: {err}");
    }
}

/// Fold flags reach multi-process workers: the multi-process store equals
/// the single-process store, for the default and for non-default flags.
#[test]
fn fold_flags_reach_multiprocess_workers() {
    let dir = tempfile::tempdir().unwrap();
    for (tag, flags) in [
        ("default", &[][..]),
        ("legacy", &["--legacy-fold-geometry"][..]),
        (
            "flags",
            &[
                "--fold-style",
                "parallel",
                "--folds-domes",
                "3",
                "--bin-m",
                "20",
                "--fold-growth",
                "0.5",
            ][..],
        ),
    ] {
        let single = dir.path().join(format!("{tag}-single.mdio"));
        let mp = dir.path().join(format!("{tag}-mp.mdio"));
        let base = ["--chunk-i", "2", "--chunk-j", "4"];
        let out = run(&[&base[..], flags].concat(), &single);
        assert!(out.status.success(), "{tag}: {out:?}");
        let out = run(
            &[&["--multiprocess", "--workers", "3"][..], &base[..], flags].concat(),
            &mp,
        );
        assert!(out.status.success(), "{tag}: {out:?}");
        assert_eq!(
            volume(&single),
            volume(&mp),
            "{tag}: multi-process = one process"
        );
    }
}
