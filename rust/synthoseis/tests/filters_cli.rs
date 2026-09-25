//! CLI filter flags: `--keep-ricker` requires `--bandpass`, and the summary
//! reports whether the Ricker wavelet is skipped.
use std::process::Command;

fn run(extra: &[&str], store: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_synthoseis"))
        .args([
            "run",
            "--e2e",
            "--chunked",
            "--shape",
            "12,10,64",
            "--chunk-i",
            "5",
        ])
        .args(extra)
        .arg("--store")
        .arg(store)
        .output()
        .expect("spawn synthoseis")
}

#[test]
fn keep_ricker_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = run(&["--bandpass", "4,30"], &dir.path().join("skip.mdio"));
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("wavelet=none"), "{stdout}");

    let out = run(
        &["--bandpass", "4,30", "--keep-ricker"],
        &dir.path().join("keep.mdio"),
    );
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("wavelet=ricker"), "{stdout}");

    let out = run(&["--keep-ricker"], &dir.path().join("bad.mdio"));
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--keep-ricker requires --bandpass"));
}
