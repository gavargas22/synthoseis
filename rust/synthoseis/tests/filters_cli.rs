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

#[test]
fn noise_flags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let read = |p: &std::path::Path| {
        synthoseis_io::MdioStore::open(p)
            .unwrap()
            .read_volume()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<u32>>()
    };
    let a = dir.path().join("a.mdio");
    let out = run(&["--bandpass", "4,30", "--noise-snr-db", "12.5"], &a);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("noise: snr=12.5 dB, seed=42, weights=radians, data_std_cutoff=seabed"),
        "{stdout}"
    );

    // Explicit seed equal to the default gives identical output; another differs.
    let b = dir.path().join("b.mdio");
    let out = run(&["--bandpass", "4,30", "--noise-snr-db", "12.5", "--noise-seed", "42"], &b);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(read(&a), read(&b));
    let c = dir.path().join("c.mdio");
    let out = run(
        &["--noise-snr-db", "12.5", "--noise-seed", "4", "--noise-legacy-weights"],
        &c,
    );
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("weights=legacy-degrees"));
    assert_ne!(read(&a), read(&c));
    // Legacy seabed mask: different data_std, so a different (rescaled) stack.
    let d = dir.path().join("d.mdio");
    let out = run(
        &["--bandpass", "4,30", "--noise-snr-db", "12.5", "--noise-legacy-seabed"],
        &d,
    );
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("data_std_cutoff=legacy-0.84-seabed"));
    assert_ne!(read(&a), read(&d));

    for bad in [["--noise-seed", "3"].as_slice(), &["--noise-legacy-seabed"]] {
        let out = run(bad, &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&out.stderr).contains("require --noise-snr-db"));
    }
}
