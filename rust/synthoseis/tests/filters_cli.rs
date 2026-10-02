//! CLI filter flags: `--keep-ricker` and `--bandpass-trailing-sample`
//! require `--bandpass`, and the summary reports the wavelet / bandpass trace.
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

/// `--bandpass-trailing-sample` requires `--bandpass`, restores the old
/// whole-trace bandpass (a different store) and is reported in the summary;
/// the default leaves the trailing sample of every trace 0.
#[test]
fn bandpass_trailing_sample_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let read = |p: &std::path::Path| synthoseis_io::MdioStore::open(p).unwrap().read_volume().unwrap();
    // The switch is a legacy-axis option (time mode always zeroes the dead
    // last sample): compare on `--legacy-depth-as-time`.
    let fixed = dir.path().join("fixed.mdio");
    let out = run(&["--bandpass", "4,30", "--legacy-depth-as-time"], &fixed);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("bandpass trace: first nk-1 samples, trailing sample 0"), "{stdout}");

    let old = dir.path().join("old.mdio");
    let out = run(
        &["--bandpass", "4,30", "--bandpass-trailing-sample", "--legacy-depth-as-time"],
        &old,
    );
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("bandpass trace: all nk samples"), "{stdout}");

    let (f, o) = (read(&fixed), read(&old));
    assert_eq!(f.len(), o.len());
    assert!(f.chunks_exact(64).all(|t| t[63].to_bits() == 0));
    assert!(o.chunks_exact(64).any(|t| t[63] != 0.0));
    assert_ne!(
        f.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        o.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );

    let out = run(&["--bandpass-trailing-sample"], &dir.path().join("bad.mdio"));
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--bandpass-trailing-sample requires --bandpass"));
    // Time mode (the default) rejects it (spec §2).
    let out = run(&["--bandpass", "4,30", "--bandpass-trailing-sample"], &dir.path().join("bad.mdio"));
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("requires --legacy-depth-as-time"));
}
