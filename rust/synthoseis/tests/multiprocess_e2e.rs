//! Real OS-process multi-process e2e via CARGO_BIN_EXE_synthoseis.
use std::process::Command;

#[test]
fn cli_multiprocess_spawns_real_workers() {
    let bin = env!("CARGO_BIN_EXE_synthoseis");
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("cli-mp.mdio");
    let status = Command::new(bin)
        .args([
            "run",
            "--e2e",
            "--chunked",
            "--multiprocess",
            "--workers",
            "4",
            "--chunk-i",
            "2",
            "--chunk-j",
            "4",
            "--chunk-k",
            "8",
            "--seed",
            "42",
            "--store",
        ])
        .arg(&store)
        .status()
        .expect("spawn synthoseis");
    assert!(status.success(), "multiprocess CLI failed: {status}");
    assert!(store.join("data").join("chunked_012").join("0.0.0").is_file());
    let plan = std::path::PathBuf::from(format!("{}.partition-plan.json", store.display()));
    assert!(plan.is_file(), "missing plan at {}", plan.display());
}
