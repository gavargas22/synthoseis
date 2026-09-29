//! CLI rock-physics flags: `--legacy-toy-depth` reproduces master 10f4dcd
//! stores bit for bit (single process and multi-process), the corrected
//! default differs, `--legacy-zoeppritz` reproduces master 33a3a93 stores
//! (the rock-physics default before the Zoeppritz fix), model flags reach
//! multi-process workers, and invalid combinations exit 2.
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

/// FNV-1a over the f32 bits of the stored angle stack.
fn store_hash(p: &Path) -> u64 {
    let v = synthoseis_io::MdioStore::open(p).unwrap().read_volume().unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

const PLAIN: &[&str] = &["--shape", "12,10,64", "--chunk-i", "5"];
const RICH: &[&str] = &[
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
];
const MP: &[&str] = &[
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

fn with<'a>(base: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    base.iter().chain(extra).copied().collect()
}

// Angle-stack hashes of stores written by the master 10f4dcd binary with
// the same flags (no rock-physics flags existed there).
const MASTER_PLAIN: u64 = 0xb55a_a488_ecfe_c81d;
const MASTER_RICH: u64 = 0x8c44_6a73_c50b_5a90;
const MASTER_MP: u64 = 0x3e27_e417_f473_649f;

#[test]
fn legacy_toy_depth_flag_reproduces_master_stores() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, base, master) in [
        ("plain", PLAIN, MASTER_PLAIN),
        ("rich", RICH, MASTER_RICH),
        ("mp", MP, MASTER_MP),
    ] {
        let legacy = dir.path().join(format!("{name}-legacy.mdio"));
        let out = run(&with(base, &["--legacy-toy-depth"]), &legacy);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("rock physics: legacy toy depth"), "{stdout}");
        assert_eq!(store_hash(&legacy), master, "{name}: legacy switch vs master");

        let default = dir.path().join(format!("{name}-default.mdio"));
        let out = run(base, &default);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("rock physics: 4 m/sample below seabed, mixing=inverse-velocity"),
            "{stdout}"
        );
        assert!(stdout.contains("zoeppritz=textbook"), "{stdout}");
        assert_ne!(store_hash(&default), master, "{name}: default must be the corrected model");
    }
}

// Angle-stack hashes of stores written by the master 33a3a93 binary (the
// rock-physics default with the legacy Zoeppritz `det` typo).
const MASTER33_PLAIN: u64 = 0xea04_aee3_0408_fd02;
const MASTER33_RICH: u64 = 0x1210_0347_2974_e5f8;
const MASTER33_MP: u64 = 0x7e4e_878d_f8b5_3e79;
const MASTER33_MP_FLAGS: u64 = 0x6c65_d8f4_c50d_19c3;
const MODEL_FLAGS: &[&str] = &[
    "--mixing",
    "backus",
    "--net-to-gross",
    "0.7",
    "--first-random-layer",
    "0",
    "--no-fluids",
];

#[test]
fn legacy_zoeppritz_flag_reproduces_master_33a3a93_stores() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mp_flags = with(MP, MODEL_FLAGS);
    for (name, base, master) in [
        ("plain", PLAIN, MASTER33_PLAIN),
        ("rich", RICH, MASTER33_RICH),
        ("mp", MP, MASTER33_MP),
        ("mp-flags", &mp_flags[..], MASTER33_MP_FLAGS),
    ] {
        let legacy = dir.path().join(format!("{name}-lz.mdio"));
        // 33a3a93 had only the planar toy geometry.
        let out = run(
            &with(base, &["--legacy-zoeppritz", "--toy-geometry", "planar"]),
            &legacy,
        );
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("zoeppritz=legacy (det typo)"), "{stdout}");
        assert_eq!(store_hash(&legacy), master, "{name}: --legacy-zoeppritz vs master 33a3a93");
        let fixed = dir.path().join(format!("{name}-fixed.mdio"));
        assert!(run(&with(base, &["--toy-geometry", "planar"]), &fixed)
            .status
            .success());
        assert_ne!(
            store_hash(&fixed),
            master,
            "{name}: default must use the textbook Zoeppritz"
        );
    }
    // Redundant with the toy switch (which implies it): still master 10f4dcd.
    let both = dir.path().join("both.mdio");
    let out = run(&with(PLAIN, &["--legacy-toy-depth", "--legacy-zoeppritz"]), &both);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(store_hash(&both), MASTER_PLAIN);
}

#[test]
fn model_flags_reach_multiprocess_workers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let flags = [
        "--mixing",
        "backus",
        "--net-to-gross",
        "0.7",
        "--first-random-layer",
        "0",
        "--no-fluids",
    ];
    // Multi-process children must build the same model as one process with
    // the same chunking.
    let single_args = with(&with(MP, &flags)[3..], &[]);
    let single = dir.path().join("single.mdio");
    let out = run(&single_args, &single);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "mixing=backus, net_to_gross=constant 0.7, first_random_layer=0, fluids=brine"
        ),
        "{stdout}"
    );
    let mp = dir.path().join("mp.mdio");
    let out = run(&with(MP, &flags), &mp);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(store_hash(&single), store_hash(&mp));

    let default = dir.path().join("default.mdio");
    assert!(run(MP, &default).status.success());
    assert_ne!(store_hash(&default), store_hash(&mp), "flags must change the model");
}

#[test]
fn invalid_rock_physics_flags_exit_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bad, msg) in [
        (["--legacy-toy-depth", "--mixing", "backus"].as_slice(), "no effect with --legacy-toy-depth"),
        (&["--legacy-toy-depth", "--no-fluids"], "no effect with --legacy-toy-depth"),
        (&["--mixing", "wyllie"], "--mixing expects inverse-velocity or backus"),
        (&["--net-to-gross", "1.5"], "net"),
    ] {
        let out = run(&with(PLAIN, bad), &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(msg), "{bad:?}: {out:?}");
    }
}

/// `--toy-geometry`: layered is the default, both geometries reach
/// multi-process workers (bit-identical to one process), and they differ.
#[test]
fn toy_geometry_flag_reaches_workers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut hashes = Vec::new();
    for geometry in [None, Some("layered"), Some("planar")] {
        let extra: Vec<&str> = geometry
            .map(|g| vec!["--toy-geometry", g])
            .unwrap_or_default();
        let name = geometry.unwrap_or("default");
        let single = dir.path().join(format!("{name}-single.mdio"));
        let out = run(&with(&MP[3..], &extra), &single);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let want = format!("toy geometry: {}", geometry.unwrap_or("layered"));
        assert!(stdout.contains(&want), "{name}: {stdout}");
        let mp = dir.path().join(format!("{name}-mp.mdio"));
        let out = run(&with(MP, &extra), &mp);
        assert!(out.status.success(), "{name}: {out:?}");
        assert_eq!(
            store_hash(&single),
            store_hash(&mp),
            "{name}: multiprocess vs single"
        );
        hashes.push(store_hash(&single));
    }
    assert_eq!(hashes[0], hashes[1], "default is layered");
    assert_ne!(hashes[1], hashes[2], "layered vs planar");
}

#[test]
fn invalid_toy_geometry_flags_exit_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bad, msg) in [
        (
            ["--legacy-toy-depth", "--toy-geometry", "layered"].as_slice(),
            "no effect with --legacy-toy-depth",
        ),
        (&["--toy-geometry", "salt"], "--toy-geometry"),
    ] {
        let out = run(&with(PLAIN, bad), &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(msg),
            "{bad:?}: {out:?}"
        );
    }
    // Explicit planar with the toy switch is redundant but accepted.
    let ok = dir.path().join("ok.mdio");
    let out = run(
        &with(PLAIN, &["--legacy-toy-depth", "--toy-geometry", "planar"]),
        &ok,
    );
    assert!(out.status.success(), "{out:?}");
    assert_eq!(store_hash(&ok), MASTER_PLAIN);
}
