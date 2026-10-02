//! CLI rock-physics flags: `--legacy-toy-depth` reproduces master 10f4dcd
//! stores bit for bit (single process and multi-process), the corrected
//! default differs, `--legacy-zoeppritz` reproduces master 33a3a93 stores
//! (the rock-physics default before the Zoeppritz fix), model flags reach
//! multi-process workers, and invalid combinations exit 2.
use std::path::Path;
use std::process::{Command, Output};

/// Every golden set below runs on the legacy depth-as-time axis
/// (`--legacy-depth-as-time`, depth-to-time spec §4): the pinned hashes are
/// stores written by master binaries before the time conversion, and must
/// pass unchanged. Time-mode runs use [`run_time`].
fn run(args: &[&str], store: &Path) -> Output {
    run_time(&with(args, &["--legacy-depth-as-time"]), store)
}

/// [`run`] without the legacy axis switch: the default time output.
fn run_time(args: &[&str], store: &Path) -> Output {
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
// `--bandpass-trailing-sample`: every RICH golden below is a store written by
// a master binary from before the bandpass trailing-sample fix.
const RICH: &[&str] = &[
    "--shape",
    "12,10,64",
    "--chunk-i",
    "5",
    "--faults",
    "3",
    "--bandpass",
    "4,30",
    "--bandpass-trailing-sample",
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

// Angle-stack hashes of stores written by the master binary after #30
// (layered geometry, alternating lithology) with the same flags.
const MASTER30_PLAIN: u64 = 0x1c8a_d8b3_5687_42d6;
const MASTER30_RICH: u64 = 0x55d8_6de1_924e_e347;
const MASTER30_MP: u64 = 0xbdd5_2772_230c_2542;
const MASTER30_DEEP: u64 = 0xb891_542c_7783_2264;
const DEEP: &[&str] = &["--shape", "24,20,160", "--faults", "3", "--chunk-i", "5"];

/// `--toy-lithology alternating` reproduces master after #30 bit for bit;
/// the Markov default differs.
#[test]
fn alternating_lithology_reproduces_master_after_30() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, base, master) in [
        ("plain", PLAIN, MASTER30_PLAIN),
        ("rich", RICH, MASTER30_RICH),
        ("mp", MP, MASTER30_MP),
        ("deep", DEEP, MASTER30_DEEP),
    ] {
        let alt = dir.path().join(format!("{name}-alt.mdio"));
        let out = run(&with(base, &["--toy-lithology", "alternating", "--no-salt"]), &alt);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("toy lithology: alternating"), "{name}: {stdout}");
        assert_eq!(store_hash(&alt), master, "{name}: --toy-lithology alternating vs master after #30");
    }
    let markov = dir.path().join("deep-markov.mdio");
    let out = run(DEEP, &markov);
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("toy lithology: markov (sand fraction"));
    assert_ne!(store_hash(&markov), MASTER30_DEEP, "default must use the Markov lithology");
}

/// Lithology options reach multi-process workers.
#[test]
fn lithology_flags_reach_multiprocess_workers() {
    let dir = tempfile::tempdir().expect("tempdir");
    for flags in [
        &[][..],
        &["--sand-layer-fraction", "0.5", "--sand-layer-thickness", "1"][..],
        &["--toy-lithology", "alternating"][..],
        &["--toy-geometry", "planar"][..],
        &["--closures-per-layer"][..],
        &["--closures-unsegmented"][..],
        &["--no-salt"][..],
        &["--salt-legacy-top-offset"][..],
    ] {
        let single = dir.path().join("single.mdio");
        let out = run(&with(&MP[3..], flags), &single);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        let mp = dir.path().join("mp.mdio");
        let out = run(&with(MP, flags), &mp);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        assert_eq!(store_hash(&single), store_hash(&mp), "{flags:?}: multiprocess vs single");
    }
}

#[test]
fn invalid_lithology_flags_exit_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bad, msg) in [
        (["--toy-geometry", "planar", "--toy-lithology", "markov"].as_slice(), "planar geometry"),
        (&["--legacy-toy-depth", "--sand-layer-fraction", "0.2"], "planar geometry"),
        (&["--toy-lithology", "alternating", "--sand-layer-thickness", "3"], "no effect with --toy-lithology alternating"),
        (&["--sand-layer-fraction", "0.7"], "unreachable"),
        (&["--sand-layer-thickness", "0.5"], "--sand-layer-thickness must be >= 1"),
        (&["--toy-lithology", "fluvial"], "--toy-lithology expects markov or alternating"),
        (&["--closures-per-layer", "--legacy-toy-depth"], "--closures-per-layer has no effect with --legacy-toy-depth"),
        (&["--closures-per-layer", "--no-fluids"], "--closures-per-layer has no effect with --legacy-toy-depth or --no-fluids"),
        (&["--closures-per-layer", "--toy-geometry", "planar"], "--closures-per-layer has no effect with the planar geometry"),
        (&["--closures-unsegmented", "--legacy-toy-depth"], "--closures-unsegmented has no effect with --legacy-toy-depth"),
        (&["--closures-unsegmented", "--no-fluids"], "--closures-unsegmented has no effect with --legacy-toy-depth, --no-fluids"),
        (&["--closures-unsegmented", "--closures-per-layer"], "--closures-unsegmented has no effect with --legacy-toy-depth, --no-fluids or --closures-per-layer"),
        (&["--closures-unsegmented", "--toy-geometry", "planar"], "--closures-unsegmented has no effect with the planar geometry"),
        (&["--no-salt", "--toy-geometry", "planar"], "--no-salt / --salt-legacy-top-offset have no effect with the planar geometry"),
        (&["--no-salt", "--legacy-toy-depth"], "--no-salt / --salt-legacy-top-offset have no effect with the planar geometry or --legacy-toy-depth"),
        (&["--salt-legacy-top-offset", "--toy-geometry", "planar"], "have no effect with the planar geometry"),
        (&["--salt-legacy-top-offset", "--no-salt"], "--salt-legacy-top-offset has no effect with --no-salt"),
    ] {
        let out = run(&with(PLAIN, bad), &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(msg), "{bad:?}: {out:?}");
    }
}

// Angle-stack hashes of stores written by the master 8b5988f binary
// (closures on every sand layer) with the same flags.
const MASTER8B_PLAIN: u64 = 0x2205_9836_d641_6f98;
const MASTER8B_RICH: u64 = 0x204c_83fa_48a4_befb;
const MASTER8B_MP: u64 = 0x2841_1c79_845b_b9a5;
const MASTER8B_DEEP: u64 = 0xd1a3_74bf_b795_15e5;
const MASTER8B_SAND: u64 = 0xcd58_f19d_3cf8_f60a;
/// Two multi-layer sand units close on the dome at this size.
const SAND: &[&str] = &[
    "--shape",
    "32,32,128",
    "--chunk-i",
    "16",
    "--sand-layer-fraction",
    "0.5",
    "--sand-layer-thickness",
    "3",
];
/// `SAND` with closures per sand unit (the default).
const SAND_UNIT: u64 = 0xee90_356c_257c_d617;

/// `--closures-per-layer` reproduces master 8b5988f bit for bit; the
/// per-unit default differs once a multi-layer sand unit closes.
#[test]
fn closures_per_layer_flag_reproduces_master_8b5988f() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, base, master) in [
        ("plain", PLAIN, MASTER8B_PLAIN),
        ("rich", RICH, MASTER8B_RICH),
        ("mp", MP, MASTER8B_MP),
        ("deep", DEEP, MASTER8B_DEEP),
        ("sand", SAND, MASTER8B_SAND),
    ] {
        let pl = dir.path().join(format!("{name}-pl.mdio"));
        let out = run(&with(base, &["--closures-per-layer", "--no-salt"]), &pl);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("closures: per sand layer (--closures-per-layer"), "{name}: {stdout}");
        assert_eq!(store_hash(&pl), master, "{name}: --closures-per-layer vs master 8b5988f");
    }
    let unit = dir.path().join("sand-unit.mdio");
    let out = run(&with(SAND, &["--no-salt"]), &unit);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("closures: per sand unit (3 units with closures, 2 multi-layer)"),
        "{stdout}"
    );
    assert_eq!(store_hash(&unit), SAND_UNIT);
    assert_ne!(SAND_UNIT, MASTER8B_SAND);
}

// Angle-stack hashes of stores written by the master ef2dc42 binary
// (closures per sand unit, unsegmented) with the same flags. PLAIN, RICH,
// MP and DEEP have no multi-layer closing unit there, so they equal the
// 8b5988f hashes; SAND is `SAND_UNIT`.
const MASTEREF_FAULTED_7: u64 = 0x7d71_8e21_c6a7_aaab;
const MASTEREF_FAULTED_4: u64 = 0xac94_e179_5801_bd3d;
/// Sandy, thin sand units, 4 faults: fault juxtaposition joins closures of
/// different units into one compartment.
const FAULTED: &[&str] = &[
    "--shape",
    "32,32,128",
    "--chunk-i",
    "16",
    "--faults",
    "4",
    "--sand-layer-fraction",
    "0.5",
    "--sand-layer-thickness",
    "1",
];
/// `FAULTED` with 3D segmentation (the default).
const SEGMENTED_FAULTED_7: u64 = 0xd928_4752_66cc_e43e;
const SEGMENTED_FAULTED_4: u64 = 0xf41a_4e80_cb38_7735;

/// `--closures-unsegmented` reproduces master ef2dc42 bit for bit; the
/// segmented default differs once a fault splits or joins closures, and
/// equals ef2dc42 when it does not.
#[test]
fn closures_unsegmented_flag_reproduces_master_ef2dc42() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, base, master, segmented_same) in [
        ("plain", PLAIN, MASTER8B_PLAIN, true),
        ("rich", RICH, MASTER8B_RICH, true),
        ("mp", MP, MASTER8B_MP, true),
        ("deep", DEEP, MASTER8B_DEEP, true),
        ("sand", SAND, SAND_UNIT, true),
        ("faulted7", &with(FAULTED, &["--seed", "7"])[..], MASTEREF_FAULTED_7, false),
        ("faulted4", &with(FAULTED, &["--seed", "4"])[..], MASTEREF_FAULTED_4, false),
    ] {
        let un = dir.path().join(format!("{name}-un.mdio"));
        let out = run(&with(base, &["--closures-unsegmented", "--no-salt"]), &un);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("unsegmented (--closures-unsegmented, master ef2dc42)"), "{name}: {stdout}");
        assert_eq!(store_hash(&un), master, "{name}: --closures-unsegmented vs master ef2dc42");
        let seg = dir.path().join(format!("{name}-seg.mdio"));
        let out = run(&with(base, &["--no-salt"]), &seg);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("3D-segmented across faults"));
        assert_eq!(store_hash(&seg) == master, segmented_same, "{name}: segmented vs ef2dc42");
    }
    for (seed, want) in [("7", SEGMENTED_FAULTED_7), ("4", SEGMENTED_FAULTED_4)] {
        let p = dir.path().join(format!("seg{seed}.mdio"));
        assert!(run(&with(FAULTED, &["--seed", seed, "--no-salt"]), &p).status.success());
        assert_eq!(store_hash(&p), want, "segmented faulted seed {seed}");
        // Tiling does not change the segmented output.
        let q = dir.path().join(format!("seg{seed}-tiles.mdio"));
        let tiled = with(&FAULTED[4..], &["--shape", "32,32,128", "--chunk-i", "3", "--chunk-j", "7", "--chunk-k", "16", "--seed", seed, "--no-salt"]);
        assert!(run(&tiled, &q).status.success());
        assert_eq!(store_hash(&q), want, "segmented faulted seed {seed}, tiles");
    }
}

// Angle-stack hashes of stores written by the master b4f4259 binary (no
// salt) with the same flags; `--no-salt` stores are byte-identical there.
const MASTERB4_PLAIN: u64 = 0x2205_9836_d641_6f98;
const MASTERB4_RICH: u64 = 0x204c_83fa_48a4_befb;
const MASTERB4_MP: u64 = 0x2841_1c79_845b_b9a5;
const MASTERB4_DEEP: u64 = 0xd1a3_74bf_b795_15e5;
const MASTERB4_SAND: u64 = 0xee90_356c_257c_d617;
const MASTERB4_FAULTED_7: u64 = 0xd928_4752_66cc_e43e;
// The same runs with the default salt body (this branch).
const SALT_PLAIN: u64 = 0x90e7_47af_ef37_a8e5;
const SALT_RICH: u64 = 0xc64a_edda_ee03_ff02;
const SALT_MP: u64 = 0x5f5f_85af_e374_3e75;
const SALT_DEEP: u64 = 0x3144_ba5f_39b5_5ee1;
const SALT_SAND: u64 = 0x5d8a_0071_9fa0_731d;
const SALT_FAULTED_7: u64 = 0x3980_7024_06ef_ee2a;

/// `--no-salt` reproduces master b4f4259 bit for bit (no `salt_labels`
/// array); the default adds the salt body, writes `data/salt_labels` equal
/// to the library salt body, and changes the output.
#[test]
fn no_salt_flag_reproduces_master_b4f4259() {
    let dir = tempfile::tempdir().expect("tempdir");
    let faulted7 = with(FAULTED, &["--seed", "7"]);
    for (name, base, master, salted, voxels) in [
        ("plain", PLAIN, MASTERB4_PLAIN, SALT_PLAIN, 212),
        ("rich", RICH, MASTERB4_RICH, SALT_RICH, 212),
        ("mp", MP, MASTERB4_MP, SALT_MP, 0),
        ("deep", DEEP, MASTERB4_DEEP, SALT_DEEP, 2996),
        ("sand", SAND, MASTERB4_SAND, SALT_SAND, 4083),
        ("faulted7", &faulted7[..], MASTERB4_FAULTED_7, SALT_FAULTED_7, 2336),
    ] {
        let off = dir.path().join(format!("{name}-off.mdio"));
        let out = run(&with(base, &["--no-salt"]), &off);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("salt: off (--no-salt, master b4f4259)"));
        assert_eq!(store_hash(&off), master, "{name}: --no-salt vs master b4f4259");
        assert!(!off.join("data").join("salt_labels").exists(), "{name}: no salt_labels without salt");
        let on = dir.path().join(format!("{name}-on.mdio"));
        let out = run(base, &on);
        assert!(out.status.success(), "{name}: {out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains(&format!("{voxels} voxels in")), "{name}: {stdout}");
        assert_eq!(store_hash(&on), salted, "{name}: salt default");
        assert_ne!(salted, master);
        let mask = synthoseis_io::MdioStore::open(&on).unwrap().read_salt_labels_u8().unwrap();
        assert_eq!(mask.iter().map(|&v| v as usize).sum::<usize>(), voxels, "{name}: salt_labels");
    }
}
