//! CLI depth-to-time flags (spec "depth-to-time conversion" §2, §4):
//! `--legacy-depth-as-time` reproduces master f3720fb2 stores bit for bit
//! (single process and multi-process), time output is the default, the
//! time flags reach multi-process workers, invalid combinations exit 2, and
//! the MDIO attributes, summary and staircase warning are reported.
use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], store: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthoseis"))
        .args(["run", "--e2e"])
        .args(whole_voxels(args))
        .arg("--store")
        .arg(store)
        .output()
        .expect("spawn synthoseis")
}

/// Partial voxels are on by default since PR B2: the goldens here are
/// whole-voxel stores, so every layered run adds `--legacy-whole-voxels`
/// (partial-voxels spec §5.4: unchanged hashes with the opt-out). Planar,
/// `--legacy-toy-depth` and runs that pick a voxel mode are left alone.
fn whole_voxels<'a>(args: &[&'a str]) -> Vec<&'a str> {
    let mut v = args.to_vec();
    let picks = |a: &&str| {
        matches!(
            *a,
            "planar"
                | "--legacy-toy-depth"
                | "--partial-voxel-reflectivity"
                | "--legacy-whole-voxels"
        )
    };
    if !args.iter().any(picks) {
        v.push("--legacy-whole-voxels");
    }
    v
}

fn with<'a>(base: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    base.iter().chain(extra).copied().collect()
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

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const PLAIN: &[&str] = &["--chunked", "--shape", "12,10,64", "--chunk-i", "5"];
const RICH: &[&str] = &[
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
];
const MP: &[&str] = &[
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
];

// Angle-stack hashes of stores written by the master 0eb937b5 binary with
// the same flags (unchanged at master f3720fb2: #38 touches only
// fault_labels, which `depth_to_time_pipeline.rs` pins) (`RICH` there with `--bandpass-trailing-sample` too, the
// variant the legacy-axis option exists for, and without it). PLAIN,
// RICH + trailing and MP equal the salt goldens of `rock_physics_cli.rs`
// (`SALT_*`), which are master's default since #34.
const MASTERD2T_PLAIN: u64 = 0x90e7_47af_ef37_a8e5;
const MASTERD2T_RICH: u64 = 0xc747_5878_6ba7_0f95;
const MASTERD2T_RICH_TRAILING: u64 = 0xc64a_edda_ee03_ff02;
const MASTERD2T_MP: u64 = 0x5f5f_85af_e374_3e75;
// The same runs in time mode (the default since this branch).
const TIME_PLAIN: u64 = 0x6aa8_1542_40b2_94e6;
const TIME_RICH: u64 = 0x02d9_a61d_29bb_4de2;
const TIME_MP: u64 = 0x5f34_69f1_e77e_3494;

#[test]
fn legacy_depth_as_time_reproduces_master_f3720fb2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let rich_trailing = with(RICH, &["--bandpass-trailing-sample"]);
    for (name, base, master, time) in [
        ("plain", PLAIN, MASTERD2T_PLAIN, Some(TIME_PLAIN)),
        ("rich", RICH, MASTERD2T_RICH, Some(TIME_RICH)),
        ("rich-trailing", &rich_trailing[..], MASTERD2T_RICH_TRAILING, None),
        ("mp", MP, MASTERD2T_MP, Some(TIME_MP)),
    ] {
        let legacy = dir.path().join(format!("{name}-legacy.mdio"));
        let out = run(&with(base, &["--legacy-depth-as-time"]), &legacy);
        assert!(out.status.success(), "{name}: {out:?}");
        assert!(!stdout(&out).contains("time axis:"), "{name}: legacy prints no time summary");
        assert_eq!(store_hash(&legacy), master, "{name}: --legacy-depth-as-time vs master f3720fb2");
        // No new root attributes on the legacy axis; digi stays 4 ms.
        let s = synthoseis_io::MdioStore::open(&legacy).unwrap();
        let attrs = s.root_attrs().unwrap();
        for k in ["time_conversion", "depth_step_m", "twt_kernel"] {
            assert!(attrs.get(k).is_none(), "{name}: legacy store has {k}");
        }
        assert_eq!(s.config().digi, 4.0);
        let Some(time) = time else { continue };
        let default = dir.path().join(format!("{name}-time.mdio"));
        let out = run(base, &default);
        assert!(out.status.success(), "{name}: {out:?}");
        assert_eq!(store_hash(&default), time, "{name}: time-mode default");
        assert_ne!(time, master, "{name}: time mode must differ from the legacy axis");
    }
}

/// Time-mode stores: `digi = dt`, the conversion attributes, the time-domain
/// shape; the summary reports the axis and the short / long columns.
#[test]
fn time_mode_attrs_and_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, args, nt, dt, kernel) in [
        ("default", PLAIN.to_vec(), 64, 4.0, "sinc"),
        ("dt2-long-linear", with(RICH, &["--dt-ms", "2", "--twt-samples", "150", "--twt-kernel", "linear"]), 150, 2.0, "linear"),
        ("classic", vec![], 8, 4.0, "sinc"),
        ("strip", vec!["--chunked", "--workers", "3", "--chunk-i", "2"], 8, 4.0, "sinc"),
        ("overlap", vec!["--chunked", "--overlap", "--chunk-i", "4"], 8, 4.0, "sinc"),
        ("mp", MP.to_vec(), 8, 4.0, "sinc"),
    ] {
        let p = dir.path().join(format!("{name}.mdio"));
        let out = run(&args, &p);
        assert!(out.status.success(), "{name}: {out:?}");
        let s = synthoseis_io::MdioStore::open(&p).unwrap();
        assert_eq!(s.shape()[2], nt, "{name}: samples");
        assert_eq!(s.config().digi, dt, "{name}: digi");
        let attrs = s.root_attrs().unwrap();
        assert_eq!(attrs["time_conversion"], "vp-twt", "{name}");
        assert_eq!(attrs["depth_step_m"], 4.0, "{name}");
        assert_eq!(attrs["twt_kernel"], kernel, "{name}");
        let so = stdout(&out);
        assert!(so.contains(&format!("time axis: two-way time from voxel Vp (vp-twt), dt={dt} ms, nt={nt}, kernel={kernel}")), "{name}: {so}");
        assert!(so.contains("time columns: base TWT") && so.contains("% (worst shortfall") && so.contains("% (worst excess"), "{name}: {so}");
    }
}

/// `--dt-ms`, `--twt-samples` and `--twt-kernel` reach multi-process
/// workers: bit-identical to one process with the same chunking.
#[test]
fn time_flags_reach_multiprocess_workers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut seen = Vec::new();
    for flags in [
        &[][..],
        &["--dt-ms", "2"][..],
        &["--twt-samples", "29"][..],
        &["--twt-kernel", "linear"][..],
        &["--legacy-depth-as-time"][..],
    ] {
        let single = dir.path().join("single.mdio");
        let out = run(&with(&with(&MP[..1], &MP[4..]), flags), &single);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        let mp = dir.path().join("mp.mdio");
        let out = run(&with(MP, flags), &mp);
        assert!(out.status.success(), "{flags:?}: {out:?}");
        assert_eq!(store_hash(&single), store_hash(&mp), "{flags:?}: multiprocess vs single");
        seen.push(store_hash(&mp));
    }
    for i in 0..seen.len() {
        for j in i + 1..seen.len() {
            assert_ne!(seen[i], seen[j], "flag sets {i} and {j} must differ");
        }
    }
}

#[test]
fn invalid_time_flags_exit_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (bad, msg) in [
        (["--legacy-depth-as-time", "--dt-ms", "2"].as_slice(), "have no effect with --legacy-depth-as-time or --legacy-toy-depth"),
        (&["--legacy-depth-as-time", "--twt-samples", "80"], "have no effect with --legacy-depth-as-time"),
        (&["--legacy-depth-as-time", "--twt-kernel", "linear"], "have no effect with --legacy-depth-as-time"),
        (&["--legacy-toy-depth", "--dt-ms", "2"], "--legacy-toy-depth"),
        (&["--legacy-toy-depth", "--twt-kernel", "sinc"], "--legacy-toy-depth"),
        (&["--bandpass", "4,30", "--bandpass-trailing-sample"], "requires --legacy-depth-as-time"),
        (&["--dt-ms", "9"], "outside 0.5-8.0"),
        (&["--dt-ms", "0.25"], "outside 0.5-8.0"),
        (&["--dt-ms", "4.5"], "too coarse"),
        (&["--bandpass", "4,30", "--keep-ricker", "--dt-ms", "8"], "too coarse"),
        (&["--twt-samples", "15"], "twt-samples 15 outside 16..=512"),
        (&["--twt-samples", "513"], "outside 16..=512"),
        (&["--twt-kernel", "nearest"], "--twt-kernel expects sinc or linear"),
        // #38's fault-label switch keeps its exit-2 rules in time mode.
        (&["--fault-labels-through-salt"], "--fault-labels-through-salt has no effect without --faults"),
        (&["--fault-labels-through-salt", "--faults", "2", "--no-salt"], "--fault-labels-through-salt has no effect with --no-salt"),
        (&["--fault-labels-through-salt", "--faults", "2", "--toy-geometry", "planar"], "has no effect with the planar geometry"),
        (&["--fault-labels-through-salt", "--faults", "2", "--legacy-toy-depth"], "--legacy-toy-depth"),
    ] {
        let out = run(&with(PLAIN, bad), &dir.path().join("bad.mdio"));
        assert_eq!(out.status.code(), Some(2), "{bad:?}: {out:?}");
        assert!(stderr(&out).contains(msg), "{bad:?}: {}", stderr(&out));
    }
    // Accepted: both legacy switches together; dt 8 ms with the bandpass alone.
    for ok in [
        ["--legacy-depth-as-time", "--legacy-toy-depth"].as_slice(),
        &["--bandpass", "4,30", "--dt-ms", "8"],
        &["--legacy-depth-as-time", "--bandpass", "4,30", "--bandpass-trailing-sample"],
        &["--fault-labels-through-salt", "--faults", "2"],
        &["--legacy-depth-as-time", "--fault-labels-through-salt", "--faults", "2"],
    ] {
        let out = run(&with(PLAIN, ok), &dir.path().join("ok.mdio"));
        assert!(out.status.success(), "{ok:?}: {out:?}");
    }
}

/// Fault-label salt mask in time mode (#38 + #39, spec §8), seed 7 at
/// 24 × 24 × 128 with 3 faults: the time-domain `fault_labels` are `fault
/// AND NOT salt`, the mask removes exactly the 79 fault ∩ salt voxels of
/// the `--fault-labels-through-salt` store (which the summary reports as
/// `masked_in_salt=79`, counted on the time cube), the angle stack is
/// unchanged. (The CLI takes `--faults` on the single-worker chunked path
/// only; the library test `time_mode_fault_salt_mask_removed_count_every_path`
/// covers the other writers.)
#[test]
fn fault_label_salt_mask_in_time_mode() {
    const FAULTED7: &[&str] = &["--chunked", "--shape", "24,24,128", "--seed", "7", "--faults", "3", "--chunk-i", "5"];
    let dir = tempfile::tempdir().expect("tempdir");
    let read = |p: &Path| {
        let s = synthoseis_io::MdioStore::open(p).unwrap();
        (s.read_fault_labels_u8().unwrap(), s.read_salt_labels_u8().unwrap(), s.config().digi)
    };
    let masked_p = dir.path().join("masked.mdio");
    let out = run(FAULTED7, &masked_p);
    assert!(out.status.success(), "{out:?}");
    assert!(stdout(&out).contains("masked_in_salt=79 "), "{}", stdout(&out));
    let through_p = dir.path().join("through.mdio");
    let out = run(&with(FAULTED7, &["--fault-labels-through-salt"]), &through_p);
    assert!(out.status.success(), "{out:?}");
    assert!(stdout(&out).contains("(--fault-labels-through-salt)"), "{}", stdout(&out));
    let (masked, salt, digi) = read(&masked_p);
    let (through, salt2, _) = read(&through_p);
    assert_eq!(digi, 4.0);
    assert_eq!(salt, salt2);
    let overlap = through.iter().zip(&salt).filter(|(&f, &s)| f == 1 && s == 1).count();
    let removed = through.iter().map(|&v| v as usize).sum::<usize>() - masked.iter().map(|&v| v as usize).sum::<usize>();
    assert_eq!((removed, overlap), (79, 79));
    for v in 0..masked.len() {
        assert_eq!(masked[v] == 1, through[v] == 1 && salt[v] == 0, "voxel {v}");
    }
    assert_eq!(store_hash(&masked_p), store_hash(&through_p), "the mask is label-only");
}

/// The depth-staircase constraint is a warning, not an error (spec §3.3):
/// dt = 2 ms on the 4 m depth grid; dt = 4 ms is quiet. `--gpu` in time
/// mode logs the CPU fallback (spec §6).
#[test]
fn staircase_warning_and_gpu_fallback_log() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = run(&with(PLAIN, &["--dt-ms", "2"]), &dir.path().join("w.mdio"));
    assert!(out.status.success(), "{out:?}");
    assert!(stderr(&out).contains("warning: depth step 4 m > Vp_min 1580 m/s x dt / 1.2 = 2.63 m"), "{}", stderr(&out));
    let out = run(PLAIN, &dir.path().join("q.mdio"));
    assert!(!stderr(&out).contains("warning: depth step"), "{}", stderr(&out));
    let gpu = dir.path().join("gpu.mdio");
    let out = run(&with(PLAIN, &["--gpu"]), &gpu);
    assert!(out.status.success(), "{out:?}");
    assert!(stderr(&out).contains("depth-to-time mode fuses on the CPU"), "{}", stderr(&out));
    assert_eq!(store_hash(&gpu), TIME_PLAIN, "--gpu time mode = CPU time mode");
}
