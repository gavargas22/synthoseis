//! Fault heave CLI (fault kinematics spec, PR 7a): heave output does not
//! depend on the tile layout, and the exit-2 rules for
//! `--fault-min-dip` / `--legacy-fault-kinematics`.

use std::path::Path;
use std::process::{Command, Output};
use synthoseis_io::MdioStore;

fn run(args: &[&str], store: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_synthoseis"))
        .args(["run", "--e2e"])
        .args(args)
        .arg("--store")
        .arg(store)
        .output()
        .expect("spawn synthoseis")
}

/// Decoded arrays of a store (the on-disk chunk layout follows the tiles,
/// so the comparison is on values, not files).
fn decoded(store: &Path) -> (Vec<u32>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let s = MdioStore::open(store).expect("open store");
    let vol = s
        .read_volume()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (
        vol,
        s.read_labels_u8().unwrap(),
        s.read_fault_labels_u8().unwrap(),
        s.read_unconformity_labels_u8().unwrap(),
    )
}

const BASE: &[&str] = &["--chunked", "--shape", "32,32,128", "--faults", "4"];

#[test]
fn heave_output_is_tile_invariant() {
    let dir = tempfile::tempdir().unwrap();
    for seed in ["1", "7", "13"] {
        let mut a: Vec<&str> = BASE.to_vec();
        a.extend(["--seed", seed]);
        let mut b = a.clone();
        b.extend(["--chunk-i", "5", "--chunk-j", "7"]);
        let (pa, pb) = (
            dir.path().join(format!("a{seed}")),
            dir.path().join(format!("b{seed}")),
        );
        let oa = run(&a, &pa);
        assert!(
            oa.status.success(),
            "{}",
            String::from_utf8_lossy(&oa.stderr)
        );
        assert!(
            String::from_utf8_lossy(&oa.stdout).contains("fault kinematics: heave, min dip 30 deg")
        );
        let ob = run(&b, &pb);
        assert!(
            ob.status.success(),
            "{}",
            String::from_utf8_lossy(&ob.stderr)
        );
        assert!(
            decoded(&pa) == decoded(&pb),
            "seed {seed}: tiles changed the output"
        );
    }
}

#[test]
fn bad_kinematics_flags_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let cases: &[&[&str]] = &[
        &["--faults", "4", "--fault-min-dip", "10"],
        &["--faults", "4", "--fault-min-dip", "61"],
        &[
            "--faults",
            "4",
            "--fault-min-dip",
            "40",
            "--legacy-fault-kinematics",
        ],
        &["--faults", "0", "--fault-min-dip", "40"],
        &["--faults", "0", "--legacy-fault-kinematics"],
        &["--fault-min-dip", "40"],
        &["--legacy-fault-kinematics"],
    ];
    for (k, extra) in cases.iter().enumerate() {
        let mut a = vec!["--chunked", "--shape", "16,16,64"];
        a.extend_from_slice(extra);
        let o = run(&a, &dir.path().join(format!("s{k}")));
        assert_eq!(
            o.status.code(),
            Some(2),
            "{extra:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    for dip in ["15", "60"] {
        let a = [
            "--chunked",
            "--shape",
            "16,16,64",
            "--faults",
            "2",
            "--fault-min-dip",
            dip,
        ];
        let o = run(&a, &dir.path().join(format!("ok{dip}")));
        assert!(
            o.status.success(),
            "dip {dip}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(String::from_utf8_lossy(&o.stdout).contains(&format!("min dip {dip} deg")));
    }
}
