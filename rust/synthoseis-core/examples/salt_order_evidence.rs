//! Salt / fault order metrics and figure dumps (spec-salt-fault-order §4):
//!
//! ```text
//! cargo run --release -p synthoseis-core --example salt_order_evidence -- metrics 1 20 [OUT_DIR]
//! ```
//!
//! prints one TSV row per seed (128×128×256, 4 faults): exposed voxels
//! (post-fault / master), stair index (post-fault / master / faults off),
//! near-salt Jaccard vs master and the warp checks. With `OUT_DIR` it also
//! dumps the three label cubes of seed `lo` as raw u8 for
//! `plot_salt_order_evidence.py`.
#[path = "../tests/salt_order_metrics/mod.rs"]
mod m;
use m::Mode;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lo, hi) = (a[2].parse::<u64>().unwrap(), a[3].parse::<u64>().unwrap());
    let out = a.get(4);
    let shape = [128, 128, 256];
    println!("seed\texp_post\texp_master\tmaster_frac\tstair_post\tstair_master\tstair_off\tpost/off\tpost/master\tjaccard\tworst\tmean_lift\tnonmono\tunset");
    for seed in lo..=hi {
        let cp = m::cfg(seed, shape, 4, Mode::PostFault);
        let Some(s) = m::salt(&cp) else {
            println!("{seed}\tno salt");
            continue;
        };
        let rp = m::run(&cp);
        let rm = m::run(&m::cfg(seed, shape, 4, Mode::Master));
        let ro = m::run(&m::cfg(seed, shape, 4, Mode::FaultsOff));
        let und = m::undragged(seed, shape);
        let (ep, em) = (m::exposed(&rp, &s, &und), m::exposed(&rm, &s, &und));
        let nonsalt = (shape[0] * shape[1] * shape[2]) - s.voxels(shape[2]);
        let (sp, sm, so) = (
            m::stair_index(&rp, &s),
            m::stair_index(&rm, &s),
            m::stair_index(&ro, &s),
        );
        let j = m::jaccard(&rp, &rm, &s);
        let (w, l, nm, un) = m::warp_checks(&rp, &s);
        println!(
            "{seed}\t{ep}\t{em}\t{:.4}%\t{sp:.5}\t{sm:.5}\t{so:.5}\t{:.3}\t{:.3}\t{j:.4}\t{w}\t{l:.2}\t{nm}\t{un}",
            100.0 * em as f64 / nonsalt as f64,
            sp / so,
            sp / sm
        );
        if let (Some(o), true) = (out, seed == lo) {
            std::fs::create_dir_all(o).unwrap();
            let w = |n: &str, b: &[u8]| {
                std::fs::File::create(format!("{o}/{n}"))
                    .unwrap()
                    .write_all(b)
                    .unwrap()
            };
            w("post_labels.u8", &rp.labels);
            w("master_labels.u8", &rm.labels);
            w("off_labels.u8", &ro.labels);
            w("faults.u8", &rp.fault_id);
            let sm: Vec<u8> = (0..shape[0] * shape[1])
                .flat_map(|q| (0..shape[2]).map(move |k| (q, k)))
                .map(|(q, k)| s.contains(q, k) as u8)
                .collect();
            w("salt.u8", &sm);
            w(
                "shape.txt",
                format!("{} {} {}", shape[0], shape[1], shape[2]).as_bytes(),
            );
        }
    }
}
