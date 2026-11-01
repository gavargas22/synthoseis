//! Dump labels with fault drag off / normal / reverse (roadmap 7c figures).
//! `cargo run --release --example fault_drag_dump -- SEED NI NJ NK OUTDIR`
use synthoseis_core::pipeline::{E2eConfig, FaultConfig};
use synthoseis_core::{generate_labels, DragParams, DragSense};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (seed, out) = (a[1].parse::<u64>().unwrap(), &a[5]);
    for (tag, sense) in [("off", None), ("normal", Some(DragSense::Normal)), ("reverse", Some(DragSense::Reverse)), ("auto", Some(DragSense::Auto))] {
        let mut c = E2eConfig::tiny(seed);
        c.inline_count = p(2);
        c.crossline_count = p(3);
        c.samples = p(4);
        c.faults = FaultConfig::with_count(4);
        c.faults.drag = sense.map(|sense| DragParams { sense, ..DragParams::default() });
        let (l, sh) = generate_labels(&c);
        std::fs::write(format!("{out}/labels_{tag}.u8"), &l).unwrap();
        eprintln!("{tag}: {sh:?}");
    }
}
