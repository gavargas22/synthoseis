//! Dump the fault labels with and without the salt mask, the salt labels,
//! layer labels and angle stack of two stores (default and
//! `--fault-labels-through-salt`) as raw files for
//! `plot_fault_salt_mask.py`:
//!
//! ```text
//! synthoseis run --e2e --chunked --faults 3 --seed 7 --shape 64,64,256 --store /tmp/m.mdio
//! synthoseis run --e2e --chunked --faults 3 --seed 7 --shape 64,64,256 --fault-labels-through-salt --store /tmp/t.mdio
//! cargo run --release -p synthoseis-core --example fault_salt_dump -- /tmp/m.mdio /tmp/t.mdio /tmp/fig
//! python rust/synthoseis-core/examples/plot_fault_salt_mask.py /tmp/fig OUT.png
//! ```
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (masked, through, out) = (&a[1], &a[2], &a[3]);
    std::fs::create_dir_all(out).unwrap();
    let m = synthoseis_io::MdioStore::open(std::path::Path::new(masked)).unwrap();
    let t = synthoseis_io::MdioStore::open(std::path::Path::new(through)).unwrap();
    let w = |n: &str, b: &[u8]| {
        std::fs::File::create(format!("{out}/{n}"))
            .unwrap()
            .write_all(b)
            .unwrap()
    };
    w("fault_masked.u8", &m.read_fault_labels_u8().unwrap());
    w("fault_through.u8", &t.read_fault_labels_u8().unwrap());
    w("salt.u8", &m.read_salt_labels_u8().unwrap());
    w("labels.u8", &m.read_labels_u8().unwrap());
    let v: Vec<u8> = m
        .read_volume()
        .unwrap()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    w("stack.f32", &v);
    let [ni, nj, nk] = m.shape();
    w("shape.txt", format!("{ni} {nj} {nk}").as_bytes());
    println!("{:?}", m.shape());
}
