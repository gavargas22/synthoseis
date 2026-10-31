//! Fault heave evidence (spec-fault-kinematics 7a §12): dumps the stack,
//! layer labels and fault labels of a legacy and a heave store as raw files
//! for `plot_heave_evidence.py`, and prints per-fault throw / heave
//! percentiles of the rebuilt fault model:
//!
//! ```text
//! synthoseis run --e2e --chunked --faults 4 --seed 7 --shape 128,128,256 --legacy-fault-kinematics --store /tmp/l.mdio
//! synthoseis run --e2e --chunked --faults 4 --seed 7 --shape 128,128,256 --store /tmp/h.mdio
//! cargo run --release -p synthoseis-core --example heave_evidence -- /tmp/l.mdio /tmp/h.mdio 7 4 /tmp/fig
//! python rust/synthoseis-core/examples/plot_heave_evidence.py /tmp/fig OUT_PREFIX
//! ```
//!
//! Stats are over the hanging-wall voxels of each fault's label band
//! (`|f − 1| / |∇f| ≤ LABEL_BAND_VOXELS`) with throw > 1 sample; "peak" is
//! the voxel of largest throw.
use std::io::Write;
use synthoseis_geo::faults::{stage_slip, FaultKinematics, LABEL_BAND_VOXELS};

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (r.floor() as usize, r.ceil() as usize);
    v[lo] + (r - lo as f64) * (v[hi] - v[lo])
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (legacy, heave, seed, count, out) = (&a[1], &a[2], a[3].parse::<u64>().unwrap(), a[4].parse::<usize>().unwrap(), &a[5]);
    std::fs::create_dir_all(out).unwrap();
    let w = |n: &str, b: &[u8]| std::fs::File::create(format!("{out}/{n}")).unwrap().write_all(b).unwrap();
    let mut shape = [0; 3];
    for (tag, p) in [("legacy", legacy), ("heave", heave)] {
        let s = synthoseis_io::MdioStore::open(std::path::Path::new(p)).unwrap();
        shape = s.shape();
        let v: Vec<u8> = s.read_volume().unwrap().iter().flat_map(|x| x.to_le_bytes()).collect();
        w(&format!("{tag}_stack.f32"), &v);
        w(&format!("{tag}_labels.u8"), &s.read_labels_u8().unwrap());
        w(&format!("{tag}_faults.u8"), &s.read_fault_labels_u8().unwrap());
    }
    let [ni, nj, nk] = shape;
    w("shape.txt", format!("{ni} {nj} {nk}").as_bytes());

    let cfg = synthoseis_core::pipeline::E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        faults: synthoseis_core::FaultConfig::with_count(count),
        ..Default::default()
    };
    let m = synthoseis_core::pipeline_stream::fault_model(&cfg).expect("faulted model");
    let FaultKinematics::Heave(p) = m.kinematics() else { panic!("expected heave") };
    let top = (nk - 1) as f64;
    let (mut all_t, mut all_h) = (Vec::new(), Vec::new());
    let mut stats = String::new();
    for (n, f) in m.faults().iter().enumerate() {
        let (mut t, mut h) = (Vec::new(), Vec::new());
        let mut peak = (0.0f64, 0.0f64);
        for i in 0..ni {
            for j in 0..nj {
                let lat = f.lateral(i, j);
                for k in 0..nk {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    let g = f.geometry.gradient_at(x, y, z);
                    let gn = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                    if !(gn > 0.0) || (f.geometry.value_at(x, y, z) - 1.0).abs() / gn > LABEL_BAND_VOXELS {
                        continue;
                    }
                    let (_, hv, d) = stage_slip(f, &p, x, y, k, lat, top);
                    if d <= 1.0 {
                        continue;
                    }
                    let hm = hv[0].hypot(hv[1]);
                    if d > peak.0 {
                        peak = (d, hm);
                    }
                    t.push(d);
                    h.push(hm);
                }
            }
        }
        all_t.extend(&t);
        all_h.extend(&h);
        let line = format!(
            "fault {}: n={} throw p5/50/95/max {:.1}/{:.1}/{:.1}/{:.1} samples, heave p5/50/95/max {:.1}/{:.1}/{:.1}/{:.1} bins, at peak throw {:.1} -> heave {:.1}\n",
            n + 1, t.len(), pct(&mut t, 5.0), pct(&mut t, 50.0), pct(&mut t, 95.0), pct(&mut t, 100.0),
            pct(&mut h, 5.0), pct(&mut h, 50.0), pct(&mut h, 95.0), pct(&mut h, 100.0), peak.0, peak.1
        );
        stats.push_str(&line);
    }
    stats.push_str(&format!(
        "all: n={} throw p5/50/95/max {:.1}/{:.1}/{:.1}/{:.1}, heave p5/50/95/max {:.1}/{:.1}/{:.1}/{:.1}\n",
        all_t.len(), pct(&mut all_t, 5.0), pct(&mut all_t, 50.0), pct(&mut all_t, 95.0), pct(&mut all_t, 100.0),
        pct(&mut all_h, 5.0), pct(&mut all_h, 50.0), pct(&mut all_h, 95.0), pct(&mut all_h, 100.0)
    ));
    // Sanity: rebuilt model labels vs the heave store's fault labels.
    let store = synthoseis_io::MdioStore::open(std::path::Path::new(heave)).unwrap().read_fault_labels_u8().unwrap();
    let mut same = 0usize;
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                same += (m.gather(i, j, k).id == store[(i * nj + j) * nk + k]) as usize;
            }
        }
    }
    stats.push_str(&format!("model/store fault-label agreement {:.4}\n", same as f64 / (ni * nj * nk) as f64));
    print!("{stats}");
    w("stats.txt", stats.as_bytes());
}
