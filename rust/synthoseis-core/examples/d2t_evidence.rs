//! Depth-to-time evidence dump (PR B figures).
//!
//! `cargo run --release -p synthoseis-core --example d2t_evidence -- OUT SEED
//! NI NJ NZ [--no-salt] [--faults N]` writes, for that cube, the depth-domain
//! inputs and both deliverables:
//!
//! * `<tag>_depth_labels.u8`, `<tag>_depth_salt.u8`, `<tag>_depth_faults.u8`
//!   (`ni·nj·nz`), `<tag>_twt.f32` (`ni·nj·(nz+1)` cell-boundary times, ms);
//! * `<tag>_time_rfc.f32`: the production time-mode fuse with no wavelet
//!   (raw two-way-time reflectivity, `ni·nj·nt`);
//! * `<tag>_{legacy,time}_{stack,labels,salt,faults}` (output shape);
//! * `<tag>.json` with the shapes and the time axis.
//!
//! `<tag>` is `seed<S>[_nosalt][_faults<N>]`. Missing label cubes are skipped.

use std::path::{Path, PathBuf};

use synthoseis_core::pipeline::{generate_tiny_cube, E2eConfig};
use synthoseis_core::{FaultConfig, TimeConfig};

fn write(dir: &Path, name: String, bytes: Vec<u8>) {
    std::fs::write(dir.join(name), bytes).expect("write");
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&a[0]);
    std::fs::create_dir_all(&dir).expect("out dir");
    let p = |k: usize| a[k].parse::<u64>().expect("number");
    let (seed, ni, nj, nz) = (p(1), p(2) as usize, p(3) as usize, p(4) as usize);
    let no_salt = a.iter().any(|x| x == "--no-salt");
    let faults = a.iter().position(|x| x == "--faults").map(|k| a[k + 1].parse::<usize>().unwrap()).unwrap_or(0);
    let mut base = E2eConfig {
        seed,
        inline_count: ni,
        crossline_count: nj,
        samples: nz,
        faults: FaultConfig {
            count: faults,
            ..FaultConfig::default()
        },
        ..E2eConfig::default()
    };
    base.rock_physics.salt = !no_salt;
    let tag = format!(
        "seed{seed}{}{}",
        if no_salt { "_nosalt" } else { "" },
        if faults > 0 { format!("_faults{faults}") } else { String::new() }
    );

    // Depth-domain inputs.
    let (labels, shape) = synthoseis_core::pipeline_stream::generate_labels(&base);
    let model = synthoseis_core::rock_physics::elastic_model(&base, &labels, shape);
    let axis = base.time_axis().expect("time mode");
    let twt = synthoseis_core::time_mode::tile_twt(&model, &labels, shape, 0, ni, 0, nj, &axis);
    write(&dir, format!("{tag}_twt.f32"), f32s(&twt.iter().map(|&x| x as f32).collect::<Vec<_>>()));
    // Production time-mode fuse without a wavelet: the raw two-way-time
    // reflectivity at the default incidence (sub-sample pull-up picks).
    let mut rfc = vec![0.0f32; ni * nj * axis.nt];
    synthoseis_core::pipeline_stream::fuse_tile_local(
        &labels,
        shape,
        0,
        ni,
        0,
        nj,
        &model,
        synthoseis_gpu::NO_WAVELET,
        synthoseis_core::pipeline_stream::DEFAULT_INCIDENCE_DEG,
        &mut rfc,
        &mut synthoseis_core::pipeline_stream::WorkingSetStats::default(),
    );
    write(&dir, format!("{tag}_time_rfc.f32"), f32s(&rfc));
    write(&dir, format!("{tag}_depth_labels.u8"), labels);
    if let Some(m) = synthoseis_core::salt::generate_salt_labels(&base) {
        write(&dir, format!("{tag}_depth_salt.u8"), m);
    }
    if let Some(m) = synthoseis_core::pipeline_stream::generate_fault_labels(&base) {
        write(&dir, format!("{tag}_depth_faults.u8"), m);
    }

    // Deliverables on the legacy axis and in time.
    let mut shapes = Vec::new();
    for (mode, time) in [("legacy", TimeConfig::legacy()), ("time", TimeConfig::default())] {
        let cfg = E2eConfig { time, ..base.clone() };
        let v = generate_tiny_cube(&cfg);
        write(&dir, format!("{tag}_{mode}_stack.f32"), f32s(&v.angle_stack));
        write(&dir, format!("{tag}_{mode}_labels.u8"), v.labels);
        if let Some(m) = synthoseis_core::time_mode::generate_salt_labels_output(&cfg) {
            write(&dir, format!("{tag}_{mode}_salt.u8"), m);
        }
        if let Some(m) = synthoseis_core::time_mode::generate_fault_labels_output(&cfg) {
            write(&dir, format!("{tag}_{mode}_faults.u8"), m);
        }
        shapes.push(format!("\"{mode}_shape\": {:?}", v.shape));
    }
    let meta = format!(
        "{{\"seed\": {seed}, \"salt\": {}, \"faults\": {faults}, \"depth_shape\": {:?}, \"dz\": {}, \"dt_ms\": {}, \"nt\": {}, \"digi_legacy\": {}, {}}}\n",
        !no_salt,
        shape,
        axis.dz,
        axis.dt_ms,
        axis.nt,
        E2eConfig { time: TimeConfig::legacy(), ..base.clone() }.digi_ms(),
        shapes.join(", ")
    );
    write(&dir, format!("{tag}.json"), meta.clone().into_bytes());
    println!("{tag}: {meta}");
}
