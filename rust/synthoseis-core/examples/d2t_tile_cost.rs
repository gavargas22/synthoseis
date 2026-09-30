//! Per-tile cost of the depth-to-time conversion core (spec §6 target: the
//! fuse tile ≤ 1.3× its legacy-mode time).
//!
//! PR A has no time-mode pipeline path, so this times a prototype of the
//! time-mode fuse tile built from the PR A kernels against the production
//! legacy fuse tile (`fuse_tile_local`, default rock physics):
//!
//! * legacy: `tile_properties` → Zoeppritz in depth → 17-tap Ricker (depth
//!   index as time), i.e. exactly `fuse_tile_local`;
//! * time mode: `tile_properties` → per column `reflectivity_time_column`
//!   (T prefix sum, Zoeppritz on the depth interfaces, band-limited
//!   insertion at T_{k+1}) → the same Ricker on the time trace.
//!
//! Usage: `cargo run --release -p synthoseis-core --example d2t_tile_cost --
//! [ni nj nk tile reps seed]` (defaults 64 64 256 16 5 7).

use std::time::Instant;

use synthoseis_core::pipeline::E2eConfig;
use synthoseis_core::pipeline_stream::fuse_tile_local;
use synthoseis_core::{elastic_model, generate_labels, TimeConfig, WorkingSetStats};
use synthoseis_seismic::{convolve_same_1d, reflectivity_time_column, ricker, TwtKernel, TwtScratch};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let arg = |i: usize, d: usize| a.get(i).copied().unwrap_or(d);
    let (ni, nj, nk, tile, reps, seed) = (arg(0, 64), arg(1, 64), arg(2, 256), arg(3, 16), arg(4, 5), arg(5, 7));
    let cfg = E2eConfig {
        seed: seed as u64,
        inline_count: ni,
        crossline_count: nj,
        samples: nk,
        ..E2eConfig::default()
    };
    let (labels, shape) = generate_labels(&cfg);
    let model = elastic_model(&cfg, &labels, shape);
    let form = model.zoeppritz_form();
    let dz = cfg.rock_physics.depth_step_m;
    let tc = TimeConfig::default();
    let nt = tc.output_samples(nk, dz);
    let wav = ricker(40.0, tc.dt_ms, 1);
    let angle = 15.0;
    let tiles: Vec<(usize, usize, usize, usize)> = (0..ni)
        .step_by(tile)
        .flat_map(|i0| (0..nj).step_by(tile).map(move |j0| (i0, (i0 + tile).min(ni), j0, (j0 + tile).min(nj))))
        .collect();

    // Legacy fuse tile.
    let legacy = || {
        let mut stats = WorkingSetStats::default();
        let mut sum = 0.0f64;
        for &(i0, i1, j0, j1) in &tiles {
            let mut out = vec![0.0f32; (i1 - i0) * (j1 - j0) * nk];
            fuse_tile_local(&labels, shape, i0, i1, j0, j1, &model, &wav, angle, &mut out, &mut stats);
            sum += out[out.len() / 2] as f64;
        }
        sum
    };
    // Time-mode prototype; also returns (props, conversion, convolution) seconds.
    let time_mode = |kernel: TwtKernel| {
        let (mut t_props, mut t_conv_in, mut t_wav) = (0.0f64, 0.0f64, 0.0f64);
        let mut sum = 0.0f64;
        let mut scratch = TwtScratch::default();
        let mut x = vec![0.0f64; nt];
        for &(i0, i1, j0, j1) in &tiles {
            let n = (i1 - i0) * (j1 - j0) * nk;
            let (mut vp, mut vs, mut rho) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            let s0 = Instant::now();
            model.tile_properties(&labels, shape, i0, i1, j0, j1, &mut vp, &mut vs, &mut rho);
            t_props += s0.elapsed().as_secs_f64();
            let mut out = vec![0.0f32; (i1 - i0) * (j1 - j0) * nt];
            for c in 0..(i1 - i0) * (j1 - j0) {
                let r = c * nk..(c + 1) * nk;
                let s1 = Instant::now();
                reflectivity_time_column(
                    &vp[r.clone()], &vs[r.clone()], &rho[r], dz, angle, form, tc.dt_ms, kernel, &mut scratch, &mut x,
                );
                let s2 = Instant::now();
                let y = convolve_same_1d(&x, &wav);
                for (o, v) in out[c * nt..(c + 1) * nt].iter_mut().zip(&y) {
                    *o = *v as f32;
                }
                t_conv_in += (s2 - s1).as_secs_f64();
                t_wav += s2.elapsed().as_secs_f64();
            }
            sum += out[out.len() / 2] as f64;
        }
        (sum, t_props, t_conv_in, t_wav)
    };

    let best = |f: &dyn Fn() -> f64| {
        (0..reps)
            .map(|_| {
                let s = Instant::now();
                std::hint::black_box(f());
                s.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    // Warm up.
    legacy();
    time_mode(TwtKernel::Sinc);
    let t_leg = best(&|| legacy());
    let t_sinc = best(&|| time_mode(TwtKernel::Sinc).0);
    let t_lin = best(&|| time_mode(TwtKernel::Linear).0);
    let (_, p, c, w) = time_mode(TwtKernel::Sinc);
    println!(
        "{ni}x{nj}x{nk} (nt {nt}), {} tiles of {tile}x{tile}, angle {angle}, best of {reps}:",
        tiles.len()
    );
    println!("  legacy fuse tile total   {:8.2} ms", 1e3 * t_leg);
    println!("  time mode, sinc          {:8.2} ms  ratio {:.3}", 1e3 * t_sinc, t_sinc / t_leg);
    println!("  time mode, linear        {:8.2} ms  ratio {:.3}", 1e3 * t_lin, t_lin / t_leg);
    println!(
        "  sinc breakdown (one run): props {:.2} ms, T + Zoeppritz + insertion {:.2} ms, Ricker {:.2} ms",
        1e3 * p,
        1e3 * c,
        1e3 * w
    );
}
