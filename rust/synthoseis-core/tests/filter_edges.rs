//! Physical filter edges in time mode (filter-edge spec §4, §7): water
//! above time 0, the model's own continuation below the window, reflect
//! sideways. PR-run gates on 32 × 32 × 128 cubes or smaller; the multi-seed
//! sweep, the 64 × 64 × 256 truth gate, the `edge_pad` sweep and the timing
//! comparison are nightly (`#[ignore = "nightly: ..."]`).
//!
//! Every gate prints its measured numbers (`cargo test -- --nocapture`).

use std::time::Instant;

use synthoseis_core::partial_voxels::{PartialVoxelConfig, PvReflectivity};
use synthoseis_core::pipeline::{generate_tiny_cube, E2eConfig};
use synthoseis_core::time_mode::{
    finish_trace_padded, partial_time_path, tile_twt, ChainScratch, EdgePads, TraceChain,
};
use synthoseis_core::{
    elastic_model, generate_chunked, generate_labels, generate_output_labels,
    run_e2e_geometry_once_seismic_many, run_e2e_multiprocess, run_e2e_streaming,
    run_e2e_streaming_overlapped, run_e2e_strip_stitched, FaultConfig, FilterConfig, NoiseConfig,
    RockPhysicsConfig, SeismicFilters, TimeAxis, ToyGeometry, TwtKernel, DEFAULT_INCIDENCE_DEG,
};
use synthoseis_seismic::{
    butterworth_bandpass, legacy_digitisation_ms, reflectivity_time_column_with_twt,
    subcell_reflectivity, IirFilter, SubcellColumn, TwtScratch, SINC_HALF_WIDTH,
};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The 32 × 32 × 128 demo cube of the spec's probes (layered dome, default
/// partial voxels, time mode).
fn demo(seed: u64, faults: usize) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: 32,
        crossline_count: 32,
        samples: 128,
        faults: FaultConfig::with_count(faults),
        geometry: ToyGeometry::Layered,
        ..E2eConfig::default()
    }
}

fn legacy(c: &E2eConfig) -> E2eConfig {
    let mut c = c.clone();
    c.time.legacy_filter_edges = true;
    c
}

fn whole_voxels(c: &E2eConfig) -> E2eConfig {
    let mut c = c.clone();
    c.rock_physics.partial_voxels = PartialVoxelConfig::whole_voxels();
    c
}

/// `--bandpass 4,30` (order 4, Ricker skipped), optional lateral filter.
fn bandpass(c: &E2eConfig, lateral: usize) -> E2eConfig {
    let noise = c.filters.noise.clone();
    E2eConfig {
        filters: FilterConfig {
            noise,
            ..FilterConfig::legacy(4.0, 30.0, lateral)
        },
        ..c.clone()
    }
}

fn with_noise(c: &E2eConfig, snr_db: f64, seed: u64) -> E2eConfig {
    let mut c = c.clone();
    c.filters.noise = NoiseConfig {
        snr_db: Some(snr_db),
        seed: Some(seed),
        ..NoiseConfig::default()
    };
    c
}

fn stack(c: &E2eConfig) -> Vec<f32> {
    generate_chunked(c).0.angle_stack
}

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn peak(v: &[f32]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()))
}

/// Bit-level and amplitude change of `new` against `old` (`(cols, nt)`).
#[derive(Debug)]
struct Churn {
    traces: usize,
    cells: usize,
    total: usize,
    /// Sample indices with at least one changed cell.
    samples: Vec<usize>,
    rel_rms: f64,
    max_of_peak: f64,
}

impl Churn {
    fn of(new: &[f32], old: &[f32], nt: usize) -> Self {
        assert_eq!(new.len(), old.len());
        let mut touched = vec![false; nt];
        let (mut traces, mut cells, mut d2, mut dmax) = (0, 0, 0.0f64, 0.0f64);
        for (a, b) in new.chunks_exact(nt).zip(old.chunks_exact(nt)) {
            let mut changed = false;
            for (k, (x, y)) in a.iter().zip(b).enumerate() {
                if x.to_bits() != y.to_bits() {
                    touched[k] = true;
                    changed = true;
                    cells += 1;
                }
                let d = *x as f64 - *y as f64;
                d2 += d * d;
                dmax = dmax.max(d.abs());
            }
            traces += changed as usize;
        }
        Churn {
            traces,
            cells,
            total: new.len(),
            samples: (0..nt).filter(|&k| touched[k]).collect(),
            rel_rms: (d2 / new.len() as f64).sqrt() / rms(new),
            max_of_peak: dmax / peak(new),
        }
    }

    fn cell_pct(&self) -> f64 {
        100.0 * self.cells as f64 / self.total as f64
    }

    fn print(&self, name: &str, nt: usize) {
        let (lo, hi) = (self.samples.first().copied(), self.samples.last().copied());
        println!(
            "{name}: traces changed {:.1} %, cells changed {:.2} %, samples {lo:?}–{hi:?} ({} touched), rel RMS {:.4}, max {:.1} % of peak",
            100.0 * self.traces as f64 / (self.total / nt) as f64,
            self.cell_pct(),
            self.samples.len(),
            self.rel_rms,
            100.0 * self.max_of_peak,
        );
    }
}

/// Per-column `T_nz` (ms, the model base) of the whole cube.
fn base_times(c: &E2eConfig) -> Vec<f64> {
    let (labels, shape) = generate_labels(c);
    let model = elastic_model(c, &labels, shape);
    let axis = c.time_axis().expect("time mode");
    let nz = shape[2];
    tile_twt(&model, &labels, shape, 0, shape[0], 0, shape[1], &axis)
        .chunks_exact(nz + 1)
        .map(|t| t[nz])
        .collect()
}

/// Raw time reflectivity of every column at 15° into `len`-sample buffers
/// through the production column routines: whole voxels, partial
/// `subcell` or partial `cell`, depending on `c`.
fn raw_columns(c: &E2eConfig, len: usize) -> Vec<Vec<f64>> {
    let (labels, shape) = generate_labels(c);
    let [ni, nj, nz] = shape;
    let model = elastic_model(c, &labels, shape);
    let axis = c.time_axis().expect("time mode");
    let mut out = Vec::with_capacity(ni * nj);
    if let Some(m) = partial_time_path(&model) {
        let tile = m.partial_tile(0, ni, 0, nj);
        let mut scratch = synthoseis_core::rock_physics::ColumnScratch::default();
        let (mut rho, mut vp, mut vs) = (vec![0f32; nz], vec![0f32; nz], vec![0f32; nz]);
        let mut col = SubcellColumn::default();
        let (mut r, mut ts) = (Vec::new(), TwtScratch::default());
        let cell = matches!(
            c.rock_physics.partial_voxels.reflectivity,
            Some(PvReflectivity::Cell)
        );
        for i in 0..ni {
            for j in 0..nj {
                let g = (i * nj + j) * nz;
                m.column_partial(
                    i,
                    j,
                    &labels[g..g + nz],
                    &tile,
                    &mut scratch,
                    &mut rho,
                    &mut vp,
                    &mut vs,
                );
                m.partial_column_twt(&scratch, &axis, &mut col);
                let mut x = vec![0f64; len];
                if cell {
                    let t = col.t_cells.clone();
                    reflectivity_time_column_with_twt(
                        &vp,
                        &vs,
                        &rho,
                        &t,
                        DEFAULT_INCIDENCE_DEG,
                        m.zoeppritz,
                        axis.dt_ms,
                        axis.kernel,
                        &mut ts,
                        &mut x,
                    );
                } else {
                    subcell_reflectivity(
                        &col,
                        DEFAULT_INCIDENCE_DEG,
                        m.zoeppritz,
                        axis.dt_ms,
                        axis.kernel,
                        &mut r,
                        &mut x,
                    );
                }
                out.push(x);
            }
        }
        return out;
    }
    let n = ni * nj * nz;
    let (mut vp, mut vs, mut rho) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    model.tile_properties(&labels, shape, 0, ni, 0, nj, &mut vp, &mut vs, &mut rho);
    let form = model.zoeppritz_form();
    let mut ts = TwtScratch::default();
    let mut t = vec![0f64; nz + 1];
    for c in 0..ni * nj {
        let r = c * nz..(c + 1) * nz;
        synthoseis_core::time_mode::column_twt(&axis, &vp[r.clone()], &mut t);
        let mut x = vec![0f64; len];
        reflectivity_time_column_with_twt(
            &vp[r.clone()],
            &vs[r.clone()],
            &rho[r],
            &t,
            DEFAULT_INCIDENCE_DEG,
            form,
            axis.dt_ms,
            axis.kernel,
            &mut ts,
            &mut x,
        );
        out.push(x);
    }
    out
}

fn axis_of(c: &E2eConfig) -> TimeAxis {
    c.time_axis().expect("time mode")
}

fn bp_4_30(axis: &TimeAxis) -> IirFilter {
    butterworth_bandpass(4.0, 30.0, legacy_digitisation_ms(axis.dt_ms), 4).unwrap()
}

/// The `1c22b653` bandpass on one window trace: odd-extension `filtfilt`
/// over `nt − 1` samples, the #36 zero on the last.
fn legacy_bandpass(f: &IirFilter, w: &mut [f32], scratch: &mut Vec<f64>) {
    let nt = w.len();
    f.filtfilt_f32(&mut w[..nt - 1], scratch).unwrap();
    w[nt - 1] = 0.0;
}

/// §7.1 prefix identity: the first `nt` samples of the `nt + P` buffer are
/// bit-identical to the `nt` buffer, for every reflectivity routine (whole
/// voxels, partial `subcell`, partial `cell`) and both kernels, seeds 7, 1
/// and 30.
#[test]
fn prefix_identity_of_the_padded_reflectivity() {
    for seed in [7u64, 1, 30] {
        for kernel in [TwtKernel::Sinc, TwtKernel::Linear] {
            let mut base = demo(seed, 3);
            base.time.kernel = kernel;
            let mut cell = base.clone();
            cell.rock_physics.partial_voxels = PartialVoxelConfig {
                reflectivity: Some(PvReflectivity::Cell),
                ..PartialVoxelConfig::on()
            };
            for (mode, c) in [
                ("whole", whole_voxels(&base)),
                ("subcell", base.clone()),
                ("cell", cell),
            ] {
                let nt = c.output_samples();
                let (w, p) = (raw_columns(&c, nt), raw_columns(&c, nt + 512));
                let mut beyond = 0;
                for (a, b) in w.iter().zip(&p) {
                    assert!(
                        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "seed {seed} {kernel:?} {mode}: window != padded prefix"
                    );
                    beyond += b[nt..].iter().any(|&v| v != 0.0) as usize;
                }
                println!("prefix seed {seed} {kernel:?} {mode}: {} columns identical, {beyond} with reflectivity below the window", w.len());
                assert!(
                    beyond > 0,
                    "seed {seed} {kernel:?} {mode}: the pad must hold the model below"
                );
            }
        }
    }
}

/// §7.5 Ricker default churn, seed 7 (and seed 1), 32 × 32 × 128: labels
/// byte-identical, the stack changes only in samples 121–127 (the Ricker's
/// reach of 7 samples), cells and rel. RMS within the predicted bands;
/// columns whose model base is well inside the window are unchanged.
#[test]
fn ricker_default_churn_only_in_the_last_7_samples() {
    for (seed, cells_band, rms_band) in [
        (7u64, (3.5, 6.0), Some((0.010, 0.025))),
        (1, (1.0, 3.0), None),
    ] {
        let c = demo(seed, 3);
        let nt = c.output_samples();
        assert_eq!(nt, 128);
        let (new, _) = generate_chunked(&c);
        let (old, _) = generate_chunked(&legacy(&c));
        assert!(new.labels == old.labels, "seed {seed}: labels");
        let ch = Churn::of(&new.angle_stack, &old.angle_stack, nt);
        ch.print(&format!("Ricker churn seed {seed} 15°"), nt);
        assert!(
            ch.samples.iter().all(|&k| (nt - 7..nt).contains(&k)),
            "seed {seed}: {:?}",
            ch.samples
        );
        let pct = ch.cell_pct();
        assert!(
            pct >= cells_band.0 && pct <= cells_band.1,
            "seed {seed}: cells changed {pct:.2} %"
        );
        if let Some((lo, hi)) = rms_band {
            assert!(
                ch.rel_rms >= lo && ch.rel_rms <= hi,
                "seed {seed}: rel RMS {:.4}",
                ch.rel_rms
            );
        }
        // Short columns: the model base plus the sinc and Ricker reach end
        // inside the window, so the zero pad was already right.
        let dt = c.time.dt_ms;
        let mut short = 0;
        for (col, t) in base_times(&c).iter().enumerate() {
            if (t / dt).ceil() as usize + SINC_HALF_WIDTH + 8 < nt {
                short += 1;
                let r = col * nt..(col + 1) * nt;
                assert_eq!(
                    bits(&new.angle_stack[r.clone()]),
                    bits(&old.angle_stack[r]),
                    "seed {seed} col {col}"
                );
            }
        }
        println!("seed {seed}: {short} short columns unchanged");
    }
    // Whole-voxel and no-fault rows of the spec's table (printed), and the
    // error profile over the last 8 samples (spec §2.1: error RMS ÷ cube
    // RMS per sample, whole voxels).
    for (name, c) in [
        ("whole voxels", whole_voxels(&demo(7, 3))),
        ("no faults", demo(7, 0)),
    ] {
        let nt = c.output_samples();
        let (new, old) = (stack(&c), stack(&legacy(&c)));
        let ch = Churn::of(&new, &old, nt);
        ch.print(&format!("Ricker churn seed 7 {name} 15°"), nt);
        assert!(ch.samples.iter().all(|&k| k >= nt - 7));
        let cube = rms(&new);
        let profile: Vec<String> = (nt - 8..nt)
            .map(|k| {
                let d: Vec<f32> = new
                    .chunks_exact(nt)
                    .zip(old.chunks_exact(nt))
                    .map(|(a, b)| a[k] - b[k])
                    .collect();
                format!("{:.3}", rms(&d) / cube)
            })
            .collect();
        println!(
            "error profile {name}, samples {}–{}: {}",
            nt - 8,
            nt - 1,
            profile.join(", ")
        );
    }
    // A long window (every column's base inside it): nothing moves.
    let mut c = demo(7, 3);
    let dt = c.time.dt_ms;
    let tmax = base_times(&c).iter().fold(0.0f64, |m, &t| m.max(t));
    c.time.samples = Some((tmax / dt).ceil() as usize + SINC_HALF_WIDTH + 16);
    let (new, old) = (stack(&c), stack(&legacy(&c)));
    assert_eq!(bits(&new), bits(&old), "long window: every column short");
    println!(
        "long window nt {}: default == legacy bit for bit",
        c.output_samples()
    );
}

/// §7.3 truth gate: seed 7, 32 × 32 × 128, `--bandpass 4,30`. The default
/// against a test-only reference with `Pb = 2048` (reflectivity to the
/// model base, then the half-space) is within 1e-6 of peak; the legacy
/// edges are off by a rel. RMS of 0.12–0.22 on every sample.
#[test]
fn bandpass_truth_gate() {
    let c = bandpass(&demo(7, 3), 1);
    let nt = c.output_samples();
    let axis = axis_of(&c);
    let pbp = bp_4_30(&axis).edge_pad().unwrap();
    let mut reference = c.clone();
    reference.time.edge_pad_override = Some(2048);
    let dt = axis.dt_ms;
    let t_base = base_times(&c).iter().fold(0.0f64, |m, &t| m.max(t)) / dt;
    assert!(
        (t_base.ceil() as usize) + SINC_HALF_WIDTH < nt + 2048,
        "the reference must reach the model base: T_nz = {t_base} samples"
    );
    let (def, truth, old) = (stack(&c), stack(&reference), stack(&legacy(&c)));
    let pk = peak(&truth);
    let err = def
        .iter()
        .zip(&truth)
        .fold(0.0f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs()))
        / pk;
    let ch = Churn::of(&def, &old, nt);
    println!(
        "truth gate seed 7 bp 4-30: Pb = {pbp}, max T_nz = {t_base:.1} samples; max |default − truth| = {err:.2e} of peak"
    );
    ch.print("bandpass default vs --legacy-filter-edges", nt);
    let ch_truth = Churn::of(&truth, &old, nt);
    println!(
        "legacy vs truth: rel RMS {:.4}, max {:.1} % of peak",
        ch_truth.rel_rms,
        100.0 * ch_truth.max_of_peak
    );
    assert!(err <= 1e-6, "default vs truth {err:e} of peak");
    assert!(
        ch.rel_rms >= 0.12 && ch.rel_rms <= 0.22,
        "legacy vs default rel RMS {:.4}",
        ch.rel_rms
    );
    assert_eq!(ch.samples.len(), nt, "all {nt} samples touched");
}

/// §7.4 shift invariance (the #39 note): the model moved down by 4, 16
/// and 40 samples (more water) gives the shifted output in the top 60
/// samples to ≤ 1e-8 of peak; the legacy edges are off by > 10 %.
#[test]
fn bandpass_shift_invariance() {
    let c = demo(7, 3);
    let nt = c.output_samples();
    let axis = axis_of(&c);
    let f = bp_4_30(&axis);
    let pb = f.edge_pad().unwrap();
    let chain = TraceChain {
        wavelet: &[],
        noise: None,
        bandpass: Some(&f),
        pads: EdgePads { top: 0, bottom: pb },
    };
    let cols: Vec<Vec<f64>> = raw_columns(&c, nt + pb).into_iter().step_by(37).collect();
    let mut cs = ChainScratch::default();
    let mut fs = Vec::new();
    let run_new = |x: &[f64], cs: &mut ChainScratch| {
        let mut y = vec![0f32; nt];
        finish_trace_padded(x, nt, &chain, 0, cs, &mut y);
        y
    };
    let base_new: Vec<Vec<f32>> = cols.iter().map(|x| run_new(x, &mut cs)).collect();
    let base_old: Vec<Vec<f32>> = cols
        .iter()
        .map(|x| {
            let mut w: Vec<f32> = x[..nt].iter().map(|&v| v as f32).collect();
            legacy_bandpass(&f, &mut w, &mut fs);
            w
        })
        .collect();
    let pk_new = base_new.iter().map(|y| peak(y)).fold(0.0, f64::max);
    let pk_old = base_old.iter().map(|y| peak(y)).fold(0.0, f64::max);
    for s in [4usize, 16, 40] {
        let (mut e_new, mut e_old) = (0.0f64, 0.0f64);
        for (k, x) in cols.iter().enumerate() {
            let mut xs = vec![0f64; s];
            xs.extend_from_slice(&x[..nt + pb - s]);
            let y = run_new(&xs, &mut cs);
            let mut w: Vec<f32> = xs[..nt].iter().map(|&v| v as f32).collect();
            legacy_bandpass(&f, &mut w, &mut fs);
            for n in 0..60 {
                e_new = e_new.max((y[s + n] as f64 - base_new[k][n] as f64).abs());
                e_old = e_old.max((w[s + n] as f64 - base_old[k][n] as f64).abs());
            }
        }
        let (e_new, e_old) = (e_new / pk_new, e_old / pk_old);
        println!("shift {s:2} samples ({} columns): physical {e_new:.2e} of peak, legacy {:.1} % of peak", cols.len(), 100.0 * e_old);
        assert!(e_new <= 1e-8, "shift {s}: {e_new:e}");
        assert!(e_old > 0.10, "shift {s}: legacy {e_old}");
    }
}

/// §7.6 noise stationarity: the noise-only hook (zero reflectivity through
/// [`finish_trace_padded`]) with `--bandpass 4,30` over 1,024 columns. The
/// per-sample noise strength at k = 0 and the last sample is 0.9–1.1 of
/// mid-trace (samples 60–66); the legacy edges give < 0.3 at k = 0.
#[test]
fn noise_is_stationary_at_both_ends() {
    let c = with_noise(&bandpass(&demo(7, 3), 1), 12.5, 7);
    let (labels, shape) = generate_labels(&c);
    let f = SeismicFilters::resolve(&c, &labels, shape)
        .unwrap()
        .unwrap();
    assert!(f.physical_edges);
    let axis = axis_of(&c);
    let nt = axis.nt;
    let noise = f.noise.as_ref().unwrap().at_angle(DEFAULT_INCIDENCE_DEG);
    let bp = f.bandpass.as_ref().unwrap();
    let pads = EdgePads::for_chain(&axis, &[], Some(f.edge_pad), true);
    assert_eq!((pads.top, pads.bottom), (f.edge_pad, f.edge_pad));
    let chain = TraceChain {
        wavelet: &[],
        noise: Some(noise),
        bandpass: Some(bp),
        pads,
    };
    let cols = shape[0] * shape[1];
    assert_eq!(cols, 1024);
    let zeros = vec![0f64; nt + pads.bottom];
    let mut cs = ChainScratch::default();
    let mut fs = Vec::new();
    let (mut new, mut old) = (vec![0f32; cols * nt], vec![0f32; cols * nt]);
    for col in 0..cols {
        finish_trace_padded(
            &zeros,
            nt,
            &chain,
            col as u64,
            &mut cs,
            &mut new[col * nt..(col + 1) * nt],
        );
        let w = &mut old[col * nt..(col + 1) * nt];
        for (k, v) in w.iter_mut().enumerate() {
            *v = noise.sample((col * nt + k) as u64);
        }
        legacy_bandpass(bp, w, &mut fs);
    }
    let profile = |y: &[f32]| {
        let std = |ks: std::ops::Range<usize>| {
            let v: Vec<f64> = y
                .chunks_exact(nt)
                .flat_map(|t| t[ks.clone()].iter().map(|&x| x as f64))
                .collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let reference = std(60..67);
        (0..nt)
            .map(|k| std(k..k + 1) / reference)
            .collect::<Vec<f64>>()
    };
    let (pn, po) = (profile(&new), profile(&old));
    println!(
        "noise strength / mid-trace: physical k=0 {:.3}, k=2 {:.3}, k=5 {:.3}, last {:.3}; legacy k=0 {:.3}, k=5 {:.3}, k=nt-2 {:.3}",
        pn[0], pn[2], pn[5], pn[nt - 1], po[0], po[5], po[nt - 2]
    );
    for k in [0, nt - 1] {
        assert!(pn[k] >= 0.9 && pn[k] <= 1.1, "k {k}: {}", pn[k]);
    }
    assert!(po[0] < 0.3, "legacy k=0 {}", po[0]);
}

/// The window noise of the physical chain is today's noise field: with
/// the wavelet and bandpass off the output is reflectivity + `sample(g)`.
#[test]
fn window_noise_keeps_todays_keys() {
    let c = with_noise(&demo(7, 3), 12.5, 7);
    let (labels, shape) = generate_labels(&c);
    let f = SeismicFilters::resolve(&c, &labels, shape)
        .unwrap()
        .unwrap();
    let axis = axis_of(&c);
    let nt = axis.nt;
    let noise = f.noise.as_ref().unwrap().at_angle(DEFAULT_INCIDENCE_DEG);
    let chain = TraceChain {
        wavelet: &[],
        noise: Some(noise),
        bandpass: None,
        pads: EdgePads { top: 9, bottom: 9 },
    };
    let mut cs = ChainScratch::default();
    let x: Vec<f64> = (0..nt + 9)
        .map(|k| (k as f64 * 0.37).sin() * 0.01)
        .collect();
    for col in [0u64, 17, 1023] {
        let mut y = vec![0f32; nt];
        finish_trace_padded(&x, nt, &chain, col, &mut cs, &mut y);
        for k in 0..nt {
            let want = x[k] as f32 + noise.sample(col * nt as u64 + k as u64);
            assert_eq!(y[k].to_bits(), want.to_bits(), "col {col} k {k}");
        }
    }
}

/// §7.8 invariance: bandpass + noise + lateral 3 with the physical edges
/// (pads recomputed in every halo column) gives bit-identical cubes on
/// every path: classic, chunked and streaming and overlapped streaming
/// (several chunk shapes, single-column and k-chunked tiles), strips,
/// multi-process and geometry-once.
#[test]
fn physical_edges_are_tiling_and_process_invariant() {
    let shape = [12, 10, 64];
    let mut c = with_noise(&bandpass(&demo(30, 3), 3), 12.5, 3);
    c.inline_count = shape[0];
    c.crossline_count = shape[1];
    c.samples = shape[2];
    for c in [c.clone(), whole_voxels(&c)] {
        let nt = c.output_samples();
        let classic = generate_tiny_cube(&c);
        let r = bits(&classic.angle_stack);
        let (labels, sh) = generate_labels(&c);
        let model = elastic_model(&c, &labels, sh);
        let out = generate_output_labels(&c, &labels, &model);
        assert_eq!(classic.labels, out.labels);
        let read = |p: &std::path::Path| {
            let s = synthoseis_io::MdioStore::open(p).unwrap();
            (bits(&s.read_volume().unwrap()), s.read_labels_u8().unwrap())
        };
        let dir = tempfile::tempdir().unwrap();
        for (n, ch) in [[1usize, 1, nt], [5, 7, nt], [3, 10, 16], [8, 5, 16]]
            .into_iter()
            .enumerate()
        {
            let cc = E2eConfig {
                chunk_shape: Some(ch),
                ..c.clone()
            };
            assert!(
                bits(&generate_chunked(&cc).0.angle_stack) == r,
                "chunked {ch:?}"
            );
            let p = dir.path().join(format!("s{n}.mdio"));
            run_e2e_streaming(&E2eConfig {
                store_path: Some(p.clone()),
                ..cc.clone()
            })
            .unwrap();
            assert!(
                read(&p) == (r.clone(), out.labels.clone()),
                "streaming {ch:?}"
            );
            let p = dir.path().join(format!("o{n}.mdio"));
            run_e2e_streaming_overlapped(&E2eConfig {
                store_path: Some(p.clone()),
                ..cc.clone()
            })
            .unwrap();
            assert!(
                read(&p) == (r.clone(), out.labels.clone()),
                "overlap {ch:?}"
            );
        }
        for workers in [2, 3] {
            let p = dir.path().join(format!("strip{workers}.mdio"));
            run_e2e_strip_stitched(
                &E2eConfig {
                    store_path: Some(p.clone()),
                    chunk_shape: Some([3, 4, 16]),
                    ..c.clone()
                },
                workers,
            )
            .unwrap();
            assert!(
                read(&p) == (r.clone(), out.labels.clone()),
                "strips {workers}"
            );
            let p = dir.path().join(format!("mp{workers}.mdio"));
            run_e2e_multiprocess(
                &E2eConfig {
                    store_path: Some(p.clone()),
                    chunk_shape: Some([2, 5, 8]),
                    ..c.clone()
                },
                workers,
            )
            .unwrap();
            assert!(
                read(&p) == (r.clone(), out.labels.clone()),
                "multiprocess {workers}"
            );
        }
        let gc = E2eConfig {
            chunk_shape: Some([4, 4, nt]),
            ..c.clone()
        };
        let (g, _) = run_e2e_geometry_once_seismic_many(&gc, &[DEFAULT_INCIDENCE_DEG]).unwrap();
        assert!(bits(&g.stacks[0].volumes.angle_stack) == r, "geometry-once");
        // The physical chain differs from the legacy one on every path.
        assert_ne!(bits(&generate_tiny_cube(&legacy(&c)).angle_stack), r);
    }
}

/// §7.9 labels everywhere: default vs `--legacy-filter-edges` label arrays
/// (`labels`, `fault_labels`, `salt_labels`) byte-identical on seeds 7, 1
/// and 30, faults 0 and 3, salt on and off (bandpass + noise + lateral 3,
/// the chain that differs most).
#[test]
fn labels_are_identical_with_and_without_the_legacy_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut n = 0;
    for seed in [7u64, 1, 30] {
        for faults in [0usize, 3] {
            for salt in [true, false] {
                let mut c = with_noise(&bandpass(&demo(seed, faults), 3), 12.5, seed);
                c.rock_physics = RockPhysicsConfig {
                    salt,
                    ..c.rock_physics.clone()
                };
                let mut got = Vec::new();
                for (tag, cc) in [("new", c.clone()), ("legacy", legacy(&c))] {
                    let p = dir
                        .path()
                        .join(format!("{seed}-{faults}-{salt}-{tag}.mdio"));
                    run_e2e_streaming(&E2eConfig {
                        store_path: Some(p.clone()),
                        chunk_shape: Some([8, 8, 128]),
                        ..cc
                    })
                    .unwrap();
                    let s = synthoseis_io::MdioStore::open(&p).unwrap();
                    let has = |a: &str| p.join("data").join(a).join(".zarray").is_file();
                    got.push((
                        s.read_labels_u8().unwrap(),
                        has("fault_labels").then(|| s.read_fault_labels_u8().unwrap()),
                        has("salt_labels").then(|| s.read_salt_labels_u8().unwrap()),
                        bits(&s.read_volume().unwrap()),
                    ));
                }
                let (a, b) = (&got[0], &got[1]);
                assert!(
                    a.0 == b.0,
                    "seed {seed} faults {faults} salt {salt}: labels"
                );
                assert!(
                    a.1 == b.1,
                    "seed {seed} faults {faults} salt {salt}: fault_labels"
                );
                assert!(
                    a.2 == b.2,
                    "seed {seed} faults {faults} salt {salt}: salt_labels"
                );
                assert_eq!(a.1.is_some(), faults > 0);
                assert_eq!(a.2.is_some(), salt);
                assert!(a.3 != b.3, "the stacks differ");
                n += 1;
            }
        }
    }
    println!("labels identical in {n} configurations");
}

/// The test-only `Pb` override reaches the chain; the default pads are
/// `h` (Ricker only), `h + Pbp` / 0 (bandpass), `Pb` / `Pb` (noise).
#[test]
fn edge_pads_follow_the_chain() {
    let c = demo(7, 3);
    let axis = axis_of(&c);
    let w = c.ricker();
    assert_eq!(w.len(), 17);
    let pbp = bp_4_30(&axis).edge_pad().unwrap();
    assert_eq!(
        EdgePads::for_chain(&axis, &w, None, false),
        EdgePads { top: 0, bottom: 8 }
    );
    assert_eq!(
        EdgePads::for_chain(&axis, &[], Some(pbp), false),
        EdgePads {
            top: 0,
            bottom: pbp
        }
    );
    assert_eq!(
        EdgePads::for_chain(&axis, &w, Some(pbp), true),
        EdgePads {
            top: 8 + pbp,
            bottom: 8 + pbp
        }
    );
    let mut o = c.clone();
    o.time.edge_pad_override = Some(2048);
    assert_eq!(
        EdgePads::for_chain(&axis_of(&o), &w, Some(pbp), false),
        EdgePads {
            top: 0,
            bottom: 2048
        }
    );
    assert!(TraceChain::wavelet_only(&axis_of(&legacy(&c)), &w).is_none());
    let f = SeismicFilters::from_config(&bandpass(&c, 1))
        .unwrap()
        .unwrap();
    assert!(f.physical_edges && !f.exclude_trailing_sample && f.edge_pad == pbp);
    let f = SeismicFilters::from_config(&legacy(&bandpass(&c, 1)))
        .unwrap()
        .unwrap();
    assert!(!f.physical_edges && f.exclude_trailing_sample && f.edge_pad == 0);
    // The legacy padlen check stays with the legacy edges only.
    let mut short = bandpass(&c, 1);
    short.time.samples = Some(20);
    assert!(SeismicFilters::from_config(&short).is_ok());
    let err = SeismicFilters::from_config(&legacy(&short)).unwrap_err();
    assert!(err.contains("needs more than 27 filtered samples"), "{err}");
}

// ---------------------------------------------------------------- nightly

/// Nightly: the Ricker churn sweep over seeds 1–30, 15° and 30°, whole and
/// partial voxels: changes only in the last 7 samples, labels identical.
#[test]
#[ignore = "nightly: Ricker churn sweep, seeds 1-30 x 2 angles x whole/partial"]
fn nightly_ricker_churn_sweep() {
    let mut worst: f64 = 0.0;
    for seed in 1..=30u64 {
        for whole in [false, true] {
            let c = if whole {
                whole_voxels(&demo(seed, 3))
            } else {
                demo(seed, 3)
            };
            let nt = c.output_samples();
            let (labels, shape) = generate_labels(&c);
            for angle in [15.0, 30.0] {
                let (new, _) =
                    synthoseis_core::generate_angle_stack_from_labels(&c, &labels, shape, angle);
                let (old, _) = synthoseis_core::generate_angle_stack_from_labels(
                    &legacy(&c),
                    &labels,
                    shape,
                    angle,
                );
                assert!(new.labels == old.labels, "seed {seed}");
                let ch = Churn::of(&new.angle_stack, &old.angle_stack, nt);
                ch.print(&format!("seed {seed} whole {whole} {angle}°"), nt);
                assert!(
                    ch.samples.iter().all(|&k| k >= nt - 7),
                    "seed {seed} whole {whole} {angle}°: {:?}",
                    ch.samples
                );
                assert!(ch.cell_pct() <= 7.0 / nt as f64 * 100.0);
                worst = worst.max(ch.max_of_peak);
            }
        }
    }
    println!("worst max change {:.1} % of peak", 100.0 * worst);
}

/// Nightly: the truth gate at 64 × 64 × 256 with noise off and on, and the
/// working set of the padded halo stays bounded by the tile.
#[test]
#[ignore = "nightly: 64x64x256 truth gate and working set"]
fn nightly_truth_gate_64() {
    let mut c = bandpass(&demo(7, 3), 3);
    c.inline_count = 64;
    c.crossline_count = 64;
    c.samples = 256;
    let nt = c.output_samples();
    let mut reference = c.clone();
    reference.time.edge_pad_override = Some(4096);
    let t_base = base_times(&c).iter().fold(0.0f64, |m, &t| m.max(t)) / c.time.dt_ms;
    assert!((t_base.ceil() as usize) + SINC_HALF_WIDTH < nt + 4096);
    let (def, stats) = generate_chunked(&c);
    let truth = stack(&reference);
    let err = def
        .angle_stack
        .iter()
        .zip(&truth)
        .fold(0.0f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs()))
        / peak(&truth);
    let ch = Churn::of(&def.angle_stack, &stack(&legacy(&c)), nt);
    println!("64x64x256 truth gate: {err:.2e} of peak; peak working set {stats:?}");
    ch.print("64x64x256 default vs legacy", nt);
    assert!(err <= 1e-6, "{err:e}");
    let (_, legacy_stats) = generate_chunked(&legacy(&c));
    println!("legacy working set {legacy_stats:?}");
    // The padded halo is counted and stays bounded by the tile: within 2x
    // of the legacy peak at nt = 256 with Pb = 369.
    assert!(
        stats.peak_temp_bytes <= 2 * legacy_stats.peak_temp_bytes,
        "{stats:?} vs {legacy_stats:?}"
    );
}

/// Nightly: `edge_pad` over corner frequencies and orders at dt 2 and 4 ms:
/// finite, below the cap, monotone in the tolerance, and the padded chain
/// against a 4× longer pad (spikes every 7 samples down to a model base
/// half-way into the pad, then the half-space) stays within 1e-5 of peak
/// (the 1e-6 per-impulse tail summed over ~100 spikes) for orders 2–5.
/// Order 6 is printed, not gated: at dt 2 ms with a 3 Hz corner the padded
/// chain differs from the long pad by ~2e-3 of peak (reported to Strata;
/// the default order is 4).
#[test]
#[ignore = "nightly: edge_pad sweep over corners and orders"]
fn nightly_edge_pad_sweep() {
    for dt in [2.0f64, 4.0] {
        for order in [2usize, 3, 4, 5, 6] {
            for (lo, hi) in [
                (3.0, 35.0),
                (4.0, 30.0),
                (5.5, 22.0),
                (6.0, 20.0),
                (3.0, 60.0),
            ] {
                if hi >= 500.0 / dt {
                    continue;
                }
                let f = butterworth_bandpass(lo, hi, legacy_digitisation_ms(dt), order).unwrap();
                let p = f.edge_pad().unwrap();
                let p7 = f.edge_pad_tol(1e-7).unwrap();
                let p5 = f.edge_pad_tol(1e-5).unwrap();
                assert!(
                    p5 <= p && p <= p7,
                    "{lo}-{hi} order {order} dt {dt}: {p5} {p} {p7}"
                );
                let nt = 128;
                let mut x = vec![0f32; nt + 4 * p];
                for (k, v) in x.iter_mut().enumerate().take(nt + p / 2) {
                    *v = if k % 7 == 3 {
                        ((k as f32) * 0.31).sin()
                    } else {
                        0.0
                    };
                }
                let mut a = x[..nt + p].to_vec();
                let mut b = x.clone();
                let mut s = Vec::new();
                f.filtfilt_padded_f32(&mut a, 0, nt, &mut s).unwrap();
                f.filtfilt_padded_f32(&mut b, 0, nt, &mut s).unwrap();
                let pk = peak(&b[..nt]);
                let e = a[..nt]
                    .iter()
                    .zip(&b[..nt])
                    .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
                    / pk;
                println!("dt {dt} order {order} {lo}-{hi} Hz: edge_pad {p} (1e-5: {p5}, 1e-7: {p7}), vs 4x pad {e:.1e} of peak");
                if order <= 5 {
                    assert!(e <= 1e-5, "{lo}-{hi} order {order} dt {dt}: {e}");
                }
            }
        }
    }
}

/// Nightly: wall time of the default and bandpass (+ noise) chains, physical
/// vs legacy edges, at nt = 128 and nt = 1500 (printed for the PR body).
#[test]
#[ignore = "nightly: timing comparison of the physical and legacy edges"]
fn nightly_timing() {
    for (nt, shape) in [(128usize, [32usize, 32, 128]), (1500, [16, 16, 1024])] {
        let mut base = demo(7, 3);
        base.inline_count = shape[0];
        base.crossline_count = shape[1];
        base.samples = shape[2];
        base.time.samples = Some(nt);
        for (name, c) in [
            ("default", base.clone()),
            ("bandpass", bandpass(&base, 1)),
            ("bandpass+noise", with_noise(&bandpass(&base, 1), 12.5, 7)),
        ] {
            let time = |c: &E2eConfig| {
                let _ = stack(c);
                let t = Instant::now();
                for _ in 0..3 {
                    let _ = stack(c);
                }
                t.elapsed().as_secs_f64() / 3.0
            };
            let (a, b) = (time(&c), time(&legacy(&c)));
            println!(
                "nt {nt} {name}: physical {a:.3} s, legacy {b:.3} s, ratio {:.2}",
                a / b
            );
        }
    }
}
