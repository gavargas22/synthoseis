//! Deterministic seismic noise (replacement for legacy `add_weighted_noise`):
//! off-by-default bit identity, legacy statistical equivalence (mean, std,
//! kurtosis, amplitude spectrum, inter-angle correlation vs the real legacy
//! code over 64 seeds, `tests/fixtures/seismic_noise.json`), chain order
//! (noise before wavelet and bandpass), and bit-identical noise for every
//! chunk shape, strip worker count, process count and geometry-once run.

use serde_json::Value;
use synthoseis_core::pipeline::{
    generate_tiny_cube, E2eConfig, FaultConfig, FilterConfig, NoiseConfig,
};
use synthoseis_core::pipeline_stream::apply_filters_to_volume;
use synthoseis_core::{
    generate_chunked, generate_chunked_at_angle, generate_labels, generate_noise,
    generate_reflectivity, noise_signal_std, run_e2e_chunked, run_e2e_geometry_once_seismic_many,
    run_e2e_multiprocess, run_e2e_streaming, run_e2e_streaming_overlapped, run_e2e_strip_stitched,
    SeismicFilters,
};
use synthoseis_io::MdioStore;
use synthoseis_seismic::{apply_wavelet_traces, ricker};
use tempfile::tempdir;

const FIXTURE: &str = include_str!("../../../tests/fixtures/seismic_noise.json");

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn angle_hash(v: &[f32]) -> u64 {
    fnv(v.iter().flat_map(|x| x.to_bits().to_le_bytes()))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn cfg(seed: u64, shape: [usize; 3], chunks: [usize; 3], faults: usize) -> E2eConfig {
    E2eConfig {
        seed,
        inline_count: shape[0],
        crossline_count: shape[1],
        samples: shape[2],
        store_path: None,
        chunk_shape: Some(chunks),
        faults: FaultConfig::with_count(faults),
        filters: FilterConfig::default(),
    }
}

fn with(chunks: [usize; 3], filters: FilterConfig) -> E2eConfig {
    E2eConfig {
        filters,
        ..cfg(10, [24, 20, 64], chunks, 3)
    }
}

fn noise(snr_db: f64, seed: u64, legacy: bool) -> NoiseConfig {
    NoiseConfig {
        snr_db: Some(snr_db),
        seed: Some(seed),
        legacy_angle_weights: legacy,
    }
}

/// Noise configs exercised by the invariance tests.
fn configs() -> Vec<FilterConfig> {
    vec![
        // Noise only: the Ricker wavelet is kept (no bandpass).
        FilterConfig {
            noise: noise(12.5, 3, false),
            ..FilterConfig::default()
        },
        // Legacy chain: reflectivity + noise, bandpass, lateral filter.
        FilterConfig {
            noise: noise(10.0, 4, false),
            ..FilterConfig::legacy(4.0, 30.0, 3)
        },
        // keep_ricker: reflectivity + noise, Ricker, bandpass, lateral.
        FilterConfig {
            keep_ricker: true,
            noise: noise(17.5, 5, false),
            ..FilterConfig::legacy(5.5, 22.0, 5)
        },
        // Exact legacy (degree) angle weights.
        FilterConfig {
            noise: noise(7.5, 6, true),
            ..FilterConfig::legacy(4.0, 30.0, 1)
        },
    ]
}

/// Noise is off by default and an explicit "off" config (seed set, no S/N)
/// changes nothing: filters-off hashes are the master goldens, and the
/// filtered keep_ricker hash is the one recorded on master 9d5d2051.
#[test]
fn noise_off_by_default_is_bit_identical() {
    assert!(!NoiseConfig::default().enabled());
    assert!(!FilterConfig::default().enabled());
    let off = NoiseConfig {
        snr_db: None,
        seed: Some(99),
        legacy_angle_weights: true,
    };
    assert!(!off.enabled());
    let base = cfg(10, [24, 20, 64], [8, 5, 64], 3);
    let (a, _) = generate_chunked(&base);
    let (b, _) = generate_chunked(&E2eConfig {
        filters: FilterConfig {
            noise: off.clone(),
            ..FilterConfig::default()
        },
        ..base.clone()
    });
    assert_eq!(bits(&a.angle_stack), bits(&b.angle_stack));
    assert!(generate_noise(&base, 15.0).is_none());
    let keep = FilterConfig {
        keep_ricker: true,
        noise: off,
        ..FilterConfig::legacy(4.0, 30.0, 3)
    };
    let (k, _) = generate_chunked(&with([8, 5, 64], keep.clone()));
    assert_eq!(angle_hash(&k.angle_stack), 0x6855_5526_d2bd_464c);
    assert_eq!(
        angle_hash(&generate_tiny_cube(&with([8, 5, 64], keep)).angle_stack),
        0x6855_5526_d2bd_464c
    );
}

/// Noise is added to the raw reflectivity, then the wavelet (unless
/// skipped), then the bandpass / lateral filter: exactly legacy
/// `add_weighted_noise` -> `postprocess_rfc_cubes`.
#[test]
fn noise_is_added_before_wavelet_and_filters() {
    let base = cfg(10, [24, 20, 64], [8, 5, 64], 3);
    let rfc = generate_reflectivity(&base, 15.0);
    let wavelet = ricker(40.0, 4.0, 1);
    let shape = [24, 20, 64];
    for fc in configs() {
        let c = with([8, 5, 64], fc.clone());
        let n = generate_noise(&c, 15.0).expect("noise on");
        assert!(n.iter().any(|&v| v != 0.0));
        let noisy_rfc: Vec<f32> = rfc.iter().zip(&n).map(|(r, e)| r + e).collect();
        let mut want = if fc.skips_ricker() {
            noisy_rfc
        } else {
            apply_wavelet_traces(&noisy_rfc, shape, &wavelet)
        };
        apply_filters_to_volume(&c, &mut want);
        let (v, _) = generate_chunked(&c);
        assert_eq!(bits(&v.angle_stack), bits(&want), "{fc:?}");
        let quiet = with(
            [8, 5, 64],
            FilterConfig {
                noise: NoiseConfig::default(),
                ..fc.clone()
            },
        );
        assert_ne!(bits(&v.angle_stack), bits(&generate_chunked(&quiet).0.angle_stack));
    }
}

#[test]
fn noise_seeds_and_angles() {
    let c = |s: Option<u64>, seed: u64| E2eConfig {
        filters: FilterConfig {
            noise: NoiseConfig {
                snr_db: Some(12.5),
                seed: s,
                legacy_angle_weights: false,
            },
            ..FilterConfig::default()
        },
        ..cfg(seed, [12, 10, 48], [4, 4, 48], 0)
    };
    let a = generate_noise(&c(Some(1), 3), 15.0).unwrap();
    let b = generate_noise(&c(Some(2), 3), 15.0).unwrap();
    assert_ne!(bits(&a), bits(&b), "different seeds differ");
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    assert!(same * 1000 < a.len(), "{same} equal samples");
    assert_eq!(bits(&a), bits(&generate_noise(&c(Some(1), 3), 15.0).unwrap()));
    // Default noise seed follows E2eConfig::seed.
    assert_eq!(
        bits(&generate_noise(&c(None, 3), 15.0).unwrap()),
        bits(&generate_noise(&c(Some(3), 3), 15.0).unwrap())
    );
    // Angles mix the same two fields (legacy shares noise_0deg/noise_45deg).
    let at5 = generate_noise(&c(Some(1), 3), 5.0).unwrap();
    assert_ne!(bits(&a), bits(&at5));
    assert!(corr(&a, &at5) > 0.99);
    // Invalid S/N is an error, not a panic.
    let bad = E2eConfig {
        filters: FilterConfig {
            noise: NoiseConfig::snr(f64::NAN),
            ..FilterConfig::default()
        },
        ..cfg(1, [8, 8, 32], [4, 4, 32], 0)
    };
    assert!(run_e2e_chunked(&bad).unwrap_err().contains("finite"));
}

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64 - ma, y as f64 - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

// ---- legacy statistical equivalence -------------------------------------

struct FieldStats {
    mean: f64,
    std: f64,
    kurtosis: f64,
    bands: Vec<f64>,
}

/// Same statistics as `generate_seismic_noise.py::field_stats`, over the
/// legacy `nk - 1` samples per trace.
fn field_stats(v: &[f32], nk: usize, spectrum: bool) -> FieldStats {
    let n = nk - 1;
    let xs: Vec<f64> = v
        .chunks_exact(nk)
        .flat_map(|t| t[..n].iter().map(|&x| x as f64))
        .collect();
    let cnt = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / cnt;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / cnt;
    let std = var.sqrt();
    let kurtosis = xs.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / cnt / (var * var) - 3.0;
    let mut bands = Vec::new();
    if spectrum {
        let nf = n / 2 + 1;
        let (cs, sn): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|m| {
                let w = 2.0 * std::f64::consts::PI * m as f64 / n as f64;
                (w.cos(), w.sin())
            })
            .unzip();
        let mut amp = vec![0.0f64; nf];
        let traces = xs.chunks_exact(n);
        let nt = traces.len() as f64;
        for t in traces {
            for (f, a) in amp.iter_mut().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (k, &x) in t.iter().enumerate() {
                    let m = (f * k) % n;
                    re += x * cs[m];
                    im -= x * sn[m];
                }
                *a += (re * re + im * im).sqrt();
            }
        }
        let norm = nt * std * (n as f64).sqrt();
        let body: Vec<f64> = amp[1..].iter().map(|a| a / norm).collect();
        // numpy.array_split(body, n_bands)
        let nb = 6;
        let (q, r) = (body.len() / nb, body.len() % nb);
        let mut at = 0;
        for b in 0..nb {
            let len = q + usize::from(b < r);
            bands.push(body[at..at + len].iter().sum::<f64>() / len as f64);
            at += len;
        }
    }
    FieldStats {
        mean,
        std,
        kurtosis,
        bands,
    }
}

fn mean_sd(v: &Value) -> (f64, f64) {
    (v["mean"].as_f64().unwrap(), v["sd"].as_f64().unwrap())
}

fn avg(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Rust noise vs the real legacy `add_weighted_noise` run on the same raw
/// reflectivity (64 legacy seeds; 16 Rust seeds, spectra from 4).
///
/// Tolerances (`SE = sd_legacy * sqrt(1/S_legacy + 1/S_rust)`, `sd_legacy` =
/// legacy seed-to-seed std of the statistic):
/// - `data_std`: relative 1e-6 (legacy reduces in f32);
/// - mean, excess kurtosis, spectrum bands, inter-angle correlation: `|Δ| <= 5 SE`;
/// - std: seed-averaged relative `|Δ| <= 0.5 %`, every seed `<= 2 %` (legacy
///   rescales by the sample std, Rust by the analytic std).
#[test]
fn noise_statistics_match_legacy() {
    let fix: Value = serde_json::from_str(FIXTURE).unwrap();
    let meta = &fix["meta"];
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let [ni, nj, nk] = [shape[0], shape[1], shape[2]];
    let snr_db = meta["snr_db"].as_f64().unwrap();
    let legacy_seeds = meta["legacy_seeds"].as_f64().unwrap();
    let base = cfg(meta["seed"].as_u64().unwrap(), [ni, nj, nk], [16, 16, nk], 0);
    let (labels, lshape) = generate_labels(&base);

    let data_std = noise_signal_std(&base, &labels, lshape);
    let legacy_std = fix["legacy_data_std"].as_f64().unwrap();
    eprintln!("data_std rust {data_std:.9e} legacy {legacy_std:.9e}");
    assert!(
        (data_std / legacy_std - 1.0).abs() < 1e-6,
        "data_std rust {data_std:e} legacy {legacy_std:e}"
    );

    let angles = [5.0, 15.0, 25.0];
    let rust_seeds = 16u64;
    let spec_seeds = 4u64;
    let mut kurt15 = [0.0f64; 2];
    for (m, (mode, legacy)) in [("legacy_degrees", true), ("radians", false)]
        .into_iter()
        .enumerate()
    {
        let reference = &fix[mode];
        let mut stats: Vec<Vec<FieldStats>> = (0..3).map(|_| Vec::new()).collect();
        let mut c525 = Vec::new();
        let mut c515 = Vec::new();
        for s in 1..=rust_seeds {
            let c = E2eConfig {
                filters: FilterConfig {
                    noise: noise(snr_db, s, legacy),
                    ..FilterConfig::default()
                },
                ..base.clone()
            };
            let f = SeismicFilters::resolve(&c, &labels, lshape).unwrap().unwrap();
            let nz = f.noise.as_ref().unwrap();
            assert_eq!(nz.data_std, Some(data_std));
            let fields: Vec<Vec<f32>> = angles
                .iter()
                .map(|&a| {
                    let mut v = vec![0.0f32; ni * nj * nk];
                    nz.at_angle(a).add_to_tile(&mut v, (0, ni), (0, nj), lshape);
                    v
                })
                .collect();
            for (x, v) in fields.iter().enumerate() {
                stats[x].push(field_stats(v, nk, s <= spec_seeds));
            }
            let trim = |v: &[f32]| -> Vec<f32> {
                v.chunks_exact(nk).flat_map(|t| t[..nk - 1].to_vec()).collect()
            };
            let (f5, f15, f25) = (trim(&fields[0]), trim(&fields[1]), trim(&fields[2]));
            c525.push(corr(&f5, &f25));
            c515.push(corr(&f5, &f15));
        }
        let rs = rust_seeds as f64;
        let se = |sd: f64, s_rust: f64| sd * (1.0 / legacy_seeds + 1.0 / s_rust).sqrt();
        for (x, a) in angles.iter().enumerate() {
            let r = &reference["angles"][format!("{a:.1}")];
            let st = &stats[x];
            let (lm, lsd) = mean_sd(&r["mean"]);
            let rm = avg(&st.iter().map(|s| s.mean).collect::<Vec<_>>());
            assert!((rm - lm).abs() <= 5.0 * se(lsd, rs), "{mode} {a} mean {rm:e} vs {lm:e}");

            let (lstd, _) = mean_sd(&r["std"]);
            let rstd: Vec<f64> = st.iter().map(|s| s.std).collect();
            assert!((avg(&rstd) / lstd - 1.0).abs() <= 0.005, "{mode} {a} std {rstd:?} vs {lstd}");
            for v in &rstd {
                assert!((v / lstd - 1.0).abs() <= 0.02, "{mode} {a} std {v} vs {lstd}");
            }

            let (lk, lksd) = mean_sd(&r["kurtosis"]);
            let rk = avg(&st.iter().map(|s| s.kurtosis).collect::<Vec<_>>());
            eprintln!(
                "{mode:>14} {a:>4}: mean {rm:+.2e} vs {lm:+.2e} (tol {:.1e}) | std {:.6e} vs \
                 {lstd:.6e} ({:+.3}%) | ex.kurt {rk:.3} vs {lk:.3} (tol {:.3})",
                5.0 * se(lsd, rs),
                avg(&rstd),
                100.0 * (avg(&rstd) / lstd - 1.0),
                5.0 * se(lksd, rs)
            );
            assert!((rk - lk).abs() <= 5.0 * se(lksd, rs), "{mode} {a} kurtosis {rk} vs {lk}");
            if *a == 15.0 {
                kurt15[m] = rk;
            }

            let lb = r["bands"]["mean"].as_array().unwrap();
            let lbsd = r["bands"]["sd"].as_array().unwrap();
            for b in 0..lb.len() {
                let rb = avg(
                    &st.iter()
                        .filter(|s| !s.bands.is_empty())
                        .map(|s| s.bands[b])
                        .collect::<Vec<_>>(),
                );
                let (l, sd) = (lb[b].as_f64().unwrap(), lbsd[b].as_f64().unwrap());
                eprintln!(
                    "{mode:>14} {a:>4}: band {b} {rb:.4} vs {l:.4} (tol {:.4})",
                    5.0 * se(sd, spec_seeds as f64)
                );
                assert!(
                    (rb - l).abs() <= 5.0 * se(sd, spec_seeds as f64),
                    "{mode} {a} band {b}: {rb} vs {l}"
                );
            }
        }
        for (key, rc) in [("corr_5_25", &c525), ("corr_5_15", &c515)] {
            let (l, sd) = mean_sd(&reference["inter_angle"][key]);
            let r = avg(rc);
            eprintln!("{mode:>14} {key}: {r:.5} vs {l:.5} (tol {:.1e})", 5.0 * se(sd, rs));
            assert!((r - l).abs() <= 5.0 * se(sd, rs), "{mode} {key}: {r} vs {l}");
        }
    }
    // The test discriminates the two weightings (legacy 15 deg mix is much
    // less heavy-tailed than the radian mix).
    let rad15 = mean_sd(&fix["radians"]["angles"]["15.0"]["kurtosis"]).0;
    assert!((kurt15[0] - rad15).abs() > 1.0, "{kurt15:?} vs {rad15}");
}

// ---- tiling / worker / process invariance --------------------------------

#[test]
fn noisy_stack_invariant_to_chunk_shape() {
    let chunk_shapes: [[usize; 3]; 6] = [
        [24, 20, 64],
        [8, 5, 64],
        [5, 7, 64],
        [1, 20, 64],
        [24, 1, 16],
        [7, 3, 32],
    ];
    for fc in configs() {
        let (reference, _) = generate_chunked(&with(chunk_shapes[0], fc.clone()));
        for chunks in &chunk_shapes[1..] {
            let (v, _) = generate_chunked(&with(*chunks, fc.clone()));
            assert_eq!(
                bits(&v.angle_stack),
                bits(&reference.angle_stack),
                "{fc:?} chunks {chunks:?}"
            );
        }
        let classic = generate_tiny_cube(&with([8, 5, 64], fc.clone()));
        assert_eq!(
            bits(&classic.angle_stack),
            bits(&reference.angle_stack),
            "classic {fc:?}"
        );
        // data_std does not depend on the chunk shape either.
        let (labels, shape) = generate_labels(&with([8, 5, 64], fc.clone()));
        let a = noise_signal_std(&with([24, 20, 64], fc.clone()), &labels, shape);
        let b = noise_signal_std(&with([7, 3, 32], fc.clone()), &labels, shape);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// Streaming, overlapped streaming, strip-stitch (2-4 workers), multi-process
/// (1-3 workers) and geometry-once all write bit-identical noisy stacks.
#[test]
fn noisy_stack_invariant_to_workers_and_paths() {
    let dir = tempdir().unwrap();
    for (n, fc) in configs().into_iter().enumerate() {
        let (reference, _) = generate_chunked(&with([24, 20, 64], fc.clone()));
        let want = bits(&reference.angle_stack);
        let read = |p: &std::path::Path| bits(&MdioStore::open(p).unwrap().read_volume().unwrap());

        for (k, chunks) in [[8, 5, 16], [3, 20, 32]].into_iter().enumerate() {
            let p = dir.path().join(format!("stream_{n}_{k}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..with(chunks, fc.clone())
            };
            let (report, _) = run_e2e_streaming(&c).expect("streaming");
            assert!(report.parity.passes_defaults());
            assert_eq!(read(&p), want, "streaming {fc:?} {chunks:?}");

            let p = dir.path().join(format!("overlap_{n}_{k}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..with(chunks, fc.clone())
            };
            run_e2e_streaming_overlapped(&c).expect("overlap");
            assert_eq!(read(&p), want, "overlap {fc:?} {chunks:?}");
        }

        for workers in [2, 3, 4] {
            let p = dir.path().join(format!("strip_{n}_{workers}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..with([5, 7, 64], fc.clone())
            };
            let (report, _) = run_e2e_strip_stitched(&c, workers).expect("strip stitch");
            assert!(report.parity.passes_defaults());
            assert_eq!(read(&p), want, "strip {fc:?} workers {workers}");
        }

        for workers in [1, 2, 3] {
            let p = dir.path().join(format!("mp_{n}_{workers}.mdio"));
            let c = E2eConfig {
                store_path: Some(p.clone()),
                ..with([8, 7, 64], fc.clone())
            };
            let (report, _) = run_e2e_multiprocess(&c, workers).expect("multiprocess");
            assert!(report.parity.passes_defaults());
            assert_eq!(bits(&report.volumes.angle_stack), want);
            assert_eq!(read(&p), want, "multiprocess {fc:?} workers {workers}");
        }

        let (geo, _) =
            run_e2e_geometry_once_seismic_many(&with([5, 7, 64], fc.clone()), &[0.0, 15.0])
                .expect("geometry once");
        assert_eq!(bits(&geo.stacks[1].volumes.angle_stack), want);
        let at0 = generate_chunked_at_angle(&with([24, 20, 64], fc.clone()), 0.0).0;
        assert_eq!(bits(&geo.stacks[0].volumes.angle_stack), bits(&at0.angle_stack));
    }
}
