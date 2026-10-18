//! CLI of the analytic folds (spec "analytic folds" §5): `--legacy-fold-geometry`
//! and the fold parameter flags.

use synthoseis_core::{FoldConfig, FoldStyle, RockPhysicsConfig, ToyGeometry};

#[derive(Debug, Clone, Default, clap::Args)]
pub struct FoldArgs {
    /// Legacy switch: no analytic folds, the layered geometry of master
    /// 7e80f2af byte for byte. Rejected with any fold flag, planar geometry
    /// or --legacy-toy-depth.
    #[arg(long, default_value_t = false)]
    pub legacy_fold_geometry: bool,
    /// Fold style: `similar` (vertical shear, default) or `parallel`
    /// (flexural slip, true thickness preserved).
    #[arg(long)]
    pub fold_style: Option<String>,
    /// Number of cylindrical folds per cube (0-4, default 1).
    #[arg(long)]
    pub folds_cylindrical: Option<usize>,
    /// Number of domes / basins per cube (0-6, default 1).
    #[arg(long)]
    pub folds_domes: Option<usize>,
    /// Lateral bin size in metres used by the folds (5-50, default 12.5).
    #[arg(long)]
    pub bin_m: Option<f64>,
    /// Fold growth exponent q (0-3, default 1; 0 = post-depositional).
    #[arg(long)]
    pub fold_growth: Option<f64>,
    /// Maximum fold dip in samples per column (0.1-2, default 1).
    #[arg(long)]
    pub fold_max_dip: Option<f64>,
}

impl FoldArgs {
    fn any_param(&self) -> bool {
        self.fold_style.is_some()
            || self.folds_cylindrical.is_some()
            || self.folds_domes.is_some()
            || self.bin_m.is_some()
            || self.fold_growth.is_some()
            || self.fold_max_dip.is_some()
    }
}

/// Apply the fold flags to `rock` (exit-2 errors as `Err`).
pub fn apply_folds(
    mut rock: RockPhysicsConfig,
    geometry: ToyGeometry,
    a: &FoldArgs,
) -> Result<RockPhysicsConfig, String> {
    let any = a.any_param() || a.legacy_fold_geometry;
    if a.any_param() && a.legacy_fold_geometry {
        return Err("fold flags have no effect with --legacy-fold-geometry".into());
    }
    if any && rock.legacy_toy_depth {
        return Err("fold flags and --legacy-fold-geometry have no effect with --legacy-toy-depth".into());
    }
    if any && geometry == ToyGeometry::Planar {
        return Err("fold flags and --legacy-fold-geometry have no effect with --toy-geometry planar".into());
    }
    rock.legacy_fold_geometry = a.legacy_fold_geometry;
    let mut f = FoldConfig::default();
    if let Some(s) = a.fold_style.as_deref() {
        f.style = FoldStyle::parse(s)?;
    }
    if let Some(v) = a.folds_cylindrical {
        f.cylindrical = v;
    }
    if let Some(v) = a.folds_domes {
        f.domes = v;
    }
    if let Some(v) = a.bin_m {
        f.bin_m = v;
    }
    if let Some(v) = a.fold_growth {
        f.growth = v;
    }
    if let Some(v) = a.fold_max_dip {
        f.max_dip = v;
    }
    f.validate()?;
    rock.fold = f;
    Ok(rock)
}

/// Worker flags that rebuild the fold config (`{:?}` so f64 round-trips).
pub fn fold_args(rock: &RockPhysicsConfig, geometry: ToyGeometry) -> Vec<String> {
    let mut a = Vec::new();
    if rock.legacy_toy_depth || geometry == ToyGeometry::Planar {
        return a;
    }
    if rock.legacy_fold_geometry {
        a.push("--legacy-fold-geometry".into());
        return a;
    }
    let (f, d) = (&rock.fold, FoldConfig::default());
    if f.style != d.style {
        a.push("--fold-style".into());
        a.push(f.style.as_str().into());
    }
    if f.cylindrical != d.cylindrical {
        a.push("--folds-cylindrical".into());
        a.push(f.cylindrical.to_string());
    }
    if f.domes != d.domes {
        a.push("--folds-domes".into());
        a.push(f.domes.to_string());
    }
    for (flag, v, dv) in [
        ("--bin-m", f.bin_m, d.bin_m),
        ("--fold-growth", f.growth, d.growth),
        ("--fold-max-dip", f.max_dip, d.max_dip),
    ] {
        if v != dv {
            a.push(flag.into());
            a.push(format!("{v:?}"));
        }
    }
    a
}
