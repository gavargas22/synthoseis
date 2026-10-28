//! CLI of the angular unconformities (spec "erosion and onlap" §5):
//! `--legacy-unconformity` and the unconformity parameter flags.

use synthoseis_core::{RockPhysicsConfig, ToyGeometry, UnconformityConfig, UnconformityShale};

#[derive(Debug, Clone, Default, clap::Args)]
pub struct UnconformityArgs {
    /// Legacy switch: no unconformities, the layered geometry of #56
    /// byte for byte. Rejected with any unconformity flag, planar geometry
    /// or --legacy-toy-depth. `--legacy-fold-geometry` implies it.
    #[arg(long, default_value_t = false)]
    pub legacy_unconformity: bool,
    /// Number of angular unconformities per cube (0-2, default 1).
    #[arg(long)]
    pub unconformities: Option<usize>,
    /// Position of the (shallower) unconformity as a fraction of the stack
    /// from the seabed (0.1-0.9, default drawn from 0.3-0.7).
    #[arg(long)]
    pub unconformity_position: Option<f64>,
    /// Quantile of the folded horizon that sets the bevel level (0.1-0.9,
    /// default drawn from 0.35-0.65).
    #[arg(long)]
    pub unconformity_level: Option<f64>,
    /// Relief of the erosion surface in metres (0-200, default drawn from
    /// 2-5 % of the cube depth).
    #[arg(long)]
    pub unconformity_relief_m: Option<f64>,
    /// Steepest dip of the erosion surface in samples per column
    /// (0.05-2, default 1).
    #[arg(long)]
    pub unconformity_max_dip: Option<f64>,
    /// Fill ratio of the ponded fill (1-6, default 2).
    #[arg(long)]
    pub unconformity_fill_ratio: Option<f64>,
    /// Shale seal at the unconformity: `below` (default), `above`, `both`
    /// or `none`.
    #[arg(long)]
    pub unconformity_shale: Option<String>,
    /// Do not add the eroded thickness to the burial of the layers below
    /// each unconformity.
    #[arg(long, default_value_t = false)]
    pub no_exhumation_burial: bool,
}

impl UnconformityArgs {
    fn any_param(&self) -> bool {
        self.unconformities.is_some()
            || self.unconformity_position.is_some()
            || self.unconformity_level.is_some()
            || self.unconformity_relief_m.is_some()
            || self.unconformity_max_dip.is_some()
            || self.unconformity_fill_ratio.is_some()
            || self.unconformity_shale.is_some()
            || self.no_exhumation_burial
    }
}

/// Apply the unconformity flags to `rock` (exit-2 errors as `Err`).
pub fn apply_unconformities(
    mut rock: RockPhysicsConfig,
    geometry: ToyGeometry,
    a: &UnconformityArgs,
) -> Result<RockPhysicsConfig, String> {
    let any = a.any_param() || a.legacy_unconformity;
    if a.any_param() && a.legacy_unconformity {
        return Err("unconformity flags have no effect with --legacy-unconformity".into());
    }
    if any && rock.legacy_toy_depth {
        return Err(
            "unconformity flags and --legacy-unconformity have no effect with --legacy-toy-depth"
                .into(),
        );
    }
    if any && geometry == ToyGeometry::Planar {
        return Err(
            "unconformity flags and --legacy-unconformity have no effect with --toy-geometry planar"
                .into(),
        );
    }
    if a.any_param() && rock.legacy_fold_geometry {
        return Err("unconformity flags have no effect with --legacy-fold-geometry".into());
    }
    if a.unconformity_position.is_some() && a.unconformities == Some(0) {
        return Err("--unconformity-position has no effect with --unconformities 0".into());
    }
    rock.legacy_unconformity = a.legacy_unconformity;
    let mut u = UnconformityConfig::default();
    if let Some(v) = a.unconformities {
        u.count = v;
    }
    u.position = a.unconformity_position;
    u.level = a.unconformity_level;
    u.relief_m = a.unconformity_relief_m;
    if let Some(v) = a.unconformity_max_dip {
        u.max_dip = v;
    }
    if let Some(v) = a.unconformity_fill_ratio {
        u.fill_ratio = v;
    }
    u.validate()?;
    rock.unconformity = u;
    if let Some(s) = a.unconformity_shale.as_deref() {
        rock.unconformity_shale = UnconformityShale::parse(s)?;
    }
    rock.exhumation_burial = !a.no_exhumation_burial;
    Ok(rock)
}

/// Worker flags that rebuild the unconformity config (`{:?}` so f64
/// round-trips).
pub fn unconformity_args(rock: &RockPhysicsConfig, geometry: ToyGeometry) -> Vec<String> {
    let mut a = Vec::new();
    if rock.legacy_toy_depth || geometry == ToyGeometry::Planar || rock.legacy_fold_geometry {
        return a;
    }
    if rock.legacy_unconformity {
        a.push("--legacy-unconformity".into());
        return a;
    }
    let (u, d) = (&rock.unconformity, UnconformityConfig::default());
    if u.count != d.count {
        a.push("--unconformities".into());
        a.push(u.count.to_string());
    }
    for (flag, v) in [
        ("--unconformity-position", u.position),
        ("--unconformity-level", u.level),
        ("--unconformity-relief-m", u.relief_m),
    ] {
        if let Some(v) = v {
            a.push(flag.into());
            a.push(format!("{v:?}"));
        }
    }
    for (flag, v, dv) in [
        ("--unconformity-max-dip", u.max_dip, d.max_dip),
        ("--unconformity-fill-ratio", u.fill_ratio, d.fill_ratio),
    ] {
        if v != dv {
            a.push(flag.into());
            a.push(format!("{v:?}"));
        }
    }
    if rock.unconformity_shale != UnconformityShale::default() {
        a.push("--unconformity-shale".into());
        a.push(rock.unconformity_shale.as_str().into());
    }
    if !rock.exhumation_burial {
        a.push("--no-exhumation-burial".into());
    }
    a
}
