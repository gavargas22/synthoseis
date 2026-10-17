//! CLI of the physical rock-physics model (`--legacy-rock-physics` and the
//! parameter flags of spec §5). See docs/rock-physics-physical.md.

use synthoseis_core::{CompactionForm, DryFrame};
use synthoseis_core::{PhysParams, RockPhysicsConfig};

#[derive(Debug, Clone, Default, clap::Args)]
pub struct PhysArgs {
    /// Legacy switch: depth-trend rock physics of master 77131867 (one
    /// polynomial per lithology and fluid), bit for bit. Default: physical
    /// chain (Athy compaction + soft-sand dry frame + Gassmann + Batzle-Wang
    /// fluids). Rejected together with any physical parameter flag.
    #[arg(long, default_value_t = false)]
    pub legacy_rock_physics: bool,
    /// Sand surface porosity φ0 (0.25-0.6, default 0.49; Sclater & Christie 1980).
    #[arg(long)]
    pub sand_phi0: Option<f64>,
    /// Sand Athy coefficient c in 1/km (0.05-1.0, default 0.27).
    #[arg(long)]
    pub sand_athy_c: Option<f64>,
    /// Shale surface porosity φ0 (0.4-0.8, default 0.63).
    #[arg(long)]
    pub shale_phi0: Option<f64>,
    /// Shale Athy coefficient c in 1/km (0.1-1.5, default 0.51).
    #[arg(long)]
    pub shale_athy_c: Option<f64>,
    /// Porosity law: `depth` (φ0 e^(−cz), default) or `stress` (φ0 e^(−P_eff/σ0)).
    #[arg(long)]
    pub compaction_form: Option<String>,
    /// σ0 in MPa for `--compaction-form stress` (5-200; default derived per
    /// lithology as (ρ_m − ρ_w)·g/c).
    #[arg(long)]
    pub athy_sigma0: Option<f64>,
    /// Dry frame: `soft-sand` (default) or `critical-porosity`.
    #[arg(long)]
    pub dry_frame: Option<String>,
    /// Seabed temperature in °C (0-25, default 4).
    #[arg(long)]
    pub seabed_temperature: Option<f64>,
    /// Geothermal gradient in °C/km (15-50, default 30).
    #[arg(long)]
    pub temperature_gradient: Option<f64>,
    /// Brine NaCl weight fraction (0-0.3, default 0.035).
    #[arg(long)]
    pub brine_salinity: Option<f64>,
    /// Gas specific gravity (0.55-1.0, default 0.6).
    #[arg(long)]
    pub gas_gravity: Option<f64>,
    /// Oil API gravity (10-50, default 32).
    #[arg(long)]
    pub oil_api: Option<f64>,
    /// Oil gas-oil ratio in L/L (0-300, default 100; 0 = dead oil).
    #[arg(long)]
    pub oil_gor: Option<f64>,
    /// Gas saturation of gas sands (0-1, default 0.8; rest brine, Wood mix).
    #[arg(long)]
    pub gas_saturation: Option<f64>,
    /// Oil saturation of oil sands (0-1, default 0.8).
    #[arg(long)]
    pub oil_saturation: Option<f64>,
}

impl PhysArgs {
    fn any_param(&self) -> bool {
        self.sand_phi0.is_some()
            || self.sand_athy_c.is_some()
            || self.shale_phi0.is_some()
            || self.shale_athy_c.is_some()
            || self.compaction_form.is_some()
            || self.athy_sigma0.is_some()
            || self.dry_frame.is_some()
            || self.seabed_temperature.is_some()
            || self.temperature_gradient.is_some()
            || self.brine_salinity.is_some()
            || self.gas_gravity.is_some()
            || self.oil_api.is_some()
            || self.oil_gor.is_some()
            || self.gas_saturation.is_some()
            || self.oil_saturation.is_some()
    }
}

/// Apply the rock-physics flags to `rock` (exit-2 errors as `Err`).
pub fn apply_rock_physics(
    mut rock: RockPhysicsConfig,
    a: &PhysArgs,
) -> Result<RockPhysicsConfig, String> {
    if a.any_param() && a.legacy_rock_physics {
        return Err(
            "physical rock-physics parameter flags have no effect with --legacy-rock-physics"
                .into(),
        );
    }
    if a.any_param() && rock.legacy_toy_depth {
        return Err(
            "physical rock-physics parameter flags have no effect with --legacy-toy-depth".into(),
        );
    }
    if a.legacy_rock_physics && rock.legacy_toy_depth {
        return Err("--legacy-rock-physics is implied by --legacy-toy-depth".into());
    }
    rock.legacy_rock_physics = a.legacy_rock_physics;
    let mut p = PhysParams::default();
    if let Some(v) = a.sand_phi0 {
        p.sand.phi0 = v;
    }
    if let Some(v) = a.sand_athy_c {
        p.sand.athy_c = v;
    }
    if let Some(v) = a.shale_phi0 {
        p.shale.phi0 = v;
    }
    if let Some(v) = a.shale_athy_c {
        p.shale.athy_c = v;
    }
    if let Some(s) = a.compaction_form.as_deref() {
        p.compaction = match s {
            "depth" => CompactionForm::Depth,
            "stress" => CompactionForm::Stress,
            o => {
                return Err(format!(
                    "--compaction-form must be depth or stress, got {o}"
                ))
            }
        };
    }
    if let Some(v) = a.athy_sigma0 {
        if p.compaction != CompactionForm::Stress {
            return Err("--athy-sigma0 needs --compaction-form stress".into());
        }
        p.sigma0 = Some(v);
    }
    if let Some(s) = a.dry_frame.as_deref() {
        p.dry_frame = match s {
            "soft-sand" => DryFrame::SoftSand,
            "critical-porosity" => DryFrame::CriticalPorosity,
            o => {
                return Err(format!(
                    "--dry-frame must be soft-sand or critical-porosity, got {o}"
                ))
            }
        };
    }
    if let Some(v) = a.seabed_temperature {
        p.t_seabed = v;
    }
    if let Some(v) = a.temperature_gradient {
        p.t_gradient = v;
    }
    if let Some(v) = a.brine_salinity {
        p.salinity = v;
    }
    if let Some(v) = a.gas_gravity {
        p.gas_gravity = v;
    }
    if let Some(v) = a.oil_api {
        p.oil_api = v;
    }
    if let Some(v) = a.oil_gor {
        p.oil_gor = v;
    }
    if let Some(v) = a.gas_saturation {
        p.gas_saturation = v;
    }
    if let Some(v) = a.oil_saturation {
        p.oil_saturation = v;
    }
    p.validate()?;
    rock.phys = p;
    Ok(rock)
}

/// Worker flags that rebuild the rock model: `--legacy-rock-physics`, or
/// every physical parameter that differs from its default (`{:?}` so f64
/// values round-trip exactly).
pub fn rock_model_args(rock: &RockPhysicsConfig) -> Vec<String> {
    let mut a = Vec::new();
    if rock.legacy_toy_depth {
        return a;
    }
    if rock.legacy_rock_physics {
        a.push("--legacy-rock-physics".into());
        return a;
    }
    let p = &rock.phys;
    let d = PhysParams::default();
    let mut f = |flag: &str, v: f64, dv: f64| {
        if v != dv {
            a.push(flag.into());
            a.push(format!("{v:?}"));
        }
    };
    f("--sand-phi0", p.sand.phi0, d.sand.phi0);
    f("--sand-athy-c", p.sand.athy_c, d.sand.athy_c);
    f("--shale-phi0", p.shale.phi0, d.shale.phi0);
    f("--shale-athy-c", p.shale.athy_c, d.shale.athy_c);
    f("--seabed-temperature", p.t_seabed, d.t_seabed);
    f("--temperature-gradient", p.t_gradient, d.t_gradient);
    f("--brine-salinity", p.salinity, d.salinity);
    f("--gas-gravity", p.gas_gravity, d.gas_gravity);
    f("--oil-api", p.oil_api, d.oil_api);
    f("--oil-gor", p.oil_gor, d.oil_gor);
    f("--gas-saturation", p.gas_saturation, d.gas_saturation);
    f("--oil-saturation", p.oil_saturation, d.oil_saturation);
    if p.compaction != d.compaction {
        a.push("--compaction-form".into());
        a.push(p.compaction.as_str().into());
    }
    if let Some(s) = p.sigma0 {
        a.push("--athy-sigma0".into());
        a.push(format!("{s:?}"));
    }
    if p.dry_frame != d.dry_frame {
        a.push("--dry-frame".into());
        a.push(p.dry_frame.as_str().into());
    }
    a
}

/// One summary line.
pub fn summary(rock: &RockPhysicsConfig) -> String {
    if rock.legacy_rock_physics || rock.legacy_toy_depth {
        "rock physics: legacy trends (--legacy-rock-physics)".into()
    } else {
        format!(
            "rock physics: physical (Athy {} + {} + Gassmann + Batzle-Wang)",
            rock.phys.compaction.as_str(),
            rock.phys.dry_frame.as_str()
        )
    }
}
