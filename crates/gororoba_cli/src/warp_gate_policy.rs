use serde::Deserialize;
use std::error::Error;
use std::path::{Path, PathBuf};

pub const CANONICAL_REQUIRED_TRACE_CHANNELS: [&str; 5] = [
    "u_rms",
    "mach_eff",
    "re_eff",
    "power_injection_proxy",
    "dissipation_proxy",
];

pub const CANONICAL_REQUIRED_SPECTRAL_CHANNELS: [&str; 5] = [
    "k_peak",
    "total_power",
    "slope",
    "triad_count",
    "triad_clustering",
];

#[derive(Debug, Clone, Copy)]
pub struct ScalarSignalThreshold {
    pub min_abs_max: f64,
    pub min_std_dev: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpGatePolicy {
    pub canonical_scalar_signal: ScalarSignalThreshold,
    pub canonical_enstrophy_signal: ScalarSignalThreshold,
    pub measured_enstrophy_signal: ScalarSignalThreshold,
    pub measured_algebra_norm_signal: ScalarSignalThreshold,
    pub measured_spectral_total_power_signal: ScalarSignalThreshold,
    pub measured_enstrophy_nonzero_fraction_min: f64,
    pub measured_algebra_norm_nonzero_fraction_min: f64,
    pub measured_spectral_total_power_nonzero_fraction_min: f64,
    pub u_rms_model_lock_max_fraction: f64,
}

impl Default for WarpGatePolicy {
    fn default() -> Self {
        Self {
            canonical_scalar_signal: ScalarSignalThreshold {
                min_abs_max: 1.0e-14,
                min_std_dev: 0.0,
            },
            canonical_enstrophy_signal: ScalarSignalThreshold {
                min_abs_max: 1.0e-20,
                min_std_dev: 0.0,
            },
            measured_enstrophy_signal: ScalarSignalThreshold {
                min_abs_max: 1.0e-20,
                min_std_dev: 0.0,
            },
            measured_algebra_norm_signal: ScalarSignalThreshold {
                min_abs_max: 1.0e-20,
                min_std_dev: 0.0,
            },
            measured_spectral_total_power_signal: ScalarSignalThreshold {
                min_abs_max: 1.0e-24,
                min_std_dev: 0.0,
            },
            measured_enstrophy_nonzero_fraction_min: 0.01,
            measured_algebra_norm_nonzero_fraction_min: 0.01,
            measured_spectral_total_power_nonzero_fraction_min: 0.01,
            u_rms_model_lock_max_fraction: 0.999,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedWarpGatePolicy {
    pub source_path: PathBuf,
    pub policy: WarpGatePolicy,
}

#[derive(Debug, Deserialize)]
struct EngineeringStandardsRoot {
    engineering_standards: EngineeringStandardsSection,
}

#[derive(Debug, Deserialize)]
struct EngineeringStandardsSection {
    warp_gate_profiles: Option<WarpGateProfilesToml>,
}

#[derive(Debug, Deserialize)]
struct WarpGateProfilesToml {
    canonical_300s: Option<Canonical300sToml>,
    canonical_300s_measured: Option<Canonical300sMeasuredToml>,
}

#[derive(Debug, Deserialize)]
struct Canonical300sToml {
    scalar_min_abs_max: Option<f64>,
    scalar_min_std_dev: Option<f64>,
    enstrophy_min_abs_max: Option<f64>,
    enstrophy_min_std_dev: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Canonical300sMeasuredToml {
    enstrophy_measured_min_abs_max: Option<f64>,
    enstrophy_measured_min_std_dev: Option<f64>,
    algebra_norm_min_abs_max: Option<f64>,
    algebra_norm_min_std_dev: Option<f64>,
    spectral_total_power_min_abs_max: Option<f64>,
    spectral_total_power_min_std_dev: Option<f64>,
    enstrophy_measured_nonzero_fraction_min: Option<f64>,
    algebra_norm_nonzero_fraction_min: Option<f64>,
    spectral_total_power_nonzero_fraction_min: Option<f64>,
    u_rms_model_lock_max_fraction: Option<f64>,
}

fn default_policy_path() -> PathBuf {
    PathBuf::from("registry/engineering_standards.toml")
}

fn policy_path_from_env() -> PathBuf {
    std::env::var("GOROROBA_WARP_GATE_POLICY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_policy_path())
}

fn apply_canonical_300s_overrides(policy: &mut WarpGatePolicy, src: Option<Canonical300sToml>) {
    if let Some(src) = src {
        if let Some(v) = src.scalar_min_abs_max {
            policy.canonical_scalar_signal.min_abs_max = v;
        }
        if let Some(v) = src.scalar_min_std_dev {
            policy.canonical_scalar_signal.min_std_dev = v;
        }
        if let Some(v) = src.enstrophy_min_abs_max {
            policy.canonical_enstrophy_signal.min_abs_max = v;
        }
        if let Some(v) = src.enstrophy_min_std_dev {
            policy.canonical_enstrophy_signal.min_std_dev = v;
        }
    }
}

fn apply_canonical_300s_measured_overrides(
    policy: &mut WarpGatePolicy,
    src: Option<Canonical300sMeasuredToml>,
) {
    if let Some(src) = src {
        if let Some(v) = src.enstrophy_measured_min_abs_max {
            policy.measured_enstrophy_signal.min_abs_max = v;
        }
        if let Some(v) = src.enstrophy_measured_min_std_dev {
            policy.measured_enstrophy_signal.min_std_dev = v;
        }
        if let Some(v) = src.algebra_norm_min_abs_max {
            policy.measured_algebra_norm_signal.min_abs_max = v;
        }
        if let Some(v) = src.algebra_norm_min_std_dev {
            policy.measured_algebra_norm_signal.min_std_dev = v;
        }
        if let Some(v) = src.spectral_total_power_min_abs_max {
            policy.measured_spectral_total_power_signal.min_abs_max = v;
        }
        if let Some(v) = src.spectral_total_power_min_std_dev {
            policy.measured_spectral_total_power_signal.min_std_dev = v;
        }
        if let Some(v) = src.enstrophy_measured_nonzero_fraction_min {
            policy.measured_enstrophy_nonzero_fraction_min = v;
        }
        if let Some(v) = src.algebra_norm_nonzero_fraction_min {
            policy.measured_algebra_norm_nonzero_fraction_min = v;
        }
        if let Some(v) = src.spectral_total_power_nonzero_fraction_min {
            policy.measured_spectral_total_power_nonzero_fraction_min = v;
        }
        if let Some(v) = src.u_rms_model_lock_max_fraction {
            policy.u_rms_model_lock_max_fraction = v;
        }
    }
}

pub fn load_warp_gate_policy() -> Result<LoadedWarpGatePolicy, Box<dyn Error>> {
    let path = policy_path_from_env();
    load_warp_gate_policy_from_path(&path)
}

pub fn load_warp_gate_policy_from_path(
    path: &Path,
) -> Result<LoadedWarpGatePolicy, Box<dyn Error>> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        std::io::Error::other(format!(
            "failed to read warp gate policy from {}: {e}",
            path.display()
        ))
    })?;
    let parsed: EngineeringStandardsRoot = toml::from_str(&raw).map_err(|e| {
        std::io::Error::other(format!(
            "failed to parse warp gate policy table from {}: {e}",
            path.display()
        ))
    })?;
    let profiles = parsed
        .engineering_standards
        .warp_gate_profiles
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "missing [engineering_standards.warp_gate_profiles] in {}",
                path.display()
            ))
        })?;

    let mut policy = WarpGatePolicy::default();
    apply_canonical_300s_overrides(&mut policy, profiles.canonical_300s);
    apply_canonical_300s_measured_overrides(&mut policy, profiles.canonical_300s_measured);

    Ok(LoadedWarpGatePolicy {
        source_path: path.to_path_buf(),
        policy,
    })
}
