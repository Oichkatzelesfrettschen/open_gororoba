use serde::Deserialize;
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy)]
pub struct WarpForcingProfile {
    pub mode_y_requested: usize,
    pub re_target: f64,
    pub max_mach: f64,
    pub rho0: f64,
    pub bf16_min_delta_ulp_ratio: f64,
    pub bf16_enforce_mode_floor: bool,
    pub enforce_mode_floor_all_precisions: bool,
}

impl Default for WarpForcingProfile {
    fn default() -> Self {
        Self {
            mode_y_requested: 1,
            re_target: 64.0,
            max_mach: 0.08,
            rho0: 1.0,
            bf16_min_delta_ulp_ratio: 1.0e-2,
            bf16_enforce_mode_floor: true,
            enforce_mode_floor_all_precisions: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedWarpForcingProfile {
    pub source_path: PathBuf,
    pub profile_name: String,
    pub resolution: usize,
    pub used_resolution_override: bool,
    pub values: WarpForcingProfile,
}

#[derive(Debug, Deserialize)]
struct EngineeringStandardsRoot {
    engineering_standards: EngineeringStandardsSection,
}

#[derive(Debug, Deserialize)]
struct EngineeringStandardsSection {
    warp_forcing_profiles: Option<WarpForcingProfilesToml>,
}

#[derive(Debug, Deserialize)]
struct WarpForcingProfilesToml {
    default: Option<String>,
    #[serde(flatten)]
    profiles: BTreeMap<String, WarpForcingProfileToml>,
}

#[derive(Debug, Deserialize)]
struct WarpForcingProfileToml {
    mode_y_requested: Option<usize>,
    re_target: Option<f64>,
    max_mach: Option<f64>,
    rho0: Option<f64>,
    bf16_min_delta_ulp_ratio: Option<f64>,
    bf16_enforce_mode_floor: Option<bool>,
    enforce_mode_floor_all_precisions: Option<bool>,
    per_resolution: Option<Vec<WarpForcingPerResolutionToml>>,
}

#[derive(Debug, Deserialize)]
struct WarpForcingPerResolutionToml {
    resolution: usize,
    mode_y_requested: Option<usize>,
    re_target: Option<f64>,
    max_mach: Option<f64>,
    rho0: Option<f64>,
    bf16_min_delta_ulp_ratio: Option<f64>,
    bf16_enforce_mode_floor: Option<bool>,
    enforce_mode_floor_all_precisions: Option<bool>,
}

fn default_policy_path() -> PathBuf {
    PathBuf::from("registry/engineering_standards.toml")
}

fn policy_path_from_env() -> PathBuf {
    std::env::var("GOROROBA_WARP_FORCING_POLICY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_policy_path())
}

fn requested_profile_name(
    requested_profile: Option<&str>,
    profile_table: &WarpForcingProfilesToml,
) -> String {
    if let Some(name) = requested_profile {
        return name.to_string();
    }
    if let Ok(name) = std::env::var("GOROROBA_WARP_FORCING_PROFILE")
        && !name.trim().is_empty()
    {
        return name;
    }
    profile_table
        .default
        .clone()
        .unwrap_or_else(|| "kolmogorov_bf16_measured_v1".to_string())
}

fn apply_profile_overrides(base: &mut WarpForcingProfile, src: &WarpForcingProfileToml) {
    if let Some(v) = src.mode_y_requested {
        base.mode_y_requested = v;
    }
    if let Some(v) = src.re_target {
        base.re_target = v;
    }
    if let Some(v) = src.max_mach {
        base.max_mach = v;
    }
    if let Some(v) = src.rho0 {
        base.rho0 = v;
    }
    if let Some(v) = src.bf16_min_delta_ulp_ratio {
        base.bf16_min_delta_ulp_ratio = v;
    }
    if let Some(v) = src.bf16_enforce_mode_floor {
        base.bf16_enforce_mode_floor = v;
    }
    if let Some(v) = src.enforce_mode_floor_all_precisions {
        base.enforce_mode_floor_all_precisions = v;
    }
}

fn apply_resolution_overrides(
    base: &mut WarpForcingProfile,
    per_resolution: Option<&[WarpForcingPerResolutionToml]>,
    resolution: usize,
) -> bool {
    let mut used = false;
    if let Some(rows) = per_resolution {
        for row in rows {
            if row.resolution != resolution {
                continue;
            }
            used = true;
            if let Some(v) = row.mode_y_requested {
                base.mode_y_requested = v;
            }
            if let Some(v) = row.re_target {
                base.re_target = v;
            }
            if let Some(v) = row.max_mach {
                base.max_mach = v;
            }
            if let Some(v) = row.rho0 {
                base.rho0 = v;
            }
            if let Some(v) = row.bf16_min_delta_ulp_ratio {
                base.bf16_min_delta_ulp_ratio = v;
            }
            if let Some(v) = row.bf16_enforce_mode_floor {
                base.bf16_enforce_mode_floor = v;
            }
            if let Some(v) = row.enforce_mode_floor_all_precisions {
                base.enforce_mode_floor_all_precisions = v;
            }
        }
    }
    used
}

fn validate_profile(values: &WarpForcingProfile) -> Result<(), Box<dyn Error>> {
    if values.mode_y_requested == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "forcing profile invalid: mode_y_requested must be >= 1",
        )
        .into());
    }
    if !values.re_target.is_finite() || values.re_target <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "forcing profile invalid: re_target must be finite and > 0, got {}",
                values.re_target
            ),
        )
        .into());
    }
    if !values.max_mach.is_finite() || values.max_mach <= 0.0 || values.max_mach > 0.30 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "forcing profile invalid: max_mach must satisfy 0 < Ma <= 0.30, got {}",
                values.max_mach
            ),
        )
        .into());
    }
    if !values.rho0.is_finite() || values.rho0 <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "forcing profile invalid: rho0 must be finite and > 0, got {}",
                values.rho0
            ),
        )
        .into());
    }
    if !values.bf16_min_delta_ulp_ratio.is_finite() || values.bf16_min_delta_ulp_ratio < 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "forcing profile invalid: bf16_min_delta_ulp_ratio must be finite and >= 0, got {}",
                values.bf16_min_delta_ulp_ratio
            ),
        )
        .into());
    }
    Ok(())
}

pub fn load_warp_forcing_profile(
    requested_profile: Option<&str>,
    resolution: usize,
) -> Result<LoadedWarpForcingProfile, Box<dyn Error>> {
    let path = policy_path_from_env();
    load_warp_forcing_profile_from_path(&path, requested_profile, resolution)
}

pub fn load_warp_forcing_profile_from_path(
    path: &Path,
    requested_profile: Option<&str>,
    resolution: usize,
) -> Result<LoadedWarpForcingProfile, Box<dyn Error>> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        std::io::Error::other(format!(
            "failed to read warp forcing policy from {}: {e}",
            path.display()
        ))
    })?;
    let parsed: EngineeringStandardsRoot = toml::from_str(&raw).map_err(|e| {
        std::io::Error::other(format!(
            "failed to parse warp forcing policy from {}: {e}",
            path.display()
        ))
    })?;
    let profile_table = parsed
        .engineering_standards
        .warp_forcing_profiles
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "missing [engineering_standards.warp_forcing_profiles] in {}",
                path.display()
            ))
        })?;

    let profile_name = requested_profile_name(requested_profile, &profile_table);
    let source = profile_table.profiles.get(&profile_name).ok_or_else(|| {
        let mut names: Vec<_> = profile_table.profiles.keys().cloned().collect();
        names.sort();
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "unknown forcing profile '{profile_name}' in {} (available: {})",
                path.display(),
                names.join(", ")
            ),
        )
    })?;

    let mut values = WarpForcingProfile::default();
    apply_profile_overrides(&mut values, source);
    let used_resolution_override =
        apply_resolution_overrides(&mut values, source.per_resolution.as_deref(), resolution);
    validate_profile(&values)?;

    Ok(LoadedWarpForcingProfile {
        source_path: path.to_path_buf(),
        profile_name,
        resolution,
        used_resolution_override,
        values,
    })
}

pub fn apply_warp_forcing_env(values: &WarpForcingProfile) {
    // SAFETY: Called single-threaded at binary startup before any parallel work.
    unsafe {
        std::env::set_var("GOROROBA_KOLMO_MODE_Y", values.mode_y_requested.to_string());
        std::env::set_var(
            "GOROROBA_KOLMO_RE_TARGET",
            format!("{:.9}", values.re_target),
        );
        std::env::set_var("GOROROBA_KOLMO_MAX_MACH", format!("{:.9}", values.max_mach));
        std::env::set_var("GOROROBA_KOLMO_RHO0", format!("{:.9}", values.rho0));
        std::env::set_var(
            "GOROROBA_BF16_MIN_DELTA_ULP_RATIO",
            format!("{:.9}", values.bf16_min_delta_ulp_ratio),
        );
        std::env::set_var(
            "GOROROBA_BF16_ENFORCE_MODE_FLOOR",
            if values.bf16_enforce_mode_floor {
                "true"
            } else {
                "false"
            },
        );
        std::env::set_var(
            "GOROROBA_ENFORCE_MODE_FLOOR_ALL_PRECISIONS",
            if values.enforce_mode_floor_all_precisions {
                "true"
            } else {
                "false"
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_policy(contents: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough for test temp naming")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("warp_forcing_policy_{nanos}.toml"));
        std::fs::write(&path, contents).expect("failed to write temp forcing policy");
        path
    }

    #[test]
    fn load_profile_applies_resolution_override() {
        let path = write_temp_policy(
            r#"
[engineering_standards.warp_forcing_profiles]
default = "test_profile"

[engineering_standards.warp_forcing_profiles.test_profile]
mode_y_requested = 1
re_target = 128.0
max_mach = 0.10
rho0 = 1.0
bf16_min_delta_ulp_ratio = 0.25
bf16_enforce_mode_floor = true
enforce_mode_floor_all_precisions = false

[[engineering_standards.warp_forcing_profiles.test_profile.per_resolution]]
resolution = 256
mode_y_requested = 2
re_target = 192.0
enforce_mode_floor_all_precisions = true
"#,
        );
        let loaded = load_warp_forcing_profile_from_path(&path, Some("test_profile"), 256)
            .expect("profile should load");
        assert!(loaded.used_resolution_override);
        assert_eq!(loaded.values.mode_y_requested, 2);
        assert_eq!(loaded.values.re_target, 192.0);
        assert!(loaded.values.enforce_mode_floor_all_precisions);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_profile_rejects_out_of_envelope_mach() {
        let path = write_temp_policy(
            r#"
[engineering_standards.warp_forcing_profiles]
default = "bad_profile"

[engineering_standards.warp_forcing_profiles.bad_profile]
mode_y_requested = 1
re_target = 128.0
max_mach = 0.45
rho0 = 1.0
bf16_min_delta_ulp_ratio = 0.25
bf16_enforce_mode_floor = true
enforce_mode_floor_all_precisions = true
"#,
        );
        let err = load_warp_forcing_profile_from_path(&path, Some("bad_profile"), 128)
            .expect_err("profile should fail validation");
        let msg = err.to_string();
        assert!(msg.contains("0 < Ma <= 0.30"));
        let _ = std::fs::remove_file(path);
    }
}
