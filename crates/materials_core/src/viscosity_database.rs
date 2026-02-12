//! Viscosity Database: Load real materials properties from TOML registry.
//!
//! This module provides access to temperature-dependent kinematic and dynamic
//! viscosity for real materials including:
//! - Helium isotopes (He-3, He-4, normal and superfluid phases)
//! - Atmosphere (air, N2, O2)
//! - Water and ice phases (Ice Ih, Ice VII)
//! - Minerals (tourmaline/elbaite)
//! - Reference fluids (glycerol)
//!
//! All data sourced from NIST, peer-reviewed literature, or authoritative handbooks.

use serde::Deserialize;
use std::collections::HashMap;

/// Material viscosity properties.
#[allow(non_snake_case)] // Unit suffixes (_K, _Pa, _GPa) are intentional
#[derive(Debug, Clone, Deserialize)]
pub struct MaterialViscosity {
    pub id: String,
    pub name: String,
    pub formula: String,
    pub phase: String,
    pub temperature_K: f64,
    pub pressure_Pa: f64,
    pub density_kg_m3: f64,
    pub dynamic_viscosity_Pa_s: Option<f64>,
    pub kinematic_viscosity_m2_s: Option<f64>,
    pub notes: Option<String>,
    pub reference: Option<String>,

    // Optional fields for special cases
    pub crystal_structure: Option<String>,
    pub quantum_viscosity: Option<bool>,
    pub quantum_circulation: Option<f64>,
    pub creep_viscosity_Pa_s: Option<f64>,
    pub shear_modulus_GPa: Option<f64>,
    pub bulk_modulus_GPa: Option<f64>,
    pub formation_pressure_GPa: Option<f64>,
    pub melting_point_K: Option<f64>,
    pub hardness_Mohs: Option<f64>,
    pub elastic_modulus_GPa: Option<f64>,
    pub disclaimer: Option<String>,
}

/// Lambda coupling strength result.
#[allow(non_snake_case)] // Unit suffixes (_K) are intentional
#[derive(Debug, Clone, Deserialize)]
pub struct LambdaResult {
    pub material_id: String,
    pub temperature_K: f64,
    pub lambda_computed: f64,
    pub coupling_regime: String,
}

/// Viscosity database loaded from TOML registry.
#[derive(Debug, Clone, Deserialize)]
struct ViscosityDatabase {
    #[serde(rename = "material")]
    materials: Vec<MaterialViscosity>,
    #[serde(rename = "lambda_result", default)]
    lambda_results: Vec<LambdaResult>,
}

/// Load viscosity database from registry TOML.
///
/// # Returns
/// HashMap mapping material ID to MaterialViscosity struct.
///
/// # Panics
/// If registry file is missing or malformed.
pub fn load_viscosity_database() -> HashMap<String, MaterialViscosity> {
    let toml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../registry/materials_viscosity.toml"
    );

    let toml_content = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", toml_path, e));

    let db: ViscosityDatabase = toml::from_str(&toml_content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", toml_path, e));

    db.materials
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect()
}

/// Get material by ID.
///
/// # Example
/// ```ignore
/// let water = get_material("water_20C").unwrap();
/// assert_eq!(water.kinematic_viscosity_m2_s, Some(1.004e-6));
/// ```
pub fn get_material(id: &str) -> Option<MaterialViscosity> {
    let db = load_viscosity_database();
    db.get(id).cloned()
}

/// List all available material IDs.
pub fn list_materials() -> Vec<String> {
    let db = load_viscosity_database();
    let mut ids: Vec<String> = db.keys().cloned().collect();
    ids.sort();
    ids
}

/// Get materials by phase.
pub fn get_materials_by_phase(phase: &str) -> Vec<MaterialViscosity> {
    let db = load_viscosity_database();
    db.values().filter(|m| m.phase == phase).cloned().collect()
}

/// Get materials with quantum properties (superfluids).
pub fn get_quantum_fluids() -> Vec<MaterialViscosity> {
    let db = load_viscosity_database();
    db.values()
        .filter(|m| m.quantum_viscosity.unwrap_or(false))
        .cloned()
        .collect()
}

/// Convert kinematic viscosity from m^2/s to LBM lattice units.
///
/// In LBM with D3Q19, tau = 3*nu + 0.5, where nu is in lattice units.
/// To convert physical viscosity to lattice units:
/// nu_lattice = nu_physical / (dx^2 / dt)
///
/// # Arguments
/// * `nu_physical` - Kinematic viscosity in m^2/s
/// * `dx` - Lattice spacing in meters
/// * `dt` - Time step in seconds
///
/// # Returns
/// Kinematic viscosity in lattice units
pub fn to_lattice_units(nu_physical: f64, dx: f64, dt: f64) -> f64 {
    nu_physical / (dx * dx / dt)
}

/// Convert lattice relaxation time tau to physical kinematic viscosity.
///
/// # Arguments
/// * `tau` - LBM relaxation time (dimensionless)
/// * `dx` - Lattice spacing in meters
/// * `dt` - Time step in seconds
///
/// # Returns
/// Kinematic viscosity in m^2/s
pub fn from_tau(tau: f64, dx: f64, dt: f64) -> f64 {
    ((tau - 0.5) / 3.0) * (dx * dx / dt)
}

/// Compute Reynolds number from material properties.
///
/// Re = U*L/nu, where:
/// - U is characteristic velocity (m/s)
/// - L is characteristic length (m)
/// - nu is kinematic viscosity (m^2/s)
///
/// # Returns
/// Reynolds number (dimensionless)
pub fn reynolds_number(velocity_m_s: f64, length_m: f64, nu_m2_s: f64) -> f64 {
    velocity_m_s * length_m / nu_m2_s
}

/// Load lambda results from registry TOML.
///
/// Returns lambda coupling strength for frustration F=0.35 (baseline).
///
/// # Returns
/// HashMap mapping material ID to LambdaResult.
///
/// # Panics
/// If registry file is missing or malformed.
pub fn load_lambda_results() -> HashMap<String, LambdaResult> {
    let toml_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../registry/materials_viscosity.toml"
    );

    let toml_content = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", toml_path, e));

    let db: ViscosityDatabase = toml::from_str(&toml_content)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", toml_path, e));

    db.lambda_results
        .into_iter()
        .map(|r| (r.material_id.clone(), r))
        .collect()
}

/// Get lambda coupling strength for material at baseline frustration (F=0.35).
///
/// Lambda measures how strongly frustration modulates viscosity:
/// - lambda >> 100: STRONG coupling (low temperature, quantum effects)
/// - lambda ~ 10-100: MODERATE coupling
/// - lambda ~ 1-10: WEAK coupling (thermal effects dominate)
/// - lambda << 1: VERY_WEAK coupling (high temperature)
///
/// # Arguments
/// * `material_id` - Material identifier (e.g., "He4_normal", "water_20C")
///
/// # Returns
/// Lambda value at F=0.35, or None if material not found
///
/// # Example
/// ```ignore
/// let lambda = get_lambda("He4_normal").unwrap();
/// assert!(lambda > 500.0); // Strong coupling at 4.2 K
/// ```
pub fn get_lambda(material_id: &str) -> Option<f64> {
    let results = load_lambda_results();
    results.get(material_id).map(|r| r.lambda_computed)
}

/// Get coupling regime for material at baseline frustration (F=0.35).
///
/// # Arguments
/// * `material_id` - Material identifier
///
/// # Returns
/// Coupling regime: "STRONG", "MODERATE", "WEAK", or "VERY_WEAK"
///
/// # Example
/// ```ignore
/// let regime = get_coupling_regime("He4_normal").unwrap();
/// assert_eq!(regime, "STRONG");
/// ```
pub fn get_coupling_regime(material_id: &str) -> Option<String> {
    let results = load_lambda_results();
    results.get(material_id).map(|r| r.coupling_regime.clone())
}

/// List all materials with lambda data.
pub fn list_lambda_materials() -> Vec<String> {
    let results = load_lambda_results();
    let mut ids: Vec<String> = results.keys().cloned().collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_database() {
        let db = load_viscosity_database();
        assert!(db.len() > 10, "Should have multiple materials");
    }

    #[test]
    fn test_get_water() {
        let water = get_material("water_20C").unwrap();
        assert_eq!(water.phase, "liquid");
        assert!((water.kinematic_viscosity_m2_s.unwrap() - 1.004e-6).abs() < 1e-9);
    }

    #[test]
    fn test_get_he4_superfluid() {
        let he4 = get_material("He4_superfluid").unwrap();
        assert_eq!(he4.phase, "superfluid");
        assert_eq!(he4.kinematic_viscosity_m2_s, Some(0.0));
        assert!(he4.quantum_viscosity.unwrap());
    }

    #[test]
    fn test_get_ice_vii() {
        let ice = get_material("ice_VII").unwrap();
        assert_eq!(
            ice.crystal_structure,
            Some("body_centered_cubic".to_string())
        );
        assert!((ice.density_kg_m3 - 1650.0).abs() < 1.0);
        assert!(ice.shear_modulus_GPa.is_some());
    }

    #[test]
    fn test_get_tourmaline() {
        let tour = get_material("tourmaline_elbaite").unwrap();
        assert_eq!(tour.phase, "solid");
        assert!((tour.density_kg_m3 - 3060.0).abs() < 1.0);
        assert!(tour.hardness_Mohs.is_some());
    }

    #[test]
    fn test_list_materials() {
        let ids = list_materials();
        assert!(ids.contains(&"He4_normal".to_string()));
        assert!(ids.contains(&"air_STP".to_string()));
        assert!(ids.contains(&"ice_VII".to_string()));
    }

    #[test]
    fn test_get_quantum_fluids() {
        let quantum = get_quantum_fluids();
        assert!(quantum.len() >= 2); // He3 and He4 superfluids
        assert!(quantum.iter().all(|m| m.quantum_viscosity.unwrap()));
    }

    #[test]
    fn test_lattice_conversion() {
        let nu_water = 1.004e-6; // m^2/s
        let dx = 1e-5; // 10 micron grid
        let dt = 1e-7; // 0.1 microsecond

        let nu_lattice = to_lattice_units(nu_water, dx, dt);
        assert!(nu_lattice > 0.0);

        // Verify round-trip
        let tau = 3.0 * nu_lattice + 0.5;
        let nu_physical = from_tau(tau, dx, dt);
        assert!((nu_physical - nu_water).abs() < 1e-12);
    }

    #[test]
    fn test_reynolds_number() {
        let nu_water = 1.004e-6; // m^2/s at 20 C
        let velocity = 0.1; // 10 cm/s
        let length = 0.01; // 1 cm

        let re = reynolds_number(velocity, length, nu_water);
        assert!((re - 996.0).abs() < 1.0); // ~1000, laminar flow
    }

    #[test]
    fn test_reynolds_regime() {
        let nu_air = 1.516e-5; // m^2/s at 20 C
        let velocity = 10.0; // 10 m/s
        let length = 0.1; // 10 cm

        let re = reynolds_number(velocity, length, nu_air);
        assert!(re > 4000.0); // Turbulent regime
    }

    #[test]
    fn test_load_lambda_results() {
        let results = load_lambda_results();
        assert!(
            results.len() >= 16,
            "Should have lambda results for all materials"
        );
    }

    #[test]
    fn test_get_lambda_he4() {
        let lambda = get_lambda("He4_normal").unwrap();
        assert!(
            (lambda - 594.7381).abs() < 0.01,
            "He4_normal lambda mismatch"
        );
    }

    #[test]
    fn test_get_lambda_water() {
        let lambda = get_lambda("water_20C").unwrap();
        assert!((lambda - 8.5209).abs() < 0.01, "water_20C lambda mismatch");
    }

    #[test]
    fn test_get_lambda_superfluid() {
        let lambda = get_lambda("He3_superfluid_A").unwrap();
        assert!(
            lambda > 900000.0,
            "He3 superfluid should have ultra-strong coupling"
        );
    }

    #[test]
    fn test_get_coupling_regime_strong() {
        let regime = get_coupling_regime("He4_normal").unwrap();
        assert_eq!(regime, "STRONG");
    }

    #[test]
    fn test_get_coupling_regime_weak() {
        let regime = get_coupling_regime("water_20C").unwrap();
        assert_eq!(regime, "WEAK");
    }

    #[test]
    fn test_get_coupling_regime_moderate() {
        let regime = get_coupling_regime("N2_liquid").unwrap();
        assert_eq!(regime, "MODERATE");
    }

    #[test]
    fn test_list_lambda_materials() {
        let ids = list_lambda_materials();
        assert!(ids.len() >= 16);
        assert!(ids.contains(&"He4_normal".to_string()));
        assert!(ids.contains(&"water_20C".to_string()));
        assert!(ids.contains(&"ice_VII".to_string()));
    }

    #[test]
    fn test_lambda_temperature_scaling() {
        // Verify lambda is inversely proportional to temperature
        let lambda_cold = get_lambda("He4_superfluid").unwrap(); // 1.5 K
        let lambda_warm = get_lambda("He4_normal").unwrap(); // 4.2 K

        assert!(
            lambda_cold > lambda_warm,
            "Colder material should have higher lambda"
        );

        // Check approximate 1/T scaling: lambda_cold / lambda_warm ~ T_warm / T_cold
        let ratio = lambda_cold / lambda_warm;
        let temp_ratio = 4.2 / 1.5;
        assert!(
            (ratio - temp_ratio).abs() < 0.2,
            "Lambda should scale as 1/T"
        );
    }
}
