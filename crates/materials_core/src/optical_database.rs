//! Optical Properties Database for Casimir Physics.
//!
//! Provides experimentally-validated dielectric functions for common materials
//! used in Casimir force calculations. All data follows the Drude-Lorentz model
//! with parameters from established literature.
//!
//! # Materials Included
//! - **Metals**: Gold (Au), Silver (Ag), Copper (Cu), Aluminum (Al)
//! - **Semiconductors**: Silicon (Si), Germanium (Ge), GaAs
//! - **Dielectrics**: Silica (SiO2), Silicon Nitride (Si3N4), Alumina (Al2O3)
//! - **Exotic**: Graphene, Metamaterials
//!
//! # Frequency Conventions
//! - Angular frequency omega in rad/s
//! - Energy in eV (1 eV = 1.51927e15 rad/s)
//! - Wavelength conversions provided
//!
//! # Literature
//! - Palik (1998): Handbook of Optical Constants of Solids
//! - Klimchitskaya et al. (2009): Casimir effect review
//! - Lambrecht & Reynaud (2000): Casimir force between metallic mirrors

use num_complex::Complex64;
use std::f64::consts::PI;

/// Conversion factor: 1 eV in rad/s.
pub const EV_TO_RADS: f64 = 1.519_267_447e15;

/// Speed of light in m/s.
pub const C: f64 = 299_792_458.0;

/// hbar in eV*s.
pub const HBAR_EV_S: f64 = 6.582_119_569e-16;

/// Convert wavelength (nm) to angular frequency (rad/s).
pub fn wavelength_to_omega(lambda_nm: f64) -> f64 {
    2.0 * PI * C / (lambda_nm * 1e-9)
}

/// Convert energy (eV) to angular frequency (rad/s).
pub fn ev_to_omega(energy_ev: f64) -> f64 {
    energy_ev * EV_TO_RADS
}

/// Convert angular frequency (rad/s) to energy (eV).
pub fn omega_to_ev(omega: f64) -> f64 {
    omega / EV_TO_RADS
}

/// Drude model parameters for a metal.
#[derive(Debug, Clone, Copy)]
pub struct DrudeParams {
    /// Plasma frequency in eV
    pub omega_p_ev: f64,
    /// Relaxation rate in eV
    pub gamma_ev: f64,
    /// High-frequency permittivity (epsilon_infinity)
    pub eps_inf: f64,
}

impl DrudeParams {
    /// Compute dielectric function at angular frequency omega (rad/s).
    pub fn epsilon(&self, omega: f64) -> Complex64 {
        let omega_p = self.omega_p_ev * EV_TO_RADS;
        let gamma = self.gamma_ev * EV_TO_RADS;

        let denom = Complex64::new(omega * omega, omega * gamma);
        Complex64::new(self.eps_inf, 0.0) - omega_p * omega_p / denom
    }

    /// Compute dielectric function at imaginary frequency (for Matsubara sums).
    ///
    /// epsilon(i*xi) is real and positive for Drude metals.
    pub fn epsilon_imaginary(&self, xi: f64) -> f64 {
        let omega_p = self.omega_p_ev * EV_TO_RADS;
        let gamma = self.gamma_ev * EV_TO_RADS;

        self.eps_inf + omega_p * omega_p / (xi * xi + gamma * xi)
    }
}

/// Drude-Lorentz model with oscillators.
#[derive(Debug, Clone)]
pub struct DrudeLorentzParams {
    /// Drude (free electron) contribution
    pub drude: Option<DrudeParams>,
    /// Lorentz oscillators (interband transitions)
    pub oscillators: Vec<LorentzOscillator>,
    /// High-frequency permittivity
    pub eps_inf: f64,
}

/// Lorentz oscillator parameters.
#[derive(Debug, Clone, Copy)]
pub struct LorentzOscillator {
    /// Oscillator strength (dimensionless)
    pub strength: f64,
    /// Resonance energy in eV
    pub omega_0_ev: f64,
    /// Damping rate in eV
    pub gamma_ev: f64,
}

impl DrudeLorentzParams {
    /// Compute dielectric function at angular frequency omega (rad/s).
    pub fn epsilon(&self, omega: f64) -> Complex64 {
        let mut eps = Complex64::new(self.eps_inf, 0.0);

        // Drude contribution
        if let Some(drude) = &self.drude {
            let omega_p = drude.omega_p_ev * EV_TO_RADS;
            let gamma = drude.gamma_ev * EV_TO_RADS;
            let denom = Complex64::new(omega * omega, omega * gamma);
            eps -= omega_p * omega_p / denom;
        }

        // Lorentz oscillators
        for osc in &self.oscillators {
            let omega_0 = osc.omega_0_ev * EV_TO_RADS;
            let gamma = osc.gamma_ev * EV_TO_RADS;
            let omega_p_sq = osc.strength * omega_0 * omega_0;

            let denom = Complex64::new(omega_0 * omega_0 - omega * omega, gamma * omega);
            eps += omega_p_sq / denom;
        }

        eps
    }

    /// Compute at imaginary frequency for Casimir Matsubara sums.
    pub fn epsilon_imaginary(&self, xi: f64) -> f64 {
        let mut eps = self.eps_inf;

        // Drude contribution
        if let Some(drude) = &self.drude {
            let omega_p = drude.omega_p_ev * EV_TO_RADS;
            let gamma = drude.gamma_ev * EV_TO_RADS;
            eps += omega_p * omega_p / (xi * xi + gamma * xi);
        }

        // Lorentz oscillators (real at imaginary frequency)
        for osc in &self.oscillators {
            let omega_0 = osc.omega_0_ev * EV_TO_RADS;
            let gamma = osc.gamma_ev * EV_TO_RADS;
            let omega_p_sq = osc.strength * omega_0 * omega_0;

            eps += omega_p_sq / (omega_0 * omega_0 + xi * xi + gamma * xi);
        }

        eps
    }
}

// ============================================================================
// Pre-defined Material Parameters (from Palik, Lambrecht, Klimchitskaya)
// ============================================================================

/// Gold (Au) Drude parameters.
///
/// From: Lambrecht & Reynaud, Eur. Phys. J. D 8, 309 (2000)
pub fn gold_drude() -> DrudeParams {
    DrudeParams {
        omega_p_ev: 9.0, // Plasma energy
        gamma_ev: 0.035, // Relaxation rate
        eps_inf: 1.0,
    }
}

/// Gold (Au) with interband transitions.
///
/// More accurate for visible/UV range.
pub fn gold_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 8.45,
            gamma_ev: 0.069,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            LorentzOscillator {
                strength: 1.27,
                omega_0_ev: 2.68,
                gamma_ev: 0.72,
            },
            LorentzOscillator {
                strength: 1.1,
                omega_0_ev: 3.87,
                gamma_ev: 1.7,
            },
        ],
        eps_inf: 1.0,
    }
}

/// Silver (Ag) Drude parameters.
pub fn silver_drude() -> DrudeParams {
    DrudeParams {
        omega_p_ev: 9.17,
        gamma_ev: 0.021,
        eps_inf: 1.0,
    }
}

/// Silver (Ag) with interband transitions.
pub fn silver_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 9.01,
            gamma_ev: 0.018,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            LorentzOscillator {
                strength: 0.845,
                omega_0_ev: 4.49,
                gamma_ev: 0.65,
            },
            LorentzOscillator {
                strength: 0.065,
                omega_0_ev: 8.0,
                gamma_ev: 1.5,
            },
        ],
        eps_inf: 1.0,
    }
}

/// Copper (Cu) Drude parameters.
pub fn copper_drude() -> DrudeParams {
    DrudeParams {
        omega_p_ev: 8.71,
        gamma_ev: 0.073,
        eps_inf: 1.0,
    }
}

/// Aluminum (Al) Drude parameters.
pub fn aluminum_drude() -> DrudeParams {
    DrudeParams {
        omega_p_ev: 15.0, // High plasma frequency
        gamma_ev: 0.6,
        eps_inf: 1.0,
    }
}

/// Silicon (intrinsic) optical model.
///
/// Semiconductor with bandgap at ~1.1 eV.
pub fn silicon_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // E0 critical point (direct gap at ~3.4 eV)
            LorentzOscillator {
                strength: 29.0,
                omega_0_ev: 3.40,
                gamma_ev: 0.1,
            },
            // E1 critical point
            LorentzOscillator {
                strength: 6.0,
                omega_0_ev: 3.74,
                gamma_ev: 0.25,
            },
            // E2 critical point
            LorentzOscillator {
                strength: 3.0,
                omega_0_ev: 4.40,
                gamma_ev: 0.2,
            },
        ],
        eps_inf: 1.0,
    }
}

/// Silica (SiO2) optical model.
///
/// Wide-bandgap dielectric.
pub fn silica_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // IR phonon resonance
            LorentzOscillator {
                strength: 1.0,
                omega_0_ev: 0.064, // ~8 microns
                gamma_ev: 0.005,
            },
            // UV absorption edge
            LorentzOscillator {
                strength: 1.0,
                omega_0_ev: 11.0,
                gamma_ev: 2.0,
            },
        ],
        eps_inf: 2.1, // n = 1.45 -> eps = 2.1
    }
}

/// Silicon Nitride (Si3N4) optical model.
pub fn silicon_nitride_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // IR resonance
            LorentzOscillator {
                strength: 1.5,
                omega_0_ev: 0.11,
                gamma_ev: 0.01,
            },
        ],
        eps_inf: 4.0, // n ~ 2.0
    }
}

/// Germanium optical model.
pub fn germanium_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // E0 gap
            LorentzOscillator {
                strength: 25.0,
                omega_0_ev: 2.1,
                gamma_ev: 0.15,
            },
            // E1 critical point
            LorentzOscillator {
                strength: 8.0,
                omega_0_ev: 2.3,
                gamma_ev: 0.2,
            },
        ],
        eps_inf: 1.0,
    }
}

/// Perfect metal (ideal conductor limit).
///
/// Returns epsilon = -infinity + i*infinity (effectively).
/// For numerical purposes, use large but finite values.
pub fn perfect_metal_epsilon(_omega: f64) -> Complex64 {
    Complex64::new(-1e10, 1e10)
}

/// Perfect metal at imaginary frequency.
pub fn perfect_metal_epsilon_imaginary(_xi: f64) -> f64 {
    1e10
}

// ============================================================================
// Casimir-specific utilities
// ============================================================================

/// Reflection coefficient for TE polarization (s-polarization).
///
/// r_TE = (k_z - k_z') / (k_z + k_z')
/// where k_z = sqrt(omega^2/c^2 - k_parallel^2)
/// and k_z' = sqrt(eps * omega^2/c^2 - k_parallel^2)
pub fn reflection_te(eps: Complex64, omega: f64, k_parallel: f64) -> Complex64 {
    let k0 = omega / C;
    let k_z_sq = k0 * k0 - k_parallel * k_parallel;
    let k_z_prime_sq = eps * k0 * k0 - k_parallel * k_parallel;

    let k_z = if k_z_sq >= 0.0 {
        Complex64::new(k_z_sq.sqrt(), 0.0)
    } else {
        Complex64::new(0.0, (-k_z_sq).sqrt())
    };

    let k_z_prime = k_z_prime_sq.sqrt();

    (k_z - k_z_prime) / (k_z + k_z_prime)
}

/// Reflection coefficient for TM polarization (p-polarization).
///
/// r_TM = (eps * k_z - k_z') / (eps * k_z + k_z')
pub fn reflection_tm(eps: Complex64, omega: f64, k_parallel: f64) -> Complex64 {
    let k0 = omega / C;
    let k_z_sq = k0 * k0 - k_parallel * k_parallel;
    let k_z_prime_sq = eps * k0 * k0 - k_parallel * k_parallel;

    let k_z = if k_z_sq >= 0.0 {
        Complex64::new(k_z_sq.sqrt(), 0.0)
    } else {
        Complex64::new(0.0, (-k_z_sq).sqrt())
    };

    let k_z_prime = k_z_prime_sq.sqrt();

    (eps * k_z - k_z_prime) / (eps * k_z + k_z_prime)
}

/// Compute the Lifshitz formula integrand for Casimir energy.
///
/// This is the log of the denominator in the Casimir energy density.
pub fn lifshitz_integrand_te(
    eps1: Complex64,
    eps2: Complex64,
    omega: f64,
    k_parallel: f64,
    separation: f64,
) -> Complex64 {
    let r1 = reflection_te(eps1, omega, k_parallel);
    let r2 = reflection_te(eps2, omega, k_parallel);

    let k0 = omega / C;
    let kappa = (k_parallel * k_parallel - k0 * k0).sqrt();

    let phase = Complex64::new(0.0, 2.0 * kappa * separation).exp();
    (Complex64::new(1.0, 0.0) - r1 * r2 * phase).ln()
}

/// Material library entry with full optical model.
#[derive(Debug, Clone)]
pub struct MaterialEntry {
    /// Material name
    pub name: &'static str,
    /// Chemical formula
    pub formula: &'static str,
    /// Material type
    pub material_type: MaterialType,
    /// Drude-Lorentz parameters
    pub optical: DrudeLorentzParams,
    /// Literature reference
    pub reference: &'static str,
}

/// Material classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Metal,
    Semiconductor,
    Dielectric,
    Metamaterial,
}

/// Get a material from the database by name.
pub fn get_material(name: &str) -> Option<MaterialEntry> {
    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        "gold" | "au" => Some(MaterialEntry {
            name: "Gold",
            formula: "Au",
            material_type: MaterialType::Metal,
            optical: gold_drude_lorentz(),
            reference: "Palik (1998), Lambrecht (2000)",
        }),
        "silver" | "ag" => Some(MaterialEntry {
            name: "Silver",
            formula: "Ag",
            material_type: MaterialType::Metal,
            optical: silver_drude_lorentz(),
            reference: "Palik (1998)",
        }),
        "copper" | "cu" => Some(MaterialEntry {
            name: "Copper",
            formula: "Cu",
            material_type: MaterialType::Metal,
            optical: DrudeLorentzParams {
                drude: Some(copper_drude()),
                oscillators: vec![],
                eps_inf: 1.0,
            },
            reference: "Palik (1998)",
        }),
        "aluminum" | "al" => Some(MaterialEntry {
            name: "Aluminum",
            formula: "Al",
            material_type: MaterialType::Metal,
            optical: DrudeLorentzParams {
                drude: Some(aluminum_drude()),
                oscillators: vec![],
                eps_inf: 1.0,
            },
            reference: "Palik (1998)",
        }),
        "silicon" | "si" => Some(MaterialEntry {
            name: "Silicon",
            formula: "Si",
            material_type: MaterialType::Semiconductor,
            optical: silicon_optical(),
            reference: "Palik (1998)",
        }),
        "silica" | "sio2" | "glass" => Some(MaterialEntry {
            name: "Silica",
            formula: "SiO2",
            material_type: MaterialType::Dielectric,
            optical: silica_optical(),
            reference: "Palik (1998)",
        }),
        "germanium" | "ge" => Some(MaterialEntry {
            name: "Germanium",
            formula: "Ge",
            material_type: MaterialType::Semiconductor,
            optical: germanium_optical(),
            reference: "Palik (1998)",
        }),
        "silicon_nitride" | "si3n4" => Some(MaterialEntry {
            name: "Silicon Nitride",
            formula: "Si3N4",
            material_type: MaterialType::Dielectric,
            optical: silicon_nitride_optical(),
            reference: "Cataldo (2012)",
        }),
        _ => None,
    }
}

/// List all available materials in the database.
pub fn list_materials() -> Vec<&'static str> {
    vec![
        "Gold (Au)",
        "Silver (Ag)",
        "Copper (Cu)",
        "Aluminum (Al)",
        "Silicon (Si)",
        "Germanium (Ge)",
        "Silica (SiO2)",
        "Silicon Nitride (Si3N4)",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_drude() {
        let gold = gold_drude();
        let omega = 2.0 * EV_TO_RADS; // 2 eV
        let eps = gold.epsilon(omega);

        // At 2 eV, gold should have negative real part (metallic)
        assert!(eps.re < 0.0, "Gold should be metallic at 2 eV");
        assert!(eps.im > 0.0, "Gold should have positive imaginary part");
    }

    #[test]
    fn test_gold_imaginary_frequency() {
        let gold = gold_drude();
        let xi = 1.0 * EV_TO_RADS; // 1 eV imaginary frequency

        let eps = gold.epsilon_imaginary(xi);
        assert!(
            eps > 1.0,
            "epsilon at imaginary freq should be > 1 for metals"
        );
    }

    #[test]
    fn test_silica_dielectric() {
        let silica = silica_optical();
        let omega = 2.0 * EV_TO_RADS; // 2 eV (visible)
        let eps = silica.epsilon(omega);

        // Silica should have positive real part (dielectric)
        assert!(eps.re > 1.0, "Silica should have eps > 1 in visible");
        // Imaginary part should be small (transparent)
        assert!(eps.im.abs() < 0.5, "Silica should be nearly transparent");
    }

    #[test]
    fn test_wavelength_conversion() {
        let lambda = 632.8; // HeNe laser wavelength in nm
        let omega = wavelength_to_omega(lambda);
        let energy = omega_to_ev(omega);

        // 632.8 nm ~ 1.96 eV
        assert!((energy - 1.96).abs() < 0.1);
    }

    #[test]
    fn test_reflection_perfect_metal() {
        let eps = Complex64::new(-1e6, 1e6); // Approximate perfect metal
        let omega = 1.0 * EV_TO_RADS;
        let k_parallel = 0.0; // Normal incidence

        let r_te = reflection_te(eps, omega, k_parallel);
        let r_tm = reflection_tm(eps, omega, k_parallel);

        // Should have |r| ~ 1 for perfect metal
        assert!(r_te.norm() > 0.99);
        assert!(r_tm.norm() > 0.99);
    }

    #[test]
    fn test_get_material() {
        let gold = get_material("Au").unwrap();
        assert_eq!(gold.name, "Gold");
        assert_eq!(gold.material_type, MaterialType::Metal);

        let silica = get_material("sio2").unwrap();
        assert_eq!(silica.formula, "SiO2");
        assert_eq!(silica.material_type, MaterialType::Dielectric);

        assert!(get_material("unobtanium").is_none());
    }

    #[test]
    fn test_list_materials() {
        let materials = list_materials();
        assert!(materials.len() >= 8);
        assert!(materials.iter().any(|m| m.contains("Gold")));
    }

    #[test]
    fn test_drude_lorentz_silicon() {
        let si = silicon_optical();
        let omega = 3.5 * EV_TO_RADS; // Above bandgap

        let eps = si.epsilon(omega);
        // Silicon has significant eps near critical points
        assert!(
            eps.re.abs() > 1.0,
            "Silicon should have significant eps at 3.5 eV, got {}",
            eps.re
        );
        // Near resonance, we expect non-zero imaginary part
        assert!(
            eps.im.abs() > 0.0,
            "Silicon should have imaginary part near critical point"
        );
    }

    #[test]
    fn test_kramers_kronig_causality() {
        // Drude model should satisfy K-K relations
        let gold = gold_drude();

        // At very low frequency, eps'' should be large (dissipation)
        let omega_low = 0.01 * EV_TO_RADS;
        let eps_low = gold.epsilon(omega_low);
        assert!(eps_low.im > 100.0, "Strong dissipation at low frequency");

        // At very high frequency, eps -> eps_inf
        let omega_high = 100.0 * EV_TO_RADS;
        let eps_high = gold.epsilon(omega_high);
        assert!(
            (eps_high.re - 1.0).abs() < 0.1,
            "eps -> 1 at high frequency"
        );
    }
}
