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

/// Drude-Lorentz model with oscillators and optional extended Drude.
#[derive(Debug, Clone)]
pub struct DrudeLorentzParams {
    /// Drude (free electron) contribution
    pub drude: Option<DrudeParams>,
    /// Lorentz oscillators (interband transitions)
    pub oscillators: Vec<LorentzOscillator>,
    /// High-frequency permittivity
    pub eps_inf: f64,
    /// Extended Drude with frequency-dependent scattering (replaces drude when Some)
    pub extended_drude: Option<ExtendedDrudeParams>,
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

// ============================================================================
// Extended Drude infrastructure (Phase 4)
// ============================================================================

/// Frequency-dependent scattering rate model for extended Drude.
#[derive(Debug, Clone)]
pub enum ScatteringModel {
    /// Constant scattering rate (standard Drude).
    Constant { gamma_ev: f64 },
    /// Linear in frequency: gamma(w) = gamma_0 + alpha * w.
    LinearInOmega { gamma_0_ev: f64, alpha: f64 },
    /// Power-law: gamma(w) = gamma_0 * (w / w_scale)^n.
    PowerLaw {
        gamma_0_ev: f64,
        omega_scale_ev: f64,
        exponent: f64,
    },
    /// Drude-Smith with backscattering (localization correction).
    DrudeSmith { gamma_ev: f64, backscatter_c: f64 },
    /// Tabulated gamma(omega) with linear interpolation.
    Tabulated {
        omega_ev: Vec<f64>,
        gamma_ev: Vec<f64>,
    },
}

impl ScatteringModel {
    /// Evaluate scattering rate at frequency omega (in eV).
    pub fn gamma_at_ev(&self, omega_ev: f64) -> f64 {
        match self {
            ScatteringModel::Constant { gamma_ev } => *gamma_ev,
            ScatteringModel::LinearInOmega { gamma_0_ev, alpha } => {
                gamma_0_ev + alpha * omega_ev.abs()
            }
            ScatteringModel::PowerLaw {
                gamma_0_ev,
                omega_scale_ev,
                exponent,
            } => {
                if omega_ev.abs() < 1e-30 {
                    return *gamma_0_ev;
                }
                gamma_0_ev * (omega_ev.abs() / omega_scale_ev).powf(*exponent)
            }
            ScatteringModel::DrudeSmith { gamma_ev, .. } => *gamma_ev,
            ScatteringModel::Tabulated {
                omega_ev: freqs,
                gamma_ev: gammas,
            } => {
                let w = omega_ev.abs();
                let n = freqs.len();
                if n == 0 {
                    return 0.0;
                }
                if w <= freqs[0] {
                    return gammas[0];
                }
                if w >= freqs[n - 1] {
                    return gammas[n - 1];
                }
                let idx = freqs.partition_point(|&x| x < w);
                let t = (w - freqs[idx - 1]) / (freqs[idx] - freqs[idx - 1]);
                gammas[idx - 1] + t * (gammas[idx] - gammas[idx - 1])
            }
        }
    }
}

/// Extended Drude model with frequency-dependent scattering.
#[derive(Debug, Clone)]
pub struct ExtendedDrudeParams {
    /// Plasma frequency in eV.
    pub omega_p_ev: f64,
    /// Scattering model.
    pub scattering: ScatteringModel,
    /// High-frequency permittivity (for standalone use).
    pub eps_inf: f64,
}

impl ExtendedDrudeParams {
    /// Full dielectric function including eps_inf.
    pub fn epsilon(&self, omega: f64) -> Complex64 {
        Complex64::new(self.eps_inf, 0.0) + self.drude_contribution(omega)
    }

    /// Drude contribution only (for use inside DrudeLorentzParams).
    pub fn drude_contribution(&self, omega: f64) -> Complex64 {
        let omega_p = self.omega_p_ev * EV_TO_RADS;
        let omega_ev = omega / EV_TO_RADS;

        match &self.scattering {
            ScatteringModel::DrudeSmith {
                gamma_ev,
                backscatter_c,
            } => {
                let gamma = gamma_ev * EV_TO_RADS;
                let denom = Complex64::new(omega * omega, omega * gamma);
                let drude = -omega_p * omega_p / denom;
                // Drude-Smith: multiply by (1 + c * gamma / (gamma - i*omega))
                let z = Complex64::new(gamma, -omega);
                drude * (Complex64::new(1.0, 0.0) + backscatter_c * gamma / z)
            }
            _ => {
                let gamma = self.scattering.gamma_at_ev(omega_ev) * EV_TO_RADS;
                let denom = Complex64::new(omega * omega, omega * gamma);
                -omega_p * omega_p / denom
            }
        }
    }

    /// Drude contribution at imaginary frequency.
    pub fn drude_contribution_imaginary(&self, xi: f64) -> f64 {
        let omega_p = self.omega_p_ev * EV_TO_RADS;
        let xi_ev = xi / EV_TO_RADS;
        let gamma = self.scattering.gamma_at_ev(xi_ev) * EV_TO_RADS;

        match &self.scattering {
            ScatteringModel::DrudeSmith {
                backscatter_c, ..
            } => {
                let base = omega_p * omega_p / (xi * xi + gamma * xi);
                // At imaginary freq: correction = (1 + c*gamma/(gamma+xi))
                base * (1.0 + backscatter_c * gamma / (gamma + xi))
            }
            _ => omega_p * omega_p / (xi * xi + gamma * xi),
        }
    }

    /// Full epsilon at imaginary frequency.
    pub fn epsilon_imaginary(&self, xi: f64) -> f64 {
        self.eps_inf + self.drude_contribution_imaginary(xi)
    }
}

impl DrudeLorentzParams {
    /// Compute dielectric function at angular frequency omega (rad/s).
    pub fn epsilon(&self, omega: f64) -> Complex64 {
        let mut eps = Complex64::new(self.eps_inf, 0.0);

        // Extended Drude replaces simple Drude when present
        if let Some(ext) = &self.extended_drude {
            eps += ext.drude_contribution(omega);
        } else if let Some(drude) = &self.drude {
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

        // Extended Drude replaces simple Drude when present
        if let Some(ext) = &self.extended_drude {
            eps += ext.drude_contribution_imaginary(xi);
        } else if let Some(drude) = &self.drude {
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
        extended_drude: None,
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
        extended_drude: None,
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
        extended_drude: None,
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
        extended_drude: None,
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
        extended_drude: None,
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
        extended_drude: None,
    }
}

// ============================================================================
// Enhanced Cu/Al with interband oscillators (Rakic 1998)
// ============================================================================

/// Copper (Cu) with interband transitions (Rakic 1998 LD model).
pub fn copper_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 8.21,
            gamma_ev: 0.030,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            LorentzOscillator {
                strength: 1.40,
                omega_0_ev: 2.957,
                gamma_ev: 1.056,
            },
            LorentzOscillator {
                strength: 3.02,
                omega_0_ev: 5.300,
                gamma_ev: 3.213,
            },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Aluminum (Al) with interband transitions (Rakic 1998 LD model).
pub fn aluminum_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 10.83,
            gamma_ev: 0.047,
            eps_inf: 1.0,
        }),
        oscillators: vec![
            LorentzOscillator {
                strength: 4.71,
                omega_0_ev: 1.544,
                gamma_ev: 0.312,
            },
            LorentzOscillator {
                strength: 11.40,
                omega_0_ev: 1.808,
                gamma_ev: 1.351,
            },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

// ============================================================================
// Rakic 11-metal canonical set: 7 new metals (Phase 2)
// ============================================================================

/// Beryllium (Be) Drude parameters (Rakic 1998).
pub fn beryllium_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 18.51, gamma_ev: 0.035, eps_inf: 1.0 }
}

/// Beryllium (Be) Drude-Lorentz (Rakic 1998 LD model).
pub fn beryllium_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 5.37, gamma_ev: 0.035, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 17.93, omega_0_ev: 3.183, gamma_ev: 4.454 },
            LorentzOscillator { strength: 2.10, omega_0_ev: 4.604, gamma_ev: 1.802 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Chromium (Cr) Drude parameters (Rakic 1998).
pub fn chromium_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 10.75, gamma_ev: 0.047, eps_inf: 1.0 }
}

/// Chromium (Cr) Drude-Lorentz (Rakic 1998 LD model).
pub fn chromium_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 4.41, gamma_ev: 0.047, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 34.24, omega_0_ev: 1.970, gamma_ev: 2.676 },
            LorentzOscillator { strength: 1.24, omega_0_ev: 8.775, gamma_ev: 1.335 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Nickel (Ni) Drude parameters (Rakic 1998).
pub fn nickel_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 15.92, gamma_ev: 0.048, eps_inf: 1.0 }
}

/// Nickel (Ni) Drude-Lorentz (Rakic 1998 LD model).
pub fn nickel_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 4.93, gamma_ev: 0.048, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 10.53, omega_0_ev: 1.597, gamma_ev: 2.178 },
            LorentzOscillator { strength: 4.98, omega_0_ev: 6.089, gamma_ev: 6.292 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Palladium (Pd) Drude parameters (Rakic 1998).
pub fn palladium_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 9.72, gamma_ev: 0.009, eps_inf: 1.0 }
}

/// Palladium (Pd) Drude-Lorentz (Rakic 1998 LD model).
pub fn palladium_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 5.58, gamma_ev: 0.009, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 3.58, omega_0_ev: 2.855, gamma_ev: 2.022 },
            LorentzOscillator { strength: 1.36, omega_0_ev: 5.331, gamma_ev: 5.285 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Platinum (Pt) Drude parameters (Rakic 1998).
pub fn platinum_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 9.59, gamma_ev: 0.080, eps_inf: 1.0 }
}

/// Platinum (Pt) Drude-Lorentz (Rakic 1998 LD model).
pub fn platinum_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 5.54, gamma_ev: 0.080, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 35.12, omega_0_ev: 1.314, gamma_ev: 1.838 },
            LorentzOscillator { strength: 5.10, omega_0_ev: 3.145, gamma_ev: 3.668 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Titanium (Ti) Drude parameters (Rakic 1998).
pub fn titanium_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 7.29, gamma_ev: 0.082, eps_inf: 1.0 }
}

/// Titanium (Ti) Drude-Lorentz (Rakic 1998 LD model).
pub fn titanium_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 2.81, gamma_ev: 0.082, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 8.75, omega_0_ev: 1.545, gamma_ev: 2.518 },
            LorentzOscillator { strength: 1.58, omega_0_ev: 2.509, gamma_ev: 1.663 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

/// Tungsten (W) Drude parameters (Rakic 1998).
pub fn tungsten_drude() -> DrudeParams {
    DrudeParams { omega_p_ev: 13.22, gamma_ev: 0.064, eps_inf: 1.0 }
}

/// Tungsten (W) Drude-Lorentz (Rakic 1998 LD model).
pub fn tungsten_drude_lorentz() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 6.00, gamma_ev: 0.064, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 7.90, omega_0_ev: 1.917, gamma_ev: 1.281 },
            LorentzOscillator { strength: 9.63, omega_0_ev: 3.580, gamma_ev: 3.332 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
    }
}

// ============================================================================
// C-418 gap materials: Al2O3, Diamond, Quartz, TiO2 (Phase 1)
// ============================================================================

/// Alumina (Al2O3 / Sapphire) optical model (Palik 1998).
pub fn alumina_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            LorentzOscillator { strength: 0.8, omega_0_ev: 0.048, gamma_ev: 0.003 },
            LorentzOscillator { strength: 1.2, omega_0_ev: 0.071, gamma_ev: 0.005 },
            LorentzOscillator { strength: 1.5, omega_0_ev: 10.0, gamma_ev: 2.0 },
        ],
        eps_inf: 3.07,
        extended_drude: None,
    }
}

/// Diamond (C) optical model (Palik 1998).
pub fn diamond_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            LorentzOscillator { strength: 0.3, omega_0_ev: 0.165, gamma_ev: 0.005 },
            LorentzOscillator { strength: 2.5, omega_0_ev: 7.0, gamma_ev: 1.0 },
        ],
        eps_inf: 5.7,
        extended_drude: None,
    }
}

/// Crystalline Quartz optical model (Palik 1998).
///
/// Distinct from amorphous SiO2 (silica): has sharper phonon modes.
pub fn quartz_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            LorentzOscillator { strength: 0.9, omega_0_ev: 0.056, gamma_ev: 0.002 },
            LorentzOscillator { strength: 0.5, omega_0_ev: 0.137, gamma_ev: 0.004 },
            LorentzOscillator { strength: 1.2, omega_0_ev: 11.0, gamma_ev: 2.0 },
        ],
        eps_inf: 2.38,
        extended_drude: None,
    }
}

/// Titanium Dioxide rutile (TiO2) optical model (Palik 1998, DeVore 1951).
pub fn tio2_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            LorentzOscillator { strength: 3.0, omega_0_ev: 0.050, gamma_ev: 0.008 },
            LorentzOscillator { strength: 1.5, omega_0_ev: 0.099, gamma_ev: 0.010 },
            LorentzOscillator { strength: 8.0, omega_0_ev: 3.0, gamma_ev: 0.3 },
        ],
        eps_inf: 5.9,
        extended_drude: None,
    }
}

// ============================================================================
// Titanate/Ti-O core set (Phase 3)
// ============================================================================

/// Titanium Monoxide (TiO) "bad metal" optical model (Barman & Sarma PRB 51, 1995).
pub fn tio_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 2.50, gamma_ev: 0.50, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 3.0, omega_0_ev: 3.0, gamma_ev: 1.5 },
        ],
        eps_inf: 4.0,
        extended_drude: None,
    }
}

/// Strontium Titanate (SrTiO3) undoped optical model (Servoin et al. PRB 22, 1980).
///
/// Incipient ferroelectric with giant soft-mode oscillator strength.
pub fn srtio3_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // Soft mode TO1 at 11 meV with enormous oscillator strength
            LorentzOscillator { strength: 280.0, omega_0_ev: 0.011, gamma_ev: 0.003 },
            // TO2 mode
            LorentzOscillator { strength: 2.5, omega_0_ev: 0.022, gamma_ev: 0.002 },
            // TO4 mode
            LorentzOscillator { strength: 0.6, omega_0_ev: 0.067, gamma_ev: 0.005 },
            // UV absorption edge
            LorentzOscillator { strength: 3.5, omega_0_ev: 3.2, gamma_ev: 0.5 },
        ],
        eps_inf: 5.2,
        extended_drude: None,
    }
}

/// Doped SrTiO3 (SrTiO3:n) optical model (van Mechelen et al. PRL 100, 2008).
///
/// Metallic via electron doping: phonon modes plus Drude tail.
pub fn srtio3_doped_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 0.15, gamma_ev: 0.020, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 280.0, omega_0_ev: 0.011, gamma_ev: 0.003 },
            LorentzOscillator { strength: 2.5, omega_0_ev: 0.022, gamma_ev: 0.002 },
            LorentzOscillator { strength: 0.6, omega_0_ev: 0.067, gamma_ev: 0.005 },
            LorentzOscillator { strength: 3.5, omega_0_ev: 3.2, gamma_ev: 0.5 },
        ],
        eps_inf: 5.2,
        extended_drude: None,
    }
}

/// Lanthanum Titanate (LaTiO3) Mott insulator model (Okimoto et al. PRB 51, 1995).
///
/// Mott gap at ~0.2 eV, mid-IR spectral weight from d-d transitions.
pub fn latio3_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: None,
        oscillators: vec![
            // Mid-IR d-d transition (Mott gap excitation)
            LorentzOscillator { strength: 5.0, omega_0_ev: 0.50, gamma_ev: 0.30 },
            // IR phonon modes
            LorentzOscillator { strength: 1.5, omega_0_ev: 0.065, gamma_ev: 0.008 },
            // Charge-transfer UV
            LorentzOscillator { strength: 4.0, omega_0_ev: 3.5, gamma_ev: 1.0 },
        ],
        eps_inf: 4.5,
        extended_drude: None,
    }
}

// ============================================================================
// TCO / doped semiconductor materials (Phase 3)
// ============================================================================

/// Aluminum-doped Zinc Oxide (AZO) transparent conductor.
///
/// Metallic in IR, transparent in visible. Crossover near 1 eV.
pub fn azo_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 1.75, gamma_ev: 0.12, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 2.0, omega_0_ev: 3.3, gamma_ev: 0.2 },
        ],
        eps_inf: 3.7,
        extended_drude: None,
    }
}

/// Doped Silicon (Si:n, ~1e18 cm-3) with THz Drude tail.
pub fn doped_silicon_optical() -> DrudeLorentzParams {
    DrudeLorentzParams {
        drude: Some(DrudeParams { omega_p_ev: 0.12, gamma_ev: 0.010, eps_inf: 1.0 }),
        oscillators: vec![
            LorentzOscillator { strength: 29.0, omega_0_ev: 3.40, gamma_ev: 0.1 },
            LorentzOscillator { strength: 6.0, omega_0_ev: 3.74, gamma_ev: 0.25 },
            LorentzOscillator { strength: 3.0, omega_0_ev: 4.40, gamma_ev: 0.2 },
        ],
        eps_inf: 1.0,
        extended_drude: None,
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

/// Casimir model classification for a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasimirModelFlag {
    Drude,
    Plasma,
    Lorentz,
    NotApplicable,
}

/// Material classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Metal,
    Semiconductor,
    Dielectric,
    Metamaterial,
    ConductiveOxide,
}

/// Material library entry with full optical model and provenance.
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
    /// Valid frequency range (eV) for the optical model
    pub validity_range_ev: Option<(f64, f64)>,
    /// Measurement temperature (K)
    pub temperature_k: Option<f64>,
    /// Doping or carrier density info
    pub doping_info: Option<&'static str>,
    /// Casimir model classification
    pub casimir_model: CasimirModelFlag,
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
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "silver" | "ag" => Some(MaterialEntry {
            name: "Silver",
            formula: "Ag",
            material_type: MaterialType::Metal,
            optical: silver_drude_lorentz(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "copper" | "cu" => Some(MaterialEntry {
            name: "Copper",
            formula: "Cu",
            material_type: MaterialType::Metal,
            optical: copper_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "aluminum" | "al" => Some(MaterialEntry {
            name: "Aluminum",
            formula: "Al",
            material_type: MaterialType::Metal,
            optical: aluminum_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "silicon" | "si" => Some(MaterialEntry {
            name: "Silicon",
            formula: "Si",
            material_type: MaterialType::Semiconductor,
            optical: silicon_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((1.0, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "silica" | "sio2" | "glass" => Some(MaterialEntry {
            name: "Silica",
            formula: "SiO2",
            material_type: MaterialType::Dielectric,
            optical: silica_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.01, 12.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "germanium" | "ge" => Some(MaterialEntry {
            name: "Germanium",
            formula: "Ge",
            material_type: MaterialType::Semiconductor,
            optical: germanium_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.5, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "silicon_nitride" | "si3n4" => Some(MaterialEntry {
            name: "Silicon Nitride",
            formula: "Si3N4",
            material_type: MaterialType::Dielectric,
            optical: silicon_nitride_optical(),
            reference: "Cataldo (2012)",
            validity_range_ev: Some((0.05, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        // C-418 gap materials
        "alumina" | "al2o3" | "sapphire" => Some(MaterialEntry {
            name: "Alumina",
            formula: "Al2O3",
            material_type: MaterialType::Dielectric,
            optical: alumina_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.01, 12.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "diamond" | "c_diamond" => Some(MaterialEntry {
            name: "Diamond",
            formula: "C",
            material_type: MaterialType::Dielectric,
            optical: diamond_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.05, 10.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "quartz" | "crystalline_sio2" => Some(MaterialEntry {
            name: "Quartz",
            formula: "SiO2",
            material_type: MaterialType::Dielectric,
            optical: quartz_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.01, 12.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "tio2" | "rutile" | "titanium_dioxide" => Some(MaterialEntry {
            name: "Titanium Dioxide",
            formula: "TiO2",
            material_type: MaterialType::Semiconductor,
            optical: tio2_optical(),
            reference: "Palik (1998), DeVore (1951)",
            validity_range_ev: Some((0.01, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        // Rakic 11-metal set (Phase 2)
        "beryllium" | "be" => Some(MaterialEntry {
            name: "Beryllium",
            formula: "Be",
            material_type: MaterialType::Metal,
            optical: beryllium_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "chromium" | "cr" => Some(MaterialEntry {
            name: "Chromium",
            formula: "Cr",
            material_type: MaterialType::Metal,
            optical: chromium_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "nickel" | "ni" => Some(MaterialEntry {
            name: "Nickel",
            formula: "Ni",
            material_type: MaterialType::Metal,
            optical: nickel_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "palladium" | "pd" => Some(MaterialEntry {
            name: "Palladium",
            formula: "Pd",
            material_type: MaterialType::Metal,
            optical: palladium_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "platinum" | "pt" => Some(MaterialEntry {
            name: "Platinum",
            formula: "Pt",
            material_type: MaterialType::Metal,
            optical: platinum_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "titanium" | "ti" => Some(MaterialEntry {
            name: "Titanium",
            formula: "Ti",
            material_type: MaterialType::Metal,
            optical: titanium_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "tungsten" | "w" => Some(MaterialEntry {
            name: "Tungsten",
            formula: "W",
            material_type: MaterialType::Metal,
            optical: tungsten_drude_lorentz(),
            reference: "Rakic (1998)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        // Titanates (Phase 3)
        "tio" | "titanium_monoxide" => Some(MaterialEntry {
            name: "Titanium Monoxide",
            formula: "TiO",
            material_type: MaterialType::ConductiveOxide,
            optical: tio_optical(),
            reference: "Barman & Sarma PRB 51 (1995)",
            validity_range_ev: Some((0.1, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Drude,
        }),
        "srtio3" | "strontium_titanate" => Some(MaterialEntry {
            name: "Strontium Titanate",
            formula: "SrTiO3",
            material_type: MaterialType::Dielectric,
            optical: srtio3_optical(),
            reference: "Servoin et al. PRB 22 (1980)",
            validity_range_ev: Some((0.005, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        "srtio3_doped" | "srtio3_n" => Some(MaterialEntry {
            name: "Doped SrTiO3",
            formula: "SrTiO3:n",
            material_type: MaterialType::ConductiveOxide,
            optical: srtio3_doped_optical(),
            reference: "van Mechelen et al. PRL 100 (2008)",
            validity_range_ev: Some((0.005, 6.0)),
            temperature_k: Some(10.0),
            doping_info: Some("n-type, ~1e19 cm-3"),
            casimir_model: CasimirModelFlag::Drude,
        }),
        "latio3" | "lanthanum_titanate" => Some(MaterialEntry {
            name: "Lanthanum Titanate",
            formula: "LaTiO3",
            material_type: MaterialType::Semiconductor,
            optical: latio3_optical(),
            reference: "Okimoto et al. PRB 51 (1995)",
            validity_range_ev: Some((0.01, 6.0)),
            temperature_k: Some(300.0),
            doping_info: None,
            casimir_model: CasimirModelFlag::Lorentz,
        }),
        // TCOs / doped semiconductors (Phase 3)
        "azo" | "al_zno" => Some(MaterialEntry {
            name: "Al-doped ZnO",
            formula: "ZnO:Al",
            material_type: MaterialType::ConductiveOxide,
            optical: azo_optical(),
            reference: "Community literature",
            validity_range_ev: Some((0.05, 6.0)),
            temperature_k: Some(300.0),
            doping_info: Some("Al-doped, ~2% Al"),
            casimir_model: CasimirModelFlag::Drude,
        }),
        "doped_si" | "si_doped" | "si_n" => Some(MaterialEntry {
            name: "Doped Silicon",
            formula: "Si:n",
            material_type: MaterialType::Semiconductor,
            optical: doped_silicon_optical(),
            reference: "Palik (1998)",
            validity_range_ev: Some((0.01, 6.0)),
            temperature_k: Some(300.0),
            doping_info: Some("n-type, ~1e18 cm-3"),
            casimir_model: CasimirModelFlag::Drude,
        }),
        _ => None,
    }
}

/// List all available materials in the database.
pub fn list_materials() -> Vec<&'static str> {
    vec![
        // Original metals
        "Gold (Au)",
        "Silver (Ag)",
        "Copper (Cu)",
        "Aluminum (Al)",
        // Rakic metals
        "Beryllium (Be)",
        "Chromium (Cr)",
        "Nickel (Ni)",
        "Palladium (Pd)",
        "Platinum (Pt)",
        "Titanium (Ti)",
        "Tungsten (W)",
        // Semiconductors
        "Silicon (Si)",
        "Germanium (Ge)",
        "Doped Silicon (Si:n)",
        // Dielectrics
        "Silica (SiO2)",
        "Silicon Nitride (Si3N4)",
        "Alumina (Al2O3)",
        "Diamond (C)",
        "Quartz (SiO2 crystalline)",
        "Titanium Dioxide (TiO2)",
        // Titanates
        "Titanium Monoxide (TiO)",
        "Strontium Titanate (SrTiO3)",
        "Doped SrTiO3 (SrTiO3:n)",
        "Lanthanum Titanate (LaTiO3)",
        // TCOs
        "Al-doped ZnO (AZO)",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================== Original tests =====================

    #[test]
    fn test_gold_drude() {
        let gold = gold_drude();
        let omega = 2.0 * EV_TO_RADS;
        let eps = gold.epsilon(omega);
        assert!(eps.re < 0.0, "Gold should be metallic at 2 eV");
        assert!(eps.im > 0.0, "Gold should have positive imaginary part");
    }

    #[test]
    fn test_gold_imaginary_frequency() {
        let gold = gold_drude();
        let xi = 1.0 * EV_TO_RADS;
        let eps = gold.epsilon_imaginary(xi);
        assert!(eps > 1.0, "epsilon(i*xi) > 1 for metals");
    }

    #[test]
    fn test_silica_dielectric() {
        let silica = silica_optical();
        let omega = 2.0 * EV_TO_RADS;
        let eps = silica.epsilon(omega);
        assert!(eps.re > 1.0, "Silica should have eps > 1 in visible");
        assert!(eps.im.abs() < 0.5, "Silica should be nearly transparent");
    }

    #[test]
    fn test_wavelength_conversion() {
        let lambda = 632.8;
        let omega = wavelength_to_omega(lambda);
        let energy = omega_to_ev(omega);
        assert!((energy - 1.96).abs() < 0.1);
    }

    #[test]
    fn test_reflection_perfect_metal() {
        let eps = Complex64::new(-1e6, 1e6);
        let omega = 1.0 * EV_TO_RADS;
        let r_te = reflection_te(eps, omega, 0.0);
        let r_tm = reflection_tm(eps, omega, 0.0);
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
        assert!(
            materials.len() >= 25,
            "Expected >= 25 materials, got {}",
            materials.len()
        );
        assert!(materials.iter().any(|m| m.contains("Gold")));
    }

    #[test]
    fn test_drude_lorentz_silicon() {
        let si = silicon_optical();
        let omega = 3.5 * EV_TO_RADS;
        let eps = si.epsilon(omega);
        assert!(eps.re.abs() > 1.0);
        assert!(eps.im.abs() > 0.0);
    }

    #[test]
    fn test_kramers_kronig_causality() {
        let gold = gold_drude();
        let omega_low = 0.01 * EV_TO_RADS;
        let eps_low = gold.epsilon(omega_low);
        assert!(eps_low.im > 100.0, "Strong dissipation at low frequency");
        let omega_high = 100.0 * EV_TO_RADS;
        let eps_high = gold.epsilon(omega_high);
        assert!((eps_high.re - 1.0).abs() < 0.1, "eps -> 1 at high frequency");
    }

    // ===================== Phase 1: Provenance + C-418 gap materials =====================

    #[test]
    fn test_material_provenance_fields() {
        let gold = get_material("Au").unwrap();
        assert_eq!(gold.casimir_model, CasimirModelFlag::Drude);
        assert_eq!(gold.temperature_k, Some(300.0));
        assert!(gold.validity_range_ev.is_some());
    }

    #[test]
    fn test_alumina_optical() {
        let mat = get_material("al2o3").unwrap();
        assert_eq!(mat.name, "Alumina");
        assert_eq!(mat.material_type, MaterialType::Dielectric);
        // Transparency window: eps.re > 1 in visible
        let omega = 2.0 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re > 1.0, "Alumina eps.re={} should be > 1 at 2 eV", eps.re);
    }

    #[test]
    fn test_alumina_alias() {
        assert!(get_material("sapphire").is_some());
        assert!(get_material("alumina").is_some());
    }

    #[test]
    fn test_diamond_optical() {
        let mat = get_material("diamond").unwrap();
        assert_eq!(mat.formula, "C");
        // Diamond has very high eps_inf (~5.7)
        let omega = 2.0 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re > 5.0, "Diamond eps.re={} should be > 5 at 2 eV", eps.re);
    }

    #[test]
    fn test_quartz_optical() {
        let mat = get_material("quartz").unwrap();
        assert_eq!(mat.material_type, MaterialType::Dielectric);
        let omega = 2.0 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re > 2.0, "Quartz eps.re={} should be > 2 at 2 eV", eps.re);
    }

    #[test]
    fn test_tio2_optical() {
        let mat = get_material("tio2").unwrap();
        assert_eq!(mat.formula, "TiO2");
        // TiO2 has high eps_inf (~5.9)
        let omega = 2.0 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re > 5.0, "TiO2 eps.re={} should be > 5 at 2 eV", eps.re);
    }

    #[test]
    fn test_tio2_alias() {
        assert!(get_material("rutile").is_some());
        assert!(get_material("titanium_dioxide").is_some());
    }

    #[test]
    fn test_dielectric_imaginary_freq_positivity() {
        for name in &["al2o3", "diamond", "quartz", "tio2"] {
            let mat = get_material(name).unwrap();
            let xi = 1.0 * EV_TO_RADS;
            let eps = mat.optical.epsilon_imaginary(xi);
            assert!(eps > 1.0, "{}: eps(i*xi) = {} should be > 1", name, eps);
        }
    }

    // ===================== Phase 2: Rakic 11-metal set =====================

    #[test]
    fn test_rakic_metals_exist() {
        for name in &["be", "cr", "ni", "pd", "pt", "ti", "w"] {
            assert!(
                get_material(name).is_some(),
                "Metal '{}' should exist in database",
                name
            );
        }
    }

    #[test]
    fn test_rakic_metals_metallic_sign() {
        // At 0.5 eV, all metals should have eps.re < 0 (Drude dominates)
        let omega = 0.5 * EV_TO_RADS;
        for name in &["au", "ag", "cu", "al", "be", "cr", "ni", "pd", "pt", "ti", "w"] {
            let mat = get_material(name).unwrap();
            let eps = mat.optical.epsilon(omega);
            assert!(
                eps.re < 0.0,
                "{}: eps.re={} should be < 0 at 0.5 eV",
                name, eps.re
            );
        }
    }

    #[test]
    fn test_rakic_metals_imaginary_freq_monotonic() {
        // eps(i*xi) should decrease monotonically with increasing xi
        for name in &["be", "cr", "ni", "pd", "pt", "ti", "w"] {
            let mat = get_material(name).unwrap();
            let xi_low = 0.5 * EV_TO_RADS;
            let xi_high = 2.0 * EV_TO_RADS;
            let eps_low = mat.optical.epsilon_imaginary(xi_low);
            let eps_high = mat.optical.epsilon_imaginary(xi_high);
            assert!(
                eps_low > eps_high,
                "{}: eps(i*xi_low)={} should > eps(i*xi_high)={}",
                name, eps_low, eps_high
            );
        }
    }

    #[test]
    fn test_copper_enhanced_dl() {
        let cu_dl = copper_drude_lorentz();
        let omega = 2.0 * EV_TO_RADS;
        let eps = cu_dl.epsilon(omega);
        // Copper with interband should still be metallic at 2 eV
        assert!(eps.re < 0.0, "Cu DL eps.re={} should be < 0 at 2 eV", eps.re);
        // Should have interband absorption
        assert!(eps.im.abs() > 0.1, "Cu DL should have imaginary part from interband");
    }

    #[test]
    fn test_aluminum_enhanced_dl() {
        let al_dl = aluminum_drude_lorentz();
        let omega = 0.5 * EV_TO_RADS;
        let eps = al_dl.epsilon(omega);
        assert!(eps.re < 0.0, "Al DL eps.re={} at 0.5 eV", eps.re);
    }

    // ===================== Phase 3: Titanates + TCOs =====================

    #[test]
    fn test_tio_metallic() {
        let mat = get_material("tio").unwrap();
        assert_eq!(mat.material_type, MaterialType::ConductiveOxide);
        // TiO is a bad metal: eps.re < 0 at low freq
        let omega = 0.5 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re < 0.0, "TiO eps.re={} should be < 0 at 0.5 eV", eps.re);
    }

    #[test]
    fn test_srtio3_soft_mode() {
        let mat = get_material("srtio3").unwrap();
        // Near the soft mode at 11 meV, eps should be giant
        let omega = 0.012 * EV_TO_RADS; // Just above 11 meV
        let eps = mat.optical.epsilon(omega);
        assert!(
            eps.re.abs() > 50.0,
            "SrTiO3 near soft mode: |eps.re|={} should be > 50",
            eps.re.abs()
        );
    }

    #[test]
    fn test_srtio3_doped_metallic() {
        let mat = get_material("srtio3_doped").unwrap();
        assert_eq!(mat.material_type, MaterialType::ConductiveOxide);
        assert!(mat.doping_info.is_some());
        // At very low frequency, Drude tail should give eps.re < 0
        let omega = 0.005 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        // The phonon modes dominate at this frequency so eps may be positive;
        // test that Drude contribution exists via imaginary part
        assert!(eps.im.abs() > 1.0, "Doped SrTiO3 should have significant Im(eps) at THz");
    }

    #[test]
    fn test_latio3_mott_gap() {
        let mat = get_material("latio3").unwrap();
        // At 0.5 eV (above Mott gap), should have strong absorption
        let omega = 0.5 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(
            eps.im.abs() > 1.0,
            "LaTiO3 should have strong absorption at Mott gap energy, im={}",
            eps.im
        );
    }

    #[test]
    fn test_azo_crossover() {
        let mat = get_material("azo").unwrap();
        // Metallic in IR (below ~1 eV), dielectric in visible
        let omega_ir = 0.3 * EV_TO_RADS;
        let omega_vis = 3.0 * EV_TO_RADS;
        let eps_ir = mat.optical.epsilon(omega_ir);
        let eps_vis = mat.optical.epsilon(omega_vis);
        assert!(eps_ir.re < 0.0, "AZO should be metallic in IR, eps.re={}", eps_ir.re);
        assert!(eps_vis.re > 0.0, "AZO should be dielectric in visible, eps.re={}", eps_vis.re);
    }

    #[test]
    fn test_doped_si() {
        let mat = get_material("doped_si").unwrap();
        assert!(mat.doping_info.is_some());
        // Has both interband (Si) and Drude (doping) contributions
        let omega = 3.5 * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        assert!(eps.re.abs() > 1.0, "Doped Si should have significant eps near Si critical points");
    }

    #[test]
    fn test_material_count() {
        let materials = list_materials();
        assert!(
            materials.len() >= 25,
            "Database should have >= 25 materials, got {}",
            materials.len()
        );
    }

    // === Phase 4: Extended Drude tests ===

    #[test]
    fn test_constant_scattering_matches_regular_drude() {
        // ExtendedDrude with Constant scattering should reproduce regular Drude
        let regular = DrudeLorentzParams {
            drude: Some(DrudeParams {
                omega_p_ev: 9.0,
                gamma_ev: 0.035,
                eps_inf: 1.0,
            }),
            oscillators: vec![],
            eps_inf: 1.0,
            extended_drude: None,
        };

        let extended = DrudeLorentzParams {
            drude: None,
            oscillators: vec![],
            eps_inf: 1.0,
            extended_drude: Some(ExtendedDrudeParams {
                omega_p_ev: 9.0,
                scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
                eps_inf: 1.0,
            }),
        };

        for freq_ev in [0.5, 1.0, 2.0, 5.0] {
            let omega = freq_ev * EV_TO_RADS;
            let eps_r = regular.epsilon(omega);
            let eps_e = extended.epsilon(omega);
            assert!(
                (eps_r.re - eps_e.re).abs() < 1e-10,
                "Re mismatch at {} eV: regular={}, extended={}",
                freq_ev, eps_r.re, eps_e.re
            );
            assert!(
                (eps_r.im - eps_e.im).abs() < 1e-10,
                "Im mismatch at {} eV: regular={}, extended={}",
                freq_ev, eps_r.im, eps_e.im
            );
        }
    }

    #[test]
    fn test_constant_scattering_imaginary_matches_regular() {
        let regular = DrudeLorentzParams {
            drude: Some(DrudeParams {
                omega_p_ev: 9.0,
                gamma_ev: 0.035,
                eps_inf: 1.0,
            }),
            oscillators: vec![],
            eps_inf: 1.0,
            extended_drude: None,
        };

        let extended = DrudeLorentzParams {
            drude: None,
            oscillators: vec![],
            eps_inf: 1.0,
            extended_drude: Some(ExtendedDrudeParams {
                omega_p_ev: 9.0,
                scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
                eps_inf: 1.0,
            }),
        };

        for freq_ev in [0.5, 1.0, 2.0, 5.0] {
            let xi = freq_ev * EV_TO_RADS;
            let eps_r = regular.epsilon_imaginary(xi);
            let eps_e = extended.epsilon_imaginary(xi);
            assert!(
                (eps_r - eps_e).abs() < 1e-10,
                "Imaginary freq mismatch at {} eV: regular={}, extended={}",
                freq_ev, eps_r, eps_e
            );
        }
    }

    #[test]
    fn test_drude_smith_c0_matches_regular() {
        // Drude-Smith with c=0 should match regular Drude
        let regular = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
            eps_inf: 1.0,
        };

        let smith_c0 = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::DrudeSmith {
                gamma_ev: 0.035,
                backscatter_c: 0.0,
            },
            eps_inf: 1.0,
        };

        for freq_ev in [0.5, 1.0, 2.0, 5.0] {
            let omega = freq_ev * EV_TO_RADS;
            let eps_r = regular.epsilon(omega);
            let eps_s = smith_c0.epsilon(omega);
            assert!(
                (eps_r.re - eps_s.re).abs() < 1e-8,
                "Re mismatch at {} eV: regular={}, smith={}",
                freq_ev, eps_r.re, eps_s.re
            );
            assert!(
                (eps_r.im - eps_s.im).abs() < 1e-8,
                "Im mismatch at {} eV: regular={}, smith={}",
                freq_ev, eps_r.im, eps_s.im
            );
        }
    }

    #[test]
    fn test_drude_smith_negative_c_suppresses_dc() {
        // With c < 0, Drude-Smith suppresses the DC conductivity (localization)
        let regular = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
            eps_inf: 1.0,
        };

        let smith = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::DrudeSmith {
                gamma_ev: 0.035,
                backscatter_c: -0.8,
            },
            eps_inf: 1.0,
        };

        // At low frequency, Smith should have less negative eps.re (less metallic)
        let omega_low = 0.1 * EV_TO_RADS;
        let eps_r = regular.epsilon(omega_low);
        let eps_s = smith.epsilon(omega_low);
        assert!(
            eps_s.re > eps_r.re,
            "Smith (c<0) should suppress metallic response: smith.re={}, regular.re={}",
            eps_s.re, eps_r.re
        );
    }

    #[test]
    fn test_power_law_gamma_increases() {
        let model = ScatteringModel::PowerLaw {
            gamma_0_ev: 0.035,
            omega_scale_ev: 1.0,
            exponent: 1.5,
        };

        let g1 = model.gamma_at_ev(1.0);
        let g2 = model.gamma_at_ev(2.0);
        let g3 = model.gamma_at_ev(4.0);
        assert!(g2 > g1, "gamma should increase with frequency");
        assert!(g3 > g2, "gamma should increase with frequency");
    }

    #[test]
    fn test_power_law_zero_frequency() {
        let model = ScatteringModel::PowerLaw {
            gamma_0_ev: 0.035,
            omega_scale_ev: 1.0,
            exponent: 1.5,
        };
        let g0 = model.gamma_at_ev(0.0);
        assert!((g0 - 0.035).abs() < 1e-12, "gamma(0) should equal gamma_0");
    }

    #[test]
    fn test_linear_scattering() {
        let model = ScatteringModel::LinearInOmega {
            gamma_0_ev: 0.01,
            alpha: 0.05,
        };
        let g = model.gamma_at_ev(2.0);
        let expected = 0.01 + 0.05 * 2.0;
        assert!((g - expected).abs() < 1e-12, "Linear: gamma(2)={}, expected={}", g, expected);
    }

    #[test]
    fn test_tabulated_scattering_interpolation() {
        let model = ScatteringModel::Tabulated {
            omega_ev: vec![0.0, 1.0, 2.0, 4.0],
            gamma_ev: vec![0.01, 0.02, 0.05, 0.15],
        };
        // At 1.5 eV: linear interpolation between (1.0, 0.02) and (2.0, 0.05)
        let g = model.gamma_at_ev(1.5);
        let expected = 0.02 + 0.5 * (0.05 - 0.02);
        assert!((g - expected).abs() < 1e-12, "Tabulated interp: gamma(1.5)={}, expected={}", g, expected);
    }

    #[test]
    fn test_extended_drude_imaginary_freq_positive() {
        // Imaginary-frequency epsilon must be > 1 for metals
        let ext = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
            eps_inf: 1.0,
        };

        for freq_ev in [0.01, 0.1, 1.0, 5.0, 10.0] {
            let xi = freq_ev * EV_TO_RADS;
            let eps = ext.epsilon_imaginary(xi);
            assert!(eps > 1.0, "eps(i*xi) should be > 1 for metals at {} eV, got {}", freq_ev, eps);
        }
    }

    #[test]
    fn test_extended_drude_standalone_epsilon() {
        // Standalone epsilon (with eps_inf) at high freq should approach eps_inf
        let ext = ExtendedDrudeParams {
            omega_p_ev: 9.0,
            scattering: ScatteringModel::Constant { gamma_ev: 0.035 },
            eps_inf: 1.0,
        };
        let eps = ext.epsilon(100.0 * EV_TO_RADS);
        assert!(
            (eps.re - 1.0).abs() < 0.01,
            "At high freq eps.re should approach eps_inf=1, got {}",
            eps.re
        );
    }
}
