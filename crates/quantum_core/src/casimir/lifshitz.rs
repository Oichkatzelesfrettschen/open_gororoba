// ============================================================================
// Lifshitz Formula with Dielectric Functions
// ============================================================================
//
// The Lifshitz formula computes Casimir forces from the frequency-dependent
// dielectric functions of the materials. Unlike the ideal PFA (perfect conductor
// limit), Lifshitz theory captures:
// - Material-dependent forces
// - Finite temperature effects
// - Frequency-dependent optical response
//
// # Literature
// - Lifshitz, Sov. Phys. JETP 2, 73 (1956) - Original theory
// - Dzyaloshinskii et al., Adv. Phys. 10, 165 (1961) - Extension to materials
// - Klimchitskaya et al., RMP 81, 1827 (2009) - Comprehensive review

use std::{f64::consts::PI, num::NonZeroUsize};

use gauss_quad::GaussLegendre;

use super::{C, HBAR, casimir_force_pfa};

const DIMENSIONLESS_KAPPA_CUTOFF: f64 = 24.0;

/// Numerical domain and Gauss-Legendre orders for the zero-temperature
/// dimensionless polar quadrature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifshitzQuadratureOptions {
    /// Upper limit for q = kappa * gap.
    pub kappa_cutoff: f64,
    /// Gauss-Legendre order along q.
    pub kappa_order: usize,
    /// Gauss-Legendre order along mu = xi / (c * kappa).
    pub angle_order: usize,
}

impl LifshitzQuadratureOptions {
    /// Construct an explicit quadrature configuration.
    pub const fn new(kappa_cutoff: f64, kappa_order: usize, angle_order: usize) -> Self {
        Self {
            kappa_cutoff,
            kappa_order,
            angle_order,
        }
    }
}

impl Default for LifshitzQuadratureOptions {
    fn default() -> Self {
        Self::new(DIMENSIONLESS_KAPPA_CUTOFF, 32, 32)
    }
}

/// Dielectric model for computing optical response at imaginary frequency.
///
/// The dielectric function at imaginary frequency \epsilon(i\xi) is always real and
/// monotonically decreasing from \epsilon(0) to 1 as \xi -> infty. This is a consequence
/// of Kramers-Kronig relations.
#[derive(Debug, Clone)]
pub enum DielectricModel {
    /// Perfect conductor: \epsilon -> infty (reflection coefficient r = 1)
    PerfectConductor,

    /// Drude model for metals: eps(i*xi) = 1 + omega_p^2 / (xi*(xi + gamma))
    /// Parameters: (plasma frequency omega_p in rad/s, damping gamma in rad/s)
    Drude { omega_p: f64, gamma: f64 },

    /// Plasma model (dissipationless Drude): eps(i*xi) = 1 + omega_p^2 / xi^2
    /// The gamma -> 0 limit of Drude; controversial for thermal Casimir effect
    Plasma { omega_p: f64 },

    /// Drude-Lorentz oscillator model (dielectrics with resonance):
    /// eps(i*xi) = 1 + Sum S_j omega_j^2 / (omega_j^2 + xi^2 + gamma_j xi)
    /// Parameters: Vec of (oscillator strength S, resonance omega_j, damping gamma_j)
    DrudeLorentz { oscillators: Vec<(f64, f64, f64)> },

    /// Tabulated dielectric data (interpolated)
    /// Parameters: (frequencies \xi in rad/s, dielectric values \epsilon(i\xi))
    Tabulated { xi: Vec<f64>, eps: Vec<f64> },
}

impl DielectricModel {
    /// Create a Drude model for gold at room temperature.
    ///
    /// Parameters from optical data:
    /// - \omega_p = 9.0 eV = 1.37e16 rad/s
    /// - \gamma = 35 meV = 5.3e13 rad/s
    pub fn gold() -> Self {
        DielectricModel::Drude {
            omega_p: 1.37e16,
            gamma: 5.3e13,
        }
    }

    /// Create a Drude model for aluminum.
    ///
    /// Parameters from optical data:
    /// - \omega_p = 12.5 eV = 1.9e16 rad/s
    /// - \gamma = 126 meV = 1.9e14 rad/s
    pub fn aluminum() -> Self {
        DielectricModel::Drude {
            omega_p: 1.9e16,
            gamma: 1.9e14,
        }
    }

    /// Create a plasma model for gold (dissipationless limit).
    pub fn gold_plasma() -> Self {
        DielectricModel::Plasma { omega_p: 1.37e16 }
    }

    /// Create a simple dielectric model for silica (SiO2).
    ///
    /// Simplified single-oscillator model:
    /// - UV resonance at ~10 eV
    /// - Static eps ~ 3.8
    pub fn silica() -> Self {
        // Single oscillator: S*w^2 / (w^2 + xi^2) where S*(w/w)^2 gives eps(0)-1
        // For eps(0) ~ 3.8, S ~ 2.8 with w ~ 1.5e16 rad/s (10 eV)
        DielectricModel::DrudeLorentz {
            oscillators: vec![(2.8, 1.5e16, 1e15)],
        }
    }

    /// Evaluate dielectric function at imaginary frequency \epsilon(i\xi).
    ///
    /// Returns the real, positive dielectric response at the imaginary
    /// frequency i\xi. The result is always >= 1 for passive materials.
    ///
    /// # Arguments
    /// * `xi` - Imaginary frequency (rad/s), must be >= 0
    pub fn epsilon_at_imaginary(&self, xi: f64) -> f64 {
        match self {
            // Perfect conductor approximated by very large but finite \epsilon
            // This avoids NaN in numerical integration while capturing
            // the essential physics (r -> 1)
            DielectricModel::PerfectConductor => 1e20,

            DielectricModel::Drude { omega_p, gamma } => {
                if xi == 0.0 {
                    f64::INFINITY // DC conductivity diverges
                } else {
                    1.0 + omega_p * omega_p / (xi * (xi + gamma))
                }
            }

            DielectricModel::Plasma { omega_p } => {
                if xi == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 + omega_p * omega_p / (xi * xi)
                }
            }

            DielectricModel::DrudeLorentz { oscillators } => {
                let mut eps = 1.0;
                for &(s, omega_j, gamma_j) in oscillators {
                    eps += s * omega_j * omega_j / (omega_j * omega_j + xi * xi + gamma_j * xi);
                }
                eps
            }

            DielectricModel::Tabulated { xi: xi_tab, eps } => {
                // Linear interpolation in log-log space
                if xi <= 0.0 || xi_tab.is_empty() {
                    return 1.0;
                }
                if xi <= xi_tab[0] {
                    return eps[0];
                }
                if xi >= *xi_tab.last().unwrap() {
                    return 1.0; // High-frequency limit
                }

                // Binary search for interval
                let idx = xi_tab.partition_point(|&x| x < xi);
                if idx == 0 || idx >= xi_tab.len() {
                    return 1.0;
                }

                // Log-log interpolation
                let x0 = xi_tab[idx - 1].ln();
                let x1 = xi_tab[idx].ln();
                let y0 = (eps[idx - 1] - 1.0).max(1e-10).ln();
                let y1 = (eps[idx] - 1.0).max(1e-10).ln();
                let t = (xi.ln() - x0) / (x1 - x0);
                let eps_interp = 1.0 + (y0 + t * (y1 - y0)).exp();
                eps_interp.max(1.0)
            }
        }
    }
}

/// Fresnel reflection coefficient at imaginary frequency (TM/p-polarization).
///
/// r_TM = (eps2 kappa1 - eps1 kappa2) / (eps2 kappa1 + eps1 kappa2)
///
/// where kappa_i = sqrt(eps_i xi^2/c^2 + k_perp^2)
///
/// # Arguments
/// * `eps1` - Dielectric of medium 1 at imaginary frequency
/// * `eps2` - Dielectric of medium 2 at imaginary frequency
/// * `xi` - Imaginary frequency (rad/s)
/// * `k_perp` - Transverse wavenumber (1/m)
pub fn fresnel_tm_imaginary(eps1: f64, eps2: f64, xi: f64, k_perp: f64) -> f64 {
    let xi_c = xi / C;
    let kappa1 = (eps1 * xi_c * xi_c + k_perp * k_perp).sqrt();
    let kappa2 = (eps2 * xi_c * xi_c + k_perp * k_perp).sqrt();

    let numer = eps2 * kappa1 - eps1 * kappa2;
    let denom = eps1 * kappa2 + eps2 * kappa1;

    if denom.abs() < 1e-30 {
        return 0.0;
    }

    numer / denom
}

/// Fresnel reflection coefficient at imaginary frequency (TE/s-polarization).
///
/// r_TE = (\kappa_1 - \kappa_2) / (\kappa_1 + \kappa_2)
///
/// For non-magnetic materials (\mu = 1).
///
/// # Arguments
/// * `eps1` - Dielectric of medium 1 at imaginary frequency
/// * `eps2` - Dielectric of medium 2 at imaginary frequency
/// * `xi` - Imaginary frequency (rad/s)
/// * `k_perp` - Transverse wavenumber (1/m)
pub fn fresnel_te_imaginary(eps1: f64, eps2: f64, xi: f64, k_perp: f64) -> f64 {
    let xi_c = xi / C;
    let kappa1 = (eps1 * xi_c * xi_c + k_perp * k_perp).sqrt();
    let kappa2 = (eps2 * xi_c * xi_c + k_perp * k_perp).sqrt();

    (kappa1 - kappa2) / (kappa1 + kappa2)
}

fn vacuum_surface_reflections(model: &DielectricModel, xi: f64, k_parallel: f64) -> (f64, f64) {
    if matches!(model, DielectricModel::PerfectConductor) {
        (1.0, -1.0)
    } else {
        let permittivity = model.epsilon_at_imaginary(xi);
        (
            fresnel_tm_imaginary(1.0, permittivity, xi, k_parallel),
            fresnel_te_imaginary(1.0, permittivity, xi, k_parallel),
        )
    }
}

fn lifshitz_dimensionless_integrals(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    options: LifshitzQuadratureOptions,
) -> (f64, f64) {
    assert!(
        gap.is_finite() && gap > 0.0,
        "gap must be finite and positive"
    );
    assert!(
        options.kappa_cutoff.is_finite() && options.kappa_cutoff > 0.0,
        "kappa cutoff must be finite and positive"
    );
    let kappa_quadrature = GaussLegendre::new(NonZeroUsize::new(options.kappa_order).expect(
        "kappa quadrature order must be non-zero",
    ));
    let angle_quadrature = GaussLegendre::new(
        NonZeroUsize::new(options.angle_order).expect("angle quadrature order must be non-zero"),
    );

    let pressure_integral = kappa_quadrature.integrate(0.0, options.kappa_cutoff, |q| {
        angle_quadrature.integrate(0.0, 1.0, |mu| {
            let xi = C * q * mu / gap;
            let k_parallel = q * (1.0 - mu * mu).sqrt() / gap;
            let (tm_1, te_1) = vacuum_surface_reflections(eps1, xi, k_parallel);
            let (tm_2, te_2) = vacuum_surface_reflections(eps2, xi, k_parallel);
            let tm_product = tm_1 * tm_2;
            let te_product = te_1 * te_2;
            let attenuation = (-2.0 * q).exp();
            let tm_round_trip = tm_product * attenuation;
            let te_round_trip = te_product * attenuation;
            q.powi(3)
                * (tm_round_trip / (1.0 - tm_round_trip) + te_round_trip / (1.0 - te_round_trip))
        })
    });

    let energy_integral = kappa_quadrature.integrate(0.0, options.kappa_cutoff, |q| {
        angle_quadrature.integrate(0.0, 1.0, |mu| {
            let xi = C * q * mu / gap;
            let k_parallel = q * (1.0 - mu * mu).sqrt() / gap;
            let (tm_1, te_1) = vacuum_surface_reflections(eps1, xi, k_parallel);
            let (tm_2, te_2) = vacuum_surface_reflections(eps2, xi, k_parallel);
            let tm_product = tm_1 * tm_2;
            let te_product = te_1 * te_2;
            let attenuation = (-2.0 * q).exp();
            q * q * ((-tm_product * attenuation).ln_1p() + (-te_product * attenuation).ln_1p())
        })
    });

    (pressure_integral, energy_integral)
}

/// Lifshitz pressure between two parallel plates at zero temperature.
///
/// Integrates over imaginary frequency and transverse momentum to compute
/// the Casimir pressure P = -dE/dA/dd between two semi-infinite slabs.
///
/// P = -(hbar/2pi^2) int_0^inf dxi int_0^inf k_perp dk_perp kappa (r_TM^2 e^{-2 kappa d}/(1-r_TM^2 e^{-2 kappa d}) + TE)
///
/// # Arguments
/// * `gap` - Surface separation (m)
/// * `eps1` - Dielectric model for surface 1
/// * `eps2` - Dielectric model for surface 2
/// * `n_xi` - Number of frequency integration points (default: 32)
/// * `n_k` - Number of momentum integration points (default: 32)
///
/// # Returns
/// Pressure in Pa (negative = attractive)
pub fn lifshitz_pressure_plates(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    n_xi: usize,
    n_k: usize,
) -> f64 {
    lifshitz_pressure_plates_with_options(
        gap,
        eps1,
        eps2,
        LifshitzQuadratureOptions::new(DIMENSIONLESS_KAPPA_CUTOFF, n_xi, n_k),
    )
}

/// Lifshitz pressure with an explicit dimensionless integration domain.
pub fn lifshitz_pressure_plates_with_options(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    options: LifshitzQuadratureOptions,
) -> f64 {
    let (pressure_integral, _) = lifshitz_dimensionless_integrals(gap, eps1, eps2, options);
    -HBAR * C * pressure_integral / (2.0 * PI * PI * gap.powi(4))
}

/// Lifshitz energy per unit area between parallel plates at zero temperature.
///
/// The integration variables are q = kappa*d and
/// mu = xi/(c*kappa). The polar domain covers every direction in the
/// nonnegative (xi/c, k_parallel) quadrant.
pub fn lifshitz_energy_plates(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    kappa_order: usize,
    angle_order: usize,
) -> f64 {
    lifshitz_energy_plates_with_options(
        gap,
        eps1,
        eps2,
        LifshitzQuadratureOptions::new(DIMENSIONLESS_KAPPA_CUTOFF, kappa_order, angle_order),
    )
}

/// Lifshitz energy per unit area with an explicit dimensionless integration domain.
pub fn lifshitz_energy_plates_with_options(
    gap: f64,
    eps1: &DielectricModel,
    eps2: &DielectricModel,
    options: LifshitzQuadratureOptions,
) -> f64 {
    let (_, energy_integral) = lifshitz_dimensionless_integrals(gap, eps1, eps2, options);
    HBAR * C * energy_integral / (4.0 * PI * PI * gap.powi(3))
}

/// Lifshitz force for sphere-plate geometry using PFA.
///
/// Combines the Lifshitz pressure with the Proximity Force Approximation:
/// F_sp = 2\piR integral P(d) d(d) evaluated at the minimum gap.
///
/// For computational efficiency, we use:
/// F_sp ~= 2\piR * gap * P(gap) * correction_factor
///
/// where the correction factor accounts for the integration over the sphere surface.
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Minimum surface-to-surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
/// * `n_xi` - Number of frequency integration points
/// * `n_k` - Number of momentum integration points
///
/// # Returns
/// Force in Newtons (negative = attractive)
pub fn lifshitz_force_sphere_plate(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
    n_xi: usize,
    n_k: usize,
) -> f64 {
    let energy = lifshitz_energy_plates(gap, eps_sphere, eps_plate, n_xi, n_k);
    2.0 * PI * radius * energy
}

/// Lifshitz force ratio: compares material-dependent force to perfect conductor.
///
/// \eta = F_Lifshitz / F_ideal_PFA
///
/// This ratio is always <= 1 for real materials and approaches 1 for
/// perfect conductors or at very short distances (high-frequency limit).
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
///
/// # Returns
/// Ratio \eta (dimensionless, 0 < \eta <= 1)
pub fn lifshitz_force_ratio(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
) -> f64 {
    let f_lifshitz = lifshitz_force_sphere_plate(radius, gap, eps_sphere, eps_plate, 32, 32);
    let f_ideal = casimir_force_pfa(radius, gap);

    if f_ideal.abs() < 1e-30 {
        return 1.0;
    }

    (f_lifshitz / f_ideal).clamp(0.0, 2.0)
}

/// Result of Lifshitz theory calculation.
#[derive(Debug, Clone)]
pub struct LifshitzResult {
    /// Force in Newtons (negative = attractive)
    pub force: f64,
    /// Pressure at the gap (Pa)
    pub pressure: f64,
    /// Ratio to ideal PFA force
    pub eta: f64,
    /// Gap used (m)
    pub gap: f64,
    /// Sphere radius (m)
    pub radius: f64,
}

/// Complete Lifshitz calculation for sphere-plate geometry.
///
/// Returns force, pressure, and ratio to ideal conductor.
///
/// # Arguments
/// * `radius` - Sphere radius (m)
/// * `gap` - Surface separation (m)
/// * `eps_sphere` - Dielectric model for sphere
/// * `eps_plate` - Dielectric model for plate
pub fn lifshitz_sphere_plate(
    radius: f64,
    gap: f64,
    eps_sphere: &DielectricModel,
    eps_plate: &DielectricModel,
) -> LifshitzResult {
    let pressure = lifshitz_pressure_plates(gap, eps_sphere, eps_plate, 32, 32);
    let energy = lifshitz_energy_plates(gap, eps_sphere, eps_plate, 32, 32);
    let force = 2.0 * PI * radius * energy;
    let f_ideal = casimir_force_pfa(radius, gap);
    let eta = if f_ideal.abs() < 1e-30 {
        1.0
    } else {
        (force / f_ideal).clamp(0.0, 2.0)
    };

    LifshitzResult {
        force,
        pressure,
        eta,
        gap,
        radius,
    }
}

/// Matsubara frequency for finite-temperature Lifshitz theory.
///
/// \xi_n = 2\pi n k_B T / hbar
///
/// At room temperature (T = 300 K), \xi_1 ~= 2.4e14 rad/s.
///
/// # Arguments
/// * `n` - Matsubara index (n = 0, 1, 2, ...)
/// * `temperature` - Temperature in Kelvin
pub fn matsubara_frequency(n: usize, temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    2.0 * PI * (n as f64) * K_B * temperature / HBAR
}

/// Characteristic thermal wavelength.
///
/// \lambda_T = hbarc / (k_B T)
///
/// At room temperature: \lambda_T ~= 7.6 \mum
///
/// # Arguments
/// * `temperature` - Temperature in Kelvin
pub fn thermal_wavelength(temperature: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    HBAR * C / (K_B * temperature)
}
