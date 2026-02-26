//! Physical and mathematical constants for photon-graviton TCMT

/// Reduced Planck constant (hbar = h / 2\pi)
/// Units: eV*s
/// Reference: CODATA 2018
pub const REDUCED_PLANCK: f64 = 6.582119569509676e-16;

/// Planck constant
/// Units: eV*s
pub const PLANCK_CONSTANT: f64 = 2.0 * std::f64::consts::PI * REDUCED_PLANCK;

/// Speed of light in vacuum
/// Units: m/s
pub const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Gravitational constant (Newton's G)
/// Units: m^3 kg^-1 s^-2
pub const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11;

/// Planck mass (m_p = sqrt(hbarc/G))
/// Units: kg
pub const PLANCK_MASS: f64 = 2.176434e-8;

/// Planck length (l_p = sqrt(hbarG/c^3))
/// Units: m
pub const PLANCK_LENGTH: f64 = 1.616255e-35;

/// Planck frequency (f_p = c / l_p = sqrt(c^5 / (hbar G)))
/// Units: Hz
pub const PLANCK_FREQUENCY: f64 = 1.854858e43;

/// Fine structure constant (\alpha = e^2 / (4\pi\epsilon_0hbarc))
/// Dimensionless
pub const FINE_STRUCTURE_CONSTANT: f64 = 7.2973525693e-3;

/// Gravitational coupling constant (relates graviton to Planck mass)
/// \alpha_grav ~= (m_electron / m_planck)^2
/// Dimensionless, extremely small
pub const GRAVITATIONAL_COUPLING_CONSTANT: f64 = 1.76859e-45;

/// Elementary charge
/// Units: Coulombs (in SI), or stored as coupling in natural units
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

/// Electron mass
/// Units: kg
pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

/// Electron mass in natural units (eV/c^2)
pub const ELECTRON_MASS_EV: f64 = 0.51099895e6;

/// Weak-field approximation validity threshold
/// TCMT approximation valid when coupling_strength / (decay_rate * resonance_freq) << 1
/// Typical threshold: 0.1 to 0.01
pub const WEAK_FIELD_THRESHOLD: f64 = 0.1;

/// Numerical precision tolerance for optical theorem checks
/// Units: dimensionless (relative error)
pub const OPTICAL_THEOREM_TOLERANCE: f64 = 1e-10;

/// Numerical precision tolerance for energy conservation checks
/// Units: relative error
pub const ENERGY_CONSERVATION_TOLERANCE: f64 = 1e-10;

/// Convergence tolerance for iterative solvers
/// Units: relative change in amplitude per iteration
pub const SOLVER_CONVERGENCE_TOLERANCE: f64 = 1e-12;

/// Maximum iterations for TCMT ODE solver
pub const MAX_SOLVER_ITERATIONS: usize = 10000;

/// Minimal coupling strength to avoid numerical issues
/// Units: [frequency], anything below this is treated as zero
pub const MIN_COUPLING_STRENGTH: f64 = 1e-20;

/// Minimal decay rate to avoid division by zero
/// Units: [frequency]
pub const MIN_DECAY_RATE: f64 = 1e-20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planck_scales_consistency() {
        // Verify Planck scales are self-consistent
        // l_p * f_p should equal c
        let computed_speed = PLANCK_LENGTH * PLANCK_FREQUENCY;
        assert!((computed_speed - SPEED_OF_LIGHT).abs() / SPEED_OF_LIGHT < 1e-6);
    }

    #[test]
    fn gravitational_constant_scaling() {
        // Simplified check: Planck mass should be positive and related to fundamental constants
        const { assert!(PLANCK_MASS > 0.0) };
        const { assert!(GRAVITATIONAL_CONSTANT > 0.0) };
        // Planck mass is roughly sqrt(hbar*c/G), dimensionally correct
        let dim_check = REDUCED_PLANCK * SPEED_OF_LIGHT / GRAVITATIONAL_CONSTANT;
        assert!(dim_check > 0.0);
    }

    #[test]
    fn coupling_constants_positive() {
        const { assert!(REDUCED_PLANCK > 0.0) };
        const { assert!(SPEED_OF_LIGHT > 0.0) };
        const { assert!(GRAVITATIONAL_CONSTANT > 0.0) };
        const { assert!(FINE_STRUCTURE_CONSTANT > 0.0) };
        const { assert!(FINE_STRUCTURE_CONSTANT < 1.0) };
    }
}
