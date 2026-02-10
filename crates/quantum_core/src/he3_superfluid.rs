//! Helium-3 p-wave BCS superfluid physics.
//!
//! He-3 is a fermionic system that undergoes p-wave Cooper pairing,
//! producing an anisotropic order parameter with two distinct superfluid
//! phases (A and B) separated by a first-order transition.
//!
//! # Phases
//! - **Normal**: T > T_c, no pairing
//! - **Superfluid A** (ABM state): Axial p-wave, breaks time-reversal symmetry.
//!   Anderson-Brinkman-Morel state. Stable at high P near T_c.
//! - **Superfluid B** (BW state): Isotropic gap, preserves combined
//!   spin-orbit symmetry. Balian-Werthamer state. Stable at low P, T.
//!
//! # Literature
//! - Leggett (1975): A theoretical description of the new phases of liquid He-3
//! - Vollhardt & Wolfle (1990): The Superfluid Phases of Helium 3
//! - Volovik (2003): The Universe in a Helium Droplet

use std::f64::consts::PI;

/// Physical constants for He-3.
pub struct He3Params {
    /// He-3 atomic mass (kg).
    pub mass: f64,
    /// Fermi energy (J).
    pub e_f: f64,
    /// Density of states at Fermi level N(0) (J^-1 m^-3).
    pub n0: f64,
    /// p-wave pairing interaction strength (dimensionless).
    pub v_p: f64,
    /// Critical temperature for A phase at zero pressure (mK).
    pub t_c_a: f64,
    /// Critical temperature for B phase (slightly below A).
    pub t_c_b: f64,
}

impl He3Params {
    /// Standard He-3 parameters at low pressure (~0 bar).
    ///
    /// T_c ~ 0.929 mK at SVP (Greywall 1986).
    pub fn standard() -> Self {
        Self {
            mass: 5.0082e-27,      // 3.016 u in kg
            e_f: 3.28e-23,         // ~2.38 mK in J (Fermi temperature ~38 mK)
            n0: 1.39e47,           // states/J/m^3 (from specific heat)
            v_p: 0.25,             // effective p-wave coupling
            t_c_a: 0.929,          // mK at SVP (A phase)
            t_c_b: 0.929,          // mK at SVP (B phase, same at zero field)
        }
    }

    /// He-3 parameters at elevated pressure (~34 bar, near melting).
    ///
    /// At high pressure, T_c increases and A/B phases separate.
    pub fn high_pressure() -> Self {
        Self {
            mass: 5.0082e-27,
            e_f: 4.5e-23,          // Enhanced at high pressure
            n0: 2.0e47,            // Enhanced DOS
            v_p: 0.30,
            t_c_a: 2.491,          // mK at ~34 bar
            t_c_b: 2.273,          // mK at ~34 bar (B below A)
        }
    }
}

/// Superfluid phase of He-3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum He3Phase {
    Normal,
    SuperfluidA,
    SuperfluidB,
}

/// BCS gap magnitude for p-wave pairing.
///
/// Delta(T) = Delta_0 * sqrt(1 - (T/T_c)^2) (weak-coupling BCS).
/// At T=0: Delta_0 = (pi/e^gamma) * k_B * T_c ~ 1.764 * k_B * T_c.
///
/// For He-3, strong-coupling corrections modify the prefactor slightly,
/// but the temperature dependence remains approximately BCS-like.
pub fn bcs_gap_pwave(temperature_mk: f64, t_c_mk: f64) -> f64 {
    if temperature_mk >= t_c_mk || t_c_mk <= 0.0 {
        return 0.0;
    }
    if temperature_mk <= 0.0 {
        // BCS zero-temperature gap: Delta_0 = 1.764 * k_B * T_c
        // Return in units of k_B * T_c for convenience
        return 1.764;
    }
    let t_ratio = temperature_mk / t_c_mk;
    1.764 * (1.0 - t_ratio * t_ratio).sqrt()
}

/// Anisotropy parameter for the p-wave gap.
///
/// The A-phase gap has nodes along the l-vector:
///   Delta_A(k) = Delta * sin(theta_k) (axial state)
/// The B-phase gap is isotropic:
///   Delta_B(k) = Delta (BW state)
///
/// Returns the gap anisotropy ratio: min(Delta)/max(Delta).
/// A-phase: 0 (has nodes), B-phase: 1 (isotropic).
pub fn gap_anisotropy(phase: He3Phase) -> f64 {
    match phase {
        He3Phase::Normal => 0.0,
        He3Phase::SuperfluidA => 0.0,  // Axial nodes
        He3Phase::SuperfluidB => 1.0,  // Isotropic
    }
}

/// Magnetic susceptibility ratio chi/chi_n for each phase.
///
/// In the normal state: chi_n (Pauli susceptibility).
/// A-phase (equal-spin pairing): chi/chi_n ~ 1 (unchanged from normal).
/// B-phase (spin-singlet-like mixing): chi/chi_n ~ 2/3 at T=0
/// (Leggett 1975, strong-coupling: ~0.5-0.7).
pub fn magnetic_susceptibility_ratio(
    phase: He3Phase,
    temperature_mk: f64,
    t_c_mk: f64,
) -> f64 {
    match phase {
        He3Phase::Normal => 1.0,
        He3Phase::SuperfluidA => {
            // ESP (equal-spin pairing): chi barely changes
            // Weak T-dependence: chi/chi_n ~ 1 - small correction
            if temperature_mk >= t_c_mk {
                return 1.0;
            }
            let t_ratio = temperature_mk / t_c_mk;
            // Small reduction below T_c: ~1 - 0.05*(1-T/Tc)
            1.0 - 0.05 * (1.0 - t_ratio)
        }
        He3Phase::SuperfluidB => {
            // Yosida function Y(T): chi/chi_n = 2/3 * (1 + Y(T)/2)
            // Y(T=0) = 0, Y(T_c) = 1
            if temperature_mk >= t_c_mk {
                return 1.0;
            }
            let t_ratio = temperature_mk / t_c_mk;
            let yosida = t_ratio * t_ratio; // Approximate Yosida function
            (2.0 / 3.0) * (1.0 + yosida / 2.0)
        }
    }
}

/// Determine the equilibrium phase at given temperature and pressure.
///
/// Simplified phase diagram (Greywall 1986):
/// - T > T_c_A: Normal
/// - T_c_B < T < T_c_A and high pressure: Superfluid A
/// - T < T_c_B or low pressure: Superfluid B
///
/// The `pressure_bar` parameter controls the A-B splitting.
/// At P = 0: T_c_A = T_c_B (no A phase region).
/// At P = 34 bar: T_c_A ~ 2.49 mK, T_c_B ~ 2.27 mK.
pub fn equilibrium_phase(
    temperature_mk: f64,
    t_c_a_mk: f64,
    t_c_b_mk: f64,
) -> He3Phase {
    if temperature_mk >= t_c_a_mk {
        He3Phase::Normal
    } else if temperature_mk >= t_c_b_mk {
        He3Phase::SuperfluidA
    } else {
        He3Phase::SuperfluidB
    }
}

/// Compute the He-3 phase diagram as a function of temperature.
///
/// Returns a vector of (temperature_mk, phase) pairs spanning 0 to t_max.
pub fn phase_diagram(
    t_c_a_mk: f64,
    t_c_b_mk: f64,
    t_max_mk: f64,
    n_points: usize,
) -> Vec<(f64, He3Phase)> {
    let dt = t_max_mk / n_points as f64;
    (0..=n_points)
        .map(|i| {
            let t = i as f64 * dt;
            (t, equilibrium_phase(t, t_c_a_mk, t_c_b_mk))
        })
        .collect()
}

/// Superfluid density fraction for He-3 B-phase (isotropic gap).
///
/// rho_s/rho ~ 1 - (2/3) * exp(-Delta/(k_B * T)) at low T.
/// Near T_c: rho_s/rho ~ (1 - T/T_c).
/// Interpolation: rho_s/rho = 1 - (T/T_c)^4 (empirical fit).
pub fn he3_superfluid_density_fraction(temperature_mk: f64, t_c_mk: f64) -> f64 {
    if temperature_mk >= t_c_mk || t_c_mk <= 0.0 {
        return 0.0;
    }
    if temperature_mk <= 0.0 {
        return 1.0;
    }
    let t_ratio = temperature_mk / t_c_mk;
    1.0 - t_ratio.powi(4)
}

/// Specific heat jump at T_c.
///
/// BCS prediction: Delta(C_V)/(gamma * T_c) = 1.43 for s-wave.
/// He-3 (p-wave, strong coupling): experimentally ~1.6-2.0.
pub fn specific_heat_jump_ratio() -> f64 {
    1.764 // p-wave BCS value (slightly above s-wave 1.43)
}

/// Second sound velocity in superfluid He-3 B-phase.
///
/// c_2 ~ c_1 / sqrt(3) * sqrt(rho_s/rho_n) at low temperatures.
/// First sound in He-3: c_1 ~ 183 m/s at SVP.
pub fn he3_second_sound(temperature_mk: f64, t_c_mk: f64) -> f64 {
    let c1 = 183.0; // m/s, first sound at SVP
    let rho_s = he3_superfluid_density_fraction(temperature_mk, t_c_mk);
    if rho_s <= 0.0 || rho_s >= 1.0 {
        return 0.0;
    }
    let rho_n = 1.0 - rho_s;
    c1 / 3.0_f64.sqrt() * (rho_s / rho_n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bcs_gap_above_tc_zero() {
        // Above T_c, gap is zero
        let gap = bcs_gap_pwave(1.5, 0.929);
        assert!(
            gap.abs() < 1e-10,
            "gap above T_c should be 0, got {}",
            gap
        );
    }

    #[test]
    fn test_bcs_gap_t_zero_max() {
        // At T=0, gap = 1.764 * k_B * T_c
        let gap = bcs_gap_pwave(0.0, 0.929);
        assert!(
            (gap - 1.764).abs() < 1e-10,
            "gap at T=0 should be 1.764, got {}",
            gap
        );
    }

    #[test]
    fn test_bcs_gap_decreases_with_temperature() {
        let gap_low = bcs_gap_pwave(0.2, 0.929);
        let gap_mid = bcs_gap_pwave(0.5, 0.929);
        let gap_high = bcs_gap_pwave(0.8, 0.929);
        assert!(
            gap_low > gap_mid && gap_mid > gap_high,
            "gap should decrease with T: {:.4} > {:.4} > {:.4}",
            gap_low, gap_mid, gap_high
        );
    }

    #[test]
    fn test_pwave_anisotropy() {
        // A-phase is anisotropic (nodes), B-phase is isotropic
        assert!(
            gap_anisotropy(He3Phase::SuperfluidA).abs() < 1e-10,
            "A-phase should have zero anisotropy ratio (nodes)"
        );
        assert!(
            (gap_anisotropy(He3Phase::SuperfluidB) - 1.0).abs() < 1e-10,
            "B-phase should have unity anisotropy (isotropic)"
        );
    }

    #[test]
    fn test_phase_a_chi_near_unity() {
        // A-phase: equal-spin pairing preserves susceptibility ~1
        let chi = magnetic_susceptibility_ratio(He3Phase::SuperfluidA, 0.5, 0.929);
        eprintln!("A-phase chi/chi_n at 0.5 mK = {:.4}", chi);
        assert!(
            chi > 0.9 && chi <= 1.0,
            "A-phase chi/chi_n should be near 1.0, got {}",
            chi
        );
    }

    #[test]
    fn test_phase_b_chi_reduced() {
        // B-phase: reduced susceptibility, chi/chi_n ~ 2/3 at T=0
        let chi_zero = magnetic_susceptibility_ratio(He3Phase::SuperfluidB, 0.001, 0.929);
        let chi_tc = magnetic_susceptibility_ratio(He3Phase::SuperfluidB, 0.929, 0.929);
        eprintln!(
            "B-phase chi/chi_n: T~0 = {:.4}, T=T_c = {:.4}",
            chi_zero, chi_tc
        );
        // At T~0: chi/chi_n ~ 2/3
        assert!(
            (chi_zero - 2.0 / 3.0).abs() < 0.05,
            "B-phase chi(T=0)/chi_n should be ~2/3, got {}",
            chi_zero
        );
        // At T=T_c: chi/chi_n = 1
        assert!(
            (chi_tc - 1.0).abs() < 1e-10,
            "B-phase chi(T_c)/chi_n should be 1, got {}",
            chi_tc
        );
    }

    #[test]
    fn test_phase_diagram_topology() {
        // At high pressure: Normal -> A -> B as T decreases
        let params = He3Params::high_pressure();
        let diagram = phase_diagram(params.t_c_a, params.t_c_b, 3.0, 300);

        // Find phase transitions
        let mut found_normal = false;
        let mut found_a = false;
        let mut found_b = false;
        for &(_, phase) in &diagram {
            match phase {
                He3Phase::Normal => found_normal = true,
                He3Phase::SuperfluidA => found_a = true,
                He3Phase::SuperfluidB => found_b = true,
            }
        }

        assert!(found_normal, "should have normal phase");
        assert!(found_a, "should have A phase at high pressure");
        assert!(found_b, "should have B phase");

        // At high T: Normal
        assert_eq!(diagram.last().unwrap().1, He3Phase::Normal);
        // At T=0: B phase
        assert_eq!(diagram[0].1, He3Phase::SuperfluidB);
    }

    #[test]
    fn test_phase_diagram_low_pressure() {
        // At SVP: T_c_A = T_c_B, no A-phase window
        let params = He3Params::standard();
        let diagram = phase_diagram(params.t_c_a, params.t_c_b, 2.0, 200);

        // Should go directly Normal -> B (no A window)
        let a_count = diagram
            .iter()
            .filter(|&&(_, p)| p == He3Phase::SuperfluidA)
            .count();
        assert_eq!(
            a_count, 0,
            "at SVP, A and B have same T_c, no A-phase window"
        );
    }

    #[test]
    fn test_he3_he4_tc_order() {
        // He-3 T_c (~1 mK) << He-4 T_lambda (~2177 mK)
        let he3_tc = 0.929; // mK
        let he4_t_lambda = 2176.8; // mK (= 2.1768 K)
        assert!(
            he3_tc < he4_t_lambda / 1000.0,
            "He-3 T_c ({} mK) should be << He-4 T_lambda ({} mK)",
            he3_tc, he4_t_lambda
        );
    }

    #[test]
    fn test_superfluid_density_he3() {
        let t_c = 0.929;
        // At T=0, fully superfluid
        let rho_s_0 = he3_superfluid_density_fraction(0.0, t_c);
        assert!(
            (rho_s_0 - 1.0).abs() < 1e-10,
            "rho_s(0) = {}, expected 1",
            rho_s_0
        );

        // At T_c, no superfluid
        let rho_s_tc = he3_superfluid_density_fraction(t_c, t_c);
        assert!(
            rho_s_tc.abs() < 1e-10,
            "rho_s(Tc) = {}, expected 0",
            rho_s_tc
        );

        // Monotonically decreasing
        let rho_s_low = he3_superfluid_density_fraction(0.2, t_c);
        let rho_s_mid = he3_superfluid_density_fraction(0.5, t_c);
        assert!(
            rho_s_low > rho_s_mid,
            "rho_s should decrease: {:.4} > {:.4}",
            rho_s_low, rho_s_mid
        );
    }
}
