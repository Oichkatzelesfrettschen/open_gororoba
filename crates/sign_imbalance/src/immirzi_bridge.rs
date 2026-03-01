//! Bridge between Cayley-Dickson imbalance density and the Barbero-Immirzi parameter.
//!
//! This module explores whether the 3/8 imbalance attractor in the signed-graph
//! vacuum has a natural mapping to the Immirzi parameter gamma from Loop Quantum
//! Gravity. The Immirzi parameter is the single free parameter in LQG that fixes
//! the area spectrum: A(j) = 8*pi*gamma*l_P^2 * sqrt(j(j+1)).
//!
//! # Physical motivation
//!
//! The imbalance density phi of a Cayley-Dickson signed graph measures the fraction
//! of imbalanced cycles (those whose product of edge signs is negative). For the
//! sedenion multiplication table, phi -> 3/8 = 0.375 (the imbalance attractor).
//!
//! The Immirzi parameter gamma encodes the minimum quantum of area. Two canonical
//! values exist:
//! - gamma_BG = ln(2)/(pi*sqrt(3)) = 0.2375 (Boltzmann-Gibbs, Meissner 2004)
//! - gamma_NZJ = ln(3)/(pi*sqrt(8)) = 0.1236 (Domagala-Lewandowski 2004)
//!
//! # Mapping candidates
//!
//! We test several physically motivated mappings phi -> gamma:
//!
//! 1. **Entropy bridge**: gamma = ln(2*phi + 1) / (pi * sqrt(3))
//!    At phi=3/8: gamma = ln(1.75)/(pi*sqrt(3)) = 0.1027 (too small)
//!
//! 2. **Linear bridge**: gamma = alpha * phi + beta, calibrated to pass through
//!    (phi=0, gamma=0) and (phi=1, gamma_max). Not physically motivated but
//!    useful as a null model.
//!
//! 3. **Imbalance-entropy bridge**: gamma = H(phi)/(pi*sqrt(3)), where
//!    H(phi) is the binary entropy. Max at phi=1/2 giving ln(2)/(pi*sqrt(3))=0.1274.
//!    At the imbalance attractor phi=3/8, gives 0.1216, within 1.7% of gamma_NZJ.
//!
//! # Key result
//!
//! The imbalance-entropy bridge maps phi=3/8 to within 1.7% of gamma_NZJ,
//! suggesting a connection between CD imbalance and the NZJ Immirzi parameter.
//! No natural mapping to gamma_BG exists: the bridge's maximum (0.1274 at phi=0.5)
//! is far below gamma_BG (0.2375).
//!
//! # References
//! - Barbero (1995): Phys. Rev. D 51, 5507
//! - Immirzi (1997): Class. Quantum Grav. 14, L177
//! - Meissner (2004): Class. Quantum Grav. 21, 5245
//! - Domagala & Lewandowski (2004): Class. Quantum Grav. 21, 5233
//! - Rovelli & Smolin (1995): Nucl. Phys. B 442, 593

use std::f64::consts::PI;

/// Vacuum imbalance attractor (3/8).
pub const VACUUM_PHI: f64 = 3.0 / 8.0;

/// Boltzmann-Gibbs Immirzi parameter.
pub const GAMMA_BG: f64 = 0.237_532_956_559_5;

/// Domagala-Lewandowski Immirzi parameter.
pub const GAMMA_NZJ: f64 = 0.123_556_531_041_5;

/// Result of mapping imbalance density to Immirzi parameter.
#[derive(Debug, Clone, Copy)]
pub struct ImmirziMappingResult {
    /// Input imbalance density phi.
    pub phi: f64,
    /// Mapped Immirzi parameter gamma_eff.
    pub gamma_eff: f64,
    /// Fractional deviation from gamma_BG: |gamma_eff - gamma_BG| / gamma_BG.
    pub deviation_bg: f64,
    /// Fractional deviation from gamma_NZJ: |gamma_eff - gamma_NZJ| / gamma_NZJ.
    pub deviation_nzj: f64,
    /// Name of the mapping used.
    pub mapping: &'static str,
}

/// Entropy bridge mapping: gamma = ln(2*phi + 1) / (pi * sqrt(3)).
///
/// Motivated by the Boltzmann-Gibbs entropy counting where ln(2) appears
/// from the j=1/2 spin degeneracy. Replacing 2 with 2*phi+1 generalizes
/// the degeneracy factor to depend on imbalance.
pub fn entropy_bridge(phi: f64) -> f64 {
    (2.0 * phi + 1.0).ln() / (PI * 3.0_f64.sqrt())
}

/// Linear bridge mapping: gamma = gamma_BG * phi / VACUUM_PHI.
///
/// Calibrated so that phi = VACUUM_PHI maps to gamma_BG. This is a null
/// model with no physical justification.
pub fn linear_bridge_bg(phi: f64) -> f64 {
    GAMMA_BG * phi / VACUUM_PHI
}

/// Logarithmic bridge mapping: gamma = gamma_BG * ln(1 + phi) / ln(1 + VACUUM_PHI).
///
/// Interpolates logarithmically: gamma(0) = 0, gamma(VACUUM_PHI) = gamma_BG.
pub fn log_bridge_bg(phi: f64) -> f64 {
    GAMMA_BG * (1.0 + phi).ln() / (1.0 + VACUUM_PHI).ln()
}

/// Imbalance-entropy bridge mapping.
///
/// gamma = [phi * ln(1/phi) + (1-phi) * ln(1/(1-phi))] / (pi * sqrt(3))
///
/// Uses the binary entropy of the imbalance fraction as the degeneracy
/// factor. The denominator pi*sqrt(3) matches the NZJ-family normalization.
/// At phi = 1/2 (maximal imbalance), H = ln(2), giving ln(2)/(pi*sqrt(3)) = 0.1274.
/// At phi = 3/8 (imbalance attractor), H(3/8)/(pi*sqrt(3)) = 0.1216, within 1.7% of gamma_NZJ.
pub fn imbalance_entropy_bridge(phi: f64) -> f64 {
    if phi <= 0.0 || phi >= 1.0 {
        return 0.0;
    }
    let h = -phi * phi.ln() - (1.0 - phi) * (1.0 - phi).ln();
    h / (PI * 3.0_f64.sqrt())
}

/// Power-law bridge: gamma = gamma_BG * (phi / VACUUM_PHI)^alpha.
///
/// The exponent alpha is a free parameter. At alpha = 1, this reduces to
/// the linear bridge. At alpha < 1, it compresses the mapping.
pub fn power_bridge_bg(phi: f64, alpha: f64) -> f64 {
    if phi <= 0.0 {
        return 0.0;
    }
    GAMMA_BG * (phi / VACUUM_PHI).powf(alpha)
}

/// Evaluate all candidate mappings at a given imbalance density.
///
/// Returns a vector of `ImmirziMappingResult` for comparison.
pub fn evaluate_all_mappings(phi: f64) -> Vec<ImmirziMappingResult> {
    let candidates: Vec<(&str, f64)> = vec![
        ("entropy_bridge", entropy_bridge(phi)),
        ("linear_bridge_bg", linear_bridge_bg(phi)),
        ("log_bridge_bg", log_bridge_bg(phi)),
        ("imbalance_entropy", imbalance_entropy_bridge(phi)),
        ("power_bridge(0.5)", power_bridge_bg(phi, 0.5)),
        ("power_bridge(2.0)", power_bridge_bg(phi, 2.0)),
    ];

    candidates
        .into_iter()
        .map(|(name, gamma_eff)| ImmirziMappingResult {
            phi,
            gamma_eff,
            deviation_bg: (gamma_eff - GAMMA_BG).abs() / GAMMA_BG,
            deviation_nzj: (gamma_eff - GAMMA_NZJ).abs() / GAMMA_NZJ,
            mapping: name,
        })
        .collect()
}

/// Find the mapping that best matches gamma_BG at the imbalance attractor.
///
/// Returns the mapping with the smallest fractional deviation from gamma_BG.
pub fn best_bg_match() -> ImmirziMappingResult {
    let results = evaluate_all_mappings(VACUUM_PHI);
    results
        .into_iter()
        .min_by(|a, b| a.deviation_bg.partial_cmp(&b.deviation_bg).unwrap())
        .unwrap()
}

/// Find the mapping that best matches gamma_NZJ at the imbalance attractor.
pub fn best_nzj_match() -> ImmirziMappingResult {
    let results = evaluate_all_mappings(VACUUM_PHI);
    results
        .into_iter()
        .min_by(|a, b| a.deviation_nzj.partial_cmp(&b.deviation_nzj).unwrap())
        .unwrap()
}

/// Maximum of the imbalance-entropy bridge: ln(2)/(pi*sqrt(3)) at phi=0.5.
pub const BRIDGE_MAX: f64 = 0.127_384_023_140_948;

/// Invert the imbalance-entropy bridge on the LEFT branch (0, 0.5).
///
/// Finds phi in (0, 0.5) such that imbalance_entropy_bridge(phi) = target.
/// Returns None if the target exceeds BRIDGE_MAX (no solution exists).
///
/// NOTE: gamma_BG = 0.2375 exceeds BRIDGE_MAX = 0.1274, so it cannot be
/// inverted. Use this for targets <= BRIDGE_MAX only (e.g., gamma_NZJ = 0.1236).
pub fn invert_entropy_bridge(target: f64) -> Option<f64> {
    if target > BRIDGE_MAX + 1e-12 || target <= 0.0 {
        return None;
    }
    // Search on the left branch (0, 0.5) where the function is monotonically increasing
    let mut lo = 1e-6;
    let mut hi = 0.5;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if imbalance_entropy_bridge(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

/// Critical imbalance density for the NZJ value.
///
/// Returns the phi value in (0, 0.5) where imbalance_entropy_bridge(phi) = gamma_NZJ.
/// This should be near the imbalance attractor 3/8 = 0.375.
pub fn invert_entropy_bridge_nzj() -> f64 {
    invert_entropy_bridge(GAMMA_NZJ).expect("gamma_NZJ is within bridge range")
}

/// Attempt to invert the bridge for gamma_BG.
///
/// Returns None because gamma_BG = 0.2375 exceeds the bridge maximum
/// of ln(2)/(pi*sqrt(3)) = 0.1274. This is an honest negative result:
/// no imbalance density maps to gamma_BG via this bridge.
pub fn invert_entropy_bridge_bg() -> Option<f64> {
    invert_entropy_bridge(GAMMA_BG)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_bridge_at_vacuum() {
        // At phi=3/8: ln(1.75)/(pi*sqrt(3)) = 0.5596/(5.4414) = 0.1028
        let gamma = entropy_bridge(VACUUM_PHI);
        assert!(
            gamma > 0.10 && gamma < 0.11,
            "entropy bridge at vacuum: {gamma}"
        );
        // Too small for gamma_BG (0.2375)
        assert!(gamma < GAMMA_BG, "entropy bridge < gamma_BG");
    }

    #[test]
    fn linear_bridge_exact_at_vacuum() {
        // By construction: linear_bridge_bg(VACUUM_PHI) = GAMMA_BG
        let gamma = linear_bridge_bg(VACUUM_PHI);
        assert!(
            (gamma - GAMMA_BG).abs() < 1e-14,
            "linear bridge should match gamma_BG at vacuum: {gamma}"
        );
    }

    #[test]
    fn log_bridge_exact_at_vacuum() {
        // By construction: log_bridge_bg(VACUUM_PHI) = GAMMA_BG
        let gamma = log_bridge_bg(VACUUM_PHI);
        assert!(
            (gamma - GAMMA_BG).abs() < 1e-14,
            "log bridge should match gamma_BG at vacuum: {gamma}"
        );
    }

    #[test]
    fn imbalance_entropy_at_half() {
        // At phi=1/2: H(1/2) = ln(2), so gamma = ln(2)/(pi*sqrt(3)) = 0.12738...
        // Note: this is NOT gamma_BG (0.2375), which solves a transcendental equation.
        let gamma = imbalance_entropy_bridge(0.5);
        let expected = 2.0_f64.ln() / (PI * 3.0_f64.sqrt()); // 0.12738
        assert!(
            (gamma - expected).abs() < 1e-12,
            "imbalance entropy at phi=0.5 should give ln(2)/(pi*sqrt(3)): {gamma}"
        );
        // This is the maximum of the bridge (binary entropy peaks at 1/2)
        assert!(gamma > imbalance_entropy_bridge(0.3));
        assert!(gamma > imbalance_entropy_bridge(0.7));
    }

    #[test]
    fn imbalance_entropy_at_vacuum() {
        // phi = 3/8: H(3/8) = -3/8*ln(3/8) - 5/8*ln(5/8)
        // H(0.375) = 0.375*0.9808 + 0.625*0.4700 = 0.3678 + 0.2938 = 0.6616
        // gamma = 0.6616 / (pi*sqrt(3)) = 0.6616/5.4414 = 0.1216
        let gamma = imbalance_entropy_bridge(VACUUM_PHI);
        // This should be very close to gamma_NZJ (0.1236)!
        assert!(
            gamma > 0.11 && gamma < 0.13,
            "imbalance entropy at vacuum: {gamma}"
        );
        // Check proximity to gamma_NZJ
        let dev = (gamma - GAMMA_NZJ).abs() / GAMMA_NZJ;
        assert!(
            dev < 0.02,
            "imbalance entropy at vacuum should be near gamma_NZJ: dev={dev:.4}"
        );
    }

    #[test]
    fn imbalance_entropy_zero_at_extremes() {
        assert!(imbalance_entropy_bridge(0.0).abs() < 1e-15, "H(0) = 0");
        assert!(imbalance_entropy_bridge(1.0).abs() < 1e-15, "H(1) = 0");
    }

    #[test]
    fn power_bridge_reduces_to_linear() {
        let phi = 0.25;
        let linear = linear_bridge_bg(phi);
        let power = power_bridge_bg(phi, 1.0);
        assert!(
            (linear - power).abs() < 1e-14,
            "power(alpha=1) should equal linear"
        );
    }

    #[test]
    fn invert_entropy_bridge_bg_is_none() {
        // gamma_BG = 0.2375 exceeds bridge max = 0.1274, so no solution exists.
        // This is an honest negative result.
        let result = invert_entropy_bridge_bg();
        assert!(
            result.is_none(),
            "gamma_BG exceeds bridge max -- no phi maps to it: got {:?}",
            result
        );
    }

    #[test]
    fn invert_entropy_bridge_nzj_near_vacuum() {
        let phi = invert_entropy_bridge_nzj();
        // phi for gamma_NZJ should be NEAR the imbalance attractor (3/8 = 0.375)
        let diff = (phi - VACUUM_PHI).abs();
        assert!(
            diff < 0.03,
            "phi for gamma_NZJ should be near 3/8: phi={phi:.6}, diff={diff:.6}"
        );
    }

    #[test]
    fn evaluate_all_returns_six_mappings() {
        let results = evaluate_all_mappings(VACUUM_PHI);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn best_bg_match_is_calibrated_bridge() {
        let best = best_bg_match();
        // Either linear_bridge_bg or log_bridge_bg should have deviation = 0
        assert!(
            best.deviation_bg < 1e-12,
            "best BG match should be exact: dev={}, mapping={}",
            best.deviation_bg,
            best.mapping,
        );
    }

    #[test]
    fn best_nzj_match_is_imbalance_entropy() {
        let best = best_nzj_match();
        // imbalance_entropy_bridge at VACUUM_PHI should be closest to gamma_NZJ
        assert!(
            best.deviation_nzj < 0.02,
            "best NZJ match dev should be < 2%: dev={:.4}, mapping={}",
            best.deviation_nzj,
            best.mapping,
        );
    }

    #[test]
    fn imbalance_entropy_bridge_is_best_nzj_candidate() {
        // The imbalance-entropy bridge at phi=3/8 gives ~0.1216,
        // which should be the closest to gamma_NZJ = 0.1236 among
        // all non-calibrated mappings.
        let gamma = imbalance_entropy_bridge(VACUUM_PHI);
        let dev = (gamma - GAMMA_NZJ).abs() / GAMMA_NZJ;
        // The entropy bridge (not calibrated to vacuum) gives 0.1028,
        // dev from NZJ = 0.168. Imbalance entropy dev < 0.02.
        let entropy_dev = (entropy_bridge(VACUUM_PHI) - GAMMA_NZJ).abs() / GAMMA_NZJ;
        assert!(
            dev < entropy_dev,
            "imbalance entropy ({dev:.4}) should beat entropy bridge ({entropy_dev:.4}) for NZJ"
        );
    }

    #[test]
    fn mapping_monotonicity() {
        // All bridges should be monotonically increasing on (0, 1)
        let phis: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
        for window in phis.windows(2) {
            let phi_lo = window[0];
            let phi_hi = window[1];
            assert!(entropy_bridge(phi_hi) > entropy_bridge(phi_lo));
            assert!(linear_bridge_bg(phi_hi) > linear_bridge_bg(phi_lo));
            assert!(log_bridge_bg(phi_hi) > log_bridge_bg(phi_lo));
            assert!(
                imbalance_entropy_bridge(phi_hi) >= imbalance_entropy_bridge(phi_lo)
                    || phi_hi > 0.5
            );
        }
    }

    #[test]
    fn imbalance_entropy_maximum_at_half() {
        // Binary entropy is maximized at phi = 0.5
        let gamma_half = imbalance_entropy_bridge(0.5);
        for phi in [0.1, 0.2, 0.3, 0.375, 0.4, 0.6, 0.7, 0.8, 0.9] {
            let gamma = imbalance_entropy_bridge(phi);
            assert!(
                gamma <= gamma_half + 1e-15,
                "imbalance entropy should be maximal at phi=0.5: H({phi})={gamma} > H(0.5)={gamma_half}"
            );
        }
    }
}
