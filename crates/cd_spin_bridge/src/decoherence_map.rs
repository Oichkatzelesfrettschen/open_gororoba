use sign_imbalance::IMBALANCE_ATTRACTOR;

/// Maps CD field observables (imbalance density, associator norms) to quantum channel parameters.
#[derive(Debug, Clone)]
pub struct DecoherenceMap {
    /// Scaling for imbalance term
    pub c_f: f64,
    /// Scaling for associator term
    pub c_a: f64,
    /// Base gamma
    pub gamma_0: f64,
    /// Temperature for lambda computation (if used)
    pub temperature: f64,
}

impl Default for DecoherenceMap {
    fn default() -> Self {
        Self {
            c_f: 10.0,
            c_a: 1.0,
            gamma_0: 0.0,
            temperature: 1.0,
        }
    }
}

impl DecoherenceMap {
    pub fn new(c_f: f64, c_a: f64, gamma_0: f64, temperature: f64) -> Self {
        Self {
            c_f,
            c_a,
            gamma_0,
            temperature,
        }
    }

    /// Computes gamma from local imbalance density and associator norm.
    /// gamma = gamma_0 + c_f * (F - 3/8)^2 + c_a * A
    pub fn gamma_from_imbalance(&self, imbalance: f64, associator_norm: f64) -> f64 {
        let f_diff = imbalance - IMBALANCE_ATTRACTOR;
        // Using squared difference as deviation from imbalance attractor implies higher energy cost/decoherence?
        // Or maybe just linear?
        // The plan suggests: gamma = gamma_0 + c_F * (F - 3/8)^2 + c_A * A

        let term_f = self.c_f * f_diff * f_diff;
        let term_a = self.c_a * associator_norm;

        self.gamma_0 + term_f + term_a
    }

    /// Computes gamma from lambda (coupling strength).
    /// gamma = kappa * lambda
    pub fn gamma_from_lambda(&self, lambda: f64, kappa: f64) -> f64 {
        kappa * lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_construction() {
        let dm = DecoherenceMap::default();
        assert_eq!(dm.c_f, 10.0);
        assert_eq!(dm.c_a, 1.0);
        assert_eq!(dm.gamma_0, 0.0);
        assert_eq!(dm.temperature, 1.0);
    }

    #[test]
    fn test_gamma_at_imbalance_attractor() {
        // At F = 3/8 (imbalance attractor), imbalance term vanishes.
        // gamma = gamma_0 + c_f * 0^2 + c_a * A = gamma_0 + c_a * A
        let dm = DecoherenceMap::default();
        let gamma = dm.gamma_from_imbalance(IMBALANCE_ATTRACTOR, 0.0);
        assert!(
            (gamma - dm.gamma_0).abs() < 1e-14,
            "At imbalance attractor with zero associator, gamma should equal gamma_0"
        );
    }

    #[test]
    fn test_gamma_nonzero_deviation() {
        // F != 3/8 should increase gamma (quadratic penalty)
        let dm = DecoherenceMap::default();
        let gamma_at_attractor = dm.gamma_from_imbalance(IMBALANCE_ATTRACTOR, 0.0);
        let gamma_deviated = dm.gamma_from_imbalance(0.5, 0.0);
        assert!(
            gamma_deviated > gamma_at_attractor,
            "Deviation from imbalance attractor should increase decoherence rate"
        );
    }

    #[test]
    fn test_gamma_associator_contribution() {
        let dm = DecoherenceMap::default();
        let gamma_no_assoc = dm.gamma_from_imbalance(IMBALANCE_ATTRACTOR, 0.0);
        let gamma_with_assoc = dm.gamma_from_imbalance(IMBALANCE_ATTRACTOR, 1.0);
        assert!(
            (gamma_with_assoc - gamma_no_assoc - dm.c_a).abs() < 1e-14,
            "Associator norm contribution should be c_a * A"
        );
    }

    #[test]
    fn test_gamma_from_lambda() {
        let dm = DecoherenceMap::default();
        let gamma = dm.gamma_from_lambda(2.0, 0.5);
        assert!((gamma - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_custom_construction() {
        let dm = DecoherenceMap::new(5.0, 2.0, 0.1, 300.0);
        assert_eq!(dm.c_f, 5.0);
        assert_eq!(dm.c_a, 2.0);
        assert_eq!(dm.gamma_0, 0.1);
        assert_eq!(dm.temperature, 300.0);
    }
}
