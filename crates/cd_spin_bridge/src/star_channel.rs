//! STAR-calibrated decoherence channel.
//!
//! WHY: The phenomenological gamma(dR) model calibrated to STAR Nature 2026
//! Fig. 4 is the validated bridge between separation dR and decoherence strength.
//! Having it as a named struct in cd_spin_bridge makes it the canonical Rust
//! implementation, matching the Haskell Bridge.hs and Python get_channel_strength().
//!
//! Calibration:
//!   gamma(dR) = gamma_0 * (1 - exp(-dR / lambda))
//!   gamma_0 = 0.611 => exp(-gamma_0) * (1/3) = 0.181 = STAR measured P (Fig. 4).
//!   lambda = 2.0 fm (characteristic decoherence scale).

/// STAR-calibrated decoherence channel for Lambda-AntiLambda pairs.
///
/// gamma(dR) = gamma_0 * (1 - exp(-dR / lambda))
#[derive(Debug, Clone)]
pub struct StarChannel {
    /// Saturation decoherence strength (STAR calibrated: 0.611).
    /// Derivation: exp(-gamma_0) * (1/3) = 0.181 => gamma_0 = -ln(0.181 * 3) = 0.611.
    pub gamma_0: f64,
    /// Characteristic decoherence scale in fm (default: 2.0 fm).
    pub lambda_fm: f64,
}

impl Default for StarChannel {
    fn default() -> Self {
        Self {
            gamma_0: 0.611,
            lambda_fm: 2.0,
        }
    }
}

impl StarChannel {
    pub fn new(gamma_0: f64, lambda_fm: f64) -> Self {
        Self { gamma_0, lambda_fm }
    }

    /// Compute the decoherence strength gamma at separation dR (fm).
    pub fn gamma_from_dr(&self, dr: f64) -> f64 {
        self.gamma_0 * (1.0 - (-dr / self.lambda_fm).exp())
    }

    /// Compute the depolarizing probability p = 1 - exp(-gamma).
    pub fn p_from_dr(&self, dr: f64) -> f64 {
        let gamma = self.gamma_from_dr(dr);
        1.0 - (-gamma).exp()
    }

    /// Compute the scalar P = exp(-gamma) / 3 predicted at separation dR.
    pub fn p_scalar_from_dr(&self, dr: f64) -> f64 {
        let gamma = self.gamma_from_dr(dr);
        (-gamma).exp() / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_zero_at_dr0() {
        let ch = StarChannel::default();
        assert!(ch.gamma_from_dr(0.0).abs() < 1e-14, "gamma(0) must be 0");
    }

    #[test]
    fn test_gamma_positive_for_positive_dr() {
        let ch = StarChannel::default();
        assert!(ch.gamma_from_dr(1.0) > 0.0);
    }

    #[test]
    fn test_gamma_saturates_at_gamma0() {
        // At large dR, gamma -> gamma_0
        let ch = StarChannel::default();
        let gamma_large = ch.gamma_from_dr(1000.0);
        assert!(
            (gamma_large - ch.gamma_0).abs() < 1e-9,
            "gamma must saturate at gamma_0={}: got {gamma_large}",
            ch.gamma_0
        );
    }

    #[test]
    fn test_star_calibration_p_at_saturation() {
        // STAR measured P = 0.181 +/- 0.035 (Fig. 4).
        // At large dR: P_sat = exp(-gamma_0) / 3 = exp(-0.611) / 3.
        let ch = StarChannel::default();
        let p_sat = ch.p_scalar_from_dr(1000.0);
        assert!(
            (p_sat - 0.181).abs() < 0.035,
            "Saturated P must match STAR measurement: P_sat={p_sat:.4}"
        );
    }

    #[test]
    fn test_p_scalar_one_third_at_dr0() {
        // At dR=0: no decoherence, P = 1/3 (physical triplet baseline).
        let ch = StarChannel::default();
        let p = ch.p_scalar_from_dr(0.0);
        assert!(
            (p - 1.0 / 3.0).abs() < 1e-12,
            "P(dR=0) must be 1/3: got {p}"
        );
    }

    #[test]
    fn test_p_decreases_with_dr() {
        let ch = StarChannel::default();
        let p0 = ch.p_scalar_from_dr(0.0);
        let p1 = ch.p_scalar_from_dr(1.0);
        let p5 = ch.p_scalar_from_dr(5.0);
        assert!(
            p0 > p1 && p1 > p5,
            "P must decrease with dR: {p0:.4} > {p1:.4} > {p5:.4}"
        );
    }

    #[test]
    fn test_custom_channel() {
        let ch = StarChannel::new(1.0, 1.0);
        let gamma = ch.gamma_from_dr(1.0);
        let expected = 1.0 * (1.0 - (-1.0_f64).exp());
        assert!((gamma - expected).abs() < 1e-14);
    }
}
