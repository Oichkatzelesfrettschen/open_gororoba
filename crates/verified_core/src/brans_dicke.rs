//! Brans-Dicke PPN parameters (Rocq-verified: C-886..C-892).
//!
//! gamma_BD(omega) = (1+omega)/(2+omega) approaches GR as omega -> inf.
//! Kernel-checked in `proofs/theories/BransDicke.v`.

/// PPN gamma in Brans-Dicke theory.
///
/// # Panics
///
/// Panics if `omega <= -2.0` (denominator zero or negative).
pub fn ppn_gamma_bd(omega: f64) -> f64 {
    assert!(omega > -2.0, "omega must be > -2, got {omega}");
    (1.0 + omega) / (2.0 + omega)
}

/// Nordtvedt parameter: deviation from strong equivalence principle.
///
/// # Panics
///
/// Panics if `omega <= -2.0`.
pub fn nordtvedt_bd(omega: f64) -> f64 {
    assert!(omega > -2.0, "omega must be > -2, got {omega}");
    1.0 / (2.0 + omega)
}

/// Gamma deviation from GR: gamma_BD - 1 = -nordtvedt_bd.
pub fn ppn_gamma_deviation(omega: f64) -> f64 {
    ppn_gamma_bd(omega) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn deviation_equals_neg_nordtvedt() {
        for &omega in &[0.1, 1.0, 10.0, 100.0, 1000.0, 43476.0] {
            let dev = ppn_gamma_deviation(omega);
            let nord = nordtvedt_bd(omega);
            assert!(
                (dev + nord).abs() < TOL,
                "deviation + nordtvedt should be 0: {dev} + {nord}"
            );
        }
    }

    #[test]
    fn gamma_range() {
        for &omega in &[0.01, 0.1, 1.0, 10.0, 100.0, 1e6] {
            let g = ppn_gamma_bd(omega);
            assert!(g > 0.5, "gamma({omega}) = {g} should be > 0.5");
            assert!(g < 1.0, "gamma({omega}) = {g} should be < 1.0");
        }
    }

    #[test]
    fn gamma_monotone() {
        let values: Vec<f64> = [0.1, 1.0, 10.0, 100.0, 1000.0]
            .iter()
            .map(|&w| ppn_gamma_bd(w))
            .collect();
        for w in values.windows(2) {
            assert!(w[1] > w[0], "gamma should be increasing");
        }
    }

    #[test]
    fn cassini_bound() {
        let omega = 43477.0;
        assert!(ppn_gamma_deviation(omega).abs() < 23.0 / 1_000_000.0);
    }

    #[test]
    fn nordtvedt_bound() {
        let omega = 1999.0;
        assert!(nordtvedt_bd(omega) < 5.0 / 10000.0);
    }

    #[test]
    fn gpb_geodetic_bound() {
        let omega = 178.0;
        assert!(ppn_gamma_deviation(omega).abs() < 56.0 / 10000.0);
    }

    #[test]
    #[should_panic]
    fn invalid_omega_panics() {
        ppn_gamma_bd(-3.0);
    }
}
