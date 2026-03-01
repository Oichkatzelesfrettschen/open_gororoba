//! Neg-dim equation of state (Rocq-verified: C-884).
//!
//! The effective EOS w_eff = -1 - eta*(alpha + 3/2) depends on (alpha, eta)
//! only through the product P = eta*(alpha + 3/2).  The 2D parameterization
//! is therefore degenerate: infinitely many (alpha, eta) pairs yield the same
//! w_eff.  Kernel-checked in `proofs/verified/C884_NegDimDegeneracy.v`.

/// Neg-dim effective equation of state.
///
/// Returns w_eff = -1 - eta * (alpha + 3/2).
pub fn neg_dim_w_eff(alpha: f64, eta: f64) -> f64 {
    -1.0 - eta * (alpha + 1.5)
}

/// The degeneracy product: the only observable combination of (alpha, eta).
pub fn neg_dim_product(alpha: f64, eta: f64) -> f64 {
    eta * (alpha + 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuum_limit() {
        // eta = 0 => w_eff = -1 (cosmological constant)
        assert!((neg_dim_w_eff(1.0, 0.0) - (-1.0)).abs() < 1e-15);
        assert!((neg_dim_w_eff(100.0, 0.0) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn product_degeneracy() {
        // Two distinct (alpha, eta) pairs with same product => same w_eff
        let w1 = neg_dim_w_eff(0.0, 1.0); // P = 1 * 1.5 = 1.5
        let w2 = neg_dim_w_eff(1.0, 0.6); // P = 0.6 * 2.5 = 1.5
        assert!((w1 - w2).abs() < 1e-14);
    }

    #[test]
    fn phantom_regime() {
        // eta > 0, alpha > -1.5 => P > 0 => w < -1 (phantom)
        assert!(neg_dim_w_eff(1.0, 0.5) < -1.0);
    }

    #[test]
    fn quintessence_regime() {
        // eta < 0, alpha > -1.5 => P < 0 => w > -1 (quintessence)
        assert!(neg_dim_w_eff(1.0, -0.5) > -1.0);
    }
}
