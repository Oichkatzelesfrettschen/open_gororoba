//! Kerr-de Sitter metric: rotating black hole with cosmological constant.
//!
//! The Kerr-de Sitter (KdS) solution describes a rotating black hole in an
//! asymptotically de Sitter (expanding) universe.  It has three parameters:
//!   M -- black hole mass
//!   a -- spin parameter (J/M)
//!   Lambda -- cosmological constant (Lambda > 0 for de Sitter)
//!
//! Key modification from Kerr: Delta = r^2 - 2Mr + a^2 - Lambda r^2/3.
//! The -Lambda r^2/3 term creates a THIRD horizon (cosmological horizon)
//! beyond which the de Sitter expansion dominates.
//!
//! Triple horizon structure: r_- (inner) < r_+ (event) < r_c (cosmological).
//! Stable orbits exist only in the exterior region r_+ < r < r_c.
//!
//! Limits:
//!   Lambda -> 0: reduces to standard Kerr metric
//!   M -> 0, a -> 0: reduces to pure de Sitter spacetime
//!
//! References:
//!   - Carter (1973): Black hole equilibrium states with cosmological constant
//!   - Griffiths & Podolsky (2009): Exact Space-Times in Einstein's GR

// ============================================================================
// Metric helper functions
// ============================================================================

/// Sigma = r^2 + a^2 cos^2(theta).
///
/// Unchanged from Kerr -- the cosmological constant does not modify Sigma.
pub fn sigma(r: f64, theta: f64, a: f64) -> f64 {
    let cos_th = theta.cos();
    r * r + a * a * cos_th * cos_th
}

/// Delta = r^2 - 2Mr + a^2 - Lambda r^2/3.
///
/// The -Lambda r^2/3 term is the key modification from Kerr.
/// This fourth-degree polynomial in r (when multiplied out) can have
/// up to four real roots; the three positive ones are the horizons.
pub fn delta(r: f64, m: f64, a: f64, lambda: f64) -> f64 {
    r * r - 2.0 * m * r + a * a - lambda * r * r / 3.0
}

/// A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta).
///
/// Uses the modified Delta with cosmological term.
pub fn big_a(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> f64 {
    let r2_plus_a2 = r * r + a * a;
    let sin_th = theta.sin();
    let d = delta(r, m, a, lambda);
    r2_plus_a2 * r2_plus_a2 - a * a * d * sin_th * sin_th
}

// ============================================================================
// Metric components
// ============================================================================

/// g_tt = -(1 - 2Mr/Sigma + Lambda r^2 sin^2(theta)/3).
pub fn g_tt(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> f64 {
    let sig = sigma(r, theta, a);
    let sin_th = theta.sin();
    -(1.0 - 2.0 * m * r / sig + lambda * r * r * sin_th * sin_th / 3.0)
}

/// g_rr = Sigma / Delta.
pub fn g_rr(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> f64 {
    sigma(r, theta, a) / delta(r, m, a, lambda)
}

/// g_{theta theta} = Sigma.
pub fn g_thth(r: f64, theta: f64, a: f64) -> f64 {
    sigma(r, theta, a)
}

/// g_{phi phi} = (r^2 + a^2 + 2Mra^2 sin^2(theta)/Sigma
///               - Lambda r^4 sin^2(theta)/3) sin^2(theta).
pub fn g_phph(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> f64 {
    let sig = sigma(r, theta, a);
    let sin_th = theta.sin();
    let sin2 = sin_th * sin_th;
    (r * r + a * a + 2.0 * m * r * a * a * sin2 / sig - lambda * r * r * r * r * sin2 / 3.0) * sin2
}

/// g_{t phi} = -2Mar sin^2(theta) / Sigma.
///
/// Unchanged from Kerr -- the cosmological constant does not modify
/// the frame dragging cross term directly.
pub fn g_tphi(r: f64, theta: f64, m: f64, a: f64) -> f64 {
    let sig = sigma(r, theta, a);
    let sin_th = theta.sin();
    -2.0 * m * r * a * sin_th * sin_th / sig
}

// ============================================================================
// Horizon structure
// ============================================================================

/// Inner (Cauchy) horizon, approximate for small Lambda.
///
/// r_- ~ r_Kerr_inner - Lambda r_Kerr_inner^3 / 3
///
/// where r_Kerr_inner = M - sqrt(M^2 - a^2).
pub fn inner_horizon(m: f64, a: f64, lambda: f64) -> f64 {
    let discriminant = (m * m - a * a).sqrt();
    let r_kerr = m - discriminant;
    r_kerr - lambda * r_kerr * r_kerr * r_kerr / 3.0
}

/// Event horizon, approximate for small Lambda.
///
/// r_+ ~ r_Kerr_outer + Lambda r_Kerr_outer^3 / 3
///
/// where r_Kerr_outer = M + sqrt(M^2 - a^2).
/// The Lambda correction pushes the event horizon slightly outward.
pub fn event_horizon(m: f64, a: f64, lambda: f64) -> f64 {
    let discriminant = (m * m - a * a).sqrt();
    let r_kerr = m + discriminant;
    r_kerr + lambda * r_kerr * r_kerr * r_kerr / 3.0
}

/// Cosmological horizon: r_c = sqrt(3 / Lambda).
///
/// This is the de Sitter horizon radius.  Beyond r_c, the universe
/// expands superluminally (objects recede faster than light).
///
/// For the observed Lambda ~ 1.1e-52 m^-2, r_c ~ 1.6e26 m ~ 5 Gpc.
pub fn cosmological_horizon(lambda: f64) -> f64 {
    (3.0 / lambda).sqrt()
}

/// Ergosphere outer boundary.
///
/// For small Lambda, approximately: r_ergo = M + sqrt(M^2 - a^2 cos^2(theta)).
/// The cosmological correction is negligible at the ergosphere scale.
pub fn ergosphere_radius(theta: f64, m: f64, a: f64) -> f64 {
    let cos_th = theta.cos();
    m + (m * m - a * a * cos_th * cos_th).sqrt()
}

// ============================================================================
// Frame dragging
// ============================================================================

/// Frame dragging angular velocity: omega = -g_{t phi} / g_{phi phi}.
pub fn frame_dragging_omega(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> f64 {
    let g_tp = g_tphi(r, theta, m, a);
    let g_pp = g_phph(r, theta, m, a, lambda);
    -g_tp / g_pp
}

// ============================================================================
// Physical validity
// ============================================================================

/// Physical Kerr-de Sitter: M > 0, Lambda > 0, M^2 >= a^2.
pub fn is_physical(m: f64, a: f64, lambda: f64) -> bool {
    m > 0.0 && lambda > 0.0 && m * m >= a * a
}

/// Check if a point is in the exterior region (between event and cosmological horizons).
///
/// This is where stable orbits and normal physics exist.
pub fn is_exterior(r: f64, m: f64, a: f64, lambda: f64) -> bool {
    let r_plus = event_horizon(m, a, lambda);
    let r_cosmo = cosmological_horizon(lambda);
    r > r_plus && r < r_cosmo
}

/// Check if a point is in the ergosphere (g_tt > 0).
pub fn is_in_ergosphere(r: f64, theta: f64, m: f64, a: f64, lambda: f64) -> bool {
    g_tt(r, theta, m, a, lambda) > 0.0
}

/// Verify that horizons are properly ordered: r_- < r_+ < r_c.
pub fn verify_horizon_ordering(m: f64, a: f64, lambda: f64) -> bool {
    if !is_physical(m, a, lambda) {
        return false;
    }
    let r_minus = inner_horizon(m, a, lambda);
    let r_plus = event_horizon(m, a, lambda);
    let r_cosmo = cosmological_horizon(lambda);
    r_minus < r_plus && r_plus < r_cosmo
}

// ============================================================================
// Cosmological constant utilities
// ============================================================================

/// Observed cosmological constant: Lambda ~ 1.1e-52 m^{-2} (Planck 2018).
///
/// In geometric units where c = G = 1, this value can be used directly.
pub const LAMBDA_OBSERVED: f64 = 1.1e-52;

/// Cosmological horizon radius for the observed Lambda.
///
/// r_c = sqrt(3/Lambda) ~ 1.65e26 m ~ 5.3 Gpc.
pub fn observed_cosmological_horizon() -> f64 {
    cosmological_horizon(LAMBDA_OBSERVED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // -- Kerr limit (Lambda -> 0) --

    #[test]
    fn test_reduces_to_kerr_delta() {
        // Lambda=0: Delta = r^2 - 2Mr + a^2 (standard Kerr)
        let d_kds = delta(10.0, 1.0, 0.5, 0.0);
        let d_kerr = 10.0 * 10.0 - 2.0 * 1.0 * 10.0 + 0.25;
        assert!((d_kds - d_kerr).abs() < 1e-14);
    }

    #[test]
    fn test_reduces_to_kerr_horizon() {
        // Lambda=0: r_+ = M + sqrt(M^2 - a^2)
        let r = event_horizon(1.0, 0.5, 0.0);
        let expected = 1.0 + (1.0 - 0.25_f64).sqrt();
        assert!((r - expected).abs() < 1e-10, "r = {r}");
    }

    // -- Triple horizon structure --

    #[test]
    fn test_horizon_ordering() {
        // For astrophysically reasonable Lambda
        let m = 1.0;
        let a = 0.5;
        let lambda = 1e-4; // much larger than observed, for testability
        assert!(verify_horizon_ordering(m, a, lambda));
    }

    #[test]
    fn test_cosmological_horizon() {
        let lambda = 1e-4;
        let r_c = cosmological_horizon(lambda);
        let expected = (3.0 / 1e-4_f64).sqrt(); // sqrt(30000) ~ 173.2
        assert!((r_c - expected).abs() < 1e-10, "r_c = {r_c}");
    }

    #[test]
    fn test_inner_less_than_event() {
        let r_minus = inner_horizon(1.0, 0.5, 1e-4);
        let r_plus = event_horizon(1.0, 0.5, 1e-4);
        assert!(r_minus < r_plus, "r- = {r_minus}, r+ = {r_plus}");
    }

    #[test]
    fn test_event_less_than_cosmological() {
        let r_plus = event_horizon(1.0, 0.5, 1e-4);
        let r_cosmo = cosmological_horizon(1e-4);
        assert!(r_plus < r_cosmo, "r+ = {r_plus}, r_c = {r_cosmo}");
    }

    // -- Lambda pushes event horizon outward --

    #[test]
    fn test_lambda_expands_event_horizon() {
        let r_kerr = event_horizon(1.0, 0.5, 0.0);
        let r_kds = event_horizon(1.0, 0.5, 1e-4);
        assert!(r_kds > r_kerr, "Lambda should push event horizon outward");
    }

    // -- Delta modified by Lambda --

    #[test]
    fn test_delta_lambda_reduces_delta() {
        // The -Lambda r^2/3 term reduces Delta compared to Kerr at the same r.
        // This is the key physical effect: Lambda "competes" with the r^2 term.
        let r = 50.0;
        let d_kerr = delta(r, 1.0, 0.5, 0.0);
        let d_kds = delta(r, 1.0, 0.5, 1e-4);
        let reduction = d_kerr - d_kds;
        let expected = 1e-4 * r * r / 3.0; // Lambda r^2 / 3
        assert!(
            (reduction - expected).abs() < 1e-10,
            "reduction = {reduction}, expected = {expected}"
        );
    }

    #[test]
    fn test_delta_positive_between_horizons() {
        // Delta should be positive in the exterior region between horizons
        let d = delta(10.0, 1.0, 0.5, 1e-4);
        assert!(d > 0.0, "Delta = {d}");
    }

    // -- Metric components --

    #[test]
    fn test_g_tt_flat_at_moderate_r() {
        // At moderate r (between horizons), g_tt should be close to Kerr
        let g_kds = g_tt(10.0, PI / 4.0, 1.0, 0.5, 1e-10);
        let sig = sigma(10.0, PI / 4.0, 0.5);
        let g_kerr = -(1.0 - 2.0 * 10.0 / sig); // Kerr g_tt (a-dependent)
        assert!(
            (g_kds - g_kerr).abs() < 1e-6,
            "g_tt(KdS) = {g_kds}, g_tt(Kerr) = {g_kerr}"
        );
    }

    #[test]
    fn test_g_rr_positive_between_horizons() {
        let r = 10.0;
        let g = g_rr(r, PI / 4.0, 1.0, 0.5, 1e-4);
        assert!(g > 0.0, "g_rr = {g}");
    }

    #[test]
    fn test_g_phph_positive() {
        let g = g_phph(10.0, PI / 4.0, 1.0, 0.5, 1e-4);
        assert!(g > 0.0);
    }

    // -- Frame dragging --

    #[test]
    fn test_frame_dragging_positive() {
        let omega = frame_dragging_omega(5.0, PI / 2.0, 1.0, 0.5, 1e-4);
        assert!(omega > 0.0);
    }

    #[test]
    fn test_frame_dragging_decreases_with_r() {
        let o1 = frame_dragging_omega(3.0, PI / 2.0, 1.0, 0.5, 1e-4);
        let o2 = frame_dragging_omega(10.0, PI / 2.0, 1.0, 0.5, 1e-4);
        assert!(o2 < o1);
    }

    // -- Ergosphere --

    #[test]
    fn test_ergosphere_at_pole_equals_horizon() {
        let m = 1.0;
        let a = 0.5;
        let r_ergo_pole = ergosphere_radius(0.0, m, a);
        let r_horizon = event_horizon(m, a, 0.0); // Kerr limit for comparison
        let expected = m + (m * m - a * a).sqrt();
        assert!(
            (r_ergo_pole - expected).abs() < 1e-10,
            "ergosphere at pole = {r_ergo_pole}, horizon = {r_horizon}"
        );
    }

    #[test]
    fn test_ergosphere_exceeds_horizon_at_equator() {
        let m = 1.0;
        let a = 0.5;
        let lambda = 1e-4;
        let r_ergo_eq = ergosphere_radius(PI / 2.0, m, a);
        let r_horizon = event_horizon(m, a, lambda);
        // Ergosphere at equator: M + sqrt(M^2) = 2M for a=0
        // For a=0.5: M + M = 2M (since cos(pi/2) = 0)
        assert!(
            r_ergo_eq > r_horizon,
            "ergosphere should extend beyond horizon"
        );
    }

    // -- Validity --

    #[test]
    fn test_is_physical() {
        assert!(is_physical(1.0, 0.5, 1e-4));
        assert!(!is_physical(1.0, 0.5, 0.0)); // Lambda must be > 0
        assert!(!is_physical(1.0, 1.5, 1e-4)); // a > M
    }

    #[test]
    fn test_exterior_region() {
        let m = 1.0;
        let a = 0.5;
        let lambda = 1e-4;
        assert!(is_exterior(10.0, m, a, lambda));
        assert!(!is_exterior(1.0, m, a, lambda)); // inside event horizon
        assert!(!is_exterior(200.0, m, a, lambda)); // outside cosmological horizon
    }

    // -- Observed Lambda --

    #[test]
    fn test_observed_cosmological_horizon() {
        let r_c = observed_cosmological_horizon();
        // sqrt(3 / 1.1e-52) ~ 1.65e26 m
        assert!(r_c > 1e26 && r_c < 2e26, "r_c = {r_c}");
    }

    #[test]
    fn test_observed_lambda_horizon_ordering() {
        // Even with observed (tiny) Lambda, horizons should be ordered
        assert!(verify_horizon_ordering(1.0, 0.5, LAMBDA_OBSERVED));
    }
}
