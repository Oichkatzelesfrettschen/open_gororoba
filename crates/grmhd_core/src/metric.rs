//! Kerr spacetime metric in Boyer-Lindquist coordinates.
//!
//! The metric components are computed from the Kerr parameters (M, a)
//! and coordinates (r, theta). All computations use geometric units (G=c=1).
//!
//! Convention: signature (-,+,+,+), coordinates (t, r, theta, phi).
//!
//! The 4-metric g_mu_nu in Boyer-Lindquist:
//!   g_tt = -(1 - 2Mr/Sigma)
//!   g_rr = Sigma/Delta
//!   g_thth = Sigma
//!   g_phph = sin^2(theta) * (r^2 + a^2 + 2Ma^2 r sin^2(theta)/Sigma)
//!   g_tph = g_pht = -2Mar sin^2(theta)/Sigma
//! where Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2Mr + a^2.

/// Kerr metric at a single spacetime point.
#[derive(Clone, Debug)]
pub struct KerrMetric {
    /// Black hole mass (geometric units, typically M=1).
    pub mass: f64,
    /// Dimensionless spin parameter a = J/M, |a| <= M.
    pub spin: f64,
}

impl KerrMetric {
    pub fn schwarzschild() -> Self {
        Self { mass: 1.0, spin: 0.0 }
    }

    pub fn kerr(spin: f64) -> Self {
        assert!(spin.abs() <= 1.0, "Spin |a/M| must be <= 1, got {}", spin);
        Self { mass: 1.0, spin }
    }

    /// Sigma = r^2 + a^2 cos^2(theta)
    #[inline]
    pub fn sigma(&self, r: f64, theta: f64) -> f64 {
        r * r + self.spin * self.spin * theta.cos().powi(2)
    }

    /// Delta = r^2 - 2Mr + a^2
    #[inline]
    pub fn delta(&self, r: f64) -> f64 {
        r * r - 2.0 * self.mass * r + self.spin * self.spin
    }

    /// Event horizon radius: r_+ = M + sqrt(M^2 - a^2)
    pub fn r_horizon(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin).sqrt()
    }

    /// ISCO radius for prograde orbits (Bardeen et al. 1972).
    pub fn r_isco(&self) -> f64 {
        let a = self.spin / self.mass;
        let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
        let z2 = (3.0 * a * a + z1 * z1).sqrt();
        self.mass * (3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt())
    }

    /// Covariant metric components g_mu_nu at (r, theta).
    /// Returns [g_tt, g_rr, g_thth, g_phph, g_tph] (5 independent components).
    pub fn gcov(&self, r: f64, theta: f64) -> [f64; 5] {
        let sig = self.sigma(r, theta);
        let del = self.delta(r);
        let a = self.spin;
        let m = self.mass;
        let sth2 = theta.sin().powi(2);

        let g_tt = -(1.0 - 2.0 * m * r / sig);
        let g_rr = sig / del;
        let g_thth = sig;
        let g_phph = sth2 * (r * r + a * a + 2.0 * m * a * a * r * sth2 / sig);
        let g_tph = -2.0 * m * a * r * sth2 / sig;

        [g_tt, g_rr, g_thth, g_phph, g_tph]
    }

    /// Contravariant metric components g^mu_nu at (r, theta).
    /// Returns [g^tt, g^rr, g^thth, g^phph, g^tph].
    pub fn gcon(&self, r: f64, theta: f64) -> [f64; 5] {
        let [g_tt, g_rr, g_thth, g_phph, g_tph] = self.gcov(r, theta);
        let det_2d = g_tt * g_phph - g_tph * g_tph;

        let gcon_tt = g_phph / det_2d;
        let gcon_rr = 1.0 / g_rr;
        let gcon_thth = 1.0 / g_thth;
        let gcon_phph = g_tt / det_2d;
        let gcon_tph = -g_tph / det_2d;

        [gcon_tt, gcon_rr, gcon_thth, gcon_phph, gcon_tph]
    }

    /// Metric determinant sqrt(-g) at (r, theta).
    pub fn sqrt_neg_g(&self, r: f64, theta: f64) -> f64 {
        let sig = self.sigma(r, theta);
        // sqrt(-g) = sqrt(Sigma) * sin(theta) for Kerr in BL coords
        sig.sqrt() * theta.sin().abs()
    }

    /// Lapse function alpha = 1/sqrt(-g^tt).
    pub fn lapse(&self, r: f64, theta: f64) -> f64 {
        let [gcon_tt, _, _, _, _] = self.gcon(r, theta);
        1.0 / (-gcon_tt).sqrt()
    }

    /// Shift vector beta^i = -g^ti/g^tt (only phi component is nonzero for Kerr).
    pub fn shift_phi(&self, r: f64, theta: f64) -> f64 {
        let [gcon_tt, _, _, _, gcon_tph] = self.gcon(r, theta);
        -gcon_tph / gcon_tt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_horizon() {
        let m = KerrMetric::schwarzschild();
        assert!((m.r_horizon() - 2.0).abs() < 1e-10, "Schwarzschild r_+ = 2M");
    }

    #[test]
    fn test_schwarzschild_isco() {
        let m = KerrMetric::schwarzschild();
        assert!((m.r_isco() - 6.0).abs() < 1e-10, "Schwarzschild ISCO = 6M");
    }

    #[test]
    fn test_kerr_horizon_spin09() {
        let m = KerrMetric::kerr(0.9);
        let rh = m.r_horizon();
        // r_+ = 1 + sqrt(1 - 0.81) = 1 + sqrt(0.19) ~ 1.4359
        assert!((rh - 1.4359).abs() < 0.001, "Kerr a=0.9 r_+ ~ 1.436, got {}", rh);
    }

    #[test]
    fn test_metric_signature() {
        let m = KerrMetric::schwarzschild();
        let [g_tt, g_rr, g_thth, g_phph, _] = m.gcov(10.0, std::f64::consts::FRAC_PI_2);
        assert!(g_tt < 0.0, "g_tt should be negative (timelike)");
        assert!(g_rr > 0.0, "g_rr should be positive (spacelike)");
        assert!(g_thth > 0.0, "g_thth should be positive");
        assert!(g_phph > 0.0, "g_phph should be positive");
    }

    #[test]
    fn test_flat_space_limit() {
        // At large r, Kerr -> Minkowski
        let m = KerrMetric::schwarzschild();
        let r = 1e6;
        let th = std::f64::consts::FRAC_PI_2;
        let [g_tt, g_rr, _, _, _] = m.gcov(r, th);
        assert!((g_tt + 1.0).abs() < 1e-5, "g_tt -> -1 at large r");
        assert!((g_rr - 1.0).abs() < 1e-5, "g_rr -> 1 at large r");
    }

    #[test]
    fn test_lapse_at_horizon() {
        let m = KerrMetric::schwarzschild();
        let rh = m.r_horizon() + 0.001; // just outside horizon
        let alpha = m.lapse(rh, std::f64::consts::FRAC_PI_2);
        assert!(alpha < 0.1 && alpha > 0.0, "Lapse should be small near horizon, got {}", alpha);
    }
}
