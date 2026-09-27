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
        Self {
            mass: 1.0,
            spin: 0.0,
        }
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
        // The Boyer-Lindquist determinant is -Sigma^2 sin^2(theta).
        sig * theta.sin().abs()
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

    fn assembled_gcov(components: [f64; 5]) -> [[f64; 4]; 4] {
        let [g_tt, g_rr, g_thth, g_phph, g_tph] = components;
        [
            [g_tt, 0.0, 0.0, g_tph],
            [0.0, g_rr, 0.0, 0.0],
            [0.0, 0.0, g_thth, 0.0],
            [g_tph, 0.0, 0.0, g_phph],
        ]
    }

    fn determinant_4x4(mut matrix: [[f64; 4]; 4]) -> f64 {
        let mut determinant = 1.0;
        for column in 0..4 {
            let pivot = (column..4)
                .max_by(|&left, &right| {
                    matrix[left][column]
                        .abs()
                        .total_cmp(&matrix[right][column].abs())
                })
                .expect("matrix column has a pivot");
            if matrix[pivot][column].abs() < f64::MIN_POSITIVE {
                return 0.0;
            }
            if pivot != column {
                matrix.swap(pivot, column);
                determinant = -determinant;
            }
            let diagonal = matrix[column][column];
            determinant *= diagonal;
            let pivot_row = matrix[column];
            for row_values in matrix.iter_mut().skip(column + 1) {
                let factor = row_values[column] / diagonal;
                for (trailing, value) in row_values.iter_mut().enumerate().skip(column + 1) {
                    *value -= factor * pivot_row[trailing];
                }
            }
        }
        determinant
    }

    fn kernel_inverse(components: [f64; 5]) -> [[f64; 4]; 4] {
        let [g_tt, g_rr, g_thth, g_phph, g_tph] = components;
        let determinant = g_tt * g_phph - g_tph * g_tph;
        let signed_floor = determinant.signum() * determinant.abs().max(1e-40);
        let inverse_determinant = 1.0 / signed_floor;
        [
            [
                g_phph * inverse_determinant,
                0.0,
                0.0,
                -g_tph * inverse_determinant,
            ],
            [0.0, 1.0 / g_rr, 0.0, 0.0],
            [0.0, 0.0, 1.0 / g_thth, 0.0],
            [
                -g_tph * inverse_determinant,
                0.0,
                0.0,
                g_tt * inverse_determinant,
            ],
        ]
    }

    #[test]
    fn test_schwarzschild_horizon() {
        let m = KerrMetric::schwarzschild();
        assert!(
            (m.r_horizon() - 2.0).abs() < 1e-10,
            "Schwarzschild r_+ = 2M"
        );
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
        assert!(
            (rh - 1.4359).abs() < 0.001,
            "Kerr a=0.9 r_+ ~ 1.436, got {}",
            rh
        );
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
    fn test_sqrt_neg_g_matches_assembled_metric_determinant() {
        let cases = [
            (3.0, 0.37, 0.0),
            (4.7, 1.13, 0.35),
            (9.5, 2.41, -0.7),
            (2.8, 0.82, 0.9),
        ];
        for (r, theta, spin) in cases {
            let metric = KerrMetric::kerr(spin);
            let determinant = determinant_4x4(assembled_gcov(metric.gcov(r, theta)));
            let expected = (-determinant).sqrt();
            let actual = metric.sqrt_neg_g(r, theta);
            let relative_error = (actual - expected).abs() / expected;
            assert!(
                relative_error < 1e-12,
                "sqrt(-g) mismatch at r={r}, theta={theta}, a={spin}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn test_kernel_inverse_block_multiplies_to_identity_on_exterior_grid() {
        for spin in [0.0, 0.4, -0.85] {
            let metric = KerrMetric::kerr(spin);
            for r in [2.5, 4.0, 12.0] {
                if r <= metric.r_horizon() {
                    continue;
                }
                for theta in [0.4, 1.2, 2.5] {
                    let covariant = assembled_gcov(metric.gcov(r, theta));
                    let contravariant = kernel_inverse(metric.gcov(r, theta));
                    let covariant_columns: [[f64; 4]; 4] = std::array::from_fn(|column| {
                        std::array::from_fn(|row| covariant[row][column])
                    });
                    for (row, inverse_row) in contravariant.iter().enumerate() {
                        for (column, covariant_column) in covariant_columns.iter().enumerate() {
                            let product: f64 = inverse_row
                                .iter()
                                .zip(covariant_column)
                                .map(|(inverse, metric)| inverse * metric)
                                .sum();
                            let expected = if row == column { 1.0 } else { 0.0 };
                            assert!(
                                (product - expected).abs() < 1e-12,
                                "inverse identity failed at r={r}, theta={theta}, a={spin}, ({row},{column})={product}"
                            );
                        }
                    }
                }
            }
        }
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
        assert!(
            alpha < 0.1 && alpha > 0.0,
            "Lapse should be small near horizon, got {}",
            alpha
        );
    }
}
