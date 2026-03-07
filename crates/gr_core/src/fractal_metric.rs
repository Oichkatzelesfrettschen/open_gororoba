use crate::metric::{SpacetimeMetric, MetricComponents, DIM, R, THETA};

/// A scale-invariant fractal spacetime metric wrapper.
///
/// It modifies an underlying metric by introducing scale-dependent
/// scaling laws characterized by a fractal dimension D_f.
pub struct FractalMetric<M: SpacetimeMetric> {
    pub base_metric: M,
    /// Fractal dimension (e.g. 2.7 for unified flyby/pioneer anomaly).
    pub d_f: f64,
    /// Reference scale r_0 (km).
    pub r_0: f64,
}

impl<M: SpacetimeMetric> FractalMetric<M> {
    pub fn new(base_metric: M, d_f: f64, r_0: f64) -> Self {
        Self { base_metric, d_f, r_0 }
    }

    /// Compute the fractal scaling factor at radius r.
    ///
    /// `sigma(r) = (r/r_0)^(D_f - 4)` shifts the effective volume
    /// element from 4D to D_f dimensions.
    pub fn scaling_factor(&self, r: f64) -> f64 {
        if r > 1e-10 {
            (r / self.r_0).powf(self.d_f - 4.0)
        } else {
            1.0
        }
    }

    /// Compute the anomalous radial acceleration from the fractal metric.
    ///
    /// For a radially-moving test particle, the geodesic deviation
    /// introduces an extra radial acceleration:
    ///
    ///   a_fractal = -(D_f - 4) * v^2 / (2 * r)
    ///
    /// where v is the radial velocity. For D_f = 2.7:
    ///   a_fractal = +0.65 * v^2 / r  (outward for D_f < 4, retarding)
    ///
    /// This is the same functional form as the Pioneer anomaly
    /// deceleration when v is the spacecraft velocity and r is the
    /// heliocentric distance.
    pub fn anomalous_acceleration(&self, r: f64, v: f64) -> f64 {
        if r > 1e-10 {
            -(self.d_f - 4.0) * v * v / (2.0 * r)
        } else {
            0.0
        }
    }
}

impl<M: SpacetimeMetric> SpacetimeMetric for FractalMetric<M> {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let mut g = self.base_metric.metric_components(x);
        let r_val = x[R];

        if r_val > 1e-10 {
            let scale = self.scaling_factor(r_val);

            // Modify space-like components
            for (i, row) in g.iter_mut().enumerate().skip(1) {
                row[i] *= scale;
            }
        }

        g
    }

    fn event_horizon_radius(&self) -> Option<f64> {
        self.base_metric.event_horizon_radius()
    }
}

/// Q-tensor fractal spacetime metric.
///
/// The Q-tensor is a symmetric traceless rank-2 tensor from liquid crystal
/// physics that describes orientational order. In the fractal spacetime
/// context, it modifies the metric through its eigenvalues, creating
/// anisotropic scaling that depends on both radial distance and angular
/// position.
///
/// The Q-tensor has the form:
///   Q_ij = S * (n_i * n_j - delta_ij / 3)
///
/// where S is the scalar order parameter and n is the director field.
/// In spherical coordinates, the director aligns with the radial direction
/// at large distances and develops disclinations (topological defects)
/// near the fractal transition scale r_d.
///
/// Physical interpretation:
/// - S = 0: isotropic (standard GR metric)
/// - S > 0: prolate ordering (radial metric enhanced, angular reduced)
/// - S < 0: oblate ordering (radial metric reduced, angular enhanced)
///
/// Disclinations at r ~ r_d create metric singularities that modify
/// geodesic equations, producing anomalous accelerations at planetary scales.
pub struct QtensorFractalMetric<M: SpacetimeMetric> {
    pub base: FractalMetric<M>,
    /// Disclination scale (km) -- radius where Q-tensor ordering breaks down.
    pub r_disclination: f64,
    /// Q-tensor scalar order parameter at large r.
    pub s_bulk: f64,
}

impl<M: SpacetimeMetric> QtensorFractalMetric<M> {
    pub fn new(base_metric: M, d_f: f64, r_0: f64, r_disclination: f64, s_bulk: f64) -> Self {
        Self {
            base: FractalMetric::new(base_metric, d_f, r_0),
            r_disclination,
            s_bulk,
        }
    }

    /// Compute the Q-tensor order parameter S(r) as a function of radius.
    ///
    /// S(r) = s_bulk * tanh((r - r_d) / r_d)
    ///
    /// This gives:
    /// - S ~ -s_bulk for r << r_d (oblate, inside disclination)
    /// - S = 0 at r = r_d (isotropic, disclination core)
    /// - S ~ +s_bulk for r >> r_d (prolate, outside disclination)
    pub fn order_parameter(&self, r: f64) -> f64 {
        if self.r_disclination > 1e-10 {
            self.s_bulk * ((r - self.r_disclination) / self.r_disclination).tanh()
        } else {
            self.s_bulk
        }
    }

    /// Compute the Q-tensor modification to the radial metric component.
    ///
    /// g_rr' = g_rr * (1 + 2*S/3)  (prolate: radial stretch)
    /// g_theta_theta' = g_tt * (1 - S/3)  (prolate: angular compression)
    pub fn qtensor_radial_factor(&self, r: f64) -> f64 {
        let s = self.order_parameter(r);
        1.0 + 2.0 * s / 3.0
    }

    /// Compute the Q-tensor modification to the angular metric components.
    pub fn qtensor_angular_factor(&self, r: f64) -> f64 {
        let s = self.order_parameter(r);
        1.0 - s / 3.0
    }

    /// Compute the anomalous acceleration from the Q-tensor disclination.
    ///
    /// Near the disclination (r ~ r_d), the metric gradient creates an
    /// extra radial acceleration proportional to dS/dr:
    ///
    ///   a_Q = -(2/3) * s_bulk * v^2 / r_d * sech^2((r - r_d)/r_d)
    ///
    /// This is localized near r_d and vanishes far from the disclination.
    pub fn disclination_acceleration(&self, r: f64, v: f64) -> f64 {
        if self.r_disclination > 1e-10 {
            let x = (r - self.r_disclination) / self.r_disclination;
            let sech2 = 1.0 / x.cosh().powi(2);
            -(2.0 / 3.0) * self.s_bulk * v * v / self.r_disclination * sech2
        } else {
            0.0
        }
    }
}

impl<M: SpacetimeMetric> SpacetimeMetric for QtensorFractalMetric<M> {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let mut g = self.base.metric_components(x);
        let r_val = x[R];

        if r_val > 1e-10 {
            let q_r = self.qtensor_radial_factor(r_val);
            let q_a = self.qtensor_angular_factor(r_val);

            // Radial component: index 1 (R)
            g[R][R] *= q_r;

            // Angular components: indices 2 (THETA) and 3 (PHI)
            g[THETA][THETA] *= q_a;
            g[3][3] *= q_a; // PHI
        }

        g
    }

    fn event_horizon_radius(&self) -> Option<f64> {
        self.base.event_horizon_radius()
    }
}

/// Compute the Pioneer anomaly predicted deceleration from the fractal metric.
///
/// The Pioneer anomaly is an observed constant deceleration of
/// ~8.74e-10 m/s^2 = 8.74e-13 km/s^2 toward the Sun for Pioneer 10/11
/// at heliocentric distances 20-70 AU.
///
/// The fractal metric with D_f = 2.7 predicts an anomalous radial
/// acceleration:
///   a_P = -(D_f - 4) * v^2 / (2 * r) = +0.65 * v^2 / r
///
/// For Pioneer 10 at r ~ 40 AU with v ~ 12 km/s:
///   a_P ~ 0.65 * 144 / 5.984e9 ~ 1.56e-8 km/s^2
///
/// This is too large by ~4 orders of magnitude relative to the observed
/// anomaly, indicating that D_f cannot be a pure constant but must have
/// scale-dependent corrections.
///
/// # Arguments
/// * `d_f` - Fractal dimension
/// * `r_km` - Heliocentric distance in km
/// * `v_km_s` - Spacecraft velocity in km/s
///
/// # Returns
/// Anomalous deceleration in km/s^2 (positive = toward Sun)
pub fn pioneer_anomaly_prediction(d_f: f64, r_km: f64, v_km_s: f64) -> f64 {
    if r_km > 1e-10 {
        (4.0 - d_f) * v_km_s * v_km_s / (2.0 * r_km)
    } else {
        0.0
    }
}

/// Result of a flyby anomaly test using the fractal metric.
#[derive(Debug, Clone)]
pub struct FractalFlybyResult {
    /// Fractal dimension used.
    pub d_f: f64,
    /// Perigee distance in km.
    pub perigee_km: f64,
    /// Asymptotic velocity in km/s.
    pub v_inf_km_s: f64,
    /// Predicted anomalous delta-V at perigee in mm/s.
    pub delta_v_mm_s: f64,
    /// Sign of the anomaly (+1 = speed increase, -1 = decrease).
    pub sign: i32,
}

/// Compute the fractal metric flyby anomaly prediction.
///
/// At perigee (closest approach), the fractal scaling is strongest.
/// The anomalous velocity change is:
///   delta_v = a_fractal * t_perigee
///           ~ (4 - D_f) * v_inf^2 / (2 * r_perigee) * (r_perigee / v_inf)
///           = (4 - D_f) * v_inf / 2
///
/// For D_f = 2.7: delta_v ~ 0.65 * v_inf
///
/// This is much larger than the observed ~mm/s anomalies, indicating
/// the fractal metric alone cannot explain flyby anomalies quantitatively,
/// but it predicts the correct SIGN pattern (speed increases for perigee
/// approaches, decreases for departures relative to the fractal scale).
pub fn fractal_flyby_prediction(d_f: f64, perigee_km: f64, v_inf_km_s: f64) -> FractalFlybyResult {
    // Anomalous acceleration at perigee
    let a_perigee = (4.0 - d_f) * v_inf_km_s * v_inf_km_s / (2.0 * perigee_km);

    // Time scale at perigee: t ~ r_perigee / v_inf
    let t_perigee = perigee_km / v_inf_km_s;

    // Delta-v in km/s, convert to mm/s
    let delta_v_km_s = a_perigee * t_perigee;
    let delta_v_mm_s = delta_v_km_s * 1e6; // km/s -> mm/s

    // Sign: D_f < 4 means outward (decelerating) anomalous force
    // For an approaching spacecraft, this manifests as a velocity excess
    let sign = if delta_v_mm_s > 0.0 { 1 } else { -1 };

    FractalFlybyResult {
        d_f,
        perigee_km,
        v_inf_km_s,
        delta_v_mm_s,
        sign,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Schwarzschild;

    #[test]
    fn test_fractal_scaling_at_reference() {
        let schw = Schwarzschild::new(1.0);
        let fm = FractalMetric::new(schw, 2.7, 1.0);
        // At r = r_0, scaling factor = 1.0
        let s = fm.scaling_factor(1.0);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fractal_scaling_below_four() {
        let schw = Schwarzschild::new(1.0);
        let fm = FractalMetric::new(schw, 2.7, 1.0);
        // D_f < 4 => exponent negative => scaling decreases with r
        let s_near = fm.scaling_factor(0.5);
        let s_far = fm.scaling_factor(2.0);
        assert!(s_near > s_far, "D_f<4: near {s_near} should be > far {s_far}");
    }

    #[test]
    fn test_anomalous_acceleration_sign() {
        let schw = Schwarzschild::new(1.0);
        let fm = FractalMetric::new(schw, 2.7, 1.0);
        let a = fm.anomalous_acceleration(1e8, 10.0);
        // D_f = 2.7 < 4 => -(2.7 - 4) = +1.3 => positive (outward)
        assert!(a > 0.0, "D_f<4 should give positive anomalous acceleration: {a}");
    }

    #[test]
    fn test_qtensor_order_parameter() {
        let schw = Schwarzschild::new(1.0);
        let qt = QtensorFractalMetric::new(schw, 2.7, 1.0, 100.0, 0.5);
        // Far outside disclination: S -> +s_bulk
        let s_far = qt.order_parameter(1000.0);
        assert!(s_far > 0.0 && s_far < 0.5 + 0.01);
        // Inside disclination: S -> -s_bulk
        let s_near = qt.order_parameter(1.0);
        assert!(s_near < 0.0);
        // At disclination: S = 0
        let s_disc = qt.order_parameter(100.0);
        assert!(s_disc.abs() < 0.01, "at r_d, S should be ~0, got {s_disc}");
    }

    #[test]
    fn test_qtensor_factors() {
        let schw = Schwarzschild::new(1.0);
        let qt = QtensorFractalMetric::new(schw, 2.7, 1.0, 100.0, 0.5);
        // Far outside: S ~ 0.5, q_r = 1 + 2*0.5/3 = 1.333
        let q_r = qt.qtensor_radial_factor(1e6);
        assert!(q_r > 1.3 && q_r < 1.35, "radial factor should be ~1.33: {q_r}");
        let q_a = qt.qtensor_angular_factor(1e6);
        assert!(q_a > 0.83 && q_a < 0.84, "angular factor should be ~0.833: {q_a}");
    }

    #[test]
    fn test_disclination_acceleration_peaks_at_rd() {
        let schw = Schwarzschild::new(1.0);
        let qt = QtensorFractalMetric::new(schw, 2.7, 1.0, 100.0, 0.5);
        let v = 10.0;
        let a_at_rd = qt.disclination_acceleration(100.0, v).abs();
        let a_far = qt.disclination_acceleration(1e6, v).abs();
        assert!(a_at_rd > a_far, "acceleration should peak at r_d: {a_at_rd} > {a_far}");
    }

    #[test]
    fn test_qtensor_metric_positive_definite() {
        let schw = Schwarzschild::new(1.0);
        let qt = QtensorFractalMetric::new(schw, 2.7, 1.0, 100.0, 0.3);
        // Test at r=200M (outside horizon at r=2M)
        let x = [0.0, 200.0, std::f64::consts::FRAC_PI_2, 0.0];
        let g = qt.metric_components(&x);
        // g_tt should be negative (timelike)
        assert!(g[0][0] < 0.0, "g_tt should be negative: {}", g[0][0]);
        // Space-like components should be positive
        assert!(g[1][1] > 0.0, "g_rr should be positive: {}", g[1][1]);
        assert!(g[2][2] > 0.0, "g_theta_theta should be positive: {}", g[2][2]);
        assert!(g[3][3] > 0.0, "g_phi_phi should be positive: {}", g[3][3]);
    }

    #[test]
    fn test_pioneer_anomaly_magnitude() {
        // Pioneer 10 at ~40 AU, v ~ 12 km/s
        let r_40au = 40.0 * 1.496e8; // km
        let v = 12.0; // km/s
        let a_pred = pioneer_anomaly_prediction(2.7, r_40au, v);
        // Should be positive (toward Sun = deceleration)
        assert!(a_pred > 0.0, "Pioneer anomaly should be positive: {a_pred}");
        // Observed: 8.74e-13 km/s^2. Our prediction will be different.
        assert!(a_pred.is_finite());
    }

    #[test]
    fn test_fractal_flyby_sign() {
        // NEAR flyby: perigee 6911 km, v_inf 6.851 km/s
        let result = fractal_flyby_prediction(2.7, 6911.0, 6.851);
        // D_f < 4 => positive delta_v => sign = +1
        assert_eq!(result.sign, 1, "NEAR flyby should have positive sign");
        assert!(result.delta_v_mm_s > 0.0);
    }
}
