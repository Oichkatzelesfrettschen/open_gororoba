//! Environmental models for flyby trajectory integration.
//!
//! Designed for static dispatch (`impl EnvironmentModel`) to avoid
//! vtable overhead in tight RK4/ABM8 inner loops.

/// Represents a state vector for a rigid body or particle.
#[derive(Debug, Clone)]
pub struct State {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}

/// Core trait for environmental models.
///
/// Implementations provide local dark matter density and anisotropy
/// at a given position and time. Generic parameters use static dispatch
/// (monomorphization) for zero-cost abstraction in the integrator.
pub trait EnvironmentModel {
    /// Computes the scalar density at a given position and time.
    fn density_scalar(&self, r: &[f64; 3], t: f64, state: &State) -> f64;

    /// Computes the environmental anisotropy/wake tensor at a given position and time.
    /// Returns a 3x3 tensor representing local dark matter / solar wind flow.
    fn anisotropy_tensor(&self, r: &[f64; 3], t: f64, state: &State) -> [[f64; 3]; 3];
}

/// Baseline Earth-only model using a simplified NFW-like 1/r^3 density profile.
///
/// Uses the same profile as dm_force.rs but evaluated in flyby coordinates (km).
/// See BIB-0302 (Navarro, Frenk & White 1996).
pub struct EarthOnlyNfwLike {
    pub base_density: f64,
    pub earth_radius_km: f64,
}

impl Default for EarthOnlyNfwLike {
    fn default() -> Self {
        Self {
            base_density: 1.0,
            earth_radius_km: 6371.0,
        }
    }
}

impl EnvironmentModel for EarthOnlyNfwLike {
    fn density_scalar(&self, r: &[f64; 3], _t: f64, _state: &State) -> f64 {
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if r_mag <= self.earth_radius_km {
            return self.base_density;
        }
        // 1/r^3 NFW-like inner cusp
        self.base_density * (self.earth_radius_km / r_mag).powi(3)
    }

    fn anisotropy_tensor(&self, _r: &[f64; 3], _t: f64, _state: &State) -> [[f64; 3]; 3] {
        // Isotropic baseline: identity tensor
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

/// Model incorporating Earth's gravitational focusing wake.
///
/// Density: rho_0 * (R_E/r)^3 * (1 + eta_wake * cos(theta))
/// where theta is the angle between position and DM wind direction.
/// See BIB-0303 (Lundberg & Edsjo 2004), BIB-0304 (Lee et al. 2021).
pub struct EarthWakeModel {
    pub base_density: f64,
    pub earth_radius_km: f64,
    /// DM wind direction in J2000 ECI (unit vector, km/s normalized).
    pub wind_direction: [f64; 3],
    /// Wake amplitude (0.0 = no wake, 0.10 = 10% enhancement downstream).
    pub eta_wake: f64,
}

impl EnvironmentModel for EarthWakeModel {
    fn density_scalar(&self, r: &[f64; 3], _t: f64, _state: &State) -> f64 {
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if r_mag <= self.earth_radius_km {
            return self.base_density;
        }
        let base = self.base_density * (self.earth_radius_km / r_mag).powi(3);
        if self.eta_wake == 0.0 {
            return base;
        }
        let w_mag = (self.wind_direction[0] * self.wind_direction[0]
            + self.wind_direction[1] * self.wind_direction[1]
            + self.wind_direction[2] * self.wind_direction[2])
            .sqrt();
        if w_mag < 1e-10 {
            return base;
        }
        let cos_wind = (r[0] * self.wind_direction[0]
            + r[1] * self.wind_direction[1]
            + r[2] * self.wind_direction[2])
            / (r_mag * w_mag);
        base * (1.0 + self.eta_wake * cos_wind)
    }

    fn anisotropy_tensor(&self, r: &[f64; 3], _t: f64, _state: &State) -> [[f64; 3]; 3] {
        // Wake-induced anisotropy: outer product of wind direction
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if r_mag <= self.earth_radius_km || self.eta_wake == 0.0 {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let w_mag = (self.wind_direction[0] * self.wind_direction[0]
            + self.wind_direction[1] * self.wind_direction[1]
            + self.wind_direction[2] * self.wind_direction[2])
            .sqrt();
        if w_mag < 1e-10 {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let w = [
            self.wind_direction[0] / w_mag,
            self.wind_direction[1] / w_mag,
            self.wind_direction[2] / w_mag,
        ];
        // Identity + eta * (w_i * w_j) wake tensor
        [
            [1.0 + self.eta_wake * w[0] * w[0], self.eta_wake * w[0] * w[1], self.eta_wake * w[0] * w[2]],
            [self.eta_wake * w[1] * w[0], 1.0 + self.eta_wake * w[1] * w[1], self.eta_wake * w[1] * w[2]],
            [self.eta_wake * w[2] * w[0], self.eta_wake * w[2] * w[1], 1.0 + self.eta_wake * w[2] * w[2]],
        ]
    }
}

/// Advanced model for heliopause distortions and dark matter "bubbles"
/// driven by multi-body (Earth-Moon-Sun) interactions.
///
/// WARNING: This model is speculative. Methods require derivations from
/// the theoretical framework in BIB-0305, BIB-0306.
pub struct SolarWindHeliosphericTensorModel {
    pub solar_wind_speed_km_s: f64,
    pub dark_matter_cross_section: f64,
}

impl EnvironmentModel for SolarWindHeliosphericTensorModel {
    fn density_scalar(&self, _r: &[f64; 3], _t: f64, _state: &State) -> f64 {
        // TODO: Implement heliospheric DM density from Boltzmann transport
        // equation coupled to solar wind MHD. Requires BIB-0305/BIB-0306
        // cross-section bounds. For now, return uniform background.
        1.0
    }

    fn anisotropy_tensor(&self, _r: &[f64; 3], _t: f64, _state: &State) -> [[f64; 3]; 3] {
        // TODO: Implement heliospheric tensor from Parker spiral + DM wind
        // interaction. Requires coupling to lbm_3d MHD solver output.
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfw_at_surface_is_base() {
        let model = EarthOnlyNfwLike::default();
        let state = State {
            position: [6371.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let rho = model.density_scalar(&state.position, 0.0, &state);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn nfw_decreases_with_altitude() {
        let model = EarthOnlyNfwLike::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let rho_low = model.density_scalar(&[7000.0, 0.0, 0.0], 0.0, &state);
        let rho_high = model.density_scalar(&[50000.0, 0.0, 0.0], 0.0, &state);
        assert!(rho_low > rho_high);
    }

    #[test]
    fn nfw_r_cubed_scaling() {
        let model = EarthOnlyNfwLike::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let r1 = 10000.0;
        let r2 = 20000.0;
        let rho1 = model.density_scalar(&[r1, 0.0, 0.0], 0.0, &state);
        let rho2 = model.density_scalar(&[r2, 0.0, 0.0], 0.0, &state);
        let ratio = rho1 / rho2;
        let expected = (r2 / r1).powi(3);
        assert!((ratio - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn wake_model_symmetric_at_zero_eta() {
        let model = EarthWakeModel {
            base_density: 1.0,
            earth_radius_km: 6371.0,
            wind_direction: [1.0, 0.0, 0.0],
            eta_wake: 0.0,
        };
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let rho_pos = model.density_scalar(&[10000.0, 0.0, 0.0], 0.0, &state);
        let rho_neg = model.density_scalar(&[-10000.0, 0.0, 0.0], 0.0, &state);
        assert!((rho_pos - rho_neg).abs() < 1e-10);
    }

    #[test]
    fn wake_model_asymmetric_with_eta() {
        let model = EarthWakeModel {
            base_density: 1.0,
            earth_radius_km: 6371.0,
            wind_direction: [1.0, 0.0, 0.0],
            eta_wake: 0.10,
        };
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        // Downstream (positive x = aligned with wind)
        let rho_down = model.density_scalar(&[10000.0, 0.0, 0.0], 0.0, &state);
        // Upstream (negative x = against wind)
        let rho_up = model.density_scalar(&[-10000.0, 0.0, 0.0], 0.0, &state);
        assert!(rho_down > rho_up, "Downstream density should exceed upstream");
    }

    #[test]
    fn wake_anisotropy_tensor_positive_definite() {
        let model = EarthWakeModel {
            base_density: 1.0,
            earth_radius_km: 6371.0,
            wind_direction: [1.0, 0.0, 0.0],
            eta_wake: 0.10,
        };
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let t = model.anisotropy_tensor(&[10000.0, 0.0, 0.0], 0.0, &state);
        // Diagonal elements should be positive
        assert!(t[0][0] > 0.0);
        assert!(t[1][1] > 0.0);
        assert!(t[2][2] > 0.0);
    }

    #[test]
    fn heliospheric_model_returns_defaults() {
        let model = SolarWindHeliosphericTensorModel {
            solar_wind_speed_km_s: 400.0,
            dark_matter_cross_section: 1e-45,
        };
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let rho = model.density_scalar(&[10000.0, 0.0, 0.0], 0.0, &state);
        assert!((rho - 1.0).abs() < 1e-10, "Stub should return 1.0");
    }

    #[test]
    fn all_models_satisfy_trait_bounds() {
        fn assert_env_model<E: EnvironmentModel>(_: &E) {}
        assert_env_model(&EarthOnlyNfwLike::default());
        assert_env_model(&EarthWakeModel {
            base_density: 1.0,
            earth_radius_km: 6371.0,
            wind_direction: [1.0, 0.0, 0.0],
            eta_wake: 0.10,
        });
        assert_env_model(&SolarWindHeliosphericTensorModel {
            solar_wind_speed_km_s: 400.0,
            dark_matter_cross_section: 1e-45,
        });
    }
}
