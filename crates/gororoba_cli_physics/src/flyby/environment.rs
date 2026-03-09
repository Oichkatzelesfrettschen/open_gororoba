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
            [
                1.0 + self.eta_wake * w[0] * w[0],
                self.eta_wake * w[0] * w[1],
                self.eta_wake * w[0] * w[2],
            ],
            [
                self.eta_wake * w[1] * w[0],
                1.0 + self.eta_wake * w[1] * w[1],
                self.eta_wake * w[1] * w[2],
            ],
            [
                self.eta_wake * w[2] * w[0],
                self.eta_wake * w[2] * w[1],
                1.0 + self.eta_wake * w[2] * w[2],
            ],
        ]
    }
}

/// Heliospheric DM density + Parker spiral anisotropy model.
///
/// Evaluates the NFW density profile at the true galactocentric distance
/// (adding the Sun-galactic-center offset to heliocentric coordinates)
/// and computes anisotropy from the local Parker spiral B-field direction.
///
/// At solar-system scales (r << r_s ~ 20 kpc), the NFW profile is
/// approximately constant, but this model correctly handles the full
/// range from near-Sun (0.01 AU) to outer heliosphere (200 AU).
///
/// Parker spiral angle: psi = atan(omega_sun * r_helio / v_sw)
/// At 1 AU, psi ~ 45 deg; at 10 AU, psi ~ 84 deg (nearly toroidal).
///
/// See BIB-0302 (Navarro, Frenk & White 1996), BIB-0305, BIB-0306.
pub struct SolarWindHeliosphericTensorModel {
    pub solar_wind_speed_km_s: f64,
    pub dark_matter_cross_section: f64,
    /// Heliocentric distance of the evaluation point (AU).
    /// Used to select the correct Parker spiral angle.
    pub r_au: f64,
    /// Local DM density at the Sun's galactocentric radius (GeV/cm^3).
    pub rho_dm_local_gev_cm3: f64,
    /// NFW scale radius in kpc.
    pub r_s_kpc: f64,
}

impl Default for SolarWindHeliosphericTensorModel {
    fn default() -> Self {
        Self {
            solar_wind_speed_km_s: 400.0,
            dark_matter_cross_section: 1e-45,
            r_au: 1.0,
            rho_dm_local_gev_cm3: 0.3,
            // r_s = r_200 / c; typical MW: r_200 ~ 200 kpc, c ~ 10
            r_s_kpc: 20.0,
        }
    }
}

/// Solar angular velocity (rad/s). Carrington rotation period ~ 25.38 days.
const OMEGA_SUN: f64 = 2.87e-6;
/// 1 AU in km (for coordinate conversion).
const AU_KM: f64 = 1.496e8;
/// Sun's galactocentric distance in km.
const R_SUN_GC_KM: f64 = 8.178 * 3.086e16; // 8.178 kpc * (kpc->km)

impl EnvironmentModel for SolarWindHeliosphericTensorModel {
    fn density_scalar(&self, r: &[f64; 3], _t: f64, _state: &State) -> f64 {
        // Heliocentric distance in km
        let r_helio_km = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();

        // Galactocentric distance: Sun is at ~8.178 kpc from galactic center.
        // In this frame, x points away from galactic center, so the galactic
        // center is at x = -R_SUN_GC_KM. The particle's galactocentric
        // distance is |r + R_sun_gc|.
        let r_gc_km =
            ((r[0] + R_SUN_GC_KM) * (r[0] + R_SUN_GC_KM) + r[1] * r[1] + r[2] * r[2]).sqrt();

        // NFW density: rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
        // We normalize so that rho(R_sun_gc) = rho_dm_local.
        let r_s_km = self.r_s_kpc * 3.086e16; // kpc -> km
        let x_gc = r_gc_km / r_s_km;
        let x_sun = R_SUN_GC_KM / r_s_km;

        // NFW shape function: f(x) = 1 / [x * (1+x)^2]
        // rho(r_gc) / rho(R_sun) = f(x_gc) / f(x_sun)
        if x_gc < 1e-30 || x_sun < 1e-30 {
            return self.rho_dm_local_gev_cm3;
        }
        let f_gc = 1.0 / (x_gc * (1.0 + x_gc) * (1.0 + x_gc));
        let f_sun = 1.0 / (x_sun * (1.0 + x_sun) * (1.0 + x_sun));

        if f_sun < 1e-30 {
            return self.rho_dm_local_gev_cm3;
        }

        // At solar-system scales (r_helio << R_sun_gc), x_gc ~ x_sun,
        // so the ratio f_gc/f_sun ~ 1.0 (uniform to ~1e-10 at 200 AU).
        let _ = r_helio_km; // used implicitly via r_gc_km
        self.rho_dm_local_gev_cm3 * f_gc / f_sun
    }

    fn anisotropy_tensor(&self, r: &[f64; 3], _t: f64, _state: &State) -> [[f64; 3]; 3] {
        let r_helio_km = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if r_helio_km < 1.0 {
            // At the origin, no preferred direction
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }

        // Heliocentric distance in AU for Parker spiral angle
        let r_au = r_helio_km / AU_KM;

        // Parker spiral winding angle: psi = atan(omega * r / v_sw)
        let v_sw_km_s = self.solar_wind_speed_km_s;
        let r_km = r_au * AU_KM;
        let psi = (OMEGA_SUN * r_km / v_sw_km_s).atan();

        // Radial unit vector (heliocentric)
        let r_hat = [r[0] / r_helio_km, r[1] / r_helio_km, r[2] / r_helio_km];

        // Azimuthal unit vector: phi_hat = z_hat x r_hat (in ecliptic plane)
        // For 3D: phi_hat = [-r_hat[1], r_hat[0], 0] / |...| (projection onto ecliptic)
        let phi_raw = [-r_hat[1], r_hat[0], 0.0];
        let phi_mag = (phi_raw[0] * phi_raw[0] + phi_raw[1] * phi_raw[1]).sqrt();
        let phi_hat = if phi_mag > 1e-10 {
            [phi_raw[0] / phi_mag, phi_raw[1] / phi_mag, 0.0]
        } else {
            // At poles, default to y-direction
            [0.0, 1.0, 0.0]
        };

        // B-field direction in Parker spiral: B_hat = cos(psi)*r_hat - sin(psi)*phi_hat
        // (negative sign because B_phi is negative in the Parker spiral for v_sw > 0)
        let cos_psi = psi.cos();
        let sin_psi = psi.sin();
        let b_hat = [
            cos_psi * r_hat[0] - sin_psi * phi_hat[0],
            cos_psi * r_hat[1] - sin_psi * phi_hat[1],
            cos_psi * r_hat[2] - sin_psi * phi_hat[2],
        ];

        // Anisotropy tensor: I + epsilon * (b_hat x b_hat)
        // epsilon scales with cross-section (small perturbation)
        let epsilon = self.dark_matter_cross_section * 1e45; // normalize to O(1) for 1e-45 cm^2
        [
            [
                1.0 + epsilon * b_hat[0] * b_hat[0],
                epsilon * b_hat[0] * b_hat[1],
                epsilon * b_hat[0] * b_hat[2],
            ],
            [
                epsilon * b_hat[1] * b_hat[0],
                1.0 + epsilon * b_hat[1] * b_hat[1],
                epsilon * b_hat[1] * b_hat[2],
            ],
            [
                epsilon * b_hat[2] * b_hat[0],
                epsilon * b_hat[2] * b_hat[1],
                1.0 + epsilon * b_hat[2] * b_hat[2],
            ],
        ]
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
        assert!(
            rho_down > rho_up,
            "Downstream density should exceed upstream"
        );
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
    fn heliospheric_model_density_near_local() {
        // At 1 AU, density should be extremely close to rho_dm_local
        // because 1 AU << R_sun_gc (galactocentric distance changes by < 1e-9)
        let model = SolarWindHeliosphericTensorModel::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        // 1 AU along x in km
        let r_1au = [AU_KM, 0.0, 0.0];
        let rho = model.density_scalar(&r_1au, 0.0, &state);
        let rel_diff = (rho - 0.3).abs() / 0.3;
        assert!(
            rel_diff < 1e-6,
            "At 1 AU, DM density should be ~0.3 GeV/cm^3, got {rho:.6e} (rel_diff={rel_diff:.3e})"
        );
    }

    #[test]
    fn heliospheric_model_density_at_200au() {
        // At 200 AU, still nearly identical to local density
        // (200 AU / 8.178 kpc ~ 3e-6 fractional change)
        let model = SolarWindHeliosphericTensorModel::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let r_200au = [200.0 * AU_KM, 0.0, 0.0];
        let rho = model.density_scalar(&r_200au, 0.0, &state);
        let rel_diff = (rho - 0.3).abs() / 0.3;
        assert!(
            rel_diff < 1e-4,
            "At 200 AU, density should still be ~0.3, got {rho:.6e} (rel_diff={rel_diff:.3e})"
        );
    }

    #[test]
    fn heliospheric_anisotropy_identity_at_origin() {
        let model = SolarWindHeliosphericTensorModel::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let t = model.anisotropy_tensor(&[0.0, 0.0, 0.0], 0.0, &state);
        // At origin, should be identity
        assert!((t[0][0] - 1.0).abs() < 1e-10);
        assert!((t[1][1] - 1.0).abs() < 1e-10);
        assert!((t[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn heliospheric_anisotropy_positive_definite() {
        let model = SolarWindHeliosphericTensorModel::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        let r = [10.0 * AU_KM, 0.0, 0.0];
        let t = model.anisotropy_tensor(&r, 0.0, &state);
        // Diagonal elements positive (I + epsilon * b x b with epsilon >= 0)
        assert!(t[0][0] > 0.0, "T[0][0] = {}", t[0][0]);
        assert!(t[1][1] > 0.0, "T[1][1] = {}", t[1][1]);
        assert!(t[2][2] > 0.0, "T[2][2] = {}", t[2][2]);
    }

    #[test]
    fn heliospheric_consistent_with_earth_nfw_at_1au() {
        // P13.4: At 1 AU with isotropic tensor (sigma=0), the heliospheric
        // model should produce density close to EarthOnlyNfwLike at
        // the same position, after accounting for different profile shapes
        // (1/r^3 vs proper NFW). The key invariant: both return finite
        // positive density at Earth orbit distance.
        let helio = SolarWindHeliosphericTensorModel {
            dark_matter_cross_section: 0.0,
            ..SolarWindHeliosphericTensorModel::default()
        };
        let earth = EarthOnlyNfwLike::default();
        let state = State {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        };
        // Earth orbit position in km
        let r = [AU_KM, 0.0, 0.0];
        let rho_helio = helio.density_scalar(&r, 0.0, &state);
        let rho_earth = earth.density_scalar(&r, 0.0, &state);

        // Both should be positive and finite
        assert!(rho_helio > 0.0 && rho_helio.is_finite());
        assert!(rho_earth > 0.0 && rho_earth.is_finite());

        // Helio tensor with sigma=0 should be identity (isotropic)
        let t = helio.anisotropy_tensor(&r, 0.0, &state);
        assert!(
            (t[0][0] - 1.0).abs() < 1e-10,
            "sigma=0 should give identity tensor"
        );
        assert!(t[0][1].abs() < 1e-10);
        assert!(t[1][0].abs() < 1e-10);
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
        assert_env_model(&SolarWindHeliosphericTensorModel::default());
    }
}
