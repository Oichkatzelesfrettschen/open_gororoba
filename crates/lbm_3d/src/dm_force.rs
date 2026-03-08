//! Dark matter gravitational force field for LBM coupling.
//!
//! Computes the gravitational acceleration from an NFW dark matter halo
//! profile on the LBM grid. The force field is static (precomputed once)
//! and added to the Lorentz force via `combine_forces()` before passing
//! to `LbmSolver3D::set_force_field()`.
//!
//! Physics: at 1 AU with canonical local DM density (0.3 GeV/cm^3) and
//! MW virial mass (1e12 Msun), the DM gravitational perturbation on solar
//! wind is delta_rho/rho < 10^-15 -- a rigorous null result confirming
//! the 15-order-of-magnitude gap between DM gravity and Lorentz forces.

use crate::boundary::GridIndex;

/// Gravitational constant in SI (m^3 kg^-1 s^-2).
const G_SI: f64 = 6.674e-11;

/// Solar mass in kg.
const M_SUN_KG: f64 = 1.989e30;

/// 1 AU in meters.
const AU_M: f64 = 1.496e11;

/// GeV/cm^3 to kg/m^3 conversion: 1 GeV/c^2 = 1.783e-27 kg, 1 cm^3 = 1e-6 m^3.
const GEV_CM3_TO_KG_M3: f64 = 1.783e-21;

/// Configuration for the dark matter gravitational force field.
#[derive(Clone, Debug)]
pub struct DmForceConfig {
    /// Local DM density at the Sun's galactocentric radius (GeV/cm^3).
    pub rho_dm_local_gev_cm3: f64,
    /// Milky Way virial mass in solar masses.
    pub m200_solar: f64,
    /// NFW concentration parameter.
    pub c200: f64,
    /// DM wind velocity in simulation frame (lattice units). Reserved for future use.
    pub v_dm_wind: [f64; 3],
    /// Gravitational focusing wake amplitude (0.0 = isotropic, >0 = upstream enhancement).
    pub eta_wake: f64,
    /// Non-dimensionalization factor: physical acceleration (m/s^2) -> LBM lattice force.
    /// Computed as delta_t^2 / delta_x where delta_x and delta_t are the LBM
    /// lattice spacing and timestep in physical units.
    pub force_scale: f64,
}

impl Default for DmForceConfig {
    fn default() -> Self {
        // Default non-dimensionalization for nx=128 grid spanning 1 AU slab
        // with v_sw = 400 km/s mapped to lattice speed 0.05:
        //   delta_x = 1 AU / 128 ~ 1.169e9 m
        //   delta_t = delta_x * (0.05 / 400e3) ~ 146.1 s
        //   force_scale = delta_t^2 / delta_x ~ 1.826e-5
        let delta_x = AU_M / 128.0;
        let v_sw_phys = 400.0e3; // m/s
        let v_sw_lattice = 0.05;
        let delta_t = delta_x * (v_sw_lattice / v_sw_phys);
        let force_scale = delta_t * delta_t / delta_x;

        Self {
            rho_dm_local_gev_cm3: 0.3,
            m200_solar: 1.0e12,
            c200: 10.0,
            v_dm_wind: [0.0, 0.0, 0.0],
            eta_wake: 0.0,
            force_scale,
        }
    }
}

/// NFW profile helper: compute enclosed mass M(<r) for given r.
///
/// NFW density: rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
/// Enclosed mass: M(<r) = 4 pi rho_s r_s^3 [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
///
/// We express this as M(<r) = M_200 * f(r/r_s) / f(c), where
/// f(x) = ln(1+x) - x/(1+x) and r_s = r_200/c.
fn nfw_enclosed_mass(r_phys: f64, m200_kg: f64, r_200: f64, c: f64) -> f64 {
    let r_s = r_200 / c;
    let x = r_phys / r_s;

    // Use ln_1p(x) for numerical stability when x << 1 (solar system scales:
    // 1 AU / r_s ~ 2.4e-10). Plain (1+x).ln() catastrophically cancels.
    let f_x = x.ln_1p() - x / (1.0 + x);
    let f_c = c.ln_1p() - c / (1.0 + c);

    if f_c.abs() < 1e-30 {
        return 0.0;
    }
    m200_kg * f_x / f_c
}

/// Precomputed dark matter gravitational force field on the LBM grid.
pub struct DmForceField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Gravitational force per grid cell in LBM lattice units, [fx, fy, fz].
    pub force: Vec<[f64; 3]>,
    /// DM density at each grid cell (kg/m^3), for diagnostic output.
    pub dm_density: Vec<f64>,
    /// Configuration used to generate this field.
    pub config: DmForceConfig,
}

impl DmForceField {
    /// Create and precompute the static DM gravitational force field.
    ///
    /// Grid mapping follows the MHD Parker spiral convention:
    /// - x is the radial direction (Sun -> outward)
    /// - y, z are transverse
    /// - r_0 = nx/2 corresponds to 1 AU
    /// - Physical radius: r_phys = (x / r0) * 1 AU
    pub fn new(nx: usize, ny: usize, nz: usize, config: DmForceConfig) -> Self {
        let n = nx * ny * nz;
        let mut force = vec![[0.0; 3]; n];
        let mut dm_density = vec![0.0; n];

        // NFW parameters in SI
        let m200_kg = config.m200_solar * M_SUN_KG;
        // Virial radius: r_200 ~ 200 kpc for MW. We derive from M_200:
        // M_200 = (4/3) pi r_200^3 * 200 * rho_crit
        // rho_crit ~ 9.47e-27 kg/m^3 (H0=67.4 km/s/Mpc)
        let rho_crit = 9.47e-27; // kg/m^3
        let r_200 = (3.0 * m200_kg / (4.0 * std::f64::consts::PI * 200.0 * rho_crit)).cbrt();

        // Local DM density in SI
        let rho_dm_si = config.rho_dm_local_gev_cm3 * GEV_CM3_TO_KG_M3;

        // Grid reference point: x = nx/2 corresponds to 1 AU
        let r0 = nx as f64 / 2.0;
        let y_center = ny as f64 / 2.0;
        let z_center = nz as f64 / 2.0;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = GridIndex::new(x, y, z).linearize(nx, ny);

                    // Map grid to physical coordinates
                    // Radial distance from Sun (along x)
                    let dx = x as f64 - 0.0; // Sun is at x=0
                    let dy = y as f64 - y_center;
                    let dz = z as f64 - z_center;
                    let r_grid = (dx * dx + dy * dy + dz * dz).sqrt().max(0.5);
                    let r_phys = (r_grid / r0) * AU_M;

                    // NFW enclosed mass at this radius
                    let m_enc = nfw_enclosed_mass(r_phys, m200_kg, r_200, config.c200);

                    // Gravitational acceleration: a = -G M(<r) / r^2
                    let a_phys = G_SI * m_enc / (r_phys * r_phys);

                    // Radial unit vector (pointing away from Sun at x=0)
                    let rhat = [dx / r_grid, dy / r_grid, dz / r_grid];

                    // Wake modulation: 1 + eta * cos(angle to DM wind direction)
                    let wake_factor = if config.eta_wake > 0.0
                        && config.v_dm_wind.iter().any(|&v| v.abs() > 1e-30)
                    {
                        let wind_mag = (config.v_dm_wind[0] * config.v_dm_wind[0]
                            + config.v_dm_wind[1] * config.v_dm_wind[1]
                            + config.v_dm_wind[2] * config.v_dm_wind[2])
                        .sqrt();
                        let cos_angle = (rhat[0] * config.v_dm_wind[0]
                            + rhat[1] * config.v_dm_wind[1]
                            + rhat[2] * config.v_dm_wind[2])
                            / wind_mag;
                        1.0 + config.eta_wake * cos_angle
                    } else {
                        1.0
                    };

                    // Force in LBM units (gravitational pull toward center, hence negative)
                    let a_lattice = a_phys * config.force_scale * wake_factor;
                    force[idx] = [
                        -a_lattice * rhat[0],
                        -a_lattice * rhat[1],
                        -a_lattice * rhat[2],
                    ];

                    // Store DM density (uniform in NFW at solar system scales)
                    dm_density[idx] = rho_dm_si;
                }
            }
        }

        Self {
            nx,
            ny,
            nz,
            force,
            dm_density,
            config,
        }
    }

    /// Get force at a linearized grid index.
    pub fn force_at(&self, idx: usize) -> [f64; 3] {
        self.force[idx]
    }

    /// Get DM density at a linearized grid index.
    pub fn dm_density_at(&self, idx: usize) -> f64 {
        self.dm_density[idx]
    }

    /// Maximum force magnitude across the grid.
    pub fn max_force_magnitude(&self) -> f64 {
        self.force
            .iter()
            .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
            .fold(0.0_f64, f64::max)
    }
}

/// Combine two force fields by element-wise vector addition.
///
/// Used to sum Lorentz + DM gravitational forces before passing to
/// `LbmSolver3D::set_force_field()`.
pub fn combine_forces(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f64; 3]> {
    debug_assert_eq!(a.len(), b.len(), "force field length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(fa, fb)| [fa[0] + fb[0], fa[1] + fb[1], fa[2] + fb[2]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_field(nx: usize, ny: usize, nz: usize) -> DmForceField {
        DmForceField::new(nx, ny, nz, DmForceConfig::default())
    }

    #[test]
    fn test_creation() {
        let field = default_field(16, 8, 8);
        assert_eq!(field.force.len(), 16 * 8 * 8);
        assert_eq!(field.dm_density.len(), 16 * 8 * 8);
        // All forces should be finite
        for f in &field.force {
            assert!(f[0].is_finite(), "non-finite fx");
            assert!(f[1].is_finite(), "non-finite fy");
            assert!(f[2].is_finite(), "non-finite fz");
        }
    }

    #[test]
    fn test_radial_symmetry() {
        // Two points equidistant from center (in y/z) at same x should have
        // equal force magnitude
        let field = default_field(16, 16, 16);
        let y_center = 8.0;
        let z_center = 8.0;

        // Point A: (8, 4, 8) -- offset -4 in y
        let idx_a = GridIndex::new(8, 4, 8).linearize(16, 16);
        // Point B: (8, 12, 8) -- offset +4 in y
        let idx_b = GridIndex::new(8, 12, 8).linearize(16, 16);

        // They have different x-offsets from Sun (at x=0), so not exact symmetry,
        // but the transverse components should be equal in magnitude and opposite in sign
        let _ = (y_center, z_center);
        assert!(
            (field.force[idx_a][1] + field.force[idx_b][1]).abs() < 1e-30,
            "y-forces should be opposite: {} vs {}",
            field.force[idx_a][1],
            field.force[idx_b][1]
        );
    }

    #[test]
    fn test_r_dependence() {
        // At solar system scales (r << r_s ~ 20 kpc), the NFW enclosed mass
        // scales as M(<r) ~ r^2 (from Taylor expansion of f(x) ~ x^2/2).
        // Therefore F = GM/r^2 ~ r^2/r^2 = const. The force magnitude should
        // be approximately equal at different radii along the x-axis.
        let field = default_field(32, 8, 8);
        let y_mid = 4;
        let z_mid = 4;

        // Near Sun: x=4 -> r_phys = (4/16)*AU = 0.25 AU
        let idx_near = GridIndex::new(4, y_mid, z_mid).linearize(32, 8);
        let mag_near = (field.force[idx_near][0].powi(2)
            + field.force[idx_near][1].powi(2)
            + field.force[idx_near][2].powi(2))
        .sqrt();

        // Far from Sun: x=28 -> r_phys = (28/16)*AU = 1.75 AU
        let idx_far = GridIndex::new(28, y_mid, z_mid).linearize(32, 8);
        let mag_far = (field.force[idx_far][0].powi(2)
            + field.force[idx_far][1].powi(2)
            + field.force[idx_far][2].powi(2))
        .sqrt();

        // Both should be nonzero and within an order of magnitude of each other
        assert!(mag_near > 0.0 && mag_far > 0.0);
        let ratio = mag_near / mag_far;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "NFW force approximately constant at sub-kpc: near={mag_near:.3e}, far={mag_far:.3e}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_center_singularity_handled() {
        // The grid point closest to Sun (x=0, y=ny/2, z=nz/2) should not
        // produce NaN or Inf due to the r.max(0.5) guard
        let field = default_field(16, 8, 8);
        let idx = GridIndex::new(0, 4, 4).linearize(16, 8);
        let f = field.force[idx];
        assert!(f[0].is_finite());
        assert!(f[1].is_finite());
        assert!(f[2].is_finite());
    }

    #[test]
    fn test_wake_asymmetry() {
        // With eta_wake > 0 and DM wind along +x, upstream (negative x direction)
        // should have enhanced force relative to downstream
        let config_wake = DmForceConfig {
            eta_wake: 0.5,
            v_dm_wind: [0.05, 0.0, 0.0],
            ..DmForceConfig::default()
        };
        let field_wake = DmForceField::new(32, 8, 8, config_wake);

        let config_iso = DmForceConfig {
            eta_wake: 0.0,
            ..DmForceConfig::default()
        };
        let field_iso = DmForceField::new(32, 8, 8, config_iso);

        // At midpoint x=16, the wake should modify the force compared to isotropic
        let idx = GridIndex::new(16, 4, 4).linearize(32, 8);
        let mag_wake = (field_wake.force[idx][0].powi(2)
            + field_wake.force[idx][1].powi(2)
            + field_wake.force[idx][2].powi(2))
        .sqrt();
        let mag_iso = (field_iso.force[idx][0].powi(2)
            + field_iso.force[idx][1].powi(2)
            + field_iso.force[idx][2].powi(2))
        .sqrt();

        // With wind along +x and rhat mostly along +x at this point,
        // cos_angle > 0, so wake_factor > 1, so mag_wake > mag_iso
        assert!(
            mag_wake > mag_iso,
            "wake should enhance force: wake={mag_wake:.3e}, iso={mag_iso:.3e}"
        );
    }

    #[test]
    fn test_combine_forces() {
        let a = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = vec![[0.1, 0.2, 0.3], [-1.0, -2.0, -3.0]];
        let c = combine_forces(&a, &b);

        assert!((c[0][0] - 1.1).abs() < 1e-14);
        assert!((c[0][1] - 2.2).abs() < 1e-14);
        assert!((c[0][2] - 3.3).abs() < 1e-14);
        assert!((c[1][0] - 3.0).abs() < 1e-14);
        assert!((c[1][1] - 3.0).abs() < 1e-14);
        assert!((c[1][2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_nfw_consistency() {
        // At r -> 0, enclosed mass -> 0
        let m200_kg = 1.0e12 * M_SUN_KG;
        let rho_crit = 9.47e-27;
        let r_200 = (3.0 * m200_kg / (4.0 * std::f64::consts::PI * 200.0 * rho_crit)).cbrt();
        let c = 10.0;

        let m_small = nfw_enclosed_mass(1.0, m200_kg, r_200, c);
        let m_large = nfw_enclosed_mass(r_200, m200_kg, r_200, c);

        // At r = r_200, enclosed mass should equal M_200
        assert!(
            (m_large - m200_kg).abs() / m200_kg < 1e-10,
            "M(<r_200) should equal M_200: got {m_large:.6e}, expected {m200_kg:.6e}"
        );

        // At tiny r, mass should be much smaller
        assert!(
            m_small < m_large * 1e-10,
            "M(<1m) should be tiny: {m_small:.6e}"
        );
    }

    #[test]
    fn test_force_scale_order_of_magnitude() {
        // DM gravitational acceleration at 1 AU: a ~ G * M_DM(<1AU) / r^2.
        // With corrected NFW (ln_1p), a ~ 1.1e-10 m/s^2 (constant across grid).
        // In lattice units: a_lattice = a_phys * force_scale ~ 1.1e-10 * 1.8e-5 ~ 2e-15.
        // This is negligible compared to LBM forcing (Lorentz ~ O(1e-3)).
        let field = default_field(32, 8, 8);
        let max_f = field.max_force_magnitude();

        // Should be extremely small but nonzero
        assert!(max_f > 0.0, "force should be nonzero");
        assert!(
            max_f < 1e-10,
            "DM force at solar system scales should be negligible: {max_f:.3e}"
        );
        assert!(
            max_f > 1e-20,
            "force should be well above f64 noise: {max_f:.3e}"
        );
    }
}
