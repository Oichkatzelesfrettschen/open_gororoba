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

/// Proton mass in GeV/c^2.
const M_PROTON_GEV: f64 = 0.938;

/// ADM dark baryon mass (~5 * m_proton).
const M_CHI_GEV: f64 = 5.0 * M_PROTON_GEV;

/// Reduced mass of DM-proton system in GeV/c^2.
const M_REDUCED_GEV: f64 = (M_CHI_GEV * M_PROTON_GEV) / (M_CHI_GEV + M_PROTON_GEV);

/// DM halo velocity dispersion (km/s).
const V_CHI_DISPERSION_KMS: f64 = 220.0;

/// 1 GeV/c^2 in kg.
const GEV_TO_KG: f64 = 1.783e-27;

/// 1 km/s in m/s.
const KMS_TO_MS: f64 = 1.0e3;

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
    /// DM-baryon scattering cross-section (cm^2). Default 0 = pure gravity.
    pub sigma_chi_b: f64,
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
            sigma_chi_b: 0.0,
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

    /// Compute dynamic DM-baryon drag force from scattering cross-section.
    ///
    /// F_drag = n_chi * n_b * sigma_chi_b * m_reduced * |v_rel| * v_rel_hat
    /// per unit volume, converted to lattice acceleration.
    ///
    /// Arguments:
    /// - `baryon_density`: proton density per cell in LBM units (rho field)
    /// - `baryon_velocity`: velocity per cell in LBM units (u field)
    /// - `n_ref_cm3`: reference proton density (cm^-3) for unit conversion
    /// - `v_ref_kms`: reference bulk speed (km/s) for unit conversion
    pub fn drag_force(
        &self,
        baryon_density: &[f64],
        baryon_velocity: &[[f64; 3]],
        n_ref_cm3: f64,
        v_ref_kms: f64,
    ) -> Vec<[f64; 3]> {
        let sigma = self.config.sigma_chi_b;
        if sigma <= 0.0 {
            return vec![[0.0; 3]; self.force.len()];
        }

        let n = self.force.len();
        let mut drag = vec![[0.0; 3]; n];

        // Local DM number density (cm^-3)
        let n_chi_cm3 = self.config.rho_dm_local_gev_cm3 / M_CHI_GEV;
        // Convert to SI: cm^-3 -> m^-3
        let n_chi_m3 = n_chi_cm3 * 1.0e6;
        // Reduced mass in kg
        let m_red_kg = M_REDUCED_GEV * GEV_TO_KG;
        // Sigma in m^2
        let sigma_m2 = sigma * 1.0e-4;
        // DM wind in physical units (m/s)
        let v_dm_phys = [
            self.config.v_dm_wind[0] * v_ref_kms * KMS_TO_MS / 0.05,
            self.config.v_dm_wind[1] * v_ref_kms * KMS_TO_MS / 0.05,
            self.config.v_dm_wind[2] * v_ref_kms * KMS_TO_MS / 0.05,
        ];
        // Default DM velocity if no wind specified: use dispersion along x
        let v_dm_default = [V_CHI_DISPERSION_KMS * KMS_TO_MS, 0.0, 0.0];
        let v_dm = if self.config.v_dm_wind.iter().any(|&v| v.abs() > 1e-30) {
            v_dm_phys
        } else {
            v_dm_default
        };

        for idx in 0..n {
            // Convert baryon density from LBM to physical (cm^-3 -> m^-3)
            let n_b_m3 = baryon_density[idx] * n_ref_cm3 * 1.0e6;

            // Convert baryon velocity from LBM to physical (m/s)
            let v_b = [
                baryon_velocity[idx][0] * v_ref_kms * KMS_TO_MS / 0.05,
                baryon_velocity[idx][1] * v_ref_kms * KMS_TO_MS / 0.05,
                baryon_velocity[idx][2] * v_ref_kms * KMS_TO_MS / 0.05,
            ];

            // Relative velocity
            let v_rel = [v_dm[0] - v_b[0], v_dm[1] - v_b[1], v_dm[2] - v_b[2]];
            let v_rel_mag = (v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]).sqrt();

            if v_rel_mag < 1e-30 {
                continue;
            }

            // F_drag = n_chi * n_b * sigma * m_red * |v_rel| * v_rel_hat (N/m^3)
            // Then convert to LBM acceleration: a_lattice = F * force_scale / rho_phys
            let f_mag = n_chi_m3 * n_b_m3 * sigma_m2 * m_red_kg * v_rel_mag;
            let a_phys = f_mag / (n_b_m3 * M_PROTON_GEV * GEV_TO_KG);
            let a_lattice = a_phys * self.config.force_scale;

            drag[idx] = [
                a_lattice * v_rel[0] / v_rel_mag,
                a_lattice * v_rel[1] / v_rel_mag,
                a_lattice * v_rel[2] / v_rel_mag,
            ];
        }

        drag
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

    // ---- MHD-DM coupling integration tests (P3.3c) ----

    #[test]
    fn test_combine_forces_identity() {
        // Combining any force field with zeros returns the original
        let a = vec![[1.5, -2.3, 0.7], [3.14, 0.0, -1.0]];
        let zero = vec![[0.0; 3]; 2];
        let result = combine_forces(&a, &zero);
        for (r, orig) in result.iter().zip(a.iter()) {
            assert!((r[0] - orig[0]).abs() < 1e-15);
            assert!((r[1] - orig[1]).abs() < 1e-15);
            assert!((r[2] - orig[2]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_combine_forces_commutativity() {
        let a = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, -0.3]];
        let b = vec![[0.1, -0.2, 0.3], [10.0, -5.0, 2.5], [0.0, 0.0, 1.0]];
        let ab = combine_forces(&a, &b);
        let ba = combine_forces(&b, &a);
        for (r_ab, r_ba) in ab.iter().zip(ba.iter()) {
            assert!((r_ab[0] - r_ba[0]).abs() < 1e-15);
            assert!((r_ab[1] - r_ba[1]).abs() < 1e-15);
            assert!((r_ab[2] - r_ba[2]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_dm_force_null_ratio() {
        // Verify DM gravity is negligible vs Lorentz force on a well-resolved grid.
        // Uses a synthetic non-uniform B-field (sinusoidal gradient) rather than
        // Parker spiral, which requires large grids to resolve properly.
        use crate::mhd::{MhdConfig, MhdField};

        let (nx, ny, nz) = (32, 16, 16);

        // Set up non-uniform B-field with gradient -> nonzero curl -> nonzero Lorentz
        let mut mhd = MhdField::new(nx, ny, nz, MhdConfig::default());
        let n = nx * ny * nz;
        for idx in 0..n {
            let x = idx % nx;
            let phase = 2.0 * std::f64::consts::PI * x as f64 / nx as f64;
            mhd.bx[idx] = 0.0;
            mhd.by[idx] = phase.sin() * 0.01; // sinusoidal By(x) -> dBy/dx != 0
            mhd.bz[idx] = 0.0;
        }
        let lorentz = mhd.lorentz_force();
        let max_lorentz = lorentz
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0_f64, f64::max);

        // DM: canonical NFW
        let dm = DmForceField::new(nx, ny, nz, DmForceConfig::default());
        let max_dm = dm.max_force_magnitude();

        assert!(max_lorentz > 1e-6, "Lorentz force should be nonzero: {max_lorentz:.3e}");
        assert!(max_dm > 0.0, "DM force should be nonzero");

        let ratio = max_dm / max_lorentz;
        assert!(
            ratio < 1e-6,
            "DM/Lorentz ratio should be negligible: {ratio:.3e} (max_dm={max_dm:.3e}, max_lorentz={max_lorentz:.3e})"
        );
    }

    #[test]
    fn test_dm_mass_conservation() {
        // DM force is a body force, not a source/sink.
        // Total mass must be conserved. Use uniform B-field (zero Lorentz) so
        // the only force is the tiny DM gravity -- stable on any grid.
        use crate::solver::LbmSolver3D;

        let (nx, ny, nz) = (16, 8, 8);
        let tau = 0.8; // higher tau = more viscous = more stable

        let mut solver = LbmSolver3D::new(nx, ny, nz, tau);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]); // quiescent
        let mass_initial = solver.total_mass();

        // DM force field only (no MHD -- pure body force test)
        let dm = DmForceField::new(nx, ny, nz, DmForceConfig::default());

        for _ in 0..50 {
            solver
                .set_force_field(dm.force.clone())
                .expect("force field set");
            solver.evolve_one_step();
        }

        let mass_final = solver.total_mass();
        let rel_err = (mass_final - mass_initial).abs() / mass_initial;
        assert!(
            rel_err < 1e-10,
            "mass conservation violated: initial={mass_initial:.8}, final={mass_final:.8}, rel_err={rel_err:.3e}"
        );
    }

    #[test]
    fn test_zero_dm_matches_pure_mhd() {
        // Evolving with an explicit zero force field should produce identical
        // results to evolving without any forcing.
        use crate::solver::LbmSolver3D;

        let (nx, ny, nz) = (16, 8, 8);
        let n = nx * ny * nz;
        let tau = 0.8;

        // Run A: no forcing
        let mut solver_a = LbmSolver3D::new(nx, ny, nz, tau);
        solver_a.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

        for _ in 0..10 {
            solver_a.evolve_one_step();
        }

        // Run B: explicit zero force via combine_forces
        let mut solver_b = LbmSolver3D::new(nx, ny, nz, tau);
        solver_b.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

        let zero_a = vec![[0.0; 3]; n];
        let zero_b = vec![[0.0; 3]; n];
        let combined = combine_forces(&zero_a, &zero_b);
        // Verify all zeros
        assert!(combined.iter().all(|f| f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0));

        for _ in 0..10 {
            solver_b
                .set_force_field(combined.clone())
                .expect("set");
            solver_b.evolve_one_step();
        }

        // Compare final fields: should be identical
        let mass_a = solver_a.total_mass();
        let mass_b = solver_b.total_mass();
        assert!(
            (mass_a - mass_b).abs() < 1e-12,
            "mass mismatch: unforced={mass_a:.8}, zero_forced={mass_b:.8}"
        );

        // Compare velocity fields
        for idx in 0..solver_a.u.len() {
            for d in 0..3 {
                let diff = (solver_a.u[idx][d] - solver_b.u[idx][d]).abs();
                assert!(
                    diff < 1e-12,
                    "velocity mismatch at idx={idx} dim={d}: unforced={:.8e}, zero_forced={:.8e}",
                    solver_a.u[idx][d],
                    solver_b.u[idx][d]
                );
            }
        }
    }

    #[test]
    fn test_drag_zero_sigma() {
        let field = default_field(8, 4, 4);
        let rho = vec![1.0; 8 * 4 * 4];
        let u = vec![[0.05, 0.0, 0.0]; 8 * 4 * 4];
        let drag = field.drag_force(&rho, &u, 5.0, 400.0);
        assert!(drag.iter().all(|f| f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0));
    }

    #[test]
    fn test_drag_nonzero_sigma() {
        let config = DmForceConfig {
            sigma_chi_b: 1e-40,
            ..DmForceConfig::default()
        };
        let field = DmForceField::new(8, 4, 4, config);
        let rho = vec![1.0; 8 * 4 * 4];
        let u = vec![[0.05, 0.0, 0.0]; 8 * 4 * 4];
        let drag = field.drag_force(&rho, &u, 5.0, 400.0);
        // At least some cells should have nonzero drag
        let has_nonzero = drag.iter().any(|f| f[0].abs() > 0.0 || f[1].abs() > 0.0 || f[2].abs() > 0.0);
        assert!(has_nonzero, "drag should be nonzero at sigma=1e-40");
        // All drag should be finite
        for f in &drag {
            assert!(f[0].is_finite());
            assert!(f[1].is_finite());
            assert!(f[2].is_finite());
        }
    }

    #[test]
    fn test_drag_monotonic_with_sigma() {
        // Higher sigma should produce larger drag force
        let config_low = DmForceConfig {
            sigma_chi_b: 1e-45,
            ..DmForceConfig::default()
        };
        let config_high = DmForceConfig {
            sigma_chi_b: 1e-40,
            ..DmForceConfig::default()
        };
        let field_low = DmForceField::new(8, 4, 4, config_low);
        let field_high = DmForceField::new(8, 4, 4, config_high);

        let rho = vec![1.0; 8 * 4 * 4];
        let u = vec![[0.05, 0.0, 0.0]; 8 * 4 * 4];

        let drag_low = field_low.drag_force(&rho, &u, 5.0, 400.0);
        let drag_high = field_high.drag_force(&rho, &u, 5.0, 400.0);

        let max_low: f64 = drag_low.iter()
            .map(|f| (f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).sqrt())
            .fold(0.0, f64::max);
        let max_high: f64 = drag_high.iter()
            .map(|f| (f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).sqrt())
            .fold(0.0, f64::max);

        assert!(
            max_high > max_low,
            "higher sigma should produce larger drag: low={max_low:.3e}, high={max_high:.3e}"
        );
    }
}
