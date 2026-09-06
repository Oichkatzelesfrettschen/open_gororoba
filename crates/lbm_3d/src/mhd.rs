//! Magnetohydrodynamics extension for the D3Q19 LBM solver.
//!
//! Adds magnetic field evolution and Lorentz force coupling for
//! magnetized plasma simulations (e.g., solar wind Parker spiral).
//!
//! The B-field evolves via the induction equation:
//!   dB/dt = curl(v x B) - eta * curl(curl(B))
//!
//! where eta is magnetic diffusivity. The implementation uses a seven-point
//! Laplacian, whose continuum identity assumes divergence-free B.
//! The Lorentz force J x B (with
//! J = curl(B)/mu_0) couples back to the LBM via the Guo forcing scheme.

use crate::boundary::GridIndex;
use crate::units::{LatticeUnits, ParkerSpiralSi, UniformCartesianMesh, UnitError};

mod integrator;
pub use integrator::{MhdIntegrator, ssp_rk3_amplification_squared};

/// Invalid transport inputs or an unrepresentable prospective update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MhdError(pub &'static str);

impl std::fmt::Display for MhdError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for MhdError {}

/// Admitted magnetic diffusivity on a unit-spacing lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MagneticDiffusivity(f64);

impl MagneticDiffusivity {
    pub fn from_lattice(value: f64) -> Result<Self, MhdError> {
        if !value.is_finite() || value < 0.0 {
            return Err(MhdError(
                "magnetic diffusivity must be finite and nonnegative",
            ));
        }
        Ok(Self(value))
    }

    pub fn from_si(value_m2_s: f64, units: &LatticeUnits) -> Result<Self, MhdError> {
        Self::from_lattice(value_m2_s)?;
        let converted = units.diffusivity_to_lattice(value_m2_s);
        if value_m2_s > 0.0 && converted == 0.0 {
            return Err(MhdError("SI magnetic diffusivity conversion underflow"));
        }
        Self::from_lattice(converted)
    }

    pub fn lattice_value(self) -> f64 {
        self.0
    }
}

fn validate_configuration(dimensions: [usize; 3], config: &MhdConfig) -> Result<usize, MhdError> {
    let cells = dimensions
        .into_iter()
        .try_fold(1usize, |product, dimension| {
            if dimension == 0 {
                return Err(MhdError("MHD dimensions must be positive"));
            }
            product
                .checked_mul(dimension)
                .ok_or(MhdError("MHD grid product overflow"))
        })?;
    if cells > (isize::MAX as usize) / std::mem::size_of::<f64>() {
        return Err(MhdError(
            "MHD array byte length exceeds addressable capacity",
        ));
    }
    MagneticDiffusivity::from_lattice(config.eta)?;
    if !config.dt_mhd.is_finite() || config.dt_mhd <= 0.0 {
        return Err(MhdError("MHD timestep must be finite and positive"));
    }
    if !config.mu_0.is_finite() || config.mu_0 <= 0.0 {
        return Err(MhdError("MHD permeability must be finite and positive"));
    }
    if !config.cleaning_rate.is_finite() || config.cleaning_rate < 0.0 {
        return Err(MhdError(
            "MHD cleaning coefficient must be finite and nonnegative",
        ));
    }
    let spectral_sum: f64 = dimensions
        .into_iter()
        .map(|dimension| {
            (std::f64::consts::PI * (dimension / 2) as f64 / dimension as f64)
                .sin()
                .powi(2)
        })
        .sum();
    let diffusion_number = config.eta * config.dt_mhd * spectral_sum;
    if !diffusion_number.is_finite() || diffusion_number > 0.5 {
        return Err(MhdError(
            "finite-periodic diffusion-only Euler bound exceeded",
        ));
    }
    Ok(cells)
}

#[cfg(test)]
mod physical_unit_tests {
    use super::*;
    use crate::units::VACUUM_PERMEABILITY_H_M;

    #[test]
    fn diffusivity_conversion_and_invalid_values() {
        let mesh = UniformCartesianMesh::new([2, 2, 2], [0.0; 3], 4.0).unwrap();
        let units = LatticeUnits::new(&mesh, 2.0, 1.0).unwrap();
        assert_eq!(
            MagneticDiffusivity::from_si(2.0, &units)
                .unwrap()
                .lattice_value(),
            0.25
        );
        assert_eq!(
            MagneticDiffusivity::from_si(0.0, &units)
                .unwrap()
                .lattice_value(),
            0.0
        );
        assert!(MagneticDiffusivity::from_si(f64::from_bits(1), &units).is_err());
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(MagneticDiffusivity::from_lattice(invalid).is_err());
            assert!(MagneticDiffusivity::from_si(invalid, &units).is_err());
        }
    }

    #[test]
    fn mutable_transport_rejects_invalid_configuration() {
        assert!(MhdField::try_new(0, 2, 2, MhdConfig::default()).is_err());
        assert!(MhdField::try_new(usize::MAX, 2, 2, MhdConfig::default()).is_err());
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            for parameter in 0..4 {
                let mut field = MhdField::new(2, 2, 2, MhdConfig::default());
                match parameter {
                    0 => field.config.eta = invalid,
                    1 => field.config.cleaning_rate = invalid,
                    2 => field.config.dt_mhd = invalid,
                    _ => field.config.mu_0 = invalid,
                }
                assert!(field.try_evolve_b_field(&[[0.0; 3]; 8]).is_err());
                assert!(
                    field
                        .bx
                        .iter()
                        .chain(&field.by)
                        .chain(&field.bz)
                        .chain(&field.psi)
                        .all(|value| value.to_bits() == 0)
                );
            }
        }
        for parameter in 0..2 {
            let mut field = MhdField::new(2, 2, 2, MhdConfig::default());
            if parameter == 0 {
                field.config.dt_mhd = 0.0;
            } else {
                field.config.mu_0 = 0.0;
            }
            assert!(field.validate_transport().is_err());
        }
        let mut field = MhdField::new(2, 2, 2, MhdConfig::default());
        field.config.eta = 0.2;
        assert!(field.validate_transport().is_err());
        field.config.eta = 0.0;
        field.psi.pop();
        assert!(field.validate_transport().is_err());
    }

    #[test]
    fn normalized_lorentz_matches_si_linear_field_across_units() {
        for timestep_s in [2.0, 4.0] {
            let mesh = UniformCartesianMesh::new([5, 3, 3], [0.0; 3], 1e6).unwrap();
            let units = LatticeUnits::new(&mesh, timestep_s, 1e-20).unwrap();
            let mut field = MhdField::new(5, 3, 3, MhdConfig::default());
            let base_t = 5e-9;
            let gradient_t_m = 1e-16;
            for z in 0..3 {
                for y in 0..3 {
                    for x in 0..5 {
                        let index = z * 15 + y * 5 + x;
                        field.by[index] = (base_t + gradient_t_m * x as f64 * mesh.spacing_m())
                            / units.magnetic_unit_t();
                    }
                }
            }
            let force = field.lorentz_force();
            for x in 1..4 {
                let expected = -gradient_t_m
                    * (base_t + gradient_t_m * x as f64 * mesh.spacing_m())
                    / VACUUM_PERMEABILITY_H_M;
                let observed = units.force_density_to_si(force[15 + 5 + x][0]);
                assert!((observed / expected - 1.0).abs() < 2e-14);
                assert_eq!(force[15 + 5 + x][1], 0.0);
                assert_eq!(force[15 + 5 + x][2], 0.0);
            }
        }
    }

    #[test]
    fn physical_parker_initializer_converts_once_and_rejects_atomically() {
        let mesh = UniformCartesianMesh::new([2, 1, 1], [1.496e11, 0.0, 0.0], 1e6).unwrap();
        let units = LatticeUnits::new(&mesh, 2.0, 1e-20).unwrap();
        let model = ParkerSpiralSi {
            radial_field_at_reference_t: 3e-9,
            reference_radius_m: 1.496e11,
            source_radius_m: 0.0,
            rotation_rad_s: 2.662e-6,
            radial_speed_m_s: 400e3,
        };
        let mut field = MhdField::new(2, 1, 1, MhdConfig::default());
        field.initialize_parker_si(&mesh, &units, &model).unwrap();
        assert!((units.magnetic_to_nt(field.bx[0]) - 3.0).abs() < 1e-14);
        let expected = model
            .field_t(1.496e11, std::f64::consts::FRAC_PI_2)
            .unwrap();
        assert!((field.by[0] * units.magnetic_unit_t() / expected[2] - 1.0).abs() < 1e-15);
        let previous = (field.bx.clone(), field.by.clone(), field.bz.clone());
        let invalid = UniformCartesianMesh::new([2, 1, 1], [0.0; 3], 1e6).unwrap();
        assert!(
            field
                .initialize_parker_si(&invalid, &units, &model)
                .is_err()
        );
        assert_eq!((field.bx, field.by, field.bz), previous);
    }
}

/// Configuration for MHD simulation parameters.
#[derive(Clone, Debug)]
pub struct MhdConfig {
    /// Reference magnetic field magnitude (nT). Parker spiral B_0 at r_0.
    pub b0_nt: f64,
    /// Solar rotation rate (rad/s). Carrington: 2.662e-6.
    pub omega: f64,
    /// Magnetic permeability of free space (H/m) in simulation units.
    /// For dimensionless LBM, set to 1.0.
    pub mu_0: f64,
    /// Magnetic diffusivity in lattice units (dx squared per LBM timestep).
    /// The compatibility name eta denotes diffusivity, not electrical resistivity.
    pub eta: f64,
    /// MHD sub-timestep relative to LBM dt. Typically 1.0.
    pub dt_mhd: f64,
    /// Algebraic divergence damping coefficient. 0.0 disables damping.
    pub cleaning_rate: f64,
}

impl Default for MhdConfig {
    fn default() -> Self {
        Self {
            b0_nt: 5.0,
            omega: 2.662e-6,
            mu_0: 1.0,
            eta: 0.0,
            dt_mhd: 1.0,
            cleaning_rate: 0.0,
        }
    }
}

/// 3D magnetic field on the LBM grid.
pub struct MhdField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// B-field x-component, flat array of length nx*ny*nz.
    pub bx: Vec<f64>,
    /// B-field y-component.
    pub by: Vec<f64>,
    /// B-field z-component.
    pub bz: Vec<f64>,
    /// Algebraic divergence damping potential psi = -cleaning_rate^2 div B.
    pub psi: Vec<f64>,
    /// Configuration parameters.
    pub config: MhdConfig,
}

impl MhdField {
    /// Create a zero-initialized MHD field on the given grid.
    pub fn new(nx: usize, ny: usize, nz: usize, config: MhdConfig) -> Self {
        Self::try_new(nx, ny, nz, config).expect("invalid MHD construction")
    }

    /// Construct a field after checking grid capacity and transport parameters.
    pub fn try_new(nx: usize, ny: usize, nz: usize, config: MhdConfig) -> Result<Self, MhdError> {
        let n = validate_configuration([nx, ny, nz], &config)?;
        Ok(Self {
            nx,
            ny,
            nz,
            bx: vec![0.0; n],
            by: vec![0.0; n],
            bz: vec![0.0; n],
            psi: vec![0.0; n],
            config,
        })
    }

    /// Check mutable fields and the isolated diffusion Euler bound.
    /// Admission does not establish stability of induction, cleaning, or coupled flow.
    pub fn validate_transport(&self) -> Result<(), MhdError> {
        let cells = validate_configuration([self.nx, self.ny, self.nz], &self.config)?;
        for component in [&self.bx, &self.by, &self.bz, &self.psi] {
            if component.len() != cells {
                return Err(MhdError("MHD component length mismatch"));
            }
            if !component.iter().all(|value| value.is_finite()) {
                return Err(MhdError("MHD input component must be finite"));
            }
        }
        Ok(())
    }

    /// Initialize an uncalibrated legacy lattice Parker construction.
    /// Use `initialize_parker_si` for declared SI parameters and mesh conversion.
    ///
    /// The Parker spiral in Cartesian coordinates centered on the Sun:
    ///   B_r = B_0 * (r_0/r)^2
    ///   B_phi = -B_0 * (r_0/r) * (Omega * r * sin(theta)) / v_sw
    ///
    /// We map the grid to a radial slab: x is radial (Sun -> outward),
    /// y and z are transverse. The grid center is at 1 AU equivalent.
    ///
    /// `v_sw` is solar wind speed in simulation units (e.g., 0.1 in LBM lattice units).
    pub fn parker_spiral_init(&mut self, v_sw: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let b0 = self.config.b0_nt;
        let omega = self.config.omega;

        // Map grid: x in [0, nx) corresponds to radial distance.
        // r_0 is at grid center x = nx/2. Radial scaling: r/r_0 = x / (nx/2).
        let r0 = nx as f64 / 2.0;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = GridIndex::new(x, y, z).linearize(nx, ny);

                    // Radial distance (avoid r=0 singularity)
                    let r = (x as f64).max(1.0);
                    let ratio = r0 / r;

                    // B_r along x-axis (radial)
                    let b_r = b0 * ratio * ratio;

                    // B_phi along y-axis (azimuthal)
                    // Parker spiral angle: tan(psi) = Omega * r / v_sw
                    let b_phi = if v_sw.abs() > 1e-15 {
                        -b0 * ratio * (omega * r) / v_sw
                    } else {
                        0.0
                    };

                    self.bx[idx] = b_r;
                    self.by[idx] = b_phi;
                    // Bz = 0 for equatorial Parker spiral
                    self.bz[idx] = 0.0;
                }
            }
        }
    }

    /// Populate normalized Cartesian B from a heliocentric SI mesh and Parker model.
    /// The mesh origin is relative to the Sun; polar axes are the model's rotation
    /// axis. The complete field is validated before mutation. Magnetic permeability
    /// becomes one in lattice units. External or bias fields require a separate model.
    pub fn initialize_parker_si(
        &mut self,
        mesh: &UniformCartesianMesh,
        units: &LatticeUnits,
        model: &ParkerSpiralSi,
    ) -> Result<(), UnitError> {
        if mesh.dimensions() != [self.nx, self.ny, self.nz] || mesh.spacing_m() != units.spacing_m()
        {
            return Err(UnitError("MHD mesh dimensions or unit spacing mismatch"));
        }
        let mut fields = Vec::with_capacity(self.bx.len());
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let position = mesh.position_m([x, y, z])?;
                    let radius = position[0].hypot(position[1]).hypot(position[2]);
                    if radius == 0.0 {
                        return Err(UnitError("Parker mesh includes the Sun"));
                    }
                    let colatitude = (position[2] / radius).clamp(-1.0, 1.0).acos();
                    let longitude = position[1].atan2(position[0]);
                    let field = model.field_t(radius, colatitude)?;
                    let normalized = [
                        field[0] * colatitude.sin() * longitude.cos() - field[2] * longitude.sin(),
                        field[0] * colatitude.sin() * longitude.sin() + field[2] * longitude.cos(),
                        field[0] * colatitude.cos(),
                    ]
                    .map(|component| component / units.magnetic_unit_t());
                    if !normalized.iter().all(|component| component.is_finite()) {
                        return Err(UnitError("normalized Parker field overflow"));
                    }
                    fields.push(normalized);
                }
            }
        }
        for (index, field) in fields.iter().enumerate() {
            self.bx[index] = field[0];
            self.by[index] = field[1];
            self.bz[index] = field[2];
        }
        self.config.mu_0 = 1.0;
        Ok(())
    }

    /// Evolve B-field by one MHD timestep using the induction equation.
    ///
    /// dB/dt = curl(v x B) + eta * Laplacian(B)
    ///
    /// Uses forward-time centered-space (FTCS) finite differences with
    /// periodic boundary conditions. For ideal MHD (eta=0), this reduces
    /// to dB/dt = curl(v x B).
    pub fn evolve_b_field(&mut self, u: &[[f64; 3]]) {
        self.try_evolve_b_field(u).expect("invalid MHD update");
    }

    /// Compute a prospective update and commit only finite admitted fields.
    pub fn try_evolve_b_field(&mut self, u: &[[f64; 3]]) -> Result<(), MhdError> {
        self.validate_transport()?;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;
        let dt = self.config.dt_mhd;
        let eta = self.config.eta;

        if u.len() != n || !u.iter().flatten().all(|value| value.is_finite()) {
            return Err(MhdError(
                "MHD velocity length mismatch or nonfinite component",
            ));
        }

        // Compute v x B at each grid point
        let mut vxb_x = vec![0.0; n];
        let mut vxb_y = vec![0.0; n];
        let mut vxb_z = vec![0.0; n];

        for idx in 0..n {
            let [ux, uy, uz] = u[idx];
            let bx_i = self.bx[idx];
            let by_i = self.by[idx];
            let bz_i = self.bz[idx];
            // v x B = (uy*Bz - uz*By, uz*Bx - ux*Bz, ux*By - uy*Bx)
            vxb_x[idx] = uy * bz_i - uz * by_i;
            vxb_y[idx] = uz * bx_i - ux * bz_i;
            vxb_z[idx] = ux * by_i - uy * bx_i;
        }

        // Compute curl(v x B) via central differences (periodic)
        let mut dbx = vec![0.0; n];
        let mut dby = vec![0.0; n];
        let mut dbz = vec![0.0; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;

                    // Periodic neighbors
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    // curl(F)_x = dFz/dy - dFy/dz
                    // curl(F)_y = dFx/dz - dFz/dx
                    // curl(F)_z = dFy/dx - dFx/dy
                    let curl_vxb_x = 0.5 * (vxb_z[yp] - vxb_z[ym]) - 0.5 * (vxb_y[zp] - vxb_y[zm]);
                    let curl_vxb_y = 0.5 * (vxb_x[zp] - vxb_x[zm]) - 0.5 * (vxb_z[xp] - vxb_z[xm]);
                    let curl_vxb_z = 0.5 * (vxb_y[xp] - vxb_y[xm]) - 0.5 * (vxb_x[yp] - vxb_x[ym]);

                    dbx[idx] = curl_vxb_x;
                    dby[idx] = curl_vxb_y;
                    dbz[idx] = curl_vxb_z;

                    // Seven-point magnetic diffusion. The continuum replacement
                    // of -curl(curl(B)) by Laplacian(B) assumes div B = 0;
                    // centered discrete curl compositions use a different stencil.
                    if eta > 0.0 {
                        let lap_bx = self.bx[xp]
                            + self.bx[xm]
                            + self.bx[yp]
                            + self.bx[ym]
                            + self.bx[zp]
                            + self.bx[zm]
                            - 6.0 * self.bx[idx];
                        let lap_by = self.by[xp]
                            + self.by[xm]
                            + self.by[yp]
                            + self.by[ym]
                            + self.by[zp]
                            + self.by[zm]
                            - 6.0 * self.by[idx];
                        let lap_bz = self.bz[xp]
                            + self.bz[xm]
                            + self.bz[yp]
                            + self.bz[ym]
                            + self.bz[zp]
                            + self.bz[zm]
                            - 6.0 * self.bz[idx];
                        dbx[idx] += eta * lap_bx;
                        dby[idx] += eta * lap_by;
                        dbz[idx] += eta * lap_bz;
                    }
                }
            }
        }

        // Euler forward step
        for idx in 0..n {
            dbx[idx] = self.bx[idx] + dt * dbx[idx];
            dby[idx] = self.by[idx] + dt * dby[idx];
            dbz[idx] = self.bz[idx] + dt * dbz[idx];
        }

        // Reuse a cross-product buffer for the prospective algebraic potential.
        let mut next_psi = vxb_x;
        next_psi.copy_from_slice(&self.psi);
        // Algebraic divergence damping, without a hyperbolic psi evolution.
        if self.config.cleaning_rate > 0.0 {
            let ch = self.config.cleaning_rate;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        let div_b = 0.5 * (dbx[xp] - dbx[xm])
                            + 0.5 * (dby[yp] - dby[ym])
                            + 0.5 * (dbz[zp] - dbz[zm]);

                        next_psi[idx] = -ch * ch * div_b;
                    }
                }
            }
            // Apply correction: B -= dt * grad(psi)
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;
                        let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                        let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                        let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                        let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                        let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                        let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                        dbx[idx] -= dt * 0.5 * (next_psi[xp] - next_psi[xm]);
                        dby[idx] -= dt * 0.5 * (next_psi[yp] - next_psi[ym]);
                        dbz[idx] -= dt * 0.5 * (next_psi[zp] - next_psi[zm]);
                    }
                }
            }
        }
        if [&dbx, &dby, &dbz, &next_psi]
            .into_iter()
            .any(|component| component.iter().any(|value| !value.is_finite()))
        {
            return Err(MhdError(
                "prospective MHD update contains nonfinite components",
            ));
        }
        self.bx = dbx;
        self.by = dby;
        self.bz = dbz;
        self.psi = next_psi;
        Ok(())
    }

    /// Compute Lorentz force density: F = J x B / mu_0.
    ///
    /// J = curl(B) via central differences, then F = J x B.
    /// Returns force array compatible with `LbmSolver3D::set_force_field()`.
    pub fn lorentz_force(&self) -> Vec<[f64; 3]> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;
        let mu_0 = self.config.mu_0;

        let mut force = vec![[0.0; 3]; n];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;

                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    // J = curl(B) / mu_0
                    let jx = (0.5 * (self.bz[yp] - self.bz[ym])
                        - 0.5 * (self.by[zp] - self.by[zm]))
                        / mu_0;
                    let jy = (0.5 * (self.bx[zp] - self.bx[zm])
                        - 0.5 * (self.bz[xp] - self.bz[xm]))
                        / mu_0;
                    let jz = (0.5 * (self.by[xp] - self.by[xm])
                        - 0.5 * (self.bx[yp] - self.bx[ym]))
                        / mu_0;

                    // F = J x B
                    let bx_i = self.bx[idx];
                    let by_i = self.by[idx];
                    let bz_i = self.bz[idx];
                    force[idx] = [
                        jy * bz_i - jz * by_i,
                        jz * bx_i - jx * bz_i,
                        jx * by_i - jy * bx_i,
                    ];
                }
            }
        }

        force
    }

    /// Compute the maximum divergence of B across the grid (should be ~0 for physical fields).
    pub fn max_div_b(&self) -> f64 {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let mut max_div = 0.0_f64;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    let div_b = 0.5 * (self.bx[xp] - self.bx[xm])
                        + 0.5 * (self.by[yp] - self.by[ym])
                        + 0.5 * (self.bz[zp] - self.bz[zm]);
                    max_div = max_div.max(div_b.abs());
                }
            }
        }
        max_div
    }

    /// Project B-field to its divergence-free component via Helmholtz decomposition.
    ///
    /// Solves the Poisson equation `Lap(phi) = div(B)` using Jacobi iteration
    /// with periodic boundary conditions, then sets `B <- B - grad(phi)`.
    /// The result satisfies `div(B) = 0` to within the solver tolerance.
    ///
    /// This should be called ONCE after initialization (e.g., `parker_spiral_init`)
    /// to remove the magnetic monopole artifacts from projecting a spherically
    /// symmetric field onto a Cartesian grid.
    ///
    /// Returns `(initial_max_div, final_max_div)` for diagnostics.
    pub fn project_divergence_free(&mut self, max_iters: usize, tol: f64) -> (f64, f64) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n = nx * ny * nz;

        let initial_div = self.max_div_b();

        // Compute RHS: div(B) at each cell
        let mut rhs = vec![0.0; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    rhs[idx] = 0.5 * (self.bx[xp] - self.bx[xm])
                        + 0.5 * (self.by[yp] - self.by[ym])
                        + 0.5 * (self.bz[zp] - self.bz[zm]);
                }
            }
        }

        // Subtract mean from RHS (Poisson with periodic BC requires zero-mean RHS)
        let mean_rhs: f64 = rhs.iter().sum::<f64>() / n as f64;
        for v in &mut rhs {
            *v -= mean_rhs;
        }

        // Solve div_h(grad_h(phi)) = rhs via Jacobi iteration.
        //
        // Critical: the discrete Poisson operator MUST be consistent with the
        // central-difference div and grad operators used in max_div_b() and
        // the correction step. Central-difference div(grad(phi)) in 1D is:
        //   0.25*(phi[x+2] - 2*phi[x] + phi[x-2])
        // which is a WIDER stencil than the standard Laplacian. We solve
        // this directly to ensure div_h(B - grad_h(phi)) = 0 exactly.
        //
        // The wide Laplacian in 3D:
        //   L_wide(phi) = 0.25 * sum_dim (phi[+2] + phi[-2] - 2*phi[0])
        // Jacobi update: phi_new = (sum_wide_neighbors/4 - rhs) * 4/6
        //   where sum_wide = phi[x+2]+phi[x-2]+phi[y+2]+phi[y-2]+phi[z+2]+phi[z-2]

        // Helper: periodic index offset by +/-2
        let wrap = |v: usize, delta: isize, size: usize| -> usize {
            ((v as isize + delta).rem_euclid(size as isize)) as usize
        };

        let mut phi = vec![0.0; n];
        let mut phi_new = vec![0.0; n];

        for _iter in 0..max_iters {
            let mut max_residual = 0.0_f64;

            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = z * (nx * ny) + y * nx + x;

                        // Wide stencil neighbors (offset +/-2)
                        let xp2 = z * (nx * ny) + y * nx + wrap(x, 2, nx);
                        let xm2 = z * (nx * ny) + y * nx + wrap(x, -2, nx);
                        let yp2 = z * (nx * ny) + wrap(y, 2, ny) * nx + x;
                        let ym2 = z * (nx * ny) + wrap(y, -2, ny) * nx + x;
                        let zp2 = wrap(z, 2, nz) * (nx * ny) + y * nx + x;
                        let zm2 = wrap(z, -2, nz) * (nx * ny) + y * nx + x;

                        let sum_wide =
                            phi[xp2] + phi[xm2] + phi[yp2] + phi[ym2] + phi[zp2] + phi[zm2];

                        // Wide Laplacian: L = 0.25*(sum_wide - 6*phi_center)
                        // Solving L(phi) = rhs:
                        //   0.25*(sum_wide - 6*phi) = rhs
                        //   phi = (sum_wide - 4*rhs) / 6
                        phi_new[idx] = (sum_wide - 4.0 * rhs[idx]) / 6.0;

                        // Residual check
                        let lap_wide = 0.25 * (sum_wide - 6.0 * phi[idx]);
                        let residual = (lap_wide - rhs[idx]).abs();
                        max_residual = max_residual.max(residual);
                    }
                }
            }

            std::mem::swap(&mut phi, &mut phi_new);

            if max_residual < tol {
                break;
            }
        }

        // Apply correction: B <- B - grad_h(phi)
        // grad_h uses the same central-difference stencil as div_h
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    let xp = z * (nx * ny) + y * nx + (x + 1) % nx;
                    let xm = z * (nx * ny) + y * nx + (x + nx - 1) % nx;
                    let yp = z * (nx * ny) + ((y + 1) % ny) * nx + x;
                    let ym = z * (nx * ny) + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * (nx * ny) + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * (nx * ny) + y * nx + x;

                    self.bx[idx] -= 0.5 * (phi[xp] - phi[xm]);
                    self.by[idx] -= 0.5 * (phi[yp] - phi[ym]);
                    self.bz[idx] -= 0.5 * (phi[zp] - phi[zm]);
                }
            }
        }

        let final_div = self.max_div_b();
        (initial_div, final_div)
    }

    /// Total magnetic energy: integral of B^2 / (2 * mu_0) over the grid.
    pub fn magnetic_energy(&self) -> f64 {
        let n = self.nx * self.ny * self.nz;
        let mut energy = 0.0;
        for idx in 0..n {
            let b_sq = self.bx[idx] * self.bx[idx]
                + self.by[idx] * self.by[idx]
                + self.bz[idx] * self.bz[idx];
            energy += b_sq;
        }
        energy / (2.0 * self.config.mu_0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mhd_field_creation() {
        let field = MhdField::new(8, 8, 8, MhdConfig::default());
        assert_eq!(field.bx.len(), 512);
        assert_eq!(field.by.len(), 512);
        assert_eq!(field.bz.len(), 512);
        assert!(field.bx.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_parker_spiral_init() {
        let mut field = MhdField::new(
            16,
            8,
            8,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field.parker_spiral_init(0.1);

        // B_r should be positive everywhere (radial outward)
        let n = 16 * 8 * 8;
        assert!(field.bx.iter().take(n).all(|&v| v > 0.0));

        // B_r should decrease with radial distance (x)
        // Compare x=2 vs x=14 at y=4, z=4
        let idx_near = GridIndex::new(2, 4, 4).linearize(16, 8);
        let idx_far = GridIndex::new(14, 4, 4).linearize(16, 8);
        assert!(field.bx[idx_near] > field.bx[idx_far]);
    }

    #[test]
    fn test_uniform_b_zero_lorentz() {
        // Uniform B-field should produce zero current (curl(B)=0) and thus zero Lorentz force
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 1.0;
            field.by[idx] = 0.0;
            field.bz[idx] = 0.0;
        }

        let force = field.lorentz_force();
        for f in &force {
            assert!(f[0].abs() < 1e-14);
            assert!(f[1].abs() < 1e-14);
            assert!(f[2].abs() < 1e-14);
        }
    }

    #[test]
    fn test_divergence_uniform_field() {
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 3.0;
            field.by[idx] = -1.0;
            field.bz[idx] = 2.0;
        }
        // Uniform field has zero divergence
        assert!(field.max_div_b() < 1e-14);
    }

    #[test]
    fn test_magnetic_energy() {
        let mut field = MhdField::new(
            4,
            4,
            4,
            MhdConfig {
                mu_0: 1.0,
                ..MhdConfig::default()
            },
        );
        let n = 4 * 4 * 4;
        for idx in 0..n {
            field.bx[idx] = 1.0;
        }
        // Energy = sum(B^2) / (2 * mu_0) = 64 * 1.0 / 2.0 = 32.0
        assert!((field.magnetic_energy() - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_evolve_frozen_in() {
        // Ideal MHD (eta=0) with zero velocity: B should not change
        let mut field = MhdField::new(
            8,
            8,
            8,
            MhdConfig {
                eta: 0.0,
                ..MhdConfig::default()
            },
        );
        let n = 8 * 8 * 8;
        // Set uniform B
        for idx in 0..n {
            field.bx[idx] = 1.0;
        }
        let u = vec![[0.0, 0.0, 0.0]; n];
        let energy_before = field.magnetic_energy();

        field.evolve_b_field(&u);

        let energy_after = field.magnetic_energy();
        assert!(
            (energy_before - energy_after).abs() < 1e-12,
            "energy_before={energy_before}, energy_after={energy_after}"
        );
    }

    #[test]
    fn test_projection_uniform_is_identity() {
        // A uniform B-field already has div(B) = 0.
        // Projection should leave it unchanged.
        let mut field = MhdField::new(8, 8, 8, MhdConfig::default());
        let n = 8 * 8 * 8;
        for idx in 0..n {
            field.bx[idx] = 3.0;
            field.by[idx] = -1.0;
            field.bz[idx] = 2.0;
        }
        let energy_before = field.magnetic_energy();
        let (init_div, final_div) = field.project_divergence_free(1000, 1e-12);

        assert!(
            init_div < 1e-14,
            "uniform field should have zero div: {init_div}"
        );
        assert!(
            final_div < 1e-14,
            "projection should preserve zero div: {final_div}"
        );

        let energy_after = field.magnetic_energy();
        let rel_change = (energy_after - energy_before).abs() / energy_before;
        assert!(
            rel_change < 1e-10,
            "projection of divergence-free field should preserve energy: rel_change={rel_change:.3e}"
        );
    }

    #[test]
    fn test_projection_parker_reduces_div() {
        // Parker spiral on Cartesian grid has large div(B).
        // Projection should reduce it by several orders of magnitude.
        let mut field = MhdField::new(
            16,
            8,
            8,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field.parker_spiral_init(0.1);

        let (init_div, final_div) = field.project_divergence_free(5000, 1e-10);

        assert!(
            init_div > 1.0,
            "Parker spiral should have significant div(B): {init_div:.3e}"
        );
        assert!(
            final_div < init_div * 1e-3,
            "projection should reduce div(B) by >3 orders: {init_div:.3e} -> {final_div:.3e}"
        );
    }

    #[test]
    fn test_projection_preserves_energy_order() {
        // Projection removes the irrotational (gradient) component.
        // For a Parker spiral, the solenoidal component carries most of the energy.
        // Energy should remain within the same order of magnitude.
        let mut field = MhdField::new(
            16,
            8,
            8,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field.parker_spiral_init(0.1);
        let energy_before = field.magnetic_energy();

        field.project_divergence_free(5000, 1e-10);

        let energy_after = field.magnetic_energy();
        // Energy may decrease (we're removing the gradient component) but should
        // stay within a factor of 10
        assert!(
            energy_after > energy_before * 0.1,
            "energy should remain same order: before={energy_before:.3e}, after={energy_after:.3e}"
        );
        assert!(
            energy_after <= energy_before * 1.01,
            "energy should not increase: before={energy_before:.3e}, after={energy_after:.3e}"
        );
    }

    #[test]
    fn test_projection_preserves_br_monotonicity() {
        // After projection, B_r should still decrease with x (radial distance)
        // along the midline, though values will be modified.
        let mut field = MhdField::new(
            16,
            8,
            8,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field.parker_spiral_init(0.1);
        field.project_divergence_free(5000, 1e-10);

        // Check B_r decreases from x=2 to x=14 at midline
        let idx_near = GridIndex::new(2, 4, 4).linearize(16, 8);
        let idx_far = GridIndex::new(14, 4, 4).linearize(16, 8);
        assert!(
            field.bx[idx_near] > field.bx[idx_far],
            "B_r should still decrease with radius after projection: near={:.3e}, far={:.3e}",
            field.bx[idx_near],
            field.bx[idx_far]
        );
    }

    #[test]
    fn test_projection_idempotent() {
        // Applying projection twice: the second pass should find div(B) already
        // near zero and make negligible further changes.
        let mut field = MhdField::new(
            16,
            8,
            8,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field.parker_spiral_init(0.1);

        // First projection
        let (_, div_after_1) = field.project_divergence_free(5000, 1e-12);
        let energy_after_1 = field.magnetic_energy();

        // Second projection
        let (div_before_2, div_after_2) = field.project_divergence_free(5000, 1e-12);
        let energy_after_2 = field.magnetic_energy();

        // div_before_2 should equal div_after_1 (same field)
        assert!(
            (div_before_2 - div_after_1).abs() < 1e-14,
            "second pass initial div should match first pass final: {div_before_2:.3e} vs {div_after_1:.3e}"
        );

        // Second projection should not change energy
        let rel_energy = (energy_after_2 - energy_after_1).abs() / energy_after_1;
        assert!(
            rel_energy < 1e-10,
            "idempotent projection should preserve energy: rel_change={rel_energy:.3e}"
        );

        // div_after_2 should be same or smaller
        assert!(
            div_after_2 <= div_after_1 * 1.01,
            "second projection should not increase div: {div_after_1:.3e} -> {div_after_2:.3e}"
        );
    }

    #[test]
    fn test_lorentz_force_scales_with_b0() {
        // F_Lorentz = J x B where J = curl(B)/mu_0.
        // For Parker spiral, B ~ b0, J ~ b0, so F ~ b0^2.
        // Doubling b0 should quadruple the max Lorentz force.
        let (nx, ny, nz) = (16, 8, 8);

        let mut field_lo = MhdField::new(
            nx,
            ny,
            nz,
            MhdConfig {
                b0_nt: 5.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field_lo.parker_spiral_init(0.05);
        let lorentz_lo = field_lo.lorentz_force();
        let max_lo = lorentz_lo
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0_f64, f64::max);

        let mut field_hi = MhdField::new(
            nx,
            ny,
            nz,
            MhdConfig {
                b0_nt: 10.0,
                omega: 2.662e-6,
                ..MhdConfig::default()
            },
        );
        field_hi.parker_spiral_init(0.05);
        let lorentz_hi = field_hi.lorentz_force();
        let max_hi = lorentz_hi
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0_f64, f64::max);

        // b0 ratio = 10/5 = 2, so force ratio should be ~4 (quadratic)
        let ratio = max_hi / max_lo;
        assert!(
            (ratio - 4.0).abs() < 0.5,
            "Lorentz force should scale as B^2: max_lo={max_lo:.3e}, max_hi={max_hi:.3e}, ratio={ratio:.2}"
        );
    }
}
