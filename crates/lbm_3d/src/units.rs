//! Uniform-mesh physical units and an explicitly declared steady Parker model.

/// Conventional permeability used by the classical MHD unit model, in H/m.
pub const VACUUM_PERMEABILITY_H_M: f64 = 4.0 * std::f64::consts::PI * 1e-7;

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("invalid physical unit input: {0}")]
pub struct UnitError(pub &'static str);

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[derive(Clone, Debug)]
pub struct UniformCartesianMesh {
    dimensions: [usize; 3],
    origin_m: [f64; 3],
    spacing_m: f64,
}

impl UniformCartesianMesh {
    pub fn new(
        dimensions: [usize; 3],
        origin_m: [f64; 3],
        spacing_m: f64,
    ) -> Result<Self, UnitError> {
        if dimensions.contains(&0)
            || dimensions
                .iter()
                .try_fold(1usize, |count, &size| count.checked_mul(size))
                .is_none()
        {
            return Err(UnitError("mesh dimensions"));
        }
        if !positive(spacing_m) || !origin_m.iter().all(|value| value.is_finite()) {
            return Err(UnitError("mesh origin or spacing"));
        }
        for axis in 0..3 {
            if !(origin_m[axis] + (dimensions[axis] - 1) as f64 * spacing_m).is_finite() {
                return Err(UnitError("mesh coordinate overflow"));
            }
        }
        Ok(Self {
            dimensions,
            origin_m,
            spacing_m,
        })
    }
    pub fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }
    pub fn spacing_m(&self) -> f64 {
        self.spacing_m
    }
    pub fn origin_m(&self) -> [f64; 3] {
        self.origin_m
    }
    pub fn position_m(&self, index: [usize; 3]) -> Result<[f64; 3], UnitError> {
        if index
            .iter()
            .zip(self.dimensions)
            .any(|(&index, size)| index >= size)
        {
            return Err(UnitError("mesh index"));
        }
        Ok(std::array::from_fn(|axis| {
            self.origin_m[axis] + index[axis] as f64 * self.spacing_m
        }))
    }
}

#[derive(Clone, Debug)]
pub struct LatticeUnits {
    spacing_m: f64,
    timestep_s: f64,
    density_ref_kg_m3: f64,
    velocity_unit_m_s: f64,
    magnetic_unit_t: f64,
    acceleration_unit_m_s2: f64,
    force_density_unit_n_m3: f64,
    diffusivity_unit_m2_s: f64,
}

impl LatticeUnits {
    pub fn new(
        mesh: &UniformCartesianMesh,
        timestep_s: f64,
        density_ref_kg_m3: f64,
    ) -> Result<Self, UnitError> {
        if !positive(timestep_s) || !positive(density_ref_kg_m3) {
            return Err(UnitError("timestep or reference mass density"));
        }
        let spacing_m = mesh.spacing_m;
        let velocity_unit_m_s = spacing_m / timestep_s;
        let magnetic_unit_t =
            (VACUUM_PERMEABILITY_H_M * density_ref_kg_m3).sqrt() * velocity_unit_m_s;
        let acceleration_unit_m_s2 = velocity_unit_m_s / timestep_s;
        let force_density_unit_n_m3 = density_ref_kg_m3 * acceleration_unit_m_s2;
        let diffusivity_unit_m2_s = spacing_m * velocity_unit_m_s;
        if ![
            velocity_unit_m_s,
            magnetic_unit_t,
            acceleration_unit_m_s2,
            force_density_unit_n_m3,
            diffusivity_unit_m2_s,
        ]
        .into_iter()
        .all(positive)
        {
            return Err(UnitError("derived unit overflow or underflow"));
        }
        Ok(Self {
            spacing_m,
            timestep_s,
            density_ref_kg_m3,
            velocity_unit_m_s,
            magnetic_unit_t,
            acceleration_unit_m_s2,
            force_density_unit_n_m3,
            diffusivity_unit_m2_s,
        })
    }
    pub fn spacing_m(&self) -> f64 {
        self.spacing_m
    }
    pub fn timestep_s(&self) -> f64 {
        self.timestep_s
    }
    pub fn density_ref_kg_m3(&self) -> f64 {
        self.density_ref_kg_m3
    }
    pub fn velocity_unit_m_s(&self) -> f64 {
        self.velocity_unit_m_s
    }
    pub fn magnetic_unit_t(&self) -> f64 {
        self.magnetic_unit_t
    }
    pub fn velocity_to_lattice(&self, value_m_s: f64) -> f64 {
        value_m_s / self.velocity_unit_m_s
    }
    pub fn velocity_to_si(&self, value: f64) -> f64 {
        value * self.velocity_unit_m_s
    }
    pub fn magnetic_nt_to_lattice(&self, value_nt: f64) -> f64 {
        value_nt * 1e-9 / self.magnetic_unit_t
    }
    pub fn magnetic_to_nt(&self, value: f64) -> f64 {
        value * self.magnetic_unit_t * 1e9
    }
    pub fn density_to_lattice(&self, value_kg_m3: f64) -> f64 {
        value_kg_m3 / self.density_ref_kg_m3
    }
    pub fn density_to_si(&self, value: f64) -> f64 {
        value * self.density_ref_kg_m3
    }
    pub fn acceleration_to_lattice(&self, value_m_s2: f64) -> f64 {
        value_m_s2 / self.acceleration_unit_m_s2
    }
    pub fn acceleration_to_si(&self, value: f64) -> f64 {
        value * self.acceleration_unit_m_s2
    }
    pub fn force_density_to_lattice(&self, value_n_m3: f64) -> f64 {
        value_n_m3 / self.force_density_unit_n_m3
    }
    pub fn force_density_to_si(&self, value: f64) -> f64 {
        value * self.force_density_unit_n_m3
    }
    /// Magnetic diffusivity is eta_electrical/mu0, rather than electrical resistivity.
    pub fn diffusivity_to_lattice(&self, value_m2_s: f64) -> f64 {
        value_m2_s / self.diffusivity_unit_m2_s
    }
    pub fn diffusivity_to_si(&self, value: f64) -> f64 {
        value * self.diffusivity_unit_m2_s
    }
}

/// Axisymmetric steady radial-flow ideal-MHD construction; no external field is added.
/// The source radius and colatitude are explicit model inputs. The flow speed
/// is constant with radius, and the azimuthal winding neglects flow rotation
/// outside the source surface.
#[derive(Clone, Debug)]
pub struct ParkerSpiralSi {
    pub radial_field_at_reference_t: f64,
    pub reference_radius_m: f64,
    pub source_radius_m: f64,
    pub rotation_rad_s: f64,
    pub radial_speed_m_s: f64,
}

impl ParkerSpiralSi {
    /// Return spherical components [B_r, B_theta, B_phi] in tesla.
    pub fn field_t(&self, radius_m: f64, colatitude_rad: f64) -> Result<[f64; 3], UnitError> {
        if !positive(self.reference_radius_m)
            || !positive(self.radial_speed_m_s)
            || !positive(radius_m)
            || !self.source_radius_m.is_finite()
            || self.source_radius_m < 0.0
            || self.reference_radius_m < self.source_radius_m
            || radius_m < self.source_radius_m
            || !self.radial_field_at_reference_t.is_finite()
            || !self.rotation_rad_s.is_finite()
            || !(0.0..=std::f64::consts::PI).contains(&colatitude_rad)
        {
            return Err(UnitError("Parker model parameters or evaluation position"));
        }
        let radial =
            self.radial_field_at_reference_t * (self.reference_radius_m / radius_m).powi(2);
        let azimuthal = -radial
            * self.rotation_rad_s
            * (radius_m - self.source_radius_m)
            * colatitude_rad.sin()
            / self.radial_speed_m_s;
        if !radial.is_finite() || !azimuthal.is_finite() {
            return Err(UnitError("Parker field overflow"));
        }
        Ok([radial, 0.0, azimuthal])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= expected.abs() * 3e-15);
    }
    #[test]
    fn physical_unit_roundtrips_and_changed_lattice_units() {
        for spacing in [1e6, 2e6] {
            let mesh = UniformCartesianMesh::new([4, 4, 4], [0.0; 3], spacing).unwrap();
            let units = LatticeUnits::new(&mesh, 2.0, 1e-20).unwrap();
            close(
                units.velocity_to_si(units.velocity_to_lattice(400e3)),
                400e3,
            );
            close(
                units.magnetic_to_nt(units.magnetic_nt_to_lattice(-5.0)),
                -5.0,
            );
            close(
                units.acceleration_to_si(units.acceleration_to_lattice(3e-10)),
                3e-10,
            );
            close(
                units.force_density_to_si(units.force_density_to_lattice(7e-20)),
                7e-20,
            );
            close(
                units.diffusivity_to_si(units.diffusivity_to_lattice(4e8)),
                4e8,
            );
            close(units.density_to_si(units.density_to_lattice(3e-20)), 3e-20);
            close(
                units.magnetic_unit_t(),
                (VACUUM_PERMEABILITY_H_M * 1e-20).sqrt() * spacing / 2.0,
            );
        }
    }
    #[test]
    fn parker_radial_scaling_and_source_surface() {
        let mut model = ParkerSpiralSi {
            radial_field_at_reference_t: 3e-9,
            reference_radius_m: 1.496e11,
            source_radius_m: 0.0,
            rotation_rad_s: 2.662e-6,
            radial_speed_m_s: 400e3,
        };
        let inner = model
            .field_t(model.reference_radius_m, std::f64::consts::FRAC_PI_2)
            .unwrap();
        let outer = model
            .field_t(2.0 * model.reference_radius_m, std::f64::consts::FRAC_PI_2)
            .unwrap();
        close(outer[0] / inner[0], 0.25);
        close(outer[2] / inner[2], 0.5);
        close(inner[2] / inner[0], -2.662e-6 * 1.496e11 / 400e3);
        model.source_radius_m = model.reference_radius_m;
        assert_eq!(
            model.field_t(model.reference_radius_m, 1.0).unwrap()[2],
            0.0
        );
        assert!(model.field_t(0.5 * model.reference_radius_m, 1.0).is_err());
    }
    #[test]
    fn invalid_mesh_scales_and_coordinates_are_rejected() {
        assert!(UniformCartesianMesh::new([0, 1, 1], [0.0; 3], 1.0).is_err());
        assert!(UniformCartesianMesh::new([usize::MAX, 2, 1], [0.0; 3], 1.0).is_err());
        assert!(UniformCartesianMesh::new([1; 3], [f64::NAN, 0.0, 0.0], 1.0).is_err());
        assert!(UniformCartesianMesh::new([1; 3], [0.0; 3], 0.0).is_err());
        let mesh = UniformCartesianMesh::new([2; 3], [1.0; 3], 3.0).unwrap();
        assert_eq!(mesh.position_m([1; 3]).unwrap(), [4.0; 3]);
        assert!(mesh.position_m([2, 0, 0]).is_err());
        assert!(LatticeUnits::new(&mesh, 0.0, 1.0).is_err());
        assert!(LatticeUnits::new(&mesh, 1.0, -1.0).is_err());
        assert!(LatticeUnits::new(&mesh, 1e-300, 1.0).is_err());
    }
}
