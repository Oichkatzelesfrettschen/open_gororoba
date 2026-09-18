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
    /// Return nu = (tau - 1/2) * dx^2 / (3 * dt) for D3Q19 BGK.
    pub fn kinematic_viscosity_from_tau(&self, tau: f64) -> Result<f64, UnitError> {
        if !tau.is_finite() || tau <= 0.5 {
            return Err(UnitError("BGK relaxation time must exceed one half"));
        }
        let viscosity = self.diffusivity_to_si((tau - 0.5) / 3.0);
        if !positive(viscosity) {
            return Err(UnitError("physical viscosity overflow or underflow"));
        }
        Ok(viscosity)
    }
    /// Return a finite BGK relaxation time representing positive physical viscosity.
    pub fn tau_from_kinematic_viscosity(&self, value_m2_s: f64) -> Result<f64, UnitError> {
        if !positive(value_m2_s) {
            return Err(UnitError("physical viscosity must be finite and positive"));
        }
        let tau = 0.5 + 3.0 * self.diffusivity_to_lattice(value_m2_s);
        self.kinematic_viscosity_from_tau(tau)?;
        Ok(tau)
    }
}

/// Comparison metadata for an unforced, periodic, constant-viscosity BGK run.
///
/// The initial-data identifier names the same dimensionless spatial profile on
/// every mesh; `velocity_scale_lattice` supplies its amplitude. Callers must
/// initialize that profile and exclude forcing, filtering and viscosity changes.
/// These metadata checks do not inspect solver state or certify continuum error.
#[derive(Clone, Debug)]
pub struct PeriodicFlowParameters {
    mesh: UniformCartesianMesh,
    units: LatticeUnits,
    tau: f64,
    initial_data_id: String,
    velocity_scale_lattice: f64,
    steps: usize,
    domain_lengths_m: [f64; 3],
    viscosity_m2_s: f64,
    velocity_scale_m_s: f64,
    end_time_s: f64,
}

fn exact_count(value: usize) -> Result<f64, UnitError> {
    if value as u128 > 1_u128 << 53 {
        return Err(UnitError("count exceeds the exact binary64 integer range"));
    }
    Ok(value as f64)
}

impl PeriodicFlowParameters {
    pub fn new(
        mesh: UniformCartesianMesh,
        units: LatticeUnits,
        tau: f64,
        initial_data_id: impl Into<String>,
        velocity_scale_lattice: f64,
        steps: usize,
    ) -> Result<Self, UnitError> {
        let initial_data_id = initial_data_id.into();
        if mesh.spacing_m() != units.spacing_m() {
            return Err(UnitError("mesh and lattice-unit spacing differ"));
        }
        if initial_data_id.trim().is_empty()
            || !velocity_scale_lattice.is_finite()
            || velocity_scale_lattice < 0.0
            || steps == 0
        {
            return Err(UnitError(
                "initial-data identity, velocity scale or step count",
            ));
        }
        let mut domain_lengths_m = [0.0; 3];
        for (axis, length) in domain_lengths_m.iter_mut().enumerate() {
            // A periodic mesh spans N spacings, not the N-1 between stored endpoints.
            *length = exact_count(mesh.dimensions()[axis])? * mesh.spacing_m();
            if !positive(*length) || !(mesh.origin_m()[axis] + *length).is_finite() {
                return Err(UnitError("periodic domain extent overflow or underflow"));
            }
        }
        let viscosity_m2_s = units.kinematic_viscosity_from_tau(tau)?;
        let velocity_scale_m_s = units.velocity_to_si(velocity_scale_lattice);
        let end_time_s = exact_count(steps)? * units.timestep_s();
        if !velocity_scale_m_s.is_finite()
            || (velocity_scale_lattice > 0.0 && velocity_scale_m_s == 0.0)
            || !positive(end_time_s)
        {
            return Err(UnitError(
                "physical velocity or endpoint overflow or underflow",
            ));
        }
        Ok(Self {
            mesh,
            units,
            tau,
            initial_data_id,
            velocity_scale_lattice,
            steps,
            domain_lengths_m,
            viscosity_m2_s,
            velocity_scale_m_s,
            end_time_s,
        })
    }
    pub fn mesh(&self) -> &UniformCartesianMesh {
        &self.mesh
    }
    pub fn units(&self) -> &LatticeUnits {
        &self.units
    }
    pub fn tau(&self) -> f64 {
        self.tau
    }
    pub fn initial_data_id(&self) -> &str {
        &self.initial_data_id
    }
    pub fn velocity_scale_lattice(&self) -> f64 {
        self.velocity_scale_lattice
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn domain_lengths_m(&self) -> [f64; 3] {
        self.domain_lengths_m
    }
    pub fn viscosity_m2_s(&self) -> f64 {
        self.viscosity_m2_s
    }
    pub fn velocity_scale_m_s(&self) -> f64 {
        self.velocity_scale_m_s
    }
    pub fn end_time_s(&self) -> f64 {
        self.end_time_s
    }
    /// Reject different physical parameters or declared initial-data profiles.
    ///
    /// Derived binary64 quantities use a relative tolerance of 32 * EPSILON,
    /// with no absolute tolerance near zero. This accounts for unit-conversion
    /// rounding; it is not an interval proof of equality of real-valued inputs.
    pub fn require_same_continuum_problem(&self, other: &Self) -> Result<(), UnitError> {
        fn equivalent(left: f64, right: f64) -> bool {
            if left == right {
                return true;
            }
            let scale = left.abs().max(right.abs());
            (left / scale - right / scale).abs() <= 32.0 * f64::EPSILON
        }
        if self.initial_data_id != other.initial_data_id {
            return Err(UnitError("initial-data profiles differ"));
        }
        for axis in 0..3 {
            if !equivalent(self.mesh.origin_m()[axis], other.mesh.origin_m()[axis])
                || !equivalent(self.domain_lengths_m[axis], other.domain_lengths_m[axis])
            {
                return Err(UnitError("physical periodic domains differ"));
            }
        }
        for (left, right, error) in [
            (
                self.viscosity_m2_s,
                other.viscosity_m2_s,
                "physical viscosities differ",
            ),
            (
                self.velocity_scale_m_s,
                other.velocity_scale_m_s,
                "physical initial velocities differ",
            ),
            (
                self.end_time_s,
                other.end_time_s,
                "physical endpoints differ",
            ),
            (
                self.units.density_ref_kg_m3(),
                other.units.density_ref_kg_m3(),
                "reference mass densities differ",
            ),
        ] {
            if !equivalent(left, right) {
                return Err(UnitError(error));
            }
        }
        Ok(())
    }
    /// Refine dx by an integer factor and dt by its square at fixed physical data.
    pub fn diffusive_refinement(&self, factor: u32) -> Result<Self, UnitError> {
        if factor < 2 {
            return Err(UnitError("refinement factor must be at least two"));
        }
        let factor_count = usize::try_from(factor)
            .map_err(|_| UnitError("refinement factor exceeds index range"))?;
        let factor_squared = factor_count
            .checked_mul(factor_count)
            .ok_or(UnitError("refinement factor square overflow"))?;
        let mut dimensions = self.mesh.dimensions();
        for size in &mut dimensions {
            *size = size
                .checked_mul(factor_count)
                .ok_or(UnitError("refined mesh dimension overflow"))?;
        }
        let steps = self
            .steps
            .checked_mul(factor_squared)
            .ok_or(UnitError("refined step count overflow"))?;
        let mesh = UniformCartesianMesh::new(
            dimensions,
            self.mesh.origin_m(),
            self.mesh.spacing_m() / f64::from(factor),
        )?;
        let units = LatticeUnits::new(
            &mesh,
            self.units.timestep_s() / exact_count(factor_squared)?,
            self.units.density_ref_kg_m3(),
        )?;
        let velocity_scale_lattice = self.velocity_scale_lattice / f64::from(factor);
        if self.velocity_scale_lattice > 0.0 && velocity_scale_lattice == 0.0 {
            return Err(UnitError("refined lattice velocity underflow"));
        }
        let refined = Self::new(
            mesh,
            units,
            self.tau,
            self.initial_data_id.clone(),
            velocity_scale_lattice,
            steps,
        )?;
        self.require_same_continuum_problem(&refined)?;
        Ok(refined)
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
    fn periodic_flow() -> PeriodicFlowParameters {
        let mesh = UniformCartesianMesh::new([8, 8, 2], [0.0; 3], 1.0).unwrap();
        let units = LatticeUnits::new(&mesh, 1.0, 1.0).unwrap();
        PeriodicFlowParameters::new(mesh, units, 0.8, "xy-vortex", 0.005, 20).unwrap()
    }
    #[test]
    fn bgk_viscosity_conversion_rejects_nonphysical_and_unrepresentable_values() {
        let flow = periodic_flow();
        let units = flow.units();
        close(units.kinematic_viscosity_from_tau(0.8).unwrap(), 0.1);
        close(units.tau_from_kinematic_viscosity(0.1).unwrap(), 0.8);
        for tau in [0.0, 0.5, -1.0, f64::INFINITY, f64::NAN] {
            assert!(units.kinematic_viscosity_from_tau(tau).is_err());
        }
        for viscosity in [
            0.0,
            -1.0,
            f64::INFINITY,
            f64::NAN,
            f64::MAX,
            f64::from_bits(1),
        ] {
            assert!(units.tau_from_kinematic_viscosity(viscosity).is_err());
        }
    }
    #[test]
    fn diffusive_refinement_preserves_physical_data_and_scales_lattice_parameters() {
        let coarse = periodic_flow();
        for factor in [2, 3] {
            let fine = coarse.diffusive_refinement(factor).unwrap();
            let scale = f64::from(factor);
            assert_eq!(
                fine.mesh().dimensions(),
                [
                    8 * factor as usize,
                    8 * factor as usize,
                    2 * factor as usize
                ]
            );
            close(fine.units().spacing_m(), coarse.units().spacing_m() / scale);
            close(
                fine.units().timestep_s(),
                coarse.units().timestep_s() / scale.powi(2),
            );
            close(
                fine.velocity_scale_lattice(),
                coarse.velocity_scale_lattice() / scale,
            );
            assert_eq!(fine.steps(), coarse.steps() * (factor * factor) as usize);
            assert_eq!(fine.tau(), coarse.tau());
            coarse.require_same_continuum_problem(&fine).unwrap();
            fine.require_same_continuum_problem(&coarse).unwrap();
            close(fine.viscosity_m2_s(), coarse.viscosity_m2_s());
            close(fine.velocity_scale_m_s(), coarse.velocity_scale_m_s());
            close(fine.end_time_s(), coarse.end_time_s());
        }
    }
    #[test]
    fn continuum_comparison_rejects_old_lattice_parameters_and_changed_data() {
        let coarse = periodic_flow();
        let fine = coarse.diffusive_refinement(2).unwrap();
        for (tau, profile, velocity, steps) in [
            (
                fine.tau(),
                fine.initial_data_id(),
                coarse.velocity_scale_lattice(),
                fine.steps(),
            ),
            (
                fine.tau(),
                fine.initial_data_id(),
                fine.velocity_scale_lattice(),
                coarse.steps(),
            ),
            (
                0.9,
                fine.initial_data_id(),
                fine.velocity_scale_lattice(),
                fine.steps(),
            ),
            (
                fine.tau(),
                "different-profile",
                fine.velocity_scale_lattice(),
                fine.steps(),
            ),
        ] {
            let mismatched = PeriodicFlowParameters::new(
                fine.mesh().clone(),
                fine.units().clone(),
                tau,
                profile,
                velocity,
                steps,
            )
            .unwrap();
            assert!(coarse.require_same_continuum_problem(&mismatched).is_err());
        }
        for (dimensions, origin, density) in [
            ([16, 8, 2], [0.0; 3], 1.0),
            ([8, 8, 2], [0.0, 0.125, 0.0], 1.0),
            ([8, 8, 2], [0.0; 3], 2.0),
        ] {
            let mesh = UniformCartesianMesh::new(dimensions, origin, 1.0).unwrap();
            let units = LatticeUnits::new(&mesh, 1.0, density).unwrap();
            let mismatched = PeriodicFlowParameters::new(
                mesh,
                units,
                coarse.tau(),
                coarse.initial_data_id(),
                coarse.velocity_scale_lattice(),
                coarse.steps(),
            )
            .unwrap();
            assert!(coarse.require_same_continuum_problem(&mismatched).is_err());
        }
    }
    #[test]
    fn invalid_flow_metadata_and_refinement_overflow_are_rejected() {
        let flow = periodic_flow();
        for (profile, velocity, steps) in [
            (" ", 0.005, 20),
            ("xy-vortex", f64::NAN, 20),
            ("xy-vortex", -0.005, 20),
            ("xy-vortex", 0.005, 0),
        ] {
            assert!(
                PeriodicFlowParameters::new(
                    flow.mesh().clone(),
                    flow.units().clone(),
                    flow.tau(),
                    profile,
                    velocity,
                    steps,
                )
                .is_err()
            );
        }
        let other_mesh = UniformCartesianMesh::new([8, 8, 2], [0.0; 3], 0.5).unwrap();
        assert!(
            PeriodicFlowParameters::new(
                other_mesh,
                flow.units().clone(),
                flow.tau(),
                "xy-vortex",
                0.005,
                20,
            )
            .is_err()
        );
        for factor in [0, 1, u32::MAX] {
            assert!(flow.diffusive_refinement(factor).is_err());
        }
        if let Ok(steps) = usize::try_from((1_u128 << 53) + 1) {
            assert!(
                PeriodicFlowParameters::new(
                    flow.mesh().clone(),
                    flow.units().clone(),
                    flow.tau(),
                    "xy-vortex",
                    0.005,
                    steps,
                )
                .is_err()
            );
        }
        let tiny = PeriodicFlowParameters::new(
            flow.mesh().clone(),
            flow.units().clone(),
            flow.tau(),
            "xy-vortex",
            f64::from_bits(1),
            20,
        )
        .unwrap();
        assert!(tiny.diffusive_refinement(2).is_err());
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
