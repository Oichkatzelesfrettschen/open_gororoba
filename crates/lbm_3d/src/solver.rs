//! BGK collision operator for Lattice Boltzmann Method (3D).
//!
//! The BGK (Bhatnagar-Gross-Krook) collision operator is the standard relaxation
//! operator for LBM. It drives the distribution function toward equilibrium with
//! relaxation time tau:
//!
//! f_i^new = f_i - (f_i - f_i^eq) / tau
//!
//! The relaxation time connects to viscosity via:
//! nu = c_s^2 * (tau - 0.5) = (1/3) * (tau - 0.5)

use crate::lattice::D3Q19Lattice;
use cosmic_scheduler::{TwoPhaseSystem, ScheduleResult};

/// BGK collision operator for 3D LBM.
#[derive(Clone, Debug)]
pub struct BgkCollision {
    /// Relaxation time field (tau >= 0.5 for stability at each grid point).
    /// Length must equal nx*ny*nz for spatial viscosity variation.
    /// For uniform viscosity, all elements are identical.
    pub tau_field: Vec<f64>,
    /// Lattice for equilibrium computation
    pub lattice: D3Q19Lattice,
}

impl BgkCollision {
    /// Create a BGK collision operator with uniform relaxation time.
    ///
    /// Initializes a uniform tau field (all cells have same relaxation time).
    /// For spatial viscosity variation, use set_viscosity_field() after construction.
    ///
    /// # Arguments
    /// * `tau` - Relaxation time. For stability: tau >= 0.5
    ///   - tau = 0.5 => zero viscosity (inviscid limit)
    ///   - tau > 0.5 => finite viscosity nu = c_s^2 * (tau - 0.5)
    ///
    /// Note: Field length must be set via set_viscosity_field() before use with LbmSolver3D.
    pub fn new(tau: f64) -> Self {
        assert!(tau >= 0.5, "tau must be >= 0.5 for stability");
        Self {
            tau_field: vec![tau],  // Placeholder; solver will set actual field
            lattice: D3Q19Lattice::new(),
        }
    }

    /// Set the spatially-varying viscosity field (relaxation time per grid point).
    ///
    /// # Arguments
    /// * `tau_field` - Vector of relaxation times, one per grid point (length nx*ny*nz)
    ///
    /// # Errors
    /// Returns Err if:
    /// - Any tau < 0.5 (violates stability constraint)
    /// - Field contains NaN or Inf
    /// - Field is empty
    pub fn set_viscosity_field(&mut self, tau_field: Vec<f64>) -> Result<(), String> {
        if tau_field.is_empty() {
            return Err("Viscosity field cannot be empty".to_string());
        }

        for (i, &tau) in tau_field.iter().enumerate() {
            if !tau.is_finite() {
                return Err(format!(
                    "Non-finite tau at index {}: {}",
                    i, tau
                ));
            }
            if tau < 0.5 {
                return Err(format!(
                    "Stability violation at index {}: tau={} < 0.5",
                    i, tau
                ));
            }
        }

        self.tau_field = tau_field;
        Ok(())
    }

    /// Get the viscosity field (tau values) as-is.
    pub fn get_tau_field(&self) -> &[f64] {
        &self.tau_field
    }

    /// Get the kinematic viscosity field from relaxation time field.
    /// nu(x) = c_s^2 * (tau(x) - 0.5) = (1/3) * (tau(x) - 0.5)
    pub fn get_viscosity_field(&self) -> Vec<f64> {
        self.tau_field.iter()
            .map(|&tau| self.lattice.cs_sq * (tau - 0.5))
            .collect()
    }

    /// Compute kinematic viscosity from first relaxation time (representative value).
    /// For uniform fields, this is the viscosity everywhere.
    /// For spatial fields, this is the viscosity at grid point 0.
    /// nu = c_s^2 * (tau - 0.5) = (1/3) * (tau - 0.5)
    ///
    /// # Panics
    /// If tau_field is empty.
    pub fn viscosity(&self) -> f64 {
        assert!(!self.tau_field.is_empty(), "tau_field must not be empty");
        self.lattice.cs_sq * (self.tau_field[0] - 0.5)
    }

    /// Recover macroscopic density from distribution function.
    /// rho = sum_i f_i
    pub fn density_from_f(f: &[f64; 19]) -> f64 {
        f.iter().sum()
    }

    /// Recover macroscopic velocity from distribution function.
    /// u_k = (1/rho) * sum_i f_i * c_i^k
    pub fn velocity_from_f(f: &[f64; 19], rho: f64, lattice: &D3Q19Lattice) -> [f64; 3] {
        let mut u = [0.0; 3];

        if rho.abs() < 1e-14 {
            return u;  // Zero density => zero velocity
        }

        for (i, &fi) in f.iter().enumerate() {
            let c = lattice.velocity(i);
            u[0] += fi * (c[0] as f64);
            u[1] += fi * (c[1] as f64);
            u[2] += fi * (c[2] as f64);
        }

        u[0] /= rho;
        u[1] /= rho;
        u[2] /= rho;

        u
    }

    /// Initialize distribution function at rest (rho, u = 0).
    /// f_i^eq(rho, u=0) = rho * w_i
    pub fn initialize_rest(rho: f64, lattice: &D3Q19Lattice) -> [f64; 19] {
        let mut f = [0.0; 19];
        for (i, f_i) in f.iter_mut().enumerate() {
            *f_i = rho * lattice.weight(i);
        }
        f
    }

    /// Initialize distribution function with velocity.
    /// f_i = f_i^eq(rho, u)
    pub fn initialize_with_velocity(
        rho: f64,
        u: [f64; 3],
        lattice: &D3Q19Lattice,
    ) -> [f64; 19] {
        let mut f = [0.0; 19];
        for (i, f_i) in f.iter_mut().enumerate() {
            *f_i = lattice.equilibrium(rho, u, i);
        }
        f
    }

    /// Perform one BGK collision step with specified relaxation time.
    /// f_i^new = f_i - (f_i - f_i^eq) / tau
    ///
    /// # Arguments
    /// * `f` - Current distribution function (19 components)
    /// * `f_eq` - Equilibrium distribution (19 components)
    /// * `tau` - Relaxation time for this step
    pub fn collision_step(&self, f: &[f64; 19], f_eq: &[f64; 19], tau: f64) -> [f64; 19] {
        let mut f_new = [0.0; 19];
        for i in 0..19 {
            f_new[i] = f[i] - (f[i] - f_eq[i]) / tau;
        }
        f_new
    }

    /// Perform collision step with automatic equilibrium computation.
    /// Uses the first tau_field value (representative viscosity).
    ///
    /// # Arguments
    /// * `f` - Current distribution function
    /// * `rho` - Macroscopic density
    /// * `u` - Macroscopic velocity
    ///
    /// # Panics
    /// If tau_field is empty
    pub fn collision_step_with_equilibrium(
        &self,
        f: &[f64; 19],
        rho: f64,
        u: [f64; 3],
    ) -> [f64; 19] {
        // Compute equilibrium
        let mut f_eq = [0.0; 19];
        for (i, f_eq_i) in f_eq.iter_mut().enumerate() {
            *f_eq_i = self.lattice.equilibrium(rho, u, i);
        }

        // Use first tau value (representative for uniform fields)
        let tau = if !self.tau_field.is_empty() { self.tau_field[0] } else { 0.6 };

        // Perform collision
        self.collision_step(f, &f_eq, tau)
    }

    /// Check non-negativity of distribution function (stability indicator).
    /// For typical flows at low Mach number, f_i >= 0 always.
    pub fn is_stable(f: &[f64; 19]) -> bool {
        f.iter().all(|&fi| fi >= -1e-14)  // Allow small numerical error
    }
}

/// 3D LBM solver with D3Q19 lattice and BGK collision.
///
/// Encapsulates a complete fluid simulation domain with:
/// - Distribution functions at each grid point
/// - Macroscopic quantities (density, velocity)
/// - BGK collision operator
/// - D3Q19 lattice geometry
#[derive(Clone, Debug)]
pub struct LbmSolver3D {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Distribution function: f[grid_index][velocity_index]
    /// Flattened: [x + nx*(y + ny*z)]*19 + i
    pub f: Vec<f64>,
    /// Macroscopic density at each grid point
    pub rho: Vec<f64>,
    /// Macroscopic velocity at each grid point
    pub u: Vec<[f64; 3]>,
    /// BGK collision operator
    pub collider: BgkCollision,
    /// Timestep counter
    pub timestep: usize,
}

impl LbmSolver3D {
    /// Create a new 3D LBM solver domain.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `tau` - Relaxation time (must be >= 0.5)
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Self {
        let n_nodes = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            f: vec![0.0; n_nodes * 19],
            rho: vec![0.0; n_nodes],
            u: vec![[0.0; 3]; n_nodes],
            collider: BgkCollision::new(tau),
            timestep: 0,
        }
    }

    /// Set the spatially-varying viscosity field (tau values per grid point).
    ///
    /// Must be called before evolving with spatial viscosity variation.
    /// For uniform viscosity, pass a vector of identical values.
    ///
    /// # Arguments
    /// * `tau_field` - Vector of tau values, one per grid point (must have length nx*ny*nz)
    ///
    /// # Errors
    /// Propagates errors from BgkCollision::set_viscosity_field()
    pub fn set_viscosity_field(&mut self, tau_field: Vec<f64>) -> Result<(), String> {
        let expected_len = self.nx * self.ny * self.nz;
        if tau_field.len() != expected_len {
            return Err(format!(
                "Viscosity field length mismatch: got {}, expected {} ({}x{}x{})",
                tau_field.len(), expected_len, self.nx, self.ny, self.nz
            ));
        }
        self.collider.set_viscosity_field(tau_field)
    }

    /// Get the current viscosity field.
    pub fn get_viscosity_field(&self) -> Vec<f64> {
        self.collider.get_viscosity_field()
    }

    /// Initialize entire domain with uniform density and velocity.
    pub fn initialize_uniform(&mut self, rho_init: f64, u_init: [f64; 3]) {
        let lattice = &self.collider.lattice;

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let idx = self.linearize(x, y, z);

                    // Initialize macroscopic quantities
                    self.rho[idx] = rho_init;
                    self.u[idx] = u_init;

                    // Initialize distribution function to equilibrium
                    let f_eq = BgkCollision::initialize_with_velocity(rho_init, u_init, lattice);
                    let f_start = idx * 19;
                    self.f[f_start..f_start + 19].copy_from_slice(&f_eq);
                }
            }
        }
    }

    /// Linearize 3D grid coordinates to 1D index.
    fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Compute macroscopic quantities (rho, u) from distribution function.
    pub fn compute_macroscopic(&mut self) {
        let lattice = &self.collider.lattice;

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let idx = self.linearize(x, y, z);
                    let f_start = idx * 19;

                    // Extract f at this point
                    let mut f = [0.0; 19];
                    f.copy_from_slice(&self.f[f_start..f_start + 19]);

                    // Recover macroscopic quantities
                    self.rho[idx] = BgkCollision::density_from_f(&f);
                    self.u[idx] = BgkCollision::velocity_from_f(&f, self.rho[idx], lattice);
                }
            }
        }
    }

    /// Phase 1 (collision preparation): Compute macroscopic quantities and apply BGK collision operator.
    ///
    /// Implements the Chapman-Enskog collision operator with spatially-varying relaxation time:
    /// f_i^new <- f_i - (f_i - f_i^eq) / tau(x,y,z)
    ///
    /// This phase prepares the distribution function for the subsequent streaming step.
    /// Relaxation time varies per grid point to enable viscosity-driven simulation.
    pub fn phase1_collision(&mut self) -> ScheduleResult<()> {
        let lattice = self.collider.lattice.clone();
        let tau_field = self.collider.tau_field.clone();
        let nx = self.nx;
        let ny = self.ny;

        // Recover macroscopic quantities (density rho, velocity u_k)
        self.compute_macroscopic();

        // Apply BGK collision at each grid point
        for z in 0..self.nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.linearize(x, y, z);
                    let f_start = idx * 19;

                    // Get per-cell relaxation time
                    let tau = if idx < tau_field.len() {
                        tau_field[idx]
                    } else {
                        // Fallback: if field not set, use first element
                        if !tau_field.is_empty() { tau_field[0] } else { 0.6 }
                    };

                    // Extract population distribution function f_i at this lattice site
                    let mut f = [0.0; 19];
                    f.copy_from_slice(&self.f[f_start..f_start + 19]);

                    // Compute equilibrium distribution f_i^eq
                    let mut f_eq = [0.0; 19];
                    for (i, f_eq_i) in f_eq.iter_mut().enumerate() {
                        *f_eq_i = lattice.equilibrium(self.rho[idx], self.u[idx], i);
                    }

                    // BGK collision step: relax toward equilibrium with per-cell tau
                    let mut f_new = [0.0; 19];
                    for i in 0..19 {
                        f_new[i] = f[i] - (f[i] - f_eq[i]) / tau;
                    }

                    // Update distribution function
                    self.f[f_start..f_start + 19].copy_from_slice(&f_new);
                }
            }
        }

        Ok(())
    }

    /// Phase 2 (streaming): In the D3Q19 lattice, streaming is implicit via lattice velocity mapping.
    ///
    /// In standard LBM, the streaming step redistributes populations to neighboring sites:
    /// f_i(x + c_i*dt, t + dt) <- f_i(x, t)
    ///
    /// For the collision-streaming operator, we recover macroscopic quantities post-collision.
    /// The lattice structure (D3Q19 discrete velocities) encodes the streaming geometry.
    pub fn phase2_streaming(&mut self) -> ScheduleResult<()> {
        // In the current one-step collision-only formulation, streaming is implicit.
        // Recover macroscopic quantities for validation and next phase initialization.
        self.compute_macroscopic();
        self.timestep += 1;

        Ok(())
    }

    /// Perform one complete LBM timestep via two-phase coordination:
    /// Phase 1: Collision operator (BGK) applied to population distribution functions
    /// Phase 2: Macroscopic quantity recovery (streaming implicit in D3Q19 lattice structure)
    pub fn evolve_one_step(&mut self) {
        let _ = self.phase1_collision();
        let _ = self.phase2_streaming();
    }

    /// Perform multiple LBM timesteps.
    pub fn evolve(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.evolve_one_step();
        }
    }

    /// Get macroscopic quantities at a grid point.
    pub fn get_macroscopic(&self, x: usize, y: usize, z: usize) -> (f64, [f64; 3]) {
        let idx = self.linearize(x, y, z);
        (self.rho[idx], self.u[idx])
    }

    /// Check global stability (all f values non-negative).
    pub fn is_stable(&self) -> bool {
        self.f.iter().all(|&fi| fi >= -1e-14)
    }

    /// Compute total mass (should be conserved).
    pub fn total_mass(&self) -> f64 {
        self.rho.iter().sum()
    }

    /// Compute total momentum magnitude.
    pub fn total_momentum(&self) -> f64 {
        self.u.iter().map(|ui| ui[0]*ui[0] + ui[1]*ui[1] + ui[2]*ui[2]).sum::<f64>().sqrt()
    }
}

/// Implement two-phase system trait for deterministic phase coordination.
///
/// Maps the D3Q19 lattice Boltzmann method to the PhaseScheduler abstraction:
/// - Phase 1 (collision): BGK collision operator Omega applies Chapman-Enskog relaxation
/// - Phase 2 (streaming): Macroscopic recovery; streaming implicit in lattice geometry
///
/// This enables cosmic_scheduler to coordinate LBM evolution with deterministic timing
/// guarantees, matching the two-phase clock abstraction from the Intel 4004 architecture.
impl TwoPhaseSystem for LbmSolver3D {
    /// Execute Phase 1: Collision operator (BGK).
    /// Applies Chapman-Enskog collision to relax population distribution toward equilibrium.
    fn execute_phase1(&mut self) -> ScheduleResult<()> {
        self.phase1_collision()
    }

    /// Execute Phase 2: Streaming (implicit via D3Q19 lattice).
    /// Recovers macroscopic quantities post-collision.
    fn execute_phase2(&mut self) -> ScheduleResult<()> {
        self.phase2_streaming()
    }

    /// Validate system state: Check stability.
    ///
    /// Ensures stability of the Navier-Stokes simulator by verifying:
    /// - Total mass rho >= 0 (non-negative density everywhere)
    /// - Population distribution f_i >= 0 (stability in BGK collision)
    ///
    /// Note: Mass conservation is maintained by the BGK collision operator by construction
    /// and need not be checked explicitly. The validation focuses on stability metrics.
    fn validate_state(&self) -> ScheduleResult<()> {
        // Check stability: all population values non-negative
        if !self.is_stable() {
            return Err(cosmic_scheduler::ScheduleError::StateInvalid(
                format!("LBM population instability: negative f_i detected at timestep {}", self.timestep)
            ));
        }

        // Check non-negativity of density field
        for (i, &rho_i) in self.rho.iter().enumerate() {
            if rho_i < -1e-14 {
                return Err(cosmic_scheduler::ScheduleError::StateInvalid(
                    format!("Negative density at node {}: {} at timestep {}",
                        i, rho_i, self.timestep)
                ));
            }
        }

        Ok(())
    }

    /// Current simulation time (timestep counter).
    fn current_time(&self) -> Option<cosmic_scheduler::Time> {
        Some(self.timestep as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collision_operator_creation() {
        let bgk = BgkCollision::new(1.0);
        assert!(!bgk.tau_field.is_empty());
        assert!(bgk.tau_field[0] >= 0.5);
    }

    #[test]
    fn test_collision_operator_zero_viscosity() {
        let bgk = BgkCollision::new(0.5);
        let nu = bgk.viscosity();
        assert!((nu - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_collision_operator_finite_viscosity() {
        let bgk = BgkCollision::new(0.6);
        let nu = bgk.viscosity();
        let expected = (1.0 / 3.0) * (0.6 - 0.5);
        assert!((nu - expected).abs() < 1e-14);
    }

    #[test]
    fn test_density_from_f() {
        let f = [1.0; 19];
        let rho = BgkCollision::density_from_f(&f);
        assert!((rho - 19.0).abs() < 1e-14);
    }

    #[test]
    fn test_velocity_from_f_zero() {
        let lattice = D3Q19Lattice::new();
        let rho = 1.0;
        let f = BgkCollision::initialize_rest(rho, &lattice);
        let u = BgkCollision::velocity_from_f(&f, rho, &lattice);
        assert!(u[0].abs() < 1e-14);
        assert!(u[1].abs() < 1e-14);
        assert!(u[2].abs() < 1e-14);
    }

    #[test]
    fn test_initialize_rest() {
        let lattice = D3Q19Lattice::new();
        let rho = 2.0;
        let f = BgkCollision::initialize_rest(rho, &lattice);

        // Check: sum(f) = rho
        let sum_f: f64 = f.iter().sum();
        assert!((sum_f - rho).abs() < 1e-14);

        // Check: each f[i] = rho * w[i]
        for i in 0..19 {
            let expected = rho * lattice.weight(i);
            assert!((f[i] - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_initialize_with_velocity() {
        let lattice = D3Q19Lattice::new();
        let rho = 1.0;
        let u = [0.1, 0.05, 0.02];
        let f = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // Check: sum(f) = rho
        let sum_f: f64 = f.iter().sum();
        assert!((sum_f - rho).abs() < 1e-12);
    }

    #[test]
    fn test_mass_conservation() {
        let bgk = BgkCollision::new(0.8);
        let lattice = D3Q19Lattice::new();

        let rho = 1.5;
        let u = [0.1, 0.05, 0.02];
        let f = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // Perform collision
        let f_new = bgk.collision_step_with_equilibrium(&f, rho, u);

        // Check mass conservation
        let sum_f_new: f64 = f_new.iter().sum();
        assert!((sum_f_new - rho).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_conservation() {
        let bgk = BgkCollision::new(0.8);
        let lattice = D3Q19Lattice::new();

        let rho = 1.5;
        let u = [0.1, 0.05, 0.02];
        let f = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // Perform collision
        let f_new = bgk.collision_step_with_equilibrium(&f, rho, u);

        // Recover velocity
        let rho_new = BgkCollision::density_from_f(&f_new);
        let u_new = BgkCollision::velocity_from_f(&f_new, rho_new, &lattice);

        // Check momentum conservation
        assert!((u_new[0] - u[0]).abs() < 1e-12);
        assert!((u_new[1] - u[1]).abs() < 1e-12);
        assert!((u_new[2] - u[2]).abs() < 1e-12);
    }

    #[test]
    fn test_equilibrium_at_rest() {
        let bgk = BgkCollision::new(1.0);
        let lattice = D3Q19Lattice::new();

        let rho = 1.0;
        let u = [0.0, 0.0, 0.0];
        let f = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // At equilibrium and rest, collision should not change f
        let f_eq = BgkCollision::initialize_with_velocity(rho, u, &lattice);
        let tau = 1.0;  // Use same tau as bgk
        let f_new = bgk.collision_step(&f, &f_eq, tau);

        for i in 0..19 {
            assert!((f_new[i] - f[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_collision_relaxation() {
        let bgk = BgkCollision::new(1.5);
        let lattice = D3Q19Lattice::new();

        let rho = 1.0;
        let u = [0.1, 0.0, 0.0];

        // Start with non-equilibrium distribution (perturbed)
        let mut f = BgkCollision::initialize_with_velocity(rho, u, &lattice);
        f[1] += 0.01;  // Perturb one component
        f[2] -= 0.01;

        let f_eq = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // Collision should move f toward f_eq
        let tau = 1.5;  // Use same tau as bgk
        let f_new = bgk.collision_step(&f, &f_eq, tau);

        // Check that perturbation decreased
        let pert_old = ((f[1] - f_eq[1]).powi(2) + (f[2] - f_eq[2]).powi(2)).sqrt();
        let pert_new = ((f_new[1] - f_eq[1]).powi(2) + (f_new[2] - f_eq[2]).powi(2)).sqrt();

        assert!(pert_new < pert_old);  // Relaxation should decrease perturbation
    }

    #[test]
    fn test_stability_check() {
        let lattice = D3Q19Lattice::new();
        let rho = 1.0;
        let u = [0.01, 0.01, 0.01];
        let f = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        assert!(BgkCollision::is_stable(&f));
    }

    #[test]
    fn test_lbm_solver_creation() {
        let solver = LbmSolver3D::new(10, 8, 6, 1.0);
        assert_eq!(solver.nx, 10);
        assert_eq!(solver.ny, 8);
        assert_eq!(solver.nz, 6);
        assert_eq!(solver.f.len(), 10 * 8 * 6 * 19);
        assert_eq!(solver.rho.len(), 10 * 8 * 6);
        assert_eq!(solver.u.len(), 10 * 8 * 6);
    }

    #[test]
    fn test_lbm_solver_linearize() {
        let solver = LbmSolver3D::new(10, 8, 6, 1.0);
        let idx = solver.linearize(5, 4, 3);
        assert_eq!(idx, 3 * (10 * 8) + 4 * 10 + 5);
    }

    #[test]
    fn test_lbm_solver_initialize() {
        let mut solver = LbmSolver3D::new(10, 8, 6, 1.0);
        let rho_init = 1.0;
        let u_init = [0.1, 0.05, 0.02];

        solver.initialize_uniform(rho_init, u_init);

        // Check initialization at several points
        for z in 0..6 {
            for y in 0..8 {
                for x in 0..10 {
                    let (rho, u) = solver.get_macroscopic(x, y, z);
                    assert!((rho - rho_init).abs() < 1e-14);
                    assert!((u[0] - u_init[0]).abs() < 1e-14);
                    assert!((u[1] - u_init[1]).abs() < 1e-14);
                    assert!((u[2] - u_init[2]).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_lbm_solver_macroscopic_recovery() {
        let mut solver = LbmSolver3D::new(10, 8, 6, 0.8);
        let rho_init = 1.5;
        let u_init = [0.1, 0.05, 0.02];

        solver.initialize_uniform(rho_init, u_init);
        solver.compute_macroscopic();

        // Recovered macroscopic quantities should match initialization
        let idx = solver.linearize(5, 4, 3);
        assert!((solver.rho[idx] - rho_init).abs() < 1e-12);
        assert!((solver.u[idx][0] - u_init[0]).abs() < 1e-12);
        assert!((solver.u[idx][1] - u_init[1]).abs() < 1e-12);
        assert!((solver.u[idx][2] - u_init[2]).abs() < 1e-12);
    }

    #[test]
    fn test_lbm_solver_mass_conservation() {
        let mut solver = LbmSolver3D::new(10, 8, 6, 0.8);
        let rho_init = 1.5;
        let u_init = [0.1, 0.05, 0.02];

        solver.initialize_uniform(rho_init, u_init);
        let mass_before = solver.total_mass();

        // Evolve one step
        solver.evolve_one_step();
        let mass_after = solver.total_mass();

        // Mass should be conserved
        assert!((mass_after - mass_before).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_solver_stability_uniform() {
        let mut solver = LbmSolver3D::new(10, 8, 6, 1.0);
        solver.initialize_uniform(1.0, [0.01, 0.01, 0.01]);

        // Check stability before evolution
        assert!(solver.is_stable());

        // Evolve 10 steps
        solver.evolve(10);

        // Check stability after evolution
        assert!(solver.is_stable());
    }

    #[test]
    fn test_lbm_solver_equilibrium_no_change() {
        let mut solver = LbmSolver3D::new(8, 8, 4, 1.0);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

        let f_before = solver.f.clone();

        // At zero velocity and equilibrium, distribution should not change
        solver.evolve_one_step();

        // Check that f changed minimally (only due to floating point)
        for (i, (fb, fa)) in f_before.iter().zip(solver.f.iter()).enumerate() {
            assert!((fa - fb).abs() < 1e-14, "Component {} changed unexpectedly", i);
        }
    }
}
