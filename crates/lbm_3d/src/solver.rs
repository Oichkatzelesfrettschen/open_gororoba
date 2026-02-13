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
use cosmic_scheduler::{ScheduleResult, TwoPhaseSystem};

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
            tau_field: vec![tau], // Placeholder; solver will set actual field
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
                return Err(format!("Non-finite tau at index {}: {}", i, tau));
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
        self.tau_field
            .iter()
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
            return u; // Zero density => zero velocity
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
    pub fn initialize_with_velocity(rho: f64, u: [f64; 3], lattice: &D3Q19Lattice) -> [f64; 19] {
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
        let tau = if !self.tau_field.is_empty() {
            self.tau_field[0]
        } else {
            0.6
        };

        // Perform collision
        self.collision_step(f, &f_eq, tau)
    }

    /// Check non-negativity of distribution function (stability indicator).
    /// For typical flows at low Mach number, f_i >= 0 always.
    pub fn is_stable(f: &[f64; 19]) -> bool {
        f.iter().all(|&fi| fi >= -1e-14) // Allow small numerical error
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
    /// Optional external body force field (Guo forcing scheme)
    /// If None, no forcing applied. If Some, must have length nx*ny*nz.
    pub force_field: Option<Vec<[f64; 3]>>,
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
        let collider = BgkCollision::new(tau);

        // Initialize populations to equilibrium at rest (rho=1, u=0).
        // f_i^eq(rho=1, u=0) = w_i for all i.
        // This is required for physical correctness: zero populations cause
        // rho=0 which makes velocity undefined and the simulation diverges.
        let lattice = &collider.lattice;
        let mut f = vec![0.0; n_nodes * 19];
        for node in 0..n_nodes {
            let base = node * 19;
            for i in 0..19 {
                f[base + i] = lattice.weight(i);
            }
        }

        Self {
            nx,
            ny,
            nz,
            f,
            rho: vec![1.0; n_nodes],
            u: vec![[0.0; 3]; n_nodes],
            collider,
            force_field: None,
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
                tau_field.len(),
                expected_len,
                self.nx,
                self.ny,
                self.nz
            ));
        }
        self.collider.set_viscosity_field(tau_field)
    }

    /// Get the current viscosity field.
    pub fn get_viscosity_field(&self) -> Vec<f64> {
        self.collider.get_viscosity_field()
    }

    /// Set the external body force field for Guo forcing scheme.
    ///
    /// The Guo forcing method (Guo et al., 2002) adds an external force term to the LBM
    /// collision step, enabling simulation of driven flows (gravity, pressure gradients,
    /// electromagnetic forces, etc.).
    ///
    /// # Arguments
    /// * `force_field` - Vector of force vectors [F_x, F_y, F_z], one per grid point
    ///
    /// # Errors
    /// Returns Err if:
    /// - Force field length != nx*ny*nz
    /// - Any force component is NaN or Inf
    ///
    /// # Example
    /// ```ignore
    /// // Uniform gravity in -z direction
    /// let force = vec![[0.0, 0.0, -0.001]; nx*ny*nz];
    /// solver.set_force_field(force)?;
    /// ```
    pub fn set_force_field(&mut self, force_field: Vec<[f64; 3]>) -> Result<(), String> {
        let expected_len = self.nx * self.ny * self.nz;
        if force_field.len() != expected_len {
            return Err(format!(
                "Force field length mismatch: got {}, expected {} ({}x{}x{})",
                force_field.len(),
                expected_len,
                self.nx,
                self.ny,
                self.nz
            ));
        }

        // Validate all force components are finite
        for (i, &[fx, fy, fz]) in force_field.iter().enumerate() {
            if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
                return Err(format!(
                    "Non-finite force at index {}: [{}, {}, {}]",
                    i, fx, fy, fz
                ));
            }
        }

        self.force_field = Some(force_field);
        Ok(())
    }

    /// Clear the external force field (disable forcing).
    pub fn clear_force_field(&mut self) {
        self.force_field = None;
    }

    /// Check if external forcing is enabled.
    pub fn has_forcing(&self) -> bool {
        self.force_field.is_some()
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
                        if !tau_field.is_empty() {
                            tau_field[0]
                        } else {
                            0.6
                        }
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

                    // Apply Guo forcing if enabled
                    if let Some(ref force_field) = self.force_field {
                        let force = force_field[idx];
                        self.apply_guo_forcing(&mut f_new, self.u[idx], force, tau, &lattice);
                    }

                    // Update distribution function
                    self.f[f_start..f_start + 19].copy_from_slice(&f_new);
                }
            }
        }

        Ok(())
    }

    /// Apply Guo forcing term to post-collision distribution function.
    ///
    /// Implements the Guo et al. (2002) forcing scheme:
    /// delta_f_i = (1 - 1/(2*tau)) * w_i * S_i
    /// where S_i = (e_i - u)*F / c_s^2 + (e_i*u)*(e_i*F) / c_s^4
    ///
    /// This method modifies f_new in-place by adding the forcing contribution.
    ///
    /// # Arguments
    /// * `f_new` - Post-collision distribution (modified in-place)
    /// * `u` - Macroscopic velocity [u_x, u_y, u_z]
    /// * `force` - External force [F_x, F_y, F_z]
    /// * `tau` - Relaxation time at this grid point
    /// * `lattice` - D3Q19 lattice structure (for weights and velocities)
    fn apply_guo_forcing(
        &self,
        f_new: &mut [f64; 19],
        u: [f64; 3],
        force: [f64; 3],
        tau: f64,
        lattice: &D3Q19Lattice,
    ) {
        const CS2: f64 = 1.0 / 3.0; // Speed of sound squared for D3Q19
        const CS4: f64 = 1.0 / 9.0; // c_s^4

        let prefactor = 1.0 - 1.0 / (2.0 * tau);

        for (i, f_i) in f_new.iter_mut().enumerate() {
            // Lattice velocity e_i (cast from i32 to f64)
            let ei = lattice.velocities[i];
            let ei_f64 = [ei[0] as f64, ei[1] as f64, ei[2] as f64];

            // Compute (e_i - u) * F
            let ei_minus_u_dot_f = (ei_f64[0] - u[0]) * force[0]
                + (ei_f64[1] - u[1]) * force[1]
                + (ei_f64[2] - u[2]) * force[2];

            // Compute (e_i * u)
            let ei_dot_u = ei_f64[0] * u[0] + ei_f64[1] * u[1] + ei_f64[2] * u[2];

            // Compute (e_i * F)
            let ei_dot_f = ei_f64[0] * force[0] + ei_f64[1] * force[1] + ei_f64[2] * force[2];

            // Guo forcing term: S_i = (e_i - u)*F / c_s^2 + (e_i*u)*(e_i*F) / c_s^4
            let s_i = ei_minus_u_dot_f / CS2 + (ei_dot_u * ei_dot_f) / CS4;

            // Add forcing contribution: delta_f_i = (1 - 1/(2*tau)) * w_i * S_i
            let delta_f_i = prefactor * lattice.weights[i] * s_i;

            *f_i += delta_f_i;
        }
    }

    /// Phase 2 (streaming): Propagate populations along lattice velocities.
    ///
    /// Each population f_i is shifted to the neighbor in the direction of c_i
    /// with periodic boundary conditions:
    ///   f_i(x + c_i*dt, t + dt) <- f_i(x, t)
    ///
    /// Uses pull scheme: for each destination site and direction, pull from the
    /// source site (destination - c_i) with periodic wrapping.
    pub fn phase2_streaming(&mut self) -> ScheduleResult<()> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let lattice = &self.collider.lattice;

        // Allocate temporary buffer for streamed populations
        let mut f_new = vec![0.0; self.f.len()];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dst_idx = self.linearize(x, y, z);
                    let dst_start = dst_idx * 19;

                    for i in 0..19 {
                        let c = lattice.velocities[i];
                        // Pull scheme: source = destination - velocity
                        let sx = (x as i32 - c[0]).rem_euclid(nx as i32) as usize;
                        let sy = (y as i32 - c[1]).rem_euclid(ny as i32) as usize;
                        let sz = (z as i32 - c[2]).rem_euclid(nz as i32) as usize;
                        let src_idx = self.linearize(sx, sy, sz);
                        f_new[dst_start + i] = self.f[src_idx * 19 + i];
                    }
                }
            }
        }

        self.f = f_new;
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

    /// Evolve with dynamic non-Newtonian viscosity feedback.
    ///
    /// At each timestep, the local viscosity (tau) is updated based on the
    /// current strain rate and a per-cell coupling field. This enables genuine
    /// shear-thickening or shear-thinning behavior where the viscosity depends
    /// on the flow itself.
    ///
    /// The viscosity update at each cell follows:
    ///   nu_eff = nu_base * (1 + coupling[i] * (|gamma_dot| + eps)^(power_index - 1))
    ///   tau_new = 3 * nu_eff + 0.5, clamped to [tau_min, tau_max]
    ///
    /// # Arguments
    /// * `num_steps` - Number of timesteps to evolve
    /// * `coupling_field` - Per-cell coupling strength (e.g., associator norm)
    /// * `nu_base` - Base kinematic viscosity
    /// * `power_index` - Power-law exponent (n>1 = thickening, n<1 = thinning)
    /// * `tau_min` - Lower clamp for stability (>= 0.505)
    /// * `tau_max` - Upper clamp for stability
    ///
    /// Returns the final strain rate field for analysis.
    pub fn evolve_non_newtonian(
        &mut self,
        num_steps: usize,
        coupling_field: &[f64],
        nu_base: f64,
        power_index: f64,
        tau_min: f64,
        tau_max: f64,
    ) -> Vec<f64> {
        const EPS: f64 = 1e-10;
        let n_nodes = self.nx * self.ny * self.nz;
        assert_eq!(coupling_field.len(), n_nodes);
        let tau_min = tau_min.max(0.505); // Hard floor for stability

        let mut strain_rate = vec![0.0; n_nodes];

        for step in 0..num_steps {
            // Standard collision + streaming
            self.evolve_one_step();

            // Update tau based on strain rate every step (or periodically for perf)
            // First few steps: let flow develop before feedback kicks in
            if step >= 10 {
                strain_rate = self.compute_strain_rate_field();

                let mut new_tau = Vec::with_capacity(n_nodes);
                for i in 0..n_nodes {
                    let sr = strain_rate[i];
                    let coupling = coupling_field[i];
                    let strain_term = (sr + EPS).powf(power_index - 1.0);
                    let nu_eff = nu_base * (1.0 + coupling * strain_term);
                    let tau = (3.0 * nu_eff + 0.5).clamp(tau_min, tau_max);
                    new_tau.push(tau);
                }

                // Update the tau field for next collision step
                let _ = self.collider.set_viscosity_field(new_tau);
            }
        }

        strain_rate
    }

    /// Compute the strain rate magnitude field |gamma_dot| at each grid point.
    ///
    /// The strain rate tensor is:
    ///   e_ab = (du_a/dx_b + du_b/dx_a) / 2
    ///
    /// and the scalar magnitude is:
    ///   |gamma_dot| = sqrt(2 * sum_{a,b} e_ab^2)
    ///
    /// Derivatives use central finite differences with periodic boundary conditions.
    pub fn compute_strain_rate_field(&self) -> Vec<f64> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let n_nodes = nx * ny * nz;
        let mut strain_rate = vec![0.0; n_nodes];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.linearize(x, y, z);

                    // Neighbor indices with periodic BC
                    let xp = (x + 1) % nx;
                    let xm = (x + nx - 1) % nx;
                    let yp = (y + 1) % ny;
                    let ym = (y + ny - 1) % ny;
                    let zp = (z + 1) % nz;
                    let zm = (z + nz - 1) % nz;

                    // Velocity gradient tensor du_a / dx_b (central differences)
                    let u_xp = self.u[self.linearize(xp, y, z)];
                    let u_xm = self.u[self.linearize(xm, y, z)];
                    let u_yp = self.u[self.linearize(x, yp, z)];
                    let u_ym = self.u[self.linearize(x, ym, z)];
                    let u_zp = self.u[self.linearize(x, y, zp)];
                    let u_zm = self.u[self.linearize(x, y, zm)];

                    // du_a/dx (3 components)
                    let du_dx = [
                        (u_xp[0] - u_xm[0]) / 2.0,
                        (u_xp[1] - u_xm[1]) / 2.0,
                        (u_xp[2] - u_xm[2]) / 2.0,
                    ];
                    // du_a/dy (3 components)
                    let du_dy = [
                        (u_yp[0] - u_ym[0]) / 2.0,
                        (u_yp[1] - u_ym[1]) / 2.0,
                        (u_yp[2] - u_ym[2]) / 2.0,
                    ];
                    // du_a/dz (3 components)
                    let du_dz = [
                        (u_zp[0] - u_zm[0]) / 2.0,
                        (u_zp[1] - u_zm[1]) / 2.0,
                        (u_zp[2] - u_zm[2]) / 2.0,
                    ];

                    // Strain rate tensor e_ab = (du_a/dx_b + du_b/dx_a) / 2
                    // Symmetric 3x3 tensor: e[0][0], e[0][1], e[0][2], e[1][1], e[1][2], e[2][2]
                    let e00 = du_dx[0]; // du_x/dx
                    let e11 = du_dy[1]; // du_y/dy
                    let e22 = du_dz[2]; // du_z/dz
                    let e01 = (du_dy[0] + du_dx[1]) / 2.0; // (du_x/dy + du_y/dx) / 2
                    let e02 = (du_dz[0] + du_dx[2]) / 2.0; // (du_x/dz + du_z/dx) / 2
                    let e12 = (du_dz[1] + du_dy[2]) / 2.0; // (du_y/dz + du_z/dy) / 2

                    // |gamma_dot| = sqrt(2 * (e00^2 + e11^2 + e22^2 + 2*e01^2 + 2*e02^2 + 2*e12^2))
                    let sum_sq = e00 * e00
                        + e11 * e11
                        + e22 * e22
                        + 2.0 * (e01 * e01 + e02 * e02 + e12 * e12);
                    strain_rate[idx] = (2.0 * sum_sq).sqrt();
                }
            }
        }

        strain_rate
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
        self.u
            .iter()
            .map(|ui| ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2])
            .sum::<f64>()
            .sqrt()
    }

    /// Compute max velocity magnitude in the domain.
    pub fn max_velocity(&self) -> f64 {
        self.u
            .iter()
            .map(|ui| (ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2]).sqrt())
            .fold(0.0_f64, f64::max)
    }

    /// Compute mean velocity magnitude in the domain.
    pub fn mean_velocity(&self) -> f64 {
        let n = self.u.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        self.u
            .iter()
            .map(|ui| (ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2]).sqrt())
            .sum::<f64>()
            / n
    }

    /// Check the CFL condition: max velocity should be well below the lattice
    /// speed of sound c_s = 1/sqrt(3) ~ 0.577 for numerical stability.
    ///
    /// Returns (max_velocity, cfl_ratio) where cfl_ratio = max_velocity / c_s.
    /// A cfl_ratio > 0.3 is a warning; > 0.5 risks instability.
    pub fn cfl_check(&self) -> (f64, f64) {
        let cs = (1.0_f64 / 3.0).sqrt();
        let v_max = self.max_velocity();
        (v_max, v_max / cs)
    }

    /// Check if the velocity field has converged to steady state.
    ///
    /// Compares current velocity against a reference field. Returns true if
    /// max |u(current) - u(reference)| < tol.
    pub fn is_converged(&self, reference_u: &[[f64; 3]], tol: f64) -> bool {
        if reference_u.len() != self.u.len() {
            return false;
        }
        self.u
            .iter()
            .zip(reference_u.iter())
            .all(|(curr, prev)| {
                let dx = curr[0] - prev[0];
                let dy = curr[1] - prev[1];
                let dz = curr[2] - prev[2];
                (dx * dx + dy * dy + dz * dz).sqrt() < tol
            })
    }

    /// Evolve with convergence monitoring.
    ///
    /// Runs up to `max_steps` LBM timesteps, checking convergence every
    /// `check_interval` steps. Stops early if the velocity field converges
    /// within tolerance `tol`.
    ///
    /// Returns a `ConvergenceReport` with diagnostics from the evolution.
    pub fn evolve_with_diagnostics(
        &mut self,
        max_steps: usize,
        check_interval: usize,
        tol: f64,
    ) -> ConvergenceReport {
        let initial_mass = self.total_mass();
        let mut snapshots = Vec::new();
        let mut prev_u = self.u.clone();
        let mut converged = false;
        let mut steps_taken = 0;

        for step in 0..max_steps {
            self.evolve_one_step();
            steps_taken = step + 1;

            if steps_taken % check_interval == 0 || step == max_steps - 1 {
                let current_mass = self.total_mass();
                let (v_max, cfl) = self.cfl_check();

                snapshots.push(ConvergenceSnapshot {
                    step: steps_taken,
                    mass_error: (current_mass - initial_mass).abs()
                        / initial_mass.abs().max(1e-30),
                    max_velocity: v_max,
                    mean_velocity: self.mean_velocity(),
                    cfl_ratio: cfl,
                    stable: self.is_stable(),
                });

                if self.is_converged(&prev_u, tol) {
                    converged = true;
                    break;
                }

                prev_u.clone_from(&self.u);

                // Bail if unstable
                if !self.is_stable() || cfl > 0.8 {
                    break;
                }
            }
        }

        ConvergenceReport {
            steps_taken,
            converged,
            initial_mass,
            final_mass: self.total_mass(),
            snapshots,
        }
    }
}

/// Snapshot of convergence diagnostics at one point in time.
#[derive(Debug, Clone)]
pub struct ConvergenceSnapshot {
    /// Timestep number
    pub step: usize,
    /// Relative mass conservation error |M(t) - M(0)| / M(0)
    pub mass_error: f64,
    /// Max velocity magnitude in domain
    pub max_velocity: f64,
    /// Mean velocity magnitude in domain
    pub mean_velocity: f64,
    /// CFL ratio: max_velocity / c_s (warn if > 0.3, unstable if > 0.5)
    pub cfl_ratio: f64,
    /// Whether all distribution values are non-negative
    pub stable: bool,
}

/// Report from evolve_with_diagnostics.
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    /// Number of timesteps actually taken
    pub steps_taken: usize,
    /// Whether steady state was reached within tolerance
    pub converged: bool,
    /// Initial total mass
    pub initial_mass: f64,
    /// Final total mass
    pub final_mass: f64,
    /// Diagnostic snapshots taken during evolution
    pub snapshots: Vec<ConvergenceSnapshot>,
}

impl ConvergenceReport {
    /// Check if mass was conserved to the given relative tolerance.
    pub fn mass_conserved(&self, tol: f64) -> bool {
        let err = (self.final_mass - self.initial_mass).abs()
            / self.initial_mass.abs().max(1e-30);
        err < tol
    }

    /// Was the simulation stable throughout?
    pub fn always_stable(&self) -> bool {
        self.snapshots.iter().all(|s| s.stable)
    }

    /// Max CFL ratio observed during evolution.
    pub fn max_cfl(&self) -> f64 {
        self.snapshots
            .iter()
            .map(|s| s.cfl_ratio)
            .fold(0.0_f64, f64::max)
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
            return Err(cosmic_scheduler::ScheduleError::StateInvalid(format!(
                "LBM population instability: negative f_i detected at timestep {}",
                self.timestep
            )));
        }

        // Check non-negativity of density field
        for (i, &rho_i) in self.rho.iter().enumerate() {
            if rho_i < -1e-14 {
                return Err(cosmic_scheduler::ScheduleError::StateInvalid(format!(
                    "Negative density at node {}: {} at timestep {}",
                    i, rho_i, self.timestep
                )));
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
        let tau = 1.0; // Use same tau as bgk
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
        f[1] += 0.01; // Perturb one component
        f[2] -= 0.01;

        let f_eq = BgkCollision::initialize_with_velocity(rho, u, &lattice);

        // Collision should move f toward f_eq
        let tau = 1.5; // Use same tau as bgk
        let f_new = bgk.collision_step(&f, &f_eq, tau);

        // Check that perturbation decreased
        let pert_old = ((f[1] - f_eq[1]).powi(2) + (f[2] - f_eq[2]).powi(2)).sqrt();
        let pert_new = ((f_new[1] - f_eq[1]).powi(2) + (f_new[2] - f_eq[2]).powi(2)).sqrt();

        assert!(pert_new < pert_old); // Relaxation should decrease perturbation
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
    fn test_lbm_solver_equilibrium_initialization() {
        let solver = LbmSolver3D::new(4, 4, 4, 0.8);
        let lattice = D3Q19Lattice::new();

        // All cells should be initialized to equilibrium at rho=1, u=0
        for node in 0..64 {
            assert!((solver.rho[node] - 1.0).abs() < 1e-14, "rho not 1.0 at node {}", node);
            for k in 0..3 {
                assert!(solver.u[node][k].abs() < 1e-14, "u not 0 at node {}", node);
            }
            // f_i should equal w_i (equilibrium at rest with rho=1)
            let base = node * 19;
            for i in 0..19 {
                let expected = lattice.weight(i);
                assert!(
                    (solver.f[base + i] - expected).abs() < 1e-14,
                    "f[{}] = {} != w[{}] = {} at node {}",
                    i, solver.f[base + i], i, expected, node
                );
            }
        }

        // Total mass should be n_nodes (each cell has rho=1)
        assert!((solver.total_mass() - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_lbm_uniform_force_develops_flow() {
        // With equilibrium init and uniform force, velocity should grow
        let mut solver = LbmSolver3D::new(8, 8, 8, 0.8);
        let n = 8 * 8 * 8;
        let force = vec![[1e-5, 0.0, 0.0]; n];
        solver.set_force_field(force).unwrap();

        solver.evolve(50);

        let v_max = solver.max_velocity();
        assert!(v_max > 1e-6, "Force should produce nonzero velocity, got {}", v_max);
        assert!(v_max < 0.1, "Velocity should be small and stable, got {}", v_max);

        // Mass should be conserved
        assert!((solver.total_mass() - n as f64).abs() / (n as f64) < 1e-6);
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

        // At zero velocity and equilibrium, uniform distribution is invariant
        // under both collision (f = f_eq) and streaming (spatially uniform).
        solver.evolve_one_step();

        // Check that f changed minimally (only due to floating point)
        for (i, (fb, fa)) in f_before.iter().zip(solver.f.iter()).enumerate() {
            assert!(
                (fa - fb).abs() < 1e-13,
                "Component {} changed unexpectedly: before={}, after={}",
                i,
                fb,
                fa
            );
        }
    }

    #[test]
    fn test_streaming_propagates_perturbation() {
        // Place a density perturbation at one site and verify it moves
        let mut solver = LbmSolver3D::new(8, 8, 8, 1.0);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

        // Perturb f at site (3, 3, 3) for direction 1 (velocity [1,0,0])
        let src_idx = solver.linearize(3, 3, 3);
        solver.f[src_idx * 19 + 1] += 0.01;

        // Run streaming only (skip collision to isolate streaming effect)
        let _ = solver.phase2_streaming();

        // After streaming, the perturbation in direction 1 should have moved to (4,3,3)
        let dst_idx = solver.linearize(4, 3, 3);
        let original_val = 1.0 * solver.collider.lattice.weight(1);
        let delta = solver.f[dst_idx * 19 + 1] - original_val;
        assert!(
            delta.abs() > 0.005,
            "Perturbation should propagate: delta = {}",
            delta
        );
    }

    #[test]
    fn test_streaming_periodic_wrapping() {
        let mut solver = LbmSolver3D::new(4, 4, 4, 1.0);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

        // Perturb at edge site (3,2,2) in direction 1 (velocity [1,0,0])
        let edge_idx = solver.linearize(3, 2, 2);
        solver.f[edge_idx * 19 + 1] += 0.02;

        let _ = solver.phase2_streaming();

        // Should wrap to (0, 2, 2)
        let wrap_idx = solver.linearize(0, 2, 2);
        let original_val = 1.0 * solver.collider.lattice.weight(1);
        let delta = solver.f[wrap_idx * 19 + 1] - original_val;
        assert!(
            delta.abs() > 0.01,
            "Should wrap periodically: delta = {}",
            delta
        );
    }

    #[test]
    fn test_streaming_mass_conservation() {
        let mut solver = LbmSolver3D::new(8, 8, 8, 0.8);
        solver.initialize_uniform(1.5, [0.1, 0.05, 0.02]);

        let mass_before: f64 = solver.f.iter().sum();
        let _ = solver.phase2_streaming();
        let mass_after: f64 = solver.f.iter().sum();

        assert!(
            (mass_after - mass_before).abs() < 1e-10,
            "Streaming must conserve mass: before={}, after={}",
            mass_before,
            mass_after
        );
    }

    #[test]
    fn test_full_evolution_develops_velocity() {
        // With Guo forcing, the solver should develop actual flow
        let mut solver = LbmSolver3D::new(8, 8, 8, 0.8);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

        // Set spatially-varying tau field
        let n_nodes = 8 * 8 * 8;
        let tau_field = vec![0.8; n_nodes];
        solver.set_viscosity_field(tau_field).unwrap();

        // Apply a constant body force in x-direction
        let force = vec![[1e-4, 0.0, 0.0]; n_nodes];
        solver.set_force_field(force).unwrap();

        solver.evolve(50);

        // Velocity should develop in x-direction
        let max_ux: f64 = solver
            .u
            .iter()
            .map(|u| u[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_ux > 1e-6,
            "Flow should develop with forcing: max |ux| = {:.2e}",
            max_ux
        );
    }

    #[test]
    fn test_strain_rate_zero_velocity() {
        let mut solver = LbmSolver3D::new(8, 8, 8, 1.0);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
        solver.compute_macroscopic();

        let strain = solver.compute_strain_rate_field();
        for &s in &strain {
            assert!(
                s.abs() < 1e-14,
                "Strain rate should be zero for uniform field"
            );
        }
    }

    #[test]
    fn test_strain_rate_shear_flow() {
        // Set up a simple shear flow: u_x = y / ny (linear in y)
        let (nx, ny, nz) = (4, 8, 4);
        let mut solver = LbmSolver3D::new(nx, ny, nz, 1.0);
        solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

        // Override velocity to create shear
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * (nx * ny) + y * nx + x;
                    solver.u[idx] = [y as f64 / ny as f64, 0.0, 0.0];
                }
            }
        }

        let strain = solver.compute_strain_rate_field();

        // Interior points should have nonzero strain rate
        let interior_strain = strain[solver.linearize(2, 4, 2)];
        assert!(
            interior_strain > 1e-4,
            "Shear flow should produce nonzero strain: {}",
            interior_strain
        );
    }

    #[test]
    fn test_strain_rate_is_finite() {
        let mut solver = LbmSolver3D::new(8, 8, 8, 0.8);
        solver.initialize_uniform(1.0, [0.05, 0.02, 0.01]);
        solver.evolve(10);

        let strain = solver.compute_strain_rate_field();
        for &s in &strain {
            assert!(s.is_finite(), "Strain rate must be finite");
            assert!(s >= 0.0, "Strain rate magnitude must be non-negative");
        }
    }

    #[test]
    fn test_non_newtonian_evolution_stability() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let mut solver = LbmSolver3D::new(nx, ny, nz, 0.8);

        let force = vec![[1e-5, 0.0, 0.0]; n];
        solver.set_force_field(force).unwrap();

        // Coupling field: uniform moderate coupling
        let coupling = vec![0.5; n];

        let strain = solver.evolve_non_newtonian(100, &coupling, 0.1, 1.5, 0.505, 2.0);

        // Should produce stable flow
        let v_max = solver.max_velocity();
        assert!(v_max > 1e-6, "Should develop flow: v_max={}", v_max);
        assert!(v_max < 0.3, "Should remain stable: v_max={}", v_max);

        // Mass should be conserved
        let mass = solver.total_mass();
        assert!((mass - n as f64).abs() / (n as f64) < 1e-4, "Mass not conserved: {}", mass);

        // Strain rate should be finite and non-negative
        for &s in &strain {
            assert!(s.is_finite());
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_non_newtonian_thickening_increases_tau() {
        // Sinusoidal (Kolmogorov) forcing creates shear flow with velocity
        // gradients, enabling strain-rate-dependent viscosity effects.
        // Uniform forcing produces uniform velocity (zero gradients, zero effect).
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let nu_base = 0.05;
        let pi2 = std::f64::consts::PI * 2.0;

        // Kolmogorov forcing: F_x = A * sin(2*pi*y/ny)
        let amplitude = 5e-4;
        let mut force = vec![[0.0, 0.0, 0.0]; n];
        for z in 0..nz {
            for y in 0..ny {
                let fy = amplitude * (pi2 * y as f64 / ny as f64).sin();
                for x in 0..nx {
                    let idx = z * nx * ny + y * nx + x;
                    force[idx] = [fy, 0.0, 0.0];
                }
            }
        }

        // Newtonian reference: zero coupling
        let mut solver_newton = LbmSolver3D::new(nx, ny, nz, 3.0 * nu_base + 0.5);
        solver_newton.set_force_field(force.clone()).unwrap();
        let coupling_zero = vec![0.0; n];
        solver_newton.evolve_non_newtonian(300, &coupling_zero, nu_base, 2.0, 0.505, 3.0);
        let v_newton = solver_newton.max_velocity();

        // Non-Newtonian with strong shear-thickening (n=2.0)
        let mut solver_thick = LbmSolver3D::new(nx, ny, nz, 3.0 * nu_base + 0.5);
        solver_thick.set_force_field(force).unwrap();
        let coupling = vec![100.0; n];
        solver_thick.evolve_non_newtonian(300, &coupling, nu_base, 2.0, 0.505, 3.0);
        let v_thick = solver_thick.max_velocity();

        // Shear-thickening should produce LOWER velocity (higher effective viscosity)
        assert!(
            v_thick < v_newton * 0.95,
            "Thickening should reduce velocity by >5%: v_thick={} vs v_newton={}",
            v_thick, v_newton
        );
    }

    #[test]
    fn test_non_newtonian_tau_field_updated() {
        let (nx, ny, nz) = (8, 8, 8);
        let n = nx * ny * nz;
        let mut solver = LbmSolver3D::new(nx, ny, nz, 0.8);

        let force = vec![[1e-4, 0.0, 0.0]; n];
        solver.set_force_field(force).unwrap();

        // Non-uniform coupling: gradient in x
        let mut coupling = vec![0.0; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * nx * ny + y * nx + x;
                    coupling[idx] = x as f64 / nx as f64;
                }
            }
        }

        solver.evolve_non_newtonian(50, &coupling, 0.1, 1.5, 0.505, 2.0);

        // After non-Newtonian evolution, tau field should vary spatially
        let tau = solver.collider.get_tau_field();
        assert_eq!(tau.len(), n);
        let tau_min = tau.iter().cloned().fold(f64::INFINITY, f64::min);
        let tau_max = tau.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            tau_max - tau_min > 1e-6,
            "Tau should vary spatially: min={}, max={}",
            tau_min, tau_max
        );
    }
}
