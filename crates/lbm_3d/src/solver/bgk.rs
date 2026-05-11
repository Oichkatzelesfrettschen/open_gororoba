//! BGK collision operator for the 3D D3Q19 lattice.
//!
//! The BGK (Bhatnagar-Gross-Krook) collision operator is the standard
//! single-relaxation-time operator for LBM:
//!
//!   f_i^new = f_i - (f_i - f_i^eq) / tau
//!
//! The relaxation time connects to kinematic viscosity via
//! nu = c_s^2 * (tau - 0.5) = (1/3) * (tau - 0.5), so tau >= 0.5 is
//! required for positive viscosity. The MRT alternative lives in the
//! sibling `mrt` submodule.

use crate::lattice::D3Q19Lattice;

use super::{LbmError, Result};

#[inline]
fn sum_19(values: &[f64; 19]) -> f64 {
    values.iter().sum()
}

/// Collision operator selection.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum CollisionMode {
    /// Single-relaxation-time BGK (default, fast but unstable at high density contrast).
    Bgk,
    /// Multiple-relaxation-time d'Humieres (2002): ghost moments relax instantly,
    /// preventing divergence at steep NFW cusps. ~12x more FLOPs per cell but
    /// unconditionally stable for f_i positivity.
    Mrt,
}

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
    pub fn set_viscosity_field(&mut self, tau_field: Vec<f64>) -> Result<()> {
        if tau_field.is_empty() {
            return Err(LbmError::EmptyField);
        }

        for &tau in tau_field.iter() {
            if !tau.is_finite() {
                return Err(LbmError::NonFiniteValue(tau));
            }
            if tau < 0.5 {
                return Err(LbmError::StabilityViolation(tau));
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
        sum_19(f)
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
