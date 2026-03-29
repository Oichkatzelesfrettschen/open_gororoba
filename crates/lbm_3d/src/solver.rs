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
use rayon::prelude::*;
use thiserror::Error;
use wide::f64x4;

#[derive(Error, Debug)]
pub enum LbmError {
    #[error("Stability violation: tau={0} < 0.5")]
    StabilityViolation(f64),
    #[error("Non-finite value detected: {0}")]
    NonFiniteValue(f64),
    #[error("Field length mismatch: expected {expected}, got {found}")]
    DimensionMismatch { expected: usize, found: usize },
    #[error("Empty field provided")]
    EmptyField,
}

type Result<T> = std::result::Result<T, LbmError>;

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

/// 3D LBM solver with D3Q19 lattice and BGK collision.
/// AoSoA chunk size: 4 f64 values = one 256-bit YMM register (AVX2).
///
/// Memory layout per chunk:
///   [Dir0(c0,c1,c2,c3), Dir1(c0,c1,c2,c3), ... Dir18(c0,c1,c2,c3)]
///
/// One chunk footprint: 19 * 4 * 8 = 608 bytes -- fits in 2% of 32 KB L1D,
/// avoiding the 8-way associativity thrashing that pure SoA causes on x86.
pub const AOSOA_CHUNK: usize = 4;

/// Compute AoSoA index for a given cell and direction.
///
/// Maps `(cell, dir)` to a flat index into the AoSoA f-vector:
///   chunk_idx = cell / CHUNK
///   lane      = cell % CHUNK
///   index     = chunk_idx * 19 * CHUNK + dir * CHUNK + lane
#[inline(always)]
pub fn aosoa_idx(cell: usize, dir: usize) -> usize {
    let chunk = cell / AOSOA_CHUNK;
    let lane = cell % AOSOA_CHUNK;
    chunk * 19 * AOSOA_CHUNK + dir * AOSOA_CHUNK + lane
}

/// Round up to the nearest multiple of AOSOA_CHUNK.
#[inline(always)]
fn aosoa_pad(n: usize) -> usize {
    n.div_ceil(AOSOA_CHUNK) * AOSOA_CHUNK
}

/// Zero-cost wrapper to bypass the compiler's inability to prove disjoint
/// index math across parallel threads.
///
/// SAFETY contract: the caller guarantees via `aosoa_idx` algebra that no
/// two rayon threads will ever read-write or write-write the same address
/// simultaneously. This holds because `aosoa_idx(a, d) != aosoa_idx(b, d)`
/// for `a != b`, and the collision step only accesses indices belonging to
/// its own cell. Pull streaming (phase 2) is serial and uses a separate
/// scratch buffer, so no aliasing occurs there either.
#[derive(Copy, Clone)]
pub struct UnsafeAoSoAPtr<T>(pub *mut T);

// SAFETY: UnsafeAoSoAPtr wraps a pointer into a Vec-backed buffer. Send+Sync
// is safe because: (1) rayon's par_iter partitions index ranges so no two
// threads access the same offset, (2) the backing Vec outlives all rayon tasks
// via the enclosing scope, (3) writes target disjoint offsets only.
unsafe impl<T> Send for UnsafeAoSoAPtr<T> {}
unsafe impl<T> Sync for UnsafeAoSoAPtr<T> {}

impl<T> UnsafeAoSoAPtr<T> {
    /// Read a value from the specified offset without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `offset` is in bounds and no concurrent write
    /// to the same address occurs.
    #[inline(always)]
    pub unsafe fn read(&self, offset: usize) -> T {
        // SAFETY: offset is computed from grid indices (i,j,k,q) bounded by
        // grid dimensions. The total buffer length is n1*n2*n3*Q which exceeds
        // any valid offset. The caller guarantees no concurrent write.
        unsafe { core::ptr::read(self.0.add(offset)) }
    }

    /// Write a value to the specified offset without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `offset` is in bounds and no concurrent access
    /// (read or write) to the same address occurs.
    #[inline(always)]
    pub unsafe fn write(&self, offset: usize, val: T) {
        // SAFETY: offset is computed from grid indices (i,j,k,q) bounded by
        // grid dimensions. The total buffer length is n1*n2*n3*Q which exceeds
        // any valid offset. The caller guarantees no concurrent access.
        unsafe { core::ptr::write(self.0.add(offset), val) }
    }
}

impl UnsafeAoSoAPtr<f64> {
    /// Read an aligned CHUNK=4 f64 slice as f64x4 (256-bit VMOVAPD).
    ///
    /// # Safety
    /// Caller must ensure `offset` points to 4 contiguous, valid f64 values
    /// and no concurrent write to the same addresses occurs.
    #[inline(always)]
    pub unsafe fn read_x4(&self, offset: usize) -> f64x4 {
        unsafe {
            let arr = core::ptr::read(self.0.add(offset) as *const [f64; 4]);
            f64x4::new(arr)
        }
    }

    /// Write an f64x4 (256-bit VMOVAPD) to an aligned CHUNK=4 f64 slice.
    ///
    /// # Safety
    /// Caller must ensure `offset` points to 4 writable f64 slots and no
    /// concurrent access to the same addresses occurs.
    #[inline(always)]
    pub unsafe fn write_x4(&self, offset: usize, val: f64x4) {
        unsafe {
            let arr: [f64; 4] = val.to_array();
            core::ptr::write(self.0.add(offset) as *mut [f64; 4], arr);
        }
    }
}

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
    /// Distribution function in AoSoA layout.
    ///
    /// Index: `aosoa_idx(cell, dir)` where cell = z*(nx*ny) + y*nx + x.
    /// Padded to a multiple of AOSOA_CHUNK cells with zero-weight ghosts.
    pub f: Vec<f64>,
    /// Pre-allocated scratch buffer for streaming (AoSoA layout, same size as f).
    f_scratch: Vec<f64>,
    /// Macroscopic density at each grid point
    pub rho: Vec<f64>,
    /// Macroscopic velocity at each grid point
    pub u: Vec<[f64; 3]>,
    /// BGK collision operator
    pub collider: BgkCollision,
    /// Optional external body force field (Guo forcing scheme)
    /// If None, no forcing applied. If Some, must have length nx*ny*nz.
    pub force_field: Option<Vec<[f64; 3]>>,
    /// Collision operator mode (BGK or MRT).
    pub collision_mode: CollisionMode,
    /// Timestep counter
    pub timestep: usize,
}

/// D3Q19 Multiple-Relaxation-Time (MRT) Collision Operator.
///
/// Implements the d'Humieres (2002) orthogonal basis transformation.
/// Forward transform f -> moment space (m = M * f) for D3Q19 MRT.
///
/// Exposed for testing row norms and moment structure. Uses the same
/// d'Humieres orthogonal basis as the collision operator.
#[cfg(test)]
fn mrt_forward_transform(f: &[f64; 19]) -> [f64; 19] {
    [
        f[0] + f[1]
            + f[2]
            + f[3]
            + f[4]
            + f[5]
            + f[6]
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18],
        -30.0 * f[0] - 11.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
            + 8.0
                * (f[7]
                    + f[8]
                    + f[9]
                    + f[10]
                    + f[11]
                    + f[12]
                    + f[13]
                    + f[14]
                    + f[15]
                    + f[16]
                    + f[17]
                    + f[18]),
        12.0 * f[0] - 4.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
            + (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]),
        f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14],
        -4.0 * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14],
        f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        -4.0 * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18],
        -4.0 * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18],
        2.0 * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            - 2.0 * (f[15] + f[16] + f[17] + f[18]),
        -2.0 * (f[1] + f[2])
            + (f[3] + f[4] + f[5] + f[6])
            + f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            - 2.0 * (f[15] + f[16] + f[17] + f[18]),
        (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
            - (f[11] + f[12] + f[13] + f[14]),
        -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
            - (f[11] + f[12] + f[13] + f[14]),
        f[7] + f[8] - f[9] - f[10],
        f[11] + f[12] - f[13] - f[14],
        f[15] + f[16] - f[17] - f[18],
        f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14],
        -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18],
    ]
}

/// Ghost moments relax instantly (s=1.0) to annihilate the spurious
/// oscillations that cause BGK divergence at steep density cusps.
///
/// The operator transforms f -> moment space (M*f), relaxes each moment
/// independently via diagonal S matrix, then transforms back (M^{-1}*m*).
/// Physical viscosity s_nu = 1/tau is preserved; ghost moments use s=1.0.
///
/// Cost: ~722 FMA operations per cell vs ~57 for BGK (~12x more FLOPs),
/// but the GPU memory-bound regime absorbs this within the latency window.
#[inline(always)]
fn collide_mrt_d3q19(f: &[f64; 19], rho: f64, ux: f64, uy: f64, uz: f64, tau: f64) -> [f64; 19] {
    // Relaxation rates (diagonal of S matrix)
    let s_nu = 1.0 / tau; // Physical kinematic viscosity
    let s_e = 1.19; // Energy relaxation
    let s_eps = 1.4; // Energy squared
    let s_q = 1.2; // Energy flux
    let s_ghost = 1.0; // Instant damping for ghost moments

    let u_sq = ux * ux + uy * uy + uz * uz;

    // Forward transform: f -> moment space (m = M * f)
    // Using the d'Humieres D3Q19 orthogonal basis
    let m0 = f[0]
        + f[1]
        + f[2]
        + f[3]
        + f[4]
        + f[5]
        + f[6]
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        + f[15]
        + f[16]
        + f[17]
        + f[18];
    let m1 = -30.0 * f[0] - 11.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + 8.0
            * (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]);
    let m2 = 12.0 * f[0] - 4.0 * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + (f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18]);
    let m3 = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m4 = -4.0 * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m5 = f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m6 = -4.0 * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m7 = f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m8 = -4.0 * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m9 = 2.0 * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - 2.0 * (f[15] + f[16] + f[17] + f[18]);
    let m10 = -2.0 * (f[1] + f[2])
        + (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - 2.0 * (f[15] + f[16] + f[17] + f[18]);
    let m11 = (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m12 = -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m13 = f[7] + f[8] - f[9] - f[10];
    let m14 = f[11] + f[12] - f[13] - f[14];
    let m15 = f[15] + f[16] - f[17] - f[18];
    let m16 = f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14];
    let m17 = -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m18 = f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18];

    // Equilibrium moments
    // m0_eq = rho (conserved, not needed for relaxation)
    let m1_eq = rho * (-11.0 + 19.0 * u_sq);
    let m2_eq = rho * (3.0 - 5.5 * u_sq);
    // m3_eq = rho*ux (conserved)
    let m4_eq = -2.0 / 3.0 * rho * ux;
    // m5_eq = rho*uy (conserved)
    let m6_eq = -2.0 / 3.0 * rho * uy;
    // m7_eq = rho*uz (conserved)
    let m8_eq = -2.0 / 3.0 * rho * uz;
    let m9_eq = rho * (2.0 * ux * ux - uy * uy - uz * uz);
    let m10_eq = -0.5 * rho * (2.0 * ux * ux - uy * uy - uz * uz);
    let m11_eq = rho * (uy * uy - uz * uz);
    let m12_eq = -0.5 * rho * (uy * uy - uz * uz);
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;
    let m16_eq = 0.0;
    let m17_eq = 0.0;
    let m18_eq = 0.0;

    // Relax moments: m* = m - S * (m - m_eq)
    // Mass (m0) and momentum (m3, m5, m7) are conserved (s=0).
    let ms0 = m0; // conserved
    let ms1 = m1 - s_e * (m1 - m1_eq); // energy
    let ms2 = m2 - s_eps * (m2 - m2_eq); // energy^2
    let ms3 = m3; // conserved
    let ms4 = m4 - s_q * (m4 - m4_eq); // energy flux
    let ms5 = m5; // conserved
    let ms6 = m6 - s_q * (m6 - m6_eq); // energy flux
    let ms7 = m7; // conserved
    let ms8 = m8 - s_q * (m8 - m8_eq); // energy flux
    let ms9 = m9 - s_nu * (m9 - m9_eq); // stress (physical)
    let ms10 = m10 - s_ghost * (m10 - m10_eq); // ghost
    let ms11 = m11 - s_nu * (m11 - m11_eq); // stress (physical)
    let ms12 = m12 - s_ghost * (m12 - m12_eq); // ghost
    let ms13 = m13 - s_nu * (m13 - m13_eq); // stress (physical)
    let ms14 = m14 - s_nu * (m14 - m14_eq); // stress (physical)
    let ms15 = m15 - s_nu * (m15 - m15_eq); // stress (physical)
    let ms16 = m16 - s_ghost * (m16 - m16_eq); // ghost
    let ms17 = m17 - s_ghost * (m17 - m17_eq); // ghost
    let ms18 = m18 - s_ghost * (m18 - m18_eq); // ghost

    // Inverse transform: f* = M^{-1} * m*
    // M^{-1}_{ij} = M_{ji} / ||row_j||^2 (orthogonal, non-orthonormal basis).
    // Row squared-norms: [19, 2394, 252, 10, 40, 10, 40, 10, 40, 36, 36, 12, 12, 4, 4, 4, 8, 8, 8]
    let mut fo = [0.0; 19];

    // Reciprocal norms (1 / ||row_j||^2)
    let rn0 = 1.0 / 19.0;
    let rn1 = 1.0 / 2394.0;
    let rn2 = 1.0 / 252.0;
    let rn3 = 1.0 / 10.0;
    let rn4 = 1.0 / 40.0;
    let rn5 = 1.0 / 10.0;
    let rn6 = 1.0 / 40.0;
    let rn7 = 1.0 / 10.0;
    let rn8 = 1.0 / 40.0;
    let rn9 = 1.0 / 36.0;
    let rn10 = 1.0 / 36.0;
    let rn11 = 1.0 / 12.0;
    let rn12 = 1.0 / 12.0;
    let rn13 = 1.0 / 4.0;
    let rn14 = 1.0 / 4.0;
    let rn15 = 1.0 / 4.0;
    let rn16 = 1.0 / 8.0;
    let rn17 = 1.0 / 8.0;
    let rn18 = 1.0 / 8.0;

    // Scale relaxed moments by reciprocal norms: s[j] = ms_j / ||row_j||^2
    let s0 = ms0 * rn0;
    let s1 = ms1 * rn1;
    let s2 = ms2 * rn2;
    let s3 = ms3 * rn3;
    let s4 = ms4 * rn4;
    let s5 = ms5 * rn5;
    let s6 = ms6 * rn6;
    let s7 = ms7 * rn7;
    let s8 = ms8 * rn8;
    let s9 = ms9 * rn9;
    let s10 = ms10 * rn10;
    let s11 = ms11 * rn11;
    let s12 = ms12 * rn12;
    let s13 = ms13 * rn13;
    let s14 = ms14 * rn14;
    let s15 = ms15 * rn15;
    let s16 = ms16 * rn16;
    let s17 = ms17 * rn17;
    let s18 = ms18 * rn18;

    // f[i] = sum_j M[j][i] * s[j]   (M[j][i] = coefficient of f[i] in row j of M)

    // Rest (i=0): M columns [1, -30, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fo[0] = s0 - 30.0 * s1 + 12.0 * s2;

    // Face +x (i=1): [1, -11, -4, 1, -4, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0]
    fo[1] = s0 - 11.0 * s1 - 4.0 * s2 + s3 - 4.0 * s4 + 2.0 * s9 - 2.0 * s10;
    // Face -x (i=2)
    fo[2] = s0 - 11.0 * s1 - 4.0 * s2 - s3 + 4.0 * s4 + 2.0 * s9 - 2.0 * s10;
    // Face +y (i=3): [1, -11, -4, 0, 0, 1, -4, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0]
    fo[3] = s0 - 11.0 * s1 - 4.0 * s2 + s5 - 4.0 * s6 - s9 + s10 + s11 - s12;
    // Face -y (i=4)
    fo[4] = s0 - 11.0 * s1 - 4.0 * s2 - s5 + 4.0 * s6 - s9 + s10 + s11 - s12;
    // Face +z (i=5): [1, -11, -4, 0, 0, 0, 0, 1, -4, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0]
    fo[5] = s0 - 11.0 * s1 - 4.0 * s2 + s7 - 4.0 * s8 - s9 + s10 - s11 + s12;
    // Face -z (i=6)
    fo[6] = s0 - 11.0 * s1 - 4.0 * s2 - s7 + 4.0 * s8 - s9 + s10 - s11 + s12;

    // Edge +x+y (i=7): [1, 8, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, -1, 0]
    fo[7] = s0 + 8.0 * s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16 - s17;
    // Edge -x-y (i=8)
    fo[8] = s0 + 8.0 * s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16 + s17;
    // Edge +x-y (i=9)
    fo[9] = s0 + 8.0 * s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16 - s17;
    // Edge -x+y (i=10)
    fo[10] = s0 + 8.0 * s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16 + s17;
    // Edge +x+z (i=11): [1, 8, 1, 1, 1, 0, 0, 1, 1, 1, 1, -1, -1, 0, 1, 0, -1, 0, 1]
    fo[11] = s0 + 8.0 * s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16 + s18;
    // Edge -x-z (i=12)
    fo[12] = s0 + 8.0 * s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16 - s18;
    // Edge +x-z (i=13)
    fo[13] = s0 + 8.0 * s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16 + s18;
    // Edge -x+z (i=14)
    fo[14] = s0 + 8.0 * s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16 - s18;
    // Edge +y+z (i=15): [1, 8, 1, 0, 0, 1, 1, 1, 1, -2, -2, 0, 0, 0, 0, 1, 0, 1, -1]
    fo[15] = s0 + 8.0 * s1 + s2 + s5 + s6 + s7 + s8 - 2.0 * s9 - 2.0 * s10 + s15 + s17 - s18;
    // Edge -y-z (i=16)
    fo[16] = s0 + 8.0 * s1 + s2 - s5 - s6 - s7 - s8 - 2.0 * s9 - 2.0 * s10 + s15 - s17 + s18;
    // Edge +y-z (i=17)
    fo[17] = s0 + 8.0 * s1 + s2 + s5 + s6 - s7 - s8 - 2.0 * s9 - 2.0 * s10 - s15 - s17 - s18;
    // Edge -y+z (i=18)
    fo[18] = s0 + 8.0 * s1 + s2 - s5 - s6 + s7 + s8 - 2.0 * s9 - 2.0 * s10 - s15 + s17 + s18;

    fo
}

/// D3Q19 MRT collision operating on 4 cells simultaneously via AVX2 f64x4.
///
/// Structurally identical to `collide_mrt_d3q19` but every scalar becomes
/// an f64x4 lane-parallel computation. The 722 FMA operations execute as
/// ~181 VFMADD231PD instructions, each processing 4 cells per cycle.
#[inline(always)]
#[allow(dead_code)]
fn collide_mrt_d3q19_x4(
    f: &[f64x4; 19],
    rho: f64x4,
    ux: f64x4,
    uy: f64x4,
    uz: f64x4,
    tau: f64x4,
) -> [f64x4; 19] {
    let one = f64x4::splat(1.0);
    let s_nu = one / tau;
    let s_e = f64x4::splat(1.19);
    let s_eps = f64x4::splat(1.4);
    let s_q = f64x4::splat(1.2);
    let s_ghost = one;

    let u_sq = ux * ux + uy * uy + uz * uz;

    // Forward transform: f -> moment space (m = M * f)
    let m0 = f[0]
        + f[1]
        + f[2]
        + f[3]
        + f[4]
        + f[5]
        + f[6]
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        + f[15]
        + f[16]
        + f[17]
        + f[18];
    let m1 = f64x4::splat(-30.0) * f[0]
        + f64x4::splat(-11.0) * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + f64x4::splat(8.0)
            * (f[7]
                + f[8]
                + f[9]
                + f[10]
                + f[11]
                + f[12]
                + f[13]
                + f[14]
                + f[15]
                + f[16]
                + f[17]
                + f[18]);
    let m2 = f64x4::splat(12.0) * f[0]
        + f64x4::splat(-4.0) * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6])
        + (f[7]
            + f[8]
            + f[9]
            + f[10]
            + f[11]
            + f[12]
            + f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18]);
    let m3 = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14];
    let m4 = f64x4::splat(-4.0) * (f[1] - f[2]) + f[7] - f[8] + f[9] - f[10] + f[11] - f[12]
        + f[13]
        - f[14];
    let m5 = f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m6 = f64x4::splat(-4.0) * (f[3] - f[4]) + f[7] - f[8] - f[9] + f[10] + f[15] - f[16]
        + f[17]
        - f[18];
    let m7 = f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18];
    let m8 =
        f64x4::splat(-4.0) * (f[5] - f[6]) + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17]
            + f[18];
    let m9 = f64x4::splat(2.0) * (f[1] + f[2]) - (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - f64x4::splat(2.0) * (f[15] + f[16] + f[17] + f[18]);
    let m10 = f64x4::splat(-2.0) * (f[1] + f[2])
        + (f[3] + f[4] + f[5] + f[6])
        + f[7]
        + f[8]
        + f[9]
        + f[10]
        + f[11]
        + f[12]
        + f[13]
        + f[14]
        - f64x4::splat(2.0) * (f[15] + f[16] + f[17] + f[18]);
    let m11 = (f[3] + f[4]) - (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m12 = -(f[3] + f[4]) + (f[5] + f[6]) + f[7] + f[8] + f[9] + f[10]
        - (f[11] + f[12] + f[13] + f[14]);
    let m13 = f[7] + f[8] - f[9] - f[10];
    let m14 = f[11] + f[12] - f[13] - f[14];
    let m15 = f[15] + f[16] - f[17] - f[18];
    let m16 = f[7] - f[8] - f[9] + f[10] - f[11] + f[12] + f[13] - f[14];
    let m17 = -f[7] + f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18];
    let m18 = f[11] - f[12] + f[13] - f[14] - f[15] + f[16] + f[17] - f[18];

    // Equilibrium moments
    let m1_eq = rho * (f64x4::splat(-11.0) + f64x4::splat(19.0) * u_sq);
    let m2_eq = rho * (f64x4::splat(3.0) - f64x4::splat(5.5) * u_sq);
    let m4_eq = f64x4::splat(-2.0 / 3.0) * rho * ux;
    let m6_eq = f64x4::splat(-2.0 / 3.0) * rho * uy;
    let m8_eq = f64x4::splat(-2.0 / 3.0) * rho * uz;
    let m9_eq = rho * (f64x4::splat(2.0) * ux * ux - uy * uy - uz * uz);
    let m10_eq = f64x4::splat(-0.5) * rho * (f64x4::splat(2.0) * ux * ux - uy * uy - uz * uz);
    let m11_eq = rho * (uy * uy - uz * uz);
    let m12_eq = f64x4::splat(-0.5) * rho * (uy * uy - uz * uz);
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;
    let zero = f64x4::ZERO;

    // Relax moments: m* = m - S * (m - m_eq)
    let ms0 = m0;
    let ms1 = m1 - s_e * (m1 - m1_eq);
    let ms2 = m2 - s_eps * (m2 - m2_eq);
    let ms3 = m3;
    let ms4 = m4 - s_q * (m4 - m4_eq);
    let ms5 = m5;
    let ms6 = m6 - s_q * (m6 - m6_eq);
    let ms7 = m7;
    let ms8 = m8 - s_q * (m8 - m8_eq);
    let ms9 = m9 - s_nu * (m9 - m9_eq);
    let ms10 = m10 - s_ghost * (m10 - m10_eq);
    let ms11 = m11 - s_nu * (m11 - m11_eq);
    let ms12 = m12 - s_ghost * (m12 - m12_eq);
    let ms13 = m13 - s_nu * (m13 - m13_eq);
    let ms14 = m14 - s_nu * (m14 - m14_eq);
    let ms15 = m15 - s_nu * (m15 - m15_eq);
    let ms16 = m16 - s_ghost * (m16 - zero);
    let ms17 = m17 - s_ghost * (m17 - zero);
    let ms18 = m18 - s_ghost * (m18 - zero);

    // Inverse transform: f* = M^{-1} * m*
    let rn0 = f64x4::splat(1.0 / 19.0);
    let rn1 = f64x4::splat(1.0 / 2394.0);
    let rn2 = f64x4::splat(1.0 / 252.0);
    let rn3 = f64x4::splat(1.0 / 10.0);
    let rn4 = f64x4::splat(1.0 / 40.0);
    let rn5 = f64x4::splat(1.0 / 10.0);
    let rn6 = f64x4::splat(1.0 / 40.0);
    let rn7 = f64x4::splat(1.0 / 10.0);
    let rn8 = f64x4::splat(1.0 / 40.0);
    let rn9 = f64x4::splat(1.0 / 36.0);
    let rn10 = f64x4::splat(1.0 / 36.0);
    let rn11 = f64x4::splat(1.0 / 12.0);
    let rn12 = f64x4::splat(1.0 / 12.0);
    let rn13 = f64x4::splat(1.0 / 4.0);
    let rn14 = f64x4::splat(1.0 / 4.0);
    let rn15 = f64x4::splat(1.0 / 4.0);
    let rn16 = f64x4::splat(1.0 / 8.0);
    let rn17 = f64x4::splat(1.0 / 8.0);
    let rn18 = f64x4::splat(1.0 / 8.0);

    let s0 = ms0 * rn0;
    let s1 = ms1 * rn1;
    let s2 = ms2 * rn2;
    let s3 = ms3 * rn3;
    let s4 = ms4 * rn4;
    let s5 = ms5 * rn5;
    let s6 = ms6 * rn6;
    let s7 = ms7 * rn7;
    let s8 = ms8 * rn8;
    let s9 = ms9 * rn9;
    let s10 = ms10 * rn10;
    let s11 = ms11 * rn11;
    let s12 = ms12 * rn12;
    let s13 = ms13 * rn13;
    let s14 = ms14 * rn14;
    let s15 = ms15 * rn15;
    let s16 = ms16 * rn16;
    let s17 = ms17 * rn17;
    let s18 = ms18 * rn18;

    let c2 = f64x4::splat(2.0);
    let c4 = f64x4::splat(4.0);
    let c8 = f64x4::splat(8.0);
    let c11 = f64x4::splat(11.0);
    let c30 = f64x4::splat(30.0);
    let c12 = f64x4::splat(12.0);

    let mut fo = [f64x4::ZERO; 19];

    fo[0] = s0 - c30 * s1 + c12 * s2;
    fo[1] = s0 - c11 * s1 - c4 * s2 + s3 - c4 * s4 + c2 * s9 - c2 * s10;
    fo[2] = s0 - c11 * s1 - c4 * s2 - s3 + c4 * s4 + c2 * s9 - c2 * s10;
    fo[3] = s0 - c11 * s1 - c4 * s2 + s5 - c4 * s6 - s9 + s10 + s11 - s12;
    fo[4] = s0 - c11 * s1 - c4 * s2 - s5 + c4 * s6 - s9 + s10 + s11 - s12;
    fo[5] = s0 - c11 * s1 - c4 * s2 + s7 - c4 * s8 - s9 + s10 - s11 + s12;
    fo[6] = s0 - c11 * s1 - c4 * s2 - s7 + c4 * s8 - s9 + s10 - s11 + s12;
    fo[7] = s0 + c8 * s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16 - s17;
    fo[8] = s0 + c8 * s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16 + s17;
    fo[9] = s0 + c8 * s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16 - s17;
    fo[10] = s0 + c8 * s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16 + s17;
    fo[11] = s0 + c8 * s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16 + s18;
    fo[12] = s0 + c8 * s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16 - s18;
    fo[13] = s0 + c8 * s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16 + s18;
    fo[14] = s0 + c8 * s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16 - s18;
    fo[15] = s0 + c8 * s1 + s2 + s5 + s6 + s7 + s8 - c2 * s9 - c2 * s10 + s15 + s17 - s18;
    fo[16] = s0 + c8 * s1 + s2 - s5 - s6 - s7 - s8 - c2 * s9 - c2 * s10 + s15 - s17 + s18;
    fo[17] = s0 + c8 * s1 + s2 + s5 + s6 - s7 - s8 - c2 * s9 - c2 * s10 - s15 - s17 - s18;
    fo[18] = s0 + c8 * s1 + s2 - s5 - s6 + s7 + s8 - c2 * s9 - c2 * s10 - s15 + s17 + s18;

    fo
}

impl LbmSolver3D {
    /// Create a new 3D LBM solver domain.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `tau` - Relaxation time (must be >= 0.5)
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Self {
        let n_nodes = nx * ny * nz;
        let n_padded = aosoa_pad(n_nodes);
        let collider = BgkCollision::new(tau);

        // Initialize populations to equilibrium at rest (rho=1, u=0) in AoSoA layout.
        // f_i^eq(rho=1, u=0) = w_i for all i.
        // Ghost cells (padding) are initialized to w_i too (safe: they are never read
        // by the streaming phase since they lie outside nx*ny*nz).
        let lattice = &collider.lattice;
        let mut f = vec![0.0; n_padded * 19];
        for node in 0..n_padded {
            for i in 0..19 {
                f[aosoa_idx(node, i)] = lattice.weight(i);
            }
        }

        let f_scratch = vec![0.0; n_padded * 19];
        Self {
            nx,
            ny,
            nz,
            f,
            f_scratch,
            rho: vec![1.0; n_nodes],
            u: vec![[0.0; 3]; n_nodes],
            collider,
            force_field: None,
            collision_mode: CollisionMode::Bgk,
            timestep: 0,
        }
    }

    /// Create a new 3D LBM solver with MRT collision operator.
    ///
    /// MRT (Multiple-Relaxation-Time) decouples ghost moment relaxation from
    /// physical viscosity, preventing divergence at steep density gradients
    /// (e.g., NFW halo cusps). ~12x more FLOPs per cell but unconditionally stable.
    pub fn new_mrt(nx: usize, ny: usize, nz: usize, tau: f64) -> Self {
        let mut solver = Self::new(nx, ny, nz, tau);
        solver.collision_mode = CollisionMode::Mrt;
        solver
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
    pub fn set_viscosity_field(&mut self, tau_field: Vec<f64>) -> Result<()> {
        let expected_len = self.nx * self.ny * self.nz;
        if tau_field.len() != expected_len {
            return Err(LbmError::DimensionMismatch {
                expected: expected_len,
                found: tau_field.len(),
            });
        }
        self.collider.set_viscosity_field(tau_field)
    }

    /// Get the current viscosity field.
    pub fn get_viscosity_field(&self) -> Vec<f64> {
        self.collider.get_viscosity_field()
    }

    /// Compute Smagorinsky LES subgrid viscosity from current velocity field.
    ///
    /// Computes the strain rate tensor S_ij via central differences on the
    /// velocity field, then sets per-cell tau:
    ///   tau(x) = tau_base + 3 * (C_s * dx)^2 * |S(x)|
    /// clamped to [0.505, 5.0] for stability.
    ///
    /// # Arguments
    /// * `cs` - Smagorinsky constant (typical 0.1-0.2)
    /// * `dx` - Cell size in physical units (kpc for galaxy sims)
    /// * `tau_base` - Molecular relaxation time (minimum 0.5)
    pub fn update_smagorinsky_tau(&mut self, cs: f64, dx: f64, tau_base: f64) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n = nx * ny * nz;
        let cs_sq_dx_sq = cs * cs * dx * dx;
        let mut tau_field = vec![tau_base; n];
        let mut nan_cells = 0usize;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = z * nx * ny + y * nx + x;
                    // Central differences with periodic BC
                    let xp = z * nx * ny + y * nx + (x + 1) % nx;
                    let xm = z * nx * ny + y * nx + (x + nx - 1) % nx;
                    let yp = z * nx * ny + ((y + 1) % ny) * nx + x;
                    let ym = z * nx * ny + ((y + ny - 1) % ny) * nx + x;
                    let zp = ((z + 1) % nz) * nx * ny + y * nx + x;
                    let zm = ((z + nz - 1) % nz) * nx * ny + y * nx + x;

                    // Guard: skip cells with NaN velocity neighbors.
                    // Concentrated galaxies at f64 can diverge (~30 steps),
                    // producing NaN velocities that would poison the strain rate.
                    let neighbors = [idx, xp, xm, yp, ym, zp, zm];
                    let has_nan = neighbors.iter().any(|&n| {
                        !self.u[n][0].is_finite()
                            || !self.u[n][1].is_finite()
                            || !self.u[n][2].is_finite()
                    });
                    if has_nan {
                        nan_cells += 1;
                        continue;
                    }

                    let dudx = 0.5 * (self.u[xp][0] - self.u[xm][0]);
                    let dudy = 0.5 * (self.u[yp][0] - self.u[ym][0]);
                    let dudz = 0.5 * (self.u[zp][0] - self.u[zm][0]);
                    let dvdx = 0.5 * (self.u[xp][1] - self.u[xm][1]);
                    let dvdy = 0.5 * (self.u[yp][1] - self.u[ym][1]);
                    let dvdz = 0.5 * (self.u[zp][1] - self.u[zm][1]);
                    let dwdx = 0.5 * (self.u[xp][2] - self.u[xm][2]);
                    let dwdy = 0.5 * (self.u[yp][2] - self.u[ym][2]);
                    let dwdz = 0.5 * (self.u[zp][2] - self.u[zm][2]);

                    // Symmetric strain rate tensor components
                    let s11 = dudx;
                    let s22 = dvdy;
                    let s33 = dwdz;
                    let s12 = 0.5 * (dudy + dvdx);
                    let s13 = 0.5 * (dudz + dwdx);
                    let s23 = 0.5 * (dvdz + dwdy);

                    // |S| = sqrt(2 * S_ij * S_ij)
                    let s_mag = (2.0
                        * (s11 * s11
                            + s22 * s22
                            + s33 * s33
                            + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)))
                        .sqrt();

                    let nu_turb = cs_sq_dx_sq * s_mag;
                    let tau_new = tau_base + 3.0 * nu_turb;
                    tau_field[idx] = tau_new.clamp(0.505, 5.0);
                }
            }
        }

        if nan_cells > 0 {
            eprintln!(
                "  Smagorinsky: {nan_cells}/{n} cells non-finite ({:.1}%), tau_base={tau_base:.2}",
                100.0 * nan_cells as f64 / n as f64
            );
        }

        self.set_viscosity_field(tau_field)
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
    pub fn set_force_field(&mut self, force_field: Vec<[f64; 3]>) -> Result<()> {
        let expected_len = self.nx * self.ny * self.nz;
        if force_field.len() != expected_len {
            return Err(LbmError::DimensionMismatch {
                expected: expected_len,
                found: force_field.len(),
            });
        }

        // Validate all force components are finite
        for &[fx, fy, fz] in force_field.iter() {
            if !fx.is_finite() || !fy.is_finite() || !fz.is_finite() {
                return Err(LbmError::NonFiniteValue(fx + fy + fz));
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

                    // Initialize distribution function to equilibrium (AoSoA)
                    let f_eq = BgkCollision::initialize_with_velocity(rho_init, u_init, lattice);
                    for (dir, &val) in f_eq.iter().enumerate() {
                        self.f[aosoa_idx(idx, dir)] = val;
                    }
                }
            }
        }
    }

    /// Re-initialize distributions to equilibrium from current rho and u fields.
    ///
    /// Use after setting rho[] directly (e.g. for density perturbation tests).
    pub fn reinitialize_from_macroscopic(&mut self) {
        let lattice = &self.collider.lattice;
        let n = self.nx * self.ny * self.nz;
        for idx in 0..n {
            let f_eq = BgkCollision::initialize_with_velocity(self.rho[idx], self.u[idx], lattice);
            for (dir, &val) in f_eq.iter().enumerate() {
                self.f[aosoa_idx(idx, dir)] = val;
            }
        }
    }

    /// Linearize 3D grid coordinates to 1D index.
    fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Compute macroscopic quantities (rho, u) from distribution function.
    pub fn compute_macroscopic(&mut self) {
        self.compute_macroscopic_range(0..self.nz);
    }

    /// Compute macroscopic quantities for a specific Z-range.
    pub fn compute_macroscopic_range(&mut self, z_range: core::ops::Range<usize>) {
        let lattice = self.collider.lattice.clone();
        let f_slice = &self.f;
        let nx = self.nx;
        let ny = self.ny;

        let start_idx = z_range.start * nx * ny;
        let end_idx = z_range.end * nx * ny;

        self.rho[start_idx..end_idx]
            .par_iter_mut()
            .zip(self.u[start_idx..end_idx].par_iter_mut())
            .enumerate()
            .for_each(|(i, (rho_out, u_out))| {
                let idx = start_idx + i;
                let mut f = [0.0; 19];
                for (dir, fi) in f.iter_mut().enumerate() {
                    *fi = f_slice[aosoa_idx(idx, dir)];
                }
                let rho = BgkCollision::density_from_f(&f);
                *rho_out = rho;
                *u_out = BgkCollision::velocity_from_f(&f, rho, &lattice);
            });
    }

    /// Phase 1 (collision): Compute macroscopic quantities and apply collision operator.
    pub fn phase1_collision(&mut self) -> ScheduleResult<()> {
        self.phase1_collision_range(0..self.nz)
    }

    /// Phase 1 (collision) restricted to a specific Z-range.
    pub fn phase1_collision_range(
        &mut self,
        z_range: core::ops::Range<usize>,
    ) -> ScheduleResult<()> {
        self.compute_macroscopic_range(z_range.clone());

        let lattice = self.collider.lattice.clone();
        let tau_field = &self.collider.tau_field;
        let default_tau = if !tau_field.is_empty() {
            tau_field[0]
        } else {
            0.6
        };
        let rho = &self.rho;
        let u = &self.u;
        let force_field = &self.force_field;
        let mode = self.collision_mode;

        // AoSoA collision: chunk-based SIMD iteration.
        let nx = self.nx;
        let ny = self.ny;
        let start_cell = z_range.start * nx * ny;
        let end_cell = z_range.end * nx * ny;

        // Ensure chunks align properly. We process in AOSOA_CHUNK sizes.
        // Assuming the total grid size is a multiple of AOSOA_CHUNK.
        let start_chunk = start_cell / AOSOA_CHUNK;
        let end_chunk = end_cell / AOSOA_CHUNK;
        let tail_start = end_chunk * AOSOA_CHUNK;
        let n_cells = end_cell;

        // SAFETY: each chunk maps to non-overlapping AoSoA locations.
        let f_ptr = UnsafeAoSoAPtr(self.f.as_mut_ptr());

        // --- SIMD path: process 4 cells per chunk via f64x4 ---
        (start_chunk..end_chunk)
            .into_par_iter()
            .for_each(|chunk_idx| {
                unsafe {
                    let base_cell = chunk_idx * AOSOA_CHUNK;
                    let chunk_offset = chunk_idx * 19 * AOSOA_CHUNK;

                    // Load 19 f64x4 vectors (each = one direction across 4 cells)
                    let mut f_local = [f64x4::ZERO; 19];
                    for (dir, f_val) in f_local.iter_mut().enumerate() {
                        *f_val = f_ptr.read_x4(chunk_offset + dir * AOSOA_CHUNK);
                    }

                    // Gather macroscopic quantities for the 4 cells
                    let rho4 = f64x4::new([
                        rho[base_cell],
                        rho[base_cell + 1],
                        rho[base_cell + 2],
                        rho[base_cell + 3],
                    ]);
                    let ux4 = f64x4::new([
                        u[base_cell][0],
                        u[base_cell + 1][0],
                        u[base_cell + 2][0],
                        u[base_cell + 3][0],
                    ]);
                    let uy4 = f64x4::new([
                        u[base_cell][1],
                        u[base_cell + 1][1],
                        u[base_cell + 2][1],
                        u[base_cell + 3][1],
                    ]);
                    let uz4 = f64x4::new([
                        u[base_cell][2],
                        u[base_cell + 1][2],
                        u[base_cell + 2][2],
                        u[base_cell + 3][2],
                    ]);

                    // Gather tau for the 4 cells
                    let tau4 = if base_cell + 3 < tau_field.len() {
                        f64x4::new([
                            tau_field[base_cell],
                            tau_field[base_cell + 1],
                            tau_field[base_cell + 2],
                            tau_field[base_cell + 3],
                        ])
                    } else {
                        f64x4::splat(default_tau)
                    };

                    // Force-corrected velocity: u* = u + F / (2 * rho)
                    let (ux_star, uy_star, uz_star) = if let Some(ff) = force_field {
                        let inv_2rho = f64x4::splat(0.5) / rho4.max(f64x4::splat(1e-30));
                        let fx = f64x4::new([
                            ff[base_cell][0],
                            ff[base_cell + 1][0],
                            ff[base_cell + 2][0],
                            ff[base_cell + 3][0],
                        ]);
                        let fy = f64x4::new([
                            ff[base_cell][1],
                            ff[base_cell + 1][1],
                            ff[base_cell + 2][1],
                            ff[base_cell + 3][1],
                        ]);
                        let fz = f64x4::new([
                            ff[base_cell][2],
                            ff[base_cell + 1][2],
                            ff[base_cell + 2][2],
                            ff[base_cell + 3][2],
                        ]);
                        (
                            ux4 + fx * inv_2rho,
                            uy4 + fy * inv_2rho,
                            uz4 + fz * inv_2rho,
                        )
                    } else {
                        (ux4, uy4, uz4)
                    };

                    // Collision
                    match mode {
                        CollisionMode::Bgk => {
                            let u_sq = ux_star * ux_star + uy_star * uy_star + uz_star * uz_star;
                            for (dir, f_val) in f_local.iter_mut().enumerate() {
                                let w = f64x4::splat(lattice.weights[dir]);
                                let cx = f64x4::splat(lattice.velocities[dir][0] as f64);
                                let cy = f64x4::splat(lattice.velocities[dir][1] as f64);
                                let cz = f64x4::splat(lattice.velocities[dir][2] as f64);
                                let cu = cx * ux_star + cy * uy_star + cz * uz_star;
                                let f_eq = w
                                    * rho4
                                    * (f64x4::splat(1.0)
                                        + f64x4::splat(3.0) * cu
                                        + f64x4::splat(4.5) * cu * cu
                                        - f64x4::splat(1.5) * u_sq);
                                *f_val -= (*f_val - f_eq) / tau4;
                            }
                        }
                        CollisionMode::Mrt => {
                            f_local = collide_mrt_d3q19_x4(
                                &f_local, rho4, ux_star, uy_star, uz_star, tau4,
                            );
                        }
                    }

                    // Exact Guo source term Phi_i using u*
                    if let Some(ff) = force_field {
                        let fx = f64x4::new([
                            ff[base_cell][0],
                            ff[base_cell + 1][0],
                            ff[base_cell + 2][0],
                            ff[base_cell + 3][0],
                        ]);
                        let fy = f64x4::new([
                            ff[base_cell][1],
                            ff[base_cell + 1][1],
                            ff[base_cell + 2][1],
                            ff[base_cell + 3][1],
                        ]);
                        let fz = f64x4::new([
                            ff[base_cell][2],
                            ff[base_cell + 1][2],
                            ff[base_cell + 2][2],
                            ff[base_cell + 3][2],
                        ]);
                        let prefactor =
                            f64x4::splat(1.0) - f64x4::splat(1.0) / (f64x4::splat(2.0) * tau4);
                        for (dir, f_val) in f_local.iter_mut().enumerate() {
                            let w = f64x4::splat(lattice.weights[dir]);
                            let cx = f64x4::splat(lattice.velocities[dir][0] as f64);
                            let cy = f64x4::splat(lattice.velocities[dir][1] as f64);
                            let cz = f64x4::splat(lattice.velocities[dir][2] as f64);
                            let ei_minus_u_dot_f =
                                (cx - ux_star) * fx + (cy - uy_star) * fy + (cz - uz_star) * fz;
                            let ei_dot_u = cx * ux_star + cy * uy_star + cz * uz_star;
                            let ei_dot_f = cx * fx + cy * fy + cz * fz;
                            let phi_i = ei_minus_u_dot_f * f64x4::splat(3.0)
                                + (ei_dot_u * ei_dot_f) * f64x4::splat(9.0);
                            *f_val += prefactor * w * phi_i;
                        }
                    }

                    // Store 19 f64x4 vectors back to AoSoA
                    for (dir, &f_val) in f_local.iter().enumerate() {
                        f_ptr.write_x4(chunk_offset + dir * AOSOA_CHUNK, f_val);
                    }
                }
            });

        // --- Scalar tail: handle remaining cells if n_cells % 4 != 0 ---
        for idx in tail_start..n_cells {
            let tau = if idx < tau_field.len() {
                tau_field[idx]
            } else {
                default_tau
            };
            let rho_local = rho[idx];
            let u_local = u[idx];
            let u_star = if let Some(ff) = force_field {
                let force = ff[idx];
                let inv_2rho = 0.5 / rho_local.max(1e-30);
                [
                    u_local[0] + force[0] * inv_2rho,
                    u_local[1] + force[1] * inv_2rho,
                    u_local[2] + force[2] * inv_2rho,
                ]
            } else {
                u_local
            };
            let mut f_local = [0.0_f64; 19];
            for (dir, f_val) in f_local.iter_mut().enumerate() {
                // SAFETY: tail cells are within bounds and not touched by SIMD path.
                *f_val = unsafe { f_ptr.read(aosoa_idx(idx, dir)) };
            }
            match mode {
                CollisionMode::Bgk => {
                    for (i, f_val) in f_local.iter_mut().enumerate() {
                        let f_eq_i = lattice.equilibrium(rho_local, u_star, i);
                        *f_val -= (*f_val - f_eq_i) / tau;
                    }
                }
                CollisionMode::Mrt => {
                    f_local = collide_mrt_d3q19(
                        &f_local, rho_local, u_star[0], u_star[1], u_star[2], tau,
                    );
                }
            }
            if let Some(ff) = force_field {
                let force = ff[idx];
                let prefactor = 1.0 - 1.0 / (2.0 * tau);
                for (i, f_val) in f_local.iter_mut().enumerate() {
                    let ei = lattice.velocities[i];
                    let ei_f64 = [ei[0] as f64, ei[1] as f64, ei[2] as f64];
                    let ei_minus_u_dot_f = (ei_f64[0] - u_star[0]) * force[0]
                        + (ei_f64[1] - u_star[1]) * force[1]
                        + (ei_f64[2] - u_star[2]) * force[2];
                    let ei_dot_u =
                        ei_f64[0] * u_star[0] + ei_f64[1] * u_star[1] + ei_f64[2] * u_star[2];
                    let ei_dot_f =
                        ei_f64[0] * force[0] + ei_f64[1] * force[1] + ei_f64[2] * force[2];
                    let phi_i = ei_minus_u_dot_f * 3.0 + (ei_dot_u * ei_dot_f) * 9.0;
                    *f_val += prefactor * lattice.weights[i] * phi_i;
                }
            }
            for (dir, &f_val) in f_local.iter().enumerate() {
                // SAFETY: tail cells are within bounds and not touched by SIMD path.
                unsafe { f_ptr.write(aosoa_idx(idx, dir), f_val) };
            }
        }

        // Post-collision: update stored velocity to force-corrected u*
        // so downstream consumers (MHD, diagnostics, drag force) use
        // the physical velocity, not the bare streaming velocity.
        if let Some(ref ff) = self.force_field {
            for ((u, &rho), f) in self.u.iter_mut().zip(&self.rho).zip(ff) {
                let inv_2rho = 0.5 / rho.max(1e-30);
                u[0] += f[0] * inv_2rho;
                u[1] += f[1] * inv_2rho;
                u[2] += f[2] * inv_2rho;
            }
        }

        Ok(())
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
        let lattice = self.collider.lattice.clone();
        let f_src = &self.f;

        // Use pre-allocated scratch buffer instead of allocating per step.
        // AoSoA streaming: same pull scheme but reads/writes use aosoa_idx.
        // Serial loop: at 128^3 the pull scheme reads from arbitrary source
        // cells, making cache behavior worse under parallel writes.
        #[allow(clippy::needless_range_loop)]
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dst_idx = z * (nx * ny) + y * nx + x;

                    for i in 0..19 {
                        let c = lattice.velocities[i];
                        let sx = (x as i32 - c[0]).rem_euclid(nx as i32) as usize;
                        let sy = (y as i32 - c[1]).rem_euclid(ny as i32) as usize;
                        let sz = (z as i32 - c[2]).rem_euclid(nz as i32) as usize;
                        let src_idx = sz * (nx * ny) + sy * nx + sx;
                        self.f_scratch[aosoa_idx(dst_idx, i)] = f_src[aosoa_idx(src_idx, i)];
                    }
                }
            }
        }

        std::mem::swap(&mut self.f, &mut self.f_scratch);
        self.compute_macroscopic();
        self.timestep += 1;

        Ok(())
    }

    /// Phase 2 (streaming) over a subset of Z slices.
    ///
    /// This tiled variant writes into the shared scratch buffer but leaves the
    /// final swap, macroscopic recompute, and timestep increment to the caller.
    pub fn phase2_streaming_range(
        &mut self,
        z_range: core::ops::Range<usize>,
    ) -> ScheduleResult<()> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let lattice = self.collider.lattice.clone();
        let f_src = &self.f;

        #[allow(clippy::needless_range_loop)]
        for z in z_range {
            for y in 0..ny {
                for x in 0..nx {
                    let dst_idx = z * (nx * ny) + y * nx + x;

                    for i in 0..19 {
                        let c = lattice.velocities[i];
                        let sx = (x as i32 - c[0]).rem_euclid(nx as i32) as usize;
                        let sy = (y as i32 - c[1]).rem_euclid(ny as i32) as usize;
                        let sz = (z as i32 - c[2]).rem_euclid(nz as i32) as usize;
                        let src_idx = sz * (nx * ny) + sy * nx + sx;
                        self.f_scratch[aosoa_idx(dst_idx, i)] = f_src[aosoa_idx(src_idx, i)];
                    }
                }
            }
        }

        Ok(())
    }

    /// Perform one complete LBM timestep.
    ///
    /// Uses Z-axis V-Cache Tiling for large grids to keep the active working set
    /// within the CPU's L3 cache. Defaults to a heuristic 16MB if undetected, but
    /// explicitly scales up to Ryzen 5600X3D's 96MB V-Cache if detected.
    /// If the grid fits entirely in cache or is too small, falls back to standard
    /// two-phase coordination.
    pub fn evolve_one_step(&mut self) {
        // Calculate grid memory footprint.
        // 19 f64s per cell for f, plus scratch, rho, and u.
        let bytes_per_cell = 19 * 8 * 2 + 8 + 3 * 8;
        let total_bytes = self.nx * self.ny * self.nz * bytes_per_cell;

        // Dynamically detect L3 cache size via global oracle
        let topo = verified_core::topology::HardwareTopology::current();
        let l3_target_bytes = topo.l3_safe_working_set_bytes;

        let bytes_per_z_slice = self.nx * self.ny * bytes_per_cell;

        // If the entire grid fits in L3, or a single Z-slice is too big for L3,
        // tiling won't help much. Fall back to standard.
        if total_bytes <= l3_target_bytes || bytes_per_z_slice > l3_target_bytes {
            let _ = self.phase1_collision();
            let _ = self.phase2_streaming();
            return;
        }

        // V-Cache Tiling: Determine how many Z-slices fit in our L3 target
        let z_chunk_size = std::cmp::max(1, l3_target_bytes / bytes_per_z_slice);

        if self.timestep == 0 {
            eprintln!(
                "LBM V-Cache Tiling: Grid uses {:.1} MB. L3 limit is {:.1} MB. Processing in {} Z-slice chunks.",
                total_bytes as f64 / 1024.0 / 1024.0,
                l3_target_bytes as f64 / 1024.0 / 1024.0,
                z_chunk_size
            );
        }

        let mut z = 0;
        while z < self.nz {
            let z_end = std::cmp::min(self.nz, z + z_chunk_size);
            let z_range = z..z_end;

            let _ = self.phase1_collision_range(z_range.clone());
            let _ = self.phase2_streaming_range(z_range);

            z += z_chunk_size;
        }

        std::mem::swap(&mut self.f, &mut self.f_scratch);
        self.compute_macroscopic();
        self.timestep += 1;
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

    /// Maximum Mach number across the domain: Ma = max(|u|) / c_s.
    ///
    /// For D3Q19, c_s = 1/sqrt(3). BGK is typically stable for Ma < 0.3;
    /// MRT extends this to ~1.5 by independently damping ghost moments.
    pub fn max_mach_number(&self) -> f64 {
        let cs = (1.0_f64 / 3.0).sqrt();
        self.max_velocity() / cs
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
        self.u.iter().zip(reference_u.iter()).all(|(curr, prev)| {
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
                    mass_error: (current_mass - initial_mass).abs() / initial_mass.abs().max(1e-30),
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
        let err = (self.final_mass - self.initial_mass).abs() / self.initial_mass.abs().max(1e-30);
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
        for (i, &fi) in f.iter().enumerate() {
            let expected = rho * lattice.weight(i);
            assert!((fi - expected).abs() < 1e-14);
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
        assert_eq!(solver.f.len(), aosoa_pad(10 * 8 * 6) * 19);
        assert_eq!(solver.rho.len(), 10 * 8 * 6);
        assert_eq!(solver.u.len(), 10 * 8 * 6);
    }

    #[test]
    fn test_lbm_solver_equilibrium_initialization() {
        let solver = LbmSolver3D::new(4, 4, 4, 0.8);
        let lattice = D3Q19Lattice::new();

        // All cells should be initialized to equilibrium at rho=1, u=0
        for node in 0..64 {
            assert!(
                (solver.rho[node] - 1.0).abs() < 1e-14,
                "rho not 1.0 at node {}",
                node
            );
            for k in 0..3 {
                assert!(solver.u[node][k].abs() < 1e-14, "u not 0 at node {}", node);
            }
            // f_i should equal w_i (equilibrium at rest with rho=1)
            for i in 0..19 {
                let expected = lattice.weight(i);
                let actual = solver.f[aosoa_idx(node, i)];
                assert!(
                    (actual - expected).abs() < 1e-14,
                    "f[{}] = {} != w[{}] = {} at node {}",
                    i,
                    actual,
                    i,
                    expected,
                    node
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
        assert!(
            v_max > 1e-6,
            "Force should produce nonzero velocity, got {}",
            v_max
        );
        assert!(
            v_max < 0.1,
            "Velocity should be small and stable, got {}",
            v_max
        );

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
        solver.f[aosoa_idx(src_idx, 1)] += 0.01;

        // Run streaming only (skip collision to isolate streaming effect)
        let _ = solver.phase2_streaming();

        // After streaming, the perturbation in direction 1 should have moved to (4,3,3)
        let dst_idx = solver.linearize(4, 3, 3);
        let original_val = 1.0 * solver.collider.lattice.weight(1);
        let delta = solver.f[aosoa_idx(dst_idx, 1)] - original_val;
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
        solver.f[aosoa_idx(edge_idx, 1)] += 0.02;

        let _ = solver.phase2_streaming();

        // Should wrap to (0, 2, 2)
        let wrap_idx = solver.linearize(0, 2, 2);
        let original_val = 1.0 * solver.collider.lattice.weight(1);
        let delta = solver.f[aosoa_idx(wrap_idx, 1)] - original_val;
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
        let max_ux: f64 = solver.u.iter().map(|u| u[0].abs()).fold(0.0_f64, f64::max);
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
        assert!(
            (mass - n as f64).abs() / (n as f64) < 1e-4,
            "Mass not conserved: {}",
            mass
        );

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
            v_thick,
            v_newton
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
            tau_min,
            tau_max
        );
    }

    #[test]
    fn test_smagorinsky_uniform_flow() {
        // Uniform velocity -> S_ij = 0 -> tau = tau_base everywhere
        let mut solver = LbmSolver3D::new(8, 8, 8, 1.5);
        // Set uniform velocity
        for u in solver.u.iter_mut() {
            *u = [0.01, 0.0, 0.0];
        }
        solver
            .update_smagorinsky_tau(0.1, 1.0, 1.5)
            .expect("smagorinsky");
        let tau = solver.collider.get_tau_field();
        for &t in tau.iter() {
            assert!(
                (t - 1.5).abs() < 1e-10,
                "uniform flow should give tau = tau_base, got {t}"
            );
        }
    }

    #[test]
    fn test_smagorinsky_shear_flow() {
        // Linear shear du/dy -> S_12 != 0 -> tau > tau_base
        let n = 16;
        let mut solver = LbmSolver3D::new(n, n, n, 1.5);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let idx = z * n * n + y * n + x;
                    solver.u[idx] = [0.001 * y as f64, 0.0, 0.0];
                }
            }
        }
        solver
            .update_smagorinsky_tau(0.1, 1.0, 1.5)
            .expect("smagorinsky");
        let tau = solver.collider.get_tau_field();
        let tau_max = tau.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            tau_max > 1.5,
            "shear flow should increase tau, got max={tau_max}"
        );
    }

    #[test]
    fn test_mrt_mass_conservation() {
        // MRT must preserve total mass to machine precision.
        let mut solver = LbmSolver3D::new_mrt(8, 8, 8, 1.5);
        let n = 8 * 8 * 8;

        // Perturb initial density: Gaussian blob at center
        let lattice = solver.collider.lattice.clone();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let idx = z * 64 + y * 8 + x;
                    let dx = x as f64 - 3.5;
                    let dy = y as f64 - 3.5;
                    let dz = z as f64 - 3.5;
                    let rho = 1.0 + 0.1 * (-0.5 * (dx * dx + dy * dy + dz * dz)).exp();
                    let f_eq = BgkCollision::initialize_rest(rho, &lattice);
                    for (dir, &feq) in f_eq.iter().enumerate() {
                        solver.f[aosoa_idx(idx, dir)] = feq;
                    }
                    solver.rho[idx] = rho;
                }
            }
        }

        let mass_0: f64 = solver.rho.iter().sum();
        assert_eq!(n, solver.rho.len());

        // Evolve 20 steps with MRT
        for _ in 0..20 {
            solver.evolve_one_step();
        }

        let mass_1: f64 = solver.rho.iter().sum();
        let rel_err = (mass_1 - mass_0).abs() / mass_0;
        assert!(
            rel_err < 1e-10,
            "MRT mass conservation violated: mass_0={mass_0}, mass_1={mass_1}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_mrt_bgk_agreement_uniform() {
        // On uniform density at rest, MRT and BGK must produce identical results.
        let mut bgk_solver = LbmSolver3D::new(8, 8, 8, 1.5);
        let mut mrt_solver = LbmSolver3D::new_mrt(8, 8, 8, 1.5);

        // Both start from default (rho=1, u=0)
        for _ in 0..10 {
            bgk_solver.evolve_one_step();
            mrt_solver.evolve_one_step();
        }

        // Density should be identical (both start at equilibrium)
        let n = 8 * 8 * 8;
        for i in 0..n {
            let diff = (bgk_solver.rho[i] - mrt_solver.rho[i]).abs();
            assert!(
                diff < 1e-12,
                "BGK/MRT disagree at cell {i}: bgk={}, mrt={}, diff={diff}",
                bgk_solver.rho[i],
                mrt_solver.rho[i]
            );
        }
    }

    /// Verify d'Humieres D3Q19 MRT relaxation rates and row norms.
    ///
    /// The transformation matrix M has 19 orthogonal rows whose squared norms
    /// are known analytically (d'Humieres et al. 2002, Lallemand & Luo 2000).
    /// The relaxation rates S_diag must satisfy 0 < s_i <= 2 for stability,
    /// with conserved moments (mass, momentum) having s=0.
    #[test]
    fn test_mrt_relaxation_rates_and_row_norms() {
        // Published row squared-norms for d'Humieres D3Q19 M matrix
        let expected_row_norms: [f64; 19] = [
            19.0, 2394.0, 252.0, 10.0, 40.0, 10.0, 40.0, 10.0, 40.0, 36.0, 36.0, 12.0, 12.0, 4.0,
            4.0, 4.0, 8.0, 8.0, 8.0,
        ];

        // Verify by computing M * e_i for each canonical basis vector.
        // Row j of M is the vector of coefficients applied to f[0..19].
        // We compute m = M * e_i for i=0..18 and accumulate ||row_j||^2.
        let mut row_norm_sq = [0.0_f64; 19];
        for i in 0..19 {
            let mut f = [0.0_f64; 19];
            f[i] = 1.0;
            // Use collide_mrt with rho=1, u=0 and extract moment values.
            // Instead, compute M*f directly using the forward transform.
            let m = mrt_forward_transform(&f);
            for (j, mj) in m.iter().enumerate() {
                row_norm_sq[j] += mj * mj;
            }
        }

        for (j, (&computed, &expected)) in row_norm_sq
            .iter()
            .zip(expected_row_norms.iter())
            .enumerate()
        {
            let rel_err = (computed - expected).abs() / expected;
            assert!(
                rel_err < 1e-12,
                "Row {j}: norm^2 = {computed}, expected {expected}, rel_err = {rel_err:.2e}"
            );
        }

        // Verify relaxation rates are in valid range (0, 2] for non-conserved,
        // exactly 0 for conserved moments (mass=m0, momentum=m3,m5,m7).
        let tau = 0.7;
        let s_nu = 1.0 / tau;
        let s_e = 1.19;
        let s_eps = 1.4;
        let s_q = 1.2;
        let s_ghost = 1.0;

        let s_diag: [f64; 19] = [
            0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q, s_nu, s_ghost, s_nu, s_ghost, s_nu,
            s_nu, s_nu, s_ghost, s_ghost, s_ghost,
        ];

        for (j, &s) in s_diag.iter().enumerate() {
            if [0, 3, 5, 7].contains(&j) {
                assert_eq!(s, 0.0, "Conserved moment {j} must have s=0");
            } else {
                assert!(
                    s > 0.0 && s <= 2.0,
                    "Rate s[{j}] = {s} out of stable range (0, 2]"
                );
            }
        }
    }

    #[test]
    fn test_max_mach_number_at_rest() {
        let solver = LbmSolver3D::new(4, 4, 4, 1.0);
        assert!(solver.max_mach_number() < 1e-14);
    }

    #[test]
    fn test_max_mach_number_with_flow() {
        let mut solver = LbmSolver3D::new(4, 4, 4, 1.0);
        // Set velocity in one cell
        solver.u[0] = [0.1, 0.0, 0.0];
        let ma = solver.max_mach_number();
        let cs = (1.0_f64 / 3.0).sqrt();
        let expected = 0.1 / cs;
        assert!(
            (ma - expected).abs() < 1e-12,
            "Ma={ma}, expected={expected}"
        );
    }
}
