//! Sedenion foliation: time-foliated 4D sedenion field on ADM hypersurfaces.
//!
//! Extends the existing `SedenionField` (3D spatial grid) to a sequence of
//! time slices connected via the ADM 3+1 decomposition. Each slice is a
//! `SedenionField` representing the sedenion algebra state on a t=const
//! hypersurface.
//!
//! The ADM lapse and shift propagate frustration between slices:
//! - Shift vector advects frustration: F(t+dt, x) ~ F(t, x - beta*dt/alpha)
//! - Extrinsic curvature sources frustration perturbations
//! - The vacuum attractor F = 3/8 is a stable fixed point
//!
//! # Key hypothesis
//!
//! York time (expansion scalar) and frustration density are anti-correlated:
//! expanding regions (theta > 0) have lower frustration, contracting regions
//! (theta < 0) have higher frustration. This is the viscous vacuum thesis
//! applied to the 3+1 foliation.
//!
//! # References
//! - This module (open_gororoba non-canonical exploration)
//! - Arnowitt, Deser, Misner (1962): ADM formalism
//! - de Marrais (2000): Sedenion zero-divisor structure

use crate::bridge::SedenionField;
use gr_core::adm::{AdmDecomposition, ExtrinsicCurvatureData};
use gr_core::spatial::trace_symmetric;

/// Vacuum frustration attractor value (3/8).
///
/// Re-exported from bridge for convenience. Frustration densities converge
/// to this value in the absence of external sources.
pub const VACUUM_ATTRACTOR: f64 = 0.375;

/// Default Cayley-Dickson dimension for sedenion fields.
const SEDENION_DIM: usize = 16;

/// Time-foliated sedenion field on ADM hypersurfaces.
///
/// A stack of `SedenionField` instances, one per t=const hypersurface.
/// The slices share spatial grid dimensions (nx, ny, nz) and are connected
/// by the ADM evolution equations.
#[derive(Clone, Debug)]
pub struct SedenionFoliation {
    /// Individual 3D sedenion fields, one per time slice.
    pub slices: Vec<SedenionField>,
    /// Number of time slices.
    pub n_t: usize,
    /// Time step between slices.
    pub dt: f64,
    /// Spatial grid spacing (assumed uniform and isotropic).
    pub dx: f64,
}

impl SedenionFoliation {
    /// Create a uniform foliation with n_t identical slices.
    ///
    /// Each slice is initialized to the uniform sedenion field (e_0 = 1).
    pub fn uniform(nx: usize, ny: usize, nz: usize, n_t: usize, dt: f64, dx: f64) -> Self {
        let slices: Vec<SedenionField> = (0..n_t)
            .map(|_| SedenionField::uniform(nx, ny, nz))
            .collect();
        Self {
            slices,
            n_t,
            dt,
            dx,
        }
    }

    /// Create a foliation from pre-built slices.
    ///
    /// All slices must share the same spatial dimensions.
    pub fn from_slices(slices: Vec<SedenionField>, dt: f64, dx: f64) -> Self {
        let n_t = slices.len();
        assert!(n_t > 0, "foliation must have at least one slice");
        let (nx, ny, nz) = (slices[0].nx, slices[0].ny, slices[0].nz);
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(
                (s.nx, s.ny, s.nz),
                (nx, ny, nz),
                "slice {i} dimensions ({},{},{}) != expected ({nx},{ny},{nz})",
                s.nx,
                s.ny,
                s.nz,
            );
        }
        Self {
            slices,
            n_t,
            dt,
            dx,
        }
    }

    /// Get the spatial grid dimensions from the first slice.
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.slices[0].nx, self.slices[0].ny, self.slices[0].nz)
    }

    /// Compute frustration density on a given time slice.
    ///
    /// Delegates to `SedenionField::local_frustration_density` at dim=16.
    pub fn frustration_at_slice(&self, slice_idx: usize) -> Vec<f64> {
        assert!(slice_idx < self.n_t, "slice index out of bounds");
        self.slices[slice_idx].local_frustration_density(SEDENION_DIM)
    }
}

// ============================================================================
// ADM-coupled evolution
// ============================================================================

/// Evolve frustration from one time slice to the next using ADM data.
///
/// The evolution equation is a first-order advection with source:
///
///   F(t+dt, x) = F(t, x_src) + dt * source_term
///
/// where:
///   x_src = x - beta * dt / alpha    (shift advection)
///   source_term = -kappa * (F - 3/8) * K    (curvature sourcing)
///
/// The source term drives frustration toward the vacuum attractor (3/8)
/// when the trace K is nonzero. kappa is a dimensionless coupling constant.
///
/// Uses trilinear interpolation for the advected lookup at x_src.
pub fn evolve_frustration_adm(
    foliation: &mut SedenionFoliation,
    slice_idx: usize,
    adm: &AdmDecomposition,
    extrinsic: &ExtrinsicCurvatureData,
    kappa: f64,
) {
    assert!(
        slice_idx + 1 < foliation.n_t,
        "cannot evolve past the last slice"
    );

    let (nx, ny, nz) = foliation.grid_dims();
    let dt = foliation.dt;
    let dx = foliation.dx;

    // Compute frustration on the source slice
    let frustration = foliation.frustration_at_slice(slice_idx);

    // Trace of extrinsic curvature: K = gamma^{ij} K_{ij}
    let k_trace = trace_symmetric(&adm.spatial_metric_inv, &extrinsic.k_ij);

    // Evolve each spatial point
    let mut new_data = foliation.slices[slice_idx].data.clone();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = z * (nx * ny) + y * nx + x;

                // Source coordinates after shift advection:
                // x_src^i = x^i - beta^i * dt / alpha
                let advection_factor = dt / (adm.lapse * dx);
                let x_src = x as f64 - adm.shift[0] * advection_factor;
                let y_src = y as f64 - adm.shift[1] * advection_factor;
                let z_src = z as f64 - adm.shift[2] * advection_factor;

                // Trilinear interpolation of frustration at source point
                let f_src = trilinear_interp_periodic(
                    &frustration, nx, ny, nz, x_src, y_src, z_src,
                );

                // Source term: drives toward vacuum attractor
                let source = -kappa * (f_src - VACUUM_ATTRACTOR) * k_trace;

                // Update frustration at the target point
                let f_new = f_src + dt * source;

                // Clamp to physical range [0, 0.5]
                let f_clamped = f_new.clamp(0.0, 0.5);

                // Modulate the sedenion field to produce the target frustration.
                // Scale components to shift frustration toward f_clamped.
                // The simplest approach: scale the scalar (e_0) component relative
                // to others to shift the triangle-weight balance.
                let f_current = frustration[idx];
                if (f_current - VACUUM_ATTRACTOR).abs() > 1e-15 {
                    let ratio = (f_clamped - VACUUM_ATTRACTOR) / (f_current - VACUUM_ATTRACTOR);
                    // Scale non-scalar components by ratio to modulate frustration
                    let scale = 1.0 + 0.1 * (ratio - 1.0);
                    for val in &mut new_data[idx][1..16] {
                        *val *= scale;
                    }
                }
            }
        }
    }

    // Write the evolved data to the next slice
    foliation.slices[slice_idx + 1].data = new_data;
}

/// Compute the algebraic contribution to the Hamiltonian constraint.
///
/// At each spatial point, the frustration deviation from the vacuum attractor
/// sources an effective energy density:
///
///   rho_eff(x) = (1/2) * nu(F) * K^2
///
/// where nu(F) = nu_0 * exp(beta * (F - 3/8)) is the exponential viscosity
/// coupling and K = gamma^{ij} K_{ij} is the trace of extrinsic curvature.
///
/// Returns a Vec of effective energy densities, one per spatial grid point.
pub fn frustration_hamiltonian_source(
    foliation: &SedenionFoliation,
    slice_idx: usize,
    adm: &AdmDecomposition,
    extrinsic: &ExtrinsicCurvatureData,
    nu_0: f64,
    beta_coupling: f64,
) -> Vec<f64> {
    let frustration = foliation.frustration_at_slice(slice_idx);
    let k_trace = trace_symmetric(&adm.spatial_metric_inv, &extrinsic.k_ij);
    let k_sq = k_trace * k_trace;

    frustration
        .iter()
        .map(|&f| {
            let nu = nu_0 * (beta_coupling * (f - VACUUM_ATTRACTOR)).exp();
            0.5 * nu * k_sq
        })
        .collect()
}

/// Compute (York time, frustration) pairs for correlation analysis.
///
/// Returns a Vec of (theta, F) pairs at each spatial point on the given slice.
/// The hypothesis: theta and F are anti-correlated. Expanding regions
/// (theta > 0) should have lower frustration, contracting regions (theta < 0)
/// should have higher frustration.
///
/// If a spatially-varying ADM decomposition is available (one per grid point),
/// pass the per-point York times. Otherwise, for a uniform background, the
/// York time from the extrinsic curvature data is used at all points.
pub fn york_time_frustration_map(
    foliation: &SedenionFoliation,
    slice_idx: usize,
    york_time: f64,
) -> Vec<(f64, f64)> {
    let frustration = foliation.frustration_at_slice(slice_idx);
    frustration.iter().map(|&f| (york_time, f)).collect()
}

/// Compute spatially-varying York time vs frustration pairs.
///
/// Takes a Vec of per-point York times (one per spatial grid point)
/// and pairs them with the per-point frustration on the given slice.
pub fn york_time_frustration_map_spatial(
    foliation: &SedenionFoliation,
    slice_idx: usize,
    york_times: &[f64],
) -> Vec<(f64, f64)> {
    let frustration = foliation.frustration_at_slice(slice_idx);
    assert_eq!(
        york_times.len(),
        frustration.len(),
        "york_times length {} != frustration length {}",
        york_times.len(),
        frustration.len(),
    );
    york_times
        .iter()
        .zip(frustration.iter())
        .map(|(&theta, &f)| (theta, f))
        .collect()
}

// ============================================================================
// Interpolation helper
// ============================================================================

/// Trilinear interpolation on a periodic 3D grid.
///
/// Given fractional coordinates (fx, fy, fz), interpolate the scalar field
/// using the 8 surrounding grid points with periodic boundary conditions.
fn trilinear_interp_periodic(
    data: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    fx: f64,
    fy: f64,
    fz: f64,
) -> f64 {
    // Wrap to [0, N) with periodic boundaries
    let wrap = |v: f64, n: usize| -> (usize, usize, f64) {
        let n_f = n as f64;
        let v_wrapped = ((v % n_f) + n_f) % n_f;
        let i0 = v_wrapped.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let frac = v_wrapped - v_wrapped.floor();
        (i0, i1, frac)
    };

    let (x0, x1, xf) = wrap(fx, nx);
    let (y0, y1, yf) = wrap(fy, ny);
    let (z0, z1, zf) = wrap(fz, nz);

    let idx = |x: usize, y: usize, z: usize| -> usize { z * (nx * ny) + y * nx + x };

    // 8-point trilinear interpolation
    let c000 = data[idx(x0, y0, z0)];
    let c100 = data[idx(x1, y0, z0)];
    let c010 = data[idx(x0, y1, z0)];
    let c110 = data[idx(x1, y1, z0)];
    let c001 = data[idx(x0, y0, z1)];
    let c101 = data[idx(x1, y0, z1)];
    let c011 = data[idx(x0, y1, z1)];
    let c111 = data[idx(x1, y1, z1)];

    let c00 = c000 * (1.0 - xf) + c100 * xf;
    let c01 = c001 * (1.0 - xf) + c101 * xf;
    let c10 = c010 * (1.0 - xf) + c110 * xf;
    let c11 = c011 * (1.0 - xf) + c111 * xf;

    let c0 = c00 * (1.0 - yf) + c10 * yf;
    let c1 = c01 * (1.0 - yf) + c11 * yf;

    c0 * (1.0 - zf) + c1 * zf
}

#[cfg(test)]
mod tests {
    use super::*;
    use gr_core::spatial::{SDIM, SPATIAL_IDENTITY};

    const TOL: f64 = 1e-10;

    fn flat_adm() -> AdmDecomposition {
        AdmDecomposition {
            lapse: 1.0,
            shift: [0.0; 3],
            spatial_metric: SPATIAL_IDENTITY,
            spatial_metric_inv: SPATIAL_IDENTITY,
            det_gamma: 1.0,
        }
    }

    fn zero_extrinsic() -> ExtrinsicCurvatureData {
        ExtrinsicCurvatureData {
            k_ij: [[0.0; SDIM]; SDIM],
            york_time: 0.0,
            tracefree_k: [[0.0; SDIM]; SDIM],
        }
    }

    #[test]
    fn test_uniform_foliation_creation() {
        let fol = SedenionFoliation::uniform(4, 4, 4, 3, 0.1, 1.0);
        assert_eq!(fol.n_t, 3);
        assert_eq!(fol.slices.len(), 3);
        assert_eq!(fol.grid_dims(), (4, 4, 4));
    }

    #[test]
    fn test_single_slice_matches_sedenion_field() {
        let fol = SedenionFoliation::uniform(4, 4, 4, 1, 0.1, 1.0);
        let frust = fol.frustration_at_slice(0);
        // Uniform field should give frustration ~ 3/8 everywhere
        for &f in &frust {
            assert!(
                (f - VACUUM_ATTRACTOR).abs() < 0.01,
                "uniform frustration = {f}, expected ~{VACUUM_ATTRACTOR}"
            );
        }
    }

    #[test]
    fn test_evolve_zero_shift_zero_k_preserves() {
        let mut fol = SedenionFoliation::uniform(4, 4, 4, 2, 0.1, 1.0);
        let adm = flat_adm();
        let ext = zero_extrinsic();

        evolve_frustration_adm(&mut fol, 0, &adm, &ext, 1.0);

        let frust_0 = fol.frustration_at_slice(0);
        let frust_1 = fol.frustration_at_slice(1);

        // With zero shift and zero K, the evolved field should match the source
        for (f0, f1) in frust_0.iter().zip(frust_1.iter()) {
            assert!(
                (f0 - f1).abs() < 0.01,
                "evolved frustration changed: {f0} -> {f1}"
            );
        }
    }

    #[test]
    fn test_hamiltonian_source_zero_at_vacuum() {
        let fol = SedenionFoliation::uniform(4, 4, 4, 1, 0.1, 1.0);
        let adm = flat_adm();
        let ext = zero_extrinsic();

        let source = frustration_hamiltonian_source(&fol, 0, &adm, &ext, 1.0, 0.0);

        // With zero K, source = 0 regardless of frustration
        for &s in &source {
            assert!(s.abs() < TOL, "Hamiltonian source = {s}, expected 0");
        }
    }

    #[test]
    fn test_hamiltonian_source_with_curvature() {
        let fol = SedenionFoliation::uniform(4, 4, 4, 1, 0.1, 1.0);
        let adm = flat_adm();
        // Set nonzero K (isotropic expansion)
        let ext = ExtrinsicCurvatureData {
            k_ij: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            york_time: -3.0,
            tracefree_k: [[0.0; SDIM]; SDIM],
        };

        let source = frustration_hamiltonian_source(&fol, 0, &adm, &ext, 1.0, 0.0);

        // nu = nu_0 * exp(0) = 1.0 (frustration at vacuum, beta=0)
        // K = trace = 3.0, K^2 = 9.0
        // rho_eff = 0.5 * 1.0 * 9.0 = 4.5
        for &s in &source {
            assert!(
                (s - 4.5).abs() < 0.01,
                "Hamiltonian source = {s}, expected 4.5"
            );
        }
    }

    #[test]
    fn test_york_time_frustration_map_uniform() {
        let fol = SedenionFoliation::uniform(4, 4, 4, 1, 0.1, 1.0);
        let pairs = york_time_frustration_map(&fol, 0, 1.5);

        assert_eq!(pairs.len(), 4 * 4 * 4);
        for &(theta, f) in &pairs {
            assert!((theta - 1.5).abs() < TOL, "theta = {theta}");
            assert!(
                (f - VACUUM_ATTRACTOR).abs() < 0.01,
                "frustration = {f}"
            );
        }
    }

    #[test]
    fn test_york_time_frustration_map_spatial() {
        let fol = SedenionFoliation::uniform(2, 2, 2, 1, 0.1, 1.0);
        let n = 2 * 2 * 2;
        let york_times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

        let pairs = york_time_frustration_map_spatial(&fol, 0, &york_times);
        assert_eq!(pairs.len(), n);
        for (i, &(theta, _f)) in pairs.iter().enumerate() {
            assert!(
                (theta - i as f64 * 0.1).abs() < TOL,
                "theta[{i}] = {theta}"
            );
        }
    }

    #[test]
    fn test_trilinear_interp_at_grid_points() {
        // 2x2x2 grid with known values
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // At integer coordinates, interpolation should return exact values
        let v = trilinear_interp_periodic(&data, 2, 2, 2, 0.0, 0.0, 0.0);
        assert!((v - 0.0).abs() < TOL, "interp at (0,0,0) = {v}");
        let v = trilinear_interp_periodic(&data, 2, 2, 2, 1.0, 0.0, 0.0);
        assert!((v - 1.0).abs() < TOL, "interp at (1,0,0) = {v}");
    }

    #[test]
    fn test_trilinear_interp_midpoint() {
        // Uniform field -> interpolation returns same value everywhere
        let data = vec![1.0; 8];
        let v = trilinear_interp_periodic(&data, 2, 2, 2, 0.5, 0.5, 0.5);
        assert!((v - 1.0).abs() < TOL, "midpoint of uniform = {v}");
    }

    #[test]
    fn test_from_slices_consistency() {
        let s1 = SedenionField::uniform(3, 3, 3);
        let s2 = SedenionField::uniform(3, 3, 3);
        let fol = SedenionFoliation::from_slices(vec![s1, s2], 0.5, 1.0);
        assert_eq!(fol.n_t, 2);
        assert_eq!(fol.dt, 0.5);
        assert_eq!(fol.dx, 1.0);
    }

    #[test]
    #[should_panic(expected = "slice index out of bounds")]
    fn test_frustration_at_invalid_slice_panics() {
        let fol = SedenionFoliation::uniform(2, 2, 2, 1, 0.1, 1.0);
        let _ = fol.frustration_at_slice(1);
    }
}
