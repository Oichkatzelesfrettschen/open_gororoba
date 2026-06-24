//! 3D Algebraic Field Evolution.
//!
//! Implements rigorous evolution of hypercomplex fields on a 3D lattice,
//! including advection by fluid velocity and non-associative imbalance
//! coupling.

use crate::simulation::state_3d::{AlgebraicField3D, LbmBackend3D};
use ndarray::Array3;

// Implement AlgebraicField3D for concrete types
impl AlgebraicField3D for Array3<[f64; 16]> {} // For Sedenion field

/// Evolve the 3D algebraic field.
pub fn evolve_algebra_3d(
    algebra: &mut Array3<[f64; 16]>,
    fluid: &LbmBackend3D,
    curvature: &Array3<f64>,
    coupling: f64,
) {
    advect_sedenion_field(algebra, fluid, coupling);
    apply_imbalance_3d(algebra, fluid, curvature, coupling);
}

fn advect_sedenion_field(field: &mut Array3<[f64; 16]>, fluid: &LbmBackend3D, coupling: f64) {
    let (nx, ny, nz) = field.dim();
    let old_field = field.clone();

    // Get velocities from fluid backend
    let u_field: Vec<[f64; 3]> = match fluid {
        LbmBackend3D::Cpu(solver) => solver.u.clone(),
        #[cfg(feature = "gpu")]
        LbmBackend3D::Gpu(solver) => {
            // Need to ensure solver.u is synced
            solver
                .u
                .iter()
                .map(|&[fx, fy, fz]| [fx as f64, fy as f64, fz as f64])
                .collect()
        }
        #[cfg(not(feature = "gpu"))]
        LbmBackend3D::Gpu => panic!("GPU backend not enabled"),
    };

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = z * (nx * ny) + y * nx + x;
                let [ux, uy, uz] = u_field[idx];

                // Semi-Lagrangian back-trace
                let sx = (x as f64 - ux * coupling).rem_euclid(nx as f64) as usize;
                let sy = (y as f64 - uy * coupling).rem_euclid(ny as f64) as usize;
                let sz = (z as f64 - uz * coupling).rem_euclid(nz as f64) as usize;

                field[[x, y, z]] = old_field[[sx, sy, sz]];
            }
        }
    }
}

fn apply_imbalance_3d(
    field: &mut Array3<[f64; 16]>,
    _fluid: &LbmBackend3D,
    curvature: &Array3<f64>,
    coupling: f64,
) {
    let (nx, ny, nz) = field.dim();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let k_scalar = curvature[[x, y, z]];

                // Imbalance source: inject "algebraic noise" proportional to curvature AND coupling
                let mut phi = field[[x, y, z]];
                for v in phi.iter_mut().skip(8) {
                    *v += k_scalar * coupling * 0.1;
                }
                field[[x, y, z]] = phi;
            }
        }
    }
}
