//! Landau-de Gennes Q-tensor formalism for nematic liquid crystals.
//!
//! This module implements the orientational order description of rod-like
//! molecules in the nematic mesophase using the symmetric, traceless Q-tensor.
//!
//! # Overview
//! - **Q-tensor**: Symmetric traceless 3x3 matrix encoding orientational order
//! - **Scalar order parameter S**: Largest eigenvalue of Q scaled by 3/2
//! - **Director**: Principal eigenvector (headless: n and -n are equivalent)
//! - **Frank-Oseen elasticity**: Splay, twist, bend elastic energy density
//!
//! # Key Structures
//! - `QTensor`: The Landau-de Gennes order parameter
//! - `FrankConstants`: K1 (splay), K2 (twist), K3 (bend) elastic moduli
//!
//! # References
//! - de Gennes & Prost (1993), The Physics of Liquid Crystals
//! - Virga (1994), Variational Theories for Liquid Crystals
//! - Stewart (2004), The Static and Dynamic Continuum Theory of Liquid Crystals

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

// ============================================================================
// Q-tensor
// ============================================================================

/// Landau-de Gennes Q-tensor for nematic order.
///
/// Q is a symmetric, traceless 3x3 matrix with 5 independent components.
/// It describes the orientational order of rod-like molecules:
/// - Isotropic phase: Q = 0
/// - Uniaxial nematic: Q = S(nn^T - I/3) where n is the director, S in [0,1]
/// - Biaxial: Q has two distinct non-zero eigenvalues
#[derive(Clone, Debug)]
pub struct QTensor {
    /// Symmetric traceless 3x3 matrix.
    /// Stored as full Matrix3 but enforced traceless at construction.
    pub q: Matrix3<f64>,
}

/// Frank-Oseen elastic constants for nematic liquid crystals.
/// Typical values for 5CB (4-cyano-4'-pentylbiphenyl) at 25 C.
#[derive(Clone, Debug)]
pub struct FrankConstants {
    /// Splay elastic constant (N), typical ~6.2e-12 for 5CB
    pub k1_splay: f64,
    /// Twist elastic constant (N), typical ~3.9e-12 for 5CB
    pub k2_twist: f64,
    /// Bend elastic constant (N), typical ~8.2e-12 for 5CB
    pub k3_bend: f64,
}

// ============================================================================
// QTensor implementation
// ============================================================================

impl QTensor {
    /// Create a Q-tensor from a symmetric matrix, enforcing tracelessness.
    /// Subtracts tr(q)/3 * I from the input to ensure trace = 0.
    pub fn new(q: Matrix3<f64>) -> Self {
        let tr = q.trace();
        let correction = Matrix3::identity() * (tr / 3.0);
        Self { q: q - correction }
    }

    /// Create the zero (isotropic) Q-tensor.
    pub fn isotropic() -> Self {
        Self {
            q: Matrix3::zeros(),
        }
    }

    /// Scalar order parameter S = largest eigenvalue of Q * 3/2.
    /// Returns 0 for isotropic, approaches 1 for perfectly aligned nematic.
    pub fn scalar_order_parameter(&self) -> f64 {
        let eigen = SymmetricEigen::new(self.q);
        let eigenvalues = eigen.eigenvalues;
        // Find the eigenvalue with the largest absolute value, preserving sign.
        // For a physical nematic Q-tensor the largest eigenvalue is positive
        // and equals 2S/3.
        let mut max_val = eigenvalues[0];
        for i in 1..3 {
            if eigenvalues[i].abs() > max_val.abs() {
                max_val = eigenvalues[i];
            }
        }
        max_val * 1.5
    }

    /// Director: eigenvector corresponding to the largest eigenvalue.
    /// Returns None for isotropic phase (S ~ 0).
    pub fn director(&self) -> Option<Vector3<f64>> {
        let eigen = SymmetricEigen::new(self.q);
        let eigenvalues = eigen.eigenvalues;

        // Find index of eigenvalue with largest absolute value
        let mut max_idx = 0;
        let mut max_abs = eigenvalues[0].abs();
        for i in 1..3 {
            if eigenvalues[i].abs() > max_abs {
                max_abs = eigenvalues[i].abs();
                max_idx = i;
            }
        }

        // If all eigenvalues are near zero, the phase is isotropic
        if max_abs < 1e-10 {
            return None;
        }

        // The eigenvectors matrix has columns as eigenvectors
        Some(eigen.eigenvectors.column(max_idx).into_owned())
    }

    /// Trace of Q (should be ~0 by construction).
    pub fn trace(&self) -> f64 {
        self.q.trace()
    }
}

// ============================================================================
// Free functions
// ============================================================================

/// Construct Q-tensor from director and scalar order parameter.
/// Q = S * (n*n^T - I/3)
pub fn nematic_from_director(n: &Vector3<f64>, s: f64) -> QTensor {
    let nn_t = n * n.transpose();
    let identity = Matrix3::identity();
    let q = (nn_t - identity / 3.0) * s;
    QTensor { q }
}

/// Frank-Oseen elastic free energy density for a director field.
///
/// Given the director n and its spatial gradients (dn/dx, dn/dy, dn/dz),
/// computes:
///   f = (K1/2)(div n)^2 + (K2/2)(n . curl n)^2 + (K3/2)(n x curl n)^2
///
/// For a spatially uniform director, all gradients vanish, so f_elastic = 0.
pub fn frank_energy_density(
    n: &Vector3<f64>,
    dn_dx: &Vector3<f64>,
    dn_dy: &Vector3<f64>,
    dn_dz: &Vector3<f64>,
    fc: &FrankConstants,
) -> f64 {
    // Divergence: div(n) = dn_x/dx + dn_y/dy + dn_z/dz
    let div_n = dn_dx.x + dn_dy.y + dn_dz.z;

    // Curl: curl(n) = ( dn_z/dy - dn_y/dz,
    //                    dn_x/dz - dn_z/dx,
    //                    dn_y/dx - dn_x/dy )
    let curl_n = Vector3::new(
        dn_dy.z - dn_dz.y,
        dn_dz.x - dn_dx.z,
        dn_dx.y - dn_dy.x,
    );

    // n . curl(n) -- twist contribution
    let n_dot_curl = n.dot(&curl_n);

    // n x curl(n) -- bend contribution
    let n_cross_curl = n.cross(&curl_n);
    let bend_sq = n_cross_curl.norm_squared();

    0.5 * fc.k1_splay * div_n * div_n
        + 0.5 * fc.k2_twist * n_dot_curl * n_dot_curl
        + 0.5 * fc.k3_bend * bend_sq
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qtensor_traceless() {
        let q = Matrix3::new(1.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 3.0);
        let qt = QTensor::new(q);
        assert!(
            qt.trace().abs() < 1e-12,
            "Q must be traceless, got trace={}",
            qt.trace()
        );
    }

    #[test]
    fn test_isotropic_zero_order() {
        let qt = QTensor::isotropic();
        assert!(qt.scalar_order_parameter().abs() < 1e-12);
        assert!(qt.director().is_none());
    }

    #[test]
    fn test_perfect_nematic() {
        let n = Vector3::new(0.0, 0.0, 1.0);
        let qt = nematic_from_director(&n, 1.0);
        let s = qt.scalar_order_parameter();
        assert!(
            (s - 1.0).abs() < 1e-10,
            "Perfect nematic S={}, expected 1.0",
            s
        );
        let dir = qt.director().expect("Should have director");
        // Director can be +z or -z (headless)
        assert!((dir.z.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frank_energy_uniform_zero() {
        let n = Vector3::new(0.0, 0.0, 1.0);
        let zero = Vector3::zeros();
        let fc = FrankConstants {
            k1_splay: 6.2e-12,
            k2_twist: 3.9e-12,
            k3_bend: 8.2e-12,
        };
        let f = frank_energy_density(&n, &zero, &zero, &zero, &fc);
        assert!(
            f.abs() < 1e-30,
            "Uniform director should have zero elastic energy"
        );
    }

    #[test]
    fn test_nematic_from_director_traceless() {
        let n = Vector3::new(1.0, 1.0, 1.0).normalize();
        let qt = nematic_from_director(&n, 0.7);
        assert!(qt.trace().abs() < 1e-12);
    }

    #[test]
    fn test_frank_pure_splay() {
        // Director along z, but with dn_x/dx != 0 (splay only)
        let n = Vector3::new(0.0, 0.0, 1.0);
        // div(n) = dn_x/dx + dn_y/dy + dn_z/dz.
        // For pure splay: set dn_x/dx = 0.1, rest zero.
        let dn_dx = Vector3::new(0.1, 0.0, 0.0);
        let dn_dy = Vector3::zeros();
        let dn_dz = Vector3::zeros();
        let fc = FrankConstants {
            k1_splay: 1.0,
            k2_twist: 0.0,
            k3_bend: 0.0,
        };
        let f = frank_energy_density(&n, &dn_dx, &dn_dy, &dn_dz, &fc);
        // div(n) = 0.1, f = 0.5 * 1.0 * 0.01 = 0.005
        assert!(
            (f - 0.005).abs() < 1e-14,
            "Pure splay energy = {}, expected 0.005",
            f
        );
    }

    #[test]
    fn test_scalar_order_half() {
        let n = Vector3::new(1.0, 0.0, 0.0);
        let qt = nematic_from_director(&n, 0.5);
        let s = qt.scalar_order_parameter();
        assert!(
            (s - 0.5).abs() < 1e-10,
            "Half-order nematic S={}, expected 0.5",
            s
        );
    }

    #[test]
    fn test_director_x_axis() {
        let n = Vector3::new(1.0, 0.0, 0.0);
        let qt = nematic_from_director(&n, 0.8);
        let dir = qt.director().expect("Should have director");
        // Director should be along x (or -x)
        assert!(
            (dir.x.abs() - 1.0).abs() < 1e-10,
            "Director x-component = {}, expected +/-1",
            dir.x
        );
    }

    #[test]
    fn test_qtensor_symmetry_preserved() {
        let q = Matrix3::new(0.3, 0.1, 0.2, 0.1, -0.1, 0.05, 0.2, 0.05, -0.2);
        let qt = QTensor::new(q);
        // Check symmetry: q[i][j] == q[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qt.q[(i, j)] - qt.q[(j, i)]).abs() < 1e-14,
                    "Q not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
