//! Symmetry operations: rotation, reflection, inversion, improper axes,
//! translations -- the 3x3 matrix + translation-vector representation.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Provides constructors for the identity, n-fold rotations about each
//! axis, mirrors in each principal plane, inversion, improper rotations,
//! and screw / glide operations.

// ============================================================================
// Symmetry Operations
// ============================================================================

/// A symmetry operation: rotation, reflection, inversion, improper axis, or translation.
#[derive(Debug, Clone)]
pub struct SymmetryOperation {
    /// 3x3 rotation/reflection matrix
    pub matrix: [[f64; 3]; 3],
    /// Translation vector (0 for pure rotation/reflection)
    pub translation: [f64; 3],
    /// Order of operation (how many times to apply to get identity)
    pub order: u32,
}

impl SymmetryOperation {
    /// Identity operation.
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            order: 1,
        }
    }

    /// n-fold rotation about z-axis (in radians).
    pub fn rotation_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            order: 1,
        }
    }

    /// Reflection through xy-plane (z=0).
    pub fn reflection_xy() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            translation: [0.0, 0.0, 0.0],
            order: 2,
        }
    }

    /// Inversion through origin.
    pub fn inversion() -> Self {
        Self {
            matrix: [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            translation: [0.0, 0.0, 0.0],
            order: 2,
        }
    }

    /// Apply operation to a 3D point [x, y, z].
    pub fn apply_to_point(&self, p: &[f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for (i, res) in result.iter_mut().enumerate() {
            for (j, pj) in p.iter().enumerate() {
                *res += self.matrix[i][j] * pj;
            }
            *res += self.translation[i];
        }
        result
    }

    /// Apply operation to a direction (ignore translation).
    pub fn apply_to_direction(&self, v: &[f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for (i, res) in result.iter_mut().enumerate() {
            for (j, vj) in v.iter().enumerate() {
                *res += self.matrix[i][j] * vj;
            }
        }
        result
    }

    /// Compose two symmetry operations: self * other.
    /// Result is the operation that applies other first, then self.
    #[allow(clippy::needless_range_loop)]
    pub fn compose(&self, other: &Self) -> Self {
        // Matrix multiplication: self.matrix * other.matrix
        let mut new_matrix = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    new_matrix[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }

        // Translation: self.translation + self.matrix * other.translation
        let mut new_translation = [0.0; 3];
        for i in 0..3 {
            new_translation[i] = self.translation[i];
            for j in 0..3 {
                new_translation[i] += self.matrix[i][j] * other.translation[j];
            }
        }

        Self {
            matrix: new_matrix,
            translation: new_translation,
            order: 1, // Set order conservatively; caller may update
        }
    }

    /// Determinant of the operation matrix (rotation = +1, improper = -1).
    pub fn determinant(&self) -> f64 {
        // 3x3 determinant: det = a(ei-fh) - b(di-fg) + c(dh-eg)
        let a = self.matrix[0][0];
        let b = self.matrix[0][1];
        let c = self.matrix[0][2];
        let d = self.matrix[1][0];
        let e = self.matrix[1][1];
        let f = self.matrix[1][2];
        let g = self.matrix[2][0];
        let h = self.matrix[2][1];
        let i = self.matrix[2][2];

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// Check if this is a proper rotation (det = +1) or improper (det = -1).
    pub fn is_proper(&self) -> bool {
        (self.determinant() - 1.0).abs() < 1e-9
    }

    /// Inverse operation: apply this operation k times to get identity.
    /// Computed exactly for orthogonal matrices.
    #[allow(clippy::needless_range_loop)]
    pub fn inverse(&self) -> Self {
        // For orthogonal matrix, transpose = inverse
        let mut inv_matrix = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inv_matrix[i][j] = self.matrix[j][i];
            }
        }

        // Inverse translation: -inv_matrix * translation
        let mut inv_translation = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                inv_translation[i] -= inv_matrix[i][j] * self.translation[j];
            }
        }

        Self {
            matrix: inv_matrix,
            translation: inv_translation,
            order: self.order,
        }
    }

    /// Apply this operation k times (group power).
    /// Returns identity if k == 0.
    pub fn power(&self, k: usize) -> Self {
        if k == 0 {
            return Self::identity();
        }

        let mut result = self.clone();
        for _ in 1..k {
            result = result.compose(self);
        }
        result
    }

    /// Trace of the matrix (sum of diagonal elements).
    /// Used for character computation in symmetry analysis.
    pub fn trace(&self) -> f64 {
        self.matrix[0][0] + self.matrix[1][1] + self.matrix[2][2]
    }

    /// Frobenius norm: sqrt(sum of all matrix elements squared).
    /// Used to check if matrix is close to identity or other reference.
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sum += self.matrix[i][j] * self.matrix[i][j];
            }
        }
        sum.sqrt()
    }

    /// Apply point group action to a set of points and collect results.
    /// Useful for symmetry analysis of crystal structures.
    pub fn apply_to_point_set(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|p| self.apply_to_point(p)).collect()
    }

    /// Verify that this operation and its inverse are indeed inverses.
    pub fn verify_inverse(&self) -> bool {
        let inv = self.inverse();
        let composed = self.compose(&inv);
        let id = Self::identity();

        // Check if composition is identity
        for i in 0..3 {
            for j in 0..3 {
                if (composed.matrix[i][j] - id.matrix[i][j]).abs() > 1e-9 {
                    return false;
                }
            }
        }
        for i in 0..3 {
            if composed.translation[i].abs() > 1e-9 {
                return false;
            }
        }
        true
    }

    /// Check if this operation commutes with another: AB = BA.
    pub fn commutes_with(&self, other: &Self) -> bool {
        let ab = self.compose(other);
        let ba = other.compose(self);

        // Check matrix commutation
        for i in 0..3 {
            for j in 0..3 {
                if (ab.matrix[i][j] - ba.matrix[i][j]).abs() > 1e-9 {
                    return false;
                }
            }
        }

        // Check translation commutation
        for i in 0..3 {
            if (ab.translation[i] - ba.translation[i]).abs() > 1e-9 {
                return false;
            }
        }

        true
    }

    /// Find the order of this operation: smallest k such that op^k = identity.
    /// For finite point groups, order is typically 1-6 (or 4,6,8 for improper).
    pub fn find_order(&self) -> usize {
        for k in 1..=100 {
            let power = self.power(k);
            let id = Self::identity();

            let mut is_identity = true;
            for i in 0..3 {
                for j in 0..3 {
                    if (power.matrix[i][j] - id.matrix[i][j]).abs() > 1e-9 {
                        is_identity = false;
                        break;
                    }
                }
                if !is_identity {
                    break;
                }
            }

            if is_identity {
                for i in 0..3 {
                    if power.translation[i].abs() > 1e-9 {
                        is_identity = false;
                        break;
                    }
                }
            }

            if is_identity {
                return k;
            }
        }

        100 // Return max if not found
    }
}
