//! E7 Structure Constants.
//!
//! Implements N(alpha, beta) for the E7 Lie algebra Chevalley basis.
//!
//! In a simply-laced algebra (ADE type, including E7), the structure
//! constants satisfy:
//! - N(alpha, beta) = 0 if alpha + beta is NOT a root
//! - N(alpha, beta) = +/-1 if alpha + beta IS a root
//! - Antisymmetry: N(alpha, beta) = -N(beta, alpha)
//!
//! The sign is determined by a fixed total ordering on roots (Chevalley-Tits
//! orientation). We use lexicographic ordering of root coordinates, which
//! is deterministic and platform-independent.

use super::e7_geometry::E7Root;

/// Calculate structure constant N(alpha, beta).
///
/// Returns +1.0, -1.0, or 0.0 following the rules:
/// - 0 if alpha + beta is not a root (norm^2 != 2)
/// - +1 if alpha < beta in lexicographic order and alpha + beta is a root
/// - -1 if alpha > beta in lexicographic order and alpha + beta is a root
pub fn structure_constant(alpha: &E7Root, beta: &E7Root) -> f64 {
    // Compute sum vector
    let mut sum_norm_sq = 0.0;
    for d in 0..8 {
        let s = alpha.root.coords[d] + beta.root.coords[d];
        sum_norm_sq += s * s;
    }

    // In E8/E7, all roots have norm^2 = 2
    if (sum_norm_sq - 2.0).abs() > 1e-10 {
        return 0.0;
    }

    // Determine sign via lexicographic ordering of coordinates
    match lex_cmp(&alpha.root.coords, &beta.root.coords) {
        std::cmp::Ordering::Less => 1.0,
        std::cmp::Ordering::Greater => -1.0,
        // alpha == beta implies alpha + beta has norm^2 = 8, not 2
        // so this branch is unreachable given the norm check above
        std::cmp::Ordering::Equal => 0.0,
    }
}

/// Lexicographic comparison of coordinate arrays.
///
/// Compares element-by-element with epsilon tolerance for floating-point.
/// Elements differing by less than 1e-12 are considered equal.
fn lex_cmp(a: &[f64; 8], b: &[f64; 8]) -> std::cmp::Ordering {
    for i in 0..8 {
        let diff = a[i] - b[i];
        if diff > 1e-12 {
            return std::cmp::Ordering::Greater;
        }
        if diff < -1e-12 {
            return std::cmp::Ordering::Less;
        }
    }
    std::cmp::Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::e7_geometry::generate_e7_roots;

    #[test]
    fn test_antisymmetry() {
        let roots = generate_e7_roots();
        // Check N(a,b) = -N(b,a) for all pairs that produce roots
        let mut nonzero_count = 0;
        for i in 0..roots.len().min(30) {
            for j in (i + 1)..roots.len().min(30) {
                let n_ab = structure_constant(&roots[i], &roots[j]);
                let n_ba = structure_constant(&roots[j], &roots[i]);
                assert!(
                    (n_ab + n_ba).abs() < 1e-14,
                    "Antisymmetry violated for roots {} and {}: N(a,b)={}, N(b,a)={}",
                    i,
                    j,
                    n_ab,
                    n_ba
                );
                if n_ab.abs() > 0.5 {
                    nonzero_count += 1;
                }
            }
        }
        assert!(
            nonzero_count > 0,
            "Should find at least some non-zero structure constants"
        );
    }

    #[test]
    fn test_zero_when_sum_not_root() {
        let roots = generate_e7_roots();
        // Find a pair where alpha + beta is NOT a root (norm != 2)
        let a = &roots[0];
        // The negative of a root is also a root, so a + (-a) = 0 (norm 0, not a root)
        let neg_a = E7Root {
            root: crate::lie::e8_lattice::E8Root::new({
                let mut c = [0.0; 8];
                for (i, val) in a.root.coords.iter().enumerate() {
                    c[i] = -val;
                }
                c
            }),
        };
        assert_eq!(structure_constant(a, &neg_a), 0.0);
    }

    #[test]
    fn test_values_are_unit() {
        let roots = generate_e7_roots();
        for i in 0..roots.len() {
            for j in 0..roots.len() {
                let n = structure_constant(&roots[i], &roots[j]);
                assert!(
                    n == 0.0 || n == 1.0 || n == -1.0,
                    "Structure constant must be 0, +1, or -1, got {}",
                    n
                );
            }
        }
    }

    #[test]
    fn test_lex_cmp_deterministic() {
        let a = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // a > b because a[0]=1.0 > b[0]=0.0
        assert_eq!(lex_cmp(&a, &b), std::cmp::Ordering::Greater);
        assert_eq!(lex_cmp(&b, &a), std::cmp::Ordering::Less);
        assert_eq!(lex_cmp(&a, &a), std::cmp::Ordering::Equal);
    }
}
