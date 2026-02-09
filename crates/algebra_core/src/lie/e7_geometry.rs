//! E7 Geometry and Triad Interaction Maps.
//!
//! E7 is the exceptional Lie algebra of rank 7 and dimension 133.
//! Its root system has 126 roots in 7-dimensional space, embedded
//! in our 8-dimensional E8 coordinate basis.
//!
//! # Construction
//! E7 is constructed as the subset of E8 roots orthogonal to a fixed
//! reference root alpha_1 = (1,-1,0,0,0,0,0,0). The orthogonal
//! complement of a single root in E8 gives exactly the E7 root system
//! (126 roots), since the centralizer decomposes as E7 x A1.
//!
//! # Triad Dynamics
//! A "triad" is a triple (k, p, q) of roots satisfying k + p + q = 0,
//! equivalently q = -(k + p) where k + p is also a root. In the context
//! of turbulence, these represent resonant three-wave interactions.
//! The structure constant N(k, p) from e7_structure determines the
//! interaction strength.

use std::collections::HashMap;

use super::e7_structure::structure_constant;
use super::e8_lattice::{generate_e8_roots, E8Root};

/// An E7 root vector (subset of E8 root system).
#[derive(Debug, Clone, PartialEq)]
pub struct E7Root {
    pub root: E8Root,
}

/// A triad of interacting modes (k, p, q) with k + p + q = 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Triad {
    pub k: E7Root,
    pub p: E7Root,
    pub q: E7Root,
    /// Structure constant |N(k, p)| determining interaction strength.
    pub interaction_strength: f64,
}

/// Discretize coordinates to integer keys for hash-based lookup.
///
/// E8 roots have coordinates that are either integers or half-integers
/// (multiples of 0.5). Multiplying by 2 gives integers in all cases.
fn coord_key(coords: &[f64; 8]) -> [i64; 8] {
    let mut key = [0i64; 8];
    for (i, &c) in coords.iter().enumerate() {
        key[i] = (c * 2.0).round() as i64;
    }
    key
}

/// Generate E7 roots by selecting E8 roots orthogonal to a reference.
///
/// The reference root is alpha_1 = (1, -1, 0, 0, 0, 0, 0, 0), the first
/// simple root of E8 in the epsilon basis. The 126 roots orthogonal to
/// alpha_1 form the E7 root system.
pub fn generate_e7_roots() -> Vec<E7Root> {
    let e8_roots = generate_e8_roots();
    let ref_root = E8Root::new([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let e7_roots: Vec<E7Root> = e8_roots
        .into_iter()
        .filter(|r| r.inner_product(&ref_root).abs() < 1e-10)
        .map(|r| E7Root { root: r })
        .collect();

    assert_eq!(
        e7_roots.len(),
        126,
        "E7 root system must have exactly 126 roots"
    );
    e7_roots
}

/// Find all triads (k, p, q) in the E7 root set with k + p + q = 0.
///
/// Uses hash-based lookup for O(N^2) total complexity (N=126).
/// Each ordered pair (k, p) with k != p determines q = -(k+p);
/// we check whether q exists in the root set via HashMap lookup.
/// Triads are deduplicated by requiring canonical index ordering i < j < q_idx.
pub fn find_e7_triads(roots: &[E7Root]) -> Vec<Triad> {
    // Build lookup table: discretized coordinates -> index
    let root_map: HashMap<[i64; 8], usize> = roots
        .iter()
        .enumerate()
        .map(|(i, r)| (coord_key(&r.root.coords), i))
        .collect();

    let mut triads = Vec::new();

    for (i, k) in roots.iter().enumerate() {
        for (j, p) in roots.iter().enumerate() {
            if i == j {
                continue;
            }

            // q = -(k + p)
            let mut q_coords = [0.0; 8];
            for (d, q_c) in q_coords.iter_mut().enumerate() {
                *q_c = -(k.root.coords[d] + p.root.coords[d]);
            }

            let q_key = coord_key(&q_coords);
            if let Some(&q_idx) = root_map.get(&q_key) {
                // Canonical ordering: emit each unordered {k,p,q} exactly once
                if i < j && j < q_idx {
                    let strength = structure_constant(k, p).abs();
                    triads.push(Triad {
                        k: k.clone(),
                        p: p.clone(),
                        q: roots[q_idx].clone(),
                        interaction_strength: strength,
                    });
                }
            }
        }
    }

    triads
}

/// Project an E8 root vector onto a 2D plane for visualization.
///
/// Uses a pseudo-Coxeter projection that reveals the high symmetry of
/// the root system. The projection vectors alternate odd/even coordinates,
/// giving a visually appealing distribution that preserves the root system's
/// symmetry structure.
pub fn project_to_plane(root: &E8Root) -> (f64, f64) {
    let c = &root.coords;
    // Alternating coordinate sums approximate the Coxeter plane.
    // A full implementation would eigendecompose the Coxeter element.
    let u = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let v = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let x: f64 = c.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
    let y: f64 = c.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e7_root_count() {
        let roots = generate_e7_roots();
        assert_eq!(roots.len(), 126);
    }

    #[test]
    fn test_e7_roots_orthogonal_to_reference() {
        let roots = generate_e7_roots();
        let ref_root = E8Root::new([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for r in &roots {
            assert!(
                r.root.inner_product(&ref_root).abs() < 1e-10,
                "E7 root {:?} not orthogonal to reference",
                r.root.coords
            );
        }
    }

    #[test]
    fn test_e7_roots_have_norm_2() {
        let roots = generate_e7_roots();
        for r in &roots {
            assert!(
                (r.root.norm_sq - 2.0).abs() < 1e-10,
                "E7 root {:?} has norm^2 = {}, expected 2.0",
                r.root.coords,
                r.root.norm_sq
            );
        }
    }

    #[test]
    fn test_triads_are_nonempty() {
        let roots = generate_e7_roots();
        let triads = find_e7_triads(&roots);
        assert!(
            !triads.is_empty(),
            "E7 should have triads (root pairs whose sum is a root)"
        );
    }

    #[test]
    fn test_triad_closure() {
        let roots = generate_e7_roots();
        let triads = find_e7_triads(&roots);

        for triad in &triads {
            for d in 0..8 {
                let sum =
                    triad.k.root.coords[d] + triad.p.root.coords[d] + triad.q.root.coords[d];
                assert!(
                    sum.abs() < 1e-10,
                    "Triad closure violated: k+p+q[{}] = {}",
                    d,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_triads_have_valid_interaction_strength() {
        let roots = generate_e7_roots();
        let triads = find_e7_triads(&roots);

        for triad in &triads {
            assert!(
                triad.interaction_strength == 0.0 || triad.interaction_strength == 1.0,
                "Interaction strength should be 0 or 1, got {}",
                triad.interaction_strength
            );
        }
    }

    #[test]
    fn test_triads_no_duplicates() {
        let roots = generate_e7_roots();
        let triads = find_e7_triads(&roots);

        let mut seen = std::collections::HashSet::new();
        for t in &triads {
            let mut keys = [
                coord_key(&t.k.root.coords),
                coord_key(&t.p.root.coords),
                coord_key(&t.q.root.coords),
            ];
            keys.sort();
            assert!(seen.insert(keys), "Duplicate triad found");
        }
    }

    #[test]
    fn test_projection_bounded() {
        let roots = generate_e7_roots();
        for r in &roots {
            let (x, y) = project_to_plane(&r.root);
            assert!(
                x.abs() < 10.0 && y.abs() < 10.0,
                "Projection out of bounds: ({}, {})",
                x,
                y
            );
        }
    }
}
