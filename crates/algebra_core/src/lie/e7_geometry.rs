//! E7 Geometry and Triad Interaction Maps.
//!
//! E7 is the exceptional Lie algebra of rank 7 and dimension 133.
//! Its root system has 126 roots.
//!
//! In the context of fluid turbulence, E7 geometry models the
//! interaction of triad modes in the energy cascade.
//!
//! # Construction
//! We construct E7 as the subsystem of E8 orthogonal to a specific root
//! (typically the highest root or a simple root).
//!
//! # Triad Dynamics
//! - Roots represent flow modes.
//! - A "Triad" is a set of three roots {k, p, q} such that k + p + q = 0.
//! - The structure constants of E7 determine the interaction strength.

use super::e8_lattice::{generate_e8_roots, E8Root};

/// An E7 root vector (subset of E8).
#[derive(Debug, Clone, PartialEq)]
pub struct E7Root {
    pub root: E8Root,
}

/// A Triad of interacting modes {k, p, q}.
#[derive(Debug, Clone, PartialEq)]
pub struct Triad {
    pub k: E7Root,
    pub p: E7Root,
    pub q: E7Root,
    pub interaction_strength: f64,
}

/// Generate E7 roots by filtering E8 roots.
///
/// We select roots orthogonal to the canonical 8th direction in E8 basis,
/// or specifically orthogonal to the highest root of E8 relative to E7 embedding.
///
/// Here we define E7 as roots r in E8 such that (r, alpha_fixed) = 0.
/// We use alpha = (0,0,0,0,0,0,0,1) + (0,0,0,0,0,0,0,-1) ... wait.
///
/// E8 standard roots:
/// Type 1: Permutations of (+/-1, +/-1, 0^6)
/// Type 2: (+/-0.5)^8 (even minus signs)
///
/// We select roots orthogonal to `w = (0,0,0,0,0,0,1,-1)` (which is a root).
///
pub fn generate_e7_roots() -> Vec<E7Root> {
    let e8_roots = generate_e8_roots();
    // Reference vector for orthogonality: e_8 - e_7 (in standard basis this is a root)
    // Actually, let's use the standard canonical embedding where E7 lives in the hyperplane
    // orthogonal to a specific vector.
    //
    // Using Bourbaki: E7 roots are E8 roots involving only coordinates 1..7?
    // No, E8 basis is complex.
    //
    // Let's use the orthogonality constraint against the last simple root of E8 if we want E7.
    // But e8_simple_roots() uses a specific basis.
    //
    // Let's pick a specific root `ref_root` from the generated set.
    // The highest root of E8 is usually (1, 1, 0, 0, 0, 0, 0, 0)? No.
    //
    // Let's pick `ref_root` = (1, -1, 0, 0, 0, 0, 0, 0) (alpha_1 in our basis).
    // The subset orthogonal to this should be E7.
    
    // Let's verify with counts:
    // We need exactly 126 roots.
    
    let ref_root = E8Root::new([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    
    let mut e7_roots = Vec::new();
    for r in e8_roots {
        // Inner product should be 0
        if r.inner_product(&ref_root).abs() < 1e-10 {
            e7_roots.push(E7Root { root: r });
        }
    }
    
    // NOTE: Orthogonal complement of a root in E8 is E7 + A1? No, Centralizer is E7 x A1.
    // Roots in orthogonal complement are roots of E7 + roots of A1.
    // Roots of A1 are just +/- ref_root? No, ref_root is orthogonal to itself? No (norm 2).
    // Roots orthogonal to ref_root do NOT include ref_root.
    // So we just get E7 roots.
    //
    // Wait, E8 roots = 240.
    // Roots with (r, alpha) = 0: 126.
    // 126 is exactly |Phi(E7)|.
    
    assert_eq!(e7_roots.len(), 126, "Should find exactly 126 E7 roots orthogonal to reference");
    e7_roots
}

/// Find all valid triads {k, p, q} in the E7 set such that k + p + q = 0.
pub fn find_e7_triads(roots: &[E7Root]) -> Vec<Triad> {
    let mut triads = Vec::new();
    
    // Brute force O(N^2) since q is determined by -k-p
    // 126^2 = 15876 iterations, very fast.
    
    for (i, k) in roots.iter().enumerate() {
        for (j, p) in roots.iter().enumerate() {
            if i == j { continue; }
            
            // q = -(k + p)
            let mut q_coords = [0.0; 8];
            for d in 0..8 {
                q_coords[d] = -(k.root.coords[d] + p.root.coords[d]);
            }
            let q_cand = E8Root::new(q_coords);
            
            // Check if q is a valid root in the set
            // (k+p+q=0 implies q must be a root if k,p are roots and they form a triad in a Lie algebra,
            // specifically if k+p is a root. k+p might NOT be a root).
            
            if q_cand.is_valid_root() {
                // Check if q exists in our E7 set
                if let Some(q_ref) = roots.iter().find(|r| 
                    r.root.coords.iter().zip(q_cand.coords.iter()).all(|(a,b)| (a-b).abs() < 1e-10)
                ) {
                    // Valid triad found
                    triads.push(Triad {
                        k: k.clone(),
                        p: p.clone(),
                        q: q_ref.clone(),
                        interaction_strength: 1.0, // Placeholder for structure constant
                    });
                }
            }
        }
    }
    
    triads
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
    fn test_triad_closure() {
        let roots = generate_e7_roots();
        let triads = find_e7_triads(&roots);
        
        // E7 is a Lie algebra, so [x,y] is in L.
        // If roots alpha, beta are such that alpha+beta is a root, they form a triad (-alpha-beta).
        // Number of triads should be related to structure constants.
        // Just checking we found some.
        assert!(triads.len() > 0);
    }
}
