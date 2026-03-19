//! Topological-Algebraic Bridge Theory.
//!
//! Explores tripotent ($p^3 = p$) geometry, fractional associativity,
//! and stability strata in hypercomplex algebras.
//!
//! Distilled from advanced theoretical frameworks regarding quantum foam and
//! exceptional symmetries.

use crate::construction::cayley_dickson::{cd_multiply};
use crate::physics::sedenion_field::Sedenion;

/// Normalized measure of non-associativity: A(a,b,c) = ||(ab)c - a(bc)|| / (||a|| ||b|| ||c||).
///
/// Gives a continuous measure of how associative three elements are.
/// A = 0 means fully associative. A = 1 means maximally non-associative.
pub fn fractional_associativity(a: &Sedenion, b: &Sedenion, c: &Sedenion) -> f64 {
    let ab = cd_multiply(a, b);
    let bc = cd_multiply(b, c);
    let ab_c = cd_multiply(&ab, c);
    let a_bc = cd_multiply(a, &bc);
    
    let mut diff = [0.0; 16];
    for i in 0..16 {
        diff[i] = ab_c[i] - a_bc[i];
    }
    let norm_diff = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
    
    let norm_a = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_c = c.iter().map(|&x| x * x).sum::<f64>().sqrt();
    
    let denom = norm_a * norm_b * norm_c;
    if denom < 1e-12 {
        0.0
    } else {
        norm_diff / denom
    }
}

/// Computes the stability transition function T(x, y).
///
/// T(x, y) = sup { A(a,b,c) | a,b,c in span{x,y} }.
/// This implementation approximates the supremum by random sampling
/// the 2D subspace spanned by x and y.
pub fn stability_transition(x: &Sedenion, y: &Sedenion, samples: usize) -> f64 {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let mut max_a = 0.0_f64;

    let sample_subspace = |rng: &mut StdRng| -> Sedenion {
        let alpha: f64 = rng.gen_range(-1.0..1.0);
        let beta: f64 = rng.gen_range(-1.0..1.0);
        let mut v = [0.0; 16];
        for i in 0..16 {
            v[i] = alpha * x[i] + beta * y[i];
        }
        v
    };

    for _ in 0..samples {
        let a = sample_subspace(&mut rng);
        let b = sample_subspace(&mut rng);
        let c = sample_subspace(&mut rng);
        
        let a_val = fractional_associativity(&a, &b, &c);
        if a_val > max_a {
            max_a = a_val;
        }
    }
    max_a
}

/// Checks if an element acts as a tripotent: p^3 = p.
pub fn is_tripotent(p: &Sedenion, tol: f64) -> bool {
    let p2 = cd_multiply(p, p);
    let p3 = cd_multiply(&p2, p);
    
    for i in 0..16 {
        if (p3[i] - p[i]).abs() > tol {
            return false;
        }
    }
    true
}

/// Classifies an element into stability strata based on its local associativity domain.
///
/// S0 (Perfect): A = 0
/// S1 (Near): 0 < A <= eps
/// S2 (Flux): eps < A <= delta
/// S3 (Chaos): A > delta
pub enum StabilityStratum {
    Perfect,
    Near,
    Flux,
    Chaos,
}

pub fn classify_stability(a: &Sedenion, b: &Sedenion, c: &Sedenion, eps: f64, delta: f64) -> StabilityStratum {
    let a_val = fractional_associativity(a, b, c);
    if a_val <= 1e-12 {
        StabilityStratum::Perfect
    } else if a_val <= eps {
        StabilityStratum::Near
    } else if a_val <= delta {
        StabilityStratum::Flux
    } else {
        StabilityStratum::Chaos
    }
}
