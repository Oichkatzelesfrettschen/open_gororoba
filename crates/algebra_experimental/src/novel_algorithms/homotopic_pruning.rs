//! Homotopic Neural Pruning (Stasheff Dropout)
//!
//! Traditional neural networks prune weights based on magnitude (L1/L2 norm).
//! This algorithm prunes weights based on topological obstruction, utilizing
//! the Stasheff polytopes (associahedra) from non-associative algebras.
//!
//! We treat a sequence of 3 neural layers as a non-associative triplet.
//! If the weights form an "associator defect" (a non-zero associator), they
//! are considered structurally critical (a topological obstruction). If they
//! strictly associate, they are algebraically redundant and can be pruned.

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

/// Represents a neural network weight tensor mapped into 16D space.
pub type SedenionWeight = [f64; 16];

/// Evaluates the $A_3$ Stasheff constraint (the Associator) for three adjacent weight matrices.
/// [W1, W2, W3] = (W1 * W2) * W3 - W1 * (W2 * W3)
pub fn evaluate_a3_constraint(
    w1: &SedenionWeight,
    w2: &SedenionWeight,
    w3: &SedenionWeight,
) -> f64 {
    let w12: [f64; 16] = cd_multiply(w1, w2).try_into().unwrap();
    let w23: [f64; 16] = cd_multiply(w2, w3).try_into().unwrap();

    let left_eval: [f64; 16] = cd_multiply(&w12, w3).try_into().unwrap();
    let right_eval: [f64; 16] = cd_multiply(w1, &w23).try_into().unwrap();

    let mut defect = [0.0; 16];
    for i in 0..16 {
        defect[i] = left_eval[i] - right_eval[i];
    }

    cd_norm_sq(&defect).sqrt()
}

/// **Homotopic Neural Pruning**
/// Iterates over a sequence of layers. Prunes (zeros out) weights that
/// exhibit high associativity (defect < threshold). It preserves the weights
/// that generate non-associative topological defects, arguing they contain
/// the high-dimensional nonlinear routing.
pub fn prune_by_homotopy(
    layer1: &mut [SedenionWeight],
    layer2: &mut [SedenionWeight],
    layer3: &mut [SedenionWeight],
    associativity_threshold: f64,
) -> usize {
    assert_eq!(layer1.len(), layer2.len());
    assert_eq!(layer2.len(), layer3.len());

    let mut pruned_count = 0;

    for i in 0..layer1.len() {
        let defect = evaluate_a3_constraint(&layer1[i], &layer2[i], &layer3[i]);

        // If the defect is very small, the operations are essentially associative.
        // We consider this a linear/trivial combination and prune the middle layer.
        if defect < associativity_threshold {
            layer2[i] = [0.0; 16];
            pruned_count += 1;
        }
    }

    pruned_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homotopic_pruning() {
        let mut l1 = vec![[1.0; 16], [0.0; 16]];
        let mut l2 = vec![[2.0; 16], [1.0; 16]];
        let mut l3 = vec![[3.0; 16], [0.5; 16]];

        // Inject a known non-associative interaction in the second node
        l1[1][1] = 1.0;
        l1[1][10] = 1.0; // e1 + e10
        l2[1][15] = 1.0;
        l2[1][4] = -1.0; // e15 - e4
        l3[1][8] = 1.0; // e8

        let pruned = prune_by_homotopy(&mut l1, &mut l2, &mut l3, 1e-5);

        // Node 0 is trivial (associative) and should be pruned.
        // Node 1 is non-associative and should be preserved.
        assert_eq!(pruned, 1);
        assert_eq!(l2[0][0], 0.0); // Pruned
        assert_ne!(l2[1][15], 0.0); // Preserved
    }
}
