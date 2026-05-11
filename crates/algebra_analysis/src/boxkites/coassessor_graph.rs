//! Co-assessor graph construction and connected-component analysis.
//!
//! The 42 primitive sedenion assessors form a graph where edges connect
//! pairs whose diagonal zero-divisors multiply to zero for some sign
//! combination (s, t in {+1, -1}). The graph splits into exactly seven
//! connected components, one per missing octonion index 1..=7. Each
//! component is the vertex set of a single box-kite (octahedron).
//!
//! - `primitive_assessors`: enumerates the 42 valid (low, high) pairs.
//! - `diagonal_zero_product`: tests for a zero diagonal product and
//!   returns the witnessing sign pair.
//! - `are_coassessors`: boolean form of the above.
//! - `build_coassessor_graph`: pairwise scan into an adjacency map.
//! - `find_connected_components`: deterministic BFS-based component
//!   extraction sorted by lowest-index member.
//! - `compute_strut_signature`: maps a component to its missing
//!   octonion index (1..=7).

use std::collections::{HashMap, HashSet, VecDeque};

use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};

use super::Assessor;

/// Generate all 42 primitive assessors for sedenions.
///
/// Excludes (i, 8) and (i, i+8) which don't participate in diagonal zero-products.
pub fn primitive_assessors() -> Vec<Assessor> {
    let mut assessors = Vec::with_capacity(42);

    for low in 1..=7 {
        for high in 8..=15 {
            // Exclude (i, 8) - high index = 8
            if high == 8 {
                continue;
            }
            // Exclude (i, i+8) - "identity" pairs
            if high == low + 8 {
                continue;
            }
            assessors.push(Assessor::new(low, high));
        }
    }

    debug_assert_eq!(
        assessors.len(),
        42,
        "Should have exactly 42 primitive assessors"
    );
    assessors
}

/// Check if two assessors have a diagonal zero-product.
///
/// Returns Some((s, t)) if (e_low1 + s*e_high1) * (e_low2 + t*e_high2) = 0,
/// where s, t in {+1, -1}.
pub fn diagonal_zero_product(a: &Assessor, b: &Assessor, atol: f64) -> Option<(i8, i8)> {
    for s in [-1.0, 1.0] {
        for t in [-1.0, 1.0] {
            let v1 = a.diagonal(s);
            let v2 = b.diagonal(t);
            let product = cd_multiply(&v1, &v2);
            let norm = cd_norm_sq(&product).sqrt();
            if norm < atol {
                return Some((s as i8, t as i8));
            }
        }
    }
    None
}

/// Check if two assessors are co-assessors (have any diagonal zero-product).
pub fn are_coassessors(a: &Assessor, b: &Assessor, atol: f64) -> bool {
    diagonal_zero_product(a, b, atol).is_some()
}

/// Build the co-assessor adjacency graph.
///
/// Returns a map from assessor index to set of adjacent assessor indices.
pub fn build_coassessor_graph(assessors: &[Assessor], atol: f64) -> HashMap<usize, HashSet<usize>> {
    let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();

    // Initialize all vertices
    for i in 0..assessors.len() {
        graph.insert(i, HashSet::new());
    }

    // Add edges for co-assessor pairs
    for i in 0..assessors.len() {
        for j in (i + 1)..assessors.len() {
            if are_coassessors(&assessors[i], &assessors[j], atol) {
                graph.get_mut(&i).unwrap().insert(j);
                graph.get_mut(&j).unwrap().insert(i);
            }
        }
    }

    graph
}

/// Find connected components in the co-assessor graph.
///
/// Returns a vector of components, where each component is a vector of assessor indices.
pub fn find_connected_components(graph: &HashMap<usize, HashSet<usize>>) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &start in graph.keys() {
        if visited.contains(&start) {
            continue;
        }

        // BFS to find component
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);
            if let Some(neighbors) = graph.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component.sort();
        components.push(component);
    }

    // Sort components by first element for determinism
    components.sort_by_key(|c| c.first().copied().unwrap_or(0));
    components
}

/// Compute the strut signature for a box-kite.
///
/// The strut signature is the octonion index (1-7) missing from the low indices
/// of the box-kite's assessors.
pub fn compute_strut_signature(assessors: &[Assessor]) -> usize {
    let low_indices: HashSet<usize> = assessors.iter().map(|a| a.low).collect();
    for i in 1..=7 {
        if !low_indices.contains(&i) {
            return i;
        }
    }
    0 // Should never happen for valid box-kites
}
