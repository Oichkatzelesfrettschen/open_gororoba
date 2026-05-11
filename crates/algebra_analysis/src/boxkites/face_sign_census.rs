//! Triangular-face sign-pattern classification and per-component census.
//!
//! At dim=16 (sedenions), the census is 42 TwoSameOneOpp + 14 AllOpposite
//! (C-479). This subsystem generalizes to any power-of-two CD dimension by
//! walking the zero-divisor graph motif components and classifying each
//! triangular face's three edge signs.
//!
//! - `FaceSignPattern`: order-independent enum (AllSame, TwoSameOneOpp,
//!   OneSameTwoOpp, AllOpposite).
//! - `classify_face_pattern`: counts the number of Same edges in a
//!   3-tuple of EdgeSignType and returns the pattern.
//! - `edge_sign_type_exact`: integer-exact edge-sign classifier for a
//!   single cross-assessor pair at any dimension.
//! - `ComponentFaceCensus`: per-component breakdown (n_nodes, n_edges,
//!   n_triangles, pattern_counts).
//! - `GenericFaceSignCensus`: aggregate census across all components.
//! - `generic_face_sign_census`: builds the full census by iterating
//!   motif components and triangle-classifying each.

use std::collections::{HashMap, HashSet};

use super::{CrossPair, EdgeSignType, diagonal_zero_products_exact};

/// Normalized face sign pattern (order-independent classification of a
/// triangular face's three edge signs).
///
/// At dim=16 (sedenions), the census is 42 TwoSameOneOpp + 14 AllOpposite
/// (C-479). This enum supports census computation at any dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceSignPattern {
    /// All 3 edges Same-sign (Blues).
    AllSame,
    /// 2 Same + 1 Opposite (trefoil variant I).
    TwoSameOneOpp,
    /// 1 Same + 2 Opposite (trefoil variant II).
    OneSameTwoOpp,
    /// All 3 edges Opposite-sign (triple-zigzag).
    AllOpposite,
}

/// Classify a triangular face by counting how many of its three edges
/// are Same vs Opposite sign type.
pub fn classify_face_pattern(signs: &[EdgeSignType; 3]) -> FaceSignPattern {
    let n_same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
    match n_same {
        3 => FaceSignPattern::AllSame,
        2 => FaceSignPattern::TwoSameOneOpp,
        1 => FaceSignPattern::OneSameTwoOpp,
        0 => FaceSignPattern::AllOpposite,
        _ => unreachable!("triangle has exactly 3 edges, n_same is 0..=3"),
    }
}

/// Integer-exact edge sign classification for cross-assessor pairs at any dimension.
///
/// Returns `Same` if solutions include (1,1) or (-1,-1), `Opposite` otherwise.
/// Panics if the pair has no diagonal zero-products (not co-assessors).
pub fn edge_sign_type_exact(dim: usize, a: CrossPair, b: CrossPair) -> EdgeSignType {
    let sols = diagonal_zero_products_exact(dim, a, b);
    assert!(
        !sols.is_empty(),
        "No diagonal zero-products for {:?}--{:?} at dim={}",
        a,
        b,
        dim
    );
    if sols.contains(&(1, 1)) || sols.contains(&(-1, -1)) {
        EdgeSignType::Same
    } else {
        EdgeSignType::Opposite
    }
}

/// Per-component face sign census result.
#[derive(Debug, Clone)]
pub struct ComponentFaceCensus {
    /// Component index in the motif component list.
    pub component_idx: usize,
    /// Number of nodes in this component.
    pub n_nodes: usize,
    /// Number of edges in this component.
    pub n_edges: usize,
    /// Triangular faces found (as node triples, sorted).
    pub n_triangles: usize,
    /// Count of each face sign pattern.
    pub pattern_counts: HashMap<FaceSignPattern, usize>,
}

/// Complete face sign census across all motif components at a given dimension.
#[derive(Debug, Clone)]
pub struct GenericFaceSignCensus {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of motif components.
    pub n_components: usize,
    /// Total triangular faces across all components.
    pub total_triangles: usize,
    /// Aggregate pattern counts across all components.
    pub total_pattern_counts: HashMap<FaceSignPattern, usize>,
    /// Per-component breakdown.
    pub per_component: Vec<ComponentFaceCensus>,
    /// Whether all components with triangles have the same pattern distribution.
    pub uniform_across_components: bool,
}

/// Compute the face sign census for all motif components at a given CD dimension.
///
/// For each connected component of the zero-divisor graph, finds all triangular
/// faces (3-cliques), classifies each face's three edge signs as Same or Opposite,
/// and aggregates the face sign pattern distribution.
///
/// At dim=16 this reproduces C-479 (42 TwoSameOneOpp + 14 AllOpposite).
pub fn generic_face_sign_census(dim: usize) -> GenericFaceSignCensus {
    let components = super::motif_components_for_cross_assessors(dim);
    let mut per_component = Vec::new();
    let mut total_counts: HashMap<FaceSignPattern, usize> = HashMap::new();
    let mut total_triangles = 0usize;

    // Track distribution of first non-trivial component for uniformity check
    let mut first_dist: Option<HashMap<FaceSignPattern, usize>> = None;
    let mut uniform = true;

    for (comp_idx, comp) in components.iter().enumerate() {
        // Build adjacency set for fast triangle detection
        let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
        let adj: HashSet<(CrossPair, CrossPair)> = comp
            .edges
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();

        // Find all triangles (3-cliques): for each edge (u,v), find common neighbors w > v
        let mut triangles: Vec<[CrossPair; 3]> = Vec::new();
        for &(u, v) in &comp.edges {
            for &w in &nodes {
                if w <= v {
                    continue;
                }
                if adj.contains(&(u, w)) && adj.contains(&(v, w)) {
                    triangles.push([u, v, w]);
                }
            }
        }

        // Classify each triangle's edge signs
        let mut pattern_counts: HashMap<FaceSignPattern, usize> = HashMap::new();
        for tri in &triangles {
            let signs = [
                edge_sign_type_exact(dim, tri[0], tri[1]),
                edge_sign_type_exact(dim, tri[1], tri[2]),
                edge_sign_type_exact(dim, tri[0], tri[2]),
            ];
            let pattern = classify_face_pattern(&signs);
            *pattern_counts.entry(pattern).or_insert(0) += 1;
            *total_counts.entry(pattern).or_insert(0) += 1;
        }

        total_triangles += triangles.len();

        // Uniformity check: compare to first non-empty distribution
        if !triangles.is_empty() {
            if let Some(ref first) = first_dist {
                if &pattern_counts != first {
                    uniform = false;
                }
            } else {
                first_dist = Some(pattern_counts.clone());
            }
        }

        per_component.push(ComponentFaceCensus {
            component_idx: comp_idx,
            n_nodes: comp.nodes.len(),
            n_edges: comp.edges.len(),
            n_triangles: triangles.len(),
            pattern_counts,
        });
    }

    GenericFaceSignCensus {
        dim,
        n_components: components.len(),
        total_triangles,
        total_pattern_counts: total_counts,
        per_component,
        uniform_across_components: uniform,
    }
}
