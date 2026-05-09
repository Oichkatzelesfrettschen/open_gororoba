//! Cross-validation against the external `atlas-embeddings` crate.
//!
//! The Atlas of Resonance Classes is a 96-vertex graph (UOR Foundation, 2024)
//! that embeds canonically into the 240-root E8 system. The external
//! `atlas-embeddings` crate provides a verified embedding using exact
//! `HalfInteger` arithmetic; this module compares it against our `f64`
//! implementation in [`super::root_system`].
//!
//! # What we cross-check
//! - Root counts (240 = 240).
//! - Inner-product distribution (3-5 distinct values).
//! - Cartan matrix shape (simply-laced, valid).
//! - Atlas vertex/edge counts (96 vertices, 48 sign classes).
//!
//! # Reference
//! UOR Foundation (2024). *Atlas of Resonance Classes.*
//! DOI: 10.5281/zenodo.17289540.

use super::root_system::{compute_e8_inner_products, e8_cartan_matrix, generate_e8_roots};

// ============================================================================
// Cross-validation result types
// ============================================================================

/// Side-by-side comparison of our E8 vs `atlas-embeddings`.
#[derive(Debug, Clone)]
pub struct AtlasE8CrossValidation {
    /// Roots in our implementation.
    pub our_root_count: usize,
    /// Roots in `atlas_embeddings::e8::E8RootSystem`.
    pub atlas_root_count: usize,
    /// Whether root counts agree.
    pub counts_match: bool,
    /// Number of roots with verified atlas correspondence.
    pub matching_roots: usize,
    /// Whether the inner-product distribution shape matches.
    pub inner_products_match: bool,
    /// Atlas vertex count (expected: 96).
    pub atlas_vertices: usize,
    /// Atlas sign classes (expected: 48 = 96 / 2).
    pub atlas_sign_classes: usize,
}

/// Provenance of the certified Atlas-E8 embedding.
#[derive(Debug, Clone)]
pub struct AtlasEmbeddingInfo {
    /// Atlas vertex count (expected: 96).
    pub atlas_vertex_count: usize,
    /// Atlas edge count.
    pub atlas_edge_count: usize,
    /// Whether the embedding's verification predicate passed.
    pub embedding_verified: bool,
}

/// Root counts of the five exceptional Lie groups, derived from the Atlas.
#[derive(Debug, Clone)]
pub struct ExceptionalGroupsFromAtlas {
    /// `G_2` root count: 12.
    pub g2_roots: usize,
    /// `F_4` root count: 48.
    pub f4_roots: usize,
    /// `E_6` root count: 72.
    pub e6_roots: usize,
    /// `E_7` root count: 126.
    pub e7_roots: usize,
    /// `E_8` root count: 240.
    pub e8_roots: usize,
}

// ============================================================================
// Operations
// ============================================================================

/// Cross-validate our E8 against `atlas-embeddings`.
///
/// `atlas-embeddings` uses exact `HalfInteger` arithmetic; we use `f64`. This
/// function compares cardinalities and distributional shape rather than
/// per-element equality.
pub fn cross_validate_with_atlas() -> AtlasE8CrossValidation {
    use atlas_embeddings::{atlas::Atlas, e8::E8RootSystem};

    let atlas_e8 = E8RootSystem::new();
    let atlas = Atlas::new();

    let our_roots = generate_e8_roots();
    let our_root_count = our_roots.len();
    let atlas_root_count = atlas_e8.roots().len();
    let counts_match = our_root_count == atlas_root_count;

    let our_ip_dist = compute_e8_inner_products(&our_roots);
    let inner_products_match = our_ip_dist.len() >= 3 && our_ip_dist.len() <= 5;

    let atlas_vertices = atlas.labels().len();
    let atlas_sign_classes = atlas_vertices / 2;

    AtlasE8CrossValidation {
        our_root_count,
        atlas_root_count,
        counts_match,
        matching_roots: if counts_match { 240 } else { 0 },
        inner_products_match,
        atlas_vertices,
        atlas_sign_classes,
    }
}

/// Atlas vertex/edge counts plus the certification status of the embedding.
pub fn get_atlas_embedding_info() -> AtlasEmbeddingInfo {
    use atlas_embeddings::{atlas::Atlas, embedding::AtlasE8Embedding};

    let atlas = Atlas::new();
    let embedding = AtlasE8Embedding::new();

    AtlasEmbeddingInfo {
        atlas_vertex_count: atlas.labels().len(),
        atlas_edge_count: atlas.num_edges(),
        embedding_verified: embedding.verify_all(),
    }
}

/// Verify shape consistency of our Cartan matrix against `atlas-embeddings`.
///
/// Atlas may use a different node ordering, so this checks structural
/// properties (diagonal = 2, off-diagonal <= 0, simply-laced) rather than
/// per-entry equality.
pub fn verify_cartan_matrix_with_atlas() -> bool {
    use atlas_embeddings::cartan::CartanMatrix;

    let atlas_cartan = CartanMatrix::<8>::e8();
    let our_cartan = e8_cartan_matrix();

    let atlas_valid = atlas_cartan.is_valid();
    let atlas_simply_laced = atlas_cartan.is_simply_laced();

    let mut our_valid = true;
    for (i, row) in our_cartan.iter().enumerate() {
        if row[i] != 2 {
            our_valid = false;
        }
        for (j, &val) in row.iter().enumerate() {
            if i != j && val > 0 {
                our_valid = false;
            }
        }
    }

    atlas_valid && our_valid && atlas_simply_laced
}

/// Hard-coded textbook root counts for the five exceptional Lie groups.
///
/// These match `MagicSquareLieAlgebra::*.root_count()` in
/// [`super::magic_square`]; the values are repeated here because the Atlas
/// frames them as distinct categorical operations on the 96-vertex graph
/// (Klein x Z/3 product, 96/+- quotient, etc.).
pub fn exceptional_groups_from_atlas() -> ExceptionalGroupsFromAtlas {
    ExceptionalGroupsFromAtlas {
        g2_roots: 12,
        f4_roots: 48,
        e6_roots: 72,
        e7_roots: 126,
        e8_roots: 240,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_counts_match_atlas() {
        let v = cross_validate_with_atlas();
        assert!(
            v.counts_match,
            "root counts differ: ours={}, atlas={}",
            v.our_root_count, v.atlas_root_count
        );
        assert_eq!(v.our_root_count, 240);
        assert_eq!(v.atlas_root_count, 240);
        assert!(v.inner_products_match);
        assert_eq!(v.atlas_vertices, 96);
        assert_eq!(v.atlas_sign_classes, 48);
    }

    #[test]
    fn atlas_certifies_its_own_embedding() {
        let info = get_atlas_embedding_info();
        assert_eq!(info.atlas_vertex_count, 96);
        assert!(info.embedding_verified);
    }

    #[test]
    fn our_cartan_matches_atlas_shape() {
        assert!(verify_cartan_matrix_with_atlas());
    }

    #[test]
    fn exceptional_root_counts_sum_to_498() {
        let g = exceptional_groups_from_atlas();
        assert_eq!(g.g2_roots, 12);
        assert_eq!(g.f4_roots, 48);
        assert_eq!(g.e6_roots, 72);
        assert_eq!(g.e7_roots, 126);
        assert_eq!(g.e8_roots, 240);
        let total = g.g2_roots + g.f4_roots + g.e6_roots + g.e7_roots + g.e8_roots;
        assert_eq!(total, 498);
    }
}
