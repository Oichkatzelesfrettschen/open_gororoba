//! Embedding of Moonshine coefficients into Cayley-Dickson tower dimensions.
//!
//! Tests whether the CD algebraic structure has natural "slots" for Monster
//! irreducible representations. The CD tower dimensions span 2^4 through 2^12:
//!   16 (Sedenion), 32 (Pathion), 64 (Chingon), 128 (Routon),
//!   256 (Voudon), 512 (Eriston), 1024 (DekaVoudon),
//!   2048 (Icosikaivoudon), 4096 (CD-4096)
//!
//! The AVT (Alternativity Violation Tensor) at each dimension has a
//! characteristic number of violations. If Monster irrep dimensions
//! divide cleanly into AVT violation counts or CD basis dimensions,
//! the embedding is "resonant".
//!
//! See BIB-0307 (Moreno 1998), BIB-0308 (Cawagas 2004) for CD structure
//! and BIB-0311 (Conway & Norton 1979), BIB-0312 (Borcherds 1992) for Moonshine.

use super::moonshine::{J_COEFFICIENTS, MONSTER_REP_DIMENSIONS};
pub use super::{
    cd_tower_violations::{
        BOX_KITE_VIOLATION_COUNTS, CD_TOWER, CD_VIOLATION_COUNTS, EXACT_ALTERNATOR_COUNTS,
        box_kite_violation_count, enumerate_violation_count, enumerate_violation_count_fast,
        enumerate_violation_count_tiled, estimate_violation_count, exact_violation_count,
        violation_count_best,
    },
    leech_pathion::{
        CrossSectorResult, DARK_SECTOR_DIM, DarkSectorSummary, LEECH_DIM, LeechCdEmbedding,
        LeechPathionEmbedding, PATHION_DIM, sweep_leech_tower,
    },
};

/// Result of embedding a single Monster irrep into a CD dimension.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// CD tower dimension.
    pub cd_dim: usize,
    /// CD algebra name.
    pub cd_name: String,
    /// Monster irrep index (0-based).
    pub rep_index: usize,
    /// Monster irrep dimension.
    pub rep_dim: u64,
    /// How many copies of the irrep fit in the CD dimension.
    pub copies: u64,
    /// Remainder after embedding.
    pub remainder: u64,
    /// Whether the embedding is exact (remainder = 0).
    pub exact: bool,
    /// Ratio: rep_dim / cd_dim (measures how "large" the irrep is relative to CD).
    pub size_ratio: f64,
}

/// Result of the full embedding analysis.
#[derive(Debug, Clone)]
pub struct EmbeddingAnalysis {
    /// Per-rep, per-CD-dim embedding results.
    pub results: Vec<EmbeddingResult>,
    /// Number of exact embeddings found.
    pub num_exact: usize,
    /// CD dimensions that can host at least one Monster irrep exactly.
    pub resonant_dims: Vec<usize>,
}

/// Analyze how Monster irreps embed into CD tower dimensions.
///
/// For each (rep, cd_dim) pair, computes:
/// - How many full copies fit (rep_dim / cd_dim or cd_dim / rep_dim)
/// - Whether the embedding is exact (divides evenly)
/// - The size ratio
pub fn analyze_embeddings() -> EmbeddingAnalysis {
    let reps: Vec<(usize, u64)> = MONSTER_REP_DIMENSIONS
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, d)| *d > 0)
        .collect();

    let mut results = Vec::new();
    let mut resonant_dims = Vec::new();

    for &(cd_dim, cd_name) in &CD_TOWER {
        let mut has_exact = false;

        for &(rep_idx, rep_dim) in &reps {
            let cd_u64 = cd_dim as u64;

            // Check both directions: rep into CD space, or CD into rep space
            let (copies, remainder, exact) = if rep_dim <= cd_u64 {
                let c = cd_u64 / rep_dim;
                let r = cd_u64 % rep_dim;
                (c, r, r == 0)
            } else {
                let c = rep_dim / cd_u64;
                let r = rep_dim % cd_u64;
                (c, r, r == 0)
            };

            if exact {
                has_exact = true;
            }

            results.push(EmbeddingResult {
                cd_dim,
                cd_name: cd_name.to_string(),
                rep_index: rep_idx,
                rep_dim,
                copies,
                remainder,
                exact,
                size_ratio: rep_dim as f64 / cd_dim as f64,
            });
        }

        if has_exact {
            resonant_dims.push(cd_dim);
        }
    }

    let num_exact = results.iter().filter(|r| r.exact).count();

    EmbeddingAnalysis {
        results,
        num_exact,
        resonant_dims,
    }
}

/// Check whether j-coefficients have special relationships with CD dimensions.
///
/// Tests divisibility: c_n mod cd_dim = 0 would indicate a natural embedding.
pub fn j_coefficient_cd_divisibility() -> Vec<(usize, usize, u64, bool)> {
    let mut results = Vec::new();

    for (i, &coeff) in J_COEFFICIENTS.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        for &(cd_dim, _) in &CD_TOWER {
            let divisible = coeff % cd_dim as u64 == 0;
            results.push((i + 1, cd_dim, coeff, divisible));
        }
    }

    results
}

/// Count how many j-coefficients are divisible by each CD dimension.
pub fn divisibility_census() -> Vec<(usize, &'static str, usize, usize)> {
    let nonzero_count = J_COEFFICIENTS.iter().filter(|&&c| c > 0).count();

    CD_TOWER
        .iter()
        .map(|&(cd_dim, cd_name)| {
            let div_count = J_COEFFICIENTS
                .iter()
                .filter(|&&c| c > 0 && c % cd_dim as u64 == 0)
                .count();
            (cd_dim, cd_name, div_count, nonzero_count)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cd_tower_dimensions_are_powers_of_two() {
        for &(dim, _) in &CD_TOWER {
            assert!(dim.is_power_of_two(), "{} is not a power of 2", dim);
        }
        // Verify full 2^4..2^12 coverage
        assert_eq!(CD_TOWER.first().unwrap().0, 16);
        assert_eq!(CD_TOWER.last().unwrap().0, 4096);
        assert_eq!(CD_TOWER.len(), 9);
    }

    #[test]
    fn test_embedding_analysis_runs() {
        let analysis = analyze_embeddings();
        assert!(!analysis.results.is_empty());
    }

    #[test]
    fn test_trivial_rep_embeds_everywhere() {
        let analysis = analyze_embeddings();
        // Trivial rep (dim 1) divides every CD dimension exactly
        let trivial_exact: Vec<_> = analysis
            .results
            .iter()
            .filter(|r| r.rep_dim == 1 && r.exact)
            .collect();
        assert_eq!(
            trivial_exact.len(),
            CD_TOWER.len(),
            "Trivial rep should embed exactly in all CD dimensions"
        );
    }

    #[test]
    fn test_c1_divisibility() {
        // c_1 = 196884. Check which CD dimensions divide it.
        let c1 = 196884u64;
        // 196884 / 4 = 49221, so 4 divides it
        assert_eq!(c1 % 4, 0);
        // 196884 / 16 = 12305.25, so 16 does NOT divide it
        assert_ne!(c1 % 16, 0);
    }

    #[test]
    fn test_divisibility_census_counts() {
        let census = divisibility_census();
        assert_eq!(census.len(), CD_TOWER.len());
        for &(_, _, div_count, total) in &census {
            assert!(div_count <= total);
        }
    }
}
