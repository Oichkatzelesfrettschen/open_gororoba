//! SIMD ecosystem evaluation: wide vs pulp vs portable-simd.
//!
//! Current state: we use `wide` crate for f32x8 SIMD operations.
//! This module documents the evaluation and provides the migration path.
//!
//! # Comparison Matrix
//!
//! | Feature             | wide (current) | pulp           | portable-simd   |
//! |---------------------|---------------|----------------|-----------------|
//! | Stability           | Stable Rust   | Stable Rust    | Nightly only    |
//! | Runtime CPUID       | No (manual)   | Yes (built-in) | No              |
//! | f32x8 type          | Yes           | Via dispatch    | Yes             |
//! | cmp_gt + move_mask  | Yes           | Via dispatch    | Yes             |
//! | Downloads           | 16M           | 8.5M           | N/A (nightly)   |
//! | Used by             | Many crates   | faer (fastest)  | stdlib          |
//! | Our usage sites     | simd_codebook | Would replace   | Future          |
//!
//! # Assessment
//!
//! **pulp** advantages:
//! - Built-in runtime CPUID detection (replaces our dispatch.rs)
//! - Powers faer (fastest pure-Rust linear algebra, outperforms OpenBLAS)
//! - Cleaner API for multi-architecture dispatch
//! - Battle-tested at 8.5M downloads
//!
//! **wide** advantages:
//! - Already integrated and working
//! - Simpler API for basic SIMD types (f32x8, CmpGt trait)
//! - Zero migration cost
//! - Sufficient for our codebook broadcast-compare pattern
//!
//! # Recommendation
//!
//! `wide` is retained. The SIMD usage is limited to:
//! 1. simd_codebook.rs: f32x8 broadcast-compare-popcount (7 boundaries)
//! 2. Potential WHT butterfly SIMD (not yet implemented)
//!
//! pulp's advantage (runtime CPUID) is already handled by our dispatch.rs.
//! Migration to pulp would be warranted if:
//! - We need NEON/SVE support (ARM targets)
//! - We implement SIMD-level WHT butterfly (pulp's dispatch model is better)
//! - faer integration requires pulp types
//!
//! The migration path is straightforward: replace `wide::f32x8` with
//! pulp's `f32x8` type via a feature flag, keeping wide as default.

/// Document the SIMD usage sites that would need migration.
pub fn simd_usage_inventory() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "simd_codebook.rs",
            "f32x8 broadcast-compare-popcount",
            "wide::f32x8 + CmpGt",
        ),
        (
            "dispatch.rs",
            "runtime CPUID detection",
            "std::arch::is_x86_feature_detected!",
        ),
        (
            "rotation.rs",
            "WHT butterfly (scalar currently)",
            "could benefit from SIMD",
        ),
        ("sign_pack.rs", "u64 popcount", "native u64::count_ones()"),
        (
            "fixed_point.rs",
            "i32/i64 multiply-accumulate",
            "native integer ops",
        ),
    ]
}

/// Evaluate whether SIMD upgrade would help for each site.
pub fn migration_benefit_assessment() -> Vec<(&'static str, &'static str, bool)> {
    vec![
        ("simd_codebook", "Already 2.4x via wide f32x8", false),
        (
            "WHT butterfly",
            "Would benefit from 4-wide f64x4 butterfly",
            true,
        ),
        ("dot product", "Could parallelize S@query projection", true),
        ("sign inner product", "Already optimal via popcount", false),
        ("fixed-point MAC", "Integer MAC is already efficient", false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_inventory() {
        let sites = simd_usage_inventory();
        assert!(sites.len() >= 5);
        // Verify all expected sites are listed
        assert!(sites.iter().any(|(f, _, _)| *f == "simd_codebook.rs"));
        assert!(sites.iter().any(|(f, _, _)| *f == "dispatch.rs"));
    }

    #[test]
    fn test_migration_assessment() {
        let assess = migration_benefit_assessment();
        // Count how many sites would benefit from migration
        let beneficial: Vec<_> = assess.iter().filter(|(_, _, b)| *b).collect();
        println!("SIMD migration benefit:");
        for (site, note, benefit) in &assess {
            println!(
                "  {:>20}: {} [{}]",
                site,
                note,
                if *benefit { "BENEFIT" } else { "no change" }
            );
        }
        println!(
            "Sites that would benefit: {}/{}",
            beneficial.len(),
            assess.len()
        );
        // At least some sites should benefit, but not all
        assert!(!beneficial.is_empty() && beneficial.len() < assess.len());
    }
}
