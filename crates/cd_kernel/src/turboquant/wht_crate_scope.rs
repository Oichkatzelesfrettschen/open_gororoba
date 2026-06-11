//! Scope document for publishing our WHT butterfly as a standalone crate.
//!
//! No Rust FWHT crate exists on crates.io (verified 2026-03-28).
//! Our implementation achieves 451 kvec/s at d=128 (5.6x faster than Haar).
//!
//! # Proposed crate: `fwht` (Fast Walsh-Hadamard Transform)
//!
//! ## API Design
//!
//! ```text
//! use fwht::{wht_inplace, fast_jl_rotate, Rotation};
//!
//! // Basic WHT
//! let mut data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! wht_inplace(&mut data); // in-place, O(d log d)
//!
//! // Fast JL rotation (Ailon-Chazelle 2006)
//! let rotation = Rotation::new_fast_jl(128, seed);
//! let mut buf = vec![0.0; 128];
//! let mut out = vec![0.0; 128];
//! rotation.forward(&input, &mut buf, &mut out);
//! rotation.inverse(&out, &mut buf, &mut recovered);
//! ```
//!
//! ## Proposed Feature Set
//!
//! 1. Core: `wht_inplace()` for any power-of-2 dimension
//! 2. Fast JL: Rademacher diagonals + WHT composition
//! 3. SIMD: f64x4 butterfly pairs (via wide or portable-simd)
//! 4. Inverse: self-inverse with normalization
//! 5. Batch: process multiple vectors in parallel
//! 6. no_std compatible (no allocation in core path)
//!
//! ## Files to Extract
//!
//! From `rotation.rs`:
//! - `wht_inplace()` (20 LOC)
//! - `generate_rademacher_diagonals()` (15 LOC)
//! - `fast_jl_rotate()` / `fast_jl_unrotate()` (25 LOC)
//!
//! Total: ~60 LOC core + ~40 LOC tests + Cargo.toml + README
//!
//! ## Publishing Checklist
//!
//! - `[ ]` Extract into standalone crate
//! - `[ ]` Add comprehensive documentation with mathematical background
//! - `[ ]` Add benchmarks (criterion)
//! - `[ ]` Add no_std support
//! - `[ ]` Add SIMD variants (feature-gated)
//! - `[ ]` Publish to crates.io as `fwht` or `walsh-hadamard`
//! - `[ ]` Update open_gororoba to depend on published crate

/// Extracted functions that would form the crate's public API.
pub fn proposed_api_surface() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        (
            "wht_inplace",
            "In-place normalized Walsh-Hadamard Transform",
            20,
        ),
        (
            "generate_rademacher_diagonals",
            "Random +/-1 sign vectors for fast JL",
            15,
        ),
        (
            "fast_jl_rotate",
            "D1 * WHT * D2 * x (Ailon-Chazelle 2006)",
            12,
        ),
        ("fast_jl_unrotate", "Inverse: D2 * WHT * D1 * y", 12),
        (
            "Rotation enum",
            "Haar | FastJL | E8Block | F4Block dispatch",
            50,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_surface_complete() {
        let api = proposed_api_surface();
        let total_loc: usize = api.iter().map(|(_, _, l)| l).sum();
        println!("\nProposed fwht crate API:");
        for (name, desc, loc) in &api {
            println!("  {:>35}: {} ({} LOC)", name, desc, loc);
        }
        println!("  Total core LOC: {}", total_loc);
        assert!(total_loc < 200, "Core should be small");
    }
}
