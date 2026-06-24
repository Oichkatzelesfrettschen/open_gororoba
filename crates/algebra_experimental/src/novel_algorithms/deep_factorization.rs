//! Hyper-dimensional block-factorization (64D/128D)
//!
//! Heuristic search for sparse tensor contraction patterns in 64D/128D CD algebras.
//! Inspired by the Cariow & Cariowa family of fast hypercomplex multipliers.
//!
//! # Evidentiary classification: C (exploratory conjecture)
//!
//! The block density measurement is a heuristic proxy for multiplication
//! savings. It counts XOR-distance between index pairs as a proxy for zero-entry
//! density in the CD sign table. This is not the exact Cariow algorithm.
//!
//! For the verified multiplication count analysis, see:
//! `cd_kernel::cayley_dickson::cariow_factorization` (evid T/C, exact counts).
//!
//! # WHY XOR-distance as a density proxy
//!
//! In the Cayley-Dickson construction, the sign of e_i * e_j is determined by the
//! bit pattern of i XOR j.  Pairs where XOR has many set bits tend to contribute
//! to off-diagonal sign-cancellation blocks. The heuristic `xor.count_ones() > 2`
//! is a rough proxy for "this block has high structural interference." A
//! production factorizer must use direct sign-table analysis before treating
//! these blocks as multiplication savings.
//!
//! # See also
//!
//! - `cd_kernel::cayley_dickson::cariow_factorization` -- exact multiplication counts
//! - `cd_kernel::cayley_dickson::trigintaduonion` -- dim=32 mul_standard vs mul_optimized

/// Represents a sparse block in a hypercomplex multiplication matrix.
#[derive(Debug, Clone)]
pub struct TensorBlock {
    pub row_start: usize,
    pub col_start: usize,
    pub size: usize,
    /// Density of non-zero entries. If 0.0, the block is structurally null.
    pub density: f64,
}

/// Scan a CD multiplication proxy for low-density tensor blocks.
pub fn heuristic_tensor_contraction(dimension: usize, threshold: f64) -> Vec<TensorBlock> {
    assert!(dimension >= 16 && dimension.is_power_of_two());

    let mut blocks = Vec::new();
    let step = 8;

    for i in (0..dimension).step_by(step) {
        for j in (0..dimension).step_by(step) {
            let density = measure_block_density(i, j, step, dimension);
            if density < threshold {
                blocks.push(TensorBlock {
                    row_start: i,
                    col_start: j,
                    size: step,
                    density,
                });
            }
        }
    }

    // Sort blocks to find the largest contiguous zero-regions for SIMD masking
    blocks.sort_by(|a, b| a.density.partial_cmp(&b.density).unwrap());
    blocks
}

/// Measure a proxy for non-associative and non-commutative interference in a block.
fn measure_block_density(row: usize, col: usize, size: usize, _dim: usize) -> f64 {
    let mut non_zero_count = 0;
    for r in row..(row + size) {
        for c in col..(col + size) {
            let xor_val = r ^ c;
            if xor_val.count_ones() > 2 {
                non_zero_count += 1;
            }
        }
    }

    non_zero_count as f64 / (size * size) as f64
}

/// Compatibility spelling for older experimental callers.
pub fn heurstic_tensor_contraction(dimension: usize, threshold: f64) -> Vec<TensorBlock> {
    heuristic_tensor_contraction(dimension, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_64d_factorization_discovery() {
        let blocks = heuristic_tensor_contraction(64, 0.5);
        assert!(!blocks.is_empty());
        assert!(blocks[0].density < 0.5);
    }
}
