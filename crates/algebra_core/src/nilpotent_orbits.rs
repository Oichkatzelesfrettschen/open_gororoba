//! Nilpotent orbits: Jordan types, nilpotency index, and classification.
//!
//! Provides tools for analyzing nilpotent matrices and their Jordan structure.
//!
//! # Literature
//!
//! ## Foundational
//! - Collingwood, D.H. & McGovern, W.M. (1993). Nilpotent Orbits in Semisimple
//!   Lie Algebras. Van Nostrand Reinhold.
//! - Jantzen, J.C. (2004). Nilpotent Orbits in Representation Theory.
//!   In: Lie Theory (Progress in Mathematics, vol 228). Birkhauser.
//!
//! ## Computational
//! - de Graaf, W.A. (2000). Lie Algebras: Theory and Algorithms.
//!   North-Holland Mathematical Library, Vol. 56.
//! - GAP System (2023). Package SLA for Lie algebras.
//!
//! # Physics Context in This Repo
//! Nilpotent structures appear in:
//! - **Gauge theory**: Nilpotent orbits classify instanton moduli
//! - **Representation theory**: Jordan decomposition of Lie algebra elements
//! - **Higgs bundles**: Hitchin fibration over nilpotent cone
//!
//! # Key Concepts
//! - Jordan type: partition of n giving block sizes
//! - Nilpotency index: smallest k with A^k = 0
//! - Orbit: conjugacy class under GL(n)

use nalgebra::DMatrix;
use std::fmt;

/// Jordan type (partition) for a nilpotent endomorphism.
///
/// Block sizes are stored in descending order. For example,
/// the partition (3, 2, 2, 1) represents a nilpotent matrix
/// with one 3x3 block, two 2x2 blocks, and one 1x1 block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JordanType {
    /// Block sizes in descending order.
    blocks: Vec<usize>,
}

impl JordanType {
    /// Create a Jordan type from block sizes.
    ///
    /// Blocks are automatically sorted in descending order.
    ///
    /// # Panics
    /// If any block size is zero.
    pub fn new(mut blocks: Vec<usize>) -> Self {
        if blocks.iter().any(|&b| b == 0) {
            panic!("Block sizes must be positive");
        }
        blocks.sort_by(|a, b| b.cmp(a));
        JordanType { blocks }
    }

    /// Create from a partition notation (already descending).
    pub fn from_partition(blocks: Vec<usize>) -> Self {
        JordanType::new(blocks)
    }

    /// Get the block sizes.
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Total dimension (sum of block sizes).
    pub fn dimension(&self) -> usize {
        self.blocks.iter().sum()
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Maximum block size (equals nilpotency index).
    pub fn max_block(&self) -> usize {
        self.blocks.first().copied().unwrap_or(0)
    }

    /// Check if this is a regular nilpotent (single block).
    pub fn is_regular(&self) -> bool {
        self.blocks.len() == 1
    }

    /// Check if this is the zero nilpotent (all 1x1 blocks).
    pub fn is_zero(&self) -> bool {
        self.blocks.iter().all(|&b| b == 1)
    }

    /// Compute the dual partition (transpose of Young diagram).
    pub fn dual(&self) -> JordanType {
        if self.blocks.is_empty() {
            return JordanType { blocks: vec![] };
        }

        let max_size = self.max_block();
        let mut dual_blocks = Vec::with_capacity(max_size);

        for i in 1..=max_size {
            let count = self.blocks.iter().filter(|&&b| b >= i).count();
            if count > 0 {
                dual_blocks.push(count);
            }
        }

        JordanType { blocks: dual_blocks }
    }
}

impl fmt::Display for JordanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &b) in self.blocks.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", b)?;
        }
        write!(f, ")")
    }
}

/// Result of nilpotent matrix analysis.
#[derive(Debug, Clone)]
pub struct NilpotentAnalysis {
    /// Jordan type of the matrix.
    pub jordan_type: JordanType,
    /// Nilpotency index k (smallest k with A^k = 0).
    pub nilpotency_index: usize,
    /// Dimension of the matrix.
    pub dimension: usize,
    /// Sequence of kernel dimensions dim(ker(A^k)) for k = 0, 1, 2, ...
    pub kernel_dims: Vec<usize>,
}

/// Check if a matrix is nilpotent and compute its nilpotency index.
///
/// Returns `Some(k)` where k is the smallest positive integer with A^k = 0,
/// or `None` if the matrix is not nilpotent within n steps.
///
/// # Arguments
/// * `a` - Square matrix to check
/// * `tolerance` - Threshold for considering a matrix as zero
///
/// # Example
/// ```
/// use algebra_core::nilpotent_orbits::nilpotency_index;
/// use nalgebra::DMatrix;
///
/// // Nilpotent matrix [[0, 1], [0, 0]]
/// let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
/// assert_eq!(nilpotency_index(&a, 1e-10), Some(2));
/// ```
pub fn nilpotency_index(a: &DMatrix<f64>, tolerance: f64) -> Option<usize> {
    if a.nrows() != a.ncols() {
        return None;
    }

    let n = a.nrows();
    if n == 0 {
        return Some(1);
    }

    let mut power = a.clone();
    for k in 1..=n {
        if is_zero_matrix(&power, tolerance) {
            return Some(k);
        }
        power = &power * a;
    }

    None
}

/// Check if a matrix is approximately zero.
fn is_zero_matrix(m: &DMatrix<f64>, tolerance: f64) -> bool {
    m.iter().all(|&x| x.abs() < tolerance)
}

/// Compute the kernel dimension of a matrix.
fn kernel_dimension(m: &DMatrix<f64>, tolerance: f64) -> usize {
    let svd = m.clone().svd(false, false);
    let n = m.ncols();

    // Count singular values below tolerance
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s.abs() > tolerance)
        .count();

    n - rank
}

/// Compute the Jordan type of a nilpotent matrix.
///
/// Uses the kernel dimension method:
/// - d_k = dim(ker(A^k)) for k = 0, 1, 2, ...
/// - c_k = d_k - d_{k-1} = number of blocks of size >= k
/// - Number of blocks of exact size k = c_k - c_{k+1}
///
/// # Arguments
/// * `a` - Square nilpotent matrix
/// * `tolerance` - Threshold for numerical comparisons
///
/// # Returns
/// `Some(NilpotentAnalysis)` if the matrix is nilpotent, `None` otherwise.
///
/// # Example
/// ```
/// use algebra_core::nilpotent_orbits::jordan_type_nilpotent;
/// use nalgebra::DMatrix;
///
/// // Jordan block J_3: [[0,1,0],[0,0,1],[0,0,0]]
/// let j3 = DMatrix::from_row_slice(3, 3, &[
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0,
///     0.0, 0.0, 0.0,
/// ]);
/// let analysis = jordan_type_nilpotent(&j3, 1e-10).unwrap();
/// assert_eq!(analysis.jordan_type.blocks(), &[3]);
/// ```
pub fn jordan_type_nilpotent(a: &DMatrix<f64>, tolerance: f64) -> Option<NilpotentAnalysis> {
    if a.nrows() != a.ncols() {
        return None;
    }

    let n = a.nrows();
    if n == 0 {
        return Some(NilpotentAnalysis {
            jordan_type: JordanType { blocks: vec![] },
            nilpotency_index: 1,
            dimension: 0,
            kernel_dims: vec![0],
        });
    }

    // Check nilpotency and get index
    let nil_index = nilpotency_index(a, tolerance)?;

    // Compute kernel dimensions d_k = dim(ker(A^k))
    let mut kernel_dims = vec![0usize]; // d_0 = 0 (ker(I) = {0})
    let mut power = a.clone();

    for _k in 1..=n {
        let dim_ker = kernel_dimension(&power, tolerance);
        kernel_dims.push(dim_ker);

        if dim_ker == n {
            // Full kernel reached, matrix is zero
            break;
        }

        power = &power * a;
    }

    // Pad kernel_dims if needed
    while kernel_dims.len() <= n {
        kernel_dims.push(n);
    }

    // c_k = d_k - d_{k-1}: number of blocks with size >= k
    let mut c = vec![0usize; n + 2];
    for k in 1..=n {
        c[k] = kernel_dims[k].saturating_sub(kernel_dims[k - 1]);
    }

    // Extract block sizes
    let mut blocks = Vec::new();
    for k in 1..=n {
        let exact_count = c[k].saturating_sub(c.get(k + 1).copied().unwrap_or(0));
        for _ in 0..exact_count {
            blocks.push(k);
        }
    }

    blocks.sort_by(|a, b| b.cmp(a));

    // Verify sum equals n
    let block_sum: usize = blocks.iter().sum();
    if block_sum != n {
        // Numerical issues - return best effort
        if blocks.is_empty() && n > 0 {
            blocks = vec![1; n];
        }
    }

    Some(NilpotentAnalysis {
        jordan_type: JordanType { blocks },
        nilpotency_index: nil_index,
        dimension: n,
        kernel_dims,
    })
}

/// Create a standard nilpotent Jordan block of given size.
///
/// Returns the n x n matrix with 1s on the superdiagonal and 0s elsewhere.
pub fn jordan_block(n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n.saturating_sub(1) {
        m[(i, i + 1)] = 1.0;
    }
    m
}

/// Create a nilpotent matrix with given Jordan type.
///
/// Constructs a block-diagonal matrix with Jordan blocks of the specified sizes.
pub fn matrix_from_jordan_type(jt: &JordanType) -> DMatrix<f64> {
    let n = jt.dimension();
    if n == 0 {
        return DMatrix::zeros(0, 0);
    }

    let mut m = DMatrix::zeros(n, n);
    let mut offset = 0;

    for &block_size in jt.blocks() {
        for i in 0..block_size.saturating_sub(1) {
            m[(offset + i, offset + i + 1)] = 1.0;
        }
        offset += block_size;
    }

    m
}

/// Enumerate all partitions of n (Jordan types for nilpotent matrices in gl_n).
pub fn enumerate_partitions(n: usize) -> Vec<JordanType> {
    let mut result = Vec::new();
    enumerate_partitions_rec(n, n, &mut vec![], &mut result);
    result
}

fn enumerate_partitions_rec(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<JordanType>,
) {
    if remaining == 0 {
        result.push(JordanType {
            blocks: current.clone(),
        });
        return;
    }

    for part in (1..=remaining.min(max_part)).rev() {
        current.push(part);
        enumerate_partitions_rec(remaining - part, part, current, result);
        current.pop();
    }
}

/// Number of partitions of n (partition function p(n)).
pub fn partition_count(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    // Dynamic programming
    let mut dp = vec![0usize; n + 1];
    dp[0] = 1;

    for i in 1..=n {
        for j in i..=n {
            dp[j] += dp[j - i];
        }
    }

    dp[n]
}

/// Compare two Jordan types by dominance order.
///
/// Type A dominates B if sum(A[1..k]) >= sum(B[1..k]) for all k.
/// Returns:
/// - Some(Ordering::Greater) if A dominates B strictly
/// - Some(Ordering::Less) if B dominates A strictly
/// - Some(Ordering::Equal) if A == B
/// - None if incomparable
pub fn dominance_order(a: &JordanType, b: &JordanType) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;

    if a == b {
        return Some(Ordering::Equal);
    }

    let n_a = a.dimension();
    let n_b = b.dimension();

    if n_a != n_b {
        return None;
    }

    let mut sum_a = 0usize;
    let mut sum_b = 0usize;
    let mut a_dominates = true;
    let mut b_dominates = true;

    let max_len = a.blocks.len().max(b.blocks.len());

    for k in 0..max_len {
        sum_a += a.blocks.get(k).copied().unwrap_or(0);
        sum_b += b.blocks.get(k).copied().unwrap_or(0);

        if sum_a < sum_b {
            a_dominates = false;
        }
        if sum_b < sum_a {
            b_dominates = false;
        }
    }

    if a_dominates && !b_dominates {
        Some(Ordering::Greater)
    } else if b_dominates && !a_dominates {
        Some(Ordering::Less)
    } else if a_dominates && b_dominates {
        Some(Ordering::Equal)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jordan_type_creation() {
        let jt = JordanType::new(vec![1, 3, 2]);
        assert_eq!(jt.blocks(), &[3, 2, 1]);
        assert_eq!(jt.dimension(), 6);
        assert_eq!(jt.num_blocks(), 3);
        assert_eq!(jt.max_block(), 3);
    }

    #[test]
    fn test_jordan_type_regular() {
        let jt = JordanType::new(vec![5]);
        assert!(jt.is_regular());
        assert!(!jt.is_zero());
    }

    #[test]
    fn test_jordan_type_zero() {
        let jt = JordanType::new(vec![1, 1, 1]);
        assert!(!jt.is_regular());
        assert!(jt.is_zero());
    }

    #[test]
    fn test_jordan_type_dual() {
        // Partition (3, 2, 1) has dual (3, 2, 1) - self-dual
        let jt = JordanType::new(vec![3, 2, 1]);
        let dual = jt.dual();
        assert_eq!(dual.blocks(), &[3, 2, 1]);

        // Partition (4, 1) has dual (2, 1, 1, 1)
        let jt2 = JordanType::new(vec![4, 1]);
        let dual2 = jt2.dual();
        assert_eq!(dual2.blocks(), &[2, 1, 1, 1]);
    }

    #[test]
    fn test_nilpotency_index_basic() {
        // 2x2 nilpotent [[0, 1], [0, 0]]
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(nilpotency_index(&a, 1e-10), Some(2));

        // Zero matrix
        let z = DMatrix::zeros(3, 3);
        assert_eq!(nilpotency_index(&z, 1e-10), Some(1));
    }

    #[test]
    fn test_nilpotency_index_not_nilpotent() {
        // Identity matrix is not nilpotent
        let id = DMatrix::identity(3, 3);
        assert_eq!(nilpotency_index(&id, 1e-10), None);
    }

    #[test]
    fn test_jordan_block() {
        let j3 = jordan_block(3);
        assert_eq!(j3[(0, 1)], 1.0);
        assert_eq!(j3[(1, 2)], 1.0);
        assert_eq!(j3[(0, 0)], 0.0);
        assert_eq!(j3[(2, 2)], 0.0);

        // Should be nilpotent of index 3
        assert_eq!(nilpotency_index(&j3, 1e-10), Some(3));
    }

    #[test]
    fn test_jordan_type_from_matrix() {
        // Single Jordan block of size 4
        let j4 = jordan_block(4);
        let analysis = jordan_type_nilpotent(&j4, 1e-10).unwrap();
        assert_eq!(analysis.jordan_type.blocks(), &[4]);
        assert_eq!(analysis.nilpotency_index, 4);
    }

    #[test]
    fn test_matrix_from_jordan_type() {
        let jt = JordanType::new(vec![3, 2]);
        let m = matrix_from_jordan_type(&jt);

        assert_eq!(m.nrows(), 5);
        assert_eq!(m.ncols(), 5);

        // Recover Jordan type
        let analysis = jordan_type_nilpotent(&m, 1e-10).unwrap();
        assert_eq!(analysis.jordan_type.blocks(), &[3, 2]);
    }

    #[test]
    fn test_enumerate_partitions() {
        // p(4) = 5: (4), (3,1), (2,2), (2,1,1), (1,1,1,1)
        let parts = enumerate_partitions(4);
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn test_partition_count() {
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
        assert_eq!(partition_count(2), 2);
        assert_eq!(partition_count(3), 3);
        assert_eq!(partition_count(4), 5);
        assert_eq!(partition_count(5), 7);
        assert_eq!(partition_count(10), 42);
    }

    #[test]
    fn test_dominance_order() {
        use std::cmp::Ordering;

        let a = JordanType::new(vec![3, 1, 1]);
        let b = JordanType::new(vec![2, 2, 1]);

        // (3,1,1) dominates (2,2,1): 3 >= 2, 3+1=4 >= 2+2=4, 3+1+1=5 = 2+2+1=5
        let ord = dominance_order(&a, &b);
        assert_eq!(ord, Some(Ordering::Greater));

        // Equal
        let c = JordanType::new(vec![2, 2, 1]);
        assert_eq!(dominance_order(&b, &c), Some(Ordering::Equal));
    }

    #[test]
    fn test_display() {
        let jt = JordanType::new(vec![3, 2, 1]);
        assert_eq!(format!("{}", jt), "(3, 2, 1)");
    }
}
