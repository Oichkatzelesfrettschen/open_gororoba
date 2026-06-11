//! Layer 2: Elevated Addition.
//!
//! Component-wise addition of lattice vectors (Z^8) plus its
//! interpretation back in the codebook. Two arithmetic regimes:
//!
//!   * **Z-elevated** (`elevated_add`, `elevated_addition_table`):
//!     ordinary integer +; the sum may leave {-1, 0, 1}^8, in which
//!     case the result is `OutOfBounds`. Otherwise the trinary sum
//!     is decoded against the dictionary -- either `InCodebook` or
//!     `OutOfCodebook`.
//!   * **F_3-elevated** (`elevated_add_f3`, `elevated_addition_table_f3`):
//!     sum modulo 3 lifted to {-1, 0, 1}; always stays in the
//!     trinary set so only `InCodebook` / `OutOfCodebook` apply.
//!
//! Public surface re-exported from `codebook` via `pub use`:
//!   * `lattice_add`, `try_narrow_to_lattice`
//!   * `ElevatedResult`, `ElevatedAdditionStats`
//!   * `ElevatedResultF3`, `ElevatedAdditionStatsF3`
//!   * The four `EncodingDictionary` methods are added to the same
//!     struct via a second `impl` block (Rust allows multiple impl
//!     blocks across modules in the same crate).

use super::{
    EncodingDictionary,
    lambda_predicates::LatticeVector,
    lattice_arith::{lattice_add_f3, lattice_diff},
};

// ============================================================================
// Layer 2: Elevated Addition
// ============================================================================

/// Component-wise addition of two lattice vectors in Z^8.
///
/// This is ordinary integer addition. The result may leave the trinary
/// set {-1, 0, 1}^8, which is why elevated addition must check membership
/// before decoding.
pub fn lattice_add(a: &LatticeVector, b: &LatticeVector) -> [i32; 8] {
    let mut result = [0i32; 8];
    for (r, (&x, &y)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
        *r = x as i32 + y as i32;
    }
    result
}

/// Try to narrow a Z^8 vector back to a trinary lattice vector.
/// Returns None if any coordinate is outside [-1, 1].
pub fn try_narrow_to_lattice(v: &[i32; 8]) -> Option<LatticeVector> {
    let mut result = [0i8; 8];
    for (r, &x) in result.iter_mut().zip(v.iter()) {
        if !(-1..=1).contains(&x) {
            return None;
        }
        *r = x as i8;
    }
    Some(result)
}

/// Result of an elevated addition Phi(a) + Phi(b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElevatedResult {
    /// The sum lands in the codebook and decodes to basis element c.
    InCodebook {
        sum_vec: LatticeVector,
        decoded_index: usize,
    },
    /// The sum is a valid trinary vector but not in the codebook.
    OutOfCodebook { sum_vec: LatticeVector },
    /// The sum leaves {-1, 0, 1}^8 entirely (some coordinate exceeds bounds).
    OutOfBounds { sum_vec: [i32; 8] },
}

/// Statistics for the elevated addition table.
#[derive(Debug, Clone)]
pub struct ElevatedAdditionStats {
    /// Total number of ordered pairs (a, b) tested.
    pub total_pairs: usize,
    /// Number of pairs where Phi(a)+Phi(b) decodes to some c in the dictionary.
    pub in_codebook: usize,
    /// Number of pairs where sum is trinary but not in codebook.
    pub out_of_codebook: usize,
    /// Number of pairs where sum leaves {-1,0,1}^8.
    pub out_of_bounds: usize,
    /// Closure rate: in_codebook / total_pairs.
    pub closure_rate: f64,
    /// Whether the operation is commutative (Phi(a)+Phi(b) = Phi(b)+Phi(a) always).
    pub is_commutative: bool,
    /// Number of basis elements b that act as identity (Phi(a)+Phi(b)=Phi(a) for all a).
    pub identity_count: usize,
}

/// Result of an F_3-elevated addition Phi(a) +_3 Phi(b).
///
/// Unlike Z-elevated addition, F_3-addition always stays in {-1,0,1}^8
/// (it wraps around), so there is no OutOfBounds variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElevatedResultF3 {
    /// The F_3-sum lands in the codebook and decodes to basis element c.
    InCodebook {
        sum_vec: LatticeVector,
        decoded_index: usize,
    },
    /// The F_3-sum is a valid trinary vector but not in the codebook.
    OutOfCodebook { sum_vec: LatticeVector },
}

/// Statistics for the F_3-elevated addition table.
#[derive(Debug, Clone)]
pub struct ElevatedAdditionStatsF3 {
    /// Total number of ordered pairs (a, b) tested.
    pub total_pairs: usize,
    /// Number of pairs where Phi(a)+_3 Phi(b) decodes to some c.
    pub in_codebook: usize,
    /// Number of pairs where F_3-sum is trinary but not in codebook.
    pub out_of_codebook: usize,
    /// Closure rate: in_codebook / total_pairs.
    pub closure_rate: f64,
    /// Whether F_3-addition is commutative on this codebook.
    pub is_commutative: bool,
    /// Number of basis elements that act as identity.
    pub identity_count: usize,
    /// Fraction of tested triples where (a+b)+c = a+(b+c) in F_3.
    pub associativity_rate: f64,
    /// Number of triples tested for associativity.
    pub associativity_triples_tested: usize,
}

impl EncodingDictionary {
    /// Perform elevated addition: compute Phi(a) + Phi(b) and try to decode.
    ///
    /// Returns an `ElevatedResult` describing where the sum lands:
    /// - `InCodebook`: the sum is in the dictionary and decodes to basis index c
    /// - `OutOfCodebook`: the sum is trinary but not a codeword
    /// - `OutOfBounds`: the sum has coordinates outside [-1, 1]
    pub fn elevated_add(&self, a: usize, b: usize) -> Option<ElevatedResult> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let sum = lattice_add(lv_a, lv_b);

        match try_narrow_to_lattice(&sum) {
            Some(narrow) => match self.decode(&narrow) {
                Some(c) => Some(ElevatedResult::InCodebook {
                    sum_vec: narrow,
                    decoded_index: c,
                }),
                None => Some(ElevatedResult::OutOfCodebook { sum_vec: narrow }),
            },
            None => Some(ElevatedResult::OutOfBounds { sum_vec: sum }),
        }
    }

    /// Compute the full elevated addition table for all ordered pairs (a, b).
    ///
    /// Returns an n x n table where `table[a][b] = elevated_add(a, b)`.
    /// This table captures the complete lattice-addition structure of the codebook.
    pub fn elevated_addition_table(&self) -> Vec<Vec<ElevatedResult>> {
        let n = self.dim();
        let mut table = Vec::with_capacity(n);
        for a in 0..n {
            let mut row = Vec::with_capacity(n);
            for b in 0..n {
                row.push(
                    self.elevated_add(a, b)
                        .expect("basis indices should be valid"),
                );
            }
            table.push(row);
        }
        table
    }

    /// Compute summary statistics for the elevated addition table.
    pub fn elevated_addition_stats(&self) -> ElevatedAdditionStats {
        let n = self.dim();
        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut out_of_bounds = 0usize;
        let mut is_commutative = true;
        // Build the table for commutativity check
        let table = self.elevated_addition_table();

        for (a, row_a) in table.iter().enumerate() {
            for (b, result) in row_a.iter().enumerate() {
                match result {
                    ElevatedResult::InCodebook { .. } => in_codebook += 1,
                    ElevatedResult::OutOfCodebook { .. } => out_of_codebook += 1,
                    ElevatedResult::OutOfBounds { .. } => out_of_bounds += 1,
                }
                // Commutativity: check table[a][b] == table[b][a]
                if a < b && *result != table[b][a] {
                    is_commutative = false;
                }
            }
        }

        // Identity check: b is identity if Phi(a)+Phi(b) decodes to a for all a
        let identity_count = (0..n)
            .filter(|&b| {
                table.iter().enumerate().all(|(a, row)| {
                    matches!(&row[b],
                        ElevatedResult::InCodebook { decoded_index, .. }
                        if *decoded_index == a
                    )
                })
            })
            .count();

        let total_pairs = n * n;
        ElevatedAdditionStats {
            total_pairs,
            in_codebook,
            out_of_codebook,
            out_of_bounds,
            closure_rate: in_codebook as f64 / total_pairs as f64,
            is_commutative,
            identity_count,
        }
    }

    /// For a given basis element b, compute the "translation orbit":
    /// the set of basis elements a for which Phi(a) + Phi(b) is in the codebook.
    pub fn translation_orbit(&self, b: usize) -> Vec<(usize, usize)> {
        let n = self.dim();
        let mut orbit = Vec::new();
        for a in 0..n {
            if let Some(ElevatedResult::InCodebook { decoded_index, .. }) = self.elevated_add(a, b)
            {
                orbit.push((a, decoded_index));
            }
        }
        orbit
    }

    /// Perform F_3-elevated addition: Phi(a) +_3 Phi(b) mod 3, then decode.
    ///
    /// Unlike Z-addition, F_3-addition always produces a trinary vector
    /// (it wraps around: -1+(-1)=1, 1+1=-1). The result is always either
    /// InCodebook or OutOfCodebook, never OutOfBounds.
    pub fn elevated_add_f3(&self, a: usize, b: usize) -> Option<ElevatedResultF3> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let sum = lattice_add_f3(lv_a, lv_b);

        match self.decode(&sum) {
            Some(c) => Some(ElevatedResultF3::InCodebook {
                sum_vec: sum,
                decoded_index: c,
            }),
            None => Some(ElevatedResultF3::OutOfCodebook { sum_vec: sum }),
        }
    }

    /// Compute the F_3-elevated addition table and statistics.
    pub fn elevated_addition_stats_f3(&self) -> ElevatedAdditionStatsF3 {
        let n = self.dim();
        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut is_commutative = true;
        // Precompute all results for commutativity check
        let mut table: Vec<Vec<ElevatedResultF3>> = Vec::with_capacity(n);
        for a in 0..n {
            let mut row = Vec::with_capacity(n);
            for b in 0..n {
                row.push(
                    self.elevated_add_f3(a, b)
                        .expect("basis indices should be valid"),
                );
            }
            table.push(row);
        }

        for (a, row_a) in table.iter().enumerate() {
            for (b, result) in row_a.iter().enumerate() {
                match result {
                    ElevatedResultF3::InCodebook { .. } => in_codebook += 1,
                    ElevatedResultF3::OutOfCodebook { .. } => out_of_codebook += 1,
                }
                if a < b && *result != table[b][a] {
                    is_commutative = false;
                }
            }
        }

        // Identity check in F_3
        let identity_count = (0..n)
            .filter(|&b| {
                table.iter().enumerate().all(|(a, row)| {
                    matches!(&row[b],
                        ElevatedResultF3::InCodebook { decoded_index, .. }
                        if *decoded_index == a
                    )
                })
            })
            .count();

        // Associativity check (sample-based for large dictionaries)
        let mut associative_triples = 0usize;
        let mut total_triples = 0usize;
        let limit = n.min(32); // sample at most 32^3 triples
        for a in 0..limit {
            for b in 0..limit {
                for c in 0..limit {
                    total_triples += 1;
                    // (a + b) + c vs a + (b + c)
                    let ab = lattice_add_f3(self.encode(a).unwrap(), self.encode(b).unwrap());
                    let abc_left = lattice_add_f3(&ab, self.encode(c).unwrap());

                    let bc = lattice_add_f3(self.encode(b).unwrap(), self.encode(c).unwrap());
                    let abc_right = lattice_add_f3(self.encode(a).unwrap(), &bc);

                    if abc_left == abc_right {
                        associative_triples += 1;
                    }
                }
            }
        }

        let total_pairs = n * n;
        ElevatedAdditionStatsF3 {
            total_pairs,
            in_codebook,
            out_of_codebook,
            closure_rate: in_codebook as f64 / total_pairs as f64,
            is_commutative,
            identity_count,
            associativity_rate: associative_triples as f64 / total_triples as f64,
            associativity_triples_tested: total_triples,
        }
    }

    /// Compute the F_3-elevated difference: Phi(a) - Phi(b) in Z, then decode.
    ///
    /// Returns InCodebook if the difference is trinary and in the dictionary,
    /// OutOfCodebook if trinary but not in dictionary, OutOfBounds if
    /// difference leaves {-1,0,1}^8.
    pub fn elevated_diff(&self, a: usize, b: usize) -> Option<ElevatedResult> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let diff = lattice_diff(lv_a, lv_b);

        match try_narrow_to_lattice(&diff) {
            Some(narrow) => match self.decode(&narrow) {
                Some(c) => Some(ElevatedResult::InCodebook {
                    sum_vec: narrow,
                    decoded_index: c,
                }),
                None => Some(ElevatedResult::OutOfCodebook { sum_vec: narrow }),
            },
            None => Some(ElevatedResult::OutOfBounds { sum_vec: diff }),
        }
    }
}
