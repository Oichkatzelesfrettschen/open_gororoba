//! Cayley-Dickson basis product sign table: the primitive truth of the algebra.
//!
//! # Prerequisite modules: none (this is the foundation)
//! # Depended on by: [`arith`], [`signature`], [`sedenion`]
//!
//! # Why this module matters
//!
//! Every CD algebra is completely determined by its sign table: the function
//! `cd_basis_mul_sign(dim, p, q)` that returns +1 or -1 for `e_p * e_q = sign * e_{p^q}`.
//! The product index `p ^ q` (XOR) is a consequence of the doubling construction --
//! it encodes which basis element results from the multiplication.
//!
//! The sign is computed recursively through the tower: quaternion signs determine
//! octonion signs, which determine sedenion signs, and so on. This recursion
//! mirrors the Cayley-Dickson doubling itself. At each level, the sign depends
//! on which "half" of the algebra the operands live in (p < half vs p >= half).
//!
//! # Key insight
//!
//! The sign table is NOT arbitrary. It satisfies constraints imposed by the
//! doubling formula: `(a,b)(c,d) = (ac - conj(d)*b, d*a + b*conj(c))`.
//! The four sub-products each contribute different sign patterns. The recursive
//! `cd_basis_mul_sign` function implements exactly these constraints.
//!
//! For dim <= 8 (octonions), the algebra is alternative and the sign table
//! satisfies the Moufang identities. For dim >= 16, zero divisors appear
//! and the sign table encodes non-trivial algebraic defects.

/// Compute the sign in the Cayley-Dickson basis product: `e_p * e_q = sign * e_{p^q}`.
pub fn cd_basis_mul_sign(dim: usize, p: usize, q: usize) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert!(p < dim && q < dim);

    if dim == 1 {
        return 1;
    }

    let half = dim / 2;
    if p < half && q < half {
        return cd_basis_mul_sign(half, p, q);
    }
    if p < half && q >= half {
        return cd_basis_mul_sign(half, q - half, p);
    }
    if p >= half && q < half {
        let s = cd_basis_mul_sign(half, p - half, q);
        return if q == 0 { s } else { -s };
    }

    let qh = q - half;
    let ph = p - half;
    if qh == 0 {
        return -1;
    }
    cd_basis_mul_sign(half, qh, ph)
}

#[inline(always)]
pub fn cd_basis_mul_sign_iter(dim: usize, mut p: usize, mut q: usize) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert!(p < dim && q < dim);

    let mut sign = 1i32;
    let mut half = dim >> 1;

    while half > 0 {
        let p_hi = p >= half;
        let q_hi = q >= half;

        match (p_hi, q_hi) {
            (false, false) => {}
            (false, true) => {
                let qh = q - half;
                q = p;
                p = qh;
            }
            (true, false) => {
                p -= half;
                if q != 0 {
                    sign = -sign;
                }
            }
            (true, true) => {
                let qh = q - half;
                let ph = p - half;
                if qh == 0 {
                    return -sign;
                }
                p = qh;
                q = ph;
            }
        }

        half >>= 1;
    }

    sign
}

/// Precomputed sign table for a fixed Cayley-Dickson dimension.
///
/// Uses `bitvec::BitVec` for type-safe bit-packed storage.
/// Bit = 1 means sign = -1; bit = 0 means sign = +1.
/// Layout: row-major, `bits[p * dim + q]` encodes sign(p, q).
///
/// # Why bitvec instead of manual Vec<u64>?
///
/// - Eliminates manual `idx / 64` and `idx % 64` indexing
/// - Provides `.count_ones()`, `.iter()`, bitwise AND/OR/XOR
/// - Prevents off-by-one bit indexing bugs
/// - Same memory footprint (1 bit per entry)
/// - Zero runtime overhead (compiles to identical machine code)
#[derive(Clone)]
pub struct SignTable {
    dim: usize,
    bits: bitvec::vec::BitVec<u64, bitvec::order::Lsb0>,
}

impl SignTable {
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two() && dim >= 1);
        let total_bits = dim * dim;
        let mut bits = bitvec::vec::BitVec::<u64, bitvec::order::Lsb0>::repeat(false, total_bits);

        for p in 0..dim {
            for q in 0..dim {
                let s = cd_basis_mul_sign_iter(dim, p, q);
                if s == -1 {
                    bits.set(p * dim + q, true);
                }
            }
        }

        SignTable { dim, bits }
    }

    #[inline(always)]
    pub fn sign(&self, p: usize, q: usize) -> i32 {
        debug_assert!(p < self.dim && q < self.dim);
        let is_neg = self.bits[p * self.dim + q];
        if is_neg { -1 } else { 1 }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn size_bytes(&self) -> usize {
        self.bits.as_raw_slice().len() * 8
    }

    /// Count negative signs in the table.
    /// Uses bitvec's optimized `.count_ones()`.
    pub fn count_negative(&self) -> usize {
        self.bits.count_ones()
    }

    /// Count positive signs (total - negative - diagonal zeros).
    pub fn count_positive(&self) -> usize {
        self.dim * self.dim - self.count_negative()
    }

    /// Access the raw u64 backing store.
    pub fn raw_words(&self) -> &[u64] {
        self.bits.as_raw_slice()
    }

    /// Get a bit-slice view of row p (dim bits starting at p*dim).
    #[inline(always)]
    pub fn row_bits(&self, p: usize) -> &bitvec::slice::BitSlice<u64, bitvec::order::Lsb0> {
        debug_assert!(p < self.dim);
        let start = p * self.dim;
        &self.bits[start..start + self.dim]
    }

    /// Get the raw u64 words for row p.
    ///
    /// For dim >= 64, rows are word-aligned and this returns a slice
    /// of dim/64 words. For dim < 64, the entire table fits in fewer
    /// words and row boundaries cross word boundaries -- use
    /// `row_bits()` instead for sub-word access.
    ///
    /// # Panics
    ///
    /// Panics if dim < 64 (use `row_bits()` for small dimensions).
    #[inline(always)]
    pub fn row_words(&self, p: usize) -> &[u64] {
        debug_assert!(p < self.dim);
        debug_assert!(
            self.dim >= 64,
            "row_words requires dim >= 64; use row_bits() for smaller"
        );
        let words_per_row = self.dim / 64;
        let start = p * words_per_row;
        &self.bits.as_raw_slice()[start..start + words_per_row]
    }

    /// Multiply two sparse CD elements using sign-table lookups.
    ///
    /// Each pair `(i, a_i) * (j, b_j)` contributes `(i^j, sign(i,j) * a_i * b_j)`.
    /// Results are accumulated into a dense buffer, filtered, and returned sparse.
    ///
    /// WHY: at 512D with ~4-component sparse elements this is ~16 sign lookups vs
    /// ~262K recursive `cd_multiply` calls per product.
    pub fn sparse_multiply(
        &self,
        a_sparse: &[(usize, f64)],
        b_sparse: &[(usize, f64)],
    ) -> Vec<(usize, f64)> {
        let mut accum = vec![0.0f64; self.dim];
        let mut touched = Vec::with_capacity(a_sparse.len() * b_sparse.len());
        for &(i, ai) in a_sparse {
            for &(j, bj) in b_sparse {
                let out_idx = i ^ j;
                let sign = self.sign(i, j) as f64;
                let old = accum[out_idx];
                if old == 0.0 {
                    touched.push(out_idx);
                }
                accum[out_idx] = old + sign * ai * bj;
            }
        }
        touched
            .into_iter()
            .filter_map(|idx| {
                let v = accum[idx];
                accum[idx] = 0.0;
                if v.abs() > 1e-20 {
                    Some((idx, v))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute the CD associator `[a, x, b] = (a*x)*b - a*(x*b)` in sparse form,
    /// returning the sum of all output components.
    pub fn sparse_associator_sum(
        &self,
        a_sparse: &[(usize, f64)],
        x_sparse: &[(usize, f64)],
        b_sparse: &[(usize, f64)],
    ) -> f64 {
        let ax = self.sparse_multiply(a_sparse, x_sparse);
        let left = self.sparse_multiply(&ax, b_sparse);
        let xb = self.sparse_multiply(x_sparse, b_sparse);
        let right = self.sparse_multiply(a_sparse, &xb);
        let left_sum: f64 = left.iter().map(|&(_, v)| v).sum();
        let right_sum: f64 = right.iter().map(|&(_, v)| v).sum();
        left_sum - right_sum
    }

    /// Convert a dense coefficient slice to sparse `(basis_index, coefficient)` form.
    /// Filters out entries with `|val| <= 1e-15`.
    ///
    /// Produces the input format expected by `sparse_multiply` and `sparse_associator_sum`.
    pub fn to_sparse(v: &[f64]) -> Vec<(usize, f64)> {
        v.iter()
            .enumerate()
            .filter(|&(_, &val)| val.abs() > 1e-15)
            .map(|(i, &val)| (i, val))
            .collect()
    }
}

impl std::fmt::Debug for SignTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignTable {{ dim={}, size={} bytes }}",
            self.dim,
            self.size_bytes()
        )
    }
}
