//! Twisted Group Algebra Representation of Cayley-Dickson Algebras.
//!
//! Codifies the explicit twist function sigma(A, B) from:
//! Ren & Zhao (2022), "The twisted group algebra structure of the Cayley-Dickson algebra".
//!
//! e_A * e_B = (-1)^sigma(A, B) * e_{A ^ B}

/// The Ren-Zhao twist function sigma(A, B) computed modulo 2.
///
/// sigma(A + a_n*2^n, B + b_n*2^n) = sigma(A, B)*(1 + b_n) + sigma(B, A)*b_n + a_n*sigma(B, B) + a_n*b_n
/// where A, B < 2^n and a_n, b_n are the n-th bits.
pub fn cd_twist(a: usize, b: usize) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    if a == b {
        return 1;
    }

    // Find the highest bit position involved
    let max_val = a.max(b);
    let n = (usize::BITS - max_val.leading_zeros()) as i32;
    
    sigma_recursive(a, b, n - 1)
}

fn sigma_recursive(a: usize, b: usize, bit: i32) -> u8 {
    if bit < 0 {
        return 0;
    }
    
    let mask = 1 << bit;
    let a_low = a & (mask - 1);
    let b_low = b & (mask - 1);
    let a_n = ((a >> bit) & 1) as u8;
    let b_n = ((b >> bit) & 1) as u8;
    
    let s_ab = sigma_recursive(a_low, b_low, bit - 1);
    let s_ba = sigma_recursive(b_low, a_low, bit - 1);
    let s_bb = sigma_recursive(b_low, b_low, bit - 1);
    
    // Formula: s_ab * (1 + b_n) + s_ba * b_n + a_n * s_bb + a_n * b_n
    let term1 = s_ab * (1 - b_n); // (1 + b_n) mod 2 is (1 - b_n)
    let term2 = s_ba * b_n;
    let term3 = a_n * s_bb;
    let term4 = a_n * b_n;
    
    (term1 + term2 + term3 + term4) % 2
}

/// A precomputed twist table for fast sign lookups.
pub struct TwistTable {
    pub dim: usize,
    table: Vec<u8>,
}

impl TwistTable {
    pub fn new(dim: usize) -> Self {
        let mut table = vec![0u8; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                table[a * dim + b] = cd_twist(a, b);
            }
        }
        TwistTable { dim, table }
    }

    #[inline(always)]
    pub fn sign(&self, a: usize, b: usize) -> i32 {
        if self.table[a * self.dim + b] == 0 {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

    #[test]
    fn test_ren_zhao_vs_kernel() {
        let dims = [2, 4, 8, 16, 32, 64];
        for dim in dims {
            let table = TwistTable::new(dim);
            for a in 0..dim {
                for b in 0..dim {
                    let s1 = table.sign(a, b);
                    let s2 = cd_basis_mul_sign_iter(dim, a, b);
                    assert_eq!(s1, s2, "Sign mismatch at dim {}, indices ({}, {})", dim, a, b);
                }
            }
        }
    }
}
