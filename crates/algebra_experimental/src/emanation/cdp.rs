//! CDP signed-product engine (L1 in de Marrais's "Presto! Digitization"
//! Appendix).
//!
//! Faithful translation of de Marrais's M(LI, RI) function from
//! LotusScript. Given two Cayley-Dickson basis indices, returns a SIGNED
//! product: `sign * (LI XOR RI)`.
//!
//! The algorithm:
//! 1. QSIGNS[4x4] quaternion base case (hard-coded multiplication table)
//! 2. Handle negative inputs (absorb signs into NegTally accumulator)
//! 3. XorRoot = LI XOR RI (the product index, assuming we know the sign)
//! 4. Recursive reduction: strip highest bits while toggling NegTally,
//!    until we reach the quaternion base case or a termination condition
//!
//! Reference: de Marrais (2006), arXiv:math/0603281, Appendix pp.20-27.
//! The recursion terminates either at the quaternion base case or when
//! one index becomes zero.

/// Quaternion multiplication sign table (indices 0..3).
///
/// QSigns[i][j] gives the sign of e_i * e_j in the quaternion subalgebra.
/// Layout:
///   e0=1 (real), e1=i, e2=j, e3=k
///   e1*e2 = +e3, e2*e1 = -e3
///   e2*e3 = +e1, e3*e2 = -e1
///   e3*e1 = +e2, e1*e3 = -e2
///   e_i*e_i = -1 for i>0
pub(crate) const QSIGNS: [[i8; 4]; 4] = [
    [1, 1, 1, 1],   // e0 * e_j = +e_j
    [1, -1, 1, -1], // e1: e1*e0=+1, e1*e1=-1, e1*e2=+e3, e1*e3=-e2
    [1, -1, -1, 1], // e2: e2*e0=+1, e2*e1=-e3, e2*e2=-1, e2*e3=+e1
    [1, 1, -1, -1], // e3: e3*e0=+1, e3*e1=+e2, e3*e2=-e1, e3*e3=-1
];

/// De Marrais's M function: signed Cayley-Dickson basis product.
///
/// Given basis indices `li` and `ri`, returns `sign * (li XOR ri)` as a
/// signed integer. The product index is `|result|` and the sign is `signum(result)`.
///
/// Special case: `M(0, 0) = +1` (real * real = +real).
/// For `li == ri > 0`: returns `-(li XOR ri) = 0`, but we return the sign
/// separately since the product index is 0 (real unit).
///
/// Returns `(product_index, sign)` where `e_li * e_ri = sign * e_{product_index}`.
pub fn cdp_signed_product(li: usize, ri: usize) -> (usize, i8) {
    // The product index is always li XOR ri.
    let xor_root = li ^ ri;

    let mut neg_tally: i8 = 1;
    let mut l = li;
    let mut r = ri;

    loop {
        // Termination: either index is 0 => product is the other index with current sign.
        if l == 0 || r == 0 {
            break;
        }

        // Termination: l == r => e_i * e_i = -1 (imaginary squaring).
        if l == r {
            neg_tally = -neg_tally;
            break;
        }

        let l_bits = bit_length(l);
        let r_bits = bit_length(r);

        // Quaternion base case: both indices fit in 2 bits (0..3).
        if l_bits < 3 && r_bits < 3 {
            neg_tally *= QSIGNS[l][r];
            break;
        }

        if l_bits == r_bits {
            // Both indices arise from the same generator G = 2^(l_bits - 1).
            let g = 1usize << (l_bits - 1);

            if l == g {
                // l is the generator itself: triplet = (l XOR r, l=G, r)
                // Sign is positive (l < r, standard ordering).
                break;
            }
            if r == g {
                // r is the generator: triplet = (l XOR r, r=G, l)
                // Reversed from standard => negate.
                neg_tally = -neg_tally;
                break;
            }
            if (l ^ r) == g {
                // XOR product equals generator: triplet = (lo, G, hi)
                // Sign depends on ordering: if r > l, negate.
                if r > l {
                    neg_tally = -neg_tally;
                }
                break;
            }

            // General case: both in same doubling level.
            // For generator G, row = G + a, col = G + b => product = (-1) * a * b
            neg_tally = -neg_tally;
            l -= g;
            r -= g;
            continue; // RECURSIVE
        }

        if l_bits < r_bits {
            // l is in a lower doubling level than r.
            let g = 1usize << (r_bits - 1);

            if r == g {
                // r is the generator of its level.
                break;
            }
            if (l ^ r) == g {
                // XOR equals generator => negate.
                neg_tally = -neg_tally;
                break;
            }

            // Strip generator from r, negate.
            neg_tally = -neg_tally;
            r -= g;
            continue; // RECURSIVE
        }

        // r_bits < l_bits: r is in a lower doubling level than l.
        {
            let g = 1usize << (l_bits - 1);

            if (l ^ r) == g {
                // XOR equals generator.
                break;
            }

            neg_tally = -neg_tally;

            if l == g {
                // l is the generator of its level.
                break;
            }

            // Strip generator from l.
            l -= g;
            continue; // RECURSIVE
        }
    }

    (xor_root, neg_tally)
}

/// Number of bits needed to represent `n` (equivalent to floor(log2(n)) + 1).
/// Returns 0 for n == 0.
pub(crate) fn bit_length(n: usize) -> u32 {
    if n == 0 {
        0
    } else {
        usize::BITS - n.leading_zeros()
    }
}
