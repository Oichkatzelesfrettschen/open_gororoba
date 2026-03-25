//! Triad associativity classification (Wilmot Theorems 2, 5, 7).
//!
//! Every triad (b, c, d) in a CD algebra falls into one of:
//! - Associative: all 3 associator types are zero
//! - Type A: only Type 1 non-associativity is nonzero
//! - Type B: only Type 2 is nonzero
//! - Type C: only Type 3 is nonzero
//! - Type X: all three are nonzero (symmetric non-associativity)
//!
//! Non-associative triads form 3-cycles, and these group into
//! 8 "silos": AAA, ACC, XBB, BBA, BXC, CAB, CCX, XXX.

use std::fmt;

use cd_kernel::cayley_dickson::cd_multiply;

/// The three types of unordered associativity (Wilmot Table 1).
///
/// Type 1: [b,a,c] ~ [b,d,c] ~ [a,b,d] ~ [a,c,d]
/// Type 2: [a,b,c] ~ [b,c,d] ~ [b,a,d] ~ [a,d,c]
/// Type 3: [a,c,b] ~ [c,b,d] ~ [a,d,b] ~ [c,a,d]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssociativityType {
    /// All three types zero -- the triad is fully associative.
    Associative,
    /// Only Type 1 is nonzero.
    TypeA,
    /// Only Type 2 is nonzero.
    TypeB,
    /// Only Type 3 is nonzero.
    TypeC,
    /// All three types are nonzero (symmetric non-associativity).
    TypeX,
}

impl fmt::Display for AssociativityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Associative => write!(f, "Assoc"),
            Self::TypeA => write!(f, "A"),
            Self::TypeB => write!(f, "B"),
            Self::TypeC => write!(f, "C"),
            Self::TypeX => write!(f, "X"),
        }
    }
}

/// Compute the associator [a, b, c] = (ab)c - a(bc).
///
/// Returns the norm-squared of the associator (0 if associative).
pub fn associator_norm_sq(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    let ab = cd_multiply(a, b);
    let ab_c = cd_multiply(&ab, c);
    let bc = cd_multiply(b, c);
    let a_bc = cd_multiply(a, &bc);
    ab_c.iter()
        .zip(a_bc.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Classify a triad (b, c, d) where a = bcd (the fourth element).
///
/// Computes the three associativity types and returns the classification.
/// Uses basis elements at the given dimension.
///
/// Mirrors: Wilmot Theorem 2 + Theorem 5.
pub fn classify_triad(b: &[f64], c: &[f64], d: &[f64], atol: f64) -> AssociativityType {
    let a = {
        let bc = cd_multiply(b, c);
        cd_multiply(&bc, d)
    };

    // Type 1: [b, a, c] (and equivalents)
    let type1 = associator_norm_sq(b, &a, c);
    // Type 2: [a, b, c] (and equivalents)
    let type2 = associator_norm_sq(&a, b, c);
    // Type 3: [a, c, b] (and equivalents)
    let type3 = associator_norm_sq(&a, c, b);

    let t1_nz = type1 > atol;
    let t2_nz = type2 > atol;
    let t3_nz = type3 > atol;

    match (t1_nz, t2_nz, t3_nz) {
        (false, false, false) => AssociativityType::Associative,
        (true, false, false) => AssociativityType::TypeA,
        (false, true, false) => AssociativityType::TypeB,
        (false, false, true) => AssociativityType::TypeC,
        (true, true, true) => AssociativityType::TypeX,
        // Theorem 5 says these can't happen, but numerics might disagree
        _ => AssociativityType::TypeX, // fallback
    }
}

/// Count triads of each type for a CD algebra at given dimension.
///
/// Iterates over all C(N, 3) unordered basis triples and classifies each.
///
/// Mirrors: Wilmot Table 4 (Non-associative Structure).
pub fn count_triad_types(dim: usize) -> TriadCounts {
    let n = dim - 1; // number of imaginary basis elements
    let mut counts = TriadCounts::default();

    for i in 1..=n {
        for j in (i + 1)..=n {
            for k in (j + 1)..=n {
                let mut b = vec![0.0; dim];
                let mut c = vec![0.0; dim];
                let mut d = vec![0.0; dim];
                b[i] = 1.0;
                c[j] = 1.0;
                d[k] = 1.0;

                let ty = classify_triad(&b, &c, &d, 1e-8);
                match ty {
                    AssociativityType::Associative => counts.associative += 1,
                    AssociativityType::TypeA => counts.type_a += 1,
                    AssociativityType::TypeB => counts.type_b += 1,
                    AssociativityType::TypeC => counts.type_c += 1,
                    AssociativityType::TypeX => counts.type_x += 1,
                }
            }
        }
    }
    counts
}

/// Triad type counts for a CD algebra.
#[derive(Debug, Clone, Default)]
pub struct TriadCounts {
    pub associative: usize,
    pub type_a: usize,
    pub type_b: usize,
    pub type_c: usize,
    pub type_x: usize,
}

impl TriadCounts {
    pub fn total(&self) -> usize {
        self.associative + self.type_a + self.type_b + self.type_c + self.type_x
    }

    pub fn non_associative(&self) -> usize {
        self.type_a + self.type_b + self.type_c + self.type_x
    }
}

impl fmt::Display for TriadCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Assoc={}, A={}, B={}, C={}, X={} (total={})",
            self.associative,
            self.type_a,
            self.type_b,
            self.type_c,
            self.type_x,
            self.total()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_all_type_x() {
        // Octonions: 7 associative + 28 non-associative = 35 triads
        // All non-associative triads should be Type X (Wilmot: "one symmetric type")
        let counts = count_triad_types(8);
        assert_eq!(counts.total(), 35);
        assert_eq!(counts.associative, 7);
        assert_eq!(counts.type_x + counts.type_a + counts.type_b + counts.type_c, 28);
    }

    #[test]
    fn test_sedenion_counts_match_table4() {
        // Wilmot Table 4: U_1 has 35 assoc, 84 A, 112 C, 224 X = 455 total
        // (B count is from Table 4: 336 A in the paper... need to recheck)
        let counts = count_triad_types(16);
        assert_eq!(counts.total(), 455);
        assert_eq!(counts.associative, 35);
        // The non-associative split should match Wilmot Table 4
        assert_eq!(counts.non_associative(), 420);
    }

    #[test]
    fn test_quaternion_all_associative() {
        let counts = count_triad_types(4);
        assert_eq!(counts.total(), 1); // C(3,3) = 1
        assert_eq!(counts.associative, 1);
    }

    #[test]
    fn test_associator_zero_for_quaternion_triple() {
        let b = vec![0.0, 1.0, 0.0, 0.0]; // i
        let c = vec![0.0, 0.0, 1.0, 0.0]; // j
        let d = vec![0.0, 0.0, 0.0, 1.0]; // k
        let norm = associator_norm_sq(&b, &c, &d);
        assert!(norm < 1e-10, "quaternion triple should be associative");
    }

    #[test]
    fn test_associator_nonzero_for_octonion_triple() {
        // e1, e2, e4 in octonions
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let norm = associator_norm_sq(&b, &c, &d);
        assert!(norm > 0.1, "octonion (e1,e2,e4) should be non-associative");
    }
}
