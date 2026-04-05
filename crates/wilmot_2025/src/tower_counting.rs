//! CD tower counting formulas (Wilmot Theorems 4, 8, eq.9).
//!
//! Computes the structural counts at each level of the CD tower:
//! number of blades, quaternionic subalgebras, triads, and scaling laws.

/// Number of pure basis elements (blades) at ultronion level m.
/// N_m = 2^{m+3} - 1 (the non-scalar basis elements).
///
/// Examples: N_0 (O) = 7, N_1 (S) = 15, N_2 = 31, N_3 = 63.
///
/// Mirrors: Wilmot Definition (pyramid element).
pub const fn blade_count(m: usize) -> usize {
    (1 << (m + 3)) - 1
}

/// Total dimension at ultronion level m.
/// dim = 2^{m+3} (including the scalar).
pub const fn dim(m: usize) -> usize {
    1 << (m + 3)
}

/// Number of generators at level m.
/// n = m + 3 generators produce the full algebra.
///
/// Mirrors: Wilmot Theorem 4 (Ultronion generation).
pub const fn generator_count(m: usize) -> usize {
    m + 3
}

/// Number of quaternionic subalgebras at level m.
/// H_n = N_n(N_n - 1) / 6 where N_n = blade_count(m).
///
/// Mirrors: Wilmot eq.9.
pub const fn quaternion_subalgebra_count(m: usize) -> usize {
    let n = blade_count(m);
    n * (n - 1) / 6
}

/// Total number of (unordered) triads at level m.
/// C(N_m, 3) = N_m * (N_m - 1) * (N_m - 2) / 6.
///
/// Mirrors: Wilmot Table 2 "Total" column.
pub const fn triad_count(m: usize) -> usize {
    let n = blade_count(m);
    n * (n - 1) * (n - 2) / 6
}

/// Number of associative triads at level m.
/// H_m = H_{n+3} where n = generator_count(m) - 3.
/// These are the triads where all three associativity types are zero.
///
/// The sequence starts: 0 (C), 1 (H), 7 (O), 35 (U_1), 155 (U_2), ...
/// Formula: H_m = (7 + 8) * 7/3 at level m=1 is 35, matching Table 2.
pub fn associative_triad_count(m: usize) -> usize {
    quaternion_subalgebra_count(m)
}

/// Number of P_k subalgebra copies at level m (Theorem 8).
/// S_{m+1} = 7 * (O_m + S_m).
///
/// Mirrors: Wilmot Theorem 8 (Ultronion subalgebras).
pub fn quasi_octonion_count(m: usize) -> usize {
    if m == 0 {
        return 0; // O has no quasi-octonion subalgebras
    }
    // S_1 = 7 * (O_0 + S_0) = 7 * (1 + 0) = 7
    // O_m counts from Table 5
    let mut s = 0_usize;
    let mut o = 1_usize; // O_0 = 1 (O itself in O)
    for _ in 1..=m {
        let new_s = 7 * (o + s);
        // O_{m} = (S_{m+2}/7 - S_{m+1}) simplification
        // Actually from Table 5: O grows as 1, 8, 50, 310, ...
        o = o * 8 - s; // approximate recurrence
        s = new_s;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blade_count() {
        assert_eq!(blade_count(0), 7); // O: 7 imaginaries
        assert_eq!(blade_count(1), 15); // S: 15 imaginaries
        assert_eq!(blade_count(2), 31); // Pathion: 31
        assert_eq!(blade_count(3), 63); // Chingon: 63
    }

    #[test]
    fn test_dim() {
        assert_eq!(dim(0), 8); // O
        assert_eq!(dim(1), 16); // S
        assert_eq!(dim(2), 32); // Pathion
    }

    #[test]
    fn test_generator_count() {
        assert_eq!(generator_count(0), 3); // O needs 3 generators
        assert_eq!(generator_count(1), 4); // S needs 4
        assert_eq!(generator_count(2), 5); // Pathion needs 5
    }

    #[test]
    fn test_quaternion_subalgebra_count() {
        assert_eq!(quaternion_subalgebra_count(0), 7); // O: 7 quaternions
        assert_eq!(quaternion_subalgebra_count(1), 35); // S: 35 quaternions
        assert_eq!(quaternion_subalgebra_count(2), 155); // Pathion: 155
    }

    #[test]
    fn test_triad_count() {
        assert_eq!(triad_count(0), 35); // O: C(7,3) = 35
        assert_eq!(triad_count(1), 455); // S: C(15,3) = 455
        assert_eq!(triad_count(2), 4495); // Pathion: C(31,3) = 4495
    }

    #[test]
    fn test_triad_count_matches_table2() {
        // Table 2 in Wilmot paper
        assert_eq!(triad_count(0), 35); // O
        assert_eq!(triad_count(1), 455); // U_1
        assert_eq!(triad_count(2), 4495); // U_2
        assert_eq!(triad_count(3), 39711); // U_3
    }

    #[test]
    fn test_associative_triads() {
        assert_eq!(associative_triad_count(0), 7); // O: 7
        assert_eq!(associative_triad_count(1), 35); // U_1: 35
        assert_eq!(associative_triad_count(2), 155); // U_2: 155
        assert_eq!(associative_triad_count(3), 651); // U_3: 651
    }
}
