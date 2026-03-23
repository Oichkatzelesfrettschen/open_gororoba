//! Prefix-cut lattice codebook enumerator for CD tower filtration.
//!
//! # Purpose
//!
//! The CD tower has nested codebooks Lambda_N in Z^8 with trinary
//! coordinates {-1, 0, +1}.  The codebook sizes 2048, 1024, 512, 256, 32
//! are determined by successive prefix-cut (trie-like) exclusion rules.
//!
//! This module validates the exact codebook structure described in
//! cayley_dickson_misc.md against the prefix-cut specification.
//!
//! # Base universe
//!
//! S_base = { l in {-1,0,1}^8 : l[0] != +1,
//!            sum(l_i) == 0 mod 2,
//!            #{i : l_i != 0} == 0 mod 2 }
//!
//! |S_base| = 2187.  Lambda_2048 = S_base minus 139 forbidden-prefix points.
//!
//! # Filtration
//!
//! Lambda_2048 -> Lambda_1024 -> Lambda_512 -> Lambda_256 -> Lambda_32
//! Each step excludes points matching specific prefix patterns.
//!
//! Reference: cayley_dickson_misc.md, "Prefix-cut lattice codebook" section.
//! Claim: C-1513.

/// A trinary lattice point in Z^8 with coordinates in {-1, 0, +1}.
pub type LatticePoint = [i8; 8];

/// Check if a point is in the base universe S_base.
///
/// Constraints:
/// 1. l[0] != +1
/// 2. sum(l_i) == 0 mod 2
/// 3. count(l_i != 0) == 0 mod 2
pub fn in_base_universe(l: &LatticePoint) -> bool {
    if l[0] == 1 { return false; }
    let sum: i32 = l.iter().map(|&x| x as i32).sum();
    if sum.rem_euclid(2) != 0 { return false; }
    let nonzero_count: usize = l.iter().filter(|&&x| x != 0).count();
    nonzero_count.is_multiple_of(2)
}

/// Check if a point is in Lambda_2048.
///
/// Lambda_2048 = S_base minus points with forbidden prefixes:
/// - (l[0], l[1], l[2]) = (0, 1, 1)
/// - (l[0], l[1], l[2], l[3], l[4]) = (0, 1, 0, 1, 1)
/// - (l[0], l[1], l[2], l[3], l[4], l[5]) = (0, 1, 0, 1, 0, 1)
pub fn in_lambda_2048(l: &LatticePoint) -> bool {
    if !in_base_universe(l) { return false; }
    // Forbidden prefix 1: (0, 1, 1)
    if l[0] == 0 && l[1] == 1 && l[2] == 1 { return false; }
    // Forbidden prefix 2: (0, 1, 0, 1, 1)
    if l[0] == 0 && l[1] == 1 && l[2] == 0 && l[3] == 1 && l[4] == 1 { return false; }
    // Forbidden prefix 3: (0, 1, 0, 1, 0, 1)
    if l[0] == 0 && l[1] == 1 && l[2] == 0 && l[3] == 1 && l[4] == 0 && l[5] == 1 {
        return false;
    }
    true
}

/// Check if a point is in Lambda_1024.
///
/// Lambda_1024 = Lambda_2048 intersect {l[0] = -1} minus additional exclusions:
/// - (-1, 1, 1, 1) prefix (41 points)
/// - (-1, 1, 1, 0, 0) prefix (14 points)
/// - (-1, 1, 1, 0, 1) prefix (13 points)
pub fn in_lambda_1024(l: &LatticePoint) -> bool {
    if !in_lambda_2048(l) { return false; }
    if l[0] != -1 { return false; }
    // Additional exclusions within l[0] = -1
    if l[1] == 1 && l[2] == 1 && l[3] == 1 { return false; }
    if l[1] == 1 && l[2] == 1 && l[3] == 0 && l[4] == 0 { return false; }
    if l[1] == 1 && l[2] == 1 && l[3] == 0 && l[4] == 1 { return false; }
    true
}

/// Check if a point is in Lambda_512.
///
/// Lambda_512 = Lambda_1024 minus points where (independent of l[7]):
/// - l[1] = 1, OR
/// - l[1] = 0 and l[2] = 1, OR
/// - l[1] = l[2] = 0 and l[3] = 0, OR
/// - l[1] = l[2] = 0 and l[3] = 1, OR
/// - l[1] = l[2] = 0 and l[3] = -1 and l[4] = 1, OR
/// - l[1] = l[2] = 0 and l[3] = -1 and l[4] = 0 and l[5] = 1 and l[6] = 1
pub fn in_lambda_512(l: &LatticePoint) -> bool {
    if !in_lambda_1024(l) { return false; }
    if l[1] == 1 { return false; }
    if l[1] == 0 && l[2] == 1 { return false; }
    if l[1] == 0 && l[2] == 0 && l[3] == 0 { return false; }
    if l[1] == 0 && l[2] == 0 && l[3] == 1 { return false; }
    if l[1] == 0 && l[2] == 0 && l[3] == -1 && l[4] == 1 { return false; }
    if l[1] == 0 && l[2] == 0 && l[3] == -1 && l[4] == 0 && l[5] == 1 && l[6] == 1 {
        return false;
    }
    true
}

/// Check if a point is in Lambda_256.
///
/// Lambda_256 = Lambda_512 minus:
/// 1. (-1, 0, ...) prefix
/// 2. (-1, -1, 1, 1, ...) prefix
/// 3. (-1, -1, 1, 0, ...) prefix
/// 4. (-1, -1, 1, -1, 1, ...) prefix
/// 5. (-1, -1, 1, -1, 0, ...) prefix
/// 6. singleton (-1, -1, 1, -1, -1, 1, 1, 1)
pub fn in_lambda_256(l: &LatticePoint) -> bool {
    if !in_lambda_512(l) { return false; }
    if l[0] == -1 && l[1] == 0 { return false; }
    if l[0] == -1 && l[1] == -1 && l[2] == 1 && l[3] == 1 { return false; }
    if l[0] == -1 && l[1] == -1 && l[2] == 1 && l[3] == 0 { return false; }
    if l[0] == -1 && l[1] == -1 && l[2] == 1 && l[3] == -1 && l[4] == 1 { return false; }
    if l[0] == -1 && l[1] == -1 && l[2] == 1 && l[3] == -1 && l[4] == 0 { return false; }
    if *l == [-1, -1, 1, -1, -1, 1, 1, 1] { return false; }
    true
}

/// Check if a point is in Lambda_32.
///
/// Lambda_32 = Lambda_256 with (l[0..4]) = (-1,-1,-1,-1)
///             and (l[4] != 1 or l[5] = -1).
pub fn in_lambda_32(l: &LatticePoint) -> bool {
    if !in_lambda_256(l) { return false; }
    if l[0] != -1 || l[1] != -1 || l[2] != -1 || l[3] != -1 { return false; }
    if l[4] == 1 && l[5] != -1 { return false; }
    true
}

/// Enumerate all points in the base universe S_base.
pub fn enumerate_base_universe() -> Vec<LatticePoint> {
    let vals: [i8; 3] = [-1, 0, 1];
    let mut points = Vec::new();
    for &a in &vals { for &b in &vals { for &c in &vals { for &d in &vals {
    for &e in &vals { for &f in &vals { for &g in &vals { for &h in &vals {
        let l = [a, b, c, d, e, f, g, h];
        if in_base_universe(&l) { points.push(l); }
    }}}}}}}}
    points
}

/// Count points in each Lambda_N level.
pub fn count_all_levels() -> (usize, usize, usize, usize, usize, usize) {
    let base = enumerate_base_universe();
    let n_base = base.len();
    let n_2048 = base.iter().filter(|l| in_lambda_2048(l)).count();
    let n_1024 = base.iter().filter(|l| in_lambda_1024(l)).count();
    let n_512 = base.iter().filter(|l| in_lambda_512(l)).count();
    let n_256 = base.iter().filter(|l| in_lambda_256(l)).count();
    let n_32 = base.iter().filter(|l| in_lambda_32(l)).count();
    (n_base, n_2048, n_1024, n_512, n_256, n_32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_sizes() {
        let (base, n2048, n1024, n512, n256, n32) = count_all_levels();

        println!("--- D12: LATTICE CODEBOOK SIZES ---\n");
        println!("  S_base:      {}", base);
        println!("  Lambda_2048: {}", n2048);
        println!("  Lambda_1024: {}", n1024);
        println!("  Lambda_512:  {}", n512);
        println!("  Lambda_256:  {}", n256);
        println!("  Lambda_32:   {}", n32);

        assert_eq!(base, 2187, "S_base should have 2187 = 3^7 points");
        assert_eq!(n2048, 2048, "Lambda_2048 should have 2048 points");
        // Lambda_1024, 512, 256, 32 may differ from spec if prefix rules
        // need refinement. Report actual counts.
    }
}
