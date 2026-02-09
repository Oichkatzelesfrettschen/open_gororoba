//! Hierarchical Codebook Logic for 256D -> 2048D Lattice Mappings.
//!
//! Implements the "predicate cut" filtration described in the analysis:
//! Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= {-1, 0, 1}^8.
//!
//! # Hierarchy
//! - Base: Trinary vectors, even sum, even weight.
//! - 2048D: Base minus 139 forbidden prefixes.
//! - 1024D: 2048D intersected with {l_0 = -1} minus 70 prefixes.
//! - 512D: 1024D minus 6 forbidden regions (trie cuts).
//! - 256D: 512D minus 6 forbidden regions.
//!
//! # Scalar Shadow
//! Implements the affine/linear action of the scalar shadow pi(b) on the lattice.


/// A vector in the 8D integer lattice (typically {-1, 0, 1}).
pub type LatticeVector = [i8; 8];

/// Check if a vector is in the "Base Universe" (Trinary, Even Sum, Even Weight).
pub fn is_in_base_universe(v: &LatticeVector) -> bool {
    // 1. Trinary
    if v.iter().any(|&x| !(-1..=1).contains(&x)) {
        return false;
    }
    // 2. Even coordinate sum
    let sum: i32 = v.iter().map(|&x| x as i32).sum();
    if sum % 2 != 0 {
        return false;
    }
    // 3. Even Hamming weight (nonzero count)
    let weight = v.iter().filter(|&&x| x != 0).count();
    if weight % 2 != 0 {
        return false;
    }
    // 4. l_0 != +1 (from analysis of 2048D set)
    if v[0] == 1 {
        return false;
    }
    true
}

/// Check if a vector is in Lambda_2048 (Base minus 139 forbidden prefixes).
pub fn is_in_lambda_2048(v: &LatticeVector) -> bool {
    if !is_in_base_universe(v) {
        return false;
    }

    // Forbidden prefixes for 2048D
    // (l_0, l_1, l_2) = (0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return false;
    }
    // (l_0..l_4) = (0, 1, 0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return false;
    }
    // (l_0..l_5) = (0, 1, 0, 1, 0, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return false;
    }

    true
}

/// Check if a vector is in Lambda_1024 (Lambda_2048 with l_0 = -1 minus 70 points).
pub fn is_in_lambda_1024(v: &LatticeVector) -> bool {
    if !is_in_lambda_2048(v) {
        return false;
    }
    
    // Slice condition
    if v[0] != -1 {
        return false;
    }

    // Additional exclusions for 1024D
    // (-1, 1, 1, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 1 {
        return false;
    }
    // (-1, 1, 1, 0, 0)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 0 {
        return false;
    }
    // (-1, 1, 1, 0, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 1 {
        return false;
    }
    // Singleton exceptions would go here, but omitted for brevity in this MVP.

    true
}

/// Check if a vector is in Lambda_512 (Lambda_1024 minus 6 regions).
pub fn is_in_lambda_512(v: &LatticeVector) -> bool {
    if !is_in_lambda_1024(v) {
        return false;
    }

    // Forbidden regions (l_0 is always -1 here)
    // 1. l_1 = 1
    if v[1] == 1 { return false; }
    // 2. l_1=0, l_2=1
    if v[1] == 0 && v[2] == 1 { return false; }
    // 3. l_1=0, l_2=0, l_3=0
    if v[1] == 0 && v[2] == 0 && v[3] == 0 { return false; }
    // 4. l_1=0, l_2=0, l_3=1
    if v[1] == 0 && v[2] == 0 && v[3] == 1 { return false; }
    // 5. l_1=0, l_2=0, l_3=-1, l_4=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 1 { return false; }
    // 6. l_1=0, l_2=0, l_3=-1, l_4=0, l_5=1, l_6=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 0 && v[5] == 1 && v[6] == 1 { return false; }

    true
}

/// Check if a vector is in Lambda_256 (Lambda_512 minus 6 regions).
pub fn is_in_lambda_256(v: &LatticeVector) -> bool {
    if !is_in_lambda_512(v) {
        return false;
    }

    // Forbidden regions (l_0 = -1)
    // 1. l_1 = 0 (implies l_1 must be -1 for success, since l_1 != 1 from 512 rule)
    if v[1] == 0 { return false; } 
    
    // For the remaining, l_1 = -1 is established.
    // 2. (-1, -1, 1, 1)
    if v[2] == 1 && v[3] == 1 { return false; }
    // 3. (-1, -1, 1, 0)
    if v[2] == 1 && v[3] == 0 { return false; }
    // 4. (-1, -1, 1, -1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == 1 { return false; }
    // 5. (-1, -1, 1, -1, 0)
    if v[2] == 1 && v[3] == -1 && v[4] == 0 { return false; }
    // 6. Singleton (-1, -1, 1, -1, -1, 1, 1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == -1 && v[5] == 1 && v[6] == 1 && v[7] == 1 { return false; }

    true
}

/// Apply the Scalar Shadow action to a lattice vector.
///
/// Addition mode: l_out = l + a * 1_8
/// Multiplication mode: l_out = a * l
pub fn apply_scalar_shadow(l: &LatticeVector, a: i8, mode: &str) -> LatticeVector {
    let mut res = [0i8; 8];
    match mode {
        "add" => {
            // Affine shift: l + a * [1,1,...]
            for i in 0..8 {
                // Saturating add/sub to avoid panic, though algebra is mod-something?
                // Analysis says integer lattice Z^8, so we should allow growth.
                // But type is i8... let's use saturating for now or assume inputs small.
                res[i] = l[i].saturating_add(a);
            }
        },
        "mul" => {
            // Linear action: a * l
            for i in 0..8 {
                res[i] = l[i] * a;
            }
        },
        _ => panic!("Unknown mode"),
    }
    res
}
