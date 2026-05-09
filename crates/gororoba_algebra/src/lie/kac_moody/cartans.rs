//! Generalized Cartan matrices for the E-series and the classical `A_n`, `D_n` families.
//!
//! All E-series factories share the **branch-at-node-4** Dynkin numbering, so
//! they extend cleanly via the affine node `8` attached to the far end of the
//! long arm at node `0` (`E_9`), and beyond (`E_10`, `E_11`).
//!
//! Sister convention: [`crate::lie::e8::root_system::e8_cartan_matrix`]
//! returns the same 8x8 entries as [`e8_cartan`] but as a `[[i32; 8]; 8]`
//! rather than a [`super::GeneralizedCartanMatrix`].

use super::GeneralizedCartanMatrix;

// === E-series Cartan matrices ===

/// E8 Cartan matrix (finite, simply-laced, exceptional).
///
/// Node numbering matches the standard root vectors in `E9RootSystem::new()`:
///
/// ```text
///     0 -- 1 -- 2 -- 3 -- 4 -- 5
///                          |
///                          6 -- 7
/// ```
///
/// Branching at node 4 (degree 3). This is the Gram matrix of the root vectors:
///   alpha_0 = (1,-1,0,0,0,0,0,0), alpha_1 = (0,1,-1,0,0,0,0,0), ...,
///   alpha_6 = (0,0,0,0,0,1,1,0), alpha_7 = (-1/2,-1/2,...,-1/2).
///
/// NOTE on Dynkin node numbering: this module uses a 0-indexed convention
/// where the branch node is at index 4, optimized for E9/E10 affine extension
/// (node 8 attaches to node 0 at the far end of the long arm). This differs
/// from the Bourbaki convention used in `e8_lattice.rs`, where the branch
/// node is at index 2 (1-indexed: node 3). Both are isomorphic; the mapping
/// between conventions is a relabeling of the Dynkin diagram nodes.
pub fn e8_cartan() -> GeneralizedCartanMatrix {
    GeneralizedCartanMatrix::from_array([
        [2, -1, 0, 0, 0, 0, 0, 0],   // 0: connects to 1
        [-1, 2, -1, 0, 0, 0, 0, 0],  // 1: connects to 0, 2
        [0, -1, 2, -1, 0, 0, 0, 0],  // 2: connects to 1, 3
        [0, 0, -1, 2, -1, 0, 0, 0],  // 3: connects to 2, 4
        [0, 0, 0, -1, 2, -1, -1, 0], // 4: connects to 3, 5, 6 (branch)
        [0, 0, 0, 0, -1, 2, 0, 0],   // 5: connects to 4
        [0, 0, 0, 0, -1, 0, 2, -1],  // 6: connects to 4, 7
        [0, 0, 0, 0, 0, 0, -1, 2],   // 7: connects to 6
    ])
    .expect("E8 Cartan matrix is valid")
}

/// E9 = E8^{(1)} Cartan matrix (affine extension of E8).
///
/// E9 is the affine Kac-Moody algebra associated with E8.
/// It has rank 9 with determinant 0 (one null direction).
/// Important in heterotic string theory compactifications.
///
/// Node numbering: E8 nodes 0-7 (matching `e8_cartan()` and root vectors),
/// plus affine extension node 8 connected to node 0 (end of the long arm).
///
/// The highest root of E8 (in our 0-indexed root-vector convention) is:
///   theta = 2*alpha_0 + 3*alpha_1 + 4*alpha_2 + 5*alpha_3
///         + 6*alpha_4 + 3*alpha_5 + 4*alpha_6 + 2*alpha_7
///         = (1, 0, 0, 0, 0, 0, 0, -1)
/// with <theta, alpha_0> = 1, <theta, alpha_i> = 0 for i > 0.
/// The affine root is alpha_8 = delta - theta, connecting to node 0.
///
/// ```text
///     8 -- 0 -- 1 -- 2 -- 3 -- 4 -- 5
///                               |
///                               6 -- 7
/// ```
pub fn e9_cartan() -> GeneralizedCartanMatrix {
    GeneralizedCartanMatrix::from_array([
        [2, -1, 0, 0, 0, 0, 0, 0, -1],  // 0: connects to 1, 8
        [-1, 2, -1, 0, 0, 0, 0, 0, 0],  // 1: connects to 0, 2
        [0, -1, 2, -1, 0, 0, 0, 0, 0],  // 2: connects to 1, 3
        [0, 0, -1, 2, -1, 0, 0, 0, 0],  // 3: connects to 2, 4
        [0, 0, 0, -1, 2, -1, -1, 0, 0], // 4: connects to 3, 5, 6 (branch)
        [0, 0, 0, 0, -1, 2, 0, 0, 0],   // 5: connects to 4
        [0, 0, 0, 0, -1, 0, 2, -1, 0],  // 6: connects to 4, 7
        [0, 0, 0, 0, 0, 0, -1, 2, 0],   // 7: connects to 6
        [-1, 0, 0, 0, 0, 0, 0, 0, 2],   // 8 (affine): connects to 0
    ])
    .expect("E9 Cartan matrix is valid")
}

/// E10 Cartan matrix (hyperbolic, Lorentzian signature).
///
/// E10 is conjectured to be a symmetry of M-theory (Damour-Henneaux-Nicolai).
/// The Cartan matrix has signature (9, 1).
///
/// Node numbering: E8 nodes 0-7, affine node 8, hyperbolic node 9.
/// Consistent with `e8_cartan()` / `e9_cartan()` and root vectors.
///
/// ```text
///     9 -- 8 -- 0 -- 1 -- 2 -- 3 -- 4 -- 5
///                                    |
///                                    6 -- 7
/// ```
pub fn e10_cartan() -> GeneralizedCartanMatrix {
    GeneralizedCartanMatrix::from_array([
        [2, -1, 0, 0, 0, 0, 0, 0, -1, 0],  // 0: connects to 1, 8
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],  // 1: connects to 0, 2
        [0, -1, 2, -1, 0, 0, 0, 0, 0, 0],  // 2: connects to 1, 3
        [0, 0, -1, 2, -1, 0, 0, 0, 0, 0],  // 3: connects to 2, 4
        [0, 0, 0, -1, 2, -1, -1, 0, 0, 0], // 4: connects to 3, 5, 6 (branch)
        [0, 0, 0, 0, -1, 2, 0, 0, 0, 0],   // 5: connects to 4
        [0, 0, 0, 0, -1, 0, 2, -1, 0, 0],  // 6: connects to 4, 7
        [0, 0, 0, 0, 0, 0, -1, 2, 0, 0],   // 7: connects to 6
        [-1, 0, 0, 0, 0, 0, 0, 0, 2, -1],  // 8 (affine): connects to 0, 9
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2],   // 9 (hyperbolic): connects to 8
    ])
    .expect("E10 Cartan matrix is valid")
}

/// E11 Cartan matrix (very extended E8).
///
/// E11 is proposed as a hidden symmetry of 11D supergravity (West 2001).
/// Contains E10 as a subalgebra.
///
/// Node numbering: E8 nodes 0-7, affine 8, hyperbolic 9, very extended 10.
///
/// ```text
///     10 -- 9 -- 8 -- 0 -- 1 -- 2 -- 3 -- 4 -- 5
///                                          |
///                                          6 -- 7
/// ```
pub fn e11_cartan() -> GeneralizedCartanMatrix {
    GeneralizedCartanMatrix::from_array([
        [2, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],  // 0: connects to 1, 8
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],  // 1: connects to 0, 2
        [0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0],  // 2: connects to 1, 3
        [0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0],  // 3: connects to 2, 4
        [0, 0, 0, -1, 2, -1, -1, 0, 0, 0, 0], // 4: connects to 3, 5, 6 (branch)
        [0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0],   // 5: connects to 4
        [0, 0, 0, 0, -1, 0, 2, -1, 0, 0, 0],  // 6: connects to 4, 7
        [0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0],   // 7: connects to 6
        [-1, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0],  // 8 (affine): connects to 0, 9
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1],  // 9 (hyperbolic): connects to 8, 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2],   // 10 (very extended): connects to 9
    ])
    .expect("E11 Cartan matrix is valid")
}

/// Create the A_n Cartan matrix (SL(n+1)).
pub fn a_n_cartan(n: usize) -> GeneralizedCartanMatrix {
    let mut entries = vec![vec![0; n]; n];
    for i in 0..n {
        entries[i][i] = 2;
        if i > 0 {
            entries[i][i - 1] = -1;
        }
        if i < n - 1 {
            entries[i][i + 1] = -1;
        }
    }
    GeneralizedCartanMatrix::new(entries).expect("A_n Cartan matrix is valid")
}

/// Create the D_n Cartan matrix (SO(2n)).
pub fn d_n_cartan(n: usize) -> GeneralizedCartanMatrix {
    assert!(n >= 4, "D_n requires n >= 4");
    let mut entries = vec![vec![0; n]; n];

    // Linear chain 0 - 1 - 2 - ... - (n-3) - (n-2)
    for (i, row) in entries.iter_mut().enumerate() {
        row[i] = 2;
    }
    for i in 0..(n - 2) {
        entries[i][i + 1] = -1;
        entries[i + 1][i] = -1;
    }

    // Branch: (n-3) connected to both (n-2) and (n-1)
    entries[n - 3][n - 1] = -1;
    entries[n - 1][n - 3] = -1;

    GeneralizedCartanMatrix::new(entries).expect("D_n Cartan matrix is valid")
}
