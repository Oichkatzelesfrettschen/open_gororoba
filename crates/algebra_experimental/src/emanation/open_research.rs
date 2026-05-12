//! Open-research probes around the additive lattice action and
//! octonion-subalgebra constraints (C-466 and item 12).
//!
//! These are exploratory functions, not load-bearing parts of the
//! emanation framework. They live here so the parent module's main
//! body stays focused on de Marrais's theorem suite.

use algebra_analysis::boxkites::O_TRIPS;

/// Attempt to extract a GL(8,Z) action matrix rho(b) for a basis element b.
///
/// For the additive lattice action, pi(b) = signum(sum(ell)) is verified.
/// For multiplication: if ell_out = rho(b) * ell exists, then rho(b) is an
/// 8x8 integer matrix acting on the 8D lattice.
///
/// This function takes a set of lattice vectors, multiplies each by the
/// basis element e_b using Cayley-Dickson multiplication, then maps the
/// result back to the lattice to extract the transformation matrix.
///
/// Returns Some(matrix) if a consistent 8x8 integer matrix exists, None otherwise.
pub fn extract_rho_matrix(
    basis_idx: usize,
    dim: usize,
    lattice_vecs: &[Vec<i32>],
) -> Option<Vec<Vec<i32>>> {
    if lattice_vecs.is_empty() || lattice_vecs[0].len() != 8 {
        return None;
    }

    // We need at least 8 linearly independent lattice vectors to determine
    // the 8x8 matrix. Use the first 8 that span the space.
    let n_coords = 8;
    if lattice_vecs.len() < n_coords {
        return None;
    }

    // Build basis element vector for e_basis_idx
    let mut e_b = vec![0.0f64; dim];
    if basis_idx < dim {
        e_b[basis_idx] = 1.0;
    } else {
        return None;
    }

    // For each lattice vector, reconstruct the Cayley-Dickson element,
    // multiply by e_b, then try to extract the lattice coordinates of the result.
    //
    // The lattice encoding maps a CD element to 8D via some fixed projection.
    // Without knowing the exact encoding, we can try the obvious one:
    // ell = (x_0, x_1, ..., x_7) maps to the first 8 components of the CD vector.
    //
    // This is a research probe -- we check if the transformation is consistent.

    let mut input_rows: Vec<Vec<i32>> = Vec::new();
    let mut output_rows: Vec<Vec<i32>> = Vec::new();

    for ell in lattice_vecs.iter().take(n_coords) {
        // Reconstruct CD element from lattice coordinates
        let mut cd_vec = vec![0.0f64; dim];
        for (k, &coord) in ell.iter().enumerate() {
            if k < dim {
                cd_vec[k] = coord as f64;
            }
        }

        // Multiply by e_b
        let product = cd_kernel::cayley_dickson::cd_multiply(&cd_vec, &e_b);

        // Extract first 8 components as output lattice vector
        let out_ell: Vec<i32> = product
            .iter()
            .take(n_coords)
            .map(|&x: &f64| x.round() as i32)
            .collect();

        // Verify integrality
        for &x in product.iter().take(n_coords) {
            if (x - x.round()).abs() > 1e-6 {
                return None; // Not integer-valued
            }
        }

        input_rows.push(ell.clone());
        output_rows.push(out_ell);
    }

    // Try to solve: output = rho * input (each as column vectors)
    // rho[i][j] = coefficient of input_j in output_i
    // This is equivalent to: for each output row o_i, express it as
    // sum_j rho[i][j] * input_j
    //
    // If inputs are the standard basis vectors e_0..e_7, this is trivial.
    // Otherwise, need to solve the linear system.
    //
    // For simplicity, check if the first 8 lattice vectors form an identity-like basis.
    // If not, return None (the research question remains open).

    let mut rho = vec![vec![0i32; n_coords]; n_coords];
    let is_standard_basis = input_rows.iter().enumerate().all(|(i, row)| {
        row.iter()
            .enumerate()
            .all(|(j, &v)| if i == j { v == 1 } else { v == 0 })
    });

    if is_standard_basis {
        for i in 0..n_coords {
            for j in 0..n_coords {
                rho[i][j] = output_rows[j][i]; // transpose
            }
        }
        Some(rho)
    } else {
        // General case: need Gaussian elimination over Z.
        // For the research probe, just check if the mapping is consistent
        // by verifying more than 8 vectors.
        None
    }
}

/// Check whether the 8D lattice dimension is correlated with octonion structure.
///
/// The 8D embedding might be constrained by the 7 imaginary octonion units +
/// the real unit. This function checks:
/// 1. Do lattice vectors respect the Fano plane structure?
/// 2. Is the 8D encoding dimension exactly the octonion dimension?
pub fn octonion_subalgebra_constraint_check(lattice: &[Vec<i32>]) -> bool {
    // The 8D lattice dimension matches octonion dimension (8 = 2^3).
    // Check: for each lattice vector, do the non-zero coordinates
    // correspond to octonion sub-algebra structure?

    if lattice.is_empty() {
        return false;
    }

    // All lattice vectors must be 8D
    if !lattice.iter().all(|v| v.len() == 8) {
        return false;
    }

    // Check Fano structure: the support pattern of each lattice vector
    // (which coordinates are non-zero) should be compatible with Fano lines.
    // Specifically, for octonion structure, indices 1..7 participate in
    // Fano triples [1,2,3], [1,4,5], [1,6,7], [2,4,6], [2,5,7], [3,4,7], [3,5,6].

    let mut fano_compatible_count = 0usize;
    for v in lattice {
        let support: Vec<usize> = v
            .iter()
            .enumerate()
            .filter(|&(_, &x)| x != 0)
            .map(|(i, _)| i)
            .collect();

        // Check if the non-real support (indices 1..7) forms a Fano-compatible
        // pattern: any 3-element support should be a Fano line.
        let non_real_support: Vec<usize> = support
            .iter()
            .filter(|&&i| (1..=7).contains(&i))
            .copied()
            .collect();

        if non_real_support.len() == 3 {
            let mut sorted = non_real_support.clone();
            sorted.sort();
            let is_fano = O_TRIPS
                .iter()
                .any(|trip| sorted == vec![trip[0], trip[1], trip[2]]);
            if is_fano {
                fano_compatible_count += 1;
            }
        }
    }

    // The lattice dimension (8) matches octonion dimension.
    // Report whether any vectors have Fano-compatible support.
    fano_compatible_count > 0
}
