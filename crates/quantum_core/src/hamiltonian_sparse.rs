//! Sparse Hamiltonian pilot for larger spin systems.
//!
//! This module keeps the existing dense Hamiltonian builder intact while
//! adding a feature-gated nalgebra-sparse representation for the same model.

use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex64;

fn index_to_coords(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(dims.len());
    for &dim in dims {
        coords.push(idx % dim);
        idx /= dim;
    }
    coords.reverse();
    coords
}

fn distance(i: usize, j: usize, dims: &[usize]) -> f64 {
    let coords_i = index_to_coords(i, dims);
    let coords_j = index_to_coords(j, dims);
    coords_i
        .iter()
        .zip(coords_j.iter())
        .map(|(lhs, rhs)| {
            let delta = (*lhs as f64) - (*rhs as f64);
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn spin_z(basis_state: usize, spin: usize, n_spins: usize) -> f64 {
    let bit = (basis_state >> (n_spins - 1 - spin)) & 1;
    if bit == 0 { 1.0 } else { -1.0 }
}

fn zz_diagonal_energy(basis_state: usize, dims: &[usize], alpha: f64, j: f64) -> f64 {
    let n_spins: usize = dims.iter().product();
    let mut energy = 0.0;
    for lhs in 0..n_spins {
        for rhs in (lhs + 1)..n_spins {
            let dist = distance(lhs, rhs, dims);
            if dist <= f64::EPSILON {
                continue;
            }
            let coupling = j / dist.powf(alpha);
            energy +=
                coupling * spin_z(basis_state, lhs, n_spins) * spin_z(basis_state, rhs, n_spins);
        }
    }
    energy
}

/// Build a sparse COO Hamiltonian for the long-range transverse-field Ising model.
pub fn build_sparse_hamiltonian_coo(
    dims: &[usize],
    alpha: f64,
    g: f64,
    j: f64,
) -> CooMatrix<Complex64> {
    let n_spins: usize = dims.iter().product();
    let hilbert_dim = 1usize << n_spins;
    let mut coo = CooMatrix::new(hilbert_dim, hilbert_dim);

    for basis_state in 0..hilbert_dim {
        let diag = zz_diagonal_energy(basis_state, dims, alpha, j);
        if diag != 0.0 {
            coo.push(basis_state, basis_state, Complex64::new(diag, 0.0));
        }

        for spin in 0..n_spins {
            let flipped = basis_state ^ (1usize << (n_spins - 1 - spin));
            coo.push(basis_state, flipped, Complex64::new(g, 0.0));
        }
    }

    coo
}

/// Build a sparse CSR Hamiltonian for iterative workflows and larger spin systems.
pub fn build_sparse_hamiltonian(
    dims: &[usize],
    alpha: f64,
    g: f64,
    j: f64,
) -> CsrMatrix<Complex64> {
    let coo = build_sparse_hamiltonian_coo(dims, alpha, g, j);
    CsrMatrix::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_builder_shapes_match_expected_hilbert_dimension() {
        let dims = vec![3];
        let sparse = build_sparse_hamiltonian(&dims, 2.0, 0.4, 1.2);
        let hilbert_dim = 1usize << dims.iter().product::<usize>();

        assert_eq!(sparse.nrows(), hilbert_dim);
        assert_eq!(sparse.ncols(), hilbert_dim);
        assert!(
            sparse.nnz() >= hilbert_dim,
            "sparse builder should at least encode diagonal terms"
        );
    }

    #[test]
    fn test_sparse_coo_contains_transverse_field_terms() {
        let dims = vec![2];
        let sparse = build_sparse_hamiltonian_coo(&dims, 2.0, 0.5, 1.0);
        let mut has_off_diagonal = false;
        for (row, col, value) in sparse.triplet_iter() {
            if row != col && value.re.abs() > 0.0 {
                has_off_diagonal = true;
                break;
            }
        }
        assert!(has_off_diagonal, "expected off-diagonal spin-flip entries");
    }
}
