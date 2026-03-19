use crate::codebook::LatticeVector;

/// Projects a 3D sky vector to an N-dimensional basis (e.g., 1024D) using 
/// Octonion Fano-plane incidence geometry (Claim C-458).
pub fn project_sky_to_basis(sky: &[f64; 3], lattice: &[LatticeVector], dimension: usize) -> Vec<f64> {
    let mut basis = vec![0.0; dimension];
    let o_trips: [[usize; 3]; 7] = [
        [1, 2, 3], [1, 4, 5], [1, 6, 7], [2, 4, 6], [2, 5, 7], [3, 4, 7], [3, 5, 6]
    ];
    for (idx, lattice_vector) in lattice.iter().take(dimension).enumerate() {
        let mut fano_dot = 0.0;
        for trip in o_trips.iter() {
            fano_dot += sky[0] * (lattice_vector[trip[0]] as f64)
                      + sky[1] * (lattice_vector[trip[1]] as f64)
                      + sky[2] * (lattice_vector[trip[2]] as f64);
        }
        // Incorporate l_0 for parity (real part)
        basis[idx] = fano_dot * (lattice_vector[0] as f64);
    }
    basis
}
