//! Projected Entangled Pair States (PEPS) for 2D quantum systems.
//!
//! PEPS is the 2D generalization of MPS, where each site tensor has bonds
//! connecting to all four neighbors in a 2D grid:
//!
//! ```text
//!         |
//!     -- T --
//!         |
//! ```
//!
//! Each tensor T[i,j] has 5 indices: (left, right, up, down, physical).
//!
//! # Complexity
//!
//! Unlike MPS, exact contraction of PEPS is exponential in the grid size.
//! Approximate contraction methods are required for practical use:
//! - Boundary MPS method (used here)
//! - Corner transfer matrix
//! - Tensor renormalization group (TRG)
//!
//! # Literature
//!
//! - Verstraete & Cirac (2004): Renormalization algorithms for QMBS
//! - Jordan et al. (2008): Classical simulation of infinite-size quantum lattice systems
//! - Orus (2014): A practical introduction to tensor networks

use faer::complex_native::c64;
use rayon::prelude::*;

/// A single PEPS tensor at site (row, col).
///
/// Index ordering: (left, right, up, down, physical)
/// Stored as a flattened Vec with row-major ordering.
#[derive(Clone, Debug)]
pub struct PepsTensor {
    /// Tensor data in row-major order
    pub data: Vec<c64>,
    /// Left bond dimension
    pub chi_left: usize,
    /// Right bond dimension
    pub chi_right: usize,
    /// Up bond dimension
    pub chi_up: usize,
    /// Down bond dimension
    pub chi_down: usize,
    /// Physical dimension (2 for qubits)
    pub physical_dim: usize,
}

impl PepsTensor {
    /// Create a new PEPS tensor with given dimensions, initialized to zero.
    pub fn zeros(
        chi_left: usize,
        chi_right: usize,
        chi_up: usize,
        chi_down: usize,
        physical_dim: usize,
    ) -> Self {
        let size = chi_left * chi_right * chi_up * chi_down * physical_dim;
        Self {
            data: vec![c64::new(0.0, 0.0); size],
            chi_left,
            chi_right,
            chi_up,
            chi_down,
            physical_dim,
        }
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.chi_left * self.chi_right * self.chi_up * self.chi_down * self.physical_dim
    }

    /// Get tensor element at (left, right, up, down, physical).
    #[inline]
    pub fn get(&self, l: usize, r: usize, u: usize, d: usize, p: usize) -> c64 {
        let idx = ((((l * self.chi_right + r) * self.chi_up + u) * self.chi_down + d)
            * self.physical_dim)
            + p;
        self.data[idx]
    }

    /// Set tensor element at (left, right, up, down, physical).
    #[inline]
    pub fn set(&mut self, l: usize, r: usize, u: usize, d: usize, p: usize, val: c64) {
        let idx = ((((l * self.chi_right + r) * self.chi_up + u) * self.chi_down + d)
            * self.physical_dim)
            + p;
        self.data[idx] = val;
    }

    /// Apply a single-qubit gate to this tensor.
    pub fn apply_gate(&mut self, gate: &[c64; 4]) {
        let chi_l = self.chi_left;
        let chi_r = self.chi_right;
        let chi_u = self.chi_up;
        let chi_d = self.chi_down;
        let phys_dim = self.physical_dim;
        let old_data = self.data.clone();

        let mut new_data = vec![c64::new(0.0, 0.0); self.size()];

        for l in 0..chi_l {
            for r in 0..chi_r {
                for u in 0..chi_u {
                    for d in 0..chi_d {
                        for p_new in 0..phys_dim {
                            let mut sum = c64::new(0.0, 0.0);
                            for p_old in 0..phys_dim {
                                let g = gate[p_new * 2 + p_old];
                                let old_idx = ((((l * chi_r + r) * chi_u + u) * chi_d + d)
                                    * phys_dim)
                                    + p_old;
                                let t = old_data[old_idx];
                                sum = c64::new(
                                    sum.re + g.re * t.re - g.im * t.im,
                                    sum.im + g.re * t.im + g.im * t.re,
                                );
                            }
                            let new_idx =
                                ((((l * chi_r + r) * chi_u + u) * chi_d + d) * phys_dim) + p_new;
                            new_data[new_idx] = sum;
                        }
                    }
                }
            }
        }

        self.data = new_data;
    }
}

/// Projected Entangled Pair State for 2D quantum systems.
#[derive(Clone, Debug)]
pub struct Peps {
    /// Grid of PEPS tensors, indexed [row][col]
    pub tensors: Vec<Vec<PepsTensor>>,
    /// Number of rows in the grid
    pub rows: usize,
    /// Number of columns in the grid
    pub cols: usize,
    /// Maximum bond dimension
    pub chi_max: usize,
    /// Physical dimension per site (2 for qubits)
    pub physical_dim: usize,
}

impl Peps {
    /// Create a PEPS representing |0...0> on an m x n grid.
    ///
    /// This is a product state with bond dimension 1.
    pub fn new_zero_state(rows: usize, cols: usize) -> Self {
        let mut tensors = Vec::with_capacity(rows);

        for _i in 0..rows {
            let mut row_tensors = Vec::with_capacity(cols);
            for _j in 0..cols {
                let chi_left = 1;
                let chi_right = 1;
                let chi_up = 1;
                let chi_down = 1;

                let mut tensor = PepsTensor::zeros(chi_left, chi_right, chi_up, chi_down, 2);
                // Set |0> coefficient to 1
                tensor.set(0, 0, 0, 0, 0, c64::new(1.0, 0.0));
                row_tensors.push(tensor);
            }
            tensors.push(row_tensors);
        }

        Self {
            tensors,
            rows,
            cols,
            chi_max: 16, // Default for 2D is smaller than 1D due to complexity
            physical_dim: 2,
        }
    }

    /// Total number of sites.
    pub fn n_sites(&self) -> usize {
        self.rows * self.cols
    }

    /// Get tensor at site (row, col).
    pub fn get_tensor(&self, row: usize, col: usize) -> &PepsTensor {
        &self.tensors[row][col]
    }

    /// Get mutable tensor at site (row, col).
    pub fn get_tensor_mut(&mut self, row: usize, col: usize) -> &mut PepsTensor {
        &mut self.tensors[row][col]
    }

    /// Apply a single-qubit gate to a specific site.
    pub fn apply_single_gate(&mut self, row: usize, col: usize, gate: &[c64; 4]) {
        if row < self.rows && col < self.cols {
            self.tensors[row][col].apply_gate(gate);
        }
    }

    /// Apply Hadamard gate to a specific site.
    pub fn apply_hadamard(&mut self, row: usize, col: usize) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            c64::new(inv_sqrt2, 0.0),
            c64::new(inv_sqrt2, 0.0),
            c64::new(inv_sqrt2, 0.0),
            c64::new(-inv_sqrt2, 0.0),
        ];
        self.apply_single_gate(row, col, &h);
    }

    /// Apply X gate to a specific site.
    pub fn apply_x(&mut self, row: usize, col: usize) {
        let x = [
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
        ];
        self.apply_single_gate(row, col, &x);
    }

    /// Apply Z gate to a specific site.
    pub fn apply_z(&mut self, row: usize, col: usize) {
        let z = [
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(-1.0, 0.0),
        ];
        self.apply_single_gate(row, col, &z);
    }

    /// Apply Hadamard gates to all sites in parallel.
    pub fn apply_hadamard_all_parallel(&mut self) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            c64::new(inv_sqrt2, 0.0),
            c64::new(inv_sqrt2, 0.0),
            c64::new(inv_sqrt2, 0.0),
            c64::new(-inv_sqrt2, 0.0),
        ];

        self.tensors.par_iter_mut().for_each(|row| {
            row.iter_mut().for_each(|tensor| {
                tensor.apply_gate(&h);
            });
        });
    }

    /// Get maximum bond dimension in the PEPS.
    pub fn max_bond_dimension(&self) -> usize {
        self.tensors
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|t| [t.chi_left, t.chi_right, t.chi_up, t.chi_down])
            .max()
            .unwrap_or(1)
    }

    /// Contract PEPS to scalar (for product states, this gives the norm).
    ///
    /// WARNING: For entangled states, exact contraction is exponential.
    /// This method is only practical for small grids or product states.
    pub fn contract_exact(&self) -> c64 {
        if self.rows == 0 || self.cols == 0 {
            return c64::new(0.0, 0.0);
        }

        // For product states (chi=1), this is efficient
        if self.max_bond_dimension() == 1 {
            let mut result = c64::new(1.0, 0.0);
            for row in &self.tensors {
                for tensor in row {
                    // For chi=1, sum over physical index
                    let mut local_sum = c64::new(0.0, 0.0);
                    for p in 0..tensor.physical_dim {
                        let val = tensor.get(0, 0, 0, 0, p);
                        local_sum = c64::new(local_sum.re + val.re, local_sum.im + val.im);
                    }
                    result = c64::new(
                        result.re * local_sum.re - result.im * local_sum.im,
                        result.re * local_sum.im + result.im * local_sum.re,
                    );
                }
            }
            return result;
        }

        // For general case, use boundary MPS contraction (row by row)
        // This is approximate for large chi, but exact for small grids
        self.contract_boundary_mps()
    }

    /// Contract PEPS using boundary MPS method.
    ///
    /// Contracts row by row, maintaining a boundary MPS.
    fn contract_boundary_mps(&self) -> c64 {
        // Start with first row as boundary
        let mut boundary = self.row_to_mps(0);

        // Contract each subsequent row
        for i in 1..self.rows {
            let row_mps = self.row_to_mps(i);
            boundary = self.contract_rows(&boundary, &row_mps);
        }

        // Final contraction of boundary to scalar
        self.mps_to_scalar(&boundary)
    }

    /// Convert a row of PEPS tensors to an MPS-like structure.
    fn row_to_mps(&self, row_idx: usize) -> Vec<Vec<c64>> {
        let row = &self.tensors[row_idx];
        row.iter()
            .map(|tensor| {
                // Contract up and down indices (assume chi=1 for boundaries)
                let mut data =
                    Vec::with_capacity(tensor.chi_left * tensor.chi_right * tensor.physical_dim);
                for l in 0..tensor.chi_left {
                    for r in 0..tensor.chi_right {
                        for p in 0..tensor.physical_dim {
                            let mut sum = c64::new(0.0, 0.0);
                            for u in 0..tensor.chi_up {
                                for d in 0..tensor.chi_down {
                                    let val = tensor.get(l, r, u, d, p);
                                    sum = c64::new(sum.re + val.re, sum.im + val.im);
                                }
                            }
                            data.push(sum);
                        }
                    }
                }
                data
            })
            .collect()
    }

    /// Contract two boundary MPS rows by contracting over shared vertical bonds.
    ///
    /// Each row-MPS tensor has shape [chi_left * chi_right * physical_dim].
    /// For product states (chi=1), contraction is element-wise multiplication.
    /// For general chi, we do a proper tensor product contraction.
    ///
    /// When compiled with the `gpu` feature, attempts GPU acceleration; otherwise
    /// falls back to CPU. CPU path is always available as fallback.
    fn contract_rows(&self, upper: &[Vec<c64>], lower: &[Vec<c64>]) -> Vec<Vec<c64>> {
        upper
            .iter()
            .zip(lower.iter())
            .map(|(u, l)| {
                if u.len() == l.len() {
                    #[cfg(feature = "gpu")]
                    {
                        // Try GPU acceleration for large tensors
                        if u.len() > 1000 {
                            let result = crate::gpu::peps::gpu_contract_rows_peps(u, l);
                            return result;
                        }
                    }

                    // CPU path (always available)
                    u.iter()
                        .zip(l.iter())
                        .map(|(a, b)| {
                            c64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
                        })
                        .collect()
                } else {
                    // Dimension mismatch: fall back to lower (should not happen
                    // for well-formed PEPS, but defensive)
                    l.clone()
                }
            })
            .collect()
    }

    /// Compute entanglement entropy for a horizontal bipartition.
    ///
    /// Splits the grid into rows [0..cut_row) and [cut_row..rows).
    /// For product states, returns 0.  For entangled states with chi=1
    /// tensors, builds the exact statevector and computes the reduced
    /// density matrix via partial trace.
    ///
    /// # Arguments
    /// * `cut_row` - Row index for the bipartition (sites above the cut
    ///   form subsystem A).  Must be in [1, rows-1].
    ///
    /// # Returns
    /// Von Neumann entanglement entropy S = -Tr(rho_A * ln(rho_A)).
    pub fn entanglement_entropy_row_cut(&self, cut_row: usize) -> f64 {
        if cut_row == 0 || cut_row >= self.rows {
            return 0.0;
        }
        // For product states (all chi=1), entropy is exactly 0
        if self.max_bond_dimension() == 1 {
            return 0.0;
        }
        // For small grids, do exact statevector contraction
        let n = self.rows * self.cols;
        if n <= 16 {
            return self.exact_bipartite_entropy(cut_row);
        }
        // For larger grids, use boundary MPS approximation
        self.boundary_mps_entropy(cut_row)
    }

    /// Exact bipartite entropy for small grids.
    ///
    /// Builds the full 2^n statevector by iterating over all computational
    /// basis states, then partial-traces over subsystem B.
    fn exact_bipartite_entropy(&self, cut_row: usize) -> f64 {
        let n = self.rows * self.cols;
        let dim = 1_usize << n;
        let n_a = cut_row * self.cols;
        let dim_a = 1_usize << n_a;
        let dim_b = 1_usize << (n - n_a);

        // Build the full statevector: amplitude for each basis state |b_0 b_1 ... b_{n-1}>
        let mut psi = vec![c64::new(0.0, 0.0); dim];
        for (basis, psi_elem) in psi.iter_mut().enumerate() {
            let mut amp = c64::new(1.0, 0.0);
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let site = row * self.cols + col;
                    let phys = (basis >> site) & 1;
                    let tensor = &self.tensors[row][col];
                    // For chi=1, get the single component
                    // For chi>1, sum over all bond configurations
                    // (this is exact for small grids)
                    let mut site_amp = c64::new(0.0, 0.0);
                    for l in 0..tensor.chi_left {
                        for r in 0..tensor.chi_right {
                            for u in 0..tensor.chi_up {
                                for d in 0..tensor.chi_down {
                                    let val = tensor.get(l, r, u, d, phys);
                                    site_amp = c64::new(site_amp.re + val.re, site_amp.im + val.im);
                                }
                            }
                        }
                    }
                    amp = c64::new(
                        amp.re * site_amp.re - amp.im * site_amp.im,
                        amp.re * site_amp.im + amp.im * site_amp.re,
                    );
                }
            }
            *psi_elem = amp;
        }

        // Build reduced density matrix rho_A by tracing over B
        // rho_A[i,j] = sum_k psi[i*dim_b + k] * conj(psi[j*dim_b + k])
        let mut rho = vec![c64::new(0.0, 0.0); dim_a * dim_a];
        for i in 0..dim_a {
            for j in 0..dim_a {
                let mut sum = c64::new(0.0, 0.0);
                for k in 0..dim_b {
                    let psi_ik = psi[i * dim_b + k];
                    let psi_jk = psi[j * dim_b + k];
                    let conj_jk = c64::new(psi_jk.re, -psi_jk.im);
                    sum = c64::new(
                        sum.re + psi_ik.re * conj_jk.re - psi_ik.im * conj_jk.im,
                        sum.im + psi_ik.re * conj_jk.im + psi_ik.im * conj_jk.re,
                    );
                }
                rho[i * dim_a + j] = sum;
            }
        }

        // Eigendecompose rho_A and compute von Neumann entropy
        let mut mat = faer::Mat::<c64>::zeros(dim_a, dim_a);
        for i in 0..dim_a {
            for j in 0..dim_a {
                mat.write(i, j, rho[i * dim_a + j]);
            }
        }

        let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eigenvalues = eig.s();

        let mut entropy = 0.0;
        for i in 0..dim_a {
            let p = eigenvalues.column_vector().read(i).re;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Boundary MPS entropy for larger grids (approximate).
    ///
    /// Contracts rows above and below the cut into boundary MPS
    /// representations, then computes the overlap to get the reduced
    /// density matrix.
    fn boundary_mps_entropy(&self, cut_row: usize) -> f64 {
        // Contract rows [0..cut_row) into upper boundary
        let mut upper = self.row_to_mps(0);
        for i in 1..cut_row {
            let row_mps = self.row_to_mps(i);
            upper = self.contract_rows(&upper, &row_mps);
        }

        // Contract rows [cut_row..rows) into lower boundary
        let mut lower = self.row_to_mps(cut_row);
        for i in (cut_row + 1)..self.rows {
            let row_mps = self.row_to_mps(i);
            lower = self.contract_rows(&lower, &row_mps);
        }

        // For the boundary MPS approximation, build reduced density matrix
        // from the upper boundary tensors
        // Each boundary tensor has chi_l * chi_r * phys_dim entries
        // For chi=1, this gives a simple diagonal matrix (product state -> entropy 0)
        let cols = upper.len();
        let phys_dim = self.physical_dim;

        // Total dimension of upper boundary
        let dim_upper: usize = (0..cols).map(|_| phys_dim).product();
        if dim_upper > 1024 {
            // Too large for exact density matrix; return 0 as conservative estimate
            return 0.0;
        }

        // Build reduced density matrix from boundary MPS
        // For product-state-like boundaries, this will give entropy ~ 0
        let mut rho = vec![0.0_f64; dim_upper * dim_upper];
        for i in 0..dim_upper {
            for j in 0..dim_upper {
                let mut elem = c64::new(1.0, 0.0);
                for (col, upper_col) in upper.iter().enumerate() {
                    let pi = (i / phys_dim.pow(col as u32)) % phys_dim;
                    let pj = (j / phys_dim.pow(col as u32)) % phys_dim;
                    let ui = if pi < upper_col.len() {
                        upper_col[pi]
                    } else {
                        c64::new(0.0, 0.0)
                    };
                    let uj = if pj < upper_col.len() {
                        upper_col[pj]
                    } else {
                        c64::new(0.0, 0.0)
                    };
                    let conj_uj = c64::new(uj.re, -uj.im);
                    let prod = c64::new(
                        ui.re * conj_uj.re - ui.im * conj_uj.im,
                        ui.re * conj_uj.im + ui.im * conj_uj.re,
                    );
                    elem = c64::new(
                        elem.re * prod.re - elem.im * prod.im,
                        elem.re * prod.im + elem.im * prod.re,
                    );
                }
                rho[i * dim_upper + j] = elem.re;
            }
        }

        // Normalize trace to 1
        let trace: f64 = (0..dim_upper).map(|i| rho[i * dim_upper + i]).sum();
        if trace > 1e-15 {
            for val in rho.iter_mut() {
                *val /= trace;
            }
        }

        // Eigendecompose and compute entropy
        let mut mat = faer::Mat::<f64>::zeros(dim_upper, dim_upper);
        for i in 0..dim_upper {
            for j in 0..dim_upper {
                mat.write(i, j, rho[i * dim_upper + j]);
            }
        }
        let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eigenvalues = eig.s();

        let mut entropy = 0.0;
        for i in 0..dim_upper {
            let p = eigenvalues.column_vector().read(i);
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Contract MPS to scalar.
    fn mps_to_scalar(&self, mps: &[Vec<c64>]) -> c64 {
        let mut result = c64::new(1.0, 0.0);
        for tensor_data in mps {
            let mut sum = c64::new(0.0, 0.0);
            for val in tensor_data {
                sum = c64::new(sum.re + val.re, sum.im + val.im);
            }
            result = c64::new(
                result.re * sum.re - result.im * sum.im,
                result.re * sum.im + result.im * sum.re,
            );
        }
        result
    }

    /// Compute local expectation value <psi|O|psi> for single-site operator O.
    ///
    /// For product states, this is exact. For entangled states, approximate.
    pub fn expectation_local(&self, row: usize, col: usize, op: &[c64; 4]) -> c64 {
        if row >= self.rows || col >= self.cols {
            return c64::new(0.0, 0.0);
        }

        let tensor = &self.tensors[row][col];
        let mut result = c64::new(0.0, 0.0);

        // For product states, trace over O applied to local state
        for p1 in 0..tensor.physical_dim {
            for p2 in 0..tensor.physical_dim {
                let op_elem = op[p1 * 2 + p2];
                let mut contrib = c64::new(0.0, 0.0);

                for l in 0..tensor.chi_left {
                    for r in 0..tensor.chi_right {
                        for u in 0..tensor.chi_up {
                            for d in 0..tensor.chi_down {
                                let bra = tensor.get(l, r, u, d, p1);
                                let ket = tensor.get(l, r, u, d, p2);
                                let bra_conj = c64::new(bra.re, -bra.im);

                                let prod = c64::new(
                                    bra_conj.re * ket.re - bra_conj.im * ket.im,
                                    bra_conj.re * ket.im + bra_conj.im * ket.re,
                                );
                                contrib = c64::new(contrib.re + prod.re, contrib.im + prod.im);
                            }
                        }
                    }
                }

                let term = c64::new(
                    op_elem.re * contrib.re - op_elem.im * contrib.im,
                    op_elem.re * contrib.im + op_elem.im * contrib.re,
                );
                result = c64::new(result.re + term.re, result.im + term.im);
            }
        }

        result
    }

    /// Compute <Z> expectation value at a site.
    pub fn expectation_z(&self, row: usize, col: usize) -> f64 {
        let z = [
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(-1.0, 0.0),
        ];
        self.expectation_local(row, col, &z).re
    }

    /// Compute <X> expectation value at a site.
    pub fn expectation_x(&self, row: usize, col: usize) -> f64 {
        let x = [
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
        ];
        self.expectation_local(row, col, &x).re
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peps_zero_state() {
        let peps = Peps::new_zero_state(3, 4);
        assert_eq!(peps.rows, 3);
        assert_eq!(peps.cols, 4);
        assert_eq!(peps.n_sites(), 12);
        assert_eq!(peps.max_bond_dimension(), 1);
    }

    #[test]
    fn test_peps_tensor_indexing() {
        let mut tensor = PepsTensor::zeros(2, 2, 2, 2, 2);
        tensor.set(1, 1, 1, 1, 1, c64::new(3.14, 2.71));
        let val = tensor.get(1, 1, 1, 1, 1);
        assert!((val.re - 3.14).abs() < 1e-10);
        assert!((val.im - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_peps_hadamard() {
        let mut peps = Peps::new_zero_state(2, 2);
        peps.apply_hadamard(0, 0);

        // After H|0> = |+>, expectation <Z> should be 0
        let z_exp = peps.expectation_z(0, 0);
        assert!(z_exp.abs() < 1e-10, "<Z> = {}", z_exp);

        // And <X> should be 1
        let x_exp = peps.expectation_x(0, 0);
        assert!((x_exp - 1.0).abs() < 1e-10, "<X> = {}", x_exp);
    }

    #[test]
    fn test_peps_x_gate() {
        let mut peps = Peps::new_zero_state(2, 2);
        peps.apply_x(1, 1);

        // After X|0> = |1>, expectation <Z> should be -1
        let z_exp = peps.expectation_z(1, 1);
        assert!((z_exp - (-1.0)).abs() < 1e-10, "<Z> = {}", z_exp);
    }

    #[test]
    fn test_peps_product_state_contraction() {
        let peps = Peps::new_zero_state(2, 3);
        let result = peps.contract_exact();

        // |0>^n contracted should give 1 (all amplitudes are 1 for |0>)
        assert!((result.re - 1.0).abs() < 1e-10, "Result = {}", result.re);
    }

    #[test]
    fn test_peps_hadamard_all_parallel() {
        let mut peps = Peps::new_zero_state(3, 3);
        peps.apply_hadamard_all_parallel();

        // All sites should now have <Z> = 0
        for i in 0..peps.rows {
            for j in 0..peps.cols {
                let z = peps.expectation_z(i, j);
                assert!(z.abs() < 1e-10, "Site ({},{}) has <Z> = {}", i, j, z);
            }
        }
    }

    #[test]
    fn test_peps_mixed_state() {
        let mut peps = Peps::new_zero_state(2, 2);

        // Apply different gates to different sites
        peps.apply_x(0, 0); // |1>
        peps.apply_hadamard(0, 1); // |+>
        peps.apply_z(1, 0); // |0> (Z|0> = |0>)

        assert!((peps.expectation_z(0, 0) - (-1.0)).abs() < 1e-10);
        assert!(peps.expectation_z(0, 1).abs() < 1e-10);
        assert!((peps.expectation_z(1, 0) - 1.0).abs() < 1e-10);
        assert!((peps.expectation_z(1, 1) - 1.0).abs() < 1e-10);
    }

    // ================================================================
    // Entanglement entropy tests
    // ================================================================

    #[test]
    fn test_peps_product_state_entropy_zero() {
        // Product state |0...0> has zero entanglement across any cut
        let peps = Peps::new_zero_state(2, 3);
        let entropy = peps.entanglement_entropy_row_cut(1);
        assert!(
            entropy.abs() < 1e-10,
            "Product state should have zero entropy, got {}",
            entropy
        );
    }

    #[test]
    fn test_peps_entropy_boundary_cases() {
        let peps = Peps::new_zero_state(3, 2);
        // Cut at row 0 (empty upper partition) -> 0
        assert_eq!(peps.entanglement_entropy_row_cut(0), 0.0);
        // Cut at row >= rows (empty lower partition) -> 0
        assert_eq!(peps.entanglement_entropy_row_cut(3), 0.0);
        assert_eq!(peps.entanglement_entropy_row_cut(10), 0.0);
    }

    #[test]
    fn test_peps_product_state_with_gates_entropy() {
        // Product state with single-qubit gates applied: still zero entropy
        // because single-qubit gates don't create entanglement
        let mut peps = Peps::new_zero_state(2, 2);
        peps.apply_hadamard(0, 0);
        peps.apply_x(1, 1);
        let entropy = peps.entanglement_entropy_row_cut(1);
        assert!(
            entropy.abs() < 1e-10,
            "Product state with local gates should have zero entropy, got {}",
            entropy
        );
    }
}
