//! Symmetry-Adapted TT-Cross for sharply peaked tensors.
//!
//! Implements the pivoting schemes based on translational and permutational
//! symmetry, specifically optimized for crystalline Configurational Integrals.
//!
//! # Literature
//! - Truong et al. (2025): Solving configurational integrals by tensor networks
//! - Oseledets & Tyrtyshnikov (2010): TT-cross approximation for multidimensional arrays

use crate::tt_train::{TTCore, TTTrain};
use ndarray::Array3;
use faer::Mat;

/// Build a rank-1 TT approximation using a single dominant pivot multi-index j.
///
/// Ref: Paper Section II.B, "Rank-1 symmetry-adapted scheme".
pub fn build_rank1_symmetry_adapted<F>(
    d: usize,
    n: usize,
    pivot: &[usize],
    mut f: F,
) -> TTTrain 
where F: FnMut(&[usize]) -> f64 {
    let f_pivot = f(pivot);
    let mut cores = Vec::with_capacity(d);
    
    for k in 0..d {
        // G_k(1, i_k, 1) = f(j_1, ..., i_k, ..., j_d)
        let mut core_data = Array3::<f64>::zeros((1, n, 1));
        let mut idx = pivot.to_vec();
        for i in 0..n {
            idx[k] = i;
            core_data[[0, i, 0]] = f(&idx);
        }
        cores.push(TTCore { data: core_data });
    }

    // Rank-1 intersection matrices are scalars 1 / f_pivot
    let mut intersection_matrices = Vec::with_capacity(d - 1);
    let inv_f = 1.0 / f_pivot;
    for _ in 0..(d - 1) {
        let mut m = Mat::<f64>::zeros(1, 1);
        m.write(0, 0, inv_f);
        intersection_matrices.push(m);
    }

    TTTrain { cores, intersection_matrices }
}

/// Build a rank-2 TT approximation using two super-diagonal anchors.
///
/// This implementation fixes the duplicated core line error identified in the audit.
/// Ref: Paper Section II.C, "Rank-2 symmetry-adapted scheme".
pub fn build_rank2_symmetry_adapted<F>(
    d: usize,
    n: usize,
    j1: &[usize],
    j2: &[usize],
    mut f: F,
) -> TTTrain
where F: FnMut(&[usize]) -> f64 {
    let mut cores = Vec::with_capacity(d);
    let mut intersection_matrices = Vec::with_capacity(d - 1);

    // Compute intersection matrices S_k and their inverses M_k
    for k in 0..(d - 1) {
        let mut s_k = Mat::<f64>::zeros(2, 2);
        
        // S_k[alpha, beta] = f( J_alpha(<k), j_beta(k), J_beta(>k) )
        // Actually simpler for rank-2 symmetry-adapted: 
        // S_k is the submatrix of the k-th unfolding.
        // We use the two multi-indices j1 and j2 as anchors.
        
        let idx11 = j1.to_vec();
        let mut idx12 = j1.to_vec();
        for m in k+1..d { idx12[m] = j2[m]; }
        
        let mut idx21 = j2.to_vec();
        for m in k+1..d { idx21[m] = j1[m]; }
        
        let idx22 = j2.to_vec();

        let v11 = f(&idx11);
        let v12 = f(&idx12);
        let v21 = f(&idx21);
        let v22 = f(&idx22);

        s_k.write(0, 0, v11);
        s_k.write(0, 1, v12);
        s_k.write(1, 0, v21);
        s_k.write(1, 1, v22);

        let det = v11 * v22 - v12 * v21;
        println!("DEBUG: S_{} values: [[{:.2e}, {:.2e}], [{:.2e}, {:.2e}]] det={:.2e}", k, v11, v12, v21, v22, det);
        let mut inv = Mat::<f64>::zeros(2, 2);
        if det.abs() > 1e-15 {
            inv.write(0, 0, v22 / det);
            inv.write(0, 1, -v12 / det);
            inv.write(1, 0, -v21 / det);
            inv.write(1, 1, v11 / det);
        }
        intersection_matrices.push(inv);
    }

    for k in 0..d {
        let mut core_data = if k == 0 {
            Array3::<f64>::zeros((1, n, 2))
        } else if k == d - 1 {
            Array3::<f64>::zeros((2, n, 1))
        } else {
            Array3::<f64>::zeros((2, n, 2))
        };

        let r_prev = core_data.shape()[0];
        let r_next = core_data.shape()[2];

        for i in 0..n {
            for alpha in 0..r_prev {
                for beta in 0..r_next {
                    // Corrected core selection logic based on nested CUR
                    let mut idx = if alpha == 0 { j1.to_vec() } else { j2.to_vec() };
                    idx[k] = i;
                    if beta == 1 && k < d - 1 {
                        for m in k+1..d { idx[m] = j2[m]; }
                    } else if beta == 0 && k < d - 1 {
                        for m in k+1..d { idx[m] = j1[m]; }
                    }
                    core_data[[alpha, i, beta]] = f(&idx);
                }
            }
        }
        cores.push(TTCore { data: core_data });
    }

    TTTrain { cores, intersection_matrices }
}
