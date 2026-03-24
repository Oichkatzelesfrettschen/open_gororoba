//! Tensor-Train (TT) data structures.
//!
//! Implements the compressed representation of a high-dimensional array
//! as a chain of 3-way cores.

use faer::Mat;
use ndarray::Array3;

/// A single core in the Tensor-Train.
/// Shape: (r_prev, n_k, r_next)
pub struct TTCore {
    pub data: Array3<f64>,
}

/// A Tensor-Train decomposition of a high-dimensional tensor.
pub struct TTTrain {
    pub cores: Vec<TTCore>,
    /// Intersection matrices M_k (size r_k x r_k)
    /// In standard TT, these are often absorbed into cores,
    /// but in TT-cross/CUR they are explicit inverses of submatrices.
    pub intersection_matrices: Vec<Mat<f64>>,
}

impl TTTrain {
    /// Evaluate the tensor at a specific multi-index (i_1, ..., i_d).
    pub fn get(&self, indices: &[usize]) -> f64 {
        let d = self.cores.len();
        assert_eq!(indices.len(), d);

        // Initial vector (1 x r_1) from first core
        let first_core = &self.cores[0].data;
        let mut res = Mat::<f64>::zeros(1, first_core.shape()[2]);
        for j in 0..first_core.shape()[2] {
            res.write(0, j, first_core[[0, indices[0], j]]);
        }

        for k in 1..d {
            // Apply intersection matrix M_{k-1}
            if let Some(m) = self.intersection_matrices.get(k - 1) {
                res = &res * m;
            }

            // Contract with next core G_k(:, i_k, :)
            let core = &self.cores[k].data;
            let r_prev = core.shape()[0];
            let r_next = core.shape()[2];
            let mut slice = Mat::<f64>::zeros(r_prev, r_next);
            for i in 0..r_prev {
                for j in 0..r_next {
                    slice.write(i, j, core[[i, indices[k], j]]);
                }
            }
            res = &res * slice;
        }

        res.read(0, 0)
    }
}
