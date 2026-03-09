use crate::construction::cayley_dickson::cd_basis_mul_sign;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use super::TensorAVT;

pub(crate) fn tensor_avt_cpu_sign_table(dim: usize) -> Arc<Vec<i8>> {
    static TABLES: OnceLock<Mutex<HashMap<usize, Arc<Vec<i8>>>>> = OnceLock::new();
    let tables = TABLES.get_or_init(|| Mutex::new(HashMap::new()));

    {
        let tables = tables.lock().expect("tensor_avt sign cache poisoned");
        if let Some(table) = tables.get(&dim) {
            return table.clone();
        }
    }

    let mut table = vec![0i8; dim * dim];
    for i in 0..dim {
        let row = &mut table[i * dim..(i + 1) * dim];
        for (j, cell) in row.iter_mut().enumerate() {
            let src = i ^ j;
            *cell = cd_basis_mul_sign(dim, src, j) as i8;
        }
    }
    let table = Arc::new(table);

    let mut tables = tables.lock().expect("tensor_avt sign cache poisoned");
    tables.entry(dim).or_insert_with(|| table.clone()).clone()
}

#[derive(Clone)]
pub(crate) struct TensorAvtCpuMulSession {
    pub(crate) left: Vec<f32>,
    pub(crate) right: Vec<f32>,
    pub(crate) output: Vec<f32>,
}

impl TensorAvtCpuMulSession {
    pub(crate) fn new(dim: usize, max_batch_size: usize) -> Self {
        Self {
            left: vec![0.0; dim],
            right: vec![0.0; dim * max_batch_size],
            output: vec![0.0; dim * max_batch_size],
        }
    }
}

#[derive(Clone)]
pub(crate) struct TensorAvtCpuNormSession {
    pub(crate) vectors: Vec<f32>,
    pub(crate) norms: Vec<f32>,
}

impl TensorAvtCpuNormSession {
    pub(crate) fn new(dim: usize, max_vectors: usize) -> Self {
        Self {
            vectors: vec![0.0; dim * max_vectors],
            norms: vec![0.0; max_vectors],
        }
    }
}

impl TensorAVT {
    pub(crate) fn compute_cd_mul_cpu(&self, a: &[f32], x: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(x.len(), self.dim, "x must have dim elements");
        let signs = tensor_avt_cpu_sign_table(self.dim);
        let mut y = vec![0.0f32; self.dim];
        for (i, yi) in y.iter_mut().enumerate() {
            let sign_row = &signs[i * self.dim..(i + 1) * self.dim];
            for (j, xj) in x.iter().enumerate() {
                let src = i ^ j;
                *yi += a[src] * (sign_row[j] as f32) * xj;
            }
        }
        Ok(y)
    }

    pub(crate) fn compute_cd_mul_batch_cpu(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a.len(), self.dim, "a must have dim elements");
        assert_eq!(
            x_batch.len(),
            batch_size * self.dim,
            "x_batch must have batch_size * dim elements"
        );
        let signs = tensor_avt_cpu_sign_table(self.dim);
        let mut y_batch = vec![0.0f32; batch_size * self.dim];
        for b in 0..batch_size {
            let x_offset = b * self.dim;
            let y_offset = b * self.dim;
            for (i, yi) in y_batch[y_offset..y_offset + self.dim]
                .iter_mut()
                .enumerate()
            {
                let sign_row = &signs[i * self.dim..(i + 1) * self.dim];
                for j in 0..self.dim {
                    let src = i ^ j;
                    *yi += a[src] * (sign_row[j] as f32) * x_batch[x_offset + j];
                }
            }
        }
        Ok(y_batch)
    }

    pub(crate) fn compute_norm_sq_batch_cpu(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(
            vectors.len(),
            n_vectors * self.dim,
            "vectors length must be n_vectors * dim"
        );
        let mut norms = Vec::with_capacity(n_vectors);
        for v_idx in 0..n_vectors {
            let start = v_idx * self.dim;
            let end = start + self.dim;
            let norm_sq: f32 = vectors[start..end].iter().map(|x| x * x).sum();
            norms.push(norm_sq);
        }
        Ok(norms)
    }
}
