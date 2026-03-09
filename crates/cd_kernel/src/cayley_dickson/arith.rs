/// Cayley-Dickson conjugation: negate all imaginary components.
/// x* = (x0, -x1, -x2, ..., -x_{n-1})
#[inline]
pub fn cd_conjugate(x: &[f64]) -> Vec<f64> {
    let mut res = x.to_vec();
    for v in res[1..].iter_mut() {
        *v = -*v;
    }
    res
}

/// Cayley-Dickson multiplication via the doubling formula.
///
/// (a,b)(c,d) = (ac - d*b, da + bc*)
///
/// where `*` denotes conjugation. Recursion bottoms out at dim=1
/// (scalar multiplication).
pub fn cd_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_c_r = cd_conjugate(c_r);
    let term1 = cd_multiply(a_l, c_l);
    let term2 = cd_multiply(&conj_c_r, a_r);

    let conj_c_l = cd_conjugate(c_l);
    let term3 = cd_multiply(c_r, a_l);
    let term4 = cd_multiply(a_r, &conj_c_l);

    let mut result = Vec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] - term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

/// Cayley-Dickson multiplication with conjugated doubling (chirality inversion).
pub fn cd_multiply_conjugated(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_a_r = cd_conjugate(a_r);
    let term1 = cd_multiply_conjugated(a_l, c_l);
    let term2 = cd_multiply_conjugated(c_r, &conj_a_r);

    let term3 = cd_multiply_conjugated(&conj_a_r, c_l);
    let term4 = cd_multiply_conjugated(c_r, a_l);

    let mut result = Vec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] - term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

/// Squared Euclidean norm: sum of squares of all components.
#[inline]
pub fn cd_norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

/// Check if two slices are element-wise close within tolerance.
#[inline]
pub(crate) fn allclose(a: &[f64], b: &[f64], atol: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= atol)
}

/// Non-allocating Cayley-Dickson multiplication using a pre-allocated workspace.
pub fn cd_multiply_into(a: &[f64], b: &[f64], res: &mut [f64], workspace: &mut [f64]) {
    let dim = a.len();
    assert_eq!(b.len(), dim);
    assert_eq!(res.len(), dim);
    assert!(workspace.len() >= dim * 2);

    if dim == 1 {
        res[0] = a[0] * b[0];
        return;
    }

    let tmp = cd_multiply(a, b);
    res.copy_from_slice(&tmp);
}

/// In-place Cayley-Dickson multiplication.
pub fn cd_multiply_mut(a: &mut Vec<f64>, b: &[f64]) {
    let dim = a.len();
    let mut res = vec![0.0; dim];
    let mut workspace = vec![0.0; dim * 2];
    cd_multiply_into(a, b, &mut res, &mut workspace);
    *a = res;
}

/// Construct the left-multiplication matrix L_a where L_a[i][j] = (a * e_j)[i].
pub fn left_mult_operator(a: &[f64], dim: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), dim);
    let mut matrix = vec![0.0; dim * dim];
    let mut basis = vec![0.0; dim];
    for j in 0..dim {
        if j > 0 {
            basis[j - 1] = 0.0;
        }
        basis[j] = 1.0;
        let col = cd_multiply(a, &basis);
        for i in 0..dim {
            matrix[i * dim + j] = col[i];
        }
    }
    matrix
}
