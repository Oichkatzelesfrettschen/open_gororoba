use wide::f64x4;

use super::arith::{cd_conjugate, cd_norm_sq};

#[inline]
fn cd_conjugate_simd(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 4 {
        return cd_conjugate(x);
    }

    let mut res = Vec::with_capacity(n);
    res.push(x[0]);

    let neg_mask = f64x4::from([-1.0, -1.0, -1.0, -1.0]);
    let mut i = 1;

    while i + 4 <= n {
        let chunk = f64x4::from([x[i], x[i + 1], x[i + 2], x[i + 3]]);
        let negated = chunk * neg_mask;
        res.extend_from_slice(&negated.to_array());
        i += 4;
    }

    while i < n {
        res.push(-x[i]);
        i += 1;
    }

    res
}

#[inline]
fn sub_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    if n < 4 {
        return a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    }

    let mut res = Vec::with_capacity(n);
    let mut i = 0;

    while i + 4 <= n {
        let va = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let vb = f64x4::from([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let diff = va - vb;
        res.extend_from_slice(&diff.to_array());
        i += 4;
    }

    while i < n {
        res.push(a[i] - b[i]);
        i += 1;
    }

    res
}

#[inline]
fn add_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    if n < 4 {
        return a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    }

    let mut res = Vec::with_capacity(n);
    let mut i = 0;

    while i + 4 <= n {
        let va = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let vb = f64x4::from([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let sum = va + vb;
        res.extend_from_slice(&sum.to_array());
        i += 4;
    }

    while i < n {
        res.push(a[i] + b[i]);
        i += 1;
    }

    res
}

/// SIMD-accelerated Cayley-Dickson multiplication.
pub fn cd_multiply_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    if dim == 2 {
        return vec![a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_c_r = if half >= 4 {
        cd_conjugate_simd(c_r)
    } else {
        cd_conjugate(c_r)
    };
    let conj_c_l = if half >= 4 {
        cd_conjugate_simd(c_l)
    } else {
        cd_conjugate(c_l)
    };

    let term1 = cd_multiply_simd(a_l, c_l);
    let term2 = cd_multiply_simd(&conj_c_r, a_r);
    let term3 = cd_multiply_simd(c_r, a_l);
    let term4 = cd_multiply_simd(a_r, &conj_c_l);

    let left = if half >= 4 {
        sub_simd(&term1, &term2)
    } else {
        term1.iter().zip(&term2).map(|(x, y)| x - y).collect()
    };

    let right = if half >= 4 {
        add_simd(&term3, &term4)
    } else {
        term3.iter().zip(&term4).map(|(x, y)| x + y).collect()
    };

    let mut result = Vec::with_capacity(dim);
    result.extend_from_slice(&left);
    result.extend_from_slice(&right);
    result
}

/// SIMD-accelerated squared norm.
#[inline]
pub fn cd_norm_sq_simd(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 4 {
        return cd_norm_sq(a);
    }

    let mut sum = f64x4::ZERO;
    let mut i = 0;

    while i + 4 <= n {
        let v = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        sum += v * v;
        i += 1 + 3;
    }

    let arr = sum.to_array();
    let mut total = arr[0] + arr[1] + arr[2] + arr[3];

    while i < n {
        total += a[i] * a[i];
        i += 1;
    }

    total
}
