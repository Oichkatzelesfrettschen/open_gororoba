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

/// Flat quaternion (4D CD) multiply using f64x4 -- zero heap allocation.
///
/// The CD quaternion product `p = q * r` can be expressed as the matrix-vector
/// product `p = M(q) * r` where:
/// ```text
/// M(q) = [[q0, -q1, -q2, -q3],
///          [q1,  q0, -q3,  q2],
///          [q2,  q3,  q0, -q1],
///          [q3, -q2,  q1,  q0]]
/// ```
/// This decomposes into 4 broadcast-scale-accumulate operations on shuffled
/// copies of `r`, fitting naturally into 256-bit SIMD (f64x4 = one quaternion).
///
/// # Why this matters for the CD tower
/// An AVX2 `__m256d` register holds exactly 4 f64 = one quaternion.  Two
/// registers = one octonion.  Four = one sedenion.  The CD doubling formula
/// `(a,b)(c,d) = (ac - d*b, da + bc*)` then chains these flat multiplies
/// without recursive Vec allocation.
#[inline]
pub fn quaternion_multiply_flat(q: &[f64; 4], r: &[f64; 4]) -> [f64; 4] {
    let vr = f64x4::from(*r);

    // Broadcast each quaternion component
    let q0 = f64x4::from([q[0]; 4]);
    let q1 = f64x4::from([q[1]; 4]);
    let q2 = f64x4::from([q[2]; 4]);
    let q3 = f64x4::from([q[3]; 4]);

    // Shuffled/negated copies of r for column decomposition of M(q).
    // Derived from standard CD: p0 = q0r0 - q1r1 - q2r2 - q3r3, etc.
    let r_col0 = vr; 
    let r_col1 = f64x4::from([-r[1], r[0], -r[3], r[2]]);
    let r_col2 = f64x4::from([-r[2], r[3], r[0], -r[1]]);
    let r_col3 = f64x4::from([-r[3], -r[2], r[1], r[0]]);

    // Accumulate: p = q0*col0 + q1*col1 + q2*col2 + q3*col3
    let result = q0 * r_col0 + q1 * r_col1 + q2 * r_col2 + q3 * r_col3;
    result.to_array()
}

/// Flat octonion (8D CD) multiply using quaternion_multiply_flat -- zero heap allocation.
///
/// Uses the CD doubling formula: `(a,b)(c,d) = (ac - d*b, da + bc*)`
/// where `a, b, c, d` are quaternions and `*` is conjugation.
#[inline]
pub fn octonion_multiply_flat(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let a_l: [f64; 4] = [a[0], a[1], a[2], a[3]];
    let a_r: [f64; 4] = [a[4], a[5], a[6], a[7]];
    let c_l: [f64; 4] = [b[0], b[1], b[2], b[3]];
    let c_r: [f64; 4] = [b[4], b[5], b[6], b[7]];

    // CD conjugation: keep real, negate imaginary.
    let conj_mask = f64x4::from([1.0, -1.0, -1.0, -1.0]);
    let conj_c_r = (f64x4::from(c_r) * conj_mask).to_array();
    let conj_c_l = (f64x4::from(c_l) * conj_mask).to_array();

    // (a,b)(c,d) = (ac - d*b, da + bc*)
    let ac = quaternion_multiply_flat(&a_l, &c_l);
    let conj_cr_ar = quaternion_multiply_flat(&conj_c_r, &a_r);
    let cr_al = quaternion_multiply_flat(&c_r, &a_l);
    let ar_conj_cl = quaternion_multiply_flat(&a_r, &conj_c_l);

    let left = f64x4::from(ac) - f64x4::from(conj_cr_ar);
    let right = f64x4::from(cr_al) + f64x4::from(ar_conj_cl);

    let l = left.to_array();
    let r = right.to_array();
    [l[0], l[1], l[2], l[3], r[0], r[1], r[2], r[3]]
}

/// Flat sedenion (16D CD) multiply using octonion_multiply_flat -- zero heap allocation.
#[inline]
pub fn sedenion_multiply_flat(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let a_l: [f64; 8] = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
    let a_r: [f64; 8] = [a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]];
    let c_l: [f64; 8] = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    let c_r: [f64; 8] = [b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]];

    let conj_lo = f64x4::from([1.0, -1.0, -1.0, -1.0]);
    let conj_hi = f64x4::from([-1.0, -1.0, -1.0, -1.0]);
    let cr_lo = f64x4::from([c_r[0], c_r[1], c_r[2], c_r[3]]);
    let cr_hi = f64x4::from([c_r[4], c_r[5], c_r[6], c_r[7]]);
    let cl_lo = f64x4::from([c_l[0], c_l[1], c_l[2], c_l[3]]);
    let cl_hi = f64x4::from([c_l[4], c_l[5], c_l[6], c_l[7]]);
    let conj_cr_lo = (cr_lo * conj_lo).to_array();
    let conj_cr_hi = (cr_hi * conj_hi).to_array();
    let conj_cl_lo = (cl_lo * conj_lo).to_array();
    let conj_cl_hi = (cl_hi * conj_hi).to_array();
    
    let conj_c_r: [f64; 8] = [
        conj_cr_lo[0], conj_cr_lo[1], conj_cr_lo[2], conj_cr_lo[3],
        conj_cr_hi[0], conj_cr_hi[1], conj_cr_hi[2], conj_cr_hi[3],
    ];
    let conj_c_l: [f64; 8] = [
        conj_cl_lo[0], conj_cl_lo[1], conj_cl_lo[2], conj_cl_lo[3],
        conj_cl_hi[0], conj_cl_hi[1], conj_cl_hi[2], conj_cl_hi[3],
    ];

    let ac = octonion_multiply_flat(&a_l, &c_l);
    let conj_cr_ar = octonion_multiply_flat(&conj_c_r, &a_r);
    let cr_al = octonion_multiply_flat(&c_r, &a_l);
    let ar_conj_cl = octonion_multiply_flat(&a_r, &conj_c_l);

    let left_lo = f64x4::from([ac[0], ac[1], ac[2], ac[3]]) - f64x4::from([conj_cr_ar[0], conj_cr_ar[1], conj_cr_ar[2], conj_cr_ar[3]]);
    let left_hi = f64x4::from([ac[4], ac[5], ac[6], ac[7]]) - f64x4::from([conj_cr_ar[4], conj_cr_ar[5], conj_cr_ar[6], conj_cr_ar[7]]);
    let right_lo = f64x4::from([cr_al[0], cr_al[1], cr_al[2], cr_al[3]]) + f64x4::from([ar_conj_cl[0], ar_conj_cl[1], ar_conj_cl[2], ar_conj_cl[3]]);
    let right_hi = f64x4::from([cr_al[4], cr_al[5], cr_al[6], ar_conj_cl[7]]) + f64x4::from([ar_conj_cl[4], ar_conj_cl[5], ar_conj_cl[6], ar_conj_cl[7]]);

    let ll = left_lo.to_array();
    let lh = left_hi.to_array();
    let rl = right_lo.to_array();
    let rh = right_hi.to_array();
    [
        ll[0], ll[1], ll[2], ll[3], lh[0], lh[1], lh[2], lh[3], rl[0], rl[1], rl[2], rl[3], rh[0],
        rh[1], rh[2], rh[3],
    ]
}

/// Flat sedenion (16D CD) associator using sedenion_multiply_flat.
#[inline]
pub fn sedenion_associator_flat(a: &[f64; 16], b: &[f64; 16], c: &[f64; 16]) -> [f64; 16] {
    let ab = sedenion_multiply_flat(a, b);
    let abc_left = sedenion_multiply_flat(&ab, c);
    let bc = sedenion_multiply_flat(b, c);
    let abc_right = sedenion_multiply_flat(a, &bc);

    let mut assoc = [0.0_f64; 16];
    for i in 0..4 {
        let l = f64x4::from([abc_left[i*4], abc_left[i*4+1], abc_left[i*4+2], abc_left[i*4+3]]);
        let r = f64x4::from([abc_right[i*4], abc_right[i*4+1], abc_right[i*4+2], abc_right[i*4+3]]);
        assoc[i*4..(i+1)*4].copy_from_slice(&(l - r).to_array());
    }
    assoc
}

/// Sedenion associator norm squared using AVX2+FMA.
#[inline]
pub fn sedenion_associator_norm_sq_flat(a: &[f64; 16], b: &[f64; 16], c: &[f64; 16]) -> f64 {
    let assoc = sedenion_associator_flat(a, b, c);
    crate::avx2_primitives::avx2_norm_sq_16(&assoc)
}

// ---------------------------------------------------------------------------
// Flat scalar baselines (no SIMD, no Vec -- pure unrolled arithmetic)
// ---------------------------------------------------------------------------

/// Flat scalar quaternion multiply -- no SIMD, no Vec, fully unrolled.
#[inline]
pub fn quaternion_multiply_scalar_flat(q: &[f64; 4], r: &[f64; 4]) -> [f64; 4] {
    [
        q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
        q[0] * r[1] + q[1] * r[0] - q[3] * r[2] + q[2] * r[3],
        q[0] * r[2] + q[2] * r[0] + q[3] * r[1] - q[1] * r[3],
        q[0] * r[3] + q[3] * r[0] + q[1] * r[2] - q[2] * r[1],
    ]
}

/// Flat scalar octonion multiply -- no SIMD, fixed arrays, CD doubling.
#[inline]
pub fn octonion_multiply_scalar_flat(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let a_l = [a[0], a[1], a[2], a[3]];
    let a_r = [a[4], a[5], a[6], a[7]];
    let c_l = [b[0], b[1], b[2], b[3]];
    let c_r = [b[4], b[5], b[6], b[7]];
    let conj_c_r = [c_r[0], -c_r[1], -c_r[2], -c_r[3]];
    let conj_c_l = [c_l[0], -c_l[1], -c_l[2], -c_l[3]];
    let ac = quaternion_multiply_scalar_flat(&a_l, &c_l);
    let dcb = quaternion_multiply_scalar_flat(&conj_c_r, &a_r);
    let da = quaternion_multiply_scalar_flat(&c_r, &a_l);
    let bcc = quaternion_multiply_scalar_flat(&a_r, &conj_c_l);
    [
        ac[0] - dcb[0], ac[1] - dcb[1], ac[2] - dcb[2], ac[3] - dcb[3],
        da[0] + bcc[0], da[1] + bcc[1], da[2] + bcc[2], da[3] + bcc[3],
    ]
}

// ---------------------------------------------------------------------------
// Slice-based flat multiply API (zero-copy from sub-slices)
// ---------------------------------------------------------------------------

/// CD multiply on sub-slices: `out[..dim] = a[..dim] * b[..dim]`.
pub fn cd_multiply_flat_into(a: &[f64], b: &[f64], out: &mut [f64], dim: usize) {
    debug_assert!(a.len() >= dim && b.len() >= dim && out.len() >= dim);
    match dim {
        1 => out[0] = a[0] * b[0],
        2 => {
            out[0] = a[0] * b[0] - a[1] * b[1];
            out[1] = a[0] * b[1] + a[1] * b[0];
        }
        4 => {
            let qa: [f64; 4] = [a[0], a[1], a[2], a[3]];
            let qb: [f64; 4] = [b[0], b[1], b[2], b[3]];
            let r = quaternion_multiply_flat(&qa, &qb);
            out[..4].copy_from_slice(&r);
        }
        8 => {
            let oa: [f64; 8] = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let ob: [f64; 8] = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
            let r = octonion_multiply_flat(&oa, &ob);
            out[..8].copy_from_slice(&r);
        }
        16 => {
            let sa: [f64; 16] = core::array::from_fn(|i| a[i]);
            let sb: [f64; 16] = core::array::from_fn(|i| b[i]);
            let r = sedenion_multiply_flat(&sa, &sb);
            out[..16].copy_from_slice(&r);
        }
        32 => {
            let half = 16;
            let mut conj_cr = [0.0_f64; 16];
            let mut conj_cl = [0.0_f64; 16];
            cd_conjugate_into(&b[half..], &mut conj_cr, half);
            cd_conjugate_into(&b[..half], &mut conj_cl, half);

            let mut t1 = [0.0_f64; 16];
            let mut t2 = [0.0_f64; 16];
            let mut t3 = [0.0_f64; 16];
            let mut t4 = [0.0_f64; 16];
            cd_multiply_flat_into(&a[..half], &b[..half], &mut t1, half);
            cd_multiply_flat_into(&conj_cr, &a[half..], &mut t2, half);
            cd_multiply_flat_into(&b[half..], &a[..half], &mut t3, half);
            cd_multiply_flat_into(&a[half..], &conj_cl, &mut t4, half);

            for i in 0..half {
                out[i] = t1[i] - t2[i];
                out[half + i] = t3[i] + t4[i];
            }
        }
        d if d >= 64 && d.is_power_of_two() => {
            let half = d / 2;
            let mut conj_cr = vec![0.0_f64; half];
            let mut conj_cl = vec![0.0_f64; half];
            cd_conjugate_into(&b[half..], &mut conj_cr, half);
            cd_conjugate_into(&b[..half], &mut conj_cl, half);

            let mut t1 = vec![0.0_f64; half];
            let mut t2 = vec![0.0_f64; half];
            let mut t3 = vec![0.0_f64; half];
            let mut t4 = vec![0.0_f64; half];
            cd_multiply_flat_into(&a[..half], &b[..half], &mut t1, half);
            cd_multiply_flat_into(&conj_cr, &a[half..], &mut t2, half);
            cd_multiply_flat_into(&b[half..], &a[..half], &mut t3, half);
            cd_multiply_flat_into(&a[half..], &conj_cl, &mut t4, half);

            for i in 0..half {
                out[i] = t1[i] - t2[i];
                out[half + i] = t3[i] + t4[i];
            }
        }
        _ => panic!("cd_multiply_flat_into: unsupported dim {dim}"),
    }
}

/// CD conjugation into a pre-allocated buffer.
#[inline]
fn cd_conjugate_into(src: &[f64], dst: &mut [f64], dim: usize) {
    debug_assert!(src.len() >= dim && dst.len() >= dim);
    dst[0] = src[0];
    let conj_mask = f64x4::from([1.0, -1.0, -1.0, -1.0]);
    let neg_mask = f64x4::from([-1.0, -1.0, -1.0, -1.0]);
    if dim >= 4 {
        let v = f64x4::from([src[0], src[1], src[2], src[3]]) * conj_mask;
        dst[0..4].copy_from_slice(&v.to_array());
    }
    let mut i = 4;
    while i + 4 <= dim {
        let v = f64x4::from([src[i], src[i + 1], src[i + 2], src[i + 3]]) * neg_mask;
        dst[i..i+4].copy_from_slice(&v.to_array());
        i += 4;
    }
    while i < dim {
        dst[i] = -src[i];
        i += 1;
    }
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
        i += 4;
    }

    let arr = sum.to_array();
    let mut total = arr[0] + arr[1] + arr[2] + arr[3];

    while i < n {
        total += a[i] * a[i];
        i += 1;
    }

    total
}
