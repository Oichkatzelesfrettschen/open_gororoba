use super::{arith::cd_conjugate, signs::cd_basis_mul_sign};

/// Signature for a parameterized Cayley-Dickson algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdSignature {
    gammas: Vec<i32>,
}

impl CdSignature {
    pub fn uniform(dim: usize, gamma: i32) -> Self {
        assert!(dim.is_power_of_two() && dim >= 2);
        assert!(gamma == -1 || gamma == 1);
        let n = dim.trailing_zeros() as usize;
        CdSignature {
            gammas: vec![gamma; n],
        }
    }

    pub fn standard(dim: usize) -> Self {
        Self::uniform(dim, -1)
    }

    pub fn split(dim: usize) -> Self {
        Self::uniform(dim, 1)
    }

    pub fn from_gammas(gammas: &[i32]) -> Self {
        assert!(!gammas.is_empty());
        assert!(gammas.iter().all(|&g| g == -1 || g == 1));
        CdSignature {
            gammas: gammas.to_vec(),
        }
    }

    pub fn dim(&self) -> usize {
        1 << self.gammas.len()
    }

    pub fn n_levels(&self) -> usize {
        self.gammas.len()
    }

    pub fn gamma(&self, level: usize) -> i32 {
        self.gammas[level]
    }

    pub fn is_standard(&self) -> bool {
        self.gammas.iter().all(|&g| g == -1)
    }

    pub fn is_split(&self) -> bool {
        self.gammas.iter().all(|&g| g == 1)
    }

    pub fn gammas(&self) -> &[i32] {
        &self.gammas
    }
}

pub fn cd_basis_mul_sign_split(dim: usize, p: usize, q: usize, sig: &CdSignature) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert_eq!(dim, sig.dim());
    debug_assert!(p < dim && q < dim);
    cd_basis_mul_sign_split_inner(dim, p, q, &sig.gammas)
}

fn cd_basis_mul_sign_split_inner(dim: usize, p: usize, q: usize, gammas: &[i32]) -> i32 {
    if dim == 1 {
        return 1;
    }

    let half = dim / 2;
    let gamma = gammas[gammas.len() - 1];
    let inner = &gammas[..gammas.len() - 1];

    if p < half && q < half {
        return cd_basis_mul_sign_split_inner(half, p, q, inner);
    }
    if p < half && q >= half {
        return cd_basis_mul_sign_split_inner(half, q - half, p, inner);
    }
    if p >= half && q < half {
        let s = cd_basis_mul_sign_split_inner(half, p - half, q, inner);
        return if q == 0 { s } else { -s };
    }

    let qh = q - half;
    let ph = p - half;
    if qh == 0 {
        return gamma;
    }
    -gamma * cd_basis_mul_sign_split_inner(half, qh, ph, inner)
}

pub fn cd_basis_mul_sign_split_iter(
    dim: usize,
    mut p: usize,
    mut q: usize,
    sig: &CdSignature,
) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert_eq!(dim, sig.dim());
    debug_assert!(p < dim && q < dim);

    let mut sign = 1i32;
    let mut half = dim >> 1;
    let n = sig.n_levels();
    let mut level = n;

    while half > 0 {
        level -= 1;
        let gamma = sig.gammas[level];
        let p_hi = p >= half;
        let q_hi = q >= half;

        match (p_hi, q_hi) {
            (false, false) => {}
            (false, true) => {
                let qh = q - half;
                q = p;
                p = qh;
            }
            (true, false) => {
                p -= half;
                if q != 0 {
                    sign = -sign;
                }
            }
            (true, true) => {
                let qh = q - half;
                let ph = p - half;
                if qh == 0 {
                    return gamma * sign;
                }
                sign *= -gamma;
                p = qh;
                q = ph;
            }
        }

        half >>= 1;
    }

    sign
}

pub fn cd_multiply_split(a: &[f64], b: &[f64], sig: &CdSignature) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(dim, sig.dim());
    cd_multiply_split_inner(a, b, &sig.gammas)
}

fn cd_multiply_split_inner(a: &[f64], b: &[f64], gammas: &[i32]) -> Vec<f64> {
    let dim = a.len();
    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    let half = dim / 2;
    let gamma = gammas[gammas.len() - 1] as f64;
    let inner = &gammas[..gammas.len() - 1];
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_c_r = cd_conjugate(c_r);
    let term1 = cd_multiply_split_inner(a_l, c_l, inner);
    let term2 = cd_multiply_split_inner(&conj_c_r, a_r, inner);
    let conj_c_l = cd_conjugate(c_l);
    let term3 = cd_multiply_split_inner(c_r, a_l, inner);
    let term4 = cd_multiply_split_inner(a_r, &conj_c_l, inner);

    let mut result = Vec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] + gamma * term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

pub fn cd_mul_table_split(sig: &CdSignature) -> Vec<Vec<(usize, i32)>> {
    let dim = sig.dim();
    let mut table = Vec::with_capacity(dim);
    for p in 0..dim {
        let mut row = Vec::with_capacity(dim);
        for q in 0..dim {
            let mut ep = vec![0.0; dim];
            ep[p] = 1.0;
            let mut eq = vec![0.0; dim];
            eq[q] = 1.0;
            let result = cd_multiply_split(&ep, &eq, sig);

            let mut idx = 0;
            let mut s = 1i32;
            for (k, &v) in result.iter().enumerate() {
                if v.abs() > 0.5 {
                    idx = k;
                    s = if v > 0.0 { 1 } else { -1 };
                    break;
                }
            }
            row.push((idx, s));
        }
        table.push(row);
    }
    table
}

/// Precomputed sign table for a parameterized CD algebra.
#[derive(Clone)]
pub struct SplitSignTable {
    dim: usize,
    sig: CdSignature,
    bits: Vec<u64>,
}

impl SplitSignTable {
    pub fn new(sig: &CdSignature) -> Self {
        let dim = sig.dim();
        let total_bits = dim * dim;
        let n_words = total_bits.div_ceil(64);
        let mut bits = vec![0u64; n_words];

        for p in 0..dim {
            for q in 0..dim {
                let s = cd_basis_mul_sign_split_iter(dim, p, q, sig);
                if s == -1 {
                    let idx = p * dim + q;
                    bits[idx / 64] |= 1u64 << (idx % 64);
                }
            }
        }

        SplitSignTable {
            dim,
            sig: sig.clone(),
            bits,
        }
    }

    #[inline(always)]
    pub fn sign(&self, p: usize, q: usize) -> i32 {
        debug_assert!(p < self.dim && q < self.dim);
        let idx = p * self.dim + q;
        let word = self.bits[idx / 64];
        let bit = (word >> (idx % 64)) & 1;
        1 - 2 * (bit as i32)
    }

    pub fn signature(&self) -> &CdSignature {
        &self.sig
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl std::fmt::Debug for SplitSignTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SplitSignTable {{ dim={}, sig={:?} }}",
            self.dim, self.sig
        )
    }
}

#[allow(dead_code)]
fn _standard_anchor(dim: usize, p: usize, q: usize) -> i32 {
    cd_basis_mul_sign(dim, p, q)
}
