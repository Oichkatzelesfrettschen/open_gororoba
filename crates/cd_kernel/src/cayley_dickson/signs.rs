/// Compute the sign in the Cayley-Dickson basis product: `e_p * e_q = sign * e_{p^q}`.
pub fn cd_basis_mul_sign(dim: usize, p: usize, q: usize) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert!(p < dim && q < dim);

    if dim == 1 {
        return 1;
    }

    let half = dim / 2;
    if p < half && q < half {
        return cd_basis_mul_sign(half, p, q);
    }
    if p < half && q >= half {
        return cd_basis_mul_sign(half, q - half, p);
    }
    if p >= half && q < half {
        let s = cd_basis_mul_sign(half, p - half, q);
        return if q == 0 { s } else { -s };
    }

    let qh = q - half;
    let ph = p - half;
    if qh == 0 {
        return -1;
    }
    cd_basis_mul_sign(half, qh, ph)
}

#[inline(always)]
pub fn cd_basis_mul_sign_iter(dim: usize, mut p: usize, mut q: usize) -> i32 {
    debug_assert!(dim.is_power_of_two() && dim >= 1);
    debug_assert!(p < dim && q < dim);

    let mut sign = 1i32;
    let mut half = dim >> 1;

    while half > 0 {
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
                    return -sign;
                }
                p = qh;
                q = ph;
            }
        }

        half >>= 1;
    }

    sign
}

/// Precomputed sign table for a fixed Cayley-Dickson dimension.
#[derive(Clone)]
pub struct SignTable {
    dim: usize,
    bits: Vec<u64>,
}

impl SignTable {
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two() && dim >= 1);
        let total_bits = dim * dim;
        let n_words = total_bits.div_ceil(64);
        let mut bits = vec![0u64; n_words];

        for p in 0..dim {
            for q in 0..dim {
                let s = cd_basis_mul_sign_iter(dim, p, q);
                if s == -1 {
                    let idx = p * dim + q;
                    bits[idx / 64] |= 1u64 << (idx % 64);
                }
            }
        }

        SignTable { dim, bits }
    }

    #[inline(always)]
    pub fn sign(&self, p: usize, q: usize) -> i32 {
        debug_assert!(p < self.dim && q < self.dim);
        let idx = p * self.dim + q;
        let word = self.bits[idx / 64];
        let bit = (word >> (idx % 64)) & 1;
        1 - 2 * (bit as i32)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    #[inline(always)]
    pub fn row_words(&self, p: usize) -> &[u64] {
        debug_assert!(p < self.dim);
        let words_per_row = self.dim.div_ceil(64);
        let start = p * words_per_row;
        &self.bits[start..start + words_per_row]
    }
}

impl std::fmt::Debug for SignTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignTable {{ dim={}, size={} bytes }}",
            self.dim,
            self.size_bytes()
        )
    }
}
