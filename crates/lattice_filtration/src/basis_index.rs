//! Basis/lattice indexing helpers for filtration simulations.

/// Compact index codec for signed basis events.
#[derive(Debug, Clone, Copy)]
pub struct BasisIndexCodec {
    pub dim: usize,
}

impl BasisIndexCodec {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        Self { dim }
    }

    /// Encode `(basis, sign)` into a compact integer.
    pub fn encode(self, basis: usize, sign: i32) -> u64 {
        let b = (basis % self.dim) as u64;
        let s = if sign < 0 { 1u64 } else { 0u64 };
        (b << 1) | s
    }

    /// Decode integer back to `(basis, sign)`.
    pub fn decode(self, code: u64) -> (usize, i32) {
        let sign = if (code & 1) == 1 { -1 } else { 1 };
        let basis = ((code >> 1) as usize) % self.dim;
        (basis, sign)
    }
}

/// Project a 16D sedenion sample to an 8D lattice integer vector.
pub fn project_to_lattice(sample: &[f64; 16], scale: f64) -> [i32; 8] {
    let mut out = [0i32; 8];
    for i in 0..8 {
        let lo = sample[i];
        let hi = sample[i + 8];
        out[i] = ((lo - hi) * scale).round() as i32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_roundtrip() {
        let c = BasisIndexCodec::new(16);
        let code = c.encode(11, -1);
        let (basis, sign) = c.decode(code);
        assert_eq!(basis, 11);
        assert_eq!(sign, -1);
    }

    #[test]
    fn test_project_to_lattice() {
        let mut s = [0.0_f64; 16];
        s[0] = 2.0;
        s[8] = 1.0;
        let v = project_to_lattice(&s, 10.0);
        assert_eq!(v[0], 10);
    }
}
