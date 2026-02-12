//! Bit-level source layer.

use crate::traits::BitSourceLayer;

/// Deterministic Fibonacci-based bit source.
#[derive(Debug, Clone, Copy, Default)]
pub struct FibonacciBitSource;

impl BitSourceLayer for FibonacciBitSource {
    fn sample_words(&self, n: usize) -> Vec<u64> {
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        let mut a = 1_u64;
        let mut b = 1_u64;
        for i in 0..n {
            let v = if i < 2 {
                1
            } else {
                let c = a.wrapping_add(b);
                a = b;
                b = c;
                c
            };
            out.push(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::BitSourceLayer;

    #[test]
    fn test_fibonacci_source_size() {
        let src = FibonacciBitSource;
        let words = src.sample_words(10);
        assert_eq!(words.len(), 10);
        assert_eq!(words[0], 1);
        assert_eq!(words[1], 1);
    }
}
