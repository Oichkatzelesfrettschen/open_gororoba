/// Kahan compensated summation: O(eps) error regardless of accumulation length.
///
/// Standard f64 sum of n terms has error O(n * eps).
/// Kahan reduces this to O(eps) by tracking a running compensation term.
/// For Berry phase f_sum with n_grid=200 (40,000 terms), this ensures
/// |error| < 2.2e-16 instead of ~8.9e-12, guaranteeing correct rounding
/// to the nearest integer for Chern number quantization.
pub(crate) struct KahanSum {
    sum: f64,
    compensation: f64,
}

impl KahanSum {
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    pub fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    pub fn total(&self) -> f64 {
        self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_exact_for_small_n() {
        let mut ks = KahanSum::new();
        for i in 1..=100 {
            ks.add(i as f64);
        }
        assert!((ks.total() - 5050.0).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_sum_large_plus_small() {
        // Classic failure case for naive summation: large + many small
        let mut ks = KahanSum::new();
        ks.add(1e16);
        for _ in 0..10_000 {
            ks.add(1.0);
        }
        ks.add(-1e16);
        // Naive sum would lose the small additions entirely
        assert!((ks.total() - 10_000.0).abs() < 1.0);
    }
}
