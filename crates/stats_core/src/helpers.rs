//! Canonical statistical helper functions.
//!
//! These are simple, commonly needed functions that were duplicated across
//! multiple binaries. They live in stats_core as the Tier 1 home for
//! statistical operations.

/// Arithmetic mean of a slice. Returns 0.0 for empty input.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation given pre-computed mean.
///
/// Uses Bessel correction (N-1 denominator). Returns 0.0 for fewer than
/// 2 elements.
pub fn std_dev(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

/// Median of a mutable slice. Sorts in place. Returns 0.0 for empty input.
///
/// For even-length slices, returns the average of the two middle elements.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Compute a histogram of values.
///
/// Returns (bin_centers, counts).
pub fn histogram(values: &[f64], n_bins: usize, min: f64, max: f64) -> (Vec<f64>, Vec<usize>) {
    if values.is_empty() || n_bins == 0 || max <= min {
        return (vec![], vec![]);
    }

    let mut counts = vec![0; n_bins];
    let bin_width = (max - min) / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();

    for &v in values {
        if v >= min && v <= max {
            let mut bin = ((v - min) / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            counts[bin] += 1;
        }
    }

    (bin_centers, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_empty() {
        assert!((mean(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean(&[42.0]) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_basic() {
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mu = mean(&vals);
        let sd = std_dev(&vals, mu);
        // Known population std dev is 2.0, sample std dev ~2.138
        assert!((sd - 2.138).abs() < 0.01, "std_dev={}", sd);
    }

    #[test]
    fn test_std_dev_empty() {
        assert!((std_dev(&[], 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_single() {
        assert!((std_dev(&[5.0], 5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_odd() {
        let mut vals = vec![3.0, 1.0, 2.0];
        assert!((median(&mut vals) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_even() {
        let mut vals = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median(&mut vals) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_empty() {
        assert!((median(&mut []) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_single() {
        let mut vals = vec![42.0];
        assert!((median(&mut vals) - 42.0).abs() < 1e-12);
    }
}
