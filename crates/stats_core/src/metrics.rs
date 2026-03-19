use crate::helpers::mean;

/// Root-mean-square of a slice.
pub fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }
}

/// Root-mean-square of values after subtracting a pre-computed mean.
pub fn centered_rms(values: &[f64], mu: f64) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / values.len() as f64).sqrt()
    }
}

/// Pearson correlation coefficient between two slices.
pub fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let mu_l = mean(left);
    let mu_r = mean(right);
    let mut num = 0.0;
    let mut var_l = 0.0;
    let mut var_r = 0.0;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let dl = l - mu_l;
        let dr = r - mu_r;
        num += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }
    let denom = (var_l * var_r).sqrt();
    if denom > 0.0 {
        Some(num / denom)
    } else {
        Some(0.0)
    }
}

/// Percentage drop from baseline to fitted value.
pub fn percent_drop(baseline: f64, fitted: f64) -> f64 {
    if baseline <= 0.0 {
        0.0
    } else {
        100.0 * (baseline - fitted) / baseline
    }
}
