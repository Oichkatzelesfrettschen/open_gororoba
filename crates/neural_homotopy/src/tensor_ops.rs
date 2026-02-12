//! Small tensor/vector utilities for loss-curve comparisons.

/// Min-max normalize to [0, 1].
pub fn min_max_normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-12);
    values.iter().map(|v| (v - min) / span).collect()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut an = 0.0;
    let mut bn = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        an += a[i] * a[i];
        bn += b[i] * b[i];
    }
    if an <= 0.0 || bn <= 0.0 {
        0.0
    } else {
        dot / (an.sqrt() * bn.sqrt())
    }
}

/// Alignment score in [0,1] from normalized cosine similarity.
pub fn alignment_score(loss_curve: &[f64], hubble_curve: &[f64]) -> f64 {
    let lhs = min_max_normalize(loss_curve);
    let rhs = min_max_normalize(hubble_curve);
    let c = cosine_similarity(&lhs, &rhs);
    ((c + 1.0) * 0.5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_max_normalize_bounds() {
        let out = min_max_normalize(&[2.0, 4.0, 6.0]);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_similarity_identity() {
        let c = cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]);
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_alignment_score_bounds() {
        let s = alignment_score(&[1.0, 0.8, 0.6], &[0.2, 0.4, 0.6]);
        assert!((0.0..=1.0).contains(&s));
    }
}
