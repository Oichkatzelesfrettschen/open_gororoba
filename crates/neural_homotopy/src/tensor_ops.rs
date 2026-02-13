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

/// Weighted alignment score using inverse-variance weights.
///
/// Each point is weighted by `1 / sigma_i^2` where `sigma_i` is the
/// error bar at that index. Points with smaller errors contribute more
/// to the alignment. Falls back to uniform weights if no errors provided.
pub fn weighted_alignment_score(
    loss_curve: &[f64],
    hubble_curve: &[f64],
    errors: Option<&[f64]>,
) -> f64 {
    let lhs = min_max_normalize(loss_curve);
    let rhs = min_max_normalize(hubble_curve);
    let n = lhs.len().min(rhs.len());
    if n == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut an = 0.0;
    let mut bn = 0.0;

    for i in 0..n {
        let w = match errors {
            Some(e) if i < e.len() && e[i] > 1e-30 => 1.0 / (e[i] * e[i]),
            _ => 1.0,
        };
        dot += w * lhs[i] * rhs[i];
        an += w * lhs[i] * lhs[i];
        bn += w * rhs[i] * rhs[i];
    }

    let c = if an <= 0.0 || bn <= 0.0 {
        0.0
    } else {
        dot / (an.sqrt() * bn.sqrt())
    };
    ((c + 1.0) * 0.5).clamp(0.0, 1.0)
}

/// Chi-squared goodness-of-fit between predicted and observed curves.
///
/// chi^2 = sum_i [ (pred_i - obs_i)^2 / sigma_i^2 ]
///
/// Uses Planck 2018 H_0 = 67.4 km/s/Mpc fractional error (0.74/67.4)
/// as uniform error bar if none provided. Returns (chi2, dof, reduced_chi2).
pub fn chi_squared_fit(
    predicted: &[f64],
    observed: &[f64],
    errors: Option<&[f64]>,
) -> (f64, usize, f64) {
    let n = predicted.len().min(observed.len());
    if n == 0 {
        return (0.0, 0, f64::NAN);
    }

    // Planck 2018 fractional uncertainty on H_0: sigma/H_0 ~ 0.011
    let default_sigma = 0.011;

    let mut chi2 = 0.0;
    for i in 0..n {
        let sigma = match errors {
            Some(e) if i < e.len() && e[i] > 1e-30 => e[i],
            _ => {
                // Scale default sigma by local value magnitude
                let scale = observed[i].abs().max(1e-30);
                default_sigma * scale
            }
        };
        let residual = predicted[i] - observed[i];
        chi2 += (residual * residual) / (sigma * sigma);
    }

    let dof = n.saturating_sub(1).max(1);
    let reduced = chi2 / dof as f64;
    (chi2, dof, reduced)
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

    #[test]
    fn test_weighted_alignment_uniform_equals_unweighted() {
        let loss = [1.0, 0.8, 0.6, 0.4];
        let hubble = [0.9, 0.7, 0.5, 0.3];
        let unweighted = alignment_score(&loss, &hubble);
        let weighted = weighted_alignment_score(&loss, &hubble, None);
        assert!(
            (unweighted - weighted).abs() < 1e-10,
            "Uniform weights should equal unweighted: {} vs {}",
            unweighted,
            weighted
        );
    }

    #[test]
    fn test_weighted_alignment_score_bounds() {
        let loss = [1.0, 0.8, 0.6, 0.4];
        let hubble = [0.9, 0.7, 0.5, 0.3];
        let errors = [0.01, 0.02, 0.05, 0.10];
        let s = weighted_alignment_score(&loss, &hubble, Some(&errors));
        assert!(
            (0.0..=1.0).contains(&s),
            "Weighted score should be in [0,1]: {}",
            s
        );
    }

    #[test]
    fn test_weighted_alignment_emphasizes_low_error() {
        // Two curves: identical in first half, divergent in second half.
        // With small errors on first half and large on second, should score high.
        // With small errors on second half and large on first, should score low.
        let pred = [1.0, 0.8, 0.6, 0.4];
        let obs_match_first = [1.0, 0.8, 0.0, 0.0]; // matches first, diverges second

        let errors_first_tight = [0.01, 0.01, 10.0, 10.0];
        let errors_second_tight = [10.0, 10.0, 0.01, 0.01];

        let s_first = weighted_alignment_score(&pred, &obs_match_first, Some(&errors_first_tight));
        let s_second =
            weighted_alignment_score(&pred, &obs_match_first, Some(&errors_second_tight));

        // When tight errors are on the matching region, score should be higher
        assert!(
            s_first > s_second,
            "Score should be higher when tight errors match: {} vs {}",
            s_first,
            s_second
        );
    }

    #[test]
    fn test_chi_squared_perfect_fit() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let (chi2, dof, _reduced) = chi_squared_fit(&a, &a, None);
        assert!(
            chi2 < 1e-20,
            "Perfect fit should have chi2 ~ 0: {}",
            chi2
        );
        assert_eq!(dof, 3);
    }

    #[test]
    fn test_chi_squared_with_errors() {
        let pred = [1.0, 2.0, 3.0];
        let obs = [1.1, 2.0, 2.9];
        let errs = [0.1, 0.1, 0.1];
        let (chi2, dof, reduced) = chi_squared_fit(&pred, &obs, Some(&errs));
        // chi2 = (0.1/0.1)^2 + (0/0.1)^2 + (0.1/0.1)^2 = 1 + 0 + 1 = 2
        assert!(
            (chi2 - 2.0).abs() < 1e-10,
            "Expected chi2=2, got {}",
            chi2
        );
        assert_eq!(dof, 2);
        assert!((reduced - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_empty() {
        let (chi2, dof, reduced) = chi_squared_fit(&[], &[], None);
        assert_eq!(chi2, 0.0);
        assert_eq!(dof, 0);
        assert!(reduced.is_nan());
    }
}
