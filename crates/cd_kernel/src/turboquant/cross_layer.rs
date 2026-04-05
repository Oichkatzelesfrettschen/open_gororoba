//! Cross-layer KV cache compression (MiniCache/XQuant pattern).
//!
//! Compresses the KV cache across layers (depth dimension), which is
//! orthogonal to per-vector quantization (TurboQuant operates within
//! each layer).  The two compose: apply cross-layer first, then
//! quantize the merged result.
//!
//! # MiniCache pattern (NeurIPS 2024, arXiv 2405.14366)
//!
//! Adjacent layers in the middle-to-deep part of a transformer have
//! high KV state similarity.  MiniCache decomposes each state into
//! magnitude and direction components, then merges adjacent layers
//! using SLERP (Spherical Linear Interpolation) on directions while
//! preserving original magnitudes.
//!
//! Stored per merged pair: interpolated direction, both magnitudes,
//! and the inter-layer angle (for quality control).
//!
//! # XQuant pattern (arXiv 2510.11236)
//!
//! Cross-layer compression via shared quantization parameters.
//! Achieves sub-1.4-bit effective quantization by exploiting
//! cross-layer redundancy.

/// Cross-layer compressed KV cache entry.
#[derive(Clone, Debug)]
pub struct CrossLayerCompressed {
    /// SLERP-merged direction (unit vector, d elements).
    pub merged_direction: Vec<f64>,
    /// Magnitude of layer n's KV state.
    pub magnitude_n: f64,
    /// Magnitude of layer n+1's KV state.
    pub magnitude_n1: f64,
    /// Angle between the two layers' directions (for quality control).
    pub inter_layer_angle: f64,
    /// SLERP interpolation parameter (typically 0.5 for equal weighting).
    pub slerp_t: f64,
}

/// Decompose a vector into magnitude and direction.
fn mag_dir(v: &[f64]) -> (f64, Vec<f64>) {
    let mag: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag < 1e-15 {
        return (0.0, vec![0.0; v.len()]);
    }
    let dir: Vec<f64> = v.iter().map(|x| x / mag).collect();
    (mag, dir)
}

/// Compute the angle between two unit vectors.
fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0).acos()
}

/// Spherical Linear Interpolation (SLERP) between two unit vectors.
///
/// Returns a unit vector that is the geodesic interpolation at parameter t:
///   slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
/// where omega is the angle between a and b.
pub fn slerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    let omega = angle_between(a, b);
    if omega.abs() < 1e-10 {
        // Nearly identical: just return a (or b)
        return a.to_vec();
    }
    let sin_omega = omega.sin();
    let w_a = ((1.0 - t) * omega).sin() / sin_omega;
    let w_b = (t * omega).sin() / sin_omega;

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| w_a * x + w_b * y)
        .collect()
}

/// Merge two adjacent layers' KV states via SLERP.
///
/// Returns the compressed representation with the merged direction,
/// both magnitudes, and the inter-layer angle.
pub fn merge_adjacent_layers(
    layer_n: &[f64],
    layer_n1: &[f64],
    slerp_t: f64,
) -> CrossLayerCompressed {
    let (mag_n, dir_n) = mag_dir(layer_n);
    let (mag_n1, dir_n1) = mag_dir(layer_n1);
    let angle = angle_between(&dir_n, &dir_n1);
    let merged = slerp(&dir_n, &dir_n1, slerp_t);

    CrossLayerCompressed {
        merged_direction: merged,
        magnitude_n: mag_n,
        magnitude_n1: mag_n1,
        inter_layer_angle: angle,
        slerp_t,
    }
}

/// Reconstruct layer n from cross-layer compressed representation.
pub fn reconstruct_layer_n(compressed: &CrossLayerCompressed) -> Vec<f64> {
    compressed
        .merged_direction
        .iter()
        .map(|&d| d * compressed.magnitude_n)
        .collect()
}

/// Reconstruct layer n+1 from cross-layer compressed representation.
pub fn reconstruct_layer_n1(compressed: &CrossLayerCompressed) -> Vec<f64> {
    compressed
        .merged_direction
        .iter()
        .map(|&d| d * compressed.magnitude_n1)
        .collect()
}

/// Compression ratio from cross-layer merging.
///
/// Two layers of d elements (2*d total) compressed to:
/// d (merged direction) + 2 (magnitudes) + 1 (angle) + 1 (slerp_t) = d + 4
/// Ratio = 2d / (d + 4)  -- approaches 2x for large d.
pub fn compression_ratio(d: usize) -> f64 {
    (2 * d) as f64 / (d + 4) as f64
}

/// Memory savings from cross-layer compression across all layers.
///
/// For n_layers layers, we merge pairs: n_layers/2 compressed entries.
/// Each compressed entry stores d + 4 values instead of 2*d.
pub fn total_savings(d: usize, n_layers: usize) -> (usize, usize) {
    let original = n_layers * d;
    let compressed = (n_layers / 2) * (d + 4) + (n_layers % 2) * d;
    (original, compressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slerp_endpoints() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let at_0 = slerp(&a, &b, 0.0);
        let at_1 = slerp(&a, &b, 1.0);

        assert!((at_0[0] - 1.0).abs() < 1e-10, "slerp(a,b,0) should be a");
        assert!((at_1[1] - 1.0).abs() < 1e-10, "slerp(a,b,1) should be b");
    }

    #[test]
    fn test_slerp_midpoint() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mid = slerp(&a, &b, 0.5);

        // Midpoint between x-axis and y-axis is (1/sqrt(2), 1/sqrt(2), 0)
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((mid[0] - expected).abs() < 1e-10);
        assert!((mid[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_merge_reconstruct_quality() {
        let d = 128;
        let layer_n: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let layer_n1: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin() + 0.05).collect();

        let compressed = merge_adjacent_layers(&layer_n, &layer_n1, 0.5);
        let recon_n = reconstruct_layer_n(&compressed);
        let _recon_n1 = reconstruct_layer_n1(&compressed);

        // Cosine similarity between original and reconstructed
        let cos_n: f64 = {
            let dot: f64 = layer_n.iter().zip(recon_n.iter()).map(|(a, b)| a * b).sum();
            let na: f64 = layer_n.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = recon_n.iter().map(|x| x * x).sum::<f64>().sqrt();
            dot / (na * nb)
        };

        println!(
            "Cross-layer merge (d={}): angle={:.4} rad, cos_n={:.6}",
            d, compressed.inter_layer_angle, cos_n
        );

        // For similar layers, reconstruction should be good
        assert!(cos_n > 0.9, "Reconstruction quality too low: cos={}", cos_n);
    }

    #[test]
    fn test_compression_ratio() {
        assert!((compression_ratio(128) - 1.94).abs() < 0.01);
        assert!((compression_ratio(64) - 1.88).abs() < 0.01);
        println!(
            "Cross-layer compression ratio: d=128: {:.2}x, d=64: {:.2}x",
            compression_ratio(128),
            compression_ratio(64)
        );
    }

    #[test]
    fn test_total_savings() {
        let (orig, comp) = total_savings(128, 36);
        println!(
            "36 layers, d=128: {} -> {} values ({:.1}x)",
            orig,
            comp,
            orig as f64 / comp as f64
        );
        assert!(comp < orig);
    }
}
