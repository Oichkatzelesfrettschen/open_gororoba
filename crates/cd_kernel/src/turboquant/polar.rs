//! Hierarchical PolarQuant: recursive polar decomposition for quantization.
//!
//! Based on PolarQuant (arXiv 2502.02617) and the turboquant (abdelstark)
//! crate's hierarchical polar decomposition.
//!
//! # Algorithm
//!
//! At each level, pair coordinates (x_{2i}, x_{2i+1}) into polar form:
//!   r_i = sqrt(x_{2i}^2 + x_{2i+1}^2)
//!   theta_i = atan2(x_{2i+1}, x_{2i})
//!
//! After random rotation, the angles theta_i follow a known, data-independent
//! distribution (approximately uniform on `[-pi, pi]` for Gaussian inputs).
//! This enables a shared codebook without calibration.
//!
//! The recursion applies log2(d) levels, halving the dimension each time:
//!   Level 0: d coordinates -> d/2 (radius, angle) pairs
//!   Level 1: d/2 radii -> d/4 (radius, angle) pairs
//!   ...
//!   Level k: 2 values -> 1 (radius, angle) pair
//!
//! # Advantage over Lloyd-Max scalar quantization
//!
//! PolarQuant quantizes angles (uniform distribution -> uniform codebook)
//! separately from radii (chi-distribution -> optimized codebook).
//! This structural decomposition can achieve lower distortion at certain
//! bit-widths compared to treating all coordinates identically.

/// Polar pair: (radius, angle) from two Cartesian coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PolarPair {
    pub radius: f64,
    pub angle: f64,
}

/// Convert a pair of Cartesian coordinates to polar form.
#[inline]
pub fn to_polar(x: f64, y: f64) -> PolarPair {
    PolarPair {
        radius: (x * x + y * y).sqrt(),
        angle: y.atan2(x),
    }
}

/// Convert polar back to Cartesian.
#[inline]
pub fn from_polar(p: &PolarPair) -> (f64, f64) {
    (p.radius * p.angle.cos(), p.radius * p.angle.sin())
}

/// One level of polar decomposition: d coordinates -> d/2 polar pairs.
pub fn polar_decompose_level(coords: &[f64]) -> Vec<PolarPair> {
    let n = coords.len() / 2;
    (0..n)
        .map(|i| to_polar(coords[2 * i], coords[2 * i + 1]))
        .collect()
}

/// Inverse: d/2 polar pairs -> d coordinates.
pub fn polar_reconstruct_level(pairs: &[PolarPair]) -> Vec<f64> {
    let mut coords = Vec::with_capacity(pairs.len() * 2);
    for p in pairs {
        let (x, y) = from_polar(p);
        coords.push(x);
        coords.push(y);
    }
    coords
}

/// Full hierarchical polar decomposition: d coordinates -> log2(d) levels.
///
/// Returns (all_angles, final_radius) where:
/// - all_angles`[level]` contains the angles at that decomposition level
/// - final_radius is the single remaining radius after all levels
///
/// Requires d to be a power of 2.
pub fn hierarchical_decompose(coords: &[f64]) -> (Vec<Vec<f64>>, f64) {
    let d = coords.len();
    assert!(d.is_power_of_two() && d >= 2);
    let n_levels = d.trailing_zeros() as usize;

    let mut all_angles = Vec::with_capacity(n_levels);
    let mut current = coords.to_vec();

    for _level in 0..n_levels {
        let pairs = polar_decompose_level(&current);
        let angles: Vec<f64> = pairs.iter().map(|p| p.angle).collect();
        let radii: Vec<f64> = pairs.iter().map(|p| p.radius).collect();

        all_angles.push(angles);

        if radii.len() == 1 {
            return (all_angles, radii[0]);
        }
        current = radii;
    }

    (all_angles, current[0])
}

/// Reconstruct from hierarchical polar decomposition.
pub fn hierarchical_reconstruct(all_angles: &[Vec<f64>], final_radius: f64) -> Vec<f64> {
    let n_levels = all_angles.len();
    let mut current = vec![final_radius];

    for level in (0..n_levels).rev() {
        let angles = &all_angles[level];
        let pairs: Vec<PolarPair> = current
            .iter()
            .zip(angles.iter())
            .map(|(&r, &a)| PolarPair {
                radius: r,
                angle: a,
            })
            .collect();
        current = polar_reconstruct_level(&pairs);
    }

    current
}

/// Quantize angles uniformly (data-independent for rotated inputs).
///
/// Angles in `[-pi, pi]` are quantized to 2^bits uniform levels.
pub fn quantize_angle(angle: f64, bits: u32) -> u16 {
    let n_levels = 1u32 << bits;
    let normalized = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
    let clamped = normalized.clamp(0.0, 1.0 - 1e-10);
    (clamped * n_levels as f64) as u16
}

/// Dequantize angle: uniform level -> angle in `[-pi, pi]`.
pub fn dequantize_angle(index: u16, bits: u32) -> f64 {
    let n_levels = 1u32 << bits;
    let frac = (index as f64 + 0.5) / n_levels as f64;
    frac * 2.0 * std::f64::consts::PI - std::f64::consts::PI
}

/// Compressed representation from hierarchical PolarQuant.
#[derive(Clone, Debug)]
pub struct PolarCompressed {
    /// Quantized angle indices at each level.
    pub angle_indices: Vec<Vec<u16>>,
    /// Final radius (stored at full precision or quantized separately).
    pub final_radius: f64,
    /// Original vector norm.
    pub vec_norm: f64,
    /// Bits per angle.
    pub angle_bits: u32,
}

/// Full PolarQuant pipeline: rotate -> hierarchical decompose -> quantize angles.
pub fn polar_quantize(
    x: &[f64],
    rotation: &super::rotation::Rotation,
    angle_bits: u32,
    buf: &mut [f64],
) -> PolarCompressed {
    let d = x.len();

    // Normalize
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let inv_norm = if norm > 1e-15 { 1.0 / norm } else { 1.0 };
    for i in 0..d {
        buf[i] = x[i] * inv_norm;
    }

    // Rotate
    let mut rotated = vec![0.0f64; d];
    {
        let (input, scratch) = buf.split_at_mut(d);
        rotation.forward(input, scratch, &mut rotated);
    }

    // Hierarchical polar decomposition
    let (all_angles, final_radius) = hierarchical_decompose(&rotated);

    // Quantize angles uniformly
    let angle_indices: Vec<Vec<u16>> = all_angles
        .iter()
        .map(|angles| {
            angles
                .iter()
                .map(|&a| quantize_angle(a, angle_bits))
                .collect()
        })
        .collect();

    PolarCompressed {
        angle_indices,
        final_radius,
        vec_norm: norm,
        angle_bits,
    }
}

/// Dequantize and reconstruct from PolarQuant.
pub fn polar_dequantize(
    compressed: &PolarCompressed,
    rotation: &super::rotation::Rotation,
    buf: &mut [f64],
    out: &mut [f64],
) {
    let bits = compressed.angle_bits;

    // Reconstruct angles
    let all_angles: Vec<Vec<f64>> = compressed
        .angle_indices
        .iter()
        .map(|indices| indices.iter().map(|&i| dequantize_angle(i, bits)).collect())
        .collect();

    // Hierarchical reconstruction
    let rotated = hierarchical_reconstruct(&all_angles, compressed.final_radius);

    let d = out.len();
    buf[..d].copy_from_slice(&rotated);

    // Unrotate
    {
        let (input, scratch) = buf.split_at_mut(d);
        rotation.inverse(input, scratch, out);
    }

    // Denormalize
    for v in out.iter_mut() {
        *v *= compressed.vec_norm;
    }
}

/// Estimate inner product directly from polar codes (no dequantization).
///
/// Novel idea from turbo-quant crate (RecursiveIntell): for paired
/// coordinates (x, y) = r*(cos(theta), sin(theta)), the dot product
/// <(x1,y1), (x2,y2)> = r1*r2*cos(theta1-theta2).
///
/// At the first decomposition level, each pair contributes:
///   r_i * (q_a * cos(theta_i) + q_b * sin(theta_i))
/// where (q_a, q_b) is the corresponding query pair.
///
/// This avoids the full dequantization + unrotation pipeline.
pub fn polar_inner_product_estimate(query_rotated: &[f64], compressed: &PolarCompressed) -> f64 {
    let d = query_rotated.len();
    let bits = compressed.angle_bits;

    // Use the first level (d/2 pairs) for the estimate
    let n_pairs = d / 2;
    let mut estimate = 0.0f64;

    // The final radius scales the entire vector
    let r_scale = compressed.final_radius * compressed.vec_norm;

    // For each pair at level 0
    for i in 0..n_pairs.min(compressed.angle_indices[0].len()) {
        let theta = dequantize_angle(compressed.angle_indices[0][i], bits);
        let q_a = query_rotated[2 * i];
        let q_b = query_rotated[2 * i + 1];

        // <(r*cos(theta), r*sin(theta)), (q_a, q_b)> = r*(q_a*cos + q_b*sin)
        estimate += q_a * theta.cos() + q_b * theta.sin();
    }

    // Scale by the hierarchical radius product (approximation)
    estimate * r_scale / (d as f64).sqrt()
}

/// Memory usage of PolarQuant compressed representation.
pub fn polar_bits_per_vector(d: usize, angle_bits: u32) -> usize {
    let n_levels = d.trailing_zeros() as usize;
    let mut total_angles = 0;
    let mut dim = d;
    for _ in 0..n_levels {
        total_angles += dim / 2;
        dim /= 2;
    }
    total_angles * angle_bits as usize + 64 + 16 // angles + radius(f64) + norm(f16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_roundtrip_single() {
        let p = to_polar(3.0, 4.0);
        assert!((p.radius - 5.0).abs() < 1e-10);
        let (x, y) = from_polar(&p);
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hierarchical_roundtrip() {
        let d = 16;
        let coords: Vec<f64> = (0..d).map(|i| (i as f64 * 0.3).sin()).collect();
        let (angles, radius) = hierarchical_decompose(&coords);
        let recon = hierarchical_reconstruct(&angles, radius);

        let max_err: f64 = coords
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-10, "Hierarchical roundtrip error: {}", max_err);
    }

    #[test]
    fn test_angle_quantize_roundtrip() {
        for bits in [2, 3, 4, 8] {
            let angle = 1.23;
            let idx = quantize_angle(angle, bits);
            let recon = dequantize_angle(idx, bits);
            let n_levels = 1u32 << bits;
            let resolution = 2.0 * std::f64::consts::PI / n_levels as f64;
            assert!(
                (angle - recon).abs() < resolution,
                "Angle roundtrip at {}-bit: {} -> {} -> {}",
                bits,
                angle,
                idx,
                recon
            );
        }
    }

    #[test]
    fn test_polar_quantize_pipeline() {
        let d = 64;
        let bits = 3;
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let rotation = super::super::rotation::Rotation::new_fast_jl(d, 42);
        let mut buf = vec![0.0f64; 3 * d];

        let compressed = polar_quantize(&x, &rotation, bits, &mut buf);
        let mut recon = vec![0.0f64; d];
        polar_dequantize(&compressed, &rotation, &mut buf, &mut recon);

        let mse: f64 = x
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;
        println!("PolarQuant MSE at d={}, {}-bit: {:.6}", d, bits, mse);

        // Compare with TurboQuant MSE
        let tq = super::super::pipeline::TurboQuantMSE::new(d, bits, 42, true);
        let tq_comp = tq.quantize(&x, &mut buf);
        let mut tq_recon = vec![0.0f64; d];
        tq.dequantize(&tq_comp, &mut buf, &mut tq_recon);
        let tq_mse: f64 = x
            .iter()
            .zip(tq_recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        println!("TurboQuant MSE at d={}, {}-bit: {:.6}", d, bits, tq_mse);
        println!("Polar/TQ ratio: {:.3}", mse / tq_mse);
    }

    #[test]
    fn test_polar_bits_per_vector() {
        let bpv = polar_bits_per_vector(128, 3);
        // d=128: 7 levels, total angles = 64+32+16+8+4+2+1 = 127
        // 127 * 3 bits + 64 (radius) + 16 (norm) = 381 + 80 = 461 bits
        println!("PolarQuant bits per vector (d=128, 3-bit): {}", bpv);
        let fp16_bits = 128 * 16;
        println!("Compression ratio: {:.1}x", fp16_bits as f64 / bpv as f64);
    }

    #[test]
    fn test_hierarchical_levels() {
        let d = 128;
        let coords: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).cos()).collect();
        let (angles, radius) = hierarchical_decompose(&coords);

        // d=128 -> 7 levels
        assert_eq!(angles.len(), 7);
        assert_eq!(angles[0].len(), 64); // level 0: 128/2 = 64 angles
        assert_eq!(angles[1].len(), 32); // level 1: 64/2 = 32 angles
        assert_eq!(angles[6].len(), 1); // level 6: 2/2 = 1 angle
        assert!(radius > 0.0);

        println!("Hierarchical decomposition of d=128:");
        for (i, a) in angles.iter().enumerate() {
            println!("  Level {}: {} angles", i, a.len());
        }
        println!("  Final radius: {:.6}", radius);
    }
}
