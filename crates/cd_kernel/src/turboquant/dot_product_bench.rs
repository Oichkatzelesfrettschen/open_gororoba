//! Dot product implementation comparison for QJL inner product hot path.
//!
//! Compares different approaches for the critical <S@query, signs> operation:
//! 1. Naive f64 loop
//! 2. BitPackedSigns POPCNT
//! 3. Q16.16 fixed-point accumulation
//! 4. f32 loop (lower precision but faster)
//!
//! This establishes the baseline that simsimd (or pulp) would need to beat.

use super::{fixed_point::Q16_16, sign_pack::BitPackedSigns};

/// Naive f64 dot product (baseline).
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// f32 dot product (lower precision, potentially faster).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum()
}

/// Sign-vector dot product via BitPackedSigns POPCNT.
pub fn dot_signs_packed(signs: &BitPackedSigns, values: &[f64]) -> f64 {
    signs.inner_product(values)
}

/// Sign-vector dot product via naive i8 loop (comparison baseline).
pub fn dot_signs_naive(signs: &[i8], values: &[f64]) -> f64 {
    signs
        .iter()
        .zip(values.iter())
        .map(|(&s, &v)| s as f64 * v)
        .sum()
}

/// Q16.16 exact dot product.
pub fn dot_q16(a: &[f64], b_q16: &[Q16_16]) -> f64 {
    let mut acc: i64 = 0;
    for (i, &v) in a.iter().enumerate() {
        Q16_16::mac(&mut acc, Q16_16::from_f64(v), b_q16[i]);
    }
    Q16_16::finalize_acc(acc).to_f64()
}

/// Benchmark all dot product methods and report throughput.
///
/// Returns (method_name, result, ops_per_call) tuples.
pub fn benchmark_dot_products(d: usize, n_iterations: usize) -> Vec<(&'static str, f64, usize)> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    use std::time::Instant;

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;

    // Generate test data
    let a: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
    let b: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
    let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let signs_i8: Vec<i8> = (0..d)
        .map(|i| if (i * 7 % 3) < 2 { 1 } else { -1 })
        .collect();
    let signs_packed = BitPackedSigns::pack(&signs_i8);
    let b_q16: Vec<Q16_16> = b.iter().map(|&v| Q16_16::from_f64(v)).collect();

    let mut results = Vec::new();

    // 1. f64 naive
    let t0 = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..n_iterations {
        sum += dot_f64(&a, &b);
    }
    let _elapsed = t0.elapsed().as_secs_f64();
    results.push(("f64_naive", sum / n_iterations as f64, d * n_iterations));

    // 2. f32 loop
    let t0 = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..n_iterations {
        sum += dot_f32(&a_f32, &b_f32);
    }
    let _elapsed_f32 = t0.elapsed().as_secs_f64();
    results.push(("f32_loop", sum / n_iterations as f64, d * n_iterations));

    // 3. Signs packed POPCNT
    let t0 = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..n_iterations {
        sum += dot_signs_packed(&signs_packed, &a);
    }
    let _elapsed_packed = t0.elapsed().as_secs_f64();
    results.push(("signs_popcnt", sum / n_iterations as f64, d * n_iterations));

    // 4. Signs naive i8
    let t0 = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..n_iterations {
        sum += dot_signs_naive(&signs_i8, &a);
    }
    let _elapsed_naive = t0.elapsed().as_secs_f64();
    results.push(("signs_naive", sum / n_iterations as f64, d * n_iterations));

    // 5. Q16.16 exact
    let t0 = Instant::now();
    let mut sum = 0.0f64;
    for _ in 0..n_iterations {
        sum += dot_q16(&a, &b_q16);
    }
    let _elapsed_q16 = t0.elapsed().as_secs_f64();
    results.push(("q16_exact", sum / n_iterations as f64, d * n_iterations));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_products_agree() {
        let d = 128;
        let a: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..d).map(|i| (i as f64 * 0.2).cos()).collect();

        let f64_result = dot_f64(&a, &b);
        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let f32_result = dot_f32(&a_f32, &b_f32);

        let b_q16: Vec<Q16_16> = b.iter().map(|&v| Q16_16::from_f64(v)).collect();
        let q16_result = dot_q16(&a, &b_q16);

        println!("f64 result: {:.10}", f64_result);
        println!("f32 result: {:.10}", f32_result);
        println!("Q16 result: {:.10}", q16_result);
        println!("f32 vs f64: {:.2e}", (f32_result - f64_result).abs());
        println!("Q16 vs f64: {:.2e}", (q16_result - f64_result).abs());

        assert!(
            (f32_result - f64_result).abs() < 1e-3,
            "f32 too far from f64"
        );
        assert!(
            (q16_result - f64_result).abs() < 1e-2,
            "Q16 too far from f64"
        );
    }

    #[test]
    fn test_sign_dot_products_agree() {
        let d = 128;
        let values: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
        let signs: Vec<i8> = (0..d).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        let packed = BitPackedSigns::pack(&signs);

        let naive = dot_signs_naive(&signs, &values);
        let popcnt = dot_signs_packed(&packed, &values);

        assert!(
            (naive - popcnt).abs() < 1e-10,
            "Sign dot products disagree: naive={}, popcnt={}",
            naive,
            popcnt
        );
    }

    #[test]
    fn test_benchmark_runs() {
        let results = benchmark_dot_products(128, 10_000);
        println!("\nDot product benchmark (d=128, 10K iterations):");
        for (name, result, ops) in &results {
            println!("  {:>15}: result={:.8}, total_ops={}", name, result, ops);
        }
        assert_eq!(results.len(), 5);
    }
}
