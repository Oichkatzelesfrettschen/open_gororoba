use algebra_core::construction::cayley_dickson::{cd_multiply, cd_norm_sq};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

fn norm_composition_entropy(dim: usize, n_samples: usize, n_bins: usize, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut norms = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let a_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let b_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();

        let a_norm = a_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm = b_raw.iter().map(|x| x * x).sum::<f64>().sqrt();

        let a: Vec<f64> = a_raw.iter().map(|x| x / a_norm).collect();
        let b: Vec<f64> = b_raw.iter().map(|x| x / b_norm).collect();

        let ab = cd_multiply(&a, &b);
        let norm = cd_norm_sq(&ab).sqrt();
        norms.push(norm);
    }

    // Bin range [0, 1.1] since norm composition gives 1.0
    let max_v = 1.1;
    let bin_width = max_v / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &n in &norms {
        let bin = ((n / bin_width) as usize).min(n_bins - 1);
        counts[bin] += 1;
    }

    let total = n_samples as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

fn main() {
    let dims = vec![2, 4, 8, 16, 32];
    let n_samples = 10000;
    let n_bins = 100;
    let seed = 42;
    
    println!("Computing Norm Composition Entropy (H_norm) across dimensions...");
    println!("Dim | Entropy (H_norm) | Mean ||ab|| | StdDev ||ab||");
    println!("----|------------------|-------------|--------------");
    for dim in dims {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut sum_v = 0.0;
        let mut sum_v2 = 0.0;
        let mut values = Vec::new();
        for _ in 0..n_samples {
             let a_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
             let b_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
             let a_norm = a_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
             let b_norm = b_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
             let a: Vec<f64> = a_raw.iter().map(|x| x / a_norm).collect();
             let b: Vec<f64> = b_raw.iter().map(|x| x / b_norm).collect();
             let ab = cd_multiply(&a, &b);
             let v = cd_norm_sq(&ab).sqrt();
             sum_v += v;
             sum_v2 += v*v;
             values.push(v);
        }
        let mean = sum_v / n_samples as f64;
        let stddev = (sum_v2 / n_samples as f64 - mean*mean).sqrt();

        let h = norm_composition_entropy(dim, n_samples, n_bins, seed);
        println!("{:3} | {:16.6} | {:11.6} | {:12.6e}", dim, h, mean, stddev);
    }
}
