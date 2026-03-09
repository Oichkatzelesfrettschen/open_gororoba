use gororoba_algebra::construction::cayley_dickson::cd_associator_norm;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

fn alternativity_entropy(dim: usize, n_samples: usize, n_bins: usize, seed: u64) -> f64 {
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

        // A(a, a, b)
        let norm = cd_associator_norm(&a, &a, &b);
        norms.push(norm);
    }

    let max_norm = norms.iter().cloned().fold(0.0_f64, f64::max);
    if max_norm < 1e-12 {
        return 0.0;
    }

    let bin_width = max_norm / n_bins as f64;
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
    let n_samples = 5000;
    let n_bins = 100;
    let seed = 42;

    println!("Computing Alternativity Violation Entropy (H_alt) across dimensions...");
    println!("Dim | Entropy (H_alt) | Max Violation");
    println!("----|-----------------|--------------");
    for dim in dims {
        // Measure max violation too
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut max_v = 0.0;
        for _ in 0..100 {
            let a_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let b_raw: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let a_norm = a_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let b_norm = b_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let a: Vec<f64> = a_raw.iter().map(|x| x / a_norm).collect();
            let b: Vec<f64> = b_raw.iter().map(|x| x / b_norm).collect();
            let v = cd_associator_norm(&a, &a, &b);
            if v > max_v {
                max_v = v;
            }
        }

        let h = alternativity_entropy(dim, n_samples, n_bins, seed);
        println!("{:3} | {:15.6} | {:13.6e}", dim, h, max_v);
    }
}
