use gororoba_algebra::analysis::entropy_census::phase_transition_analysis;

fn main() {
    let dims = vec![2, 4, 8, 16, 32];
    let n_samples = 5000;
    let n_bins = 100;
    let seed = 42;

    println!("Computing entropy across dimensions...");
    let results = phase_transition_analysis(&dims, n_samples, n_bins, seed);

    println!("Dim | Entropy (H) | ZD Density | Delta H");
    println!("----|-------------|------------|---------");
    for (dim, h, zd, delta_h) in results {
        println!("{:3} | {:11.6} | {:10.6} | {:11.6}", dim, h, zd, delta_h);
    }
}
