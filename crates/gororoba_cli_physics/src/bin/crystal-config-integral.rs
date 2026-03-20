use anyhow::Result;
use clap::Parser;
use faer::Mat;
use tensor_core::{build_rank1_symmetry_adapted, build_rank2_symmetry_adapted};

#[derive(Parser, Debug)]
#[command(author, version, about = "Symmetry-Adapted configurational integral for crystals")]
struct Args {
    /// Number of particles
    #[arg(short, long, default_value_t = 4)]
    n_particles: usize,

    /// Number of discretization points per dimension
    #[arg(short, long, default_value_t = 16)]
    grid_n: usize,

    /// Domain width around equilibrium position (in Angstroms or reduced units)
    #[arg(short, long, default_value_t = 0.5)]
    width: f64,

    /// Inverse temperature beta
    #[arg(short, long, default_value_t = 1.0)]
    beta: f64,

    /// Rank (1 or 2)
    #[arg(short, long, default_value_t = 1)]
    rank: usize,
}

fn potential(q: &[f64]) -> f64 {
    let mut energy = 0.0;
    // Harmonic trap around equilibrium
    #[allow(clippy::needless_range_loop)]
    for i in 0..q.len() {
        energy += 0.5 * (q[i] - i as f64).powi(2);
    }
    // Inter-particle bonds
    for i in 0..q.len() {
        for j in i+1..q.len() {
            let r = (q[i] - q[j]).abs();
            if r > 1e-6 {
                energy += (r - (j - i) as f64).powi(2);
            }
        }
    }
    energy
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("--- Symmetry-Adapted TT-Cross Configurational Integral ---");

    let d = args.n_particles;
    let n = args.grid_n;
    
    // Equilibrium positions q_bar = (0, 1, 2, ..., N-1)
    let q_bar: Vec<f64> = (0..d).map(|i| i as f64).collect();
    
    // Discretization offsets around q_bar
    let nodes: Vec<f64> = (0..n).map(|i| {
        let t = (i as f64) / (n as f64 - 1.0);
        args.width * (t - 0.5)
    }).collect();

    let boltzmann = |indices: &[usize]| -> f64 {
        let mut q = vec![0.0; d];
        for k in 0..d {
            q[k] = q_bar[k] + nodes[indices[k]];
        }
        (-args.beta * potential(&q)).exp()
    };

    let pivot: Vec<usize> = vec![n / 2; d];
    
    let tt = if args.rank == 1 {
        println!("Building Rank-1 approximation...");
        build_rank1_symmetry_adapted(d, n, &pivot, boltzmann)
    } else {
        println!("Building Rank-2 approximation...");
        // pivot2 must be significantly different to avoid S_k singularity
        let mut pivot2 = vec![0; d];
        for k in 0..d {
            pivot2[k] = if pivot[k] > 2 { pivot[k] - 2 } else { pivot[k] + 2 };
        }
        build_rank2_symmetry_adapted(d, n, &pivot, &pivot2, boltzmann)
    };

    // Contract Z = sum_i1 ... sum_id F(i1...id) * w1*...*wd
    // Since uniform weights for now: w = width / (n-1)
    let w = args.width / (n as f64);
    
    // Integration via TT contraction
    let mut res = Mat::<f64>::zeros(1, tt.cores[0].data.shape()[2]);
    for j in 0..tt.cores[0].data.shape()[2] {
        let mut sum_k = 0.0;
        for i in 0..n {
            sum_k += tt.cores[0].data[[0, i, j]] * w;
        }
        res.write(0, j, sum_k);
    }

    for k in 1..d {
        if let Some(m) = tt.intersection_matrices.get(k - 1) {
            res = &res * m;
        }
        let core = &tt.cores[k].data;
        let r_prev = core.shape()[0];
        let r_next = core.shape()[2];
        let mut contract = Mat::<f64>::zeros(r_prev, r_next);
        for i in 0..r_prev {
            for j in 0..r_next {
                let mut sum_i = 0.0;
                for idx_n in 0..n {
                    sum_i += core[[i, idx_n, j]] * w;
                }
                contract.write(i, j, sum_i);
            }
        }
        res = &res * contract;
    }

    let z = res.read(0, 0);
    println!("Configurational Integral Z_{} = {:.6e}", d, z);
    println!("Free Energy A = {:.6} (units of kT)", -z.ln());

    Ok(())
}
