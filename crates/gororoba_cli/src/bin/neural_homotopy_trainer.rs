//! E-029: Neural Homotopy Trainer
//!
//! Validates Thesis 3 (A-infinity correction) by:
//! 1. Building the sedenion associator tensor m_3
//! 2. Optionally predicting corrections via a Burn neural network
//! 3. Running pentagon-constrained optimization (gradient-free)
//! 4. Measuring pentagon violation before and after
//! 5. Serializing results to TOML
//!
//! The optimization searches for a correction tensor m_3 that
//! minimizes the A-infinity A_4 pentagon relation violation.

use clap::Parser;
use neural_homotopy::{
    assemble_neural_correction, optimize_with_restarts, CorrectionTensor,
    PentagonOptimizationConfig, PerturbationDataset,
};
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "neural-homotopy-trainer")]
#[command(about = "E-029: A-infinity Correction Tensor Search")]
struct Args {
    /// Perturbation noise level (0.0 = no perturbation)
    #[arg(long, default_value = "0.0")]
    perturb: f64,

    /// Number of optimization steps
    #[arg(long, default_value = "500")]
    n_steps: usize,

    /// Step size for coordinate descent
    #[arg(long, default_value = "0.1")]
    step_size: f64,

    /// L2 regularization weight
    #[arg(long, default_value = "0.0001")]
    lambda: f64,

    /// Number of random restarts
    #[arg(long, default_value = "5")]
    n_restarts: usize,

    /// Number of quadruples sampled for pentagon violation
    #[arg(long, default_value = "256")]
    n_violation_samples: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Skip neural network prediction (use algebraic ansatz only)
    #[arg(long, default_value = "false")]
    skip_neural: bool,

    /// Neural network hidden layer size
    #[arg(long, default_value = "128")]
    hidden_size: usize,

    /// Output directory for results
    #[arg(long, default_value = "data/evidence")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("E-029: Neural Homotopy Trainer");
    println!("===================================");
    println!(
        "Steps: {}, Restarts: {}, Seed: {}",
        args.n_steps, args.n_restarts, args.seed
    );
    if args.perturb > 0.0 {
        println!("Perturbation noise: {:.2}%", args.perturb * 100.0);
    }
    println!();

    // Phase 1: Build algebraic ansatz
    println!("[1/5] Building associator tensor m_3...");
    let associator = CorrectionTensor::from_associator();
    let assoc_violation = associator.pentagon_violation(args.n_violation_samples);
    let assoc_nnz = associator.nnz();
    let assoc_sparsity = associator.sparsity();
    let assoc_l2 = associator.l2_norm_sq().sqrt();
    println!("  NNZ entries: {}", assoc_nnz);
    println!("  Sparsity: {:.4}", assoc_sparsity);
    println!("  L2 norm: {:.4}", assoc_l2);
    println!("  Pentagon violation: {:.6}", assoc_violation);

    // Phase 2: Neural network prediction (optional)
    let mut neural_tensor = None;
    let mut neural_violation = f64::NAN;
    let mut neural_params = 0usize;

    if !args.skip_neural {
        println!();
        println!("[2/5] Predicting corrections via neural network (Burn NdArray)...");
        let (nt, n_params) = assemble_neural_correction(args.hidden_size);
        neural_params = n_params;
        println!("  Model params: {}", neural_params);
        println!(
            "  Architecture: 256 -> {} -> {} -> 16",
            args.hidden_size,
            args.hidden_size / 2
        );

        neural_violation = nt.pentagon_violation(args.n_violation_samples);
        println!("  Neural pentagon violation: {:.6}", neural_violation);
        println!("  Neural NNZ: {}", nt.nnz());
        neural_tensor = Some(nt);
    } else {
        println!();
        println!("[2/5] Skipping neural prediction (--skip-neural)");
    }

    // Phase 3: Choose best initial tensor
    println!();
    println!("[3/5] Selecting initial tensor for optimization...");
    let initial = if let Some(ref nt) = neural_tensor {
        if neural_violation < assoc_violation {
            println!(
                "  Using neural tensor (violation {:.6} < associator {:.6})",
                neural_violation, assoc_violation
            );
            nt.clone()
        } else {
            println!(
                "  Using associator tensor (violation {:.6} <= neural {:.6})",
                assoc_violation, neural_violation
            );
            associator.clone()
        }
    } else {
        println!("  Using associator tensor (no neural available)");
        associator.clone()
    };
    let initial_violation = initial.pentagon_violation(args.n_violation_samples);

    // Phase 4: Pentagon-constrained optimization
    println!();
    println!("[4/5] Running pentagon-constrained optimization...");
    println!(
        "  {} steps x {} restarts = {} total evaluations",
        args.n_steps,
        args.n_restarts,
        args.n_steps * args.n_restarts
    );

    let config = PentagonOptimizationConfig {
        n_steps: args.n_steps,
        step_size: args.step_size,
        step_decay: 0.998,
        lambda: args.lambda,
        n_violation_samples: args.n_violation_samples,
        seed: args.seed,
    };
    let result = optimize_with_restarts(&initial, &config, args.n_restarts);

    let improvement_pct = if initial_violation > 0.0 {
        (1.0 - result.final_violation / initial_violation) * 100.0
    } else {
        0.0
    };

    println!("  Initial violation: {:.6}", initial_violation);
    println!("  Final violation:   {:.6}", result.final_violation);
    println!("  Improvement:       {:.2}%", improvement_pct);
    println!("  Accepted steps:    {}", result.n_accepted);
    println!("  Converged:         {}", result.converged);
    println!("  Final L2 norm:     {:.6}", result.final_l2_norm_sq.sqrt());

    // Phase 5: Perturbation robustness (optional)
    let mut perturb_results = Vec::new();
    if args.perturb > 0.0 {
        println!();
        println!(
            "[5/5] Perturbation robustness test (noise={:.2}%)...",
            args.perturb * 100.0
        );
        let levels = [args.perturb * 0.5, args.perturb, args.perturb * 2.0];
        let ds = PerturbationDataset::build(&levels, args.seed);
        let counts = ds.difference_counts();
        for (level, count) in &counts {
            println!(
                "  Noise {:.2}%: {} entries changed ({:.1}%)",
                level * 100.0,
                count,
                *count as f64 / 256.0 * 100.0
            );
            perturb_results.push((*level, *count));
        }
    } else {
        println!();
        println!("[5/5] Skipping perturbation test (--perturb 0)");
    }

    // Write TOML output
    println!();
    println!("Writing output...");
    std::fs::create_dir_all(&args.output_dir)?;
    let toml_path = format!("{}/e029_neural_homotopy.toml", args.output_dir);
    let mut f = std::fs::File::create(&toml_path)?;

    writeln!(f, "[metadata]")?;
    writeln!(f, "experiment = \"E-029\"")?;
    writeln!(
        f,
        "title = \"A-infinity Correction Tensor via Neural Homotopy Search\""
    )?;
    writeln!(f, "n_steps = {}", args.n_steps)?;
    writeln!(f, "n_restarts = {}", args.n_restarts)?;
    writeln!(f, "n_violation_samples = {}", args.n_violation_samples)?;
    writeln!(f, "seed = {}", args.seed)?;
    writeln!(f, "lambda = {}", args.lambda)?;
    writeln!(f, "step_size = {}", args.step_size)?;
    writeln!(f, "perturb = {}", args.perturb)?;
    writeln!(f, "skip_neural = {}", args.skip_neural)?;
    writeln!(f)?;

    writeln!(f, "[associator]")?;
    writeln!(f, "nnz = {}", assoc_nnz)?;
    writeln!(f, "sparsity = {:.6}", assoc_sparsity)?;
    writeln!(f, "l2_norm = {:.6}", assoc_l2)?;
    writeln!(f, "pentagon_violation = {:.6}", assoc_violation)?;
    writeln!(f)?;

    if !args.skip_neural {
        writeln!(f, "[neural_model]")?;
        writeln!(f, "hidden_size = {}", args.hidden_size)?;
        writeln!(f, "n_params = {}", neural_params)?;
        writeln!(f, "pentagon_violation = {:.6}", neural_violation)?;
        if let Some(ref nt) = neural_tensor {
            writeln!(f, "nnz = {}", nt.nnz())?;
            writeln!(f, "sparsity = {:.6}", nt.sparsity())?;
        }
        writeln!(f)?;
    }

    writeln!(f, "[optimization]")?;
    writeln!(f, "initial_violation = {:.6}", initial_violation)?;
    writeln!(f, "final_violation = {:.6}", result.final_violation)?;
    writeln!(f, "improvement_pct = {:.4}", improvement_pct)?;
    writeln!(f, "n_accepted = {}", result.n_accepted)?;
    writeln!(f, "converged = {}", result.converged)?;
    writeln!(f, "final_l2_norm = {:.6}", result.final_l2_norm_sq.sqrt())?;
    writeln!(f, "final_nnz = {}", result.tensor.nnz())?;
    writeln!(f)?;

    // Loss trace summary (first, middle, last)
    let n_trace = result.loss_trace.len();
    if n_trace > 0 {
        writeln!(f, "[optimization.trace]")?;
        writeln!(f, "initial_loss = {:.6}", result.loss_trace[0])?;
        if n_trace > 1 {
            writeln!(f, "midpoint_loss = {:.6}", result.loss_trace[n_trace / 2])?;
            writeln!(f, "final_loss = {:.6}", result.loss_trace[n_trace - 1])?;
        }
        writeln!(f)?;
    }

    if !perturb_results.is_empty() {
        writeln!(f, "[perturbation]")?;
        for (level, count) in &perturb_results {
            writeln!(f, "# noise={:.4}, entries_changed={}", level, count)?;
        }
        writeln!(
            f,
            "noise_levels = {:?}",
            perturb_results.iter().map(|(l, _)| *l).collect::<Vec<_>>()
        )?;
        writeln!(
            f,
            "entries_changed = {:?}",
            perturb_results.iter().map(|(_, c)| *c).collect::<Vec<_>>()
        )?;
        writeln!(f)?;
    }

    // Optimized correction tensor summary
    writeln!(
        f,
        "{}",
        result.tensor.serialize_to_toml(args.n_violation_samples)
    )?;

    println!("  TOML: {}", toml_path);
    println!();

    // Summary
    let pass_threshold = 0.1;
    let passed = result.final_violation < pass_threshold;
    println!("===================================");
    println!("Pentagon violation threshold: {:.2}", pass_threshold);
    println!("Final violation:             {:.6}", result.final_violation);
    println!(
        "Status:                      {}",
        if passed { "PASS" } else { "NEEDS HIGHER m_4" }
    );
    println!("E-029 Complete");

    Ok(())
}
