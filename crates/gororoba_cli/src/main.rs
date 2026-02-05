//! gororoba CLI - Unified interface for physics simulation tools.
//!
//! Subcommands:
//! - algebra: Cayley-Dickson operations, zero-divisor search
//! - gr: Kerr geodesics, black hole shadows
//! - optics: GRIN ray tracing
//! - cosmology: Gravastar TOV solver
//! - quantum: MERA entropy, Ryu-Takayanagi
//! - materials: Metamaterial absorbers
//! - stats: Statistical tests

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gororoba")]
#[command(about = "Physics simulation CLI tools", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Cayley-Dickson algebra operations
    Algebra {
        #[command(subcommand)]
        cmd: AlgebraCmd,
    },
    /// General relativity: Kerr geodesics
    Gr {
        #[command(subcommand)]
        cmd: GrCmd,
    },
    /// Optics: GRIN ray tracing
    Optics {
        #[command(subcommand)]
        cmd: OpticsCmd,
    },
    /// Cosmology: Gravastar solver
    Cosmology {
        #[command(subcommand)]
        cmd: CosmologyCmd,
    },
    /// Quantum: MERA, holographic entropy
    Quantum {
        #[command(subcommand)]
        cmd: QuantumCmd,
    },
}

#[derive(Subcommand)]
enum AlgebraCmd {
    /// Search for zero-divisors in dimension dim
    ZdSearch {
        #[arg(short, long, default_value = "16")]
        dim: usize,
        #[arg(short, long, default_value = "1e-10")]
        atol: f64,
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Generate E8 roots
    E8Roots {
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Find box-kites
    BoxKites {
        #[arg(short, long, default_value = "16")]
        dim: usize,
        #[arg(short, long, default_value = "1e-10")]
        atol: f64,
    },
}

#[derive(Subcommand)]
enum GrCmd {
    /// Compute Kerr black hole shadow boundary
    Shadow {
        #[arg(short, long)]
        spin: f64,
        #[arg(short, long, default_value = "500")]
        n_points: usize,
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Trace null geodesic
    Geodesic {
        #[arg(short, long)]
        spin: f64,
        #[arg(short = 'L', long)]
        angular_momentum: f64,
        #[arg(short = 'Q', long)]
        carter: f64,
        #[arg(short, long, default_value = "10.0")]
        r0: f64,
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum OpticsCmd {
    /// Trace ray through GRIN fiber
    Fiber {
        #[arg(long, default_value = "1.5")]
        n0: f64,
        #[arg(long, default_value = "0.1")]
        g: f64,
        #[arg(long, default_value = "0.05")]
        step: f64,
        #[arg(long, default_value = "1000")]
        max_steps: usize,
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum CosmologyCmd {
    /// Solve gravastar TOV equations
    Gravastar {
        #[arg(long)]
        m_target: f64,
        #[arg(long)]
        r_v: f64,
        #[arg(long)]
        r1: f64,
        #[arg(long)]
        rho_v: f64,
        #[arg(long)]
        rho_shell: f64,
        #[arg(long, default_value = "1.0")]
        gamma: f64,
        #[arg(long, default_value = "1.0")]
        k_poly: f64,
    },
}

#[derive(Subcommand)]
enum QuantumCmd {
    /// Estimate MERA entropy
    Mera {
        #[arg(short, long)]
        subsystem_size: usize,
        #[arg(short, long, default_value = "4")]
        chi: usize,
    },
    /// Compute Bekenstein bound
    Bekenstein {
        #[arg(long)]
        radius_nm: f64,
        #[arg(long)]
        energy_ev: f64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Algebra { cmd } => handle_algebra(cmd),
        Commands::Gr { cmd } => handle_gr(cmd),
        Commands::Optics { cmd } => handle_optics(cmd),
        Commands::Cosmology { cmd } => handle_cosmology(cmd),
        Commands::Quantum { cmd } => handle_quantum(cmd),
    }
}

fn handle_algebra(cmd: AlgebraCmd) {
    use algebra_core::{find_zero_divisors, generate_e8_roots, find_box_kites};

    match cmd {
        AlgebraCmd::ZdSearch { dim, atol, output } => {
            eprintln!("Searching for zero-divisors in dimension {}...", dim);
            let zds = find_zero_divisors(dim, atol);
            eprintln!("Found {} zero-divisor pairs", zds.len());

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
                wtr.write_record(["i", "j", "k", "l", "norm"]).unwrap();
                for (i, j, k, l, norm) in &zds {
                    wtr.write_record(&[
                        i.to_string(),
                        j.to_string(),
                        k.to_string(),
                        l.to_string(),
                        norm.to_string(),
                    ]).unwrap();
                }
                wtr.flush().unwrap();
                println!("Wrote {} records to {}", zds.len(), path);
            } else {
                for (i, j, k, l, norm) in zds.iter().take(20) {
                    println!("({}, {}, {}, {}) -> norm = {:.2e}", i, j, k, l, norm);
                }
                if zds.len() > 20 {
                    println!("... and {} more", zds.len() - 20);
                }
            }
        }
        AlgebraCmd::E8Roots { output } => {
            let roots = generate_e8_roots();
            println!("Generated {} E8 roots", roots.len());

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
                wtr.write_record(["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]).unwrap();
                for root in &roots {
                    wtr.write_record(&root.coords.map(|x| x.to_string())).unwrap();
                }
                wtr.flush().unwrap();
                println!("Wrote {} roots to {}", roots.len(), path);
            }
        }
        AlgebraCmd::BoxKites { dim, atol } => {
            let bks = find_box_kites(dim, atol);
            println!("Found {} box-kites in dimension {}", bks.len(), dim);
        }
    }
}

fn handle_gr(cmd: GrCmd) {
    use gr_core::{shadow_boundary, trace_null_geodesic};
    use std::f64::consts::FRAC_PI_2;

    match cmd {
        GrCmd::Shadow { spin, n_points, output } => {
            eprintln!("Computing Kerr shadow for a = {}...", spin);
            let (alpha, beta) = shadow_boundary(spin, n_points, FRAC_PI_2);

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
                wtr.write_record(["alpha", "beta"]).unwrap();
                for (a, b) in alpha.iter().zip(beta.iter()) {
                    wtr.write_record(&[a.to_string(), b.to_string()]).unwrap();
                }
                wtr.flush().unwrap();
                println!("Wrote {} points to {}", alpha.len(), path);
            } else {
                println!("Shadow boundary has {} points", alpha.len());
                println!("Alpha range: [{:.3}, {:.3}]",
                    alpha.iter().cloned().fold(f64::INFINITY, f64::min),
                    alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            }
        }
        GrCmd::Geodesic { spin, angular_momentum, carter, r0, output } => {
            eprintln!("Tracing geodesic: a={}, L={}, Q={}, r0={}", spin, angular_momentum, carter, r0);
            let result = trace_null_geodesic(
                spin, 1.0, angular_momentum, carter,
                r0, FRAC_PI_2, 100.0, -1.0, 0.0, 2000
            );

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
                wtr.write_record(["lam", "t", "r", "theta", "phi"]).unwrap();
                for i in 0..result.lam.len() {
                    wtr.write_record(&[
                        result.lam[i].to_string(),
                        result.t[i].to_string(),
                        result.r[i].to_string(),
                        result.theta[i].to_string(),
                        result.phi[i].to_string(),
                    ]).unwrap();
                }
                wtr.flush().unwrap();
                println!("Wrote {} points to {}", result.lam.len(), path);
            } else {
                println!("Geodesic: {} points, terminated={}, reason={}",
                    result.lam.len(), result.terminated, result.termination_reason);
            }
        }
    }
}

fn handle_optics(cmd: OpticsCmd) {
    use optics_core::{Ray, trace_ray, GrinFiber};

    match cmd {
        OpticsCmd::Fiber { n0, g, step, max_steps, output } => {
            let medium = GrinFiber {
                n0,
                g,
                axis: [0.0, 0.0, 1.0],
            };
            let ray = Ray {
                pos: [0.5, 0.0, 0.0],
                dir: [0.0, 0.0, 1.0],
            };

            eprintln!("Tracing ray through GRIN fiber: n0={}, g={}", n0, g);
            let result = trace_ray(ray, &medium, step, max_steps);

            if let Some(path) = output {
                let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
                wtr.write_record(["s", "x", "y", "z", "dx", "dy", "dz"]).unwrap();
                for i in 0..result.positions.len() {
                    let p = result.positions[i];
                    let d = result.directions[i];
                    wtr.write_record(&[
                        result.arc_lengths[i].to_string(),
                        p[0].to_string(), p[1].to_string(), p[2].to_string(),
                        d[0].to_string(), d[1].to_string(), d[2].to_string(),
                    ]).unwrap();
                }
                wtr.flush().unwrap();
                println!("Wrote {} points to {}", result.positions.len(), path);
            } else {
                println!("Traced {} points, final position: ({:.4}, {:.4}, {:.4})",
                    result.positions.len(),
                    result.positions.last().unwrap()[0],
                    result.positions.last().unwrap()[1],
                    result.positions.last().unwrap()[2]);
            }
        }
    }
}

fn handle_cosmology(cmd: CosmologyCmd) {
    use cosmology_core::{solve_gravastar, GravastarConfig, PolytropicEos, AnisotropicParams};

    match cmd {
        CosmologyCmd::Gravastar { m_target, r_v: _, r1, rho_v: _, rho_shell: _, gamma, k_poly } => {
            eprintln!("Solving gravastar: M={}, gamma={}", m_target, gamma);

            let config = GravastarConfig {
                r1,
                m_target,
                compactness_target: 0.7,
                eos: PolytropicEos::new(k_poly, gamma),
                aniso: AnisotropicParams::isotropic(),
                dr: 1e-4,
                p_floor: 1e-12,
            };

            match solve_gravastar(&config) {
                Some(result) => {
                    println!("{{");
                    println!("  \"mass\": {},", result.mass);
                    println!("  \"r2\": {},", result.r2);
                    println!("  \"compactness\": {},", result.compactness);
                    println!("  \"is_stable\": {},", result.is_stable);
                    println!("  \"is_causal\": {}", result.is_causal);
                    println!("}}");
                }
                None => {
                    eprintln!("No solution found");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn handle_quantum(cmd: QuantumCmd) {
    use quantum_core::{mera_entropy_estimate, bekenstein_bound_bits};

    match cmd {
        QuantumCmd::Mera { subsystem_size, chi } => {
            let entropy = mera_entropy_estimate(subsystem_size, chi, 42);
            println!("MERA entropy estimate for L={}, chi={}: S = {:.6}", subsystem_size, chi, entropy);
        }
        QuantumCmd::Bekenstein { radius_nm, energy_ev } => {
            let bits = bekenstein_bound_bits(radius_nm, energy_ev);
            println!("Bekenstein bound: R={} nm, E={} eV -> S_max = {:.2e} bits", radius_nm, energy_ev, bits);
        }
    }
}
