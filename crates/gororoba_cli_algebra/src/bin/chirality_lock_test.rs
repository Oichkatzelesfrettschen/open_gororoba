//! Chirality Lock Test: Falsification of the holographic chirality hypothesis.
//!
//! Tests whether macroscopic vortex handedness is locked to the Sedenion
//! multiplication table orientation. The 2x2 experimental matrix is:
//!
//! | Config | CD Table  | Vorticity | Purpose                         |
//! |--------|-----------|-----------|---------------------------------|
//! | A      | Standard  | Standard  | Control baseline                |
//! | B      | Standard  | Flipped   | Can vorticity flip w/o algebra? |
//! | C      | Conjugate | Standard  | Does algebra alone flip ring?   |
//! | D      | Conjugate | Flipped   | Do two inversions cancel?       |

use algebra_core::construction::cayley_dickson::{cd_multiply, cd_multiply_conjugated};
use clap::{Parser, Subcommand};
use gororoba_engine::simulation::ChiralityMode;
use std::io::Write;

#[derive(Parser)]
#[command(
    name = "chirality-lock-test",
    about = "Chirality-algebra lock falsification"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run all 4 configurations and compare.
    Matrix {
        /// Grid dimension (N^3).
        #[arg(long, default_value = "32")]
        grid: usize,
        /// Number of evolution steps.
        #[arg(long, default_value = "300")]
        steps: usize,
        /// Output CSV path.
        #[arg(long, default_value = "data/csv/chirality_matrix.csv")]
        output: String,
    },
    /// Run a single configuration.
    Single {
        /// Chirality mode: standard, flipped, conjugate, doubleflip.
        #[arg(long)]
        mode: String,
        #[arg(long, default_value = "32")]
        grid: usize,
        #[arg(long, default_value = "300")]
        steps: usize,
    },
}

/// 3D Sedenion field with chirality-aware multiplication.
struct ChiralField {
    data: Vec<[f64; 16]>,
    nx: usize,
    ny: usize,
    nz: usize,
    mode: ChiralityMode,
}

impl ChiralField {
    fn new(nx: usize, ny: usize, nz: usize, mode: ChiralityMode) -> Self {
        let n = nx * ny * nz;
        let mut data = vec![[0.0f64; 16]; n];
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;
        let ring_r = nx as f64 / 4.0;
        let sigma = nx as f64 / 8.0;

        // Vorticity sign from chirality mode
        let vort_sign: f64 = match mode {
            ChiralityMode::Standard | ChiralityMode::Conjugate => 1.0,
            ChiralityMode::Flipped | ChiralityMode::DoubleFlip => -1.0,
        };

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let dz = z as f64 - cz;
                    let rho = (dx * dx + dy * dy).sqrt();
                    let dr = rho - ring_r;
                    let d2 = dr * dr + dz * dz;
                    let amp = (-d2 / (sigma * sigma)).exp();
                    let idx = x + nx * (y + ny * z);
                    // Toroidal vortex: tangential velocity field
                    data[idx][1] = vort_sign * amp * (dy / (rho + 1e-10));
                    data[idx][2] = vort_sign * amp * (-dx / (rho + 1e-10));
                    data[idx][3] = amp * dz / (sigma + 1e-10);
                    // Seed higher components for non-associativity
                    data[idx][8] = amp * 0.05;
                    data[idx][9] = amp * 0.03;
                }
            }
        }

        Self {
            data,
            nx,
            ny,
            nz,
            mode,
        }
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.nx * (y + self.ny * z)
    }

    /// Multiply two sedenions using the mode-appropriate table.
    fn multiply(&self, a: &[f64; 16], b: &[f64; 16]) -> Vec<f64> {
        match self.mode {
            ChiralityMode::Standard | ChiralityMode::Flipped => cd_multiply(a, b),
            ChiralityMode::Conjugate | ChiralityMode::DoubleFlip => cd_multiply_conjugated(a, b),
        }
    }

    /// Compute dominant helicity mode (sign of the helicity integral).
    fn helicity_sign(&self) -> f64 {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let mut helicity = 0.0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    // Velocity-like components (e1, e2, e3)
                    let ux = self.data[idx][1];
                    let uy = self.data[idx][2];
                    let uz = self.data[idx][3];

                    // Vorticity via finite differences (periodic)
                    let xp = self.idx((x + 1) % nx, y, z);
                    let xm = self.idx((x + nx - 1) % nx, y, z);
                    let yp = self.idx(x, (y + 1) % ny, z);
                    let ym = self.idx(x, (y + ny - 1) % ny, z);
                    let zp = self.idx(x, y, (z + 1) % nz);
                    let zm = self.idx(x, y, (z + nz - 1) % nz);

                    let duz_dy = (self.data[yp][3] - self.data[ym][3]) * 0.5;
                    let duy_dz = (self.data[zp][2] - self.data[zm][2]) * 0.5;
                    let dux_dz = (self.data[zp][1] - self.data[zm][1]) * 0.5;
                    let duz_dx = (self.data[xp][3] - self.data[xm][3]) * 0.5;
                    let duy_dx = (self.data[xp][2] - self.data[xm][2]) * 0.5;
                    let dux_dy = (self.data[yp][1] - self.data[ym][1]) * 0.5;

                    let wx = duz_dy - duy_dz;
                    let wy = dux_dz - duz_dx;
                    let wz = duy_dx - dux_dy;

                    // Helicity = u . omega
                    helicity += ux * wx + uy * wy + uz * wz;
                }
            }
        }
        helicity / (nx * ny * nz) as f64
    }

    /// Compute enstrophy (mean squared vorticity).
    fn enstrophy(&self) -> f64 {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let mut enst = 0.0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let xp = self.idx((x + 1) % nx, y, z);
                    let xm = self.idx((x + nx - 1) % nx, y, z);
                    let yp = self.idx(x, (y + 1) % ny, z);
                    let ym = self.idx(x, (y + ny - 1) % ny, z);
                    let zp = self.idx(x, y, (z + 1) % nz);
                    let zm = self.idx(x, y, (z + nz - 1) % nz);

                    let duz_dy = (self.data[yp][3] - self.data[ym][3]) * 0.5;
                    let duy_dz = (self.data[zp][2] - self.data[zm][2]) * 0.5;
                    let dux_dz = (self.data[zp][1] - self.data[zm][1]) * 0.5;
                    let duz_dx = (self.data[xp][3] - self.data[xm][3]) * 0.5;
                    let duy_dx = (self.data[xp][2] - self.data[xm][2]) * 0.5;
                    let dux_dy = (self.data[yp][1] - self.data[ym][1]) * 0.5;

                    let wx = duz_dy - duy_dz;
                    let wy = dux_dz - duz_dx;
                    let wz = duy_dx - dux_dy;
                    enst += wx * wx + wy * wy + wz * wz;
                }
            }
        }
        enst / (nx * ny * nz) as f64
    }

    /// Compute field energy.
    fn energy(&self) -> f64 {
        self.data
            .iter()
            .map(|s| s.iter().map(|v| v * v).sum::<f64>())
            .sum()
    }

    /// Mean associator norm (higher = more non-associative tension).
    fn mean_associator_norm(&self) -> f64 {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let mut total = 0.0;
        let mut count = 0usize;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx.saturating_sub(2) {
                    let a = &self.data[self.idx(x, y, z)];
                    let b = &self.data[self.idx(x + 1, y, z)];
                    let ab = self.multiply(a, b);
                    // Use norm of product as associator proxy
                    let norm: f64 = ab.iter().map(|v| v * v).sum::<f64>();
                    total += norm.sqrt();
                    count += 1;
                }
            }
        }
        if count > 0 { total / count as f64 } else { 0.0 }
    }

    /// Evolution step: diffusion + sedenion self-interaction.
    #[allow(clippy::needless_range_loop)] // multi-array stencil: c indexes old[6 neighbors][c] + self.data[idx][c]
    fn step(&mut self, dt: f64) {
        let old = self.data.clone();
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let diffusion = 0.1;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = self.idx(x, y, z);
                    let xp = self.idx((x + 1) % nx, y, z);
                    let xm = self.idx((x + nx - 1) % nx, y, z);
                    let yp = self.idx(x, (y + 1) % ny, z);
                    let ym = self.idx(x, (y + ny - 1) % ny, z);
                    let zp = self.idx(x, y, (z + 1) % nz);
                    let zm = self.idx(x, y, (z + nz - 1) % nz);

                    for c in 0..16 {
                        let laplacian = old[xp][c]
                            + old[xm][c]
                            + old[yp][c]
                            + old[ym][c]
                            + old[zp][c]
                            + old[zm][c]
                            - 6.0 * old[idx][c];
                        self.data[idx][c] = old[idx][c] + dt * diffusion * laplacian;
                    }

                    // Non-linear interaction via chirality-aware multiplication
                    let phi: [f64; 16] = self.data[idx];
                    let phi2 = self.multiply(&phi, &phi);
                    for c in 0..16 {
                        self.data[idx][c] += dt * 0.01 * phi2[c];
                    }
                }
            }
        }
    }
}

fn parse_mode(s: &str) -> ChiralityMode {
    match s.to_lowercase().as_str() {
        "standard" => ChiralityMode::Standard,
        "flipped" => ChiralityMode::Flipped,
        "conjugate" => ChiralityMode::Conjugate,
        "doubleflip" | "double_flip" => ChiralityMode::DoubleFlip,
        _ => {
            eprintln!("Unknown mode '{s}', using Standard");
            ChiralityMode::Standard
        }
    }
}

fn run_config(mode: ChiralityMode, grid: usize, steps: usize) -> (f64, f64, f64, f64, bool) {
    let mut field = ChiralField::new(grid, grid, grid, mode);
    let dt = 0.005;
    let initial_energy = field.energy();
    let threshold = initial_energy * 100.0;

    let mut stable = true;
    for _ in 0..steps {
        field.step(dt);
        let e = field.energy();
        if e > threshold || e.is_nan() || e.is_infinite() {
            stable = false;
            break;
        }
    }

    let helicity = field.helicity_sign();
    let enstrophy = field.enstrophy();
    let energy = field.energy();
    let assoc = field.mean_associator_norm();

    (helicity, enstrophy, energy, assoc, stable)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Matrix {
            grid,
            steps,
            output,
        } => {
            println!("=== Chirality Lock Test: 2x2 Matrix ===");
            println!("Grid: {grid}^3, Steps: {steps}");

            let configs = [
                ("A", ChiralityMode::Standard, "Std CD + Std vort"),
                ("B", ChiralityMode::Flipped, "Std CD + Flip vort"),
                ("C", ChiralityMode::Conjugate, "Conj CD + Std vort"),
                ("D", ChiralityMode::DoubleFlip, "Conj CD + Flip vort"),
            ];

            let mut file = std::fs::File::create(&output)?;
            writeln!(
                file,
                "config,mode,helicity,enstrophy,energy,associator_norm,stable"
            )?;

            let mut results = Vec::new();
            for (label, mode, desc) in &configs {
                println!("\n--- Config {label}: {desc} ---");
                let (hel, enst, energy, assoc, stable) = run_config(*mode, grid, steps);
                let handedness = if hel > 0.0 { "RIGHT" } else { "LEFT" };
                let stability = if stable { "STABLE" } else { "UNSTABLE" };
                println!("  Helicity: {hel:+.6e} ({handedness})");
                println!("  Enstrophy: {enst:.6e}");
                println!("  Energy: {energy:.6e}");
                println!("  Associator: {assoc:.6e}");
                println!("  Stability: {stability}");
                writeln!(
                    file,
                    "{label},{desc},{hel:.10e},{enst:.10e},{energy:.10e},{assoc:.10e},{stable}"
                )?;
                results.push((label.to_string(), hel, stable));
            }

            println!("\nWrote {output}");

            // Chirality correlation analysis
            println!("\n=== C-777 Verdict ===");
            let a_stable = results[0].2;
            let b_stable = results[1].2;
            let _c_stable = results[2].2;
            let d_stable = results[3].2;
            let a_hel = results[0].1;
            let b_hel = results[1].1;

            if a_stable && !b_stable && d_stable {
                println!("CONFIRMED: Chirality is locked to algebraic orientation.");
                println!("  Config B (Std CD + Flipped) is UNSTABLE.");
                println!("  Config D (Conj CD + Flipped) is STABLE.");
                println!("  => Algebra determines stable handedness.");
            } else if a_stable && b_stable && a_hel.signum() != b_hel.signum() {
                println!("FALSIFIED: Chirality follows initial conditions, NOT algebra.");
                println!("  Config B produces stable OPPOSITE-handed ring.");
            } else if !a_stable {
                println!("INCONCLUSIVE: Baseline (Config A) is also unstable.");
                println!("  Need larger grid or more steps to resolve.");
            } else {
                println!("AMBIGUOUS: Mixed stability pattern -- needs further analysis.");
                for (label, hel, stable) in &results {
                    let s = if *stable { "STABLE" } else { "UNSTABLE" };
                    let h = if *hel > 0.0 { "R" } else { "L" };
                    println!("  {label}: {s}, helicity={h}");
                }
            }
        }
        Cmd::Single { mode, grid, steps } => {
            let m = parse_mode(&mode);
            println!("=== Single Config: {mode} ===");
            let (hel, enst, energy, assoc, stable) = run_config(m, grid, steps);
            println!("Helicity: {hel:+.6e}");
            println!("Enstrophy: {enst:.6e}");
            println!("Energy: {energy:.6e}");
            println!("Associator: {assoc:.6e}");
            println!("Stable: {stable}");
        }
    }

    Ok(())
}
