//! Pathion Sink Test: Falsification of the Pathionic entropy dump hypothesis.
//!
//! Tests whether adding a Pathionic heat sink (virtual dim-32 absorber) stabilizes
//! the warp ring simulation beyond its natural instability onset.
//!
//! The Sedenion field (dim=16) is truncated at the Cayley-Dickson doubling boundary.
//! Energy that would propagate into dimensions 16..31 is modeled as a scalar
//! damping term proportional to the local associator norm squared.

use clap::{Parser, Subcommand};
use gororoba_algebra::construction::cayley_dickson::cd_multiply;
use gororoba_engine::simulation::PathionSink;
use std::io::Write;

#[derive(Parser)]
#[command(
    name = "pathion-sink-test",
    about = "Pathion entropy sink falsification"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run stability comparison: with vs without Pathion sink.
    Compare {
        /// Grid dimension (N^3).
        #[arg(long, default_value = "32")]
        grid: usize,
        /// Maximum simulation steps.
        #[arg(long, default_value = "500")]
        max_steps: usize,
        /// Pathion coupling constant.
        #[arg(long, default_value = "0.1")]
        coupling: f64,
        /// Pathion damping rate.
        #[arg(long, default_value = "0.01")]
        damping: f64,
        /// Output CSV path.
        #[arg(long, default_value = "data/csv/pathion_sink_compare.csv")]
        output: String,
    },
    /// Sweep coupling parameter to find stabilization threshold.
    Sweep {
        #[arg(long, default_value = "32")]
        grid: usize,
        #[arg(long, default_value = "500")]
        max_steps: usize,
        #[arg(long, default_value = "data/csv/pathion_coupling_sweep.csv")]
        output: String,
    },
}

/// Sedenion field on 3D grid: one [f64; 16] per cell.
struct SedenionField {
    data: Vec<[f64; 16]>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl SedenionField {
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let n = nx * ny * nz;
        let mut data = vec![[0.0f64; 16]; n];
        // Seed: toroidal vortex ring initial condition
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;
        let ring_r = nx as f64 / 4.0;
        let sigma = nx as f64 / 8.0;
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
                    // Seed first 4 imaginary components (sedenion e1..e4)
                    data[idx][1] = amp * (dy / (rho + 1e-10));
                    data[idx][2] = amp * (-dx / (rho + 1e-10));
                    data[idx][3] = amp * dz / (sigma + 1e-10);
                    data[idx][4] = amp * 0.1; // Small e4 seed for non-associativity
                }
            }
        }
        Self { data, nx, ny, nz }
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.nx * (y + self.ny * z)
    }

    /// Compute field energy (sum of norm squared).
    fn total_energy(&self) -> f64 {
        self.data
            .iter()
            .map(|s| s.iter().map(|v| v * v).sum::<f64>())
            .sum()
    }

    /// Simple diffusive evolution step with optional Pathion sink.
    #[allow(clippy::needless_range_loop)] // multi-array stencil: c indexes old[6 neighbors][c] + self.data[idx][c]
    fn step(&mut self, dt: f64, diffusion: f64, sink: &mut Option<PathionSink>) {
        let old = self.data.clone();
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
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

                    // Non-linear interaction: Sedenion self-product
                    let phi = self.data[idx];
                    let phi2 = cd_multiply(&phi, &phi);
                    for c in 0..16 {
                        self.data[idx][c] += dt * 0.01 * phi2[c];
                    }

                    // Pathion sink feedback
                    if let Some(s) = sink {
                        // Use magnitude of imaginary part as proxy for associator
                        let imag_norm: f64 = self.data[idx][1..].iter().map(|v| v * v).sum::<f64>();
                        let feedback = s.absorb(imag_norm.sqrt(), dt);
                        for c in 1..16 {
                            self.data[idx][c] *= 1.0 + feedback;
                        }
                    }
                }
            }
        }
    }
}

/// Run a simulation and return (instability_onset_step, final_energy, energy_history).
fn run_simulation(
    grid: usize,
    max_steps: usize,
    sink: &mut Option<PathionSink>,
) -> (usize, f64, Vec<f64>) {
    let mut field = SedenionField::new(grid, grid, grid);
    let dt = 0.005;
    let diffusion = 0.1;
    let initial_energy = field.total_energy();
    let instability_threshold = initial_energy * 100.0; // 100x growth = instability

    let mut energy_history = Vec::with_capacity(max_steps);
    let mut instability_onset = max_steps; // default: survived all steps

    for step in 0..max_steps {
        field.step(dt, diffusion, sink);
        let energy = field.total_energy();
        energy_history.push(energy);

        if energy > instability_threshold || energy.is_nan() || energy.is_infinite() {
            instability_onset = step;
            break;
        }
    }

    let final_energy = *energy_history.last().unwrap_or(&0.0);
    (instability_onset, final_energy, energy_history)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Compare {
            grid,
            max_steps,
            coupling,
            damping,
            output,
        } => {
            println!("=== Pathion Sink Stability Comparison ===");
            println!("Grid: {grid}^3, Max steps: {max_steps}");
            println!("Coupling: {coupling}, Damping: {damping}");

            // Run WITHOUT sink
            println!("\n--- Without Pathion sink ---");
            let mut no_sink: Option<PathionSink> = None;
            let (onset_no, energy_no, hist_no) = run_simulation(grid, max_steps, &mut no_sink);
            println!("Instability onset: step {onset_no}");
            println!("Final energy: {energy_no:.6e}");

            // Run WITH sink
            println!("\n--- With Pathion sink ---");
            let mut with_sink = Some(PathionSink::new(coupling, damping));
            let (onset_with, energy_with, hist_with) =
                run_simulation(grid, max_steps, &mut with_sink);
            println!("Instability onset: step {onset_with}");
            println!("Final energy: {energy_with:.6e}");
            if let Some(ref s) = with_sink {
                println!("Pathion absorbed: {:.6e}", s.accumulated);
            }

            // Write CSV
            let mut file = std::fs::File::create(&output)?;
            writeln!(file, "step,energy_no_sink,energy_with_sink")?;
            let len = hist_no.len().max(hist_with.len());
            for i in 0..len {
                let e_no = hist_no.get(i).copied().unwrap_or(f64::NAN);
                let e_with = hist_with.get(i).copied().unwrap_or(f64::NAN);
                writeln!(file, "{i},{e_no:.10e},{e_with:.10e}")?;
            }
            println!("\nWrote {output}");

            // Verdict
            println!("\n=== C-776 Verdict ===");
            if onset_with > onset_no && onset_with >= max_steps {
                println!("CONFIRMED: Pathion sink stabilizes the ring.");
                println!("  Without sink: unstable at step {onset_no}");
                println!("  With sink: survived all {max_steps} steps");
            } else if onset_with > onset_no {
                println!("PARTIAL: Pathion sink delays instability.");
                println!("  Without: step {onset_no}, With: step {onset_with}");
            } else {
                println!("FALSIFIED: Pathion sink does NOT stabilize the ring.");
                println!("  Without: step {onset_no}, With: step {onset_with}");
            }
        }
        Cmd::Sweep {
            grid,
            max_steps,
            output,
        } => {
            println!("=== Pathion Coupling Sweep ===");
            let couplings = [0.01, 0.05, 0.1, 0.5, 1.0];
            let damping = 0.01;

            // Baseline without sink
            let mut no_sink: Option<PathionSink> = None;
            let (onset_base, _, _) = run_simulation(grid, max_steps, &mut no_sink);
            println!("Baseline (no sink): instability at step {onset_base}");

            let mut file = std::fs::File::create(&output)?;
            writeln!(
                file,
                "coupling,damping,instability_onset,final_energy,absorbed"
            )?;

            for &c in &couplings {
                let mut sink = Some(PathionSink::new(c, damping));
                let (onset, energy, _) = run_simulation(grid, max_steps, &mut sink);
                let absorbed = sink.as_ref().map_or(0.0, |s| s.accumulated);
                writeln!(file, "{c},{damping},{onset},{energy:.10e},{absorbed:.10e}")?;
                println!(
                    "  coupling={c:.2}: onset={onset}, energy={energy:.2e}, absorbed={absorbed:.2e}"
                );
            }

            println!("\nWrote {output}");

            // Check monotonicity
            let onsets: Vec<usize> = couplings
                .iter()
                .map(|&c| {
                    let mut s = Some(PathionSink::new(c, damping));
                    run_simulation(grid, max_steps, &mut s).0
                })
                .collect();
            let monotone = onsets.windows(2).all(|w| w[1] >= w[0]);
            if monotone {
                println!("Instability onset is MONOTONICALLY increasing with coupling.");
            } else {
                println!("Instability onset is NOT monotone -- non-trivial coupling dependence.");
            }
        }
    }

    Ok(())
}
