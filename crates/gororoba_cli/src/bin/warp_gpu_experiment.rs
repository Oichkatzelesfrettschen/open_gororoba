//! Warp Turbulence Experiment: E7 Viscosity, P-adic Cascades, and Topology.
//!
//! Executes three experiments to validate novel hypotheses in the open_gororoba framework.
//!
//! Experiment A: Algebraic Viscosity (Stability test)
//! Experiment B: P-adic Spectral Tuning (Spectral slope vs prime)
//! Experiment C: Topological Precursor (Betti-1 vs Enstrophy)
//!
//! Usage:
//!   cargo run --bin warp-gpu-experiment --release --features gpu -- --experiment A
//!   cargo run --bin warp-gpu-experiment --release --features gpu -- --experiment B
//!   cargo run --bin warp-gpu-experiment --release --features gpu -- --experiment C

use clap::Parser;
use log::{info, warn};
use ndarray::{Array3, ArrayView3, Zip};
use num_complex::Complex64;
use rayon::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// Import core modules
use algebra_core::lie::e7_geometry::generate_e7_roots;
use lbm_core::{simulate_kolmogorov_flow, D2Q9};
use spectral_core::ndfft::{fft_3d, ifft_3d, real_to_complex_3d};
use spectral_core::warp_physics::{padic_power_spectrum, WarpRingConfig};
use stats_core::hypergraph::TriadHypergraph;

#[cfg(feature = "gpu")]
use lbm_3d_cuda::LbmSolver3DCuda;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Experiment to run (A, B, or C)
    #[arg(short, long)]
    experiment: String,

    /// Grid size (NxN or NxNxN)
    #[arg(long, default_value_t = 64)]
    size: usize,

    /// Steps to simulate
    #[arg(long, default_value_t = 1000)]
    steps: usize,

    /// Reynolds number proxy (1/viscosity)
    #[arg(long, default_value_t = 1000.0)]
    re: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = Args::parse();

    info!("=== Warp Turbulence Experiment: {} ===", args.experiment);
    info!("Grid: {}, Steps: {}, Re: {}", args.size, args.steps, args.re);

    match args.experiment.as_str() {
        "A" => run_experiment_a(&args),
        "B" => run_experiment_b(&args),
        "C" => run_experiment_c(&args),
        _ => Err("Invalid experiment. Use A, B, or C.".into()),
    }
}

// =================================================================================
// Experiment A: Algebraic Viscosity
// =================================================================================

fn run_experiment_a(args: &Args) -> Result<(), Box<dyn Error>> {
    info!("Starting Experiment A: Algebraic Viscosity Test");

    // We compare two runs: Control (No Filter) vs Experimental (E7 Filter)
    // We measure "Time to Blowup" (Enstrophy > 1e6 or NaN)

    let mut results = Vec::new();

    for case in ["Control", "E7-Filter"] {
        info!("Running Case: {}", case);
        let blowup_time = run_lbm_stability_test(args, case == "E7-Filter")?;
        info!("  Blowup Time: {}", blowup_time);
        results.push((case, blowup_time));
    }

    // Save results
    let path = Path::new("data/csv/warp_experiment_a_viscosity.csv");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    writeln!(file, "case,blowup_time")?;
    for (case, time) in results {
        writeln!(file, "{},{}", case, time)?;
    }
    info!("Saved results to {:?}", path);

    Ok(())
}

fn run_lbm_stability_test(args: &Args, use_filter: bool) -> Result<usize, Box<dyn Error>> {
    let nx = args.size;
    let ny = args.size;
    let nz = args.size; // 3D for GPU, or if CPU fallback allows 3D
    let nu = 1.0 / args.re;
    let tau = 3.0 * nu + 0.5;

    // Try GPU first
    #[cfg(feature = "gpu")]
    {
        if let Ok(mut solver) = LbmSolver3DCuda::new(nx, ny, nz, tau) {
            info!("Using GPU LBM Solver (D3Q19)");
            // Initialize with Kolmogorov-like forcing (random initial velocity)
            let mut rng = rand::thread_rng();
            use rand::Rng;
            let u_init: Vec<[f64; 3]> = (0..nx * ny * nz)
                .map(|_| {
                    [
                        rng.gen_range(-0.1..0.1),
                        rng.gen_range(-0.1..0.1),
                        rng.gen_range(-0.1..0.1),
                    ]
                })
                .collect();
            solver.initialize_custom(&vec![1.0; nx * ny * nz], &u_init)?;

            // Generate E7 mask if filtering
            let e7_mask = if use_filter {
                Some(generate_e7_spectral_mask(nx, ny, nz))
            } else {
                None
            };

            for t in 0..args.steps {
                if let Err(e) = solver.step() {
                    warn!("Solver failed at step {}: {}", t, e);
                    return Ok(t);
                }

                // Check stability
                if t % 10 == 0 {
                    solver.sync_to_host()?;
                    let enstrophy = compute_enstrophy_3d(&solver.u, nx, ny, nz);
                    if enstrophy.is_nan() || enstrophy > 1e6 {
                        warn!("Blowup detected at step {} (Enstrophy: {:.2e})", t, enstrophy);
                        return Ok(t);
                    }

                    // Apply Filter
                    if use_filter {
                        let u_vec = &solver.u;
                        // Flatten [ [ux,uy,uz], ... ] to separate arrays for FFT?
                        // ndfft expects Array3.
                        let mut ux = Array3::zeros((nx, ny, nz));
                        let mut uy = Array3::zeros((nx, ny, nz));
                        let mut uz = Array3::zeros((nx, ny, nz));
                        
                        for (idx, vel) in u_vec.iter().enumerate() {
                            let z = idx / (nx * ny);
                            let y = (idx % (nx * ny)) / nx;
                            let x = idx % nx;
                            ux[[x, y, z]] = vel[0];
                            uy[[x, y, z]] = vel[1];
                            uz[[x, y, z]] = vel[2];
                        }

                        // Filter each component
                        apply_filter_3d(&mut ux, e7_mask.as_ref().unwrap());
                        apply_filter_3d(&mut uy, e7_mask.as_ref().unwrap());
                        apply_filter_3d(&mut uz, e7_mask.as_ref().unwrap());

                        // Write back
                        let mut u_new = u_vec.clone();
                        for (idx, vel) in u_new.iter_mut().enumerate() {
                            let z = idx / (nx * ny);
                            let y = (idx % (nx * ny)) / nx;
                            let x = idx % nx;
                            vel[0] = ux[[x, y, z]];
                            vel[1] = uy[[x, y, z]];
                            vel[2] = uz[[x, y, z]];
                        }
                        
                        // Reset solver with filtered velocity
                        solver.initialize_custom(&solver.rho, &u_new)?;
                    }
                }
            }
            return Ok(args.steps);
        } else {
            warn!("GPU Init failed, falling back to CPU (2D)");
        }
    }

    // CPU Fallback (2D D2Q9)
    info!("Using CPU LBM Solver (D2Q9 - 2D Approximation)");
    let mut flow = simulate_kolmogorov_flow(nx, ny, tau, 1e-4, 1, 1); // Init
    
    // We assume simulate_kolmogorov_flow returns a struct with step() or we run loop manually?
    // lbm_core::simulate_kolmogorov_flow runs the WHOLE simulation.
    // We need granular control. We'll use D2Q9 struct directly if possible, 
    // or just run simulate_kolmogorov_flow multiple times? No, that resets state.
    // lbm_core likely doesn't expose a step-by-step public API for external loops easily.
    // Inspecting lbm_core would be needed. 
    // Assuming for now we can't easily hook into CPU loop for filtering without modifying lbm_core.
    // We will return dummy value for CPU case to avoid build errors, or panic.
    
    Err("CPU fallback for interactive filtering not implemented in this experiment script. Please enable GPU feature.".into())
}

// =================================================================================
// Experiment B: P-adic Spectral Tuning
// =================================================================================

fn run_experiment_b(args: &Args) -> Result<(), Box<dyn Error>> {
    info!("Starting Experiment B: P-adic Spectral Tuning");

    // Generate 3D Noise (Kolmogorov-like)
    let nx = args.size;
    let ny = args.size;
    let nz = args.size;
    
    info!("Generating synthetic turbulence ({}x{}x{})...", nx, ny, nz);
    // Simple synthetic field: sum of random waves with k^-5/3 amplitude
    let mut field_hat = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut rng = rand::thread_rng();
    use rand::Rng;

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let kx = if x <= nx/2 { x as f64 } else { x as f64 - nx as f64 };
                let ky = if y <= ny/2 { y as f64 } else { y as f64 - ny as f64 };
                let kz = if z <= nz/2 { z as f64 } else { z as f64 - nz as f64 };
                let k = (kx*kx + ky*ky + kz*kz).sqrt();
                
                if k > 0.0 {
                    let amplitude = k.powf(-5.0/3.0 - 1.0); // -1.0 for 3D integration factor adjustment? roughly.
                    let phase = rng.gen::<f64>() * 2.0 * PI;
                    field_hat[[x, y, z]] = Complex64::from_polar(amplitude, phase);
                }
            }
        }
    }

    // Reduce to 2D slice for reuse of existing 2D p-adic functions
    // or implement 3D p-adic spectrum. Let's implement 3D here.
    
    let mut results = Vec::new();
    for p in [2, 3, 5, 7] {
        info!("Analyzing prime p={}", p);
        let (k_bins, power) = compute_padic_spectrum_3d(&field_hat, p);
        
        // Fit slope
        let (slope, r2) = fit_power_law(&k_bins, &power);
        info!("  p={}: Slope = {:.4}, R2 = {:.4}", p, slope, r2);
        results.push((p, slope, r2));
    }

    // Save
    let path = Path::new("data/csv/warp_experiment_b_padic.csv");
    let mut file = File::create(path)?;
    writeln!(file, "prime,slope,r2")?;
    for (p, s, r) in results {
        writeln!(file, "{},{:.6},{:.6}", p, s, r)?;
    }
    info!("Saved results to {:?}", path);

    Ok(())
}

fn compute_padic_spectrum_3d(field_hat: &Array3<Complex64>, prime: u64) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = field_hat.dim();
    let k_max = nx / 2; // Assume cubic
    let mut power = vec![0.0; k_max];
    let mut counts = vec![0; k_max];

    // Helper for p-adic valuation
    fn vp(n: usize, p: u64) -> i32 {
        if n == 0 { return 0; }
        let mut v = 0;
        let mut m = n;
        while m % (p as usize) == 0 {
            m /= p as usize;
            v += 1;
        }
        v
    }

    for ((x, y, z), val) in field_hat.indexed_iter() {
        let kx = if x <= nx/2 { x as f64 } else { x as f64 - nx as f64 };
        let ky = if y <= ny/2 { y as f64 } else { y as f64 - ny as f64 };
        let kz = if z <= nz/2 { z as f64 } else { z as f64 - nz as f64 };
        let k = (kx*kx + ky*ky + kz*kz).sqrt();
        let bin = k.round() as usize;

        if bin > 0 && bin < k_max {
            let v = vp(bin, prime);
            let weight = (prime as f64).powi(-v);
            power[bin] += val.norm_sqr() * weight;
            counts[bin] += 1;
        }
    }

    let k_bins: Vec<f64> = (0..k_max).map(|i| i as f64).collect();
    // Normalize
    for i in 0..k_max {
        if counts[i] > 0 {
            power[i] /= counts[i] as f64;
        }
    }

    (k_bins, power)
}

fn fit_power_law(x: &[f64], y: &[f64]) -> (f64, f64) {
    // Linear regression on log-log
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut n = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi > 1.0 && yi > 1e-20 {
            let lx = xi.ln();
            let ly = yi.ln();
            sx += lx;
            sy += ly;
            sxx += lx * lx;
            sxy += lx * ly;
            n += 1.0;
        }
    }

    if n < 2.0 { return (0.0, 0.0); }

    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    
    // R2 approximation
    let mean_y = sy / n;
    let ss_tot: f64 = x.iter().zip(y.iter())
        .filter(|(&xi, &yi)| xi > 1.0 && yi > 1e-20)
        .map(|(_, &yi)| (yi.ln() - mean_y).powi(2))
        .sum();
    let ss_res: f64 = x.iter().zip(y.iter())
        .filter(|(&xi, &yi)| xi > 1.0 && yi > 1e-20)
        .map(|(&xi, &yi)| {
            let pred = (sy/n) + slope * (xi.ln() - sx/n); // Simplified intercept usage
            (yi.ln() - pred).powi(2)
        })
        .sum();
    
    let r2 = 1.0 - (ss_res / ss_tot);

    (slope, r2)
}

// =================================================================================
// Experiment C: Topological Precursor
// =================================================================================

fn run_experiment_c(args: &Args) -> Result<(), Box<dyn Error>> {
    info!("Starting Experiment C: Topological Precursor");
    // Run LBM, extract triads, measure Betti-1
    
    let nx = args.size;
    let ny = args.size;
    let nz = args.size;
    let steps = args.steps;
    
    // Output file
    let path = Path::new("data/csv/warp_experiment_c_topology.csv");
    let mut file = File::create(path)?;
    writeln!(file, "step,enstrophy,betti_1,active_triads")?;

    #[cfg(feature = "gpu")]
    {
        let mut solver = LbmSolver3DCuda::new(nx, ny, nz, 0.6)?; // Low viscosity
        solver.initialize_uniform(1.0, [0.05, 0.0, 0.0])?; // Shear init handled internally or via noise
        
        // Add random noise
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut u_init = vec![[0.0; 3]; nx*ny*nz];
        for i in 0..nx*ny*nz {
            u_init[i] = [
                0.05 + rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01)
            ];
        }
        solver.initialize_custom(&vec![1.0; nx*ny*nz], &u_init)?;

        for t in 0..steps {
            solver.step()?;

            if t % 50 == 0 {
                solver.sync_to_host()?;
                let enstrophy = compute_enstrophy_3d(&solver.u, nx, ny, nz);
                
                // Topological Analysis
                // 1. FFT
                // 2. Find triads (k,p,q) such that |u_k||u_p||u_q| > Threshold
                // 3. Build Hypergraph
                let (betti_1, triad_count) = compute_topology(&solver.u, nx, ny, nz);
                
                info!("Step {}: Enstrophy={:.4e}, Betti-1={}, Triads={}", t, enstrophy, betti_1, triad_count);
                writeln!(file, "{},{:.6e},{},{}", t, enstrophy, betti_1, triad_count)?;
            }
        }
    }
    
    info!("Experiment C Complete. Saved to {:?}", path);
    Ok(())
}

fn compute_topology(u_vec: &[[f64; 3]], nx: usize, ny: usize, nz: usize) -> (usize, usize) {
    // Simplified 2D analysis on the z=mid slice for speed, or coarse 3D?
    // Let's do full 3D FFT but threshold aggressively.
    
    let mut ux = Array3::zeros((nx, ny, nz));
    for (idx, vel) in u_vec.iter().enumerate() {
        let z = idx / (nx * ny);
        let y = (idx % (nx * ny)) / nx;
        let x = idx % nx;
        ux[[x, y, z]] = vel[0];
    }
    
    let u_hat = fft_3d(&real_to_complex_3d(&ux));
    
    // Extract triads
    let mut hg = TriadHypergraph::new();
    let threshold = 1.0; // Needs tuning based on energy
    
    // Random sampling of triads to keep it fast?
    // Rigorous way: iterate all k,p. 
    // Optimization: Only consider Top-N modes.
    
    // Find top 50 modes
    let mut modes: Vec<((usize, usize, usize), f64)> = Vec::new();
    for ((x, y, z), val) in u_hat.indexed_iter() {
        let mag = val.norm_sqr();
        if mag > threshold {
            modes.push(((x,y,z), mag));
        }
    }
    modes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    modes.truncate(50);
    
    // Check triads within top modes
    let mut triad_count = 0;
    for i in 0..modes.len() {
        for j in i+1..modes.len() {
            let (k, _) = modes[i];
            let (p, _) = modes[j];
            
            // q = -(k+p)
            let qx = (nx as i32 - (k.0 as i32 + p.0 as i32).rem_euclid(nx as i32)) as usize % nx;
            let qy = (ny as i32 - (k.1 as i32 + p.1 as i32).rem_euclid(ny as i32)) as usize % ny;
            let qz = (nz as i32 - (k.2 as i32 + p.2 as i32).rem_euclid(nz as i32)) as usize % nz;
            
            // Check if q is in top modes (or just high energy)
            let q_mag = u_hat[[qx, qy, qz]].norm_sqr();
            if q_mag > threshold {
                hg.add_triad(
                    hash_k(k.0, k.1, k.2),
                    hash_k(p.0, p.1, p.2),
                    hash_k(qx, qy, qz)
                );
                triad_count += 1;
            }
        }
    }
    
    (hg.betti_1(), triad_count)
}

fn hash_k(x: usize, y: usize, z: usize) -> usize {
    x + 1024 * y + 1024 * 1024 * z
}

// =================================================================================
// Utilities
// =================================================================================

fn compute_enstrophy_3d(u: &[[f64; 3]], nx: usize, ny: usize, nz: usize) -> f64 {
    // Calculate curl (vorticity) and sum squares
    // Finite difference
    let mut enstrophy = 0.0;
    
    // idx = x + y*nx + z*nx*ny
    let idx = |x: usize, y: usize, z: usize| x + y*nx + z*nx*ny;
    
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let xp = (x + 1) % nx;
                let yp = (y + 1) % ny;
                let zp = (z + 1) % nz;
                let xm = (x + nx - 1) % nx;
                let ym = (y + ny - 1) % ny;
                let zm = (z + nz - 1) % nz;
                
                let u_c = u[idx(x,y,z)];
                
                // Partial derivs (central diff)
                // dy_uz - dz_uy
                let dy_uz = (u[idx(x,yp,z)][2] - u[idx(x,ym,z)][2]) * 0.5;
                let dz_uy = (u[idx(x,y,zp)][1] - u[idx(x,y,zm)][1]) * 0.5;
                let wx = dy_uz - dz_uy;
                
                // dz_ux - dx_uz
                let dz_ux = (u[idx(x,y,zp)][0] - u[idx(x,y,zm)][0]) * 0.5;
                let dx_uz = (u[idx(xp,y,z)][2] - u[idx(xm,y,z)][2]) * 0.5;
                let wy = dz_ux - dx_uz;
                
                // dx_uy - dy_ux
                let dx_uy = (u[idx(xp,y,z)][1] - u[idx(xm,y,z)][1]) * 0.5;
                let dy_ux = (u[idx(x,yp,z)][0] - u[idx(x,ym,z)][0]) * 0.5;
                let wz = dx_uy - dy_ux;
                
                enstrophy += wx*wx + wy*wy + wz*wz;
            }
        }
    }
    
    enstrophy / (nx * ny * nz) as f64
}

fn generate_e7_spectral_mask(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    let mut mask = Array3::zeros((nx, ny, nz));
    let roots = generate_e7_roots();
    
    // Project roots to 3D grid
    // E7 is rank 7. We take dimensions 2, 3, 4 to avoid degeneracy (since x0=x1).
    for root in roots {
        let rx = root.root.coords[2];
        let ry = root.root.coords[3];
        let rz = root.root.coords[4];
        
        // Map to grid: centered at nx/2, scale?
        // E7 roots are length sqrt(2). Max coord ~1.
        // We want them to cover low-mid freq.
        let scale = 8.0; 
        
        let gx = ((rx * scale + nx as f64 / 2.0).round() as isize).rem_euclid(nx as isize) as usize;
        let gy = ((ry * scale + ny as f64 / 2.0).round() as isize).rem_euclid(ny as isize) as usize;
        let gz = ((rz * scale + nz as f64 / 2.0).round() as isize).rem_euclid(nz as isize) as usize;
        
        mask[[gx, gy, gz]] = 1.0;
        
        // Symmetry? Roots are symmetric.
    }
    
    // Ensure DC passes
    mask[[0,0,0]] = 1.0;
    
    mask
}

fn apply_filter_3d(field: &mut Array3<f64>, mask: &Array3<f64>) {
    // Real -> Complex
    let complex_field = real_to_complex_3d(field);
    let mut field_hat = fft_3d(&complex_field);
    
    // Apply Mask
    Zip::from(&mut field_hat).and(mask).for_each(|val, &m| {
        if m < 0.5 {
            *val *= 0.95; // Damping, not full cut (Soft Filter)
        }
    });
    
    // IFFT
    let field_filtered = ifft_3d(&field_hat);
    
    // Update real part
    Zip::from(field).and(&field_filtered).for_each(|out, inp| {
        *out = inp.re;
    });
}
