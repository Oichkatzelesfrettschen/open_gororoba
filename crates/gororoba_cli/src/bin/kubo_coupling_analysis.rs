//! First-principles viscosity coupling from Kubo linear-response transport.
//!
//! Computes how the spin Drude weight D_S suppresses with frustration parameter
//! lambda, deriving the functional form of the frustration-viscosity coupling
//! from exact diagonalization instead of ad hoc models.
//!
//! The key quantity: g(lambda) = D_S(0) / D_S(lambda) = viscosity enhancement ratio.
//! This replaces the tautological `nu = nu_base * (1 + alpha * f)` with a
//! first-principles result from quantum spin transport.
//!
//! Also computes K_th (thermal Drude weight) and C_V for the thermal conductivity
//! channel: kappa ~ K_th / T, giving nu_th ~ T * C_V / K_th.

use std::fs;
use std::io;
use std::path::Path;
use vacuum_frustration::kubo_transport::{
    build_cd_heisenberg, build_interpolated, exact_diagonalize, graph_frustration_index,
    kubo_transport_optimized, thermodynamic_quantities,
};

/// Transport computation dispatcher: GPU-first, CPU fallback.
struct TransportDispatcher {
    #[cfg(feature = "gpu")]
    gpu_ctx: Option<vacuum_frustration::kubo_transport_gpu::GpuKuboContext>,
    #[allow(dead_code)]
    use_gpu: bool,
}

impl TransportDispatcher {
    fn new() -> Self {
        #[cfg(feature = "gpu")]
        {
            match vacuum_frustration::kubo_transport_gpu::GpuKuboContext::new() {
                Ok(ctx) => {
                    println!("  Backend: GPU (cuSOLVER + cuBLAS)");
                    Self {
                        gpu_ctx: Some(ctx),
                        use_gpu: true,
                    }
                }
                Err(e) => {
                    println!("  Backend: CPU (GPU init failed: {})", e);
                    Self {
                        gpu_ctx: None,
                        use_gpu: false,
                    }
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("  Backend: CPU (optimized O(n^3))");
            Self { use_gpu: false }
        }
    }
}

/// Result at a single (lambda, temperature) point.
struct CouplingPoint {
    lambda: f64,
    temperature: f64,
    frustration: f64,
    drude_spin: f64,
    drude_energy: f64,
    thermal_conductivity: f64,
    total_weight_spin: f64,
    total_weight_energy: f64,
    specific_heat: f64,
}

fn compute_point(
    dispatcher: &TransportDispatcher,
    cd_dim: usize,
    lambda: f64,
    temperature: f64,
) -> CouplingPoint {
    let cd_model = build_cd_heisenberg(cd_dim, 0.0);
    let model = build_interpolated(&cd_model, lambda);
    let frustration = graph_frustration_index(&model);

    let transport;
    #[cfg(feature = "gpu")]
    {
        if let Some(ref ctx) = dispatcher.gpu_ctx {
            if let Ok(t) = ctx.kubo_transport(&model, temperature, 1e-10) {
                    transport = t;
                    return CouplingPoint {
                        lambda,
                        temperature,
                        frustration,
                        drude_spin: transport.drude_weight_spin,
                        drude_energy: transport.drude_weight_energy,
                        thermal_conductivity: transport.thermal_conductivity,
                        total_weight_spin: transport.total_weight_spin,
                        total_weight_energy: transport.total_weight_energy,
                        specific_heat: {
                            let ed = exact_diagonalize(&model);
                            thermodynamic_quantities(&ed, temperature).specific_heat
                        },
                    };
            }
        }
    }
    let _ = &dispatcher; // suppress unused warning on non-GPU builds

    transport = kubo_transport_optimized(&model, temperature, 1e-10);
    let ed = exact_diagonalize(&model);
    let thermo = thermodynamic_quantities(&ed, temperature);

    CouplingPoint {
        lambda,
        temperature,
        frustration,
        drude_spin: transport.drude_weight_spin,
        drude_energy: transport.drude_weight_energy,
        thermal_conductivity: transport.thermal_conductivity,
        total_weight_spin: transport.total_weight_spin,
        total_weight_energy: transport.total_weight_energy,
        specific_heat: thermo.specific_heat,
    }
}

/// Least-squares fit: y = a * x^b (power law) on log-log scale.
/// Returns (a, b, r_squared).
fn fit_power_law(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mut sum_lnx = 0.0;
    let mut sum_lny = 0.0;
    let mut sum_lnx2 = 0.0;
    let mut sum_lnx_lny = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi <= 0.0 || yi <= 0.0 {
            continue;
        }
        let lx = xi.ln();
        let ly = yi.ln();
        sum_lnx += lx;
        sum_lny += ly;
        sum_lnx2 += lx * lx;
        sum_lnx_lny += lx * ly;
    }

    let b = (n * sum_lnx_lny - sum_lnx * sum_lny) / (n * sum_lnx2 - sum_lnx * sum_lnx);
    let ln_a = (sum_lny - b * sum_lnx) / n;
    let a = ln_a.exp();

    // R-squared
    let mean_lny = sum_lny / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi <= 0.0 || yi <= 0.0 {
            continue;
        }
        let predicted = (a * xi.powf(b)).ln();
        let actual = yi.ln();
        ss_res += (actual - predicted).powi(2);
        ss_tot += (actual - mean_lny).powi(2);
    }
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (a, b, r2)
}

/// Least-squares fit: y = a * exp(b * x) on log-linear scale.
/// Returns (a, b, r_squared).
fn fit_exponential(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_lny = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_x_lny = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if yi <= 0.0 {
            continue;
        }
        let ly = yi.ln();
        sum_x += xi;
        sum_lny += ly;
        sum_x2 += xi * xi;
        sum_x_lny += xi * ly;
    }

    let b = (n * sum_x_lny - sum_x * sum_lny) / (n * sum_x2 - sum_x * sum_x);
    let ln_a = (sum_lny - b * sum_x) / n;
    let a = ln_a.exp();

    // R-squared
    let mean_lny = sum_lny / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if yi <= 0.0 {
            continue;
        }
        let predicted = (a * (b * xi).exp()).ln();
        let actual = yi.ln();
        ss_res += (actual - predicted).powi(2);
        ss_tot += (actual - mean_lny).powi(2);
    }
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (a, b, r2)
}

fn main() -> io::Result<()> {
    let output_dir = Path::new("data/kubo_transport");
    fs::create_dir_all(output_dir)?;

    println!("Kubo First-Principles Viscosity Coupling Analysis");
    println!("==================================================");
    let dispatcher = TransportDispatcher::new();
    println!();

    let cd_dim = 8; // Octonion: 7 sites, 128 Hilbert states (fast on GPU)
    let n_lambda = 100;
    let temperatures = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

    // -----------------------------------------------------------------------
    // Section 1: Fine lambda sweep at primary temperature (T=0.5)
    // -----------------------------------------------------------------------
    let t_primary = 0.5;
    println!("[1/4] Fine lambda sweep: dim={}, T={}, {} points", cd_dim, t_primary, n_lambda + 1);
    println!("------------------------------------------------------");

    let mut primary_points: Vec<CouplingPoint> = Vec::with_capacity(n_lambda + 1);
    for i in 0..=n_lambda {
        let lambda = i as f64 / n_lambda as f64;
        let point = compute_point(&dispatcher, cd_dim, lambda, t_primary);
        if i % 10 == 0 || i == n_lambda {
            println!(
                "  lambda={:.2}: f={:.4}, D_S={:.6e}, K_th={:.6e}, I0_S={:.6e}",
                lambda, point.frustration, point.drude_spin,
                point.thermal_conductivity, point.total_weight_spin
            );
        }
        primary_points.push(point);
    }

    // Reference values at lambda=0
    let d_s_ref = primary_points[0].drude_spin;
    let k_th_ref = primary_points[0].thermal_conductivity;
    let i0_s_ref = primary_points[0].total_weight_spin;

    println!();
    println!("  Reference (lambda=0): D_S={:.6e}, K_th={:.6e}, I0_S={:.6e}",
             d_s_ref, k_th_ref, i0_s_ref);

    // Compute ratios
    println!();
    println!("  Drude suppression ratio D_S(0)/D_S(lambda):");
    let mut lambdas_for_fit: Vec<f64> = Vec::new();
    let mut drude_ratios: Vec<f64> = Vec::new();
    let mut i0s_ratios: Vec<f64> = Vec::new();

    for p in &primary_points {
        let drude_ratio = if p.drude_spin.abs() > 1e-30 {
            d_s_ref / p.drude_spin
        } else {
            f64::MAX
        };
        let i0s_ratio = if p.total_weight_spin.abs() > 1e-30 {
            i0_s_ref / p.total_weight_spin
        } else {
            f64::MAX
        };

        if ((p.lambda * 100.0).round() as usize).is_multiple_of(10) {
            println!(
                "    lambda={:.2}: g_D={:.2}, g_I0S={:.4}, K_th={:.6e}, C_V={:.4}",
                p.lambda, drude_ratio.min(1e6), i0s_ratio,
                p.thermal_conductivity, p.specific_heat
            );
        }

        if p.lambda > 0.005 && drude_ratio < 1e6 {
            lambdas_for_fit.push(p.lambda);
            drude_ratios.push(drude_ratio);
            i0s_ratios.push(i0s_ratio);
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 2: Temperature dependence
    // -----------------------------------------------------------------------
    println!("[2/4] Temperature dependence at selected lambda values");
    println!("------------------------------------------------------");

    let lambda_probes = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
    let mut temp_dep_points: Vec<CouplingPoint> = Vec::new();

    for &lam in &lambda_probes {
        print!("  lambda={:.2}: ", lam);
        for &t in &temperatures {
            let point = compute_point(&dispatcher, cd_dim, lam, t);
            print!("D_S(T={})={:.4e} ", t, point.drude_spin);
            temp_dep_points.push(point);
        }
        println!();
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 3: Analytical fits to the Drude ratio g(lambda)
    // -----------------------------------------------------------------------
    println!("[3/4] Analytical fits to g(lambda) = D_S(0)/D_S(lambda)");
    println!("--------------------------------------------------------");

    // Skip the initial sharp drop region (lambda < 0.02) for fitting
    let fit_start = 2; // skip first two points
    let fit_lambdas: Vec<f64> = lambdas_for_fit[fit_start..].to_vec();
    let fit_ratios: Vec<f64> = drude_ratios[fit_start..].to_vec();

    // Fit 1: Power law g(lambda) = a * lambda^b
    let (a_pow, b_pow, r2_pow) = fit_power_law(&fit_lambdas, &fit_ratios);
    println!("  Power law: g(lambda) = {:.4} * lambda^{:.4}, R^2={:.6}", a_pow, b_pow, r2_pow);

    // Fit 2: Exponential g(lambda) = a * exp(b * lambda)
    let (a_exp, b_exp, r2_exp) = fit_exponential(&fit_lambdas, &fit_ratios);
    println!("  Exponential: g(lambda) = {:.4} * exp({:.4} * lambda), R^2={:.6}", a_exp, b_exp, r2_exp);

    // Fit 3: I0_S ratio (bounded, well-behaved)
    let (a_i0s, b_i0s, r2_i0s) = fit_power_law(&fit_lambdas, &i0s_ratios[fit_start..]);
    println!("  I0_S power law: g_I0S(lambda) = {:.4} * lambda^{:.4}, R^2={:.6}", a_i0s, b_i0s, r2_i0s);

    // Compute ballistic fraction B(lambda) = D_S / I0_S
    println!();
    println!("  Ballistic fraction B(lambda) = D_S(lambda) / I0_S(lambda):");
    for p in &primary_points {
        if ((p.lambda * 100.0).round() as usize).is_multiple_of(10) {
            let b_frac = if p.total_weight_spin.abs() > 1e-30 {
                p.drude_spin / p.total_weight_spin
            } else {
                0.0
            };
            println!(
                "    lambda={:.2}: B={:.6} ({}% ballistic)",
                p.lambda, b_frac, (b_frac * 100.0).round()
            );
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 4: Recommended coupling parameters
    // -----------------------------------------------------------------------
    println!("[4/4] Recommended first-principles coupling parameters");
    println!("------------------------------------------------------");

    // The first-principles coupling formula:
    // nu(f) = nu_base * g(f) where g(f) = D_S(0) / D_S(lambda_eff(f))
    // For the sedenion vacuum (dim=16), f ~ 0.375-0.385
    // Map: lambda_eff = f / f_cd where f_cd is the CD frustration index

    let f_cd = graph_frustration_index(&build_cd_heisenberg(cd_dim, 0.0));
    println!("  CD dim={} frustration index: f_cd = {:.6}", cd_dim, f_cd);

    // At f = 0.375 (vacuum attractor):
    let lambda_at_vac = 0.375 / f_cd;
    println!("  lambda at vacuum attractor (f=0.375): {:.4}", lambda_at_vac);

    // Get D_S at lambda_at_vac by interpolation from the table
    let idx_f = lambda_at_vac * n_lambda as f64;
    let idx = (idx_f as usize).min(n_lambda - 1);
    let frac = idx_f - idx as f64;
    let d_s_at_vac = primary_points[idx].drude_spin * (1.0 - frac)
        + primary_points[idx + 1].drude_spin * frac;
    let g_at_vac = d_s_ref / d_s_at_vac.max(1e-30);
    println!("  D_S at vacuum attractor: {:.6e}", d_s_at_vac);
    println!("  g(0.375) = D_S(0)/D_S(0.375) = {:.2}", g_at_vac);
    println!();

    // Recommended ViscosityCouplingModel parameters
    println!("  RECOMMENDED COUPLING MODELS (derived from Kubo):");
    println!();

    // Option A: Use the Drude ratio directly (lookup table)
    println!("  A) Drude ratio lookup table (101 points):");
    println!("     nu(f) = nu_base * g(f/f_cd) where g comes from D_S(0)/D_S(lambda)");
    println!("     Range: g(0) = 1.0 to g(1.0) = {:.1}", d_s_ref / primary_points.last().unwrap().drude_spin.max(1e-30));
    println!();

    // Option B: Exponential fit (if R^2 > 0.9)
    if r2_exp > 0.9 {
        println!("  B) Exponential fit (R^2={:.4}):", r2_exp);
        // Map from Kubo lambda to bridge frustration f:
        // g(f) = a_exp * exp(b_exp * f / f_cd)
        // For the Exponential ViscosityCouplingModel:
        // nu = nu_base * exp(-lambda * (F - F0)^2) where lambda is the coupling strength
        // This doesn't match our form. We need: nu = nu_base * a * exp(b * f)
        println!("     g(f) = {:.4} * exp({:.4} * f / {:.4})", a_exp, b_exp, f_cd);
        let eff_b = b_exp / f_cd;
        println!("     Effective: g(f) = {:.4} * exp({:.4} * f)", a_exp, eff_b);
    }
    println!();

    // Option C: Power law fit (if R^2 > 0.9)
    if r2_pow > 0.9 {
        println!("  C) Power law fit (R^2={:.4}):", r2_pow);
        println!("     g(f) = {:.4} * (f / {:.4})^{:.4}", a_pow, f_cd, b_pow);
    }
    println!();

    // Key physical result: the onset jump
    let d_s_at_001 = primary_points[1].drude_spin; // lambda = 0.01
    let onset_ratio = d_s_ref / d_s_at_001.max(1e-30);
    println!("  KEY RESULT: Onset ratio D_S(0)/D_S(0.01) = {:.1}", onset_ratio);
    println!("  ANY amount of CD frustration causes a {:.0}x jump in effective viscosity.", onset_ratio);
    println!("  After onset, g(lambda) grows slowly as lambda^{:.2}", b_pow);
    println!();

    // -----------------------------------------------------------------------
    // Write TOML output
    // -----------------------------------------------------------------------
    let toml_path = output_dir.join("coupling_analysis.toml");
    let mut toml = String::new();
    toml.push_str("# Kubo first-principles viscosity coupling analysis\n");
    toml.push_str("# D_S(lambda=0)/D_S(lambda) = viscosity enhancement ratio\n");
    toml.push_str("# Derived from exact diagonalization of CD dim=8 Heisenberg model\n\n");

    toml.push_str("[metadata]\n");
    toml.push_str(&format!("cd_dim = {}\n", cd_dim));
    toml.push_str(&format!("n_lambda = {}\n", n_lambda));
    toml.push_str(&format!("primary_temperature = {}\n", t_primary));
    toml.push_str(&format!("frustration_cd = {}\n", f_cd));
    toml.push_str(&format!("d_s_reference = {}\n", d_s_ref));
    toml.push_str(&format!("backend = \"{}\"\n\n", if dispatcher.use_gpu { "GPU" } else { "CPU" }));

    toml.push_str("[fits]\n");
    toml.push_str(&format!("power_law_a = {}\n", a_pow));
    toml.push_str(&format!("power_law_b = {}\n", b_pow));
    toml.push_str(&format!("power_law_r2 = {}\n", r2_pow));
    toml.push_str(&format!("exponential_a = {}\n", a_exp));
    toml.push_str(&format!("exponential_b = {}\n", b_exp));
    toml.push_str(&format!("exponential_r2 = {}\n", r2_exp));
    toml.push_str(&format!("i0s_power_law_a = {}\n", a_i0s));
    toml.push_str(&format!("i0s_power_law_b = {}\n", b_i0s));
    toml.push_str(&format!("i0s_power_law_r2 = {}\n\n", r2_i0s));

    // Primary sweep data
    for p in &primary_points {
        toml.push_str("[[coupling_sweep]]\n");
        toml.push_str(&format!("lambda = {}\n", p.lambda));
        toml.push_str(&format!("frustration = {}\n", p.frustration));
        toml.push_str(&format!("drude_spin = {}\n", p.drude_spin));
        toml.push_str(&format!("drude_energy = {}\n", p.drude_energy));
        toml.push_str(&format!("thermal_conductivity = {}\n", p.thermal_conductivity));
        toml.push_str(&format!("total_weight_spin = {}\n", p.total_weight_spin));
        toml.push_str(&format!("total_weight_energy = {}\n", p.total_weight_energy));
        toml.push_str(&format!("specific_heat = {}\n", p.specific_heat));
        let drude_ratio = if p.drude_spin.abs() > 1e-30 {
            d_s_ref / p.drude_spin
        } else {
            -1.0 // sentinel for infinite ratio
        };
        toml.push_str(&format!("drude_ratio = {}\n\n", drude_ratio));
    }

    // Temperature dependence data
    for p in &temp_dep_points {
        toml.push_str("[[temperature_dependence]]\n");
        toml.push_str(&format!("lambda = {}\n", p.lambda));
        toml.push_str(&format!("temperature = {}\n", p.temperature));
        toml.push_str(&format!("drude_spin = {}\n", p.drude_spin));
        toml.push_str(&format!("thermal_conductivity = {}\n", p.thermal_conductivity));
        toml.push_str(&format!("specific_heat = {}\n\n", p.specific_heat));
    }

    fs::write(&toml_path, &toml)?;
    println!("Output: {}", toml_path.display());
    println!("\nDone.");

    Ok(())
}
