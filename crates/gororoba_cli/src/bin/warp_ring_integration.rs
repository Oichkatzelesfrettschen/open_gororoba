//! Warp Ring Integration: Algebra <-> Fluid <-> P-adic Duality.
//!
//! Full pipeline connecting spectral turbulence to E7 Lie algebra structure,
//! modulated by p-adic ultrametric weights, negative-dimension kernels,
//! and metamaterial spectral filters:
//!
//! 1. Generate Kolmogorov turbulence via D2Q9 LBM solver
//! 2. Extract spectral triads (energy transfer) via 2D FFT
//! 3. Apply p-adic modulation and negative-dimension kernel (warp physics)
//! 4. Apply metamaterial spectral filter (ZD -> TMM reflectance)
//! 5. Map triads to E7 Lie algebra roots
//! 6. Build hypergraph and compute topological invariants
//! 7. Simulate warp lensing (GRIN ray tracing)
//! 8. Visualize the composite "Warp Ring"

use algebra_core::lie::e7_geometry::{find_e7_triads, generate_e7_roots, project_to_plane};
use lbm_core::simulate_kolmogorov_flow;
use lbm_core::turbulence::{extract_dominant_triads, power_spectrum};
use log::info;
use materials_core::{
    build_absorber_stack, canonical_sedenion_zd_pairs, tmm_reflection, verify_physical_realizability,
};
use num_complex::Complex64;
use optics_core::grin::{trace_ray, GrinMedium, Ray};
use plotters::prelude::*;
use plotters::style::full_palette::GREY;
use spectral_core::ndfft::{fft_2d, real_to_complex_2d};
use spectral_core::warp_physics::{
    apply_neg_dim_kernel, extract_warp_triads, padic_power_spectrum, warp_spectral_density,
    WarpRingConfig,
};
use stats_core::hypergraph::TriadHypergraph;
use std::f64::consts::PI;

/// A simple GRIN medium representing a "warp" potential.
/// n(r) = 1.0 + A * exp(-r^2 / sigma^2)
struct WarpMedium {
    amplitude: f64,
    sigma: f64,
}

impl GrinMedium for WarpMedium {
    fn gradient_and_n(&self, pos: [f64; 3]) -> ([f64; 3], f64) {
        let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        let n_val = 1.0 + self.amplitude * (-r2 / (self.sigma * self.sigma)).exp();

        let factor = (n_val - 1.0) * (-2.0 / (self.sigma * self.sigma));
        let grad = [pos[0] * factor, pos[1] * factor, pos[2] * factor];

        (grad, n_val)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("=== Warp Ring Integration (P-adic + Neg-Dim + E7) ===");

    // -- Configuration --
    let warp_config = WarpRingConfig {
        prime: 2,
        alpha: -0.5,     // Negative-dimension: anti-diffusive smoothing
        epsilon: 0.01,   // IR regularization
        domain_size: 2.0 * PI,
    };

    // -- Step 1: Generate Kolmogorov Flow via D2Q9 LBM --
    let nx = 64;
    let ny = 64;
    let lbm_tau = 0.8; // nu = (tau - 0.5)/3 = 0.1
    let lbm_force = 1e-4;
    let lbm_steps = 2000;
    info!(
        "[1/7] Running D2Q9 LBM Kolmogorov flow ({}x{}, tau={}, F={:.0e}, {} steps)...",
        nx, ny, lbm_tau, lbm_force, lbm_steps
    );
    let flow = simulate_kolmogorov_flow(nx, ny, lbm_tau, lbm_force, 1, lbm_steps);
    let u = flow.ux;
    let v = flow.uy;
    info!(
        "      Enstrophy = {:.6e}, viscosity = {:.4}",
        flow.enstrophy, flow.viscosity
    );
    let (k_axis, power) = power_spectrum(&u);
    if let Some(p0) = power.get(1).copied() {
        info!("      Spectrum diagnostic: P(k=1) = {:.6}", p0);
    }

    // -- Step 2: Extract Standard Spectral Triads --
    info!("[2/7] Extracting spectral triads...");
    let spectral_triads = extract_dominant_triads(&u, &v, 50.0);
    info!("      Found {} spectral triads (standard).", spectral_triads.len());

    // -- Step 3: Warp Physics -- P-adic Modulation + Neg-Dim Kernel --
    info!("[3/7] Applying warp physics (p={}, alpha={:.1}, eps={:.3})...",
        warp_config.prime, warp_config.alpha, warp_config.epsilon);

    // FFT the u-field (real -> complex -> 2D FFT via ndrustfft)
    let u_hat = fft_2d(&real_to_complex_2d(&u));

    // Apply negative-dimension kernel
    let u_hat_warp = apply_neg_dim_kernel(&u_hat, &warp_config);
    info!("      Neg-dim kernel applied: DC gain = {:.2}x",
        u_hat_warp[[0, 0]].norm() / u_hat[[0, 0]].norm().max(1e-30));

    // Extract warp triads (with p-adic + neg-dim weights)
    let warp_triads = extract_warp_triads(&u_hat, &warp_config, 1.0);
    info!("      Found {} warp triads (p-adic + neg-dim weighted).", warp_triads.len());

    if let Some(top) = warp_triads.first() {
        info!("      Top triad: k={:?}, padic_w={:.4}, negdim_w={:.4}, warp_w={:.4}",
            top.k, top.padic_weight, top.neg_dim_weight, top.warp_weight);
    }

    // P-adic modulated power spectrum
    let (_k_padic, p_padic) = padic_power_spectrum(&u_hat, 2.0 * PI, warp_config.prime);
    let padic_total: f64 = p_padic.iter().sum();
    info!("      P-adic spectrum total power: {:.6}", padic_total);

    // Warp spectral density
    let warp_density = warp_spectral_density(&k_axis, &power, &warp_config);
    let warp_total: f64 = warp_density.iter().sum();
    info!("      Warp spectral density total: {:.6}", warp_total);

    // -- Step 4: Materials Bridge -- ZD Metamaterial Spectral Filter --
    info!("[4/8] Computing metamaterial spectral filter (ZD -> TMM)...");
    let zd_pairs = canonical_sedenion_zd_pairs();
    let stack = build_absorber_stack(&zd_pairs, 6, 1.5);
    let verification = verify_physical_realizability(&stack);
    info!(
        "      Metamaterial stack: {} layers ({} physical, {} dielectric, {} plasmonic)",
        verification.n_total,
        verification.n_physical,
        verification.n_dielectric,
        verification.n_plasmonic
    );

    // Build TMM reflectance spectrum over the wavenumber range
    // Map turbulence wavenumber k to optical wavelength via lambda = L/k
    // (L = domain size, conceptual correspondence)
    let n_spec = k_axis.len();
    let mut material_weights = vec![1.0_f64; n_spec];
    if !stack.is_empty() {
        let n_layers: Vec<Complex64> = std::iter::once(Complex64::new(1.0, 0.0)) // incidence medium (air)
            .chain(stack.iter().map(|m| Complex64::new(m.layer.n_real, m.layer.n_imag)))
            .chain(std::iter::once(Complex64::new(1.5, 0.0))) // substrate
            .collect();
        let d_layers: Vec<f64> = std::iter::once(0.0) // incidence (semi-infinite)
            .chain(stack.iter().map(|m| m.layer.thickness_nm))
            .chain(std::iter::once(0.0)) // substrate (semi-infinite)
            .collect();

        for (idx, &k_val) in k_axis.iter().enumerate() {
            if k_val > 0.1 {
                // Map turbulence wavenumber to optical wavelength (nm)
                // Using lambda = 1000 / k as a conceptual mapping
                let wavelength_nm = 1000.0 / k_val;
                if wavelength_nm > 50.0 && wavelength_nm < 2000.0 {
                    let tmm = tmm_reflection(&n_layers, &d_layers, wavelength_nm, 0.0, true);
                    // High reflectance -> strong coupling -> higher material weight
                    material_weights[idx] = 1.0 + tmm.reflectance;
                }
            }
        }
    }
    let mat_weight_sum: f64 = material_weights.iter().sum();
    let mat_weight_max: f64 = material_weights.iter().cloned().fold(0.0_f64, f64::max);
    info!(
        "      Material spectral weight: sum={:.3}, max={:.3}",
        mat_weight_sum, mat_weight_max
    );

    // Apply material weights to warp spectral density
    let filtered_density: Vec<f64> = warp_density
        .iter()
        .zip(material_weights.iter())
        .map(|(w, m)| w * m)
        .collect();
    let filtered_total: f64 = filtered_density.iter().sum();
    info!(
        "      Material-filtered warp density: {:.6} (ratio: {:.3}x)",
        filtered_total,
        if warp_total > 0.0 {
            filtered_total / warp_total
        } else {
            0.0
        }
    );

    // -- Step 5: Map to E7 Lie Algebra --
    info!("[5/8] Mapping to E7 geometry...");
    let e7_roots = generate_e7_roots();
    let algebra_triads = find_e7_triads(&e7_roots);
    info!("      E7: {} roots, {} structural triads.", e7_roots.len(), algebra_triads.len());

    // -- Step 6: Build Hypergraph + Topological Invariants --
    info!("[6/8] Building hypergraph...");
    let mut hg = TriadHypergraph::new();

    // Map warp triads to hypergraph vertices via wavevector hash
    for t in warp_triads.iter().take(500) {
        let k_hash = ((t.k[0] + 32) * 64 + (t.k[1] + 32)) as usize;
        let p_hash = ((t.p[0] + 32) * 64 + (t.p[1] + 32)) as usize;
        let q_hash = ((t.q[0] + 32) * 64 + (t.q[1] + 32)) as usize;
        hg.add_triad(k_hash, p_hash, q_hash);
    }

    // Also add algebraic triads
    for (i, _triad) in algebra_triads.iter().enumerate().take(200) {
        hg.add_triad(
            10000 + i,
            10000 + (i + 1) % algebra_triads.len(),
            10000 + (i + 2) % algebra_triads.len(),
        );
    }

    info!("      Hypergraph: {} vertices, {} edges", hg.vertex_count(), hg.edge_count());
    info!("      Clustering coefficient: {:.4}", hg.clustering_coefficient());
    info!("      Betti-0 (components): {}", hg.betti_0());
    info!("      Betti-1 (cycles): {}", hg.betti_1());

    // Select active algebraic triads proportional to warp triad count
    let active_count = warp_triads.len().min(algebra_triads.len()) * 3;
    let active_algebra_triads: Vec<_> = algebra_triads
        .into_iter()
        .take(active_count)
        .collect();

    // -- Step 7: Warp Lensing (GRIN Optics) --
    info!("[7/8] Simulating warp lensing...");
    let warp = WarpMedium {
        amplitude: 0.5,
        sigma: 2.0,
    };
    let mut lensed_roots = Vec::new();

    for r in &e7_roots {
        let (x, y) = project_to_plane(&r.root);
        let start = [x * 0.1, y * 0.1, -10.0];
        let dir = [0.0, 0.0, 1.0];
        let ray = Ray { pos: start, dir };

        let result = trace_ray(ray, &warp, 0.1, 200);

        if let Some(end_pos) = result.positions.last() {
            lensed_roots.push((end_pos[0], end_pos[1]));
        } else {
            lensed_roots.push((x, y));
        }
    }

    // -- Step 8: Render --
    info!("[8/8] Rendering warp ring...");
    let root = BitMapBackend::new("warp_ring_integration.png", (1024, 1024)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Warp Ring: E7 x Turbulence x P-adic",
            ("sans-serif", 36).into_font().color(&WHITE),
        )
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-4.0..4.0, -4.0..4.0)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .axis_style(WHITE)
        .draw()?;

    // Layer 1: E7 roots (grey background)
    chart.draw_series(e7_roots.iter().map(|r| {
        let (x, y) = project_to_plane(&r.root);
        Circle::new((x, y), 2, GREY.filled())
    }))?;

    // Layer 2: Lensed roots (red, warp effect)
    chart.draw_series(
        lensed_roots
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, RED.filled())),
    )?;

    // Layer 3: Active algebraic triads (cyan lines, energy flow)
    for triad in &active_algebra_triads {
        let (k_x, k_y) = project_to_plane(&triad.k.root);
        let (p_x, p_y) = project_to_plane(&triad.p.root);
        let (q_x, q_y) = project_to_plane(&triad.q.root);

        let color = HSLColor(0.6, 1.0, 0.5); // Cyan
        chart.draw_series(LineSeries::new(
            vec![(k_x, k_y), (p_x, p_y), (q_x, q_y), (k_x, k_y)],
            &color.mix(0.15),
        ))?;
    }

    // Layer 4: Top warp triads (green, p-adic modulated)
    for t in warp_triads.iter().take(50) {
        let scale = 4.0 / 32.0; // Map grid indices to plot coordinates
        let kx = t.k[0] as f64 * scale;
        let ky = t.k[1] as f64 * scale;
        let px = t.p[0] as f64 * scale;
        let py = t.p[1] as f64 * scale;
        let qx = t.q[0] as f64 * scale;
        let qy = t.q[1] as f64 * scale;

        let alpha = (t.warp_weight / warp_triads[0].warp_weight).min(1.0);
        let color = HSLColor(0.33, 1.0, 0.4); // Green
        chart.draw_series(LineSeries::new(
            vec![(kx, ky), (px, py), (qx, qy), (kx, ky)],
            &color.mix(alpha * 0.5),
        ))?;
    }

    root.present()?;
    info!("Done. Output saved to 'warp_ring_integration.png'.");

    // -- Summary --
    info!("--- Warp Ring Summary ---");
    info!("  LBM: {}x{}, tau={}, {} steps", nx, ny, lbm_tau, lbm_steps);
    info!("  Spectral triads (standard): {}", spectral_triads.len());
    info!("  Warp triads (p-adic + neg-dim): {}", warp_triads.len());
    info!("  Materials: {} ZD layers, {} physical", verification.n_total, verification.n_physical);
    info!("  E7 algebraic triads: {}", active_algebra_triads.len());
    info!("  Hypergraph: V={}, E={}, C={:.4}, b0={}, b1={}",
        hg.vertex_count(), hg.edge_count(), hg.clustering_coefficient(),
        hg.betti_0(), hg.betti_1());

    Ok(())
}
