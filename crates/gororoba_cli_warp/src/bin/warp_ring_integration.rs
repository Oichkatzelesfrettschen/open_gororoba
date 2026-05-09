//! Warp Ring Integration: Algebra <-> Fluid <-> Optics <-> P-adic Duality.
//!
//! Full pipeline connecting spectral turbulence to E7 Lie algebra structure,
//! modulated by p-adic ultrametric weights, negative-dimension kernels,
//! metamaterial spectral filters, and GRIN gravitational lensing:
//!
//! 1. Generate Kolmogorov turbulence via D2Q9 LBM solver (Gororoba Engine)
//! 2. Extract spectral triads (energy transfer) via 2D FFT
//! 3. Apply p-adic modulation and negative-dimension kernel (warp physics)
//! 4. Apply metamaterial spectral filter (ZD -> TMM reflectance)
//! 5. GRIN ray tracing through warp-ring effective refractive index (NA-001)
//! 6. Map triads to E7 Lie algebra roots
//! 7. Build hypergraph and compute topological invariants
//! 8. Simulate warp lensing (SHI Integration via Gororoba Engine)
//! 9. Visualize the composite "Warp Ring" + GRIN lensed star field

use gororoba_algebra::{
    lie::e7::geometry::{find_e7_triads, generate_e7_roots, project_to_plane},
    physics::octonion_field::FieldParams,
};
// use gororoba_engine::simulation::AlgebraicField; // Unused import removed
use gororoba_engine::{SimulationConfig, SimulationState};
use gr_core::{kerr::Kerr, sedenion_geodesic::sedenion_homotopy_step};
use lbm_core::{
    CX, W,
    turbulence::{extract_dominant_triads, power_spectrum},
};
use log::info;
use materials_core::{
    build_absorber_stack, canonical_sedenion_zd_pairs, tmm_reflection,
    verify_physical_realizability,
};
// use ndarray::Array2; // Unused import removed
use num_complex::Complex64;
use optics_core::{
    grin::{GrinMedium, Ray, Vec3, trace_ray},
    tcmt::{InputField, KerrCavity, TcmtSolver},
};
use plotters::{prelude::*, style::full_palette::GREY};
use spectral_core::{
    ndfft::{fft_2d, real_to_complex_2d},
    warp_physics::{
        WarpRingConfig, apply_neg_dim_kernel, extract_warp_triads, padic_power_spectrum,
        warp_spectral_density,
    },
};
use stats_core::hypergraph::TriadHypergraph;
use std::f64::consts::PI;

/// GRIN medium modeling the warp ring as a toroidal refractive-index perturbation.
///
/// Maps the Kerr metric near a toroidal mass distribution to an effective
/// refractive index via the transformation optics / Gordon metric analogy:
///   n_eff(r) ~ 1 + 2*Phi(r)/c^2  (weak field)
///
/// The profile is a Gaussian torus: n(x,y,z) = 1 + delta_n * exp(-d^2/sigma^2)
/// where d is the distance from the ring center circle of radius R in the xy-plane.
/// The Kerr spin parameter introduces azimuthal frame-dragging asymmetry.
struct WarpRingGrinMedium {
    /// Major radius of the warp ring torus.
    ring_radius: f64,
    /// Gaussian width of the refractive index perturbation.
    sigma: f64,
    /// Peak refractive index perturbation above vacuum (n=1).
    delta_n: f64,
    /// Kerr spin parameter: modulates azimuthal asymmetry via frame dragging.
    a_spin: f64,
}

impl GrinMedium for WarpRingGrinMedium {
    fn gradient_and_n(&self, p: Vec3) -> (Vec3, f64) {
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let z = p[2];

        // Distance squared from the torus center ring
        let dr = rho - self.ring_radius;
        let d2 = dr * dr + z * z;

        // Frame-dragging asymmetry: phi-dependent perturbation from Kerr spin
        // Uses cos(phi) modulation scaled by a_spin (0 = symmetric, 0.95 = strong)
        let phi = p[1].atan2(p[0]);
        let drag_factor = 1.0 + 0.1 * self.a_spin * phi.cos();

        let gauss = (-d2 / (self.sigma * self.sigma)).exp();
        let n = 1.0 + self.delta_n * drag_factor * gauss;

        // Analytic gradient via chain rule
        let base = -2.0 * self.delta_n * drag_factor * gauss / (self.sigma * self.sigma);
        let rho_safe = rho.max(1e-10);

        // d(d2)/dx = 2*dr*(x/rho), d(d2)/dy = 2*dr*(y/rho), d(d2)/dz = 2*z
        let grad_x = base * dr * p[0] / rho_safe;
        let grad_y = base * dr * p[1] / rho_safe;
        let grad_z = base * z;

        // Azimuthal gradient contribution from frame dragging
        // d(drag)/dphi = -0.1*a_spin*sin(phi), dphi/dx = -y/rho^2, dphi/dy = x/rho^2
        let drag_grad_coeff = self.delta_n * gauss * 0.1 * self.a_spin * (-phi.sin());
        let rho2_safe = rho_safe * rho_safe;
        let drag_gx = drag_grad_coeff * (-p[1] / rho2_safe);
        let drag_gy = drag_grad_coeff * (p[0] / rho2_safe);

        ([grad_x + drag_gx, grad_y + drag_gy, grad_z], n)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("=== Warp Ring Integration (Engine-Backed) ===");

    // -- Configuration --
    let warp_config = WarpRingConfig {
        prime: 2,
        alpha: -0.5,   // Negative-dimension: anti-diffusive smoothing
        epsilon: 0.01, // IR regularization
        domain_size: 2.0 * PI,
    };

    // -- Step 1: Initialize Gororoba Engine --
    let nx = 64;
    let ny = 64;
    let lbm_tau = 0.8;
    // 500 steps: the Kolmogorov mode reaches ~40% steady state while the initial
    // perturbation modes (decay time ~1040 steps for mode (1,0)) retain ~62% amplitude.
    // This keeps all three triad legs detectable: (1,0)+(0,1)+(-1,-1)=0.
    let lbm_steps = 500;

    let sim_config = SimulationConfig {
        nx,
        ny,
        tau: lbm_tau,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: 0.1,
        coupling_algebra_fluid: 0.1,
        coupling_metric_algebra: 0.1,
    };

    let mut state = SimulationState::new(sim_config);

    // Multi-mode perturbation to seed spectral diversity.
    //
    // WHY: The laminar Kolmogorov solution is an exact NS solution with energy at
    // ONLY the (kx=0, ky=1) mode -- a pure sine wave with zero nonlinear coupling.
    // No valid triad exists from a single mode (triads require 3 distinct wavevectors).
    // Re~10 is below the turbulent bifurcation (~Re~40), so the flow stays laminar.
    //
    // Adding modes at (1,0) and (1,1) [i.e. diagonal] provides triad closure with the
    // Kolmogorov mode: (1,0) + (0,1) + (-1,-1) = (0,0). The two perturbation modes
    // decay viscously (tau_decay ~ 1/(nu*k^2) = 1037 steps for (1,0)), but remain
    // detectable for ~500 steps at 62% of initial amplitude.
    let perturb_amp = 1e-2_f64; // 1% of lattice speed, well below Ma stability limit
    for x in 0..nx {
        for y in 0..ny {
            // Mode (1,0): sin(2*pi*x/nx) excites wavevector (kx=1, ky=0) in u_x
            let ux_mode_10 = perturb_amp * (2.0 * PI * x as f64 / nx as f64).sin();
            // Mode (1,1): sin(2*pi*(x+y)/nx) excites wavevectors (1,1) and (-1,-1) in u_x
            // The (-1,-1) leg closes the triad with (0,1) and (1,0): 1+0+(-1)=0, 0+1+(-1)=0
            let ux_mode_11 = perturb_amp * (2.0 * PI * (x + y) as f64 / nx as f64).sin();
            let upx = ux_mode_10 + ux_mode_11;
            for i in 0..9 {
                let cx = CX[i] as f64;
                // Linearized Maxwell: delta_f_i = W_i * (c_{ix}/cs^2) * delta_u_x = 3*W_i*cx*upx
                state.fluid.f[[i, x, y]] += W[i] * (3.0 * cx * upx);
            }
        }
    }

    info!(
        "[1/9] Running Engine LBM ({}x{}, tau={}, {} steps)...",
        nx, ny, lbm_tau, lbm_steps
    );

    // Kolmogorov forcing parameters: sinusoidal body force fx(y) = A*sin(2*pi*n*y/ny)
    // drives large-scale turbulence at wavenumber force_mode=1. Without this force
    // the collide-stream loop thermalizes to zero velocity (no triads to find).
    // Analytical Kolmogorov solution: U_max = F / (nu * ky^2)
    // nu = (2*tau-1)/6 = 0.1, ky = 2*pi/ny = 2*pi/64 => ky^2 ~ 0.0096
    // U_max ~ 1e-4 / (0.1 * 0.0096) ~ 0.104 lu (Ma ~ 0.18, stable)
    // Re = U_max * (ny/2pi) / nu ~ 0.104 * 10.2 / 0.1 ~ 10.6
    let force_amp = 1e-4_f64; // amplitude yields Re ~ 10 for tau=0.8, 64x64 grid
    let force_mode = 1_usize; // fundamental mode drives the longest-wavelength instability

    // Run fluid simulation with Kolmogorov body forcing
    for _ in 0..lbm_steps {
        state.fluid.collide(0, ny);

        // Guo (2002) forcing scheme: f[i,x,y] += 3*W[i]*cx * fy(y) * rho[x,y]
        // where fy(y) = A * sin(2*pi*n*y/ny) is the Kolmogorov sinusoidal force.
        let (rho, _, _) = state.fluid.macroscopic();
        for i in 0..9 {
            let cx = CX[i] as f64;
            for x in 0..nx {
                for y in 0..ny {
                    let fy =
                        force_amp * (2.0 * PI * force_mode as f64 * y as f64 / ny as f64).sin();
                    state.fluid.f[[i, x, y]] += 3.0 * W[i] * cx * fy * rho[[x, y]];
                }
            }
        }

        state.fluid.stream();
    }

    let (_rho, u_array, v_array) = state.fluid.macroscopic();
    let u = u_array;
    let v = v_array;

    // Recompute diagnostics that used to come from `flow` struct
    let (k_axis, power) = power_spectrum(&u);
    // Enstrophy estimate
    let mut enstrophy = 0.0;
    for x in 0..nx {
        for y in 0..ny {
            let uy_x = (v[[(x + 1) % nx, y]] - v[[(x + nx - 1) % nx, y]]) / 2.0;
            let ux_y = (u[[x, (y + 1) % ny]] - u[[x, (y + ny - 1) % ny]]) / 2.0;
            enstrophy += (uy_x - ux_y).powi(2);
        }
    }
    enstrophy /= (nx * ny) as f64;

    // Map flow enstrophy to a Kerr spin parameter (used in Steps 4b and 7)
    let a_spin = (enstrophy * 1e2).tanh() * 0.95;

    // -- Step 2: Extract Standard Spectral Triads --
    // ... (Remainder of spectral analysis remains similar, using `u` and `v` from engine)
    info!("[2/9] Extracting spectral triads...");
    // Threshold calibrated to forcing scale: amplitude = |u_hat| / (nx*ny).
    // At force_amp=1e-4, 64x64, Re~10: dominant triple product ~ 5e-11.
    // 1e-12 captures the energetically significant triads, rejects numerical noise.
    let spectral_triads = extract_dominant_triads(&u, &v, 1e-12);
    info!(
        "      Found {} spectral triads (standard).",
        spectral_triads.len()
    );

    // -- Step 3: Warp Physics -- P-adic Modulation + Neg-Dim Kernel --
    info!(
        "[3/9] Applying warp physics (p={}, alpha={:.1}, eps={:.3})...",
        warp_config.prime, warp_config.alpha, warp_config.epsilon
    );

    // FFT the u-field (real -> complex -> 2D FFT via ndrustfft)
    let u_hat = fft_2d(&real_to_complex_2d(&u));

    // Apply negative-dimension kernel
    let u_hat_warp = apply_neg_dim_kernel(&u_hat, &warp_config);
    info!(
        "      Neg-dim kernel applied: DC gain = {:.2}x",
        u_hat_warp[[0, 0]].norm() / u_hat[[0, 0]].norm().max(1e-30)
    );

    // Extract warp triads (with p-adic + neg-dim weights)
    let warp_triads = extract_warp_triads(&u_hat, &warp_config, 1.0);
    info!(
        "      Found {} warp triads (p-adic + neg-dim weighted).",
        warp_triads.len()
    );

    if let Some(top) = warp_triads.first() {
        info!(
            "      Top triad: k={:?}, padic_w={:.4}, negdim_w={:.4}, warp_w={:.4}",
            top.k, top.padic_weight, top.neg_dim_weight, top.warp_weight
        );
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
    info!("[4/9] Computing metamaterial spectral filter (ZD -> TMM)...");
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
            .chain(
                stack
                    .iter()
                    .map(|m| Complex64::new(m.layer.n_real, m.layer.n_imag)),
            )
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

                    // -- NA-001: Nonlinear Kerr modulation via TCMT --
                    // Scale spectral power to "optical" power for nonlinearity
                    let local_power = power[idx] * 1e3;
                    let cavity = KerrCavity::from_wavelength(
                        wavelength_nm,
                        500.0,
                        500.0,
                        1.5,
                        1e-10, // n2
                        1e-18, // Veff
                    );
                    let solver = TcmtSolver::new(cavity);
                    let input = InputField::cw(local_power, cavity.omega_0);
                    let ss = solver.steady_state(&input);

                    // High reflectance + Nonlinear enhancement -> higher material weight
                    let nl_boost = ss.power_transmissions.first().copied().unwrap_or(0.0);
                    material_weights[idx] = (1.0 + tmm.reflectance) * (1.0 + nl_boost);
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

    // -- Step 4b: GRIN Ray Tracing Through Warp Ring (NA-001) --
    info!("[5/9] GRIN ray tracing through warp-ring refractive index (NA-001)...");

    // Map enstrophy to peak refractive index perturbation:
    // Weak-field GR: delta_n ~ 2*Phi/c^2 ~ r_s/r. For the simulation,
    // we scale enstrophy (vorticity squared) to a modest perturbation.
    let delta_n = (enstrophy * 1e3).tanh() * 0.3; // Saturates at 0.3
    let ring_r = 2.5; // Ring radius in plot coordinates
    let ring_sigma = 0.8; // Gaussian width

    let warp_medium = WarpRingGrinMedium {
        ring_radius: ring_r,
        sigma: ring_sigma,
        delta_n,
        a_spin,
    };

    // Generate a grid of background rays (simulating distant stars)
    // Rays propagate in the +z direction through the equatorial plane (z=0)
    let n_rays_side = 12;
    let ray_extent = 3.8; // Cover most of the [-4, 4] plot range
    let ray_z_start = -8.0;
    let ray_step = 0.05;
    let ray_max_steps = 320; // 320 * 0.05 = 16 units total path

    let mut ray_paths: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut deflections: Vec<f64> = Vec::new();

    for ix in 0..n_rays_side {
        for iy in 0..n_rays_side {
            let x0 = -ray_extent + 2.0 * ray_extent * ix as f64 / (n_rays_side - 1) as f64;
            let y0 = -ray_extent + 2.0 * ray_extent * iy as f64 / (n_rays_side - 1) as f64;

            let initial_ray = Ray {
                pos: [x0, y0, ray_z_start],
                dir: [0.0, 0.0, 1.0], // Propagating in +z
            };

            let result = trace_ray(initial_ray, &warp_medium, ray_step, ray_max_steps);

            // Record the projected (x, y) path for equatorial-plane rendering
            let path: Vec<(f64, f64)> = result.positions.iter().map(|p| (p[0], p[1])).collect();

            // Deflection angle: angle between final direction and initial +z
            if let Some(final_dir) = result.directions.last() {
                let transverse = (final_dir[0] * final_dir[0] + final_dir[1] * final_dir[1]).sqrt();
                let deflection_rad = transverse.atan2(final_dir[2].abs());
                deflections.push(deflection_rad);
            }

            ray_paths.push(path);
        }
    }

    let mean_deflection = if deflections.is_empty() {
        0.0
    } else {
        deflections.iter().sum::<f64>() / deflections.len() as f64
    };
    let max_deflection = deflections.iter().cloned().fold(0.0_f64, f64::max);

    info!(
        "      GRIN medium: delta_n={:.4}, ring_R={:.1}, sigma={:.1}, a_spin={:.3}",
        delta_n, ring_r, ring_sigma, a_spin
    );
    info!(
        "      Traced {} rays: mean_deflection={:.4} rad ({:.2} deg), max={:.4} rad ({:.2} deg)",
        ray_paths.len(),
        mean_deflection,
        mean_deflection.to_degrees(),
        max_deflection,
        max_deflection.to_degrees()
    );

    // -- Step 5: Map to E7 Lie Algebra --
    info!("[6/9] Mapping to E7 geometry...");
    let e7_roots = generate_e7_roots();
    let algebra_triads = find_e7_triads(&e7_roots);
    info!(
        "      E7: {} roots, {} structural triads.",
        e7_roots.len(),
        algebra_triads.len()
    );

    // -- Step 6: Build Hypergraph + Topological Invariants --
    info!("[7/9] Building hypergraph...");
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

    info!(
        "      Hypergraph: {} vertices, {} edges",
        hg.vertex_count(),
        hg.edge_count()
    );
    info!(
        "      Clustering coefficient: {:.4}",
        hg.clustering_coefficient()
    );
    info!("      Betti-0 (components): {}", hg.betti_0());
    info!("      Betti-1 (cycles): {}", hg.betti_1());

    // Select active algebraic triads proportional to warp triad count
    let active_count = warp_triads.len().min(algebra_triads.len()) * 3;
    let active_algebra_triads: Vec<_> = algebra_triads.into_iter().take(active_count).collect();

    // -- Step 7: Warp Lensing (SHI Integration) --
    info!("[8/9] Simulating breakthrough SHI warp lensing...");
    let kerr = Kerr::new(1.0, a_spin);
    let mut lensed_roots = Vec::new();

    for r in &e7_roots {
        let (x, y) = project_to_plane(&r.root);

        // Initial state in Boyer-Lindquist-like coordinates
        // Map 2D Coxeter projection (x, y) to (r, theta)
        let r_start = (x * x + y * y).sqrt() * 2.0 + 3.0; // Offset from horizon
        let theta_start = y.atan2(x) + PI / 2.0;

        let mut r_curr = r_start;
        let mut theta_curr = theta_start;
        let mut vr = -0.2; // Inward radial velocity
        let mut vtheta = 0.05;
        let h = 0.1; // Integration step

        // 30 steps of SHI integration
        for _ in 0..30 {
            let (r_next, theta_next, vr_next, vtheta_next) =
                sedenion_homotopy_step(&kerr, r_curr, theta_curr, vr, vtheta, h);

            r_curr = r_next;
            theta_curr = theta_next;
            vr = vr_next;
            vtheta = vtheta_next;

            if r_curr < 2.1 {
                break;
            } // Terminate near horizon
        }

        // Project back to 2D for plotting
        // Inverse mapping to get back to visualization coordinates
        let x_final = (r_curr - 3.0) / 2.0 * theta_curr.cos();
        let y_final = (r_curr - 3.0) / 2.0 * theta_curr.sin();
        lensed_roots.push((x_final, y_final));
    }

    // -- Step 8: Render --
    info!("[9/9] Rendering warp ring...");
    let root = BitMapBackend::new("warp_ring_integration.png", (1024, 1024)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Warp Ring: E7 x Turbulence x GRIN Optics",
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

    // Layer 5: GRIN-lensed background ray paths (yellow/orange, NA-001)
    // Each ray is projected to (x, y) from the 3D trace. Rays near the ring
    // are deflected more strongly, revealing the gravitational lensing pattern.
    for (i, path) in ray_paths.iter().enumerate() {
        if path.len() < 2 {
            continue;
        }
        // Color by deflection: low deflection = dim yellow, high = bright orange
        let defl_frac = if max_deflection > 1e-10 {
            (deflections[i] / max_deflection).min(1.0)
        } else {
            0.0
        };
        let hue = 0.12 - 0.06 * defl_frac; // Yellow(0.12) -> Orange(0.06)
        let color = HSLColor(hue, 1.0, 0.5);

        // Subsample the path to avoid excessive line segments
        let step = (path.len() / 40).max(1);
        let subsampled: Vec<(f64, f64)> = path.iter().step_by(step).copied().collect();

        if subsampled.len() >= 2 {
            chart.draw_series(LineSeries::new(
                subsampled,
                &color.mix(0.4 + 0.4 * defl_frac),
            ))?;
        }
    }

    // Layer 6: Warp ring outline (white dashed circle at ring_radius)
    {
        let n_ring_pts = 100;
        let ring_pts: Vec<(f64, f64)> = (0..=n_ring_pts)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / n_ring_pts as f64;
                (ring_r * theta.cos(), ring_r * theta.sin())
            })
            .collect();
        chart.draw_series(LineSeries::new(ring_pts, &WHITE.mix(0.3)))?;
    }

    root.present()?;
    info!("Done. Output saved to 'warp_ring_integration.png'.");

    // -- Summary --
    info!("--- Warp Ring Summary ---");
    info!("  LBM: {}x{}, tau={}, {} steps", nx, ny, lbm_tau, lbm_steps);
    info!("  Spectral triads (standard): {}", spectral_triads.len());
    info!("  Warp triads (p-adic + neg-dim): {}", warp_triads.len());
    info!(
        "  Materials: {} ZD layers, {} physical",
        verification.n_total, verification.n_physical
    );
    info!(
        "  GRIN lensing: delta_n={:.4}, mean_defl={:.4} rad, max_defl={:.4} rad",
        delta_n, mean_deflection, max_deflection
    );
    info!("  E7 algebraic triads: {}", active_algebra_triads.len());
    info!(
        "  Hypergraph: V={}, E={}, C={:.4}, b0={}, b1={}",
        hg.vertex_count(),
        hg.edge_count(),
        hg.clustering_coefficient(),
        hg.betti_0(),
        hg.betti_1()
    );

    Ok(())
}
