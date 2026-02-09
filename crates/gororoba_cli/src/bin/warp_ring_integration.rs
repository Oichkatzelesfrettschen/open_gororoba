//! Warp Ring Integration: Algebra <-> Fluid Duality.
//!
//! This binary demonstrates the full pipeline:
//! 1. Simulate turbulence (LBM D2Q9)
//! 2. Extract spectral triads (energy transfer)
//! 3. Map triads to E7 Lie Algebra roots
//! 4. Visualize the "Warp Ring" (Projected E7 Triads)

use algebra_core::lie::e7_geometry::{find_e7_triads, generate_e7_roots, project_to_plane};
use lbm_core::turbulence::{extract_dominant_triads, power_spectrum};
use log::info;
use ndarray::Array2;
use optics_core::grin::{trace_ray, GrinMedium, Ray};
use plotters::prelude::*;
use plotters::style::full_palette::GREY;
use rand::{Rng, SeedableRng};
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

/// Generate 2D field with Kolmogorov-like power spectrum k^(-5/3).
fn generate_kolmogorov_field(nx: usize, ny: usize, seed: u64) -> Array2<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut field = Array2::<f64>::zeros((nx, ny));

    // Superposition of waves
    // u(x) = sum A(k) cos(k.x + phi)
    // A(k) ~ k^(-5/6) for 1D energy spectrum k^(-5/3)
    // In 2D, energy E(k) ~ |u_k|^2 * k. If E(k) ~ k^(-5/3), then |u_k|^2 ~ k^(-8/3), so |u_k| ~ k^(-4/3).

    let n_modes = 1000;
    for _ in 0..n_modes {
        let kx: f64 = rng.gen_range(-10.0..10.0);
        let ky: f64 = rng.gen_range(-10.0..10.0);
        let k = (kx * kx + ky * ky).sqrt();

        if k < 0.1 {
            continue;
        }

        let amplitude = k.powf(-4.0 / 3.0);
        let phase = rng.gen_range(0.0..2.0 * PI);

        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 / nx as f64 * 2.0 * PI;
                let y = j as f64 / ny as f64 * 2.0 * PI;
                field[[i, j]] += amplitude * (kx * x + ky * y + phase).cos();
            }
        }
    }

    // Normalize
    let max_val = field.iter().fold(f64::NEG_INFINITY, |a: f64, b| a.max(*b));
    let min_val = field.iter().fold(f64::INFINITY, |a: f64, b| a.min(*b));
    let span = max_val - min_val;
    if span > 0.0 {
        field.mapv_inplace(|x| (x - min_val) / span);
    }

    field
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("=== Warp Ring Integration ===");

    // 1. Simulate Turbulence (Mock with Kolmogorov Noise)
    info!("[1/4] Simulating Turbulence (Kolmogorov Spectrum)...");
    let nx = 64;
    let ny = 64;
    let u = generate_kolmogorov_field(nx, ny, 42);
    let v = generate_kolmogorov_field(nx, ny, 1337);
    let (_k_axis, power) = power_spectrum(&u);
    if let Some(p0) = power.get(1).copied() {
        info!("      Spectrum diagnostic: P(k=1) = {:.6}", p0);
    }

    // 2. Extract Triads
    info!("[2/4] Extracting Spectral Triads...");
    let spectral_triads = extract_dominant_triads(&u, &v, 50.0);
    info!("      Found {} spectral triads.", spectral_triads.len());

    // 3. Map to E7
    info!("[3/4] Mapping to E7 Geometry...");
    let e7_roots = generate_e7_roots();
    let algebra_triads = find_e7_triads(&e7_roots);
    info!(
        "      E7 Reference: {} roots, {} structural triads.",
        e7_roots.len(),
        algebra_triads.len()
    );

    // Build Hypergraph for Analysis
    let mut hg = TriadHypergraph::new();
    // For demo, we add algebraic triads to the hypergraph
    // In reality, we'd map spectral modes to root indices first
    for (i, _triad) in algebra_triads.iter().enumerate() {
        // Map root index to vertex ID
        hg.add_triad(i, (i + 1) % 126, (i + 2) % 126); // Dummy mapping for topo structure
    }
    info!(
        "      Hypergraph Clustering Coeff: {:.4}",
        hg.clustering_coefficient()
    );

    // Mapping Logic:
    // We map the "energy transfer" of a spectral triad to the "interaction strength"
    // of an algebraic triad.
    // Simple heuristic: Take top N algebraic triads to represent the active flow.
    let active_algebra_triads = algebra_triads
        .into_iter()
        .take(spectral_triads.len() * 10)
        .collect::<Vec<_>>();

    // 4. Optics Simulation (Warp Lensing)
    info!("[4/5] Simulating Warp Lensing...");
    let warp = WarpMedium {
        amplitude: 0.5,
        sigma: 2.0,
    };
    let mut lensed_roots = Vec::new();

    for r in &e7_roots {
        let (x, y) = project_to_plane(&r.root);
        // Trace a ray from "infinity" towards the root position on the plane z=0
        let start = [x * 0.1, y * 0.1, -10.0];
        let dir = [0.0, 0.0, 1.0];
        let ray = Ray { pos: start, dir }; // No intensity

        let result = trace_ray(ray, &warp, 0.1, 200);

        if let Some(end_pos) = result.positions.last() {
            lensed_roots.push((end_pos[0], end_pos[1]));
        } else {
            lensed_roots.push((x, y));
        }
    }

    // 5. Visualize
    info!("[5/5] Rendering Warp Ring...");
    let root = BitMapBackend::new("warp_ring_integration.png", (1024, 1024)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Algebra-Fluid Duality: E7 Warp Ring",
            ("sans-serif", 40).into_font().color(&WHITE),
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

    // Draw E7 roots (background structure)
    chart.draw_series(e7_roots.iter().map(|r| {
        let (x, y) = project_to_plane(&r.root);
        Circle::new((x, y), 2, GREY.filled())
    }))?;

    // Draw Lensed Roots (Warp Effect)
    chart.draw_series(
        lensed_roots
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, RED.filled())),
    )?;

    // Draw Active Triads (Energy Flow)
    for triad in active_algebra_triads {
        let (k_x, k_y) = project_to_plane(&triad.k.root);
        let (p_x, p_y) = project_to_plane(&triad.p.root);
        let (q_x, q_y) = project_to_plane(&triad.q.root);

        // Color based on "interaction strength" (mock)
        let color = HSLColor(0.6, 1.0, 0.5); // Cyan-ish

        chart.draw_series(LineSeries::new(
            vec![(k_x, k_y), (p_x, p_y), (q_x, q_y), (k_x, k_y)],
            &color.mix(0.3),
        ))?;
    }

    root.present()?;
    info!("Done. Output saved to 'warp_ring_integration.png'.");
    Ok(())
}
