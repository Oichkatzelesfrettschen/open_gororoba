use anyhow::Result;
use num_complex::Complex64;
use plotters::prelude::*;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::path::PathBuf;

fn integrate_trapezoidal(y: &[f64], dx: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..(y.len() - 1) {
        sum += (y[i] + y[i + 1]) * 0.5 * dx;
    }
    sum
}

fn main() -> Result<()> {
    println!("--- Running Absorber Pareto Sweep (Robust) ---");

    let n = 1024;
    let l_domain = 200.0;
    let dx = (2.0 * l_domain) / n as f64;
    
    let x: Vec<f64> = (0..n).map(|i| -l_domain + i as f64 * dx).collect();
    
    let k: Vec<f64> = (0..n).map(|i| {
        let freq = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
        2.0 * PI * freq / (n as f64 * dx)
    }).collect();

    let alpha = 1.5_f64;
    let d = 0.5_f64;
    let t_total = 160.0_f64;
    let dt = 0.25_f64;
    let steps = (t_total / dt).round() as usize;
    let delta = 5.0;

    // Aggressive IC: right-moving packet
    let x0 = -120.0;
    let k0 = 1.2;
    let sig = 8.0;
    
    let mut psi0: Vec<Complex64> = x.iter().map(|&xi| {
        let amp = (-0.5 * ((xi - x0) / sig).powi(2)).exp();
        let phase = k0 * xi;
        Complex64::new(amp * phase.cos(), amp * phase.sin())
    }).collect();

    let density0: Vec<f64> = psi0.iter().map(|p| p.norm_sqr()).collect();
    let norm = integrate_trapezoidal(&density0, dx).sqrt();
    for p in &mut psi0 {
        *p /= norm;
    }

    let phase_k: Vec<Complex64> = k.iter().map(|&ki| {
        let phase = -d * ki.abs().powf(alpha) * dt;
        Complex64::new(phase.cos(), phase.sin())
    }).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let scale = 1.0 / n as f64;

    println!("Computing free evolution baseline...");
    let mut psi_free = psi0.clone();
    for _ in 0..steps {
        fft.process(&mut psi_free);
        for (p, pk) in psi_free.iter_mut().zip(phase_k.iter()) {
            *p *= pk;
        }
        ifft.process(&mut psi_free);
        for p in &mut psi_free {
            *p *= scale;
        }
    }

    let run_abs = |eta: f64, m_order: i32, xc: f64| -> Vec<Complex64> {
        let absorb: Vec<f64> = x.iter().map(|&xi| {
            if xi.abs() > xc {
                (-eta * (xi.abs() - xc).powi(m_order)).exp()
            } else {
                1.0
            }
        }).collect();

        let mut psi = psi0.clone();
        let mut local_planner = FftPlanner::new();
        let local_fft = local_planner.plan_fft_forward(n);
        let local_ifft = local_planner.plan_fft_inverse(n);

        for _ in 0..steps {
            local_fft.process(&mut psi);
            for (p, pk) in psi.iter_mut().zip(phase_k.iter()) {
                *p *= pk;
            }
            local_ifft.process(&mut psi);
            for (p, &ab) in psi.iter_mut().zip(absorb.iter()) {
                *p *= scale * ab;
            }
        }
        psi
    };

    let orders = vec![4, 6, 8];
    let etas = vec![2e-4, 5e-4, 1e-3];
    let xcs = vec![110.0, 120.0, 130.0];

    let total_runs = orders.len() * etas.len() * xcs.len();
    println!("Starting parameter sweep ({} combinations)...", total_runs);

    struct Point { m_edge: f64, e_int: f64, m: i32, eta: f64, xc: f64 }
    let mut points = Vec::new();
    let mut count = 0;

    for &m in &orders {
        for &eta in &etas {
            for &xc in &xcs {
                let psi_abs = run_abs(eta, m, xc);
                
                // Edge mass
                let mut edge_density = Vec::new();
                for i in 0..n {
                    if x[i].abs() > xc + delta {
                        edge_density.push(psi_abs[i].norm_sqr());
                    }
                }
                let m_edge = integrate_trapezoidal(&edge_density, dx);

                // Interior distortion
                let mut int_diff_sq = Vec::new();
                for i in 0..n {
                    if x[i].abs() < xc - delta {
                        int_diff_sq.push((psi_abs[i] - psi_free[i]).norm_sqr());
                    }
                }
                let e_int = integrate_trapezoidal(&int_diff_sq, dx).sqrt();

                points.push(Point { m_edge, e_int, m, eta, xc });
                count += 1;
                if count % 5 == 0 {
                    println!("Progress: {}/{}", count, total_runs);
                }
            }
        }
    }

    // Plotting
    let out_path = PathBuf::from("data/artifacts/images/absorber_pareto_3160x2820.png");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let root = BitMapBackend::new(&out_path, (3160, 2820)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let min_x = points.iter().map(|p| p.m_edge).fold(f64::INFINITY, f64::min) * 0.5;
    let max_x = points.iter().map(|p| p.m_edge).fold(0.0, f64::max) * 2.0;
    let min_y = points.iter().map(|p| p.e_int).fold(f64::INFINITY, f64::min) * 0.5;
    let max_y = points.iter().map(|p| p.e_int).fold(0.0, f64::max) * 2.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(100)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .caption("Absorber Pareto: Edge mass vs Interior distortion (alpha=1.5)", ("sans-serif", 60).into_font().color(&WHITE))
        .build_cartesian_2d((min_x..max_x).log_scale(), (min_y..max_y).log_scale())?;

    chart.configure_mesh()
        .x_desc("Edge mass M_edge (down better)")
        .y_desc("Interior distortion E_int (down better)")
        .axis_style(RGBColor(31, 41, 55))
        .label_style(("sans-serif", 30).into_font().color(&WHITE))
        .draw()?;

    for pt in &points {
        chart.draw_series(std::iter::once(Circle::new(
            (pt.m_edge, pt.e_int),
            15,
            RGBColor(100, 150, 255).mix(0.8),
        )))?;

        let label = format!("m={},eta={:.0e},xc={}", pt.m, pt.eta, pt.xc as i32);
        chart.draw_series(std::iter::once(Text::new(
            label,
            (pt.m_edge, pt.e_int),
            ("sans-serif", 20).into_font().color(&WHITE),
        )))?;
    }

    root.present()?;
    println!("Saved: {}", out_path.display());

    Ok(())
}
