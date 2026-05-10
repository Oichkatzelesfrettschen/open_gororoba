use anyhow::Result;
use optics_core::absorber_pareto::{
    pareto_sweep, FractionalSchrodingerConfig, ParetoPoint, WavePacketConfig,
};
use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("--- Running Absorber Pareto Sweep (Robust) ---");

    let cfg = FractionalSchrodingerConfig {
        n: 1024,
        l_domain: 200.0,
        alpha: 1.5,
        d_alpha: 0.5,
        dt: 0.25,
        steps: 640, // t_total = 160.0 / dt = 0.25
    };

    let wp = WavePacketConfig {
        x0: -120.0,
        k0: 1.2,
        sigma: 8.0,
    };

    let orders = vec![4, 6, 8];
    let etas = vec![2e-4, 5e-4, 1e-3];
    let xcs = vec![110.0, 120.0, 130.0];
    let delta = 5.0;

    let total_runs = orders.len() * etas.len() * xcs.len();
    println!("Starting parameter sweep ({} combinations)...", total_runs);

    let points = pareto_sweep(&cfg, &wp, &orders, &etas, &xcs, delta);

    println!("Sweep complete: {} points", points.len());
    for pt in &points {
        println!(
            "  m={} eta={:.0e} xc={}: edge={:.4e} int={:.4e}",
            pt.m, pt.eta, pt.xc, pt.m_edge, pt.e_int
        );
    }

    plot_pareto(&points)?;
    Ok(())
}

fn plot_pareto(points: &[ParetoPoint]) -> Result<()> {
    let out_path = PathBuf::from("data/artifacts/images/absorber_pareto_3160x2820.png");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let root = BitMapBackend::new(&out_path, (3160, 2820)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let min_x = points
        .iter()
        .map(|p| p.m_edge)
        .fold(f64::INFINITY, f64::min)
        * 0.5;
    let max_x = points.iter().map(|p| p.m_edge).fold(0.0, f64::max) * 2.0;
    let min_y = points.iter().map(|p| p.e_int).fold(f64::INFINITY, f64::min) * 0.5;
    let max_y = points.iter().map(|p| p.e_int).fold(0.0, f64::max) * 2.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(100)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .caption(
            "Absorber Pareto: Edge mass vs Interior distortion (alpha=1.5)",
            ("sans-serif", 60).into_font().color(&WHITE),
        )
        .build_cartesian_2d((min_x..max_x).log_scale(), (min_y..max_y).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Edge mass M_edge (down better)")
        .y_desc("Interior distortion E_int (down better)")
        .axis_style(RGBColor(31, 41, 55))
        .label_style(("sans-serif", 30).into_font().color(&WHITE))
        .draw()?;

    for pt in points {
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
