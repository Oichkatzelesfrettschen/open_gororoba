//! Visualization utilities using plotters.
//!
//! This module provides plotting capabilities that replace the Python
//! matplotlib-based visualization scripts.
//!
//! # Features
//! - Line plots for time series (entropy evolution, cosmology)
//! - Scatter plots for 2D projections (E8 roots, zero-divisors)
//! - SVG output for vector graphics

use plotters::prelude::*;
use std::error::Error;

/// Plot a simple line chart to SVG.
pub fn line_plot_svg(
    filename: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    data: &[(f64, f64)],
    color: RGBColor,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Compute bounds
    let (x_min, x_max, y_min, y_max) = compute_bounds(data);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|&(x, y)| (x, y)),
        color.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Plot multiple line series on the same chart.
pub fn multi_line_plot_svg(
    filename: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[(&str, &[(f64, f64)], RGBColor)],
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Compute overall bounds
    let all_data: Vec<(f64, f64)> = series
        .iter()
        .flat_map(|(_, data, _)| data.iter().copied())
        .collect();
    let (x_min, x_max, y_min, y_max) = compute_bounds(&all_data);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    for (name, data, color) in series {
        chart
            .draw_series(LineSeries::new(
                data.iter().map(|&(x, y)| (x, y)),
                color.stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot a 2D scatter plot to SVG.
pub fn scatter_plot_svg(
    filename: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    points: &[(f64, f64)],
    color: RGBColor,
    point_size: u32,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max, y_min, y_max) = compute_bounds(points);

    // Make axes equal for geometric plots
    let range = (x_max - x_min).max(y_max - y_min) * 1.1;
    let x_center = (x_min + x_max) / 2.0;
    let y_center = (y_min + y_max) / 2.0;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (x_center - range / 2.0)..(x_center + range / 2.0),
            (y_center - range / 2.0)..(y_center + range / 2.0),
        )?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), point_size, color.filled())),
    )?;

    root.present()?;
    Ok(())
}

/// Plot a heatmap to SVG (for 2D grids like DLA clusters).
pub fn heatmap_svg(
    filename: &str,
    title: &str,
    data: &[Vec<f64>],
    colormap: Colormap,
) -> Result<(), Box<dyn Error>> {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };

    if rows == 0 || cols == 0 {
        return Err("Empty data".into());
    }

    // Find min/max for normalization
    let (v_min, v_max) = data
        .iter()
        .flat_map(|row| row.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });

    let root = SVGBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..cols, 0..rows)?;

    chart.configure_mesh().disable_mesh().draw()?;

    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let normalized = if v_max > v_min {
                (val - v_min) / (v_max - v_min)
            } else {
                0.5
            };
            let color = colormap.get(normalized);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(j, rows - 1 - i), (j + 1, rows - i)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Simple colormap for heatmaps.
#[derive(Clone, Copy)]
pub enum Colormap {
    Viridis,
    Inferno,
    Grayscale,
}

impl Colormap {
    /// Get color for normalized value in [0, 1].
    pub fn get(&self, t: f64) -> RGBColor {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => {
                // Simplified viridis approximation
                let r = (68.0 + t * (253.0 - 68.0)) as u8;
                let g = (1.0 + t * (231.0 - 1.0)) as u8;
                let b = (84.0 + t * (37.0 - 84.0) * (1.0 - t) + t * 37.0) as u8;
                RGBColor(r, g, b)
            }
            Colormap::Inferno => {
                // Simplified inferno approximation
                let r = (t * 255.0) as u8;
                let g = (t * t * 200.0) as u8;
                let b = ((1.0 - t) * 100.0 + t * 50.0) as u8;
                RGBColor(r, g, b)
            }
            Colormap::Grayscale => {
                let v = (t * 255.0) as u8;
                RGBColor(v, v, v)
            }
        }
    }
}

/// Compute data bounds with 5% padding.
fn compute_bounds(data: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let x_min = data.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = data.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min) * 0.05;
    let y_pad = (y_max - y_min) * 0.05;

    (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
}

/// Common colors for plots.
pub mod colors {
    use plotters::style::RGBColor;

    pub const INDIGO: RGBColor = RGBColor(75, 0, 130);
    pub const CRIMSON: RGBColor = RGBColor(220, 20, 60);
    pub const PURPLE: RGBColor = RGBColor(128, 0, 128);
    pub const TEAL: RGBColor = RGBColor(0, 128, 128);
    pub const ORANGE: RGBColor = RGBColor(255, 165, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_line_plot() {
        let data: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, (i as f64 * 0.1).sin())).collect();

        let result = line_plot_svg(
            "/tmp/test_line_plot.svg",
            "Test Line Plot",
            "X",
            "Y",
            &data,
            colors::INDIGO,
        );
        assert!(result.is_ok());
        assert!(fs::metadata("/tmp/test_line_plot.svg").is_ok());
        fs::remove_file("/tmp/test_line_plot.svg").ok();
    }

    #[test]
    fn test_scatter_plot() {
        let points: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let angle = i as f64 * 0.1;
                (angle.cos(), angle.sin())
            })
            .collect();

        let result = scatter_plot_svg(
            "/tmp/test_scatter_plot.svg",
            "Test Scatter",
            "X",
            "Y",
            &points,
            colors::PURPLE,
            3,
        );
        assert!(result.is_ok());
        fs::remove_file("/tmp/test_scatter_plot.svg").ok();
    }

    #[test]
    fn test_heatmap() {
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| (0..20).map(|j| ((i + j) as f64).sin()).collect())
            .collect();

        let result = heatmap_svg("/tmp/test_heatmap.svg", "Test Heatmap", &data, Colormap::Inferno);
        assert!(result.is_ok());
        fs::remove_file("/tmp/test_heatmap.svg").ok();
    }

    #[test]
    fn test_colormap() {
        let cmap = Colormap::Viridis;
        let c0 = cmap.get(0.0);
        let c1 = cmap.get(1.0);
        assert_ne!(c0.0, c1.0); // Colors should differ
    }
}
