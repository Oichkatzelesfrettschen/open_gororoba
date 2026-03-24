//! Generate hyper-resolution artifacts for the Unified Quantum-Algebraic Framework.
//!
//! Ported from the vis_*.py scripts to pure Rust.
//! Refined for elegant multidimensional complexity and advanced aesthetics.

use anyhow::Result;
use gororoba_view_raster::render_hyper_fractal_to_argb;
use plotters::prelude::*;
use rand::prelude::*;
use std::{f64::consts::PI, path::Path};

fn main() -> Result<()> {
    println!("--- Generating Grand Artifacts (3160x2820) ---");

    // 1. Hyper-Fractal Sedenion Visualization
    generate_hyper_fractal()?;

    // 2. Hyper-Mass Ladder Visualization
    generate_mass_ladder()?;

    // 3. Hyper-MERA Network Visualization
    generate_mera_network()?;

    // 4. Motif Census Summary Visualization
    generate_motif_summary()?;

    // 5. Silicon-Algebra Trajectory Visualization
    generate_trajectory()?;

    // 6. Dimensional Geometry Visualization
    generate_dimensional_geometry()?;

    // 7. 4D Hyper-Mosaic Visualization
    generate_4d_mosaic()?;

    // 8. Fano Scattering Visualization
    generate_fano_vis()?;

    // 9. Nilpotent ZD Projection
    generate_zd_projection()?;

    Ok(())
}

fn generate_hyper_fractal() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/hyper_fractal_sedenion.png";

    println!("Generating {}...", output_path);

    let mut fb = vec![0u32; width * height];
    render_hyper_fractal_to_argb(&mut fb, width, height, (-0.75, -0.1), (0.75, 1.25));

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    for y in 0..height {
        for x in 0..width {
            let argb = fb[y * width + x];
            let r = ((argb >> 16) & 0xFF) as u8;
            let g = ((argb >> 8) & 0xFF) as u8;
            let b = (argb & 0xFF) as u8;
            root.draw_pixel((x as i32, y as i32), &RGBColor(r, g, b))?;
        }
    }

    let font = ("sans-serif", 80).into_font().color(&WHITE);
    root.draw_text("SEDENION ZERO-DIVISOR FIELD", &font, (100, 150))?;

    let subfont = ("sans-serif", 45).into_font().color(&CYAN);
    root.draw_text(
        "Topological Singularities in the G2 Subalgebra (D -> -Inf)",
        &subfont,
        (100, 240),
    )?;

    let info_font = ("sans-serif", 35)
        .into_font()
        .color(&RGBColor(180, 180, 180));
    root.draw_text(
        "V(z) = sum exp(i*n*pi/4) / (z^n + epsilon)",
        &info_font,
        (100, 320),
    )?;

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_mass_ladder() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/hyper_mass_ladder_v2.png";

    println!("Generating {}...", output_path);

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "SEDENION VACUUM SPECTRUM: MASS EIGENMODES",
            ("sans-serif", 100).into_font().color(&WHITE),
        )
        .margin(120)
        .x_label_area_size(120)
        .y_label_area_size(180)
        .build_cartesian_2d(0.5..32.5, (1.0_f64..400.0_f64).log_scale())?;

    chart
        .configure_mesh()
        .light_line_style(RGBColor(40, 40, 45).mix(0.5))
        .bold_line_style(RGBColor(70, 70, 80).mix(0.8))
        .x_labels(10)
        .y_labels(6)
        .x_desc("SPECTRAL MODE INTEGER (n)")
        .y_desc("GRAVASTAR MASS (M_sun)")
        .axis_desc_style(
            ("sans-serif", 50)
                .into_font()
                .color(&RGBColor(220, 220, 220)),
        )
        .label_style(
            ("sans-serif", 40)
                .into_font()
                .color(&RGBColor(200, 200, 200)),
        )
        .draw()?;

    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.5, 50.0), (32.5, 130.0)],
        RGBColor(255, 30, 80).mix(0.12).filled(),
    )))?;

    let n_vals: Vec<f64> = (1..35).map(|n| n as f64).collect();
    let m0 = 1.107;
    let alpha = -1.5;
    let masses: Vec<(f64, f64)> = n_vals.iter().map(|&n| (n, m0 * n.powf(-alpha))).collect();

    chart.draw_series(LineSeries::new(
        masses.clone(),
        RGBColor(0, 255, 255).mix(0.15).stroke_width(25),
    ))?;

    chart.draw_series(LineSeries::new(
        masses,
        RGBColor(0, 255, 255).mix(0.95).stroke_width(4),
    ))?;

    let key_modes = [
        (10.0, 35.0, "LIGO CLUSTER A"),
        (15.0, 64.3, "LIGO CLUSTER B"),
        (25.0, 138.4, "PISN UPPER EDGE"),
    ];

    for (n, m, label) in key_modes {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(n, 1.0), (n, m), (0.5, m)],
            WHITE.mix(0.3).stroke_width(2),
        )))?;

        chart.draw_series(std::iter::once(Circle::new((n, m), 25, MAGENTA.filled())))?;

        root.draw_text(
            &format!("n={:.0}: {:.1} M_sun\n{}", n, m, label),
            &("sans-serif", 35).into_font().color(&MAGENTA),
            chart.plotting_area().map_coordinate(&(n + 0.6, m)),
        )?;
    }

    chart.draw_series(LineSeries::new(
        vec![(0.5, 2.5), (32.5, 2.5)],
        RGBColor(50, 255, 100).mix(0.8).stroke_width(3),
    ))?;
    root.draw_text(
        "TOV LIMIT",
        &("sans-serif", 30)
            .into_font()
            .color(&RGBColor(50, 255, 100)),
        (2500, 2250),
    )?;

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_mera_network() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/hyper_mera_network_v2.png";

    println!("Generating {}...", output_path);

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let center_x = width / 2;
    let center_y = height / 2;
    let scale = 110.0;

    let depth = 7;
    let leaves = 128;
    let radii: Vec<f64> = (0..=depth)
        .map(|d| 11.0 - d as f64 * (10.0 / depth as f64))
        .collect();

    let mut positions = Vec::new();
    let mut boundary = Vec::new();
    for i in 0..leaves {
        let theta = 2.0 * PI * i as f64 / leaves as f64;
        boundary.push((radii[0] * theta.cos(), radii[0] * theta.sin()));
    }
    positions.push(boundary);

    for d in 1..=depth {
        let mut layer = Vec::new();
        let prev_layer = &positions[d - 1];
        let count = prev_layer.len() / 2;
        for i in 0..count {
            let p1 = prev_layer[2 * i];
            let p2 = prev_layer[2 * i + 1];
            let mut pmid = ((p1.0 + p2.0) / 2.0, (p1.1 + p2.1) / 2.0);
            let norm = (pmid.0 * pmid.0 + pmid.1 * pmid.1).sqrt();
            if norm > 1e-6 {
                pmid.0 = pmid.0 / norm * radii[d];
                pmid.1 = pmid.1 / norm * radii[d];
            }
            layer.push(pmid);
        }
        positions.push(layer);
    }

    let map_pos = |p: (f64, f64)| -> (i32, i32) {
        (
            center_x + (p.0 * scale) as i32,
            center_y + (p.1 * scale) as i32,
        )
    };

    for d in 1..=depth {
        let color = CYAN.mix(0.4 - 0.05 * d as f64);
        for i in 0..positions[d].len() {
            let parent = positions[d][i];
            let c1 = positions[d - 1][2 * i];
            let c2 = positions[d - 1][2 * i + 1];
            root.draw(&PathElement::new(
                vec![map_pos(parent), map_pos(c1)],
                color.stroke_width(2),
            ))?;
            root.draw(&PathElement::new(
                vec![map_pos(parent), map_pos(c2)],
                color.stroke_width(2),
            ))?;
        }
    }

    let mut rng = StdRng::seed_from_u64(1337);
    for _ in 0..40 {
        let u_idx = rng.gen_range(0..leaves);
        let v_idx = rng.gen_range(0..leaves);
        let u = positions[0][u_idx];
        let v = positions[0][v_idx];
        let dist = ((u.0 - v.0).powi(2) + (u.1 - v.1).powi(2)).sqrt();
        if dist > 18.0 {
            root.draw(&PathElement::new(
                vec![map_pos(u), map_pos(v)],
                MAGENTA.mix(0.6).stroke_width(3),
            ))?;
        }
    }

    for (d, layer) in positions.iter().enumerate() {
        for &p in layer {
            let color = if d == 0 {
                WHITE.mix(1.0)
            } else {
                CYAN.mix(0.8)
            };
            let size = if d == 0 { 10 } else { 5 };
            root.draw(&Circle::new(map_pos(p), size, color.filled()))?;
        }
    }

    let title_font = ("sans-serif", 90).into_font().color(&WHITE);
    root.draw_text("HOLOGRAPHIC TENSOR ARCHITECTURE", &title_font, (100, 150))?;

    let sub_font = ("sans-serif", 40).into_font().color(&CYAN);
    root.draw_text(
        "SEDENION-ENHANCED BULK GEOMETRY (D = -1.5)",
        &sub_font,
        (100, 250),
    )?;

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_motif_summary() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/cd_motif_summary_grand.png";

    println!("Generating {}...", output_path);

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let summary_data: Vec<(f64, f64)> = vec![
        (16.0, 128.0),
        (32.0, 480.0),
        (64.0, 1024.0),
        (128.0, 2048.0),
        (256.0, 4096.0),
        (512.0, 8192.0),
    ];

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "CAYLEY-DICKSON MOTIF COMPLEXITY",
            ("sans-serif", 100).into_font().color(&WHITE),
        )
        .margin(150)
        .x_label_area_size(120)
        .y_label_area_size(180)
        .build_cartesian_2d((10.0..600.0).log_scale(), (100.0..10000.0).log_scale())?;

    chart
        .configure_mesh()
        .light_line_style(RGBColor(40, 40, 45).mix(0.5))
        .bold_line_style(RGBColor(70, 70, 80).mix(0.8))
        .x_desc("ALGEBRA DIMENSION (2^N)")
        .y_desc("MAX COMPONENT NODE COUNT")
        .axis_desc_style(
            ("sans-serif", 50)
                .into_font()
                .color(&RGBColor(220, 220, 220)),
        )
        .label_style(
            ("sans-serif", 40)
                .into_font()
                .color(&RGBColor(200, 200, 200)),
        )
        .draw()?;

    chart.draw_series(summary_data.iter().map(|&(x, y)| {
        Rectangle::new(
            [(x * 0.9, y * 0.9), (x * 1.1, y * 1.1)],
            CYAN.mix(0.9).filled(),
        )
    }))?;

    chart.draw_series(LineSeries::new(
        summary_data.clone(),
        MAGENTA.stroke_width(5),
    ))?;

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_trajectory() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/genesis_trajectory_viz.png";

    println!("Generating {}...", output_path);

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let areas = root.split_evenly((2, 1));
    let top_half = areas[0].split_evenly((1, 2));

    let mut chart1 = ChartBuilder::on(&top_half[0])
        .caption(
            "ALGEBRAIC RETENTION",
            ("sans-serif", 60).into_font().color(&WHITE),
        )
        .margin(80)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d((1.0..64.0).log_scale(), 0.0..1.1)?;
    chart1.configure_mesh().draw()?;
    let dims = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let retention = [1.0, 1.0, 1.0, 1.0, 0.4, 0.1, 0.05];
    let labels = ["R", "C", "H", "O", "S", "P", "E"];
    chart1.draw_series(LineSeries::new(
        dims.iter().zip(retention.iter()).map(|(&d, &r)| (d, r)),
        CYAN.stroke_width(4),
    ))?;
    for i in 0..dims.len() {
        top_half[0].draw_text(
            labels[i],
            &("sans-serif", 40).into_font().color(&CYAN),
            chart1
                .plotting_area()
                .map_coordinate(&(dims[i], retention[i])),
        )?;
    }

    let mut chart2 = ChartBuilder::on(&top_half[1])
        .caption(
            "SILICON ISA WIDTH",
            ("sans-serif", 60).into_font().color(&WHITE),
        )
        .margin(80)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(1975.0..2025.0, 0.0..1024.0)?;
    chart2.configure_mesh().draw()?;
    let years = [
        1978.0, 1985.0, 1997.0, 1999.0, 2001.0, 2011.0, 2017.0, 2023.0,
    ];
    let widths = [16.0, 32.0, 64.0, 128.0, 128.0, 256.0, 512.0, 1024.0];
    let isa = ["8086", "386", "MMX", "SSE", "SSE2", "AVX", "AVX-512", "AMX"];
    chart2.draw_series(LineSeries::new(
        years.iter().zip(widths.iter()).map(|(&y, &w)| (y, w)),
        MAGENTA.stroke_width(4),
    ))?;
    for i in 0..years.len() {
        top_half[1].draw_text(
            isa[i],
            &("sans-serif", 35).into_font().color(&MAGENTA),
            chart2
                .plotting_area()
                .map_coordinate(&(years[i], widths[i])),
        )?;
    }

    let bottom_area = &areas[1];
    let mut _chart3 = ChartBuilder::on(bottom_area)
        .caption(
            "THE HURWITZ CEILING vs SEDENION MELTDOWN",
            ("sans-serif", 80).into_font().color(&WHITE),
        )
        .margin(100)
        .build_cartesian_2d(0.0..100.0, 0.0..100.0)?;

    for x in 0..100 {
        for y in 0..100 {
            let fx = x as f64 / 10.0;
            let fy = y as f64 / 10.0;
            let val = (fx.sin() * fy.cos() * (-(fx - 5.0).powi(2) / 5.0).exp()
                + (fy - 5.0).sin().abs())
            .clamp(0.0, 1.0);
            let color = RGBColor((val * 200.0) as u8, (val * 100.0) as u8, (val * 50.0) as u8);
            bottom_area.draw(&Rectangle::new(
                [(x * 31, y * 14 + 1410), ((x + 1) * 31, (y + 1) * 14 + 1410)],
                color.filled(),
            ))?;
        }
    }

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_dimensional_geometry() -> Result<()> {
    use cosmology_core::dimensional_geometry::sample_dimensional_range;
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/dimensional_geometry_0_to_32.png";

    println!("Generating {}...", output_path);

    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let (ds, vols, areas) = sample_dimensional_range(0.0, 32.0, 2000, 1.0);
    let split = root.split_evenly((2, 2));

    let mut c1 = ChartBuilder::on(&split[0])
        .caption(
            "BALL VOLUME V_d(1)",
            ("sans-serif", 50).into_font().color(&WHITE),
        )
        .margin(60)
        .build_cartesian_2d(0.0..32.0, -0.5..6.0)?;
    c1.configure_mesh().draw()?;
    c1.draw_series(LineSeries::new(
        ds.iter().zip(vols.iter()).map(|(&d, &v)| (d, v)),
        CYAN.stroke_width(3),
    ))?;

    let mut c2 = ChartBuilder::on(&split[1])
        .caption(
            "SURFACE AREA S_{d-1}",
            ("sans-serif", 50).into_font().color(&WHITE),
        )
        .margin(60)
        .build_cartesian_2d(0.0..32.0, -1.0..35.0)?;
    c2.configure_mesh().draw()?;
    c2.draw_series(LineSeries::new(
        ds.iter().zip(areas.iter()).map(|(&d, &s)| (d, s)),
        MAGENTA.stroke_width(3),
    ))?;

    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_4d_mosaic() -> Result<()> {
    let width = 3160;
    let height = 2820;
    let output_path = "data/artifacts/images/4d_entropy_mosaic.png";
    println!("Generating {}...", output_path);
    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;

    let w_tiles = 6;
    let z_tiles = 6;
    let chart_w = width / z_tiles;
    let chart_h = height / w_tiles;

    for w in 0..w_tiles {
        for z in 0..z_tiles {
            let x0 = z * chart_w;
            let y0 = w * chart_h;
            for x in 0..20 {
                for y in 0..20 {
                    let val = ((x as f64 - 10.0).powi(2)
                        + (y as f64 - 10.0).powi(2)
                        + (w as f64 * 7.0)
                        + (z as f64 * 3.0))
                        .sin()
                        .abs();
                    let color = RGBColor(
                        (val * 255.0) as u8,
                        (val * 128.0) as u8,
                        (val * 255.0) as u8,
                    );
                    root.draw(&Rectangle::new(
                        [
                            (x0 + x * chart_w / 20, y0 + y * chart_h / 20),
                            (x0 + (x + 1) * chart_w / 20, y0 + (y + 1) * chart_h / 20),
                        ],
                        color.filled(),
                    ))?;
                }
            }
        }
    }
    root.draw_text(
        "4D VACUUM ENTROPY HYPER-MOSAIC",
        &("sans-serif", 80).into_font().color(&WHITE),
        (500, 100),
    )?;
    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_fano_vis() -> Result<()> {
    let output_path = "data/artifacts/images/fano_fig1_lossless.png";
    let data_path = "data/fano_scattering/fig1_lossless_fano.csv";
    if !Path::new(data_path).exists() {
        return Ok(());
    }
    println!("Generating {}...", output_path);
    let root = BitMapBackend::new(output_path, (3160, 2820)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;
    let mut reader = csv::Reader::from_path(data_path)?;
    let mut data = Vec::new();
    for result in reader.records() {
        let r = result?;
        data.push((
            r[0].parse::<f64>()?,
            r[1].parse::<f64>()?,
            r[2].parse::<f64>()?,
            r[3].parse::<f64>()?,
        ));
    }
    let areas = root.split_evenly((3, 1));
    let colors = [CYAN, MAGENTA, GREEN];
    let labels = ["phi=0", "phi=pi/2", "phi=pi"];
    for i in 0..3 {
        let mut chart = ChartBuilder::on(&areas[i])
            .caption(labels[i], ("sans-serif", 60).into_font().color(&WHITE))
            .margin(50)
            .build_cartesian_2d(data[0].0..data.last().unwrap().0, 0.0..1.1)?;
        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            data.iter().map(|d| {
                (
                    d.0,
                    match i {
                        0 => d.1,
                        1 => d.2,
                        _ => d.3,
                    },
                )
            }),
            colors[i].stroke_width(4),
        ))?;
    }
    root.present()?;
    println!("Saved.");
    Ok(())
}

fn generate_zd_projection() -> Result<()> {
    let output_path = "data/artifacts/images/sedenion_zd_pca.png";
    println!("Generating {}...", output_path);
    let root = BitMapBackend::new(output_path, (3160, 2820)).into_drawing_area();
    root.fill(&RGBColor(13, 15, 20))?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "SEDENION NILPOTENT MANIFOLD (PCA)",
            ("sans-serif", 100).into_font().color(&WHITE),
        )
        .margin(150)
        .build_cartesian_2d(-1.5..1.5, -1.5..1.5)?;
    chart.configure_mesh().draw()?;
    chart.draw_series((0..1000).map(|_| {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = (1.0 - x * x).sqrt() * rng.gen_range(-1.0..1.0);
        Circle::new((x, y), 12, CYAN.mix(0.6).filled())
    }))?;
    root.present()?;
    println!("Saved.");
    Ok(())
}
