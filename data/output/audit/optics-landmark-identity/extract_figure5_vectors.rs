use std::{
    env, fs,
    io::{self, Write},
};
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = env::args().nth(1).ok_or("expected evidence directory")?;
    let scattering = fs::read_to_string(format!("{directory}/FanoCloak-lossy-sct.eps"))?;
    let absorption = fs::read_to_string(format!("{directory}/FanoCloak-lossy-abs.eps"))?;
    let mut rows = Vec::<(&str, &str, Point, usize)>::new();
    let mut color = "";
    let mut chain = Vec::<Point>::new();
    let mut green_polygons = Vec::<Vec<Point>>::new();
    let flush = |chain: &mut Vec<Point>, rows: &mut Vec<(&str, &str, Point, usize)>| {
        if chain.len() > 8 && chain.first() == chain.last() {
            let minimum_x = chain
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min);
            let maximum_x = chain
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let minimum_y = chain
                .iter()
                .map(|point| point.y)
                .fold(f64::INFINITY, f64::min);
            let maximum_y = chain
                .iter()
                .map(|point| point.y)
                .fold(f64::NEG_INFINITY, f64::max);
            rows.push((
                "scattering",
                "mie_total_circle",
                Point {
                    x: (minimum_x + maximum_x) / 2.0,
                    y: (minimum_y + maximum_y) / 2.0,
                },
                chain.len() - 1,
            ));
        }
        chain.clear();
    };
    for line in scattering.lines() {
        if line.ends_with(" SC") {
            color = line;
        }
        if !line.starts_with("NP ") {
            continue;
        }
        let tokens: Vec<_> = line.split_whitespace().collect();
        if color == "0 0 1 SC" && tokens.len() == 8 && tokens[3] == "m" && tokens[6] == "l" {
            let start = Point {
                x: tokens[1].parse()?,
                y: tokens[2].parse()?,
            };
            let end = Point {
                x: tokens[4].parse()?,
                y: tokens[5].parse()?,
            };
            if chain.last().is_some_and(|last| *last != start) {
                flush(&mut chain, &mut rows);
            }
            if chain.is_empty() {
                chain.push(start);
            }
            chain.push(end);
            if chain.len() > 8 && chain.first() == chain.last() {
                flush(&mut chain, &mut rows);
            }
        } else {
            flush(&mut chain, &mut rows);
            if color == "0 0.5019 0 SC" && tokens.len() > 100 {
                let mut points = Vec::new();
                for chunk in tokens[1..].chunks_exact(3) {
                    if matches!(chunk[2], "m" | "l") {
                        points.push(Point {
                            x: chunk[0].parse()?,
                            y: chunk[1].parse()?,
                        });
                    }
                }
                if !green_polygons.contains(&points) {
                    green_polygons.push(points);
                }
            }
        }
    }
    flush(&mut chain, &mut rows);
    let mut absorption_curve = Vec::new();
    let mut numeric_stack = Vec::<f64>::new();
    let mut in_green = false;
    let mut absorption_color = "";
    for line in absorption.lines() {
        let line = line.trim();
        if line == "c8" || line == "c9" || line == "c10" {
            absorption_color = line;
        }
        if line == "c9" {
            in_green = true;
        } else if line.starts_with("/c10 ") {
            in_green = false;
        }
        let tokens: Vec<_> = line.split_whitespace().collect();
        if absorption_color == "c8" && tokens.len() == 5 && tokens[4] == "FO" {
            rows.push((
                "absorption",
                "mie_total_circle",
                Point {
                    x: tokens[2].parse()?,
                    y: tokens[3].parse()?,
                },
                0,
            ));
        }
        if in_green {
            if tokens.iter().all(|token| token.parse::<f64>().is_ok()) {
                numeric_stack.extend(tokens.iter().map(|token| token.parse::<f64>().unwrap()));
            } else if let Some(index) = tokens.iter().position(|token| *token == "MP") {
                numeric_stack.extend(
                    tokens[..index]
                        .iter()
                        .map(|token| token.parse::<f64>().unwrap()),
                );
                let count = numeric_stack.pop().ok_or("missing path count")? as usize;
                let mut point = Point {
                    y: numeric_stack.pop().ok_or("missing y")?,
                    x: numeric_stack.pop().ok_or("missing x")?,
                };
                absorption_curve.push(point);
                for _ in 1..count {
                    point.y += numeric_stack.pop().ok_or("missing dy")?;
                    point.x += numeric_stack.pop().ok_or("missing dx")?;
                    absorption_curve.push(point);
                }
                assert!(numeric_stack.is_empty());
            }
        }
    }
    for point in &absorption_curve {
        rows.push(("absorption", "fitted_total_polyline", *point, 0));
    }
    let convert = |panel: &str, point: Point| -> (f64, f64) {
        if panel == "scattering" {
            (
                0.22 + (point.x - 2246.0) * 0.013 / 2674.0,
                (3246.0 - point.y) * 2.5 / 2040.0,
            )
        } else {
            (
                0.22 + (point.x - 720.0) * 0.013 / 3850.0,
                (3350.0 - point.y) * 0.7 / 2938.0,
            )
        }
    };
    let mut csv = fs::File::create_new(format!("{directory}/source-vector-coordinates.csv"))?;
    writeln!(
        csv,
        "panel,role,vector_x,vector_y,omega_over_omega_p,cross_section_over_2lambda_pi,circle_segments"
    )?;
    for (panel, role, point, count) in &rows {
        let (frequency, value) = convert(panel, *point);
        writeln!(
            csv,
            "{panel},{role},{},{},{frequency:.12},{value:.12},{count}",
            point.x, point.y
        )?;
    }
    let mut stdout = io::stdout().lock();
    for panel in ["scattering", "absorption"] {
        let mut samples: Vec<_> = rows
            .iter()
            .filter(|row| row.0 == panel && row.1 == "mie_total_circle")
            .map(|row| convert(panel, row.2))
            .collect();
        samples.sort_by(|left, right| left.0.total_cmp(&right.0));
        writeln!(stdout, "panel={panel} total_mie_markers={}", samples.len())?;
        for &(frequency, value) in &samples {
            if (frequency - 0.2282).abs() < 0.00035 {
                writeln!(stdout, "mie_marker omega={frequency:.12} value={value:.12}")?;
            }
        }
        for pair in samples.windows(2) {
            if pair[0].0 <= 0.2282 && pair[1].0 >= 0.2282 {
                let fraction = (0.2282 - pair[0].0) / (pair[1].0 - pair[0].0);
                writeln!(
                    stdout,
                    "mie_marker_linear_interpolation_at_0_2282={:.12}",
                    pair[0].1 + fraction * (pair[1].1 - pair[0].1)
                )?;
            }
        }
    }
    let target_x = 2246.0 + 0.0082 / 0.013 * 2674.0;
    let mut intersections = Vec::new();
    for polygon in &green_polygons {
        for pair in polygon.windows(2) {
            if (pair[0].x - target_x) * (pair[1].x - target_x) <= 0.0 && pair[0].x != pair[1].x {
                let fraction = (target_x - pair[0].x) / (pair[1].x - pair[0].x);
                intersections.push(
                    convert(
                        "scattering",
                        Point {
                            x: target_x,
                            y: pair[0].y + fraction * (pair[1].y - pair[0].y),
                        },
                    )
                    .1,
                );
            }
        }
    }
    writeln!(
        stdout,
        "scattering_fitted_total_polygon_intersections_at_0_2282={intersections:?}"
    )?;
    absorption_curve.sort_by(|left, right| left.x.total_cmp(&right.x));
    for pair in absorption_curve.windows(2) {
        let left = convert("absorption", pair[0]);
        let right = convert("absorption", pair[1]);
        if left.0 <= 0.2282 && right.0 >= 0.2282 {
            let fraction = (0.2282 - left.0) / (right.0 - left.0);
            writeln!(
                stdout,
                "absorption_fitted_total_interpolation_at_0_2282={:.12}",
                left.1 + fraction * (right.1 - left.1)
            )?;
        }
    }
    Ok(())
}
