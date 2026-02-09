//! tcmt-sweep: Sweep TCMT parameters for bistability analysis.
//!
//! Usage: tcmt-sweep --omega 2.0 --n-points 100 --output bistability.csv

use clap::Parser;
use optics_core::{
    find_turning_points, find_turning_points_physical, solve_normalized_cubic,
    trace_hysteresis_loop, KerrCavity,
};

#[derive(Parser)]
#[command(name = "tcmt-sweep")]
#[command(about = "Sweep TCMT parameters for optical bistability analysis")]
struct Args {
    /// Normalized detuning (Omega = delta/g)
    #[arg(long, default_value = "2.0")]
    omega: f64,

    /// Minimum normalized power u^2
    #[arg(long, default_value = "0.5")]
    u_min: f64,

    /// Maximum normalized power u^2
    #[arg(long, default_value = "3.0")]
    u_max: f64,

    /// Number of sweep points
    #[arg(long, default_value = "100")]
    n_points: usize,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Run hysteresis trace mode
    #[arg(long)]
    hysteresis: bool,

    /// Run turning point analysis mode
    #[arg(long)]
    turning_points: bool,

    /// Run full S-curve mode (all solutions at each power)
    #[arg(long)]
    s_curve: bool,

    /// Total quality factor (for physical units)
    #[arg(long, default_value = "1000.0")]
    q_total: f64,

    /// Wavelength in nm (for physical units)
    #[arg(long, default_value = "1550.0")]
    wavelength: f64,
}

fn main() {
    let args = Args::parse();

    eprintln!(
        "TCMT sweep: Omega={}, power range=[{}, {}], n_points={}",
        args.omega, args.u_min, args.u_max, args.n_points
    );

    // Check bistability condition
    let omega_crit = 3.0_f64.sqrt();
    if args.omega.abs() <= omega_crit {
        eprintln!(
            "Warning: |Omega| = {:.4} <= sqrt(3) = {:.4}: below bistability threshold",
            args.omega.abs(),
            omega_crit
        );
    }

    if args.turning_points {
        run_turning_point_analysis(&args);
    } else if args.hysteresis {
        run_hysteresis_trace(&args);
    } else if args.s_curve {
        run_s_curve(&args);
    } else {
        // Default: run S-curve
        run_s_curve(&args);
    }
}

fn run_turning_point_analysis(args: &Args) {
    eprintln!("\n=== Turning Point Analysis ===");

    match find_turning_points(args.omega) {
        Ok(result) => {
            eprintln!("Bistability detected:");
            eprintln!(
                "  Lower turning point: u^2 = {:.6}, y = {:.6}",
                result.turning_lower.u_squared, result.turning_lower.y
            );
            eprintln!(
                "  Upper turning point: u^2 = {:.6}, y = {:.6}",
                result.turning_upper.u_squared, result.turning_upper.y
            );
            eprintln!("  Hysteresis width: {:.6}", result.width_normalized);
            eprintln!("  Energy contrast: {:.6}", result.energy_contrast);
            eprintln!("  Stability margin: {:.4}", result.stability_margin);

            // Also compute physical units
            let cavity = KerrCavity::normalized(args.q_total, 1.0);
            let g = cavity.gamma_total() / 2.0;
            let detuning = args.omega * g;

            if let Ok(phys_result) = find_turning_points_physical(&cavity, detuning) {
                if let (Some(p_low), Some(p_high)) = (
                    phys_result.turning_lower.power,
                    phys_result.turning_upper.power,
                ) {
                    eprintln!("\nPhysical units (Q_total = {}):", args.q_total);
                    eprintln!("  Lower threshold power: {:.4e} W", p_low);
                    eprintln!("  Upper threshold power: {:.4e} W", p_high);
                    eprintln!(
                        "  Hysteresis width: {:.4e} W",
                        phys_result.width_power.unwrap_or(0.0)
                    );
                }
            }

            // Output to CSV if requested
            if let Some(path) = &args.output {
                let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
                wtr.write_record(["parameter", "value"]).unwrap();
                wtr.write_record(["omega", &args.omega.to_string()])
                    .unwrap();
                wtr.write_record(["u_sq_lower", &result.turning_lower.u_squared.to_string()])
                    .unwrap();
                wtr.write_record(["u_sq_upper", &result.turning_upper.u_squared.to_string()])
                    .unwrap();
                wtr.write_record(["y_lower", &result.turning_lower.y.to_string()])
                    .unwrap();
                wtr.write_record(["y_upper", &result.turning_upper.y.to_string()])
                    .unwrap();
                wtr.write_record(["width", &result.width_normalized.to_string()])
                    .unwrap();
                wtr.write_record(["energy_contrast", &result.energy_contrast.to_string()])
                    .unwrap();
                wtr.write_record(["stability_margin", &result.stability_margin.to_string()])
                    .unwrap();
                wtr.flush().unwrap();
                println!("Wrote turning point analysis to {}", path);
            }
        }
        Err(e) => {
            eprintln!("No bistability: {}", e);
        }
    }
}

fn run_hysteresis_trace(args: &Args) {
    eprintln!("\n=== Hysteresis Trace ===");

    let trace = trace_hysteresis_loop(args.omega, args.n_points, (args.u_min, args.u_max));

    if trace.has_hysteresis() {
        eprintln!("Hysteresis detected!");
        if let Some((p_low, p_high)) = trace.hysteresis_power_range() {
            eprintln!("  Hysteresis power range: [{:.4}, {:.4}]", p_low, p_high);
        }
    } else {
        eprintln!("No hysteresis in scanned range.");
    }

    // Output
    if let Some(path) = &args.output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record(["u_squared", "y_up", "y_down", "has_hysteresis"])
            .unwrap();

        for i in 0..trace.powers.len() {
            let up = trace.up_sweep[i].map(|v| v.to_string()).unwrap_or_default();
            let down = trace.down_sweep[i]
                .map(|v| v.to_string())
                .unwrap_or_default();
            let hyst = match (&trace.up_sweep[i], &trace.down_sweep[i]) {
                (Some(u), Some(d)) if (u - d).abs() > 1e-10 => "1",
                _ => "0",
            };
            wtr.write_record(&[trace.powers[i].to_string(), up, down, hyst.to_string()])
                .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote hysteresis trace to {}", path);
    } else {
        println!("u_squared,y_up,y_down,has_hysteresis");
        for i in 0..trace.powers.len() {
            let up = trace.up_sweep[i]
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default();
            let down = trace.down_sweep[i]
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default();
            let hyst = match (&trace.up_sweep[i], &trace.down_sweep[i]) {
                (Some(u), Some(d)) if (u - d).abs() > 1e-10 => "1",
                _ => "0",
            };
            println!("{:.6},{},{},{}", trace.powers[i], up, down, hyst);
        }
    }
}

fn run_s_curve(args: &Args) {
    eprintln!("\n=== S-Curve Analysis ===");

    let step = (args.u_max - args.u_min) / (args.n_points - 1).max(1) as f64;

    // Collect all data points
    let mut data: Vec<(f64, Vec<f64>, Vec<bool>)> = Vec::new();

    for i in 0..args.n_points {
        let u_sq = args.u_min + i as f64 * step;
        let result = solve_normalized_cubic(u_sq, args.omega);

        let mut sorted_y = result.y_solutions.clone();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Get stability for sorted solutions
        let mut sorted_stable = vec![false; sorted_y.len()];
        for (j, y) in result.y_solutions.iter().enumerate() {
            if let Some(pos) = sorted_y.iter().position(|&sy| (sy - y).abs() < 1e-12) {
                sorted_stable[pos] = result.stable[j];
            }
        }

        data.push((u_sq, sorted_y, sorted_stable));
    }

    // Report statistics
    let max_solutions = data.iter().map(|(_, y, _)| y.len()).max().unwrap_or(0);
    let bistable_count = data.iter().filter(|(_, y, _)| y.len() == 3).count();
    eprintln!("Max solutions at single power: {}", max_solutions);
    eprintln!(
        "Points with 3 solutions (bistable): {} / {}",
        bistable_count, args.n_points
    );

    // Output
    if let Some(path) = &args.output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record([
            "u_squared",
            "n_solutions",
            "y_1",
            "stable_1",
            "y_2",
            "stable_2",
            "y_3",
            "stable_3",
        ])
        .unwrap();

        for (u_sq, ys, stables) in &data {
            let n = ys.len();
            let y1 = ys.first().map(|v| v.to_string()).unwrap_or_default();
            let s1 = stables
                .first()
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            let y2 = ys.get(1).map(|v| v.to_string()).unwrap_or_default();
            let s2 = stables
                .get(1)
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            let y3 = ys.get(2).map(|v| v.to_string()).unwrap_or_default();
            let s3 = stables
                .get(2)
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            wtr.write_record(&[
                u_sq.to_string(),
                n.to_string(),
                y1,
                s1.to_string(),
                y2,
                s2.to_string(),
                y3,
                s3.to_string(),
            ])
            .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote S-curve data to {}", path);
    } else {
        println!("u_squared,n_solutions,y_1,stable_1,y_2,stable_2,y_3,stable_3");
        for (u_sq, ys, stables) in &data {
            let n = ys.len();
            let y1 = ys.first().map(|v| format!("{:.6}", v)).unwrap_or_default();
            let s1 = stables
                .first()
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            let y2 = ys.get(1).map(|v| format!("{:.6}", v)).unwrap_or_default();
            let s2 = stables
                .get(1)
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            let y3 = ys.get(2).map(|v| format!("{:.6}", v)).unwrap_or_default();
            let s3 = stables
                .get(2)
                .map(|v| if *v { "1" } else { "0" })
                .unwrap_or("");
            println!(
                "{:.6},{},{},{},{},{},{},{}",
                u_sq, n, y1, s1, y2, s2, y3, s3
            );
        }
    }
}
