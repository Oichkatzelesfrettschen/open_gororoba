//! frac-laplacian: Fractional Laplacian computation via spectral methods.
//!
//! Usage: frac-laplacian --mode periodic --dim 1 --s 0.5 --n 64 --output result.csv

use clap::{Parser, ValueEnum};
use ndarray::{Array2, Array3};
use spectral_core::{
    fractional_laplacian_dirichlet_1d, fractional_laplacian_dirichlet_2d,
    fractional_laplacian_periodic_1d, fractional_laplacian_periodic_2d,
    fractional_laplacian_periodic_3d,
};
use std::f64::consts::PI;

#[derive(Clone, Copy, ValueEnum)]
enum BoundaryCondition {
    Periodic,
    Dirichlet,
}

#[derive(Clone, Copy, ValueEnum)]
enum TestFunction {
    /// Single sine mode: sin(2*pi*x/L)
    Sine,
    /// Gaussian pulse: exp(-((x-0.5)^2) / (2*sigma^2))
    Gaussian,
    /// Polynomial: x^2 * (1-x) for Dirichlet, or x^2 for periodic
    Polynomial,
}

#[derive(Parser)]
#[command(name = "frac-laplacian")]
#[command(about = "Compute fractional Laplacian via spectral methods")]
struct Args {
    /// Boundary condition type
    #[arg(short, long, value_enum, default_value = "periodic")]
    mode: BoundaryCondition,

    /// Spatial dimension (1, 2, or 3)
    #[arg(short, long, default_value = "1")]
    dim: usize,

    /// Fractional power s (typically in (0, 1])
    #[arg(short, long, default_value = "0.5")]
    s: f64,

    /// Grid size in each dimension
    #[arg(short, long, default_value = "64")]
    n: usize,

    /// Domain length
    #[arg(short = 'L', long, default_value = "1.0")]
    length: f64,

    /// Test function to apply
    #[arg(short, long, value_enum, default_value = "sine")]
    function: TestFunction,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Output as JSON summary
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    eprintln!(
        "Fractional Laplacian: mode={:?}, dim={}, s={}, n={}",
        match args.mode {
            BoundaryCondition::Periodic => "periodic",
            BoundaryCondition::Dirichlet => "dirichlet",
        },
        args.dim,
        args.s,
        args.n
    );

    match args.dim {
        1 => run_1d(&args),
        2 => run_2d(&args),
        3 => run_3d(&args),
        _ => {
            eprintln!("Dimension must be 1, 2, or 3");
            std::process::exit(1);
        }
    }
}

fn run_1d(args: &Args) {
    let n = args.n;
    let l = args.length;

    let (u, x_coords) = match args.mode {
        BoundaryCondition::Periodic => {
            let x: Vec<f64> = (0..n).map(|i| i as f64 * l / n as f64).collect();
            let u = generate_test_function_1d(&x, l, args.function, false);
            (u, x)
        }
        BoundaryCondition::Dirichlet => {
            let h = l / (n + 1) as f64;
            let x: Vec<f64> = (1..=n).map(|i| i as f64 * h).collect();
            let u = generate_test_function_1d(&x, l, args.function, true);
            (u, x)
        }
    };

    let result = match args.mode {
        BoundaryCondition::Periodic => fractional_laplacian_periodic_1d(&u, args.s, l),
        BoundaryCondition::Dirichlet => fractional_laplacian_dirichlet_1d(&u, args.s, l),
    };

    // Compute statistics
    let max_u: f64 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_result: f64 = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let l2_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    let l2_result: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();

    eprintln!(
        "max|u| = {:.6e}, max|(-Delta)^s u| = {:.6e}",
        max_u, max_result
    );
    eprintln!("L2(u) = {:.6e}, L2(result) = {:.6e}", l2_u, l2_result);

    if args.json {
        println!("{{");
        println!("  \"dim\": {},", args.dim);
        println!("  \"n\": {},", n);
        println!("  \"s\": {},", args.s);
        println!("  \"max_u\": {},", max_u);
        println!("  \"max_result\": {},", max_result);
        println!("  \"l2_u\": {},", l2_u);
        println!("  \"l2_result\": {}", l2_result);
        println!("}}");
    } else if let Some(path) = &args.output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record(["x", "u", "frac_laplacian_u"]).unwrap();
        for i in 0..n {
            wtr.write_record(&[
                x_coords[i].to_string(),
                u[i].to_string(),
                result[i].to_string(),
            ])
            .unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", n, path);
    } else {
        println!("x,u,frac_laplacian_u");
        for i in 0..n.min(20) {
            println!("{},{},{}", x_coords[i], u[i], result[i]);
        }
        if n > 20 {
            println!("... and {} more points", n - 20);
        }
    }
}

fn run_2d(args: &Args) {
    let n = args.n;
    let l = args.length;

    let (u, result) = match args.mode {
        BoundaryCondition::Periodic => {
            let u = Array2::from_shape_fn((n, n), |(i, j)| {
                let x = i as f64 * l / n as f64;
                let y = j as f64 * l / n as f64;
                match args.function {
                    TestFunction::Sine => (2.0 * PI * x / l).sin() * (2.0 * PI * y / l).sin(),
                    TestFunction::Gaussian => {
                        let dx = x - 0.5 * l;
                        let dy = y - 0.5 * l;
                        (-((dx * dx + dy * dy) / (2.0 * 0.1 * 0.1))).exp()
                    }
                    TestFunction::Polynomial => x * x * y * y,
                }
            });
            let r = fractional_laplacian_periodic_2d(&u, args.s, l, l);
            (u, r)
        }
        BoundaryCondition::Dirichlet => {
            let h = l / (n + 1) as f64;
            let u = Array2::from_shape_fn((n, n), |(i, j)| {
                let x = (i + 1) as f64 * h;
                let y = (j + 1) as f64 * h;
                match args.function {
                    TestFunction::Sine => (PI * x / l).sin() * (PI * y / l).sin(),
                    TestFunction::Gaussian => {
                        let dx = x - 0.5 * l;
                        let dy = y - 0.5 * l;
                        (-((dx * dx + dy * dy) / (2.0 * 0.1 * 0.1))).exp()
                    }
                    TestFunction::Polynomial => x * (l - x) * y * (l - y),
                }
            });
            let r = fractional_laplacian_dirichlet_2d(&u, args.s, l, l);
            (u, r)
        }
    };

    let max_u: f64 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_result: f64 = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "2D grid {}x{}, max|u| = {:.6e}, max|result| = {:.6e}",
        n, n, max_u, max_result
    );

    if args.json {
        println!("{{");
        println!("  \"dim\": 2,");
        println!("  \"n\": {},", n);
        println!("  \"s\": {},", args.s);
        println!("  \"max_u\": {},", max_u);
        println!("  \"max_result\": {}", max_result);
        println!("}}");
    } else {
        println!("2D computation complete: {}x{} grid", n, n);
        println!("max|u| = {:.6e}", max_u);
        println!("max|(-Delta)^s u| = {:.6e}", max_result);
    }
}

fn run_3d(args: &Args) {
    if matches!(args.mode, BoundaryCondition::Dirichlet) {
        eprintln!("3D Dirichlet not implemented, using periodic instead");
    }

    let n = args.n;
    let l = args.length;

    let u = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
        let x = i as f64 * l / n as f64;
        let y = j as f64 * l / n as f64;
        let z = k as f64 * l / n as f64;
        match args.function {
            TestFunction::Sine => {
                (2.0 * PI * x / l).sin() * (2.0 * PI * y / l).sin() * (2.0 * PI * z / l).sin()
            }
            TestFunction::Gaussian => {
                let dx = x - 0.5 * l;
                let dy = y - 0.5 * l;
                let dz = z - 0.5 * l;
                (-((dx * dx + dy * dy + dz * dz) / (2.0 * 0.1 * 0.1))).exp()
            }
            TestFunction::Polynomial => x * x * y * y * z * z,
        }
    });

    let result = fractional_laplacian_periodic_3d(&u, args.s, l, l, l);

    let max_u: f64 = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_result: f64 = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "3D grid {}x{}x{}, max|u| = {:.6e}, max|result| = {:.6e}",
        n, n, n, max_u, max_result
    );

    if args.json {
        println!("{{");
        println!("  \"dim\": 3,");
        println!("  \"n\": {},", n);
        println!("  \"s\": {},", args.s);
        println!("  \"max_u\": {},", max_u);
        println!("  \"max_result\": {}", max_result);
        println!("}}");
    } else {
        println!("3D computation complete: {}x{}x{} grid", n, n, n);
        println!("max|u| = {:.6e}", max_u);
        println!("max|(-Delta)^s u| = {:.6e}", max_result);
    }
}

fn generate_test_function_1d(x: &[f64], l: f64, func: TestFunction, dirichlet: bool) -> Vec<f64> {
    x.iter()
        .map(|&xi| match func {
            TestFunction::Sine => {
                if dirichlet {
                    (PI * xi / l).sin()
                } else {
                    (2.0 * PI * xi / l).sin()
                }
            }
            TestFunction::Gaussian => {
                let center = 0.5 * l;
                let sigma = 0.1 * l;
                (-(xi - center).powi(2) / (2.0 * sigma * sigma)).exp()
            }
            TestFunction::Polynomial => {
                if dirichlet {
                    xi * (l - xi) // Vanishes at boundaries
                } else {
                    xi * xi
                }
            }
        })
        .collect()
}
