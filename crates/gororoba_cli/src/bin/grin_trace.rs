//! grin-trace: Trace rays through gradient-index optical media.
//!
//! Usage: grin-trace --n0 1.5 --g 0.1 --output ray.csv

use clap::Parser;
use optics_core::{Ray, trace_ray, trace_ray_absorbing, GrinFiber, AbsorbingGrinMedium};
use num_complex::Complex64;

#[derive(Parser)]
#[command(name = "grin-trace")]
#[command(about = "Trace rays through GRIN optical media")]
struct Args {
    /// Core refractive index
    #[arg(long, default_value = "1.5")]
    n0: f64,

    /// Gradient parameter
    #[arg(long, default_value = "0.1")]
    g: f64,

    /// Initial x position
    #[arg(long, default_value = "0.5")]
    x0: f64,

    /// Initial y position
    #[arg(long, default_value = "0.0")]
    y0: f64,

    /// Initial z position
    #[arg(long, default_value = "0.0")]
    z0: f64,

    /// Step size
    #[arg(long, default_value = "0.05")]
    step: f64,

    /// Maximum steps
    #[arg(long, default_value = "1000")]
    max_steps: usize,

    /// Extinction coefficient (for absorbing media)
    #[arg(long)]
    kappa: Option<f64>,

    /// Wavelength in nm (for absorbing media)
    #[arg(long, default_value = "1550.0")]
    wavelength: f64,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,
}

/// Simple absorbing fiber
struct AbsorbingFiber {
    n0: f64,
    g: f64,
    kappa: f64,
}

impl AbsorbingGrinMedium for AbsorbingFiber {
    fn gradient_and_n_complex(&self, p: optics_core::grin::Vec3) -> (optics_core::grin::Vec3, Complex64) {
        // Distance from z-axis
        let r2 = p[0] * p[0] + p[1] * p[1];
        let arg = 1.0 - self.g * self.g * r2;

        if arg <= 0.0 {
            return ([0.0, 0.0, 0.0], Complex64::new(1.0, self.kappa));
        }

        let n_real = self.n0 * arg.sqrt();
        let factor = -self.n0 * self.g * self.g / arg.sqrt();
        let grad = [factor * p[0], factor * p[1], 0.0];

        (grad, Complex64::new(n_real, self.kappa))
    }
}

fn main() {
    let args = Args::parse();

    let ray = Ray {
        pos: [args.x0, args.y0, args.z0],
        dir: [0.0, 0.0, 1.0],
    };

    eprintln!("GRIN trace: n0={}, g={}, pos=({}, {}, {})",
        args.n0, args.g, args.x0, args.y0, args.z0);

    if let Some(kappa) = args.kappa {
        // Absorbing medium
        let medium = AbsorbingFiber { n0: args.n0, g: args.g, kappa };
        let result = trace_ray_absorbing(ray, &medium, args.step, args.wavelength, args.max_steps, 1e-6);

        eprintln!("Traced {} points, final amplitude = {:.4e}",
            result.positions.len(), result.amplitudes.last().unwrap_or(&1.0));

        output_absorbing(&args.output, &result);
    } else {
        // Non-absorbing medium
        let medium = GrinFiber {
            n0: args.n0,
            g: args.g,
            axis: [0.0, 0.0, 1.0],
        };
        let result = trace_ray(ray, &medium, args.step, args.max_steps);

        eprintln!("Traced {} points", result.positions.len());

        output_result(&args.output, &result);
    }
}

fn output_result(output: &Option<String>, result: &optics_core::RayTraceResult) {
    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record(["s", "x", "y", "z", "dx", "dy", "dz"]).unwrap();
        for i in 0..result.positions.len() {
            let p = result.positions[i];
            let d = result.directions[i];
            wtr.write_record(&[
                result.arc_lengths[i].to_string(),
                p[0].to_string(), p[1].to_string(), p[2].to_string(),
                d[0].to_string(), d[1].to_string(), d[2].to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", result.positions.len(), path);
    } else {
        println!("s,x,y,z,dx,dy,dz");
        for i in 0..result.positions.len() {
            let p = result.positions[i];
            let d = result.directions[i];
            println!("{},{},{},{},{},{},{}",
                result.arc_lengths[i], p[0], p[1], p[2], d[0], d[1], d[2]);
        }
    }
}

fn output_absorbing(output: &Option<String>, result: &optics_core::RayTraceResult) {
    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
        wtr.write_record(["s", "x", "y", "z", "amplitude", "phase"]).unwrap();
        for i in 0..result.positions.len() {
            let p = result.positions[i];
            wtr.write_record(&[
                result.arc_lengths[i].to_string(),
                p[0].to_string(), p[1].to_string(), p[2].to_string(),
                result.amplitudes[i].to_string(),
                result.phases[i].to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", result.positions.len(), path);
    } else {
        println!("s,x,y,z,amplitude,phase");
        for i in 0..result.positions.len() {
            let p = result.positions[i];
            println!("{},{},{},{},{},{}",
                result.arc_lengths[i], p[0], p[1], p[2],
                result.amplitudes[i], result.phases[i]);
        }
    }
}
