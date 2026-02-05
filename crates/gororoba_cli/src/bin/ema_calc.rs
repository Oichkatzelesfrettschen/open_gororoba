//! ema-calc: Effective Medium Approximation calculator.
//!
//! Provides:
//! - Maxwell-Garnett mixing for dilute inclusions
//! - Bruggeman self-consistent EMA
//! - Drude-Lorentz dielectric models
//! - TMM reflectance calculations
//!
//! Usage:
//!   ema-calc maxwell-garnett --eps-host 1.0 --eps-inc 4.0 --f 0.2
//!   ema-calc bruggeman --eps-1 1.0 --eps-2 4.0 --f 0.5
//!   ema-calc drude --eps-inf 1.0 --omega-p 9.0 --gamma 0.1 --omega-max 15.0
//!   ema-calc tmm --n-layers 1.0,1.38,1.5 --d-layers 100 --wavelength 550

use clap::{Parser, Subcommand};
use materials_core::{
    maxwell_garnett, bruggeman, drude, drude_lorentz, LorentzOscillator,
    tmm_reflection,
};
use num_complex::Complex64;

#[derive(Parser)]
#[command(name = "ema-calc")]
#[command(about = "Effective Medium Approximation calculator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Maxwell-Garnett mixing for dilute inclusions
    MaxwellGarnett {
        /// Host permittivity (real part)
        #[arg(long, default_value = "1.0")]
        eps_host: f64,

        /// Host permittivity (imaginary part)
        #[arg(long, default_value = "0.0")]
        eps_host_im: f64,

        /// Inclusion permittivity (real part)
        #[arg(long, default_value = "4.0")]
        eps_inc: f64,

        /// Inclusion permittivity (imaginary part)
        #[arg(long, default_value = "0.0")]
        eps_inc_im: f64,

        /// Volume fraction of inclusions
        #[arg(long, default_value = "0.1")]
        f: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Bruggeman self-consistent EMA
    Bruggeman {
        /// Component 1 permittivity (real part)
        #[arg(long, default_value = "1.0")]
        eps_1: f64,

        /// Component 1 permittivity (imaginary part)
        #[arg(long, default_value = "0.0")]
        eps_1_im: f64,

        /// Component 2 permittivity (real part)
        #[arg(long, default_value = "4.0")]
        eps_2: f64,

        /// Component 2 permittivity (imaginary part)
        #[arg(long, default_value = "0.0")]
        eps_2_im: f64,

        /// Volume fraction of component 1
        #[arg(long, default_value = "0.5")]
        f: f64,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Drude-Lorentz dielectric function
    Drude {
        /// High-frequency permittivity
        #[arg(long, default_value = "1.0")]
        eps_inf: f64,

        /// Plasma frequency (eV)
        #[arg(long, default_value = "9.0")]
        omega_p: f64,

        /// Drude damping (eV)
        #[arg(long, default_value = "0.1")]
        gamma: f64,

        /// Maximum frequency (eV)
        #[arg(long, default_value = "15.0")]
        omega_max: f64,

        /// Number of frequency points
        #[arg(long, default_value = "100")]
        n_points: usize,

        /// Lorentz oscillators: "S1,w1,g1;S2,w2,g2;..."
        #[arg(long)]
        oscillators: Option<String>,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Transfer Matrix Method reflectance
    Tmm {
        /// Refractive indices: "n0,n1,n2,..." (comma-separated, can include +im for imaginary)
        #[arg(long)]
        n_layers: String,

        /// Layer thicknesses: "d1,d2,..." (in nm, for intermediate layers only)
        #[arg(long)]
        d_layers: String,

        /// Wavelength (nm)
        #[arg(long, default_value = "550.0")]
        wavelength: f64,

        /// Angle of incidence (degrees)
        #[arg(long, default_value = "0.0")]
        theta: f64,

        /// Polarization: s (TE) or p (TM)
        #[arg(long, default_value = "s")]
        polarization: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Wavelength sweep for TMM
    TmmSweep {
        /// Refractive indices (constant, comma-separated)
        #[arg(long)]
        n_layers: String,

        /// Layer thicknesses (nm, comma-separated)
        #[arg(long)]
        d_layers: String,

        /// Start wavelength (nm)
        #[arg(long, default_value = "400.0")]
        wl_start: f64,

        /// End wavelength (nm)
        #[arg(long, default_value = "800.0")]
        wl_end: f64,

        /// Number of wavelength points
        #[arg(long, default_value = "100")]
        n_points: usize,

        /// Angle of incidence (degrees)
        #[arg(long, default_value = "0.0")]
        theta: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::MaxwellGarnett { eps_host, eps_host_im, eps_inc, eps_inc_im, f, json } => {
            run_maxwell_garnett(eps_host, eps_host_im, eps_inc, eps_inc_im, f, json);
        }
        Commands::Bruggeman { eps_1, eps_1_im, eps_2, eps_2_im, f, json } => {
            run_bruggeman(eps_1, eps_1_im, eps_2, eps_2_im, f, json);
        }
        Commands::Drude { eps_inf, omega_p, gamma, omega_max, n_points, oscillators, output } => {
            run_drude(eps_inf, omega_p, gamma, omega_max, n_points, oscillators, output);
        }
        Commands::Tmm { n_layers, d_layers, wavelength, theta, polarization, json } => {
            run_tmm(&n_layers, &d_layers, wavelength, theta, &polarization, json);
        }
        Commands::TmmSweep { n_layers, d_layers, wl_start, wl_end, n_points, theta, output } => {
            run_tmm_sweep(&n_layers, &d_layers, wl_start, wl_end, n_points, theta, output);
        }
    }
}

fn run_maxwell_garnett(eps_host: f64, eps_host_im: f64, eps_inc: f64, eps_inc_im: f64, f: f64, json: bool) {
    let eh = Complex64::new(eps_host, eps_host_im);
    let ei = Complex64::new(eps_inc, eps_inc_im);
    let eps_eff = maxwell_garnett(eh, ei, f);

    if json {
        println!("{{");
        println!("  \"method\": \"maxwell-garnett\",");
        println!("  \"eps_host\": [{}, {}],", eps_host, eps_host_im);
        println!("  \"eps_inc\": [{}, {}],", eps_inc, eps_inc_im);
        println!("  \"f\": {},", f);
        println!("  \"eps_eff\": [{}, {}]", eps_eff.re, eps_eff.im);
        println!("}}");
    } else {
        println!("Maxwell-Garnett Effective Medium");
        println!("  eps_host = {} + {}i", eps_host, eps_host_im);
        println!("  eps_inc  = {} + {}i", eps_inc, eps_inc_im);
        println!("  f        = {}", f);
        println!();
        println!("Result:");
        println!("  eps_eff  = {:.6} + {:.6}i", eps_eff.re, eps_eff.im);
        println!("  n_eff    = {:.6}", eps_eff.sqrt().re);
    }
}

fn run_bruggeman(eps_1: f64, eps_1_im: f64, eps_2: f64, eps_2_im: f64, f: f64, json: bool) {
    let e1 = Complex64::new(eps_1, eps_1_im);
    let e2 = Complex64::new(eps_2, eps_2_im);
    let eps_eff = bruggeman(e1, e2, f);

    if json {
        println!("{{");
        println!("  \"method\": \"bruggeman\",");
        println!("  \"eps_1\": [{}, {}],", eps_1, eps_1_im);
        println!("  \"eps_2\": [{}, {}],", eps_2, eps_2_im);
        println!("  \"f\": {},", f);
        println!("  \"eps_eff\": [{}, {}]", eps_eff.re, eps_eff.im);
        println!("}}");
    } else {
        println!("Bruggeman Effective Medium");
        println!("  eps_1 = {} + {}i", eps_1, eps_1_im);
        println!("  eps_2 = {} + {}i", eps_2, eps_2_im);
        println!("  f     = {} (volume fraction of component 1)", f);
        println!();
        println!("Result:");
        println!("  eps_eff = {:.6} + {:.6}i", eps_eff.re, eps_eff.im);
        println!("  n_eff   = {:.6}", eps_eff.sqrt().re);
    }
}

fn run_drude(eps_inf: f64, omega_p: f64, gamma: f64, omega_max: f64, n_points: usize, oscillators: Option<String>, output: Option<String>) {
    let omega: Vec<f64> = (1..=n_points)
        .map(|i| omega_max * i as f64 / n_points as f64)
        .collect();

    let oscs: Vec<LorentzOscillator> = oscillators
        .map(|s| {
            s.split(';')
                .filter_map(|osc_str| {
                    let parts: Vec<f64> = osc_str.split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        Some(LorentzOscillator {
                            strength: parts[0],
                            omega_0: parts[1],
                            gamma: parts[2],
                        })
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let eps = if oscs.is_empty() {
        drude(&omega, eps_inf, omega_p, gamma)
    } else {
        drude_lorentz(&omega, eps_inf, omega_p, gamma, &oscs)
    };

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["omega_eV", "eps_real", "eps_imag", "n", "k"]).unwrap();
        for (w, e) in omega.iter().zip(eps.iter()) {
            let n_complex = e.sqrt();
            wtr.write_record(&[
                w.to_string(),
                e.re.to_string(),
                e.im.to_string(),
                n_complex.re.to_string(),
                n_complex.im.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", n_points, path);
    } else {
        println!("omega_eV,eps_real,eps_imag,n,k");
        for (i, (w, e)) in omega.iter().zip(eps.iter()).enumerate() {
            if i % (n_points / 10).max(1) == 0 {
                let n_complex = e.sqrt();
                println!("{:.3},{:.4},{:.4},{:.4},{:.4}", w, e.re, e.im, n_complex.re, n_complex.im);
            }
        }
    }
}

fn parse_n_layers(s: &str) -> Vec<Complex64> {
    s.split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.contains('+') {
                let parts: Vec<&str> = part.split('+').collect();
                if parts.len() == 2 {
                    let re = parts[0].trim().parse::<f64>().ok()?;
                    let im_str = parts[1].trim().trim_end_matches('i');
                    let im = im_str.parse::<f64>().ok()?;
                    Some(Complex64::new(re, im))
                } else {
                    None
                }
            } else {
                let n = part.parse::<f64>().ok()?;
                Some(Complex64::new(n, 0.0))
            }
        })
        .collect()
}

fn parse_d_layers(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .collect()
}

fn run_tmm(n_layers_str: &str, d_layers_str: &str, wavelength: f64, theta_deg: f64, polarization: &str, json: bool) {
    let n_layers = parse_n_layers(n_layers_str);
    let d_layers = parse_d_layers(d_layers_str);
    let theta = theta_deg.to_radians();
    let s_pol = polarization.to_lowercase() == "s";

    let result = tmm_reflection(&n_layers, &d_layers, wavelength, theta, s_pol);

    if json {
        println!("{{");
        println!("  \"wavelength_nm\": {},", wavelength);
        println!("  \"theta_deg\": {},", theta_deg);
        println!("  \"polarization\": \"{}\",", if s_pol { "s" } else { "p" });
        println!("  \"r_real\": {},", result.r.re);
        println!("  \"r_imag\": {},", result.r.im);
        println!("  \"reflectance\": {}", result.reflectance);
        println!("}}");
    } else {
        println!("Transfer Matrix Method");
        println!("  n_layers = {:?}", n_layers.iter().map(|n| format!("{:.3}+{:.3}i", n.re, n.im)).collect::<Vec<_>>());
        println!("  d_layers = {:?} nm", d_layers);
        println!("  lambda   = {} nm", wavelength);
        println!("  theta    = {} deg", theta_deg);
        println!("  pol      = {}", if s_pol { "s (TE)" } else { "p (TM)" });
        println!();
        println!("Result:");
        println!("  r = {:.6} + {:.6}i", result.r.re, result.r.im);
        println!("  R = {:.4} ({:.1}%)", result.reflectance, result.reflectance * 100.0);
    }
}

fn run_tmm_sweep(n_layers_str: &str, d_layers_str: &str, wl_start: f64, wl_end: f64, n_points: usize, theta_deg: f64, output: Option<String>) {
    let n_layers = parse_n_layers(n_layers_str);
    let d_layers = parse_d_layers(d_layers_str);
    let theta = theta_deg.to_radians();

    let wavelengths: Vec<f64> = (0..n_points)
        .map(|i| wl_start + (wl_end - wl_start) * i as f64 / (n_points - 1) as f64)
        .collect();

    let results: Vec<_> = wavelengths.iter()
        .map(|&wl| tmm_reflection(&n_layers, &d_layers, wl, theta, true))
        .collect();

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["wavelength_nm", "reflectance"]).unwrap();
        for (wl, r) in wavelengths.iter().zip(results.iter()) {
            wtr.write_record(&[wl.to_string(), r.reflectance.to_string()]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", n_points, path);
    } else {
        println!("wavelength_nm,reflectance");
        for (i, (wl, r)) in wavelengths.iter().zip(results.iter()).enumerate() {
            if i % (n_points / 20).max(1) == 0 {
                println!("{:.1},{:.6}", wl, r.reflectance);
            }
        }
    }
}
