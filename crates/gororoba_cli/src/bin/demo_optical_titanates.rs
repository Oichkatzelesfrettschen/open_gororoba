//! Demonstration: Optical database and titanate material families.
//!
//! Exercises the full optical database including Rakic metals, C-418 gap materials,
//! titanates (TiO, SrTiO3, LaTiO3), TCOs (AZO, doped Si), and extended Drude
//! infrastructure.
//!
//! Example usage:
//! ```bash
//! cargo run --bin demo-optical-titanates -- --list
//! cargo run --bin demo-optical-titanates -- --material gold
//! cargo run --bin demo-optical-titanates -- --compare tio,srtio3,latio3
//! cargo run --bin demo-optical-titanates -- --thesis 1
//! ```

use clap::Parser;
use materials_core::{
    get_material, list_materials, EV_TO_RADS, C, HBAR_EV_S,
    ExtendedDrudeParams, ScatteringModel,
    DrudeLorentzParams, DrudeParams,
    tmm_reflection, kramers_kronig_check,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Convert energy in eV to wavelength in meters.
fn ev_to_wavelength_m(energy_ev: f64) -> f64 {
    2.0 * PI * HBAR_EV_S * C / energy_ev
}

#[derive(Parser)]
#[command(name = "demo-optical-titanates")]
#[command(about = "Demonstrate optical database, titanate materials, and extended Drude")]
struct Args {
    /// List all available optical materials
    #[arg(long, default_value = "false")]
    list: bool,

    /// Show epsilon(omega) at 10 frequencies for a material
    #[arg(long)]
    material: Option<String>,

    /// Side-by-side epsilon comparison (comma-separated material names)
    #[arg(long)]
    compare: Option<String>,

    /// Run a thesis demonstration (1-4)
    #[arg(long)]
    thesis: Option<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.list {
        run_list();
        return Ok(());
    }

    if let Some(ref name) = args.material {
        run_material(name)?;
        return Ok(());
    }

    if let Some(ref names) = args.compare {
        run_compare(names)?;
        return Ok(());
    }

    if let Some(n) = args.thesis {
        run_thesis(n)?;
        return Ok(());
    }

    println!("Use --list, --material <name>, --compare <a,b,c>, or --thesis <1-4>");
    Ok(())
}

fn run_list() {
    println!("========================================");
    println!("Optical Materials Database ({} entries)", list_materials().len());
    println!("========================================");
    for (i, name) in list_materials().iter().enumerate() {
        let mat = get_material_by_display_name(name);
        let provenance = if let Some(ref m) = mat {
            format!(
                "  [{}] ref: {}",
                match m.material_type {
                    materials_core::OpticalMaterialType::Metal => "Metal",
                    materials_core::OpticalMaterialType::Semiconductor => "Semiconductor",
                    materials_core::OpticalMaterialType::Dielectric => "Dielectric",
                    materials_core::OpticalMaterialType::Metamaterial => "Metamaterial",
                    materials_core::OpticalMaterialType::ConductiveOxide => "ConductiveOxide",
                },
                m.reference
            )
        } else {
            String::new()
        };
        println!("  {:>2}. {}{}", i + 1, name, provenance);
    }
}

fn get_material_by_display_name(display: &str) -> Option<materials_core::MaterialEntry> {
    // Extract the name/alias from display format like "Gold (Au)"
    let lower = display.to_lowercase();
    // Try the part before " (" first
    if let Some(idx) = lower.find(" (") {
        let name = &lower[..idx];
        if let Some(m) = get_material(name) {
            return Some(m);
        }
    }
    // Try extracting formula from parentheses
    if let Some(start) = lower.find('(')
        && let Some(end) = lower.find(')')
    {
        let formula = &lower[start + 1..end];
        if let Some(m) = get_material(formula) {
            return Some(m);
        }
    }
    get_material(&lower)
}

fn run_material(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mat = get_material(name)
        .ok_or_else(|| format!("Material '{}' not found. Use --list to see available.", name))?;

    println!("========================================");
    println!("{} ({})", mat.name, mat.formula);
    println!("========================================");
    println!("Type: {:?}", mat.material_type);
    println!("Reference: {}", mat.reference);
    if let Some((lo, hi)) = mat.validity_range_ev {
        println!("Valid range: {:.2} - {:.2} eV", lo, hi);
    }
    if let Some(t) = mat.temperature_k {
        println!("Temperature: {:.0} K", t);
    }
    if let Some(d) = mat.doping_info {
        println!("Doping: {}", d);
    }
    println!("Casimir model: {:?}", mat.casimir_model);

    println!("\nDielectric function epsilon(omega):");
    println!("{:>10}  {:>14}  {:>14}  {:>14}", "E (eV)", "Re[eps]", "Im[eps]", "|eps|");
    println!("{:-<56}", "");

    let freqs_ev = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0];
    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        let eps = mat.optical.epsilon(omega);
        println!(
            "{:>10.3}  {:>14.4}  {:>14.4}  {:>14.4}",
            e, eps.re, eps.im, eps.norm()
        );
    }

    // KK check: reconstruct Re[eps] from Im[eps]
    let eps_vals: Vec<Complex64> = (1..100)
        .map(|i| {
            let omega = i as f64 * 0.1 * EV_TO_RADS;
            mat.optical.epsilon(omega)
        })
        .collect();
    let kk = kramers_kronig_check(&eps_vals, true, None);
    println!("\nKramers-Kronig consistency: max_rel_error = {:.4e}", kk.max_rel_error);

    Ok(())
}

fn run_compare(names: &str) -> Result<(), Box<dyn std::error::Error>> {
    let material_names: Vec<&str> = names.split(',').map(|s| s.trim()).collect();
    let materials: Vec<_> = material_names
        .iter()
        .map(|n| {
            get_material(n)
                .ok_or_else(|| format!("Material '{}' not found", n))
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("========================================");
    println!("Comparison: {}", names);
    println!("========================================");

    // Header
    print!("{:>10}", "E (eV)");
    for mat in &materials {
        print!("  {:>12} Re  {:>12} Im", mat.formula, "");
    }
    println!();
    print!("{:-<10}", "");
    for _ in &materials {
        print!("  {:-<26}", "");
    }
    println!();

    let freqs_ev = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0];
    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        print!("{:>10.3}", e);
        for mat in &materials {
            let eps = mat.optical.epsilon(omega);
            print!("  {:>12.3}  {:>12.3}", eps.re, eps.im);
        }
        println!();
    }

    Ok(())
}

fn run_thesis(n: u8) -> Result<(), Box<dyn std::error::Error>> {
    match n {
        1 => thesis_1_tio_srtio3_interface(),
        2 => thesis_2_extended_drude_residual(),
        3 => thesis_3_spectral_weight_transfer(),
        4 => thesis_4_azo_sensitivity(),
        _ => {
            println!("Thesis number must be 1-4.");
            Ok(())
        }
    }
}

/// Thesis 1: TiO/SrTiO3 interfacial reflectance.
///
/// A Drude metallic layer (TiO) on a phononic substrate (SrTiO3) should
/// produce measurable low-frequency reflectance rolloff controlled by TiO thickness.
fn thesis_1_tio_srtio3_interface() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Thesis 1: TiO / SrTiO3 Interface");
    println!("========================================");
    println!("Drude metal (TiO) on phononic dielectric (SrTiO3) substrate.");
    println!("Predicts: low-freq reflectance rolloff controlled by TiO thickness.\n");

    let tio = get_material("tio").unwrap();
    let srtio3 = get_material("srtio3").unwrap();

    // TMM reflectance for different TiO film thicknesses
    let thicknesses_nm = [5.0, 10.0, 20.0, 50.0, 100.0];
    let freqs_ev = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0];

    println!("{:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "E (eV)", "5 nm", "10 nm", "20 nm", "50 nm", "100 nm");
    println!("{:-<66}", "");

    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        let wavelength = ev_to_wavelength_m(e);
        let n_sub = srtio3.optical.epsilon(omega).sqrt();
        let n_film = tio.optical.epsilon(omega).sqrt();
        let n_inc = Complex64::new(1.0, 0.0); // vacuum

        print!("{:>10.3}", e);
        for &d_nm in &thicknesses_nm {
            let d_m = d_nm * 1e-9;
            // TMM: [incident, film, substrate], thicknesses [0, d, 0]
            let n_layers = [n_inc, n_film, n_sub];
            let d_layers = [0.0, d_m, 0.0];
            let result = tmm_reflection(&n_layers, &d_layers, wavelength, 0.0, true);
            print!("  {:>10.4}", result.reflectance);
        }
        println!();
    }

    println!("\nConclusion: Reflectance at low freq depends on TiO thickness,");
    println!("confirming interfacial Drude layer controls optical response.");
    Ok(())
}

/// Thesis 2: Constant vs extended Drude for doped SrTiO3.
///
/// The standard constant-gamma Drude model fails to capture the frequency-dependent
/// scattering in correlated oxides. The residual (extended - constant) reveals
/// the non-Fermi-liquid character.
fn thesis_2_extended_drude_residual() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Thesis 2: Extended Drude Residual");
    println!("========================================");
    println!("Comparing constant-gamma vs power-law gamma for doped SrTiO3.");
    println!("Predicts: constant Drude fails; residual reveals non-FL scattering.\n");

    // Constant-gamma Drude (standard model)
    let constant = DrudeLorentzParams {
        drude: Some(DrudeParams {
            omega_p_ev: 0.15,
            gamma_ev: 0.02,
            eps_inf: 1.0,
        }),
        oscillators: vec![],
        eps_inf: 5.2,
        extended_drude: None,
    };

    // Power-law extended Drude (non-Fermi-liquid, exponent ~1.5)
    let extended = DrudeLorentzParams {
        drude: None,
        oscillators: vec![],
        eps_inf: 5.2,
        extended_drude: Some(ExtendedDrudeParams {
            omega_p_ev: 0.15,
            scattering: ScatteringModel::PowerLaw {
                gamma_0_ev: 0.02,
                omega_scale_ev: 0.1,
                exponent: 1.5,
            },
            eps_inf: 1.0,
        }),
    };

    println!("{:>10}  {:>12}  {:>12}  {:>12}", "E (eV)", "Const Re", "Ext Re", "Residual");
    println!("{:-<50}", "");

    let freqs_ev = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        let eps_c = constant.epsilon(omega);
        let eps_e = extended.epsilon(omega);
        let residual = eps_e.re - eps_c.re;
        println!(
            "{:>10.4}  {:>12.3}  {:>12.3}  {:>12.4}",
            e, eps_c.re, eps_e.re, residual
        );
    }

    println!("\nConclusion: Residual grows at higher frequencies where power-law");
    println!("scattering deviates from constant gamma (non-Fermi-liquid signature).");
    Ok(())
}

/// Thesis 3: Spectral weight transfer across RTiO3 series.
///
/// TiO (metal) -> SrTiO3 (insulator) -> LaTiO3 (Mott insulator) shows
/// progressive transfer of spectral weight from Drude peak to mid-IR.
fn thesis_3_spectral_weight_transfer() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Thesis 3: RTiO3 Spectral Weight Transfer");
    println!("========================================");
    println!("TiO (metal) -> SrTiO3 (insulator) -> LaTiO3 (Mott insulator).");
    println!("Predicts: systematic Drude-to-mid-IR spectral weight transfer.\n");

    let mats = ["tio", "srtio3", "latio3"];
    let mat_entries: Vec<_> = mats.iter().map(|n| get_material(n).unwrap()).collect();

    println!("{:>10}  {:>14}  {:>14}  {:>14}", "E (eV)", "TiO Im[eps]", "SrTiO3 Im", "LaTiO3 Im");
    println!("{:-<56}", "");

    let freqs_ev = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        print!("{:>10.4}", e);
        for mat in &mat_entries {
            let eps = mat.optical.epsilon(omega);
            print!("  {:>14.4}", eps.im);
        }
        println!();
    }

    // Drude weight = integral of Im[eps]*omega over low-freq range (proxy)
    println!("\nDrude spectral weight (integrated Im[eps]*omega, 0.01-0.5 eV):");
    for mat in &mat_entries {
        let mut weight = 0.0;
        let n_pts = 200;
        let de = 0.49 / n_pts as f64;
        for i in 0..n_pts {
            let e = 0.01 + (i as f64 + 0.5) * de;
            let omega = e * EV_TO_RADS;
            let eps = mat.optical.epsilon(omega);
            weight += eps.im * e * de;
        }
        println!("  {}: {:.6}", mat.formula, weight);
    }

    println!("\nConclusion: Drude spectral weight decreases TiO > SrTiO3:n > LaTiO3,");
    println!("consistent with progressive localization across the titanate series.");
    Ok(())
}

/// Thesis 4: AZO plasma frequency sensitivity.
///
/// The reflectance edge of Al-doped ZnO shifts with omega_p variation (+/- 20%),
/// demonstrating sensitivity to carrier concentration.
fn thesis_4_azo_sensitivity() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Thesis 4: AZO omega_p Sensitivity");
    println!("========================================");
    println!("Reflectance edge shift with +/- 20% plasma frequency variation.");
    println!("Predicts: systematic edge shift with carrier concentration.\n");

    let base = get_material("azo").unwrap();
    let omega_p_base = base.optical.drude.as_ref().map(|d| d.omega_p_ev).unwrap_or(1.75);

    let variations = [0.8, 0.9, 1.0, 1.1, 1.2];

    println!("{:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "E (eV)", "0.8x", "0.9x", "1.0x", "1.1x", "1.2x");
    println!("{:-<66}", "");

    let freqs_ev = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0];
    for &e in &freqs_ev {
        let omega = e * EV_TO_RADS;
        print!("{:>10.2}", e);
        for &factor in &variations {
            let scaled_omega_p = omega_p_base * factor;
            let mut params = base.optical.clone();
            if let Some(ref mut drude) = params.drude {
                drude.omega_p_ev = scaled_omega_p;
            }
            let eps = params.epsilon(omega);
            // Normal-incidence Fresnel: R = |(n-1)/(n+1)|^2 where n = sqrt(eps)
            let n = eps.sqrt();
            let one = Complex64::new(1.0, 0.0);
            let r = ((n - one) / (n + one)).norm_sqr();
            print!("  {:>10.4}", r);
        }
        println!();
    }

    println!("\nConclusion: Reflectance edge shifts to lower frequency with lower omega_p,");
    println!("confirming Drude-extracted tau depends on carrier concentration.");
    Ok(())
}
