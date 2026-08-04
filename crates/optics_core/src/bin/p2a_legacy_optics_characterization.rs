//! Capture the pre-repair optics observables as characterization data.
//!
//! This binary records the legacy formulas and production outputs before the
//! source-faithful channel repair. Its output is a negative-control artifact,
//! not a scientific golden file.

use num_complex::Complex64;
use optics_core::{
    FanoChannel, FanoDrudeParams, extract_fano_params, fano_cross_sections_normalized,
    fano_reflection, mie_mdm_sweep, ruan_fan_mdm_fig4, ruan_fan_mdm_fig5,
};
use std::{env, fs, path::PathBuf};

fn main() {
    let mut arguments = env::args().skip(1);
    let output = arguments.next().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: p2a-legacy-optics-characterization <output.toml>");
        std::process::exit(2);
    });

    let mut lines = vec![
        "format = \"p2a-legacy-optics-characterization-v1\"".to_string(),
        "scientific_status = \"characterization only; legacy observables are not golden values\"".to_string(),
        "legacy_absorption_formula = \"-Re(S)\"".to_string(),
        "legacy_extinction_formula = \"c_sct + c_abs\"".to_string(),
        "legacy_mie_absorption_formula = \"-Re(S_l) summed over channels\"".to_string(),
        "legacy_parameter_extractor = \"peak, one-sided HWHM, endpoint phase, quarter-range fallback\"".to_string(),
        "".to_string(),
    ];

    let phi_values = [
        ("zero", 0.0),
        ("positive_half_pi", std::f64::consts::PI / 2.0),
        ("pi", std::f64::consts::PI),
    ];
    let ratio_values = [0.0, 0.1, 1.0, 10.0];
    let detuning_values = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];

    for (phi_name, phi) in phi_values {
        for gamma_ratio in ratio_values {
            for x in detuning_values {
                let channel = FanoChannel {
                    omega_0: 1.0,
                    gamma: 1.0,
                    gamma_0: gamma_ratio,
                    phi,
                    l: 0,
                };
                let amplitudes = fano_reflection(&channel, 1.0 + x);
                let scattering = fano_cross_sections_normalized(&channel, 1.0 + x);
                lines.push(format!(
                    "[[fano_row]]\nphi = \"{}\"\ngamma0_over_gamma = {}\nx = {}\nr_re = {}\nr_im = {}\ns_re = {}\ns_im = {}\nc_sct = {}\nc_abs = {}\nc_ext = {}\n",
                    phi_name,
                    gamma_ratio,
                    x,
                    amplitudes.re,
                    amplitudes.im,
                    ((amplitudes - Complex64::new(1.0, 0.0)) / 2.0).re,
                    ((amplitudes - Complex64::new(1.0, 0.0)) / 2.0).im,
                    scattering.c_sct,
                    scattering.c_abs,
                    scattering.c_ext
                ));
            }
        }
    }

    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.0,
    };
    let omegas: Vec<f64> = (0..9)
        .map(|index| 0.14 + 0.03 * index as f64 / 8.0)
        .collect();
    for (geometry_name, geometry) in [
        ("fig4", ruan_fan_mdm_fig4(&drude)),
        ("fig5", ruan_fan_mdm_fig5(&drude)),
    ] {
        lines.push(format!("[[geometry]]\nname = \"{}\"", geometry_name));
        for (index, layer) in geometry.layers.iter().enumerate() {
            lines.push(format!(
                "layer_{}_outer_radius = {}\nlayer_{}_epsilon_re = {}\nlayer_{}_epsilon_im = {}",
                index, layer.outer_radius, index, layer.epsilon.re, index, layer.epsilon.im
            ));
        }
        lines.push(format!(
            "eps_ext_re = {}\neps_ext_im = {}\n",
            geometry.eps_ext.re, geometry.eps_ext.im
        ));

        let results = mie_mdm_sweep(
            &geometry,
            1,
            &drude,
            &omegas,
            if geometry_name == "fig4" { 0 } else { 2 },
        );
        let extracted = extract_fano_params(&omegas, &results, 0);
        if let Some(channel) = extracted {
            lines.push(format!(
                "[[legacy_extracted]]\ngeometry = \"{}\"\nomega_0 = {}\ngamma = {}\ngamma_0 = {}\nphi = {}\n",
                geometry_name, channel.omega_0, channel.gamma, channel.gamma_0, channel.phi
            ));
        }
        for result in results {
            lines.push(format!(
                "[[mie_row]]\ngeometry = \"{}\"\nomega = {}\nc_sct = {}\nc_abs = {}\nc_ext = {}",
                geometry_name,
                result.omega,
                result.cross_sections.c_sct,
                result.cross_sections.c_abs,
                result.cross_sections.c_ext
            ));
            for channel in result.channels {
                lines.push(format!(
                    "l_{}_s_re = {}\nl_{}_s_im = {}\nl_{}_r_re = {}\nl_{}_r_im = {}",
                    channel.l,
                    channel.s_l.re,
                    channel.l,
                    channel.s_l.im,
                    channel.l,
                    channel.r_l.re,
                    channel.l,
                    channel.r_l.im
                ));
            }
        }
    }

    if let Some(parent) = output.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!("failed to create {}: {}", parent.display(), error);
        std::process::exit(1);
    }
    if let Err(error) = fs::write(&output, lines.join("\n")) {
        eprintln!("failed to write {}: {}", output.display(), error);
        std::process::exit(1);
    }
    println!("wrote {}", output.display());
}
