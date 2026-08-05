//! Emit deterministic measurements of the legacy SFWM oracle boundaries.
//!
//! The binary calls the pre-existing public API only to characterize the
//! quantities that the source-intake ledger quarantines. It does not implement
//! a corrected SFWM model or run a thickness reproduction sweep.

use materials_core::fused_silica_sellmeier;
use optics_core::{
    SfwmMaterialParams, SubstrateSfwmParams, rate_ratio_with_dk, sfwm_dominance_check,
    substrate_sfwm_contribution,
};
use std::f64::consts::PI;

fn main() {
    let paper_params = SfwmMaterialParams::linbo3_paper_calibrated();
    let paper_mismatches = paper_params.paper_wavevector_mismatches();
    let legacy_rate = rate_ratio_with_dk(&paper_params, &paper_mismatches);
    let legacy_dominance = sfwm_dominance_check(10.0);

    let source_chi2 = 2.5e-11_f64;
    let legacy_chi2 = 27.0e-12_f64;
    let source_chi3 = 1.5e-20_f64;
    let legacy_chi3 = 2.4e-21_f64;
    let legacy_to_source_ratio_factor =
        (legacy_chi2 / source_chi2).powi(4) * (source_chi3 / legacy_chi3).powi(2);
    let eq6_prefactor =
        2.0 * PI / (paper_params.n_sh * (paper_params.lambda_pump / 2.0) * paper_mismatches.dk_shg);
    let source_input_and_eq6_scale_estimate =
        legacy_rate.rate_ratio_cas_to_dir / legacy_to_source_ratio_factor * eq6_prefactor.powi(2);

    let fused_silica = fused_silica_sellmeier();
    let substrate = substrate_sfwm_contribution(&SubstrateSfwmParams {
        n_pump: fused_silica.refractive_index(1.030),
        n_signal: fused_silica.refractive_index(0.770),
        n_idler: fused_silica.refractive_index(1.550),
        lambda_pump: 1.030,
        lambda_signal: 0.770,
        lambda_idler: 1.550,
        chi3: 3.0e-22,
        thickness_um: 500.0,
    });

    println!("metric\tvalue");
    println!("paper_dk_sfwm_1_per_um\t{:.17e}", paper_mismatches.dk_sfwm);
    println!("paper_dk_shg_1_per_um\t{:.17e}", paper_mismatches.dk_shg);
    println!("paper_dk_spdc_1_per_um\t{:.17e}", paper_mismatches.dk_spdc);
    println!(
        "signed_identity_defect_1_per_um\t{:.17e}",
        paper_mismatches.dk_sfwm - paper_mismatches.dk_shg - paper_mismatches.dk_spdc
    );
    println!(
        "legacy_rate_ratio_cas_to_dir_at_10_um\t{:.17e}",
        legacy_rate.rate_ratio_cas_to_dir
    );
    println!(
        "legacy_dominance_ratio_at_10_um\t{:.17e}",
        legacy_dominance.dominance_ratio
    );
    println!(
        "legacy_to_source_rate_ratio_factor_from_susceptibilities\t{:.17e}",
        legacy_to_source_ratio_factor
    );
    println!(
        "eq6_prefactor_omitted_by_legacy_path\t{:.17e}",
        eq6_prefactor
    );
    println!(
        "source_input_and_eq6_scale_estimate\t{:.17e}",
        source_input_and_eq6_scale_estimate
    );
    println!(
        "substrate_standalone_rate_proxy\t{:.17e}",
        substrate.rate_proxy
    );
    println!("substrate_proxy_is_total_fraction\tfalse");
    println!("corrected_final_sweep_run\tfalse");
}
