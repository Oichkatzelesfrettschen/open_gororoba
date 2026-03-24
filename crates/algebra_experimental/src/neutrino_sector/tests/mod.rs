mod pmns;
mod lift;
mod intertwiner;
mod cp;
mod scan;
mod regression;
mod spectrum;
mod branch_transport;

// ---------------------------------------------------------------------------
// Shared test helpers (accessed by submodules as `super::*`)
// ---------------------------------------------------------------------------

/// Construct the three sedenion subalgebras and a 16D sign table.
///
/// Callers that need the `subs` reference array follow with:
///   `let subs = [&o1, &o2, &o3];`
/// (Rust ownership prevents returning a reference slice to internal data.)
#[allow(dead_code)]
pub(super) fn psi_setup() -> (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    crate::bell_inequality::SignTableCache,
) {
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    let (o1, o2, o3) = get_sedenion_subalgebras();
    let sign_table = crate::bell_inequality::SignTableCache::new(16);
    (o1, o2, o3, sign_table)
}

/// Compute the 3-angle PMNS chi-squared against PDG 2024 central values.
///
/// Uses the canonical sum: ((t_i - mu_i) / sigma_i)^2 for i in {12, 13, 23}.
/// Replaces the repeated 3-line inline pattern at every scan/cp hot loop.
pub(super) fn pdg_score(t12: f64, t13: f64, t23: f64, pdg: &super::Pdg2024) -> f64 {
    ((t12 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
        + ((t13 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
        + ((t23 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2)
}

/// Print theta_12 / theta_13 / theta_23 vs supplied PDG central values with
/// percentage errors.  Format matches the rest of the V_6 scan output.
pub(super) fn print_best_angles(
    angles: (f64, f64, f64),
    pdg_t12: f64,
    pdg_t13: f64,
    pdg_t23: f64,
) {
    println!(
        "  theta_12 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        angles.0, pdg_t12, ((angles.0 - pdg_t12) / pdg_t12 * 100.0).abs()
    );
    println!(
        "  theta_13 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        angles.1, pdg_t13, ((angles.1 - pdg_t13) / pdg_t13 * 100.0).abs()
    );
    println!(
        "  theta_23 = {:.4} deg (PDG: {:.2}, error: {:.2}%)",
        angles.2, pdg_t23, ((angles.2 - pdg_t23) / pdg_t23 * 100.0).abs()
    );
}
