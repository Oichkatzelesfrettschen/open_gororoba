//! gororoba_py: Python bindings for gororoba physics simulation library.
//!
//! This crate provides PyO3 bindings exposing:
//! - Cayley-Dickson algebras (sedenions, pathions, zero-divisors)
//! - E8 lattice and box-kite structures
//! - GRIN ray tracing
//! - Kerr geodesics
//! - Statistics utilities
//! - Quantum algorithms (Grover, MPS, PEPS, hypothesis search)
//! - Quantum hardware profiles

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, IntoPyArray};

// Re-export core crates
use algebra_core::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator_norm,
    find_zero_divisors, generate_e8_roots, e8_cartan_matrix,
    e8_weyl_group_order, find_box_kites, analyze_box_kite_symmetry,
};
use gr_core::{
    kerr_metric_quantities, photon_orbit_radius, impact_parameters,
    shadow_boundary, trace_null_geodesic,
};
use cosmology_core::bounce::{simulate_bounce, BounceParams, chi2_distance_modulus, luminosity_distance};
use lbm_core::simulate_poiseuille;
use spectral_core::fractional_laplacian_periodic_1d;
use stats_core::frechet_distance;
use materials_core::{
    build_absorber_stack, canonical_sedenion_zd_pairs,
};
use quantum_core::{
    grover_search_indices, optimal_iterations, GroverConfig,
    ThresholdOracle, QuantumHypothesisSearch,
    MatrixProductState,
    NeutralAtomProfile, SuperconductingProfile, TrappedIonProfile,
    HardwareProfile,
};

// ============================================================================
// Algebra Module
// ============================================================================

/// Multiply two Cayley-Dickson numbers.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn py_cd_multiply(py: Python<'_>, a: Vec<f64>, b: Vec<f64>) -> PyResult<Py<PyArray1<f64>>> {
    if a.len() != b.len() || !a.len().is_power_of_two() {
        return Err(PyValueError::new_err("Arrays must have same power-of-two length"));
    }
    let result = cd_multiply(&a, &b);
    Ok(result.into_pyarray(py).to_owned().into())
}

/// Conjugate a Cayley-Dickson number.
#[pyfunction]
fn py_cd_conjugate(py: Python<'_>, a: Vec<f64>) -> PyResult<Py<PyArray1<f64>>> {
    if !a.len().is_power_of_two() {
        return Err(PyValueError::new_err("Array length must be power of two"));
    }
    let result = cd_conjugate(&a);
    Ok(result.into_pyarray(py).to_owned().into())
}

/// Compute the norm of a Cayley-Dickson number.
#[pyfunction]
fn py_cd_norm(a: Vec<f64>) -> PyResult<f64> {
    if !a.len().is_power_of_two() {
        return Err(PyValueError::new_err("Array length must be power of two"));
    }
    Ok(cd_norm_sq(&a).sqrt())
}

/// Compute the associator norm ||(ab)c - a(bc)||.
#[pyfunction]
fn py_cd_associator_norm(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() || b.len() != c.len() || !a.len().is_power_of_two() {
        return Err(PyValueError::new_err("Arrays must have same power-of-two length"));
    }
    Ok(cd_associator_norm(&a, &b, &c))
}

/// Find zero-divisor pairs in a Cayley-Dickson algebra.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (dim, atol = 1e-10))]
fn py_find_zero_divisors(dim: usize, atol: f64) -> PyResult<Vec<(usize, usize, usize, usize, f64)>> {
    if !dim.is_power_of_two() || dim < 16 {
        return Err(PyValueError::new_err("Dimension must be power of two >= 16"));
    }
    Ok(find_zero_divisors(dim, atol))
}

/// Generate the 240 roots of E8.
#[pyfunction]
fn py_generate_e8_roots() -> Vec<Vec<f64>> {
    let roots = generate_e8_roots();
    roots.iter().map(|r| r.coords.to_vec()).collect()
}

/// Get the E8 Cartan matrix.
#[pyfunction]
fn py_e8_cartan_matrix() -> Vec<Vec<i32>> {
    let matrix = e8_cartan_matrix();
    matrix.iter().map(|row| row.to_vec()).collect()
}

/// Get the Weyl group order of E8.
#[pyfunction]
fn py_weyl_group_order() -> u64 {
    e8_weyl_group_order()
}

/// Find box-kite structures in a Cayley-Dickson algebra.
#[pyfunction]
#[pyo3(signature = (dim, atol = 1e-10))]
fn py_find_box_kites(dim: usize, atol: f64) -> PyResult<usize> {
    if !dim.is_power_of_two() || dim < 16 {
        return Err(PyValueError::new_err("Dimension must be power of two >= 16"));
    }
    let box_kites = find_box_kites(dim, atol);
    Ok(box_kites.len())
}

/// Analyze box-kite symmetry structure.
#[pyfunction]
#[pyo3(signature = (dim, atol = 1e-10))]
fn py_analyze_box_kite_symmetry(dim: usize, atol: f64) -> PyResult<(usize, usize)> {
    if !dim.is_power_of_two() || dim < 16 {
        return Err(PyValueError::new_err("Dimension must be power of two >= 16"));
    }
    let result = analyze_box_kite_symmetry(dim, atol);
    Ok((result.n_boxkites, result.n_assessors))
}

// ============================================================================
// GR Module
// ============================================================================

/// Compute Kerr metric quantities (sigma, delta).
#[pyfunction]
fn py_kerr_metric(r: f64, theta: f64, a: f64) -> (f64, f64) {
    kerr_metric_quantities(r, theta, a)
}

/// Compute photon orbit radii (prograde, retrograde).
#[pyfunction]
fn py_photon_orbit_radius(a: f64) -> (f64, f64) {
    photon_orbit_radius(a)
}

/// Compute impact parameters at a given photon orbit radius.
#[pyfunction]
fn py_impact_parameters(r_ph: f64, a: f64) -> (f64, f64) {
    impact_parameters(r_ph, a)
}

/// Compute Kerr black hole shadow boundary.
#[pyfunction]
#[pyo3(signature = (a, n_points = 500, theta_o = std::f64::consts::FRAC_PI_2))]
fn py_shadow_boundary(py: Python<'_>, a: f64, n_points: usize, theta_o: f64) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let (alpha, beta) = shadow_boundary(a, n_points, theta_o);
    (alpha.into_pyarray(py).to_owned().into(), beta.into_pyarray(py).to_owned().into())
}

/// Trace a null geodesic in Kerr spacetime.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (a, e, l, q, r0, theta0, lam_max = 100.0, sgn_r = -1.0, sgn_theta = 1.0, n_steps = 2000))]
fn py_trace_geodesic(
    py: Python<'_>,
    a: f64,
    e: f64,
    l: f64,
    q: f64,
    r0: f64,
    theta0: f64,
    lam_max: f64,
    sgn_r: f64,
    sgn_theta: f64,
    n_steps: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let result = trace_null_geodesic(a, e, l, q, r0, theta0, lam_max, sgn_r, sgn_theta, n_steps);
    (
        result.t.into_pyarray(py).to_owned().into(),
        result.r.into_pyarray(py).to_owned().into(),
        result.theta.into_pyarray(py).to_owned().into(),
        result.phi.into_pyarray(py).to_owned().into(),
        result.lam.into_pyarray(py).to_owned().into(),
    )
}

// ============================================================================
// Statistics Module
// ============================================================================

/// Compute discrete Frechet distance between two curves.
#[pyfunction]
fn py_frechet_distance(p: Vec<f64>, q: Vec<f64>) -> f64 {
    frechet_distance(&p, &q)
}

// ============================================================================
// Materials Module
// ============================================================================

/// Build absorber stack from zero-divisor pairs.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (max_layers = 10, base_n = 1.5))]
fn py_build_absorber_stack(max_layers: usize, base_n: f64) -> Vec<(usize, usize, usize, usize, f64, f64, f64, f64)> {
    let zd_pairs = canonical_sedenion_zd_pairs();
    let stack = build_absorber_stack(&zd_pairs, max_layers, base_n);

    stack.iter().map(|m| {
        (
            m.zd_indices.0,
            m.zd_indices.1,
            m.zd_indices.2,
            m.zd_indices.3,
            m.layer.n_real,
            m.layer.n_imag,
            m.layer.thickness_nm,
            m.product_norm,
        )
    }).collect()
}

// ============================================================================
// Cosmology Module
// ============================================================================

/// Simulate bounce cosmology evolution.
/// Returns (time, a, h, q) arrays.
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (a_init, t_end, n_steps, omega_m = 0.315, omega_l = 0.685, q_corr = 1e-6))]
fn py_simulate_bounce(
    py: Python<'_>,
    a_init: f64,
    t_end: f64,
    n_steps: usize,
    omega_m: f64,
    omega_l: f64,
    q_corr: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let params = BounceParams {
        omega_m,
        omega_l,
        q_corr,
    };
    let result = simulate_bounce(&params, t_end, n_steps, a_init);
    (
        result.time.into_pyarray(py).to_owned().into(),
        result.a.into_pyarray(py).to_owned().into(),
        result.h.into_pyarray(py).to_owned().into(),
        result.q.into_pyarray(py).to_owned().into(),
    )
}

/// Compute chi-squared for distance modulus fit.
#[pyfunction]
#[pyo3(signature = (z_obs, mu_obs, sigma_obs, omega_m, h0, q_corr))]
fn py_chi2_distance_modulus(
    z_obs: Vec<f64>,
    mu_obs: Vec<f64>,
    sigma_obs: Vec<f64>,
    omega_m: f64,
    h0: f64,
    q_corr: f64,
) -> f64 {
    chi2_distance_modulus(&z_obs, &mu_obs, &sigma_obs, omega_m, h0, q_corr)
}

/// Compute luminosity distance for a single redshift.
#[pyfunction]
#[pyo3(signature = (z, omega_m, h0, q_corr))]
fn py_luminosity_distance(z: f64, omega_m: f64, h0: f64, q_corr: f64) -> f64 {
    luminosity_distance(z, omega_m, h0, q_corr)
}

// ============================================================================
// LBM Module
// ============================================================================

/// Simulate Poiseuille flow between parallel walls.
/// Returns (y, ux_numerical, ux_analytical, max_rel_error).
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (nx, ny, tau, fx, n_steps))]
fn py_simulate_poiseuille(
    py: Python<'_>,
    nx: usize,
    ny: usize,
    tau: f64,
    fx: f64,
    n_steps: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64) {
    let result = simulate_poiseuille(nx, ny, tau, fx, n_steps);
    (
        result.y.into_pyarray(py).to_owned().into(),
        result.ux_numerical.into_pyarray(py).to_owned().into(),
        result.ux_analytical.into_pyarray(py).to_owned().into(),
        result.max_rel_error,
    )
}

// ============================================================================
// Spectral Module
// ============================================================================

/// Compute 1D fractional Laplacian using spectral method (periodic BC).
#[pyfunction]
#[pyo3(signature = (u, s, length))]
fn py_fractional_laplacian_1d(
    py: Python<'_>,
    u: Vec<f64>,
    s: f64,
    length: f64,
) -> Py<PyArray1<f64>> {
    let result = fractional_laplacian_periodic_1d(&u, s, length);
    result.into_pyarray(py).to_owned().into()
}

// ============================================================================
// Quantum Module
// ============================================================================

/// Run Grover's quantum search algorithm.
/// Returns (iterations, success_probability, top_candidates).
#[pyfunction]
#[pyo3(signature = (n_qubits, marked_indices, max_iterations = None, top_k = 5))]
fn py_grover_search(
    n_qubits: usize,
    marked_indices: Vec<usize>,
    max_iterations: Option<usize>,
    top_k: usize,
) -> PyResult<(usize, f64, Vec<usize>)> {
    if n_qubits == 0 || n_qubits > 20 {
        return Err(PyValueError::new_err("n_qubits must be in range 1..=20"));
    }
    let config = GroverConfig {
        iterations: max_iterations,
        top_k,
    };
    let result = grover_search_indices(n_qubits, &marked_indices, config);
    Ok((result.iterations, result.success_probability, result.top_candidates))
}

/// Compute optimal Grover iterations for given search space and marked count.
#[pyfunction]
fn py_optimal_grover_iterations(n_states: usize, n_marked: usize) -> usize {
    optimal_iterations(n_states, n_marked)
}

/// Run quantum hypothesis search over a parameter grid.
/// Returns (verified_solutions, speedup_factor, oracle_calls).
#[pyfunction]
#[pyo3(signature = (ranges, threshold, score_fn_name = "sum"))]
fn py_quantum_grid_search(
    ranges: Vec<(f64, f64, usize)>,
    threshold: f64,
    score_fn_name: &str,
) -> PyResult<(Vec<Vec<f64>>, f64, usize)> {
    // Create search space
    let mut search = QuantumHypothesisSearch::from_grid(ranges);

    // Create oracle based on score function name
    let result = match score_fn_name {
        "sum" => {
            let oracle = ThresholdOracle::new(|p: &[f64]| p.iter().sum::<f64>(), threshold);
            search.mark_with_oracle(&oracle);
            search.search(GroverConfig::default())
        }
        "norm" => {
            let oracle = ThresholdOracle::new(
                |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>().sqrt(),
                threshold,
            );
            search.mark_with_oracle(&oracle);
            search.search(GroverConfig::default())
        }
        "max" => {
            let oracle = ThresholdOracle::new(
                |p: &[f64]| p.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                threshold,
            );
            search.mark_with_oracle(&oracle);
            search.search(GroverConfig::default())
        }
        "min" => {
            let oracle = ThresholdOracle::new(
                |p: &[f64]| p.iter().cloned().fold(f64::INFINITY, f64::min),
                threshold,
            );
            search.mark_with_oracle(&oracle);
            search.search(GroverConfig::default())
        }
        _ => return Err(PyValueError::new_err("score_fn must be 'sum', 'norm', 'max', or 'min'")),
    };

    let solutions: Vec<Vec<f64>> = result
        .verified_solutions
        .iter()
        .map(|h| h.parameters.clone())
        .collect();

    Ok((solutions, result.speedup_factor, result.oracle_calls))
}

/// Create an MPS (Matrix Product State) in product state |0...0>.
/// Returns the MPS data as a list of tensor shapes (chi_left, physical_dim, chi_right) for verification.
#[pyfunction]
fn py_mps_zero_state(n_qubits: usize) -> Vec<(usize, usize, usize)> {
    let mps = MatrixProductState::new_zero_state(n_qubits);
    mps.tensors
        .iter()
        .map(|t| (t.chi_left, t.physical_dim, t.chi_right))
        .collect()
}

/// Get neutral atom hardware profile.
/// Returns (n_qubits, t1_us, t2_us, single_qubit_error, two_qubit_error).
#[pyfunction]
fn py_neutral_atom_profile(n_qubits: usize) -> (usize, f64, f64, f64, f64) {
    let profile = NeutralAtomProfile::new(n_qubits);
    let coherence = profile.coherence_times();
    let errors = profile.error_rates();
    (n_qubits, coherence.t1_us, coherence.t2_us, errors.single_qubit, errors.two_qubit)
}

/// Get superconducting hardware profile (IBM Eagle topology).
/// Returns (n_qubits, vendor, t1_us, t2_us, single_qubit_error, two_qubit_error).
#[pyfunction]
fn py_superconducting_ibm_profile(n_qubits: usize) -> (usize, String, f64, f64, f64, f64) {
    let profile = SuperconductingProfile::ibm(n_qubits);
    let coherence = profile.coherence_times();
    let errors = profile.error_rates();
    (
        n_qubits,
        "ibm".to_string(),
        coherence.t1_us,
        coherence.t2_us,
        errors.single_qubit,
        errors.two_qubit,
    )
}

/// Get superconducting hardware profile (Google Sycamore 2D grid).
/// Returns (rows, cols, vendor, t1_us, t2_us, single_qubit_error, two_qubit_error).
#[pyfunction]
fn py_superconducting_google_profile(rows: usize, cols: usize) -> (usize, usize, String, f64, f64, f64, f64) {
    let profile = SuperconductingProfile::google(rows, cols);
    let coherence = profile.coherence_times();
    let errors = profile.error_rates();
    (
        rows,
        cols,
        "google".to_string(),
        coherence.t1_us,
        coherence.t2_us,
        errors.single_qubit,
        errors.two_qubit,
    )
}

/// Get trapped ion hardware profile (all-to-all connectivity).
/// Returns (n_qubits, t1_us, t2_us, single_qubit_error, two_qubit_error).
#[pyfunction]
fn py_trapped_ion_profile(n_qubits: usize) -> (usize, f64, f64, f64, f64) {
    let profile = TrappedIonProfile::new(n_qubits);
    let coherence = profile.coherence_times();
    let errors = profile.error_rates();
    (
        n_qubits,
        coherence.t1_us,
        coherence.t2_us,
        errors.single_qubit,
        errors.two_qubit,
    )
}

// ============================================================================
// Module Registration
// ============================================================================

/// Python module definition.
#[pymodule]
fn gororoba_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Algebra functions
    m.add_function(wrap_pyfunction!(py_cd_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(py_cd_conjugate, m)?)?;
    m.add_function(wrap_pyfunction!(py_cd_norm, m)?)?;
    m.add_function(wrap_pyfunction!(py_cd_associator_norm, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_zero_divisors, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_e8_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_e8_cartan_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_weyl_group_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_box_kites, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_box_kite_symmetry, m)?)?;

    // GR functions
    m.add_function(wrap_pyfunction!(py_kerr_metric, m)?)?;
    m.add_function(wrap_pyfunction!(py_photon_orbit_radius, m)?)?;
    m.add_function(wrap_pyfunction!(py_impact_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(py_shadow_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(py_trace_geodesic, m)?)?;

    // Statistics functions
    m.add_function(wrap_pyfunction!(py_frechet_distance, m)?)?;

    // Materials functions
    m.add_function(wrap_pyfunction!(py_build_absorber_stack, m)?)?;

    // Cosmology functions
    m.add_function(wrap_pyfunction!(py_simulate_bounce, m)?)?;
    m.add_function(wrap_pyfunction!(py_chi2_distance_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(py_luminosity_distance, m)?)?;

    // LBM functions
    m.add_function(wrap_pyfunction!(py_simulate_poiseuille, m)?)?;

    // Spectral functions
    m.add_function(wrap_pyfunction!(py_fractional_laplacian_1d, m)?)?;

    // Quantum functions
    m.add_function(wrap_pyfunction!(py_grover_search, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimal_grover_iterations, m)?)?;
    m.add_function(wrap_pyfunction!(py_quantum_grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(py_mps_zero_state, m)?)?;
    m.add_function(wrap_pyfunction!(py_neutral_atom_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_superconducting_ibm_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_superconducting_google_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_trapped_ion_profile, m)?)?;

    Ok(())
}
