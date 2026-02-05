//! gororoba_py: Python bindings for gororoba physics simulation library.
//!
//! This crate provides PyO3 bindings exposing:
//! - Cayley-Dickson algebras (sedenions, pathions, zero-divisors)
//! - E8 lattice and box-kite structures
//! - GRIN ray tracing
//! - Kerr geodesics
//! - Statistics utilities

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
use stats_core::frechet_distance;
use materials_core::{
    build_absorber_stack, canonical_sedenion_zd_pairs,
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
    Ok((result.n_boxkites, result.n_sails))
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

    Ok(())
}
