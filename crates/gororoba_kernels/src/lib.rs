//! gororoba_kernels: Rust-accelerated Cayley-Dickson algebra for physics.
//!
//! This crate provides high-performance implementations of:
//! - Cayley-Dickson multiplication for any power-of-2 dimension
//! - Associator computation and batch operations
//! - Zero-divisor search algorithms
//! - Spectrum analysis tools
//! - Clifford algebra Cl(8) for particle physics
//! - Spectral dimension and fractal cosmology
//! - MERA tensor networks
//! - Metamaterial absorber layer mapping
//!
//! When compiled with the `python` feature (default), it also provides
//! PyO3 bindings for use from Python via numpy arrays.

pub mod algebra;
pub mod clifford;
pub mod gravastar;
pub mod holographic;
pub mod mera;
pub mod metamaterial;
pub mod spectral;
pub mod stats;
pub mod tang_mass;
pub mod zd_graphs;

// Re-export core algebra functions for Rust users
pub use algebra::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator, cd_associator_norm,
    batch_associator_norms, batch_associator_norms_sq, batch_associator_norms_parallel,
    left_mult_operator, find_zero_divisors, measure_associator_density,
    zd_spectrum_analysis, count_pathion_zero_divisors,
};

pub use clifford::{
    pauli_matrices, gamma_matrices_cl8, verify_clifford_relation,
    GammaMatrix, CliffordAlgebra,
};

pub use spectral::{
    calcagni_spectral_dimension, cdt_spectral_dimension,
    k_minus_3_spectrum, kolmogorov_spectrum, kraichnan_enstrophy_spectrum,
    parisi_sourlas_effective_dimension, parisi_sourlas_spectrum_exponent,
    analyze_k_minus_3_origin, SpectralAnalysisResult,
};

pub use mera::{
    build_mera_structure, von_neumann_entropy, mera_entropy_estimate,
    fit_log_scaling, mera_entropy_scaling_analysis, bootstrap_slope_ci,
    MeraLayer, MeraScalingResult,
};

pub use metamaterial::{
    map_zd_to_refractive_index, map_zd_norm_to_thickness, classify_material_type,
    map_zd_pair_to_layer, build_absorber_stack, verify_physical_realizability,
    canonical_sedenion_zd_pairs, MetamaterialLayer, ZdToLayerMapping,
    MaterialType, VerificationResult,
};

pub use stats::{
    frechet_distance, normalize_spectrum, frechet_null_test,
    bootstrap_ci, haar_random_unitary, frobenius_distance,
    haar_null_test, pmns_matrix, test_pmns_prediction, fit_power_law_with_ci,
    FrechetNullTestResult, BootstrapCIResult, HaarNullTestResult,
    PmnsComparisonResult, PowerLawFitResult,
};

pub use holographic::{
    bekenstein_bound_bits, verify_bekenstein_bound,
    absorber_channel_capacity, absorber_effective_radius, absorber_energy,
    RTLattice, compute_min_cut, analyze_entropy_scaling, verify_area_law,
    BekensteinBoundResult, MinCutResult, EntropyScalingResult, AreaLawResult,
    AbsorberLayer,
};

pub use gravastar::{
    PolytropicEos, AnisotropicParams, TovState, GravastarSolution,
    GravastarConfig, solve_gravastar, polytropic_stability_sweep,
    anisotropic_stability_test, StabilityResult, AnisotropicStabilityResult,
};

pub use zd_graphs::{
    build_zd_interaction_graph, analyze_zd_graph, analyze_basis_participation,
    build_associator_graph, analyze_associator_graph, zd_shortest_path,
    zd_graph_diameter, ZdGraphAnalysis, BasisParticipationResult,
    AssociatorGraphResult,
};

pub use tang_mass::{
    M_ELECTRON, M_MUON, M_TAU, RATIO_E_MU, RATIO_MU_TAU, RATIO_E_TAU,
    GenerationAssignment, MassRatioPrediction, MassNullTestResult,
    DimensionScalingResult, basis_associator_norm, canonical_sedenion_assignments,
    predict_mass_ratios, find_best_assignment, mass_ratio_null_test,
    dimension_scaling_analysis,
};

// Python bindings (only when python feature is enabled)
#[cfg(feature = "python")]
mod python_bindings {
    use num_complex::Complex64;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
    use pyo3::prelude::*;
    use crate::algebra;
    use crate::spectral;
    use crate::mera;
    use crate::metamaterial;
    use crate::stats;
    use crate::holographic;

    /// A zero-divisor pair: (i, j, k, l, norm) where (e_i + e_j) * (e_k +/- e_l) ~ 0.
    type ZdPair = (usize, usize, usize, usize, f64);

    /// Cayley-Dickson multiplication of two algebra elements.
    #[pyfunction]
    fn cd_multiply<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
        dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;

        if a_slice.len() != dim || b_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Array lengths ({}, {}) != dim ({})",
                a_slice.len(),
                b_slice.len(),
                dim,
            )));
        }
        if !dim.is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2, got {}",
                dim,
            )));
        }

        let result = algebra::cd_multiply(a_slice, b_slice);
        Ok(PyArray1::from_vec(py, result))
    }

    /// Cayley-Dickson conjugation: negate all components except the first.
    #[pyfunction]
    fn cd_conjugate<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a_slice = a.as_slice()?;
        let result = algebra::cd_conjugate(a_slice);
        Ok(PyArray1::from_vec(py, result))
    }

    /// Squared Euclidean norm of a Cayley-Dickson element.
    #[pyfunction]
    fn cd_norm_sq(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        Ok(algebra::cd_norm_sq(a_slice))
    }

    /// Measure non-associativity density for dim-dimensional CD algebra.
    #[pyfunction]
    #[pyo3(signature = (dim, trials=5000, seed=42, atol=1e-8))]
    fn measure_associator_density(
        dim: usize,
        trials: usize,
        seed: u64,
        atol: f64,
    ) -> PyResult<(f64, usize)> {
        if !dim.is_power_of_two() || dim < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2 >= 2, got {}",
                dim,
            )));
        }
        let (density, failures) = algebra::measure_associator_density(dim, trials, seed, atol);
        Ok((density, failures))
    }

    /// Construct the left-multiplication matrix L_a.
    #[pyfunction]
    fn left_mult_operator<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f64>,
        dim: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a_slice = a.as_slice()?;
        if a_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Array length ({}) != dim ({})",
                a_slice.len(),
                dim,
            )));
        }
        if !dim.is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2, got {}",
                dim,
            )));
        }
        let flat = algebra::left_mult_operator(a_slice, dim);
        let arr = PyArray1::from_vec(py, flat);
        let reshaped = arr.reshape([dim, dim])?;
        Ok(reshaped)
    }

    /// Find 2-blade zero-divisor pairs in a dim-dimensional CD algebra.
    #[pyfunction]
    #[pyo3(signature = (dim, atol=1e-10))]
    fn find_zero_divisors(dim: usize, atol: f64) -> PyResult<Vec<ZdPair>> {
        if !dim.is_power_of_two() || dim < 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2 >= 16, got {}",
                dim,
            )));
        }
        Ok(algebra::find_zero_divisors(dim, atol))
    }

    /// Compute associator norm ||A(a,b,c)|| = ||(ab)c - a(bc)||.
    #[pyfunction]
    fn cd_associator_norm(
        a: PyReadonlyArray1<'_, f64>,
        b: PyReadonlyArray1<'_, f64>,
        c: PyReadonlyArray1<'_, f64>,
        dim: usize,
    ) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        let c_slice = c.as_slice()?;

        if a_slice.len() != dim || b_slice.len() != dim || c_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Array lengths ({}, {}, {}) != dim ({})",
                a_slice.len(),
                b_slice.len(),
                c_slice.len(),
                dim,
            )));
        }
        if !dim.is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2, got {}",
                dim,
            )));
        }

        Ok(algebra::cd_associator_norm(a_slice, b_slice, c_slice))
    }

    /// Batch computation of associator norms for multiple triples.
    #[pyfunction]
    fn batch_associator_norms<'py>(
        py: Python<'py>,
        a_flat: PyReadonlyArray1<'py, f64>,
        b_flat: PyReadonlyArray1<'py, f64>,
        c_flat: PyReadonlyArray1<'py, f64>,
        dim: usize,
        n_triples: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a_slice = a_flat.as_slice()?;
        let b_slice = b_flat.as_slice()?;
        let c_slice = c_flat.as_slice()?;

        let expected_len = dim * n_triples;
        if a_slice.len() != expected_len || b_slice.len() != expected_len || c_slice.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Array lengths ({}, {}, {}) != dim * n_triples ({})",
                a_slice.len(),
                b_slice.len(),
                c_slice.len(),
                expected_len,
            )));
        }
        if !dim.is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2, got {}",
                dim,
            )));
        }

        // Use parallel version for large batches
        let norms = if n_triples >= 100 {
            algebra::batch_associator_norms_parallel(a_slice, b_slice, c_slice, dim, n_triples)
        } else {
            algebra::batch_associator_norms(a_slice, b_slice, c_slice, dim, n_triples)
        };
        Ok(PyArray1::from_vec(py, norms))
    }

    /// Batch computation of squared associator norms for statistical analysis.
    #[pyfunction]
    fn batch_associator_norms_sq<'py>(
        py: Python<'py>,
        a_flat: PyReadonlyArray1<'py, f64>,
        b_flat: PyReadonlyArray1<'py, f64>,
        c_flat: PyReadonlyArray1<'py, f64>,
        dim: usize,
        n_triples: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a_slice = a_flat.as_slice()?;
        let b_slice = b_flat.as_slice()?;
        let c_slice = c_flat.as_slice()?;

        let expected_len = dim * n_triples;
        if a_slice.len() != expected_len || b_slice.len() != expected_len || c_slice.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Array lengths ({}, {}, {}) != dim * n_triples ({})",
                a_slice.len(),
                b_slice.len(),
                c_slice.len(),
                expected_len,
            )));
        }
        if !dim.is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2, got {}",
                dim,
            )));
        }

        let norms_sq = algebra::batch_associator_norms_sq(a_slice, b_slice, c_slice, dim, n_triples);
        Ok(PyArray1::from_vec(py, norms_sq))
    }

    /// Count pathion (32D) zero-divisors via sampling.
    #[pyfunction]
    #[pyo3(signature = (n_general_samples=10000, atol=1e-10, seed=42))]
    fn count_pathion_zero_divisors(
        n_general_samples: usize,
        atol: f64,
        seed: u64,
    ) -> PyResult<(usize, usize, usize)> {
        Ok(algebra::count_pathion_zero_divisors(n_general_samples, atol, seed))
    }

    /// Analyze zero-divisor spectrum for CD algebra.
    #[pyfunction]
    #[pyo3(signature = (dim, n_samples=10000, n_bins=50, seed=42))]
    fn zd_spectrum_analysis(
        dim: usize,
        n_samples: usize,
        n_bins: usize,
        seed: u64,
    ) -> PyResult<(f64, f64, f64, Vec<usize>)> {
        if !dim.is_power_of_two() || dim < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dim must be a power of 2 >= 2, got {}",
                dim,
            )));
        }
        Ok(algebra::zd_spectrum_analysis(dim, n_samples, n_bins, seed))
    }

    // Spectral analysis bindings
    #[pyfunction]
    #[pyo3(signature = (k, alpha=0.5))]
    fn calcagni_spectral_dimension(k: f64, alpha: f64) -> f64 {
        spectral::calcagni_spectral_dimension(k, alpha)
    }

    #[pyfunction]
    fn kraichnan_enstrophy_spectrum(k: f64) -> f64 {
        spectral::kraichnan_enstrophy_spectrum(k)
    }

    #[pyfunction]
    fn kolmogorov_spectrum(k: f64) -> f64 {
        spectral::kolmogorov_spectrum(k)
    }

    #[pyfunction]
    #[pyo3(signature = (k_min=0.01, k_max=100.0, n_points=100))]
    fn analyze_k_minus_3_origin(k_min: f64, k_max: f64, n_points: usize) -> (bool, bool, bool, bool, f64, f64, f64) {
        let result = spectral::analyze_k_minus_3_origin(k_min, k_max, n_points);
        (
            result.kraichnan_matches,
            result.kolmogorov_matches,
            result.calcagni_matches,
            result.parisi_sourlas_matches,
            result.kraichnan_rms,
            result.kolmogorov_rms,
            result.calcagni_rms,
        )
    }

    // MERA bindings
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (l_values, d=2, seed=42))]
    fn mera_entropy_scaling<'py>(
        py: Python<'py>,
        l_values: Vec<usize>,
        d: usize,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, f64, bool)> {
        let result = mera::mera_entropy_scaling_analysis(&l_values, d, seed);
        let entropies = PyArray1::from_vec(py, result.entropies);
        Ok((
            entropies,
            result.slope,
            result.intercept,
            result.central_charge_estimate,
            result.log_scaling_confirmed,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (l_values, d=2, n_bootstrap=50, base_seed=42))]
    fn mera_bootstrap_ci(
        l_values: Vec<usize>,
        d: usize,
        n_bootstrap: usize,
        base_seed: u64,
    ) -> (f64, f64, f64, f64) {
        mera::bootstrap_slope_ci(&l_values, d, n_bootstrap, base_seed)
    }

    // Metamaterial bindings
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (max_layers=20, base_n=1.5))]
    fn build_metamaterial_stack<'py>(
        py: Python<'py>,
        max_layers: usize,
        base_n: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, usize, usize)> {
        let zd_pairs = metamaterial::canonical_sedenion_zd_pairs();
        let stack = metamaterial::build_absorber_stack(&zd_pairs, max_layers, base_n);
        let verification = metamaterial::verify_physical_realizability(&stack);

        let n_real: Vec<f64> = stack.iter().map(|m| m.layer.n_real).collect();
        let n_imag: Vec<f64> = stack.iter().map(|m| m.layer.n_imag).collect();
        let thickness: Vec<f64> = stack.iter().map(|m| m.layer.thickness_nm).collect();

        Ok((
            PyArray1::from_vec(py, n_real),
            PyArray1::from_vec(py, n_imag),
            PyArray1::from_vec(py, thickness),
            verification.n_physical,
            verification.n_total,
        ))
    }

    // Statistics bindings
    #[pyfunction]
    fn frechet_distance<'py>(
        _py: Python<'py>,
        p: PyReadonlyArray1<'py, f64>,
        q: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let p_slice = p.as_slice()?;
        let q_slice = q.as_slice()?;
        Ok(stats::frechet_distance(p_slice, q_slice))
    }

    #[pyfunction]
    #[pyo3(signature = (observed, reference, n_permutations=1000, seed=42))]
    fn frechet_null_test(
        observed: PyReadonlyArray1<'_, f64>,
        reference: PyReadonlyArray1<'_, f64>,
        n_permutations: usize,
        seed: u64,
    ) -> PyResult<(f64, f64, bool, f64, f64)> {
        let obs_slice = observed.as_slice()?;
        let ref_slice = reference.as_slice()?;
        let result = stats::frechet_null_test(obs_slice, ref_slice, n_permutations, seed);
        Ok((
            result.observed_distance,
            result.p_value,
            result.significant_at_005,
            result.mean_null,
            result.std_null,
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (x, y, n_bootstrap=1000, seed=42))]
    fn fit_power_law_with_ci(
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        n_bootstrap: usize,
        seed: u64,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, bool)> {
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let result = stats::fit_power_law_with_ci(x_slice, y_slice, n_bootstrap, seed);
        Ok((
            result.amplitude.point_estimate,
            result.amplitude.ci_lower,
            result.amplitude.ci_upper,
            result.exponent.point_estimate,
            result.exponent.ci_lower,
            result.exponent.ci_upper,
            result.r_squared,
            result.uncertain,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (dim, seed=42))]
    fn haar_random_unitary<'py>(
        py: Python<'py>,
        dim: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let u = stats::haar_random_unitary(dim, seed);
        let mut data = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                data.push(u[(i, j)]);
            }
        }
        let arr = PyArray1::from_vec(py, data);
        let reshaped = arr.reshape([dim, dim])?;
        Ok(reshaped)
    }

    // Holographic bindings
    #[pyfunction]
    fn bekenstein_bound_bits(_py: Python<'_>, radius: f64, energy: f64) -> f64 {
        holographic::bekenstein_bound_bits(radius, energy)
    }

    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (depth=4, seed=42))]
    fn rt_lattice_entropy_scaling<'py>(
        py: Python<'py>,
        depth: usize,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64, f64, f64, f64, bool)> {
        let lattice = holographic::RTLattice::build_tree(depth, seed);
        let result = holographic::analyze_entropy_scaling(&lattice, seed);

        let sizes: Vec<f64> = result.sizes.iter().map(|&s| s as f64).collect();

        Ok((
            PyArray1::from_vec(py, sizes),
            PyArray1::from_vec(py, result.entropies),
            result.log_coefficient,
            result.log_intercept,
            result.power_exponent,
            result.log_r_squared,
            result.log_fit_better,
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (depth=4, n_bootstrap=100, seed=42))]
    fn verify_area_law_scaling(
        depth: usize,
        n_bootstrap: usize,
        seed: u64,
    ) -> (f64, f64, f64, f64, bool, bool) {
        let lattice = holographic::RTLattice::build_tree(depth, seed);
        let result = holographic::verify_area_law(&lattice, n_bootstrap, seed);

        (
            result.exponent,
            result.ci_lower,
            result.ci_upper,
            result.r_squared,
            result.consistent_with_area_law,
            result.consistent_with_volume_law,
        )
    }

    // Gravastar bindings
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (r1=5.0, m_target=10.0, compactness=0.7, gamma=1.0, k=1.0, lambda_aniso=0.0))]
    fn solve_gravastar_py(
        r1: f64,
        m_target: f64,
        compactness: f64,
        gamma: f64,
        k: f64,
        lambda_aniso: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, bool, bool)> {
        use crate::gravastar::{GravastarConfig, PolytropicEos, AnisotropicParams, solve_gravastar};

        let config = GravastarConfig {
            r1,
            m_target,
            compactness_target: compactness,
            eos: PolytropicEos::new(k, gamma),
            aniso: AnisotropicParams::new(lambda_aniso),
            dr: 1e-3,
            p_floor: 1e-12,
        };

        solve_gravastar(&config).map(|sol| (
            sol.r1,
            sol.r2,
            sol.mass,
            sol.shell_thickness,
            sol.compactness,
            sol.surface_redshift,
            sol.is_causal,
            sol.is_stable,
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (r1, m_target, compactness, gamma_min=1.0, gamma_max=2.5, n_gamma=10))]
    fn polytropic_stability_sweep_py<'py>(
        py: Python<'py>,
        r1: f64,
        m_target: f64,
        compactness: f64,
        gamma_min: f64,
        gamma_max: f64,
        n_gamma: usize,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Vec<bool>, Option<f64>, usize) {
        use crate::gravastar::polytropic_stability_sweep;

        let result = polytropic_stability_sweep(r1, m_target, compactness, gamma_min, gamma_max, n_gamma);

        (
            PyArray1::from_vec(py, result.gammas),
            PyArray1::from_vec(py, result.masses),
            result.stable_at_gamma,
            result.gamma_critical,
            result.n_stable,
        )
    }

    // ZD graph bindings
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (dim=16, atol=1e-10))]
    fn analyze_zd_graph_py(
        dim: usize,
        atol: f64,
    ) -> (usize, usize, usize, usize, f64, usize, f64, Vec<(usize, usize)>) {
        use crate::zd_graphs::analyze_zd_graph;

        let result = analyze_zd_graph(dim, atol);

        (
            result.n_nodes,
            result.n_edges,
            result.n_components,
            result.largest_component_size,
            result.avg_degree,
            result.max_degree,
            result.clustering_coefficient,
            result.basis_participation,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (dim=16, atol=1e-10))]
    fn analyze_basis_participation_py(
        dim: usize,
        atol: f64,
    ) -> (Vec<(usize, usize)>, f64, f64, Vec<usize>) {
        use crate::zd_graphs::analyze_basis_participation;

        let result = analyze_basis_participation(dim, atol);

        (
            result.counts,
            result.entropy,
            result.gini,
            result.hub_indices,
        )
    }

    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (dim=8, threshold=1e-10))]
    fn analyze_associator_graph_py(
        dim: usize,
        threshold: f64,
    ) -> (usize, usize, f64, f64, Vec<((usize, usize, usize), f64)>) {
        use crate::zd_graphs::analyze_associator_graph;

        let result = analyze_associator_graph(dim, threshold);

        (
            result.n_nonzero_pairs,
            result.n_scc,
            result.mean_norm,
            result.max_norm,
            result.top_pairs,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (dim=16, atol=1e-10))]
    fn zd_graph_diameter_py(dim: usize, atol: f64) -> usize {
        use crate::zd_graphs::zd_graph_diameter;
        zd_graph_diameter(dim, atol)
    }

    // Tang mass ratio bindings
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    #[pyo3(signature = (dim=16, n_samples=1000, seed=42))]
    fn find_best_mass_assignment(
        dim: usize,
        n_samples: usize,
        seed: u64,
    ) -> ((usize, usize, usize), (usize, usize, usize), (usize, usize, usize), (f64, f64, f64), (f64, f64, f64), f64) {
        use crate::tang_mass::find_best_assignment;

        let result = find_best_assignment(dim, n_samples, seed);

        (
            result.assignment.electron,
            result.assignment.muon,
            result.assignment.tau,
            result.norms,
            result.predicted_ratios,
            result.rms_deviation,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (dim=16, n_permutations=1000, seed=42))]
    fn mass_ratio_null_test_py(
        dim: usize,
        n_permutations: usize,
        seed: u64,
    ) -> (f64, f64, f64, f64, usize, bool) {
        use crate::tang_mass::mass_ratio_null_test;

        let result = mass_ratio_null_test(dim, n_permutations, seed);

        (
            result.observed_rms,
            result.p_value,
            result.mean_null_rms,
            result.std_null_rms,
            result.n_permutations,
            result.significant,
        )
    }

    #[pyfunction]
    fn pdg_lepton_masses() -> (f64, f64, f64, f64, f64, f64) {
        use crate::tang_mass::{M_ELECTRON, M_MUON, M_TAU, RATIO_E_MU, RATIO_MU_TAU, RATIO_E_TAU};
        (M_ELECTRON, M_MUON, M_TAU, RATIO_E_MU, RATIO_MU_TAU, RATIO_E_TAU)
    }

    #[pymodule]
    pub fn gororoba_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Algebra functions
        m.add_function(wrap_pyfunction!(cd_multiply, m)?)?;
        m.add_function(wrap_pyfunction!(cd_conjugate, m)?)?;
        m.add_function(wrap_pyfunction!(cd_norm_sq, m)?)?;
        m.add_function(wrap_pyfunction!(measure_associator_density, m)?)?;
        m.add_function(wrap_pyfunction!(left_mult_operator, m)?)?;
        m.add_function(wrap_pyfunction!(find_zero_divisors, m)?)?;
        m.add_function(wrap_pyfunction!(cd_associator_norm, m)?)?;
        m.add_function(wrap_pyfunction!(batch_associator_norms, m)?)?;
        m.add_function(wrap_pyfunction!(batch_associator_norms_sq, m)?)?;
        m.add_function(wrap_pyfunction!(count_pathion_zero_divisors, m)?)?;
        m.add_function(wrap_pyfunction!(zd_spectrum_analysis, m)?)?;
        // Spectral functions
        m.add_function(wrap_pyfunction!(calcagni_spectral_dimension, m)?)?;
        m.add_function(wrap_pyfunction!(kraichnan_enstrophy_spectrum, m)?)?;
        m.add_function(wrap_pyfunction!(kolmogorov_spectrum, m)?)?;
        m.add_function(wrap_pyfunction!(analyze_k_minus_3_origin, m)?)?;
        // MERA functions
        m.add_function(wrap_pyfunction!(mera_entropy_scaling, m)?)?;
        m.add_function(wrap_pyfunction!(mera_bootstrap_ci, m)?)?;
        // Metamaterial functions
        m.add_function(wrap_pyfunction!(build_metamaterial_stack, m)?)?;
        // Statistics functions
        m.add_function(wrap_pyfunction!(frechet_distance, m)?)?;
        m.add_function(wrap_pyfunction!(frechet_null_test, m)?)?;
        m.add_function(wrap_pyfunction!(fit_power_law_with_ci, m)?)?;
        m.add_function(wrap_pyfunction!(haar_random_unitary, m)?)?;
        // Holographic functions
        m.add_function(wrap_pyfunction!(bekenstein_bound_bits, m)?)?;
        m.add_function(wrap_pyfunction!(rt_lattice_entropy_scaling, m)?)?;
        m.add_function(wrap_pyfunction!(verify_area_law_scaling, m)?)?;
        // Gravastar functions
        m.add_function(wrap_pyfunction!(solve_gravastar_py, m)?)?;
        m.add_function(wrap_pyfunction!(polytropic_stability_sweep_py, m)?)?;
        // ZD graph functions
        m.add_function(wrap_pyfunction!(analyze_zd_graph_py, m)?)?;
        m.add_function(wrap_pyfunction!(analyze_basis_participation_py, m)?)?;
        m.add_function(wrap_pyfunction!(analyze_associator_graph_py, m)?)?;
        m.add_function(wrap_pyfunction!(zd_graph_diameter_py, m)?)?;
        // Tang mass ratio functions
        m.add_function(wrap_pyfunction!(find_best_mass_assignment, m)?)?;
        m.add_function(wrap_pyfunction!(mass_ratio_null_test_py, m)?)?;
        m.add_function(wrap_pyfunction!(pdg_lepton_masses, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::gororoba_kernels;
