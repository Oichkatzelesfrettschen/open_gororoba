//! Unified Hypercomplex Algebra Facade
//!
//! This module provides a cohesive API for all Cayley-Dickson algebra operations,
//! harmonizing the scattered implementations across the crate into a single
//! well-organized interface.
//!
//! # Architecture
//!
//! The hypercomplex number tower follows the Cayley-Dickson construction:
//! - Dim 1: Reals (R) - associative, commutative, no zero-divisors
//! - Dim 2: Complex (C) - associative, commutative, no zero-divisors
//! - Dim 4: Quaternions (H) - associative, non-commutative, no zero-divisors
//! - Dim 8: Octonions (O) - non-associative, alternative, no zero-divisors
//! - Dim 16: Sedenions (S) - non-associative, non-alternative, HAS zero-divisors
//! - Dim 32: Pathions (P) - more zero-divisors than sedenions
//! - Dim 64+: Higher CD algebras with increasing ZD density
//!
//! # Zero-Divisor Emergence
//!
//! At dimension 16 (sedenions), zero-divisors first appear. These are non-zero
//! pairs (a, b) where a * b = 0. The structure of these ZDs is described by:
//! - de Marrais box-kites (7 octahedral structures in sedenions)
//! - Reggiani's Stiefel manifold embedding V_2(R^7)
//!
//! # Literature
//!
//! - de Marrais (2000): Box-kite structure of sedenion zero-divisors
//! - Reggiani (2024): Geometry of sedenion zero-divisors (G_2 action)
//! - Furey et al. (2024): Cl(8) -> 3 generations of fermions
//! - Tang & Tang (2023): Sedenion SU(5) mass predictions

use crate::cayley_dickson::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator, cd_associator_norm,
    batch_associator_norms_parallel,
    find_zero_divisors, find_zero_divisors_parallel, find_zero_divisors_3blade,
    find_zero_divisors_general_form, count_pathion_zero_divisors,
    zd_spectrum_analysis, measure_associator_density, GeneralFormZD,
};
use crate::zd_graphs::{
    analyze_zd_graph, analyze_basis_participation, analyze_associator_graph,
    ZdGraphAnalysis, BasisParticipationResult, AssociatorGraphResult,
};
use crate::boxkites::{
    find_box_kites, analyze_box_kite_symmetry,
    BoxKite, BoxKiteSymmetryResult,
};
use crate::octonion_field::{
    Octonion, FieldParams, EvolutionResult, DispersionResult,
    FANO_TRIPLES, build_structure_constants,
    oct_multiply, oct_conjugate, oct_norm_sq,
    hamiltonian, force, noether_charges,
    evolve, gaussian_wave_packet, standing_wave, measure_dispersion,
};

/// Known algebra dimensions in the Cayley-Dickson tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgebraDim {
    /// Real numbers (dim 1)
    Real = 1,
    /// Complex numbers (dim 2)
    Complex = 2,
    /// Quaternions (dim 4)
    Quaternion = 4,
    /// Octonions (dim 8)
    Octonion = 8,
    /// Sedenions (dim 16) - first algebra with zero-divisors
    Sedenion = 16,
    /// Pathions (dim 32)
    Pathion = 32,
    /// Chingons (dim 64)
    Chingon = 64,
    /// Routons (dim 128)
    Routon = 128,
    /// Voudons (dim 256)
    Voudon = 256,
}

impl AlgebraDim {
    /// Returns the dimension as usize.
    pub fn dim(&self) -> usize {
        *self as usize
    }

    /// Returns true if the algebra has zero-divisors.
    pub fn has_zero_divisors(&self) -> bool {
        self.dim() >= 16
    }

    /// Returns true if the algebra is associative.
    pub fn is_associative(&self) -> bool {
        self.dim() <= 4
    }

    /// Returns true if the algebra is alternative (weaker than associative).
    pub fn is_alternative(&self) -> bool {
        self.dim() <= 8
    }

    /// Returns true if the algebra is commutative.
    pub fn is_commutative(&self) -> bool {
        self.dim() <= 2
    }

    /// Returns the name of the algebra.
    pub fn name(&self) -> &'static str {
        match self {
            AlgebraDim::Real => "Real",
            AlgebraDim::Complex => "Complex",
            AlgebraDim::Quaternion => "Quaternion",
            AlgebraDim::Octonion => "Octonion",
            AlgebraDim::Sedenion => "Sedenion",
            AlgebraDim::Pathion => "Pathion",
            AlgebraDim::Chingon => "Chingon",
            AlgebraDim::Routon => "Routon",
            AlgebraDim::Voudon => "Voudon",
        }
    }

    /// Create from dimension, returns None if not a valid CD dimension.
    pub fn from_dim(dim: usize) -> Option<Self> {
        match dim {
            1 => Some(AlgebraDim::Real),
            2 => Some(AlgebraDim::Complex),
            4 => Some(AlgebraDim::Quaternion),
            8 => Some(AlgebraDim::Octonion),
            16 => Some(AlgebraDim::Sedenion),
            32 => Some(AlgebraDim::Pathion),
            64 => Some(AlgebraDim::Chingon),
            128 => Some(AlgebraDim::Routon),
            256 => Some(AlgebraDim::Voudon),
            _ => None,
        }
    }
}

/// Configuration for zero-divisor search algorithms.
#[derive(Debug, Clone)]
pub struct ZeroSearchConfig {
    /// Tolerance for considering a product as zero.
    pub tolerance: f64,
    /// Use parallel algorithms when available.
    pub parallel: bool,
    /// Maximum blade order to search (2, 3, or higher).
    pub max_blade_order: usize,
    /// Number of random samples for general-form search.
    pub n_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for ZeroSearchConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            parallel: true,
            max_blade_order: 2,
            n_samples: 10000,
            seed: 42,
        }
    }
}

/// Comprehensive zero-divisor search results.
#[derive(Debug, Clone)]
pub struct ZeroDivisorResults {
    /// 2-blade zero-divisor pairs: (i, j, k, l, norm) where (e_i +/- e_j)(e_k +/- e_l) ~ 0
    pub blade2: Vec<(usize, usize, usize, usize, f64)>,
    /// 3-blade zero-divisor pairs (if searched)
    #[allow(clippy::type_complexity)]
    pub blade3: Option<Vec<(usize, usize, usize, usize, usize, usize, f64)>>,
    /// General-form zero-divisors from random sampling
    pub general: Option<Vec<GeneralFormZD>>,
    /// Box-kite structures (sedenion only)
    pub box_kites: Option<Vec<BoxKite>>,
    /// Graph analysis results
    pub graph_analysis: Option<ZdGraphAnalysis>,
    /// Basis participation statistics
    pub basis_stats: Option<BasisParticipationResult>,
    /// Dimension of the algebra searched
    pub dimension: usize,
}

/// Unified hypercomplex algebra interface.
///
/// Provides a cohesive API for all Cayley-Dickson operations including
/// multiplication, zero-divisor search, and structural analysis.
pub struct HypercomplexAlgebra {
    dim: usize,
}

impl HypercomplexAlgebra {
    /// Create a new hypercomplex algebra of the given dimension.
    ///
    /// # Panics
    /// Panics if dimension is not a power of 2.
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two(), "Dimension must be a power of 2");
        Self { dim }
    }

    /// Create from a known algebra dimension.
    pub fn from_algebra(algebra: AlgebraDim) -> Self {
        Self { dim: algebra.dim() }
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the algebra type if it's a named dimension.
    pub fn algebra_type(&self) -> Option<AlgebraDim> {
        AlgebraDim::from_dim(self.dim)
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Multiply two hypercomplex numbers.
    pub fn multiply(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), self.dim);
        assert_eq!(b.len(), self.dim);
        cd_multiply(a, b)
    }

    /// Conjugate a hypercomplex number (negate imaginary parts).
    pub fn conjugate(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        cd_conjugate(x)
    }

    /// Squared norm ||x||^2.
    pub fn norm_sq(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.dim);
        cd_norm_sq(x)
    }

    /// Norm ||x||.
    pub fn norm(&self, x: &[f64]) -> f64 {
        self.norm_sq(x).sqrt()
    }

    /// Compute the associator [a, b, c] = (ab)c - a(bc).
    pub fn associator(&self, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), self.dim);
        assert_eq!(b.len(), self.dim);
        assert_eq!(c.len(), self.dim);
        cd_associator(a, b, c)
    }

    /// Associator norm ||[a, b, c]||.
    pub fn associator_norm(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        assert_eq!(a.len(), self.dim);
        assert_eq!(b.len(), self.dim);
        assert_eq!(c.len(), self.dim);
        cd_associator_norm(a, b, c)
    }

    /// Create a basis element e_i.
    pub fn basis(&self, i: usize) -> Vec<f64> {
        assert!(i < self.dim);
        let mut e = vec![0.0; self.dim];
        e[i] = 1.0;
        e
    }

    /// Create a 2-blade: e_i + sign * e_j.
    pub fn blade2(&self, i: usize, j: usize, sign: f64) -> Vec<f64> {
        assert!(i < self.dim && j < self.dim);
        let mut e = vec![0.0; self.dim];
        e[i] = 1.0;
        e[j] = sign;
        e
    }

    // =========================================================================
    // Zero-Divisor Search
    // =========================================================================

    /// Comprehensive zero-divisor search.
    ///
    /// Searches for zero-divisor pairs using multiple algorithms based on
    /// configuration. Returns structured results including graph analysis.
    pub fn find_zero_divisors(&self, config: &ZeroSearchConfig) -> ZeroDivisorResults {
        // 2-blade search (sequential or parallel)
        let blade2 = if config.parallel {
            find_zero_divisors_parallel(self.dim, config.tolerance)
        } else {
            find_zero_divisors(self.dim, config.tolerance)
        };

        // 3-blade search (expensive, only if requested)
        let blade3 = if config.max_blade_order >= 3 && self.dim <= 16 {
            Some(find_zero_divisors_3blade(self.dim, config.tolerance))
        } else {
            None
        };

        // General-form random sampling
        let general = if config.n_samples > 0 {
            Some(find_zero_divisors_general_form(
                self.dim,
                config.n_samples,
                config.tolerance,
                config.seed,
            ))
        } else {
            None
        };

        // Box-kites (sedenion only)
        let box_kites = if self.dim == 16 {
            Some(find_box_kites(self.dim, config.tolerance))
        } else {
            None
        };

        // Graph analysis
        let graph_analysis = if !blade2.is_empty() {
            Some(analyze_zd_graph(self.dim, config.tolerance))
        } else {
            None
        };

        // Basis participation
        let basis_stats = if !blade2.is_empty() {
            Some(analyze_basis_participation(self.dim, config.tolerance))
        } else {
            None
        };

        ZeroDivisorResults {
            blade2,
            blade3,
            general,
            box_kites,
            graph_analysis,
            basis_stats,
            dimension: self.dim,
        }
    }

    /// Quick 2-blade zero-divisor count.
    pub fn count_2blade_zd(&self, tolerance: f64) -> usize {
        find_zero_divisors_parallel(self.dim, tolerance).len()
    }

    /// Analyze associator structure.
    pub fn analyze_associators(&self, tolerance: f64) -> AssociatorGraphResult {
        analyze_associator_graph(self.dim, tolerance)
    }

    // =========================================================================
    // Box-Kite Analysis (Sedenion-specific)
    // =========================================================================

    /// Find box-kite structures (de Marrais).
    ///
    /// Returns None if dimension is not 16 (sedenions).
    pub fn find_box_kites(&self, tolerance: f64) -> Option<Vec<BoxKite>> {
        if self.dim != 16 {
            return None;
        }
        Some(find_box_kites(self.dim, tolerance))
    }

    /// Analyze box-kite symmetry (PSL(2,7) group action).
    pub fn analyze_box_kite_symmetry(&self, tolerance: f64) -> Option<BoxKiteSymmetryResult> {
        if self.dim != 16 {
            return None;
        }
        Some(analyze_box_kite_symmetry(self.dim, tolerance))
    }

    // =========================================================================
    // Non-Associativity Metrics
    // =========================================================================

    /// Measure non-associativity density.
    ///
    /// Tests random triples and returns (density%, failure_count).
    pub fn measure_non_associativity(
        &self,
        trials: usize,
        seed: u64,
        tolerance: f64,
    ) -> (f64, usize) {
        measure_associator_density(self.dim, trials, seed, tolerance)
    }

    /// Compute batch associator norms (parallel).
    pub fn batch_associator_norms(
        &self,
        a_flat: &[f64],
        b_flat: &[f64],
        c_flat: &[f64],
        n_triples: usize,
    ) -> Vec<f64> {
        batch_associator_norms_parallel(a_flat, b_flat, c_flat, self.dim, n_triples)
    }

    // =========================================================================
    // Spectrum Analysis
    // =========================================================================

    /// Analyze zero-divisor spectrum.
    ///
    /// Returns (min_norm, max_norm, mean_norm, histogram).
    pub fn zd_spectrum(
        &self,
        n_samples: usize,
        n_bins: usize,
        seed: u64,
    ) -> (f64, f64, f64, Vec<usize>) {
        zd_spectrum_analysis(self.dim, n_samples, n_bins, seed)
    }
}

// =============================================================================
// Octonion Field Operations (Specialized for dim=8)
// =============================================================================

/// Specialized interface for octonion field dynamics.
///
/// Provides Fano-plane-based multiplication and field evolution on lattices.
pub struct OctonionFieldDynamics {
    params: FieldParams,
    // Stored for future optimized multiplication using cached structure constants
    _structure_constants: [[Option<(i8, usize)>; 8]; 8],
}

impl OctonionFieldDynamics {
    /// Create a new octonion field with given parameters.
    pub fn new(params: FieldParams) -> Self {
        Self {
            params,
            _structure_constants: build_structure_constants(),
        }
    }

    /// Get Fano plane triples defining octonion multiplication.
    ///
    /// Returns 7 triples (a, b, c) where e_a * e_b = e_c (cyclic).
    pub fn fano_triples() -> &'static [(usize, usize, usize); 7] {
        &FANO_TRIPLES
    }

    /// Multiply two octonions using structure constants.
    pub fn multiply(&self, a: &Octonion, b: &Octonion) -> Octonion {
        oct_multiply(a, b)
    }

    /// Conjugate an octonion.
    pub fn conjugate(&self, x: &Octonion) -> Octonion {
        oct_conjugate(x)
    }

    /// Octonion squared norm.
    pub fn norm_sq(&self, x: &Octonion) -> f64 {
        oct_norm_sq(x)
    }

    /// Compute field Hamiltonian.
    pub fn hamiltonian(&self, phi: &[Octonion], pi: &[Octonion]) -> f64 {
        hamiltonian(phi, pi, &self.params)
    }

    /// Compute force on field.
    pub fn force(&self, phi: &[Octonion]) -> Vec<Octonion> {
        force(phi, &self.params)
    }

    /// Evolve field using Stormer-Verlet integrator.
    pub fn evolve(&self, phi0: &[Octonion], pi0: &[Octonion], n_steps: usize) -> EvolutionResult {
        evolve(phi0, pi0, &self.params, n_steps)
    }

    /// Create Gaussian wave packet initial condition.
    ///
    /// Returns (phi, pi) tuple for initial field configuration.
    pub fn gaussian_wave_packet(&self) -> (Vec<Octonion>, Vec<Octonion>) {
        gaussian_wave_packet(&self.params)
    }

    /// Create standing wave initial condition.
    ///
    /// Returns (phi, pi) tuple for initial field configuration.
    pub fn standing_wave(&self, mode: usize, amplitude: f64) -> (Vec<Octonion>, Vec<Octonion>) {
        standing_wave(&self.params, mode, amplitude)
    }

    /// Measure dispersion relation.
    ///
    /// Returns dispersion results for n_modes different wavenumbers.
    pub fn measure_dispersion(&self, n_modes: usize) -> Vec<DispersionResult> {
        measure_dispersion(&self.params, n_modes)
    }

    /// Compute Noether charges (conserved quantities).
    ///
    /// Returns 7 conserved charges corresponding to imaginary octonion components.
    pub fn noether_charges(&self, phi: &[Octonion], pi: &[Octonion]) -> [f64; 7] {
        noether_charges(phi, pi)
    }
}

// =============================================================================
// Pathion (32D) Specialized Interface
// =============================================================================

/// Specialized interface for pathion (32D) algebra.
///
/// Pathions have significantly more zero-divisors than sedenions and
/// require parallel algorithms for practical computation.
pub struct PathionAlgebra {
    algebra: HypercomplexAlgebra,
}

impl PathionAlgebra {
    /// Create a new pathion algebra.
    pub fn new() -> Self {
        Self {
            algebra: HypercomplexAlgebra::new(32),
        }
    }

    /// Count zero-divisors using all methods.
    ///
    /// Returns (n_2blade, n_3blade_sampled, n_general_sampled).
    pub fn count_zero_divisors(
        &self,
        n_samples: usize,
        tolerance: f64,
        seed: u64,
    ) -> (usize, usize, usize) {
        count_pathion_zero_divisors(n_samples, tolerance, seed)
    }

    /// Fast 2-blade zero-divisor count (parallel).
    pub fn count_2blade_zd(&self, tolerance: f64) -> usize {
        self.algebra.count_2blade_zd(tolerance)
    }

    /// Get the underlying algebra.
    pub fn algebra(&self) -> &HypercomplexAlgebra {
        &self.algebra
    }
}

impl Default for PathionAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebra_dim_properties() {
        assert!(!AlgebraDim::Real.has_zero_divisors());
        assert!(!AlgebraDim::Complex.has_zero_divisors());
        assert!(!AlgebraDim::Quaternion.has_zero_divisors());
        assert!(!AlgebraDim::Octonion.has_zero_divisors());
        assert!(AlgebraDim::Sedenion.has_zero_divisors());
        assert!(AlgebraDim::Pathion.has_zero_divisors());

        assert!(AlgebraDim::Real.is_associative());
        assert!(AlgebraDim::Complex.is_associative());
        assert!(AlgebraDim::Quaternion.is_associative());
        assert!(!AlgebraDim::Octonion.is_associative());

        assert!(AlgebraDim::Octonion.is_alternative());
        assert!(!AlgebraDim::Sedenion.is_alternative());
    }

    #[test]
    fn test_hypercomplex_multiply_quaternion() {
        let h = HypercomplexAlgebra::new(4);
        // i * j = k in quaternions
        let i = h.basis(1);
        let j = h.basis(2);
        let k = h.multiply(&i, &j);
        assert!((k[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypercomplex_multiply_octonion() {
        let o = HypercomplexAlgebra::new(8);
        // e1 * e2 = e3 in octonions (Fano triple (1,2,3))
        let e1 = o.basis(1);
        let e2 = o.basis(2);
        let e3 = o.multiply(&e1, &e2);
        assert!((e3[3] - 1.0).abs() < 1e-10, "e1*e2 should give e3, got {:?}", e3);
    }

    #[test]
    fn test_sedenion_zd_search() {
        let s = HypercomplexAlgebra::new(16);
        let config = ZeroSearchConfig {
            parallel: true,
            max_blade_order: 2,
            n_samples: 0, // Skip general search for speed
            ..Default::default()
        };
        let results = s.find_zero_divisors(&config);
        // Sedenions have exactly 84 2-blade ZD pairs
        assert!(results.blade2.len() >= 80, "Expected ~84 2-blade ZDs, got {}", results.blade2.len());
    }

    #[test]
    fn test_sedenion_box_kites() {
        let s = HypercomplexAlgebra::new(16);
        let box_kites = s.find_box_kites(1e-10).unwrap();

        // de Marrais: exactly 7 box-kites as connected components of co-assessor graph
        assert_eq!(box_kites.len(), 7, "Expected 7 box-kites, got {}", box_kites.len());

        // Symmetry analysis should validate de Marrais structure
        let symmetry = s.analyze_box_kite_symmetry(1e-10).unwrap();
        assert_eq!(symmetry.n_boxkites, 7);
        assert!(symmetry.de_marrais_valid, "Should validate de Marrais structure");
    }

    #[test]
    fn test_octonion_no_zd() {
        let o = HypercomplexAlgebra::new(8);
        let config = ZeroSearchConfig::default();
        let results = o.find_zero_divisors(&config);
        // Octonions have no zero-divisors
        assert!(results.blade2.is_empty(), "Octonions should have no ZDs");
    }

    #[test]
    fn test_pathion_more_zd() {
        let p = PathionAlgebra::new();
        let s = HypercomplexAlgebra::new(16);

        let pathion_zd = p.count_2blade_zd(1e-10);
        let sedenion_zd = s.count_2blade_zd(1e-10);

        assert!(pathion_zd > sedenion_zd,
            "Pathions should have more ZDs: {} vs {}", pathion_zd, sedenion_zd);
    }

    #[test]
    fn test_non_associativity_scaling() {
        // Quaternions are associative
        let q = HypercomplexAlgebra::new(4);
        let (density_q, _) = q.measure_non_associativity(1000, 42, 1e-10);
        assert!(density_q < 1.0, "Quaternions should be ~0% non-associative");

        // Sedenions are highly non-associative
        let s = HypercomplexAlgebra::new(16);
        let (density_s, _) = s.measure_non_associativity(1000, 42, 1e-10);
        assert!(density_s > 90.0, "Sedenions should be >90% non-associative, got {}%", density_s);
    }

    #[test]
    fn test_octonion_field_fano() {
        let triples = OctonionFieldDynamics::fano_triples();
        assert_eq!(triples.len(), 7, "Fano plane has 7 lines");

        // Verify each triple has valid indices (1-7 for imaginary units)
        for (a, b, c) in triples {
            assert!(*a >= 1 && *a <= 7);
            assert!(*b >= 1 && *b <= 7);
            assert!(*c >= 1 && *c <= 7);
        }
    }
}
